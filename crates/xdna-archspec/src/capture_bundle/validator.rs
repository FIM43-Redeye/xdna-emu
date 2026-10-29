use super::{
    canonical::{canonical_json, sha256_bytes},
    canonicalize_manifest, canonicalize_manifest_v2, parse_manifest_document, ArtifactRecord, BundleIssue,
    BundleManifest, BundleManifestV2, BundlePayload, Campaign, DependencyRequirement, ManifestDocument,
};
use crate::research_reserve::BundleLocationRoot;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File},
    io::{self, Read},
    path::{Component, Path, PathBuf},
};

const ROOT_ENTRIES: [&str; 4] = ["manifest.json", "SHA256SUMS", "raw", "derived"];

#[derive(Debug)]
pub struct BundleValidationError {
    issues: Vec<BundleIssue>,
}

impl BundleValidationError {
    pub fn issues(&self) -> &[BundleIssue] {
        &self.issues
    }
}

impl fmt::Display for BundleValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "capture bundle validation failed")?;
        for issue in &self.issues {
            write!(f, "; {}: {}", issue.path, issue.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for BundleValidationError {}

#[derive(Debug)]
pub struct ValidatedBundle {
    manifest: ManifestDocument,
    manifest_sha256: String,
    checksum_index_sha256: String,
    promotion_blockers: Vec<BundleIssue>,
}

impl ValidatedBundle {
    pub fn bundle_id(&self) -> &str {
        match &self.manifest {
            ManifestDocument::V1(manifest) => &manifest.bundle_id,
            ManifestDocument::V2(manifest) => &manifest.bundle_id,
        }
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn checksum_index_sha256(&self) -> &str {
        &self.checksum_index_sha256
    }

    pub fn is_promotion_eligible(&self) -> bool {
        self.promotion_blockers.is_empty()
    }

    pub fn promotion_blockers(&self) -> &[BundleIssue] {
        &self.promotion_blockers
    }

    pub(crate) fn campaign(&self) -> Option<&Campaign> {
        match &self.manifest {
            ManifestDocument::V1(manifest) => Some(&manifest.campaign),
            ManifestDocument::V2(manifest) => match &manifest.payload {
                BundlePayload::Observation(body) => Some(&body.campaign),
                BundlePayload::Fixture(_) => None,
            },
        }
    }

    pub(crate) fn dependencies(&self) -> &[DependencyRequirement] {
        match &self.manifest {
            ManifestDocument::V1(_) => &[],
            ManifestDocument::V2(manifest) => &manifest.dependencies,
        }
    }

    pub(crate) fn artifacts(&self) -> &[ArtifactRecord] {
        match &self.manifest {
            ManifestDocument::V1(manifest) => &manifest.artifacts,
            ManifestDocument::V2(manifest) => &manifest.artifacts,
        }
    }

    pub(crate) fn is_fixture(&self) -> bool {
        matches!(
            &self.manifest,
            ManifestDocument::V2(BundleManifestV2 { payload: BundlePayload::Fixture(_), .. })
        )
    }

    pub(crate) fn is_observation(&self) -> bool {
        matches!(
            &self.manifest,
            ManifestDocument::V2(BundleManifestV2 { payload: BundlePayload::Observation(_), .. })
        )
    }
}

#[derive(Debug)]
pub struct ValidatedBundleGraph {
    root: ValidatedBundle,
    fixtures: BTreeMap<String, ValidatedBundle>,
    promotion_blockers: Vec<BundleIssue>,
}

impl ValidatedBundleGraph {
    pub fn root_bundle_id(&self) -> &str {
        self.root.bundle_id()
    }

    pub fn bundle_count(&self) -> usize {
        1 + self.fixtures.len()
    }

    pub fn is_promotion_eligible(&self) -> bool {
        self.promotion_blockers.is_empty()
    }

    pub fn promotion_blockers(&self) -> &[BundleIssue] {
        &self.promotion_blockers
    }

    pub(crate) fn root(&self) -> &ValidatedBundle {
        &self.root
    }

    pub(crate) fn into_root(self) -> ValidatedBundle {
        self.root
    }
}

pub fn validate_bundle(root: impl AsRef<Path>) -> Result<ValidatedBundle, BundleValidationError> {
    let root = root.as_ref();
    validate_root(root)?;

    let manifest_bytes = read_file(&root.join("manifest.json"), "$.manifest")?;
    let manifest = parse_manifest_document(&manifest_bytes).map_err(|error| {
        validation_error(vec![BundleIssue { path: "$.manifest".into(), message: error.to_string() }])
    })?;
    let eligibility = manifest.validate().map_err(|error| validation_error(error.issues().to_vec()))?;
    let (canonical_manifest, canonical_manifest_bytes, canonical_checksum_bytes) = match &manifest {
        ManifestDocument::V1(manifest) => {
            let canonical =
                canonicalize_manifest(manifest).map_err(|error| validation_error(error.issues().to_vec()))?;
            let authored =
                BundleManifest { bundle_id: manifest.bundle_id.clone(), ..canonical.manifest().clone() };
            (
                ManifestDocument::V1(canonical.manifest().clone()),
                canonical_json(&authored),
                canonical.checksum_index_bytes().to_vec(),
            )
        }
        ManifestDocument::V2(manifest) => {
            let canonical = canonicalize_manifest_v2(manifest)
                .map_err(|error| validation_error(error.issues().to_vec()))?;
            let authored =
                BundleManifestV2 { bundle_id: manifest.bundle_id.clone(), ..canonical.manifest().clone() };
            (
                ManifestDocument::V2(canonical.manifest().clone()),
                canonical_json(&authored),
                canonical.checksum_index_bytes().to_vec(),
            )
        }
    };
    if manifest_bytes != canonical_manifest_bytes {
        return Err(validation_error(vec![BundleIssue {
            path: "$.manifest".into(),
            message: "manifest.json is not canonical".into(),
        }]));
    }

    let actual_artifacts = collect_artifacts(root)?;
    let declared_artifacts: BTreeSet<String> = artifacts(&canonical_manifest)
        .iter()
        .map(|artifact| artifact.path.clone())
        .collect();
    let mut issues = Vec::new();
    for path in actual_artifacts.difference(&declared_artifacts) {
        push_issue(&mut issues, format!("$.tree.{path}"), "undeclared artifact");
    }
    for path in declared_artifacts.difference(&actual_artifacts) {
        push_issue(&mut issues, format!("$.artifacts.{path}"), "declared artifact is missing");
    }
    fail_if_any(issues)?;

    let mut issues = Vec::new();
    for artifact in artifacts(&canonical_manifest) {
        let path = root.join(&artifact.path);
        match hash_file(&path) {
            Ok((size, sha256)) => {
                if size != artifact.byte_size {
                    push_issue(
                        &mut issues,
                        format!("$.artifacts.{}.byte_size", artifact.path),
                        format!("artifact size mismatch: expected {}, found {size}", artifact.byte_size),
                    );
                }
                if sha256 != artifact.sha256 {
                    push_issue(
                        &mut issues,
                        format!("$.artifacts.{}.sha256", artifact.path),
                        format!("artifact SHA-256 mismatch: expected {}, found {sha256}", artifact.sha256),
                    );
                }
            }
            Err(error) => push_issue(
                &mut issues,
                format!("$.artifacts.{}", artifact.path),
                format!("cannot read artifact: {error}"),
            ),
        }
    }
    fail_if_any(issues)?;

    let checksum_index_bytes = read_file(&root.join("SHA256SUMS"), "$.checksum_index")?;
    if checksum_index_bytes != canonical_checksum_bytes {
        return Err(validation_error(vec![BundleIssue {
            path: "$.checksum_index".into(),
            message: "SHA256SUMS is not canonical".into(),
        }]));
    }
    if bundle_id(&manifest) != bundle_id(&canonical_manifest) {
        return Err(validation_error(vec![BundleIssue {
            path: "$.manifest.bundle_id".into(),
            message: format!(
                "bundle ID mismatch: declared {}, recomputed {}",
                bundle_id(&manifest),
                bundle_id(&canonical_manifest)
            ),
        }]));
    }

    Ok(ValidatedBundle {
        manifest: canonical_manifest,
        manifest_sha256: sha256_bytes(&manifest_bytes),
        checksum_index_sha256: sha256_bytes(&checksum_index_bytes),
        promotion_blockers: eligibility.blockers().to_vec(),
    })
}

pub fn validate_bundle_graph(
    root_path: impl AsRef<Path>,
    location_root: &BundleLocationRoot,
) -> Result<ValidatedBundleGraph, BundleValidationError> {
    let root_path = root_path.as_ref();
    let locations = validate_locations(location_root)?;
    let root = validate_bundle(root_path)?;
    let root_id = root.bundle_id().to_owned();
    let mapped_root = locations.get(&root_id).ok_or_else(|| {
        graph_error("$.location_root.bundles", format!("missing bundle mapping for root {root_id}"))
    })?;
    let canonical_root_path = fs::canonicalize(root_path)
        .map_err(|error| graph_error("$.graph.root", format!("cannot resolve root bundle path: {error}")))?;
    if mapped_root != &canonical_root_path {
        return Err(graph_error(
            "$.location_root.bundles",
            format!("root bundle mapping for {root_id} does not match the requested path"),
        ));
    }

    let mut fixtures = BTreeMap::new();
    let mut visiting = BTreeSet::from([root_id.clone()]);
    let mut complete = BTreeSet::new();
    let mut blockers = prefixed_blockers("root", root.promotion_blockers());
    for dependency in root.dependencies() {
        validate_fixture_dependency(
            dependency,
            &locations,
            &mut visiting,
            &mut complete,
            &mut fixtures,
            &mut blockers,
        )?;
    }
    visiting.remove(&root_id);
    blockers.sort();
    blockers.dedup();

    Ok(ValidatedBundleGraph { root, fixtures, promotion_blockers: blockers })
}

fn validate_locations(
    location_root: &BundleLocationRoot,
) -> Result<BTreeMap<String, PathBuf>, BundleValidationError> {
    let mut issues = Vec::new();
    if location_root.alias.trim().is_empty() {
        push_issue(&mut issues, "$.location_root.alias", "bundle root alias must not be blank");
    }
    if location_root.failure_domain_id.trim().is_empty() {
        push_issue(
            &mut issues,
            "$.location_root.failure_domain_id",
            "bundle root failure-domain ID must not be blank",
        );
    }
    let canonical_root = fs::canonicalize(&location_root.path).map_err(|error| {
        graph_error("$.location_root.path", format!("cannot resolve bundle location root: {error}"))
    })?;
    let mut by_id = BTreeMap::new();
    let mut by_path = BTreeSet::new();
    for (index, entry) in location_root.bundles.iter().enumerate() {
        let path = format!("$.location_root.bundles[{index}]");
        super::validate_bundle_id(&entry.bundle_id, &format!("{path}.bundle_id"), &mut issues);
        let relative = Path::new(&entry.relative_path);
        if entry.relative_path.is_empty()
            || relative.is_absolute()
            || !relative.components().all(|component| matches!(component, Component::Normal(_)))
        {
            push_issue(
                &mut issues,
                format!("{path}.relative_path"),
                format!("invalid relative path `{}`", entry.relative_path),
            );
            continue;
        }
        let target = match fs::canonicalize(location_root.path.join(relative)) {
            Ok(target) => target,
            Err(error) => {
                push_issue(
                    &mut issues,
                    format!("{path}.relative_path"),
                    format!("cannot resolve mapped bundle location: {error}"),
                );
                continue;
            }
        };
        if !target.starts_with(&canonical_root) {
            push_issue(
                &mut issues,
                format!("{path}.relative_path"),
                "mapped bundle location escapes its declared root",
            );
            continue;
        }
        if by_id.insert(entry.bundle_id.clone(), target.clone()).is_some() {
            push_issue(
                &mut issues,
                format!("{path}.bundle_id"),
                format!("duplicate bundle ID `{}`", entry.bundle_id),
            );
        }
        if !by_path.insert(target) {
            push_issue(
                &mut issues,
                format!("{path}.relative_path"),
                format!("duplicate mapped location `{}`", entry.relative_path),
            );
        }
    }
    fail_if_any(issues)?;
    Ok(by_id)
}

fn validate_fixture_dependency(
    requirement: &DependencyRequirement,
    locations: &BTreeMap<String, PathBuf>,
    visiting: &mut BTreeSet<String>,
    complete: &mut BTreeSet<String>,
    fixtures: &mut BTreeMap<String, ValidatedBundle>,
    blockers: &mut Vec<BundleIssue>,
) -> Result<(), BundleValidationError> {
    if complete.contains(&requirement.fixture_bundle_id) {
        return validate_fixture_artifact(
            fixtures
                .get(&requirement.fixture_bundle_id)
                .expect("complete fixtures are retained"),
            requirement,
        );
    }
    if !visiting.insert(requirement.fixture_bundle_id.clone()) {
        return Err(graph_error(
            "$.graph.dependencies",
            format!("fixture dependency cycle at {}", requirement.fixture_bundle_id),
        ));
    }
    let path = locations.get(&requirement.fixture_bundle_id).ok_or_else(|| {
        graph_error(
            "$.location_root.bundles",
            format!("missing bundle mapping for {}", requirement.fixture_bundle_id),
        )
    })?;
    let fixture = validate_bundle(path)?;
    if fixture.bundle_id() != requirement.fixture_bundle_id {
        return Err(graph_error(
            "$.graph.dependencies",
            format!(
                "mapped bundle ID mismatch: expected {}, found {}",
                requirement.fixture_bundle_id,
                fixture.bundle_id()
            ),
        ));
    }
    if !fixture.is_fixture() {
        return Err(graph_error(
            "$.graph.dependencies",
            format!("dependency {} is not a v2 fixture", requirement.fixture_bundle_id),
        ));
    }
    validate_fixture_artifact(&fixture, requirement)?;
    for child in fixture.dependencies() {
        validate_fixture_dependency(child, locations, visiting, complete, fixtures, blockers)?;
    }
    blockers.extend(prefixed_blockers(&requirement.fixture_bundle_id, fixture.promotion_blockers()));
    visiting.remove(&requirement.fixture_bundle_id);
    complete.insert(requirement.fixture_bundle_id.clone());
    fixtures.insert(requirement.fixture_bundle_id.clone(), fixture);
    Ok(())
}

fn validate_fixture_artifact(
    fixture: &ValidatedBundle,
    requirement: &DependencyRequirement,
) -> Result<(), BundleValidationError> {
    let artifact = fixture
        .artifacts()
        .iter()
        .find(|artifact| artifact.path == requirement.artifact_path)
        .ok_or_else(|| {
            graph_error(
                "$.graph.dependencies",
                format!(
                    "fixture {} has no artifact path {}",
                    requirement.fixture_bundle_id, requirement.artifact_path
                ),
            )
        })?;
    if artifact.sha256 != requirement.artifact_sha256 {
        return Err(graph_error(
            "$.graph.dependencies",
            format!(
                "artifact SHA-256 mismatch for {}:{}",
                requirement.fixture_bundle_id, requirement.artifact_path
            ),
        ));
    }
    if artifact.semantic_kind != requirement.semantic_kind {
        return Err(graph_error(
            "$.graph.dependencies",
            format!(
                "artifact semantic kind mismatch for {}:{}",
                requirement.fixture_bundle_id, requirement.artifact_path
            ),
        ));
    }
    Ok(())
}

fn prefixed_blockers(prefix: &str, blockers: &[BundleIssue]) -> Vec<BundleIssue> {
    blockers
        .iter()
        .map(|blocker| BundleIssue {
            path: format!("$.graph.{prefix}{}", blocker.path.strip_prefix('$').unwrap_or(&blocker.path)),
            message: blocker.message.clone(),
        })
        .collect()
}

fn graph_error(path: impl Into<String>, message: impl Into<String>) -> BundleValidationError {
    validation_error(vec![BundleIssue { path: path.into(), message: message.into() }])
}

fn bundle_id(manifest: &ManifestDocument) -> &str {
    match manifest {
        ManifestDocument::V1(manifest) => &manifest.bundle_id,
        ManifestDocument::V2(manifest) => &manifest.bundle_id,
    }
}

fn artifacts(manifest: &ManifestDocument) -> &[ArtifactRecord] {
    match manifest {
        ManifestDocument::V1(manifest) => &manifest.artifacts,
        ManifestDocument::V2(manifest) => &manifest.artifacts,
    }
}

fn validate_root(root: &Path) -> Result<(), BundleValidationError> {
    let metadata = fs::symlink_metadata(root).map_err(|error| {
        validation_error(vec![BundleIssue {
            path: "$.root".into(),
            message: format!("cannot inspect bundle root: {error}"),
        }])
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(validation_error(vec![BundleIssue {
            path: "$.root".into(),
            message: "bundle root must be a real directory, not a symlink".into(),
        }]));
    }

    let mut found = BTreeSet::new();
    let mut issues = Vec::new();
    let entries = fs::read_dir(root).map_err(|error| {
        validation_error(vec![BundleIssue {
            path: "$.root".into(),
            message: format!("cannot read bundle root: {error}"),
        }])
    })?;
    for entry in entries {
        match entry {
            Ok(entry) => {
                let name = entry.file_name().to_string_lossy().into_owned();
                found.insert(name.clone());
                if !ROOT_ENTRIES.contains(&name.as_str()) {
                    push_issue(
                        &mut issues,
                        format!("$.root.{name}"),
                        format!("unexpected root entry `{name}`"),
                    );
                }
            }
            Err(error) => push_issue(&mut issues, "$.root", format!("cannot inspect root entry: {error}")),
        }
    }
    for required in ROOT_ENTRIES {
        if !found.contains(required) {
            push_issue(
                &mut issues,
                format!("$.root.{required}"),
                format!("missing required root entry `{required}`"),
            );
        }
    }
    if !issues.is_empty() {
        return Err(validation_error(issues));
    }

    for file in ["manifest.json", "SHA256SUMS"] {
        validate_entry_kind(&root.join(file), file, false, &mut issues);
    }
    for directory in ["raw", "derived"] {
        validate_entry_kind(&root.join(directory), directory, true, &mut issues);
    }
    fail_if_any(issues)
}

fn validate_entry_kind(path: &Path, name: &str, directory: bool, issues: &mut Vec<BundleIssue>) {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            let wrong_kind = metadata.file_type().is_symlink()
                || if directory {
                    !metadata.is_dir()
                } else {
                    !metadata.is_file()
                };
            if wrong_kind {
                push_issue(
                    issues,
                    format!("$.root.{name}"),
                    if directory {
                        "required entry must be a real directory, not a symlink"
                    } else {
                        "required entry must be a regular file, not a symlink"
                    },
                );
            }
        }
        Err(error) => {
            push_issue(issues, format!("$.root.{name}"), format!("cannot inspect required entry: {error}"))
        }
    }
}

fn collect_artifacts(root: &Path) -> Result<BTreeSet<String>, BundleValidationError> {
    let mut files = BTreeSet::new();
    let mut issues = Vec::new();
    let mut pending = vec![root.join("raw"), root.join("derived")];
    while let Some(directory) = pending.pop() {
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) => {
                push_issue(
                    &mut issues,
                    display_relative(root, &directory),
                    format!("cannot read artifact directory: {error}"),
                );
                continue;
            }
        };
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    push_issue(
                        &mut issues,
                        display_relative(root, &directory),
                        format!("cannot inspect artifact entry: {error}"),
                    );
                    continue;
                }
            };
            let path = entry.path();
            let relative = match path.strip_prefix(root).ok().and_then(Path::to_str) {
                Some(relative) => relative.replace(std::path::MAIN_SEPARATOR, "/"),
                None => {
                    push_issue(&mut issues, "$.tree", "artifact path is not canonical UTF-8");
                    continue;
                }
            };
            match fs::symlink_metadata(&path) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    push_issue(&mut issues, format!("$.tree.{relative}"), "symlink is forbidden");
                }
                Ok(metadata) if metadata.is_dir() => pending.push(path),
                Ok(metadata) if metadata.is_file() => {
                    files.insert(relative);
                }
                Ok(_) => push_issue(
                    &mut issues,
                    format!("$.tree.{relative}"),
                    "artifact entry must be a regular file or directory",
                ),
                Err(error) => push_issue(
                    &mut issues,
                    format!("$.tree.{relative}"),
                    format!("cannot inspect artifact entry: {error}"),
                ),
            }
        }
    }
    if issues.is_empty() {
        Ok(files)
    } else {
        Err(validation_error(issues))
    }
}

fn hash_file(path: &Path) -> io::Result<(u64, String)> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "artifact is not a regular file"));
    }
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        size += read as u64;
        hasher.update(&buffer[..read]);
    }
    Ok((size, format!("{:x}", hasher.finalize())))
}

fn read_file(path: &Path, issue_path: &str) -> Result<Vec<u8>, BundleValidationError> {
    fs::read(path).map_err(|error| {
        validation_error(vec![BundleIssue {
            path: issue_path.into(),
            message: format!("cannot read file: {error}"),
        }])
    })
}

fn display_relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

fn fail_if_any(issues: Vec<BundleIssue>) -> Result<(), BundleValidationError> {
    if issues.is_empty() {
        Ok(())
    } else {
        Err(validation_error(issues))
    }
}

fn validation_error(mut issues: Vec<BundleIssue>) -> BundleValidationError {
    issues.sort();
    issues.dedup();
    BundleValidationError { issues }
}

fn push_issue(issues: &mut Vec<BundleIssue>, path: impl Into<String>, message: impl Into<String>) {
    issues.push(BundleIssue { path: path.into(), message: message.into() });
}
