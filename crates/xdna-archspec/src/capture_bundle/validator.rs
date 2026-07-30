use super::{
    canonical::{canonical_json, sha256_bytes},
    canonicalize_manifest, BundleIssue, BundleManifest,
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fmt,
    fs::{self, File},
    io::{self, Read},
    path::Path,
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
    manifest: BundleManifest,
    manifest_sha256: String,
    checksum_index_sha256: String,
    promotion_blockers: Vec<BundleIssue>,
}

impl ValidatedBundle {
    pub fn bundle_id(&self) -> &str {
        &self.manifest.bundle_id
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

    pub(crate) fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }
}

pub fn validate_bundle(root: impl AsRef<Path>) -> Result<ValidatedBundle, BundleValidationError> {
    let root = root.as_ref();
    validate_root(root)?;

    let manifest_bytes = read_file(&root.join("manifest.json"), "$.manifest")?;
    let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes).map_err(|error| {
        validation_error(vec![BundleIssue { path: "$.manifest".into(), message: error.to_string() }])
    })?;
    let eligibility = manifest.validate().map_err(|error| validation_error(error.issues().to_vec()))?;
    let canonical =
        canonicalize_manifest(&manifest).map_err(|error| validation_error(error.issues().to_vec()))?;
    let canonical_authored =
        BundleManifest { bundle_id: manifest.bundle_id.clone(), ..canonical.manifest().clone() };
    if manifest_bytes != canonical_json(&canonical_authored) {
        return Err(validation_error(vec![BundleIssue {
            path: "$.manifest".into(),
            message: "manifest.json is not canonical".into(),
        }]));
    }

    let actual_artifacts = collect_artifacts(root)?;
    let declared_artifacts: BTreeSet<String> =
        manifest.artifacts.iter().map(|artifact| artifact.path.clone()).collect();
    let mut issues = Vec::new();
    for path in actual_artifacts.difference(&declared_artifacts) {
        push_issue(&mut issues, format!("$.tree.{path}"), "undeclared artifact");
    }
    for path in declared_artifacts.difference(&actual_artifacts) {
        push_issue(&mut issues, format!("$.artifacts.{path}"), "declared artifact is missing");
    }
    fail_if_any(issues)?;

    let mut issues = Vec::new();
    for artifact in &canonical.manifest().artifacts {
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
    if checksum_index_bytes != canonical.checksum_index_bytes() {
        return Err(validation_error(vec![BundleIssue {
            path: "$.checksum_index".into(),
            message: "SHA256SUMS is not canonical".into(),
        }]));
    }
    if manifest.bundle_id != canonical.bundle_id() {
        return Err(validation_error(vec![BundleIssue {
            path: "$.manifest.bundle_id".into(),
            message: format!(
                "bundle ID mismatch: declared {}, recomputed {}",
                manifest.bundle_id,
                canonical.bundle_id()
            ),
        }]));
    }

    Ok(ValidatedBundle {
        manifest: canonical.manifest().clone(),
        manifest_sha256: sha256_bytes(&manifest_bytes),
        checksum_index_sha256: sha256_bytes(&checksum_index_bytes),
        promotion_blockers: eligibility.blockers().to_vec(),
    })
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
