use super::{
    build_canonical_bundle, build_canonical_bundle_v2, validate_bundle, validate_bundle_graph,
    ArtifactRecord, ArtifactSource, EmissionPlan, EmissionPlanV2, ValidatedBundle,
};
use crate::research_reserve::{BundleLocationEntry, BundleLocationRoot};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
};

#[derive(Debug)]
pub struct BundleEmissionError {
    message: String,
}

impl fmt::Display for BundleEmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BundleEmissionError {}

pub fn emit_bundle(
    plan: &EmissionPlan,
    output: impl AsRef<Path>,
) -> Result<ValidatedBundle, BundleEmissionError> {
    emit_bundle_with(plan, output.as_ref(), |_| Ok(()))
}

pub fn emit_bundle_v2(
    plan: &EmissionPlanV2,
    output: impl AsRef<Path>,
) -> Result<ValidatedBundle, BundleEmissionError> {
    let output = output.as_ref();
    plan.validate().map_err(|error| emission_error(error.to_string()))?;
    let parent = output_parent(output)?;
    let staging = create_staging(parent)?;
    create_payload_directories(staging.path())?;

    let mut records = Vec::with_capacity(plan.artifacts.len());
    for artifact in &plan.artifacts {
        records.push(copy_artifact(artifact, staging.path())?);
    }
    let canonical = build_canonical_bundle_v2(
        plan.payload.clone(),
        plan.dependencies
            .iter()
            .map(|dependency| dependency.requirement.clone())
            .collect(),
        records,
    )
    .map_err(|error| emission_error(error.to_string()))?;
    write_new(&staging.path().join("manifest.json"), canonical.manifest_bytes())?;
    write_new(&staging.path().join("SHA256SUMS"), canonical.checksum_index_bytes())?;

    let staged_locations = graph_locations(canonical.bundle_id(), staging.path(), &plan.dependencies)?;
    let staged = validate_bundle_graph(staging.path(), &staged_locations)
        .map_err(|error| emission_error(format!("staging graph validation failed: {error}")))?;

    if fs::symlink_metadata(output).is_ok() {
        let existing_locations = graph_locations(canonical.bundle_id(), output, &plan.dependencies)?;
        let existing = validate_bundle_graph(output, &existing_locations)
            .map_err(|error| emission_error(format!("existing bundle validation failed: {error}")))?;
        if existing.root().bundle_id() != staged.root().bundle_id()
            || existing.root().manifest_sha256() != staged.root().manifest_sha256()
            || existing.root().checksum_index_sha256() != staged.root().checksum_index_sha256()
        {
            return Err(emission_error(format!(
                "output path `{}` contains a different bundle",
                output.display()
            )));
        }
        return Ok(existing.into_root());
    }
    require_absent(output)?;
    fs::rename(staging.path(), output)
        .map_err(|error| emission_error(format!("cannot publish bundle: {error}")))?;
    Ok(staged.into_root())
}

#[cfg(test)]
pub(super) fn emit_bundle_with_test_mutation(
    plan: &EmissionPlan,
    output: &Path,
    mutation: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<ValidatedBundle, BundleEmissionError> {
    emit_bundle_with(plan, output, mutation)
}

fn emit_bundle_with(
    plan: &EmissionPlan,
    output: &Path,
    before_validation: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<ValidatedBundle, BundleEmissionError> {
    plan.validate().map_err(|error| emission_error(error.to_string()))?;
    let parent = output_parent(output)?;
    require_absent(output)?;

    let staging = create_staging(parent)?;
    create_payload_directories(staging.path())?;

    let mut records = Vec::with_capacity(plan.artifacts.len());
    for artifact in &plan.artifacts {
        records.push(copy_artifact(artifact, staging.path())?);
    }
    let canonical = build_canonical_bundle(plan.campaign.clone(), records)
        .map_err(|error| emission_error(error.to_string()))?;
    write_new(&staging.path().join("manifest.json"), canonical.manifest_bytes())?;
    write_new(&staging.path().join("SHA256SUMS"), canonical.checksum_index_bytes())?;

    before_validation(staging.path())
        .map_err(|error| emission_error(format!("staging mutation failed: {error}")))?;
    let validated = validate_bundle(staging.path())
        .map_err(|error| emission_error(format!("staging validation failed: {error}")))?;
    require_absent(output)?;
    fs::rename(staging.path(), output)
        .map_err(|error| emission_error(format!("cannot publish bundle: {error}")))?;
    Ok(validated)
}

fn output_parent(output: &Path) -> Result<&Path, BundleEmissionError> {
    if !matches!(output.components().next_back(), Some(Component::Normal(_))) {
        return Err(emission_error("output must have an ordinary final path name"));
    }
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let metadata = fs::metadata(parent)
        .map_err(|error| emission_error(format!("cannot inspect output parent: {error}")))?;
    if !metadata.is_dir() {
        return Err(emission_error("output parent must be an existing directory"));
    }
    Ok(parent)
}

fn create_staging(parent: &Path) -> Result<tempfile::TempDir, BundleEmissionError> {
    tempfile::Builder::new()
        .prefix(".xdna-reserve-")
        .tempdir_in(parent)
        .map_err(|error| emission_error(format!("cannot create staging directory: {error}")))
}

fn create_payload_directories(root: &Path) -> Result<(), BundleEmissionError> {
    fs::create_dir(root.join("raw"))
        .map_err(|error| emission_error(format!("cannot create raw directory: {error}")))?;
    fs::create_dir(root.join("derived"))
        .map_err(|error| emission_error(format!("cannot create derived directory: {error}")))
}

fn graph_locations(
    root_id: &str,
    root_path: &Path,
    dependencies: &[super::DependencySource],
) -> Result<BundleLocationRoot, BundleEmissionError> {
    let mut paths = BTreeMap::from([(root_id.to_owned(), canonical_path(root_path)?)]);
    for dependency in dependencies {
        let path = canonical_path(&dependency.source_path)?;
        match paths.get(&dependency.requirement.fixture_bundle_id) {
            Some(existing) if existing != &path => {
                return Err(emission_error(format!(
                    "fixture {} has conflicting source paths",
                    dependency.requirement.fixture_bundle_id
                )));
            }
            Some(_) => {}
            None => {
                paths.insert(dependency.requirement.fixture_bundle_id.clone(), path);
            }
        }
    }
    let mut common = paths
        .values()
        .next()
        .and_then(|path| path.parent())
        .map(Path::to_owned)
        .ok_or_else(|| emission_error("cannot determine graph location root"))?;
    while !paths.values().all(|path| path.starts_with(&common)) {
        common = common
            .parent()
            .map(Path::to_owned)
            .ok_or_else(|| emission_error("bundle graph paths have no common root"))?;
    }
    let bundles = paths
        .into_iter()
        .map(|(bundle_id, path)| {
            let relative = path
                .strip_prefix(&common)
                .expect("common graph root prefixes every path")
                .to_str()
                .ok_or_else(|| emission_error("bundle graph path is not UTF-8"))?;
            Ok(BundleLocationEntry { bundle_id, relative_path: relative.into() })
        })
        .collect::<Result<_, BundleEmissionError>>()?;
    Ok(BundleLocationRoot {
        alias: "emission".into(),
        path: common,
        failure_domain_id: "emission.validation".into(),
        bundles,
    })
}

fn canonical_path(path: &Path) -> Result<PathBuf, BundleEmissionError> {
    fs::canonicalize(path)
        .map_err(|error| emission_error(format!("cannot resolve bundle path `{}`: {error}", path.display())))
}

fn copy_artifact(artifact: &ArtifactSource, staging: &Path) -> Result<ArtifactRecord, BundleEmissionError> {
    let metadata = fs::symlink_metadata(&artifact.source_path).map_err(|error| {
        emission_error(format!(
            "cannot inspect artifact source `{}`: {error}",
            artifact.source_path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(emission_error(format!(
            "artifact source `{}` must be a regular file, not a symlink",
            artifact.source_path.display()
        )));
    }

    let destination = staging.join(&artifact.path);
    let parent = destination
        .parent()
        .expect("validated artifact destinations always have a parent");
    fs::create_dir_all(parent).map_err(|error| {
        emission_error(format!("cannot create artifact destination `{}`: {error}", parent.display()))
    })?;
    let (byte_size, sha256) = copy_and_hash(&artifact.source_path, &destination)?;
    Ok(ArtifactRecord {
        path: artifact.path.clone(),
        byte_size,
        sha256,
        semantic_kind: artifact.semantic_kind.clone(),
        class: artifact.class,
        redistributability: artifact.redistributability,
        run_ids: artifact.run_ids.clone(),
        observation_ids: artifact.observation_ids.clone(),
        derivation: artifact.derivation.clone(),
    })
}

fn copy_and_hash(source: &Path, destination: &Path) -> Result<(u64, String), BundleEmissionError> {
    let mut source = File::open(source)
        .map_err(|error| emission_error(format!("cannot open artifact source: {error}")))?;
    let mut destination = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|error| emission_error(format!("cannot create artifact destination: {error}")))?;
    let mut hasher = Sha256::new();
    let mut byte_size = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = source
            .read(&mut buffer)
            .map_err(|error| emission_error(format!("cannot read artifact source: {error}")))?;
        if read == 0 {
            break;
        }
        destination
            .write_all(&buffer[..read])
            .map_err(|error| emission_error(format!("cannot write artifact destination: {error}")))?;
        hasher.update(&buffer[..read]);
        byte_size += read as u64;
    }
    Ok((byte_size, format!("{:x}", hasher.finalize())))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), BundleEmissionError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| emission_error(format!("cannot create `{}`: {error}", path.display())))?;
    file.write_all(bytes)
        .map_err(|error| emission_error(format!("cannot write `{}`: {error}", path.display())))
}

fn require_absent(path: &Path) -> Result<(), BundleEmissionError> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(emission_error(format!("cannot inspect output path: {error}"))),
        Ok(_) => Err(emission_error(format!("output path `{}` already exists", path.display()))),
    }
}

fn emission_error(message: impl Into<String>) -> BundleEmissionError {
    BundleEmissionError { message: message.into() }
}
