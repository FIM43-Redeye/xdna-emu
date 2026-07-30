use super::{
    build_canonical_bundle, validate_bundle, ArtifactRecord, ArtifactSource, EmissionPlan, ValidatedBundle,
};
use sha2::{Digest, Sha256};
use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Component, Path},
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
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !matches!(output.components().next_back(), Some(Component::Normal(_))) {
        return Err(emission_error("output must have an ordinary final path name"));
    }
    let parent_metadata = fs::metadata(parent)
        .map_err(|error| emission_error(format!("cannot inspect output parent: {error}")))?;
    if !parent_metadata.is_dir() {
        return Err(emission_error("output parent must be an existing directory"));
    }
    require_absent(output)?;

    let staging = tempfile::Builder::new()
        .prefix(".xdna-reserve-")
        .tempdir_in(parent)
        .map_err(|error| emission_error(format!("cannot create staging directory: {error}")))?;
    fs::create_dir(staging.path().join("raw"))
        .map_err(|error| emission_error(format!("cannot create raw directory: {error}")))?;
    fs::create_dir(staging.path().join("derived"))
        .map_err(|error| emission_error(format!("cannot create derived directory: {error}")))?;

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
