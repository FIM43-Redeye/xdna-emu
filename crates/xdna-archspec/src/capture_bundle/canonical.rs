use super::{
    ArtifactRecord, BundleManifest, BundleSchemaError, Campaign, ComponentPin, InputIdentity, RevisionPin,
    RunRecord, MANIFEST_SCHEMA_VERSION,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct CanonicalBundle {
    manifest: BundleManifest,
    manifest_bytes: Vec<u8>,
    checksum_index_bytes: Vec<u8>,
    manifest_sha256: String,
    checksum_index_sha256: String,
}

impl CanonicalBundle {
    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    pub fn bundle_id(&self) -> &str {
        &self.manifest.bundle_id
    }

    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest_bytes
    }

    pub fn checksum_index_bytes(&self) -> &[u8] {
        &self.checksum_index_bytes
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn checksum_index_sha256(&self) -> &str {
        &self.checksum_index_sha256
    }
}

#[derive(Serialize)]
struct ManifestPreimage<'a> {
    schema_version: u32,
    campaign: &'a Campaign,
    artifacts: &'a [ArtifactRecord],
}

pub fn build_canonical_bundle(
    campaign: Campaign,
    artifacts: Vec<ArtifactRecord>,
) -> Result<CanonicalBundle, BundleSchemaError> {
    canonicalize_manifest(&BundleManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        bundle_id: format!("bundle.sha256.{}", "0".repeat(64)),
        campaign,
        artifacts,
    })
}

pub fn canonicalize_manifest(manifest: &BundleManifest) -> Result<CanonicalBundle, BundleSchemaError> {
    manifest.validate()?;

    let mut campaign = manifest.campaign.clone();
    let mut artifacts = manifest.artifacts.clone();
    canonicalize_campaign(&mut campaign);
    canonicalize_artifacts(&mut artifacts);

    let preimage_bytes = canonical_json(&ManifestPreimage {
        schema_version: manifest.schema_version,
        campaign: &campaign,
        artifacts: &artifacts,
    });
    let bundle_id = format!("bundle.sha256.{}", sha256_bytes(&preimage_bytes));
    let manifest = BundleManifest { schema_version: manifest.schema_version, bundle_id, campaign, artifacts };
    let manifest_bytes = canonical_json(&manifest);
    let checksum_index_bytes = checksum_index(&manifest.artifacts);
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let checksum_index_sha256 = sha256_bytes(&checksum_index_bytes);

    Ok(CanonicalBundle {
        manifest,
        manifest_bytes,
        checksum_index_bytes,
        manifest_sha256,
        checksum_index_sha256,
    })
}

pub(crate) fn canonical_json(value: &impl Serialize) -> Vec<u8> {
    let mut bytes = serde_json::to_vec_pretty(value).expect("typed capture-bundle data must serialize");
    bytes.push(b'\n');
    bytes
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn checksum_index(artifacts: &[ArtifactRecord]) -> Vec<u8> {
    let mut output = String::new();
    for artifact in artifacts {
        writeln!(output, "{}  {}", artifact.sha256, artifact.path).expect("writing to String cannot fail");
    }
    output.into_bytes()
}

fn canonicalize_campaign(campaign: &mut Campaign) {
    campaign.tuple_ids.sort();
    campaign.inventory_ids.sort();
    campaign.fact_ids.sort();
    campaign.evidence_ids.sort();
    sort_components(&mut campaign.platform.kernel_modules);
    sort_components(&mut campaign.platform.xrt_components);
    sort_components(&mut campaign.platform.toolchain_components);
    campaign.stimulus.source_revisions.sort_by(revision_order);
    campaign.stimulus.inputs.sort_by(input_order);
    campaign.stimulus.initial_state.sort();
    for run in &mut campaign.runs {
        canonicalize_run(run);
    }
    campaign.runs.sort_by(|left, right| {
        (left.ordinal, left.repetition, &left.id).cmp(&(right.ordinal, right.repetition, &right.id))
    });
}

fn sort_components(components: &mut [ComponentPin]) {
    components.sort_by(|left, right| (&left.name, &left.revision).cmp(&(&right.name, &right.revision)));
}

fn revision_order(left: &RevisionPin, right: &RevisionPin) -> std::cmp::Ordering {
    (&left.repository, &left.commit).cmp(&(&right.repository, &right.commit))
}

fn input_order(left: &InputIdentity, right: &InputIdentity) -> std::cmp::Ordering {
    left.id.cmp(&right.id)
}

fn canonicalize_run(run: &mut RunRecord) {
    run.output_artifact_paths.sort();
    for observation in &mut run.observations {
        observation.artifact_paths.sort();
    }
    run.observations.sort_by(|left, right| left.id.cmp(&right.id));
    run.timing.sort_by(|left, right| {
        (&left.anchor, left.lower.value, &left.lower.unit, left.upper.value, &left.upper.unit).cmp(&(
            &right.anchor,
            right.lower.value,
            &right.lower.unit,
            right.upper.value,
            &right.upper.unit,
        ))
    });
    run.control_run_ids.sort();
}

fn canonicalize_artifacts(artifacts: &mut [ArtifactRecord]) {
    for artifact in artifacts.iter_mut() {
        artifact.run_ids.sort();
        artifact.observation_ids.sort();
        if let Some(derivation) = &mut artifact.derivation {
            derivation.source_artifact_paths.sort();
            derivation.source_bundle_ids.sort();
        }
    }
    artifacts.sort_by(|left, right| left.path.cmp(&right.path));
}
