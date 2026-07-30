use super::{
    build_canonical_bundle, canonicalize_manifest, BundleManifest, EmissionPlan,
    EMISSION_PLAN_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION,
};
use serde_json::{json, Value};
use std::path::PathBuf;

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn known(value: Value) -> Value {
    json!({ "state": "known", "value": value })
}

fn valid_campaign(architecture: &str, device_model_key: &str, driver_platform_id: &str) -> Value {
    json!({
        "id": "campaign.synthetic.capture",
        "tuple_ids": ["tuple.synthetic.device"],
        "inventory_ids": ["inventory.synthetic.firmware"],
        "fact_ids": ["fact.synthetic.lifecycle"],
        "evidence_ids": ["evidence.synthetic.capture"],
        "provenance": "current",
        "risk_class": "non_destructive",
        "outcome": "success",
        "platform": {
            "architecture": architecture,
            "device_model_key": known(json!(device_model_key)),
            "driver_platform_id": known(json!(driver_platform_id)),
            "pci": known(json!({
                "vendor_id": "1022",
                "device_id": "1502",
                "subsystem_vendor_id": "1022",
                "subsystem_device_id": "1502",
                "revision_id": "00"
            })),
            "board_identity": known(json!("integrated.reference")),
            "firmware": known(json!({
                "logical_name": "npu.sbin",
                "sha256": SHA_A
            })),
            "host_kernel": known(json!({
                "name": "linux",
                "revision": "v6.18-test",
                "sha256": known(json!(SHA_A))
            })),
            "kernel_modules": [{
                "name": "amdxdna",
                "revision": "commit.synthetic",
                "sha256": known(json!(SHA_B))
            }],
            "driver": known(json!({
                "repository": "https://example.invalid/xdna-driver",
                "commit": "commit.synthetic"
            })),
            "xrt_components": [{
                "name": "xrt-base",
                "revision": "2.23.0",
                "sha256": known(json!(SHA_A))
            }],
            "toolchain_components": [{
                "name": "mlir-aie",
                "revision": "commit.synthetic",
                "sha256": known(json!(SHA_B))
            }],
            "compiler_mode": known(json!("peano")),
            "execution_mode": known(json!("direct")),
            "reset_state": known(json!("cold")),
            "power_state": known(json!("d0")),
            "clock_state": known(json!("default")),
            "iommu_state": known(json!("enabled")),
            "address_state": known(json!("canonical"))
        },
        "stimulus": {
            "command": {
                "argv": ["runner", "--case", "synthetic"],
                "environment": {
                    "LANG": "C"
                }
            },
            "source_revisions": [{
                "repository": "https://example.invalid/fixture",
                "commit": "commit.synthetic"
            }],
            "build_recipe": known(json!({
                "logical_name": "build-recipe.json",
                "sha256": SHA_B
            })),
            "inputs": [{
                "id": "input.synthetic.payload",
                "semantic_kind": "input.binary",
                "content": {
                    "logical_name": "input.bin",
                    "sha256": SHA_A
                }
            }],
            "initial_state": ["memory.zeroed"],
            "external_events": [{
                "id": "event.synthetic.start",
                "ordinal": 0,
                "description": "start command",
                "offset": {
                    "value": 0,
                    "unit": "cycles"
                }
            }]
        },
        "runs": [{
            "id": "run.synthetic.0",
            "ordinal": 0,
            "repetition": 0,
            "completion": "complete",
            "output_artifact_paths": ["raw/output.bin"],
            "observations": [{
                "id": "observation.synthetic.output",
                "semantic_kind": "memory.snapshot",
                "artifact_paths": ["raw/output.bin"]
            }],
            "timing": [{
                "anchor": "command.start",
                "lower": {
                    "value": 0,
                    "unit": "cycles"
                },
                "upper": {
                    "value": 10,
                    "unit": "cycles"
                }
            }],
            "errors": [],
            "recovery_actions": [],
            "teardown": "clean",
            "control_run_ids": []
        }]
    })
}

fn valid_artifacts() -> Value {
    json!([
        {
            "path": "raw/output.bin",
            "byte_size": 3,
            "sha256": SHA_A,
            "semantic_kind": "memory.snapshot",
            "class": "raw",
            "redistributability": "redistributable",
            "run_ids": ["run.synthetic.0"],
            "observation_ids": ["observation.synthetic.output"],
            "derivation": null
        },
        {
            "path": "derived/output.txt",
            "byte_size": 4,
            "sha256": SHA_B,
            "semantic_kind": "analysis.summary",
            "class": "derived",
            "redistributability": "redistributable",
            "run_ids": ["run.synthetic.0"],
            "observation_ids": [],
            "derivation": {
                "source_artifact_paths": ["raw/output.bin"],
                "source_bundle_ids": [],
                "command": {
                    "argv": ["analyze", "raw/output.bin"],
                    "environment": {}
                },
                "analysis_tool": {
                    "repository": "https://example.invalid/analyzer",
                    "commit": "commit.synthetic"
                }
            }
        }
    ])
}

fn valid_manifest_value(architecture: &str, device_model_key: &str, driver_platform_id: &str) -> Value {
    json!({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "bundle_id": format!("bundle.sha256.{SHA_A}"),
        "campaign": valid_campaign(architecture, device_model_key, driver_platform_id),
        "artifacts": valid_artifacts()
    })
}

fn valid_manifest() -> BundleManifest {
    serde_json::from_value(valid_manifest_value("aie2", "npu1", "npu1")).unwrap()
}

fn valid_plan_value() -> Value {
    let mut artifacts = valid_artifacts().as_array().unwrap().clone();
    for (index, artifact) in artifacts.iter_mut().enumerate() {
        let object = artifact.as_object_mut().unwrap();
        object.remove("byte_size");
        object.remove("sha256");
        object.insert("source_path".into(), json!(format!("/capture/source-{index}")));
    }
    json!({
        "schema_version": EMISSION_PLAN_SCHEMA_VERSION,
        "campaign": valid_campaign("aie2", "npu1", "npu1"),
        "artifacts": artifacts
    })
}

fn issue_paths(error: &super::BundleSchemaError) -> Vec<&str> {
    error.issues().iter().map(|issue| issue.path.as_str()).collect()
}

#[test]
fn schema_rejects_unknown_manifest_and_plan_fields() {
    let mut manifest = valid_manifest_value("aie2", "npu1", "npu1");
    manifest.as_object_mut().unwrap().insert("surprise".into(), json!(true));
    assert!(serde_json::from_value::<BundleManifest>(manifest).is_err());

    let mut plan = valid_plan_value();
    plan.as_object_mut().unwrap().insert("surprise".into(), json!(true));
    assert!(serde_json::from_value::<EmissionPlan>(plan).is_err());
}

#[test]
fn schema_rejects_unsupported_versions() {
    let mut manifest = valid_manifest();
    manifest.schema_version = MANIFEST_SCHEMA_VERSION + 1;
    assert_eq!(issue_paths(&manifest.validate().unwrap_err()), ["$.schema_version"]);

    let mut plan: EmissionPlan = serde_json::from_value(valid_plan_value()).unwrap();
    plan.schema_version = EMISSION_PLAN_SCHEMA_VERSION + 1;
    assert_eq!(issue_paths(&plan.validate().unwrap_err()), ["$.schema_version"]);
}

#[test]
fn schema_round_trips_aie2_and_aie2p_without_device_specific_shapes() {
    for (architecture, model, platform) in [("aie2", "npu1", "npu1"), ("aie2p", "npu5_8col", "npu5")] {
        let manifest: BundleManifest =
            serde_json::from_value(valid_manifest_value(architecture, model, platform)).unwrap();
        assert!(manifest.validate().unwrap().is_promotion_eligible());
        assert_eq!(
            serde_json::to_value(manifest).unwrap()["campaign"]["platform"]["architecture"],
            architecture
        );
    }
}

#[test]
fn schema_keeps_compiler_and_execution_modes_in_platform_identity() {
    let value = valid_manifest_value("aie2", "npu1", "npu1");
    assert!(serde_json::from_value::<BundleManifest>(value)
        .unwrap()
        .validate()
        .unwrap()
        .is_promotion_eligible());
}

#[test]
fn schema_rejects_absent_required_fields() {
    let mut value = valid_manifest_value("aie2", "npu1", "npu1");
    value["campaign"].as_object_mut().unwrap().remove("platform");
    assert!(serde_json::from_value::<BundleManifest>(value).is_err());
}

#[test]
fn schema_accepts_unavailable_identity_but_blocks_promotion() {
    let mut value = valid_manifest_value("aie2", "npu1", "npu1");
    value["campaign"]["platform"]["device_model_key"] =
        json!({ "state": "unavailable", "reason": "not recorded" });
    let manifest: BundleManifest = serde_json::from_value(value).unwrap();
    let eligibility = manifest.validate().unwrap();
    assert!(!eligibility.is_promotion_eligible());
    assert_eq!(eligibility.blockers()[0].path, "$.campaign.platform.device_model_key");
}

#[test]
fn schema_allows_not_applicable_only_where_declared() {
    let mut valid = valid_manifest_value("aie2", "npu1", "npu1");
    valid["campaign"]["platform"]["board_identity"] =
        json!({ "state": "not_applicable", "reason": "integrated package" });
    assert!(serde_json::from_value::<BundleManifest>(valid)
        .unwrap()
        .validate()
        .unwrap()
        .is_promotion_eligible());

    let mut invalid = valid_manifest_value("aie2", "npu1", "npu1");
    invalid["campaign"]["platform"]["device_model_key"] =
        json!({ "state": "not_applicable", "reason": "unknown device" });
    assert_eq!(
        issue_paths(
            &serde_json::from_value::<BundleManifest>(invalid)
                .unwrap()
                .validate()
                .unwrap_err()
        ),
        ["$.campaign.platform.device_model_key"]
    );
}

#[test]
fn schema_rejects_duplicate_and_dangling_ids() {
    let mut manifest = valid_manifest();
    manifest.campaign.tuple_ids.push("tuple.synthetic.device".into());
    manifest.campaign.runs.push(manifest.campaign.runs[0].clone());
    manifest.campaign.runs[0].control_run_ids.push("run.synthetic.missing".into());

    assert_eq!(
        issue_paths(&manifest.validate().unwrap_err()),
        [
            "$.campaign.runs[0].control_run_ids[0]",
            "$.campaign.runs[1].id",
            "$.campaign.runs[1].observations[0].id",
            "$.campaign.tuple_ids[1]"
        ]
    );
}

#[test]
fn schema_rejects_dangling_artifact_run_and_observation_ids() {
    let mut manifest = valid_manifest();
    manifest.artifacts[0].run_ids = vec!["run.synthetic.missing".into()];
    manifest.artifacts[0].observation_ids = vec!["observation.synthetic.missing".into()];

    assert_eq!(
        issue_paths(&manifest.validate().unwrap_err()),
        ["$.artifacts[0].observation_ids[0]", "$.artifacts[0].run_ids[0]"]
    );
}

#[test]
fn schema_unavailable_nested_component_hash_blocks_promotion() {
    let mut manifest = valid_manifest();
    manifest.campaign.platform.host_kernel = super::Availability::Known {
        value: super::ComponentPin {
            name: "linux".into(),
            revision: "v6.18-test".into(),
            sha256: super::Availability::Unavailable { reason: "not recorded".into() },
        },
    };

    let eligibility = manifest.validate().unwrap();
    assert!(!eligibility.is_promotion_eligible());
    assert_eq!(eligibility.blockers()[0].path, "$.campaign.platform.host_kernel.sha256");
}

#[test]
fn schema_rejects_unsafe_duplicate_and_wrong_root_artifact_paths() {
    for (path, expected_path) in [
        ("raw/../escape.bin", "$.artifacts[0].path"),
        ("raw\\escape.bin", "$.artifacts[0].path"),
        ("raw//escape.bin", "$.artifacts[0].path"),
        ("derived/output.bin", "$.artifacts[0].path"),
    ] {
        let mut manifest = valid_manifest();
        manifest.artifacts[0].path = path.into();
        assert!(issue_paths(&manifest.validate().unwrap_err()).contains(&expected_path), "{path}");
    }

    let mut duplicate = valid_manifest();
    duplicate.artifacts[1].path = "raw/output.bin".into();
    assert!(issue_paths(&duplicate.validate().unwrap_err()).contains(&"$.artifacts[1].path"));
}

#[test]
fn schema_rejects_raw_and_derived_provenance_mismatches() {
    let mut raw = valid_manifest();
    raw.artifacts[0].derivation = raw.artifacts[1].derivation.clone();
    assert!(issue_paths(&raw.validate().unwrap_err()).contains(&"$.artifacts[0].derivation"));

    let mut derived = valid_manifest();
    derived.artifacts[1].derivation = None;
    assert!(issue_paths(&derived.validate().unwrap_err()).contains(&"$.artifacts[1].derivation"));
}

#[test]
fn schema_rejects_malformed_hashes_and_blank_identity_text() {
    let mut manifest = valid_manifest();
    manifest.artifacts[0].sha256 = "A".repeat(64);
    manifest.campaign.platform.kernel_modules[0].name = " ".into();

    assert_eq!(
        issue_paths(&manifest.validate().unwrap_err()),
        ["$.artifacts[0].sha256", "$.campaign.platform.kernel_modules[0].name"]
    );
}

#[test]
fn canonical_authored_order_does_not_change_bytes_or_identity() {
    let mut ordered = valid_manifest();
    ordered.campaign.tuple_ids.push("tuple.synthetic.control".into());
    ordered.campaign.platform.toolchain_components.push(super::ComponentPin {
        name: "aie-rt".into(),
        revision: "commit.control".into(),
        sha256: super::Availability::Known { value: SHA_A.into() },
    });
    let mut second_run = ordered.campaign.runs[0].clone();
    second_run.id = "run.synthetic.1".into();
    second_run.ordinal = 1;
    second_run.observations[0].id = "observation.synthetic.control".into();
    ordered.campaign.runs.push(second_run);

    let mut reversed = ordered.clone();
    reversed.artifacts.reverse();
    reversed.campaign.tuple_ids.reverse();
    reversed.campaign.platform.toolchain_components.reverse();
    reversed.campaign.runs.reverse();

    let ordered = canonicalize_manifest(&ordered).unwrap();
    let reversed = canonicalize_manifest(&reversed).unwrap();
    assert_eq!(ordered.bundle_id(), reversed.bundle_id());
    assert_eq!(ordered.manifest_bytes(), reversed.manifest_bytes());
    assert_eq!(ordered.checksum_index_bytes(), reversed.checksum_index_bytes());
}

#[test]
fn canonical_source_paths_do_not_enter_the_preimage() {
    let mut first: EmissionPlan = serde_json::from_value(valid_plan_value()).unwrap();
    let mut second = first.clone();
    first.artifacts[0].source_path = PathBuf::from("/first/source.bin");
    second.artifacts[0].source_path = PathBuf::from("/elsewhere/source.bin");

    let artifacts = valid_manifest().artifacts;
    let first = build_canonical_bundle(first.campaign, artifacts.clone()).unwrap();
    let second = build_canonical_bundle(second.campaign, artifacts).unwrap();
    assert_eq!(first.bundle_id(), second.bundle_id());
    assert_eq!(first.manifest_bytes(), second.manifest_bytes());
}

#[test]
fn canonical_identity_changes_with_metadata_or_artifact_identity() {
    let baseline = valid_manifest();
    let baseline_id = canonicalize_manifest(&baseline).unwrap().bundle_id().to_owned();

    let mut metadata = baseline.clone();
    metadata.campaign.risk_class = "controlled_reset".into();
    assert_ne!(canonicalize_manifest(&metadata).unwrap().bundle_id(), baseline_id);

    let mut size = baseline.clone();
    size.artifacts[0].byte_size += 1;
    assert_ne!(canonicalize_manifest(&size).unwrap().bundle_id(), baseline_id);

    let mut hash = baseline.clone();
    hash.artifacts[0].sha256 = SHA_B.into();
    assert_ne!(canonicalize_manifest(&hash).unwrap().bundle_id(), baseline_id);

    let mut path = baseline;
    path.artifacts[0].path = "raw/renamed.bin".into();
    path.campaign.runs[0].output_artifact_paths[0] = "raw/renamed.bin".into();
    path.campaign.runs[0].observations[0].artifact_paths[0] = "raw/renamed.bin".into();
    path.artifacts[1].derivation.as_mut().unwrap().source_artifact_paths[0] = "raw/renamed.bin".into();
    assert_ne!(canonicalize_manifest(&path).unwrap().bundle_id(), baseline_id);
}

#[test]
fn canonical_authored_bundle_id_cannot_influence_its_hash() {
    let baseline = valid_manifest();
    let expected = canonicalize_manifest(&baseline).unwrap().bundle_id().to_owned();
    let mut forged = baseline;
    forged.bundle_id = format!("bundle.sha256.{SHA_B}");
    assert_eq!(canonicalize_manifest(&forged).unwrap().bundle_id(), expected);
}

#[test]
fn canonical_json_and_checksum_bytes_are_exact_and_stable() {
    let canonical = canonicalize_manifest(&valid_manifest()).unwrap();
    assert!(canonical.manifest_bytes().starts_with(b"{\n  \"schema_version\": 1,"));
    assert!(canonical.manifest_bytes().ends_with(b"}\n"));
    assert!(!canonical.manifest_bytes().ends_with(b"}\n\n"));
    assert_eq!(
        std::str::from_utf8(canonical.checksum_index_bytes()).unwrap(),
        format!("{SHA_B}  derived/output.txt\n{SHA_A}  raw/output.bin\n")
    );
    let parsed: BundleManifest = serde_json::from_slice(canonical.manifest_bytes()).unwrap();
    assert_eq!(&parsed, canonical.manifest());
}

#[test]
fn canonical_content_and_file_digests_remain_distinct() {
    let canonical = canonicalize_manifest(&valid_manifest()).unwrap();
    let content_hash = canonical.bundle_id().strip_prefix("bundle.sha256.").unwrap();
    for hash in [content_hash, canonical.manifest_sha256(), canonical.checksum_index_sha256()] {
        assert_eq!(hash.len(), 64);
        assert!(hash.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')));
    }
    assert_ne!(content_hash, canonical.manifest_sha256());
    assert_ne!(content_hash, canonical.checksum_index_sha256());
    assert_ne!(canonical.manifest_sha256(), canonical.checksum_index_sha256());
}
