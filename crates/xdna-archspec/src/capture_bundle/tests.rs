use super::{
    build_canonical_bundle, build_canonical_bundle_v2, canonicalize_manifest, canonicalize_manifest_v2,
    emit_bundle, emit_bundle_v2, parse_emission_plan_document, parse_manifest_document, validate_bundle,
    validate_bundle_graph, ArtifactSource, BundleManifest, CanonicalBundle, DependencySource, EmissionPlan,
    EmissionPlanDocument, EmissionPlanV2, ManifestDocument, EMISSION_PLAN_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
};
use crate::research_reserve::{BundleLocationEntry, BundleLocationRoot};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::symlink,
    path::{Path, PathBuf},
};

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SHA_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
const SHA_TEST: &str = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

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

fn v2_fixture_value() -> Value {
    json!({
        "schema_version": 2,
        "bundle_id": format!("bundle.sha256.{SHA_A}"),
        "role": "fixture",
        "body": {
            "id": "fixture.synthetic.input",
            "semantic_kind": "npu_program",
            "provenance": "current",
            "source_revisions": [{
                "repository": "https://example.invalid/fixture",
                "commit": "commit.synthetic"
            }],
            "recipe": known(json!({
                "logical_name": "build-recipe.json",
                "sha256": SHA_B
            })),
            "notes": ["synthetic fixture"]
        },
        "dependencies": [],
        "artifacts": [{
            "path": "raw/input.bin",
            "byte_size": 3,
            "sha256": SHA_A,
            "semantic_kind": "input.binary",
            "class": "raw",
            "redistributability": "redistributable",
            "run_ids": [],
            "observation_ids": [],
            "derivation": null
        }]
    })
}

fn v2_observation_value() -> Value {
    json!({
        "schema_version": 2,
        "bundle_id": format!("bundle.sha256.{SHA_B}"),
        "role": "observation",
        "body": {
            "campaign": valid_campaign("aie2", "npu1", "npu1"),
            "input_references": [{
                "input_id": "input.synthetic.payload",
                "fixture_bundle_id": format!("bundle.sha256.{SHA_A}"),
                "artifact_path": "raw/input.bin"
            }]
        },
        "dependencies": [{
            "fixture_bundle_id": format!("bundle.sha256.{SHA_A}"),
            "artifact_path": "raw/input.bin",
            "artifact_sha256": SHA_A,
            "semantic_kind": "input.binary"
        }],
        "artifacts": valid_artifacts()
    })
}

fn v2_plan_value() -> Value {
    let mut fixture = v2_fixture_value();
    fixture.as_object_mut().unwrap().remove("bundle_id");
    fixture["artifacts"][0].as_object_mut().unwrap().remove("byte_size");
    fixture["artifacts"][0].as_object_mut().unwrap().remove("sha256");
    fixture["artifacts"][0]
        .as_object_mut()
        .unwrap()
        .insert("source_path".into(), json!("/capture/input.bin"));
    fixture
}

fn issue_paths(error: &super::BundleSchemaError) -> Vec<&str> {
    error.issues().iter().map(|issue| issue.path.as_str()).collect()
}

#[test]
fn v2_schema_dispatches_fixture_and_observation_roles() {
    let fixture = serde_json::to_vec(&v2_fixture_value()).unwrap();
    let fixture = parse_manifest_document(&fixture).unwrap();
    assert!(matches!(fixture, ManifestDocument::V2(_)));
    assert!(fixture.validate().unwrap().is_promotion_eligible());

    let observation = serde_json::to_vec(&v2_observation_value()).unwrap();
    let observation = parse_manifest_document(&observation).unwrap();
    assert!(matches!(observation, ManifestDocument::V2(_)));
    assert!(observation.validate().unwrap().is_promotion_eligible());

    let v1 = serde_json::to_vec(&valid_manifest()).unwrap();
    assert!(matches!(parse_manifest_document(&v1).unwrap(), ManifestDocument::V1(_)));

    let plan = serde_json::to_vec(&v2_plan_value()).unwrap();
    assert!(matches!(parse_emission_plan_document(&plan).unwrap(), EmissionPlanDocument::V2(_)));
}

#[test]
fn v2_schema_rejects_role_and_dependency_mismatches() {
    let mut fixture = v2_fixture_value();
    fixture["artifacts"][0]["run_ids"] = json!(["run.synthetic.0"]);
    let fixture = parse_manifest_document(&serde_json::to_vec(&fixture).unwrap()).unwrap();
    assert_eq!(issue_paths(&fixture.validate().unwrap_err()), ["$.artifacts[0].run_ids"]);

    let mut observation = v2_observation_value();
    observation["dependencies"][0]["artifact_sha256"] = json!(SHA_B);
    let observation = parse_manifest_document(&serde_json::to_vec(&observation).unwrap()).unwrap();
    assert_eq!(issue_paths(&observation.validate().unwrap_err()), ["$.body.input_references[0]"]);

    let mut duplicate = v2_observation_value();
    let repeated = duplicate["dependencies"][0].clone();
    duplicate["dependencies"].as_array_mut().unwrap().push(repeated);
    let duplicate = parse_manifest_document(&serde_json::to_vec(&duplicate).unwrap()).unwrap();
    assert_eq!(issue_paths(&duplicate.validate().unwrap_err()), ["$.dependencies[1]"]);
}

#[test]
fn v2_schema_rejects_unknown_or_unsupported_documents() {
    let mut unknown_role = v2_fixture_value();
    unknown_role["role"] = json!("mystery");
    assert!(parse_manifest_document(&serde_json::to_vec(&unknown_role).unwrap()).is_err());

    let mut unsupported = v2_fixture_value();
    unsupported["schema_version"] = json!(3);
    assert!(parse_manifest_document(&serde_json::to_vec(&unsupported).unwrap()).is_err());
}

#[test]
fn v2_canonical_identity_includes_dependencies_but_not_local_paths() {
    let ManifestDocument::V2(manifest) =
        parse_manifest_document(&serde_json::to_vec(&v2_observation_value()).unwrap()).unwrap()
    else {
        panic!("expected v2");
    };
    let baseline = canonicalize_manifest_v2(&manifest).unwrap();

    let mut changed = manifest.clone();
    changed.dependencies[0].semantic_kind = "input.changed".into();
    let super::BundlePayload::Observation(body) = &mut changed.payload else {
        panic!("expected observation");
    };
    body.campaign.stimulus.inputs[0].semantic_kind = "input.changed".into();
    assert_ne!(canonicalize_manifest_v2(&changed).unwrap().bundle_id(), baseline.bundle_id());

    let EmissionPlanDocument::V2(mut first) =
        parse_emission_plan_document(&serde_json::to_vec(&v2_plan_value()).unwrap()).unwrap()
    else {
        panic!("expected v2");
    };
    let mut second = first.clone();
    first.dependencies.clear();
    second.dependencies.clear();
    first.artifacts[0].source_path = PathBuf::from("/first/input.bin");
    second.artifacts[0].source_path = PathBuf::from("/other/input.bin");
    let records = vec![valid_fixture_artifact_record()];
    assert_eq!(
        build_canonical_bundle_v2(first.payload, vec![], records.clone())
            .unwrap()
            .bundle_id(),
        build_canonical_bundle_v2(second.payload, vec![], records).unwrap().bundle_id()
    );
}

#[test]
fn v2_canonical_authored_order_is_stable() {
    let ManifestDocument::V2(mut ordered) =
        parse_manifest_document(&serde_json::to_vec(&v2_observation_value()).unwrap()).unwrap()
    else {
        panic!("expected v2");
    };
    ordered.dependencies.push(super::DependencyRequirement {
        fixture_bundle_id: format!("bundle.sha256.{SHA_B}"),
        artifact_path: "raw/second.bin".into(),
        artifact_sha256: SHA_B.into(),
        semantic_kind: "input.second".into(),
    });
    let super::BundlePayload::Observation(body) = &mut ordered.payload else {
        panic!("expected observation");
    };
    body.input_references.push(super::ObservationInputReference {
        input_id: "input.synthetic.second".into(),
        fixture_bundle_id: format!("bundle.sha256.{SHA_B}"),
        artifact_path: "raw/second.bin".into(),
    });
    body.campaign.stimulus.inputs.push(super::InputIdentity {
        id: "input.synthetic.second".into(),
        semantic_kind: "input.second".into(),
        content: super::ContentPin { logical_name: "second.bin".into(), sha256: SHA_B.into() },
    });

    let mut reversed = ordered.clone();
    reversed.dependencies.reverse();
    let super::BundlePayload::Observation(body) = &mut reversed.payload else {
        panic!("expected observation");
    };
    body.input_references.reverse();
    body.campaign.stimulus.inputs.reverse();

    let ordered = canonicalize_manifest_v2(&ordered).unwrap();
    let reversed = canonicalize_manifest_v2(&reversed).unwrap();
    assert_eq!(ordered.bundle_id(), reversed.bundle_id());
    assert_eq!(ordered.manifest_bytes(), reversed.manifest_bytes());
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

#[test]
fn v1_characterization_freezes_canonical_identity() {
    let canonical = canonicalize_manifest(&valid_manifest()).unwrap();
    assert_eq!(
        canonical.bundle_id(),
        "bundle.sha256.41ecc2b7d78d4805144ab3d2e97836163803ca730df67a9ca0581042c4f2ec87"
    );
    assert_eq!(
        canonical.manifest_sha256(),
        "e0979410e312c1c82746acc385ae844119e2e6464bba88812b16fc6de521106d"
    );
    assert_eq!(
        canonical.checksum_index_sha256(),
        "4b9a34f50d4836dd1a20aa31ec752b238de3a1b115aac59b6fef7408efedc646"
    );
}

fn valid_bundle_manifest() -> BundleManifest {
    let mut manifest = valid_manifest();
    manifest.artifacts[0].sha256 = SHA_ABC.into();
    manifest.artifacts[0].byte_size = 3;
    manifest.artifacts[1].sha256 = SHA_TEST.into();
    manifest.artifacts[1].byte_size = 4;
    manifest
}

fn valid_fixture_artifact_record() -> super::ArtifactRecord {
    serde_json::from_value(v2_fixture_value()["artifacts"][0].clone()).unwrap()
}

fn valid_fixture_manifest() -> super::BundleManifestV2 {
    let ManifestDocument::V2(mut manifest) =
        parse_manifest_document(&serde_json::to_vec(&v2_fixture_value()).unwrap()).unwrap()
    else {
        panic!("expected v2");
    };
    manifest.artifacts[0].sha256 = SHA_ABC.into();
    manifest
}

fn write_bundle(root: &Path, manifest: &BundleManifest) -> CanonicalBundle {
    fs::create_dir_all(root.join("raw")).unwrap();
    fs::create_dir_all(root.join("derived")).unwrap();
    fs::write(root.join("raw/output.bin"), b"abc").unwrap();
    fs::write(root.join("derived/output.txt"), b"test").unwrap();
    let canonical = canonicalize_manifest(manifest).unwrap();
    fs::write(root.join("manifest.json"), canonical.manifest_bytes()).unwrap();
    fs::write(root.join("SHA256SUMS"), canonical.checksum_index_bytes()).unwrap();
    canonical
}

fn write_v2_fixture(root: &Path) -> super::CanonicalBundleV2 {
    fs::create_dir_all(root.join("raw")).unwrap();
    fs::create_dir_all(root.join("derived")).unwrap();
    fs::write(root.join("raw/input.bin"), b"abc").unwrap();
    let canonical = canonicalize_manifest_v2(&valid_fixture_manifest()).unwrap();
    fs::write(root.join("manifest.json"), canonical.manifest_bytes()).unwrap();
    fs::write(root.join("SHA256SUMS"), canonical.checksum_index_bytes()).unwrap();
    canonical
}

fn valid_observation_manifest(fixture_bundle_id: &str) -> super::BundleManifestV2 {
    let mut value = v2_observation_value();
    value["body"]["campaign"]["stimulus"]["inputs"][0]["content"]["sha256"] = json!(SHA_ABC);
    value["body"]["input_references"][0]["fixture_bundle_id"] = json!(fixture_bundle_id);
    value["dependencies"][0]["fixture_bundle_id"] = json!(fixture_bundle_id);
    value["dependencies"][0]["artifact_sha256"] = json!(SHA_ABC);
    value["artifacts"] = serde_json::to_value(valid_bundle_manifest().artifacts).unwrap();
    let ManifestDocument::V2(manifest) =
        parse_manifest_document(&serde_json::to_vec(&value).unwrap()).unwrap()
    else {
        panic!("expected v2");
    };
    manifest
}

fn write_v2_observation(root: &Path, fixture_bundle_id: &str) -> super::CanonicalBundleV2 {
    fs::create_dir_all(root.join("raw")).unwrap();
    fs::create_dir_all(root.join("derived")).unwrap();
    fs::write(root.join("raw/output.bin"), b"abc").unwrap();
    fs::write(root.join("derived/output.txt"), b"test").unwrap();
    let canonical = canonicalize_manifest_v2(&valid_observation_manifest(fixture_bundle_id)).unwrap();
    fs::write(root.join("manifest.json"), canonical.manifest_bytes()).unwrap();
    fs::write(root.join("SHA256SUMS"), canonical.checksum_index_bytes()).unwrap();
    canonical
}

fn graph_root(path: &Path, entries: Vec<(&str, &str)>) -> BundleLocationRoot {
    BundleLocationRoot {
        alias: "synthetic".into(),
        path: path.to_owned(),
        failure_domain_id: "failure.synthetic".into(),
        bundles: entries
            .into_iter()
            .map(|(bundle_id, relative_path)| BundleLocationEntry {
                bundle_id: bundle_id.into(),
                relative_path: relative_path.into(),
            })
            .collect(),
    }
}

fn write_valid_bundle(root: &Path) -> CanonicalBundle {
    write_bundle(root, &valid_bundle_manifest())
}

fn validation_error(root: &Path) -> String {
    validate_bundle(root).unwrap_err().to_string()
}

#[test]
fn v2_validate_accepts_a_canonical_fixture_leaf() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("fixture");
    let expected = write_v2_fixture(&root);

    let validated = validate_bundle(&root).unwrap();
    assert_eq!(validated.bundle_id(), expected.bundle_id());
    assert_eq!(validated.manifest_sha256(), expected.manifest_sha256());
    assert_eq!(validated.checksum_index_sha256(), expected.checksum_index_sha256());
    assert!(validated.is_promotion_eligible());
}

#[test]
fn graph_validates_an_observation_and_its_exact_fixture() {
    let temporary = tempfile::tempdir().unwrap();
    let fixture = write_v2_fixture(&temporary.path().join("fixture"));
    let observation = write_v2_observation(&temporary.path().join("observation"), fixture.bundle_id());
    let location = graph_root(
        temporary.path(),
        vec![(observation.bundle_id(), "observation"), (fixture.bundle_id(), "fixture")],
    );

    let graph = validate_bundle_graph(temporary.path().join("observation"), &location).unwrap();
    assert_eq!(graph.root_bundle_id(), observation.bundle_id());
    assert_eq!(graph.bundle_count(), 2);
    assert!(graph.is_promotion_eligible());
}

#[test]
fn graph_rejects_missing_or_substituted_fixture_resolution() {
    let temporary = tempfile::tempdir().unwrap();
    let fixture = write_v2_fixture(&temporary.path().join("fixture"));
    let observation = write_v2_observation(&temporary.path().join("observation"), fixture.bundle_id());

    let missing = graph_root(temporary.path(), vec![(observation.bundle_id(), "observation")]);
    assert!(validate_bundle_graph(temporary.path().join("observation"), &missing)
        .unwrap_err()
        .to_string()
        .contains("missing bundle mapping"));

    write_valid_bundle(&temporary.path().join("wrong"));
    let substituted = graph_root(
        temporary.path(),
        vec![(observation.bundle_id(), "observation"), (fixture.bundle_id(), "wrong")],
    );
    assert!(validate_bundle_graph(temporary.path().join("observation"), &substituted)
        .unwrap_err()
        .to_string()
        .contains("mapped bundle ID mismatch"));
}

#[test]
fn graph_rejects_duplicate_locations_and_fixture_artifact_mismatch() {
    let temporary = tempfile::tempdir().unwrap();
    let fixture = write_v2_fixture(&temporary.path().join("fixture"));
    let mut manifest = valid_observation_manifest(fixture.bundle_id());
    manifest.dependencies[0].semantic_kind = "wrong.kind".into();
    if let super::BundlePayload::Observation(body) = &mut manifest.payload {
        body.campaign.stimulus.inputs[0].semantic_kind = "wrong.kind".into();
    }
    let observation_root = temporary.path().join("observation");
    fs::create_dir_all(observation_root.join("raw")).unwrap();
    fs::create_dir_all(observation_root.join("derived")).unwrap();
    fs::write(observation_root.join("raw/output.bin"), b"abc").unwrap();
    fs::write(observation_root.join("derived/output.txt"), b"test").unwrap();
    let observation = canonicalize_manifest_v2(&manifest).unwrap();
    fs::write(observation_root.join("manifest.json"), observation.manifest_bytes()).unwrap();
    fs::write(observation_root.join("SHA256SUMS"), observation.checksum_index_bytes()).unwrap();

    let mismatched = graph_root(
        temporary.path(),
        vec![(observation.bundle_id(), "observation"), (fixture.bundle_id(), "fixture")],
    );
    assert!(validate_bundle_graph(&observation_root, &mismatched)
        .unwrap_err()
        .to_string()
        .contains("semantic kind mismatch"));

    let duplicate = graph_root(
        temporary.path(),
        vec![(observation.bundle_id(), "observation"), (fixture.bundle_id(), "observation")],
    );
    assert!(validate_bundle_graph(&observation_root, &duplicate)
        .unwrap_err()
        .to_string()
        .contains("duplicate mapped location"));
}

#[test]
fn graph_accepts_a_self_contained_v1_leaf() {
    let temporary = tempfile::tempdir().unwrap();
    let leaf = write_valid_bundle(&temporary.path().join("leaf"));
    let location = graph_root(temporary.path(), vec![(leaf.bundle_id(), "leaf")]);

    let graph = validate_bundle_graph(temporary.path().join("leaf"), &location).unwrap();
    assert_eq!(graph.bundle_count(), 1);
}

#[test]
fn validate_accepts_a_canonical_complete_bundle() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("bundle");
    let expected = write_valid_bundle(&root);

    let validated = validate_bundle(&root).unwrap();
    assert_eq!(validated.bundle_id(), expected.bundle_id());
    assert_eq!(validated.manifest_sha256(), expected.manifest_sha256());
    assert_eq!(validated.checksum_index_sha256(), expected.checksum_index_sha256());
    assert!(validated.is_promotion_eligible());
    assert!(validated.promotion_blockers().is_empty());
}

#[test]
fn validate_rejects_missing_and_extra_root_entries() {
    let missing_temp = tempfile::tempdir().unwrap();
    let missing = missing_temp.path().join("bundle");
    write_valid_bundle(&missing);
    fs::remove_file(missing.join("SHA256SUMS")).unwrap();
    assert!(validation_error(&missing).contains("missing required root entry `SHA256SUMS`"));

    let extra_temp = tempfile::tempdir().unwrap();
    let extra = extra_temp.path().join("bundle");
    write_valid_bundle(&extra);
    fs::write(extra.join("notes.txt"), b"extra").unwrap();
    assert!(validation_error(&extra).contains("unexpected root entry `notes.txt`"));
}

#[test]
fn validate_rejects_missing_and_extra_artifacts() {
    let missing_temp = tempfile::tempdir().unwrap();
    let missing = missing_temp.path().join("bundle");
    write_valid_bundle(&missing);
    fs::remove_file(missing.join("raw/output.bin")).unwrap();
    assert!(validation_error(&missing).contains("declared artifact is missing"));

    let extra_temp = tempfile::tempdir().unwrap();
    let extra = extra_temp.path().join("bundle");
    write_valid_bundle(&extra);
    fs::write(extra.join("raw/extra.bin"), b"extra").unwrap();
    assert!(validation_error(&extra).contains("undeclared artifact"));
}

#[test]
fn validate_rejects_altered_truncated_and_substituted_artifacts() {
    for (bytes, expected) in [
        (&b"abd"[..], "artifact SHA-256 mismatch"),
        (&b"ab"[..], "artifact size mismatch"),
        (&b"test"[..], "artifact size mismatch"),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("bundle");
        write_valid_bundle(&root);
        fs::write(root.join("raw/output.bin"), bytes).unwrap();
        assert!(validation_error(&root).contains(expected), "{bytes:?}");
    }
}

#[test]
fn validate_rejects_artifact_and_directory_symlinks() {
    let artifact_temp = tempfile::tempdir().unwrap();
    let artifact_root = artifact_temp.path().join("bundle");
    write_valid_bundle(&artifact_root);
    let outside_file = artifact_temp.path().join("outside.bin");
    fs::write(&outside_file, b"abc").unwrap();
    fs::remove_file(artifact_root.join("raw/output.bin")).unwrap();
    symlink(&outside_file, artifact_root.join("raw/output.bin")).unwrap();
    assert!(validation_error(&artifact_root).contains("symlink"));

    let directory_temp = tempfile::tempdir().unwrap();
    let directory_root = directory_temp.path().join("bundle");
    write_valid_bundle(&directory_root);
    let outside_directory = directory_temp.path().join("outside");
    fs::create_dir(&outside_directory).unwrap();
    symlink(&outside_directory, directory_root.join("raw/linked")).unwrap();
    assert!(validation_error(&directory_root).contains("symlink"));
}

#[test]
fn validate_rejects_bundle_root_and_required_entry_symlinks() {
    let root_temp = tempfile::tempdir().unwrap();
    let real_root = root_temp.path().join("real-bundle");
    write_valid_bundle(&real_root);
    let linked_root = root_temp.path().join("linked-bundle");
    symlink(&real_root, &linked_root).unwrap();
    assert!(validation_error(&linked_root).contains("bundle root must be a real directory"));

    let entry_temp = tempfile::tempdir().unwrap();
    let entry_root = entry_temp.path().join("bundle");
    write_valid_bundle(&entry_root);
    let manifest_copy = entry_temp.path().join("manifest-copy.json");
    fs::copy(entry_root.join("manifest.json"), &manifest_copy).unwrap();
    fs::remove_file(entry_root.join("manifest.json")).unwrap();
    symlink(&manifest_copy, entry_root.join("manifest.json")).unwrap();
    assert!(validation_error(&entry_root).contains("regular file, not a symlink"));
}

#[test]
fn validate_rejects_unsafe_manifest_artifact_paths() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("bundle");
    let canonical = write_valid_bundle(&root);
    let mut manifest = canonical.manifest().clone();
    manifest
        .artifacts
        .iter_mut()
        .find(|artifact| artifact.class == super::ArtifactClass::Raw)
        .unwrap()
        .path = "raw/../escape.bin".into();
    let mut bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    bytes.push(b'\n');
    fs::write(root.join("manifest.json"), bytes).unwrap();

    assert!(validation_error(&root).contains("invalid canonical raw artifact path"));
}

#[test]
fn validate_rejects_reformatted_and_altered_manifest_bytes() {
    let reformatted_temp = tempfile::tempdir().unwrap();
    let reformatted = reformatted_temp.path().join("bundle");
    let canonical = write_valid_bundle(&reformatted);
    fs::write(reformatted.join("manifest.json"), serde_json::to_vec(canonical.manifest()).unwrap()).unwrap();
    assert!(validation_error(&reformatted).contains("manifest.json is not canonical"));

    let altered_temp = tempfile::tempdir().unwrap();
    let altered = altered_temp.path().join("bundle");
    let canonical = write_valid_bundle(&altered);
    let mut manifest = canonical.manifest().clone();
    manifest.campaign.risk_class = "controlled_reset".into();
    let mut bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    bytes.push(b'\n');
    fs::write(altered.join("manifest.json"), bytes).unwrap();
    assert!(validation_error(&altered).contains("bundle ID mismatch"));
}

#[test]
fn validate_rejects_reordered_malformed_and_mismatched_checksum_index() {
    for contents in [
        format!("{SHA_ABC}  raw/output.bin\n{SHA_TEST}  derived/output.txt\n"),
        "not a checksum index\n".into(),
        format!("{SHA_A}  derived/output.txt\n{SHA_ABC}  raw/output.bin\n"),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("bundle");
        write_valid_bundle(&root);
        fs::write(root.join("SHA256SUMS"), contents).unwrap();
        assert!(validation_error(&root).contains("SHA256SUMS is not canonical"));
    }
}

#[test]
fn validate_rejects_a_forged_bundle_id() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("bundle");
    let canonical = write_valid_bundle(&root);
    let mut manifest = canonical.manifest().clone();
    manifest.bundle_id = format!("bundle.sha256.{SHA_A}");
    let mut bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    bytes.push(b'\n');
    fs::write(root.join("manifest.json"), bytes).unwrap();

    assert!(validation_error(&root).contains("bundle ID mismatch"));
}

#[test]
fn validate_returns_blocked_opaque_result_for_unavailable_provenance() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("bundle");
    let mut manifest = valid_bundle_manifest();
    manifest.campaign.platform.device_model_key =
        super::Availability::Unavailable { reason: "not recorded".into() };
    write_bundle(&root, &manifest);

    let validated = validate_bundle(&root).unwrap();
    assert!(!validated.is_promotion_eligible());
    assert_eq!(validated.promotion_blockers()[0].path, "$.campaign.platform.device_model_key");
}

pub(crate) fn emission_plan(source_root: &Path) -> EmissionPlan {
    fs::create_dir_all(source_root).unwrap();
    let mut plan: EmissionPlan = serde_json::from_value(valid_plan_value()).unwrap();
    for artifact in &mut plan.artifacts {
        let (name, bytes) = match artifact.class {
            super::ArtifactClass::Raw => ("raw-source.bin", &b"abc"[..]),
            super::ArtifactClass::Derived => ("derived-source.txt", &b"test"[..]),
        };
        artifact.source_path = source_root.join(name);
        fs::write(&artifact.source_path, bytes).unwrap();
    }
    plan
}

fn artifact_source(record: super::ArtifactRecord, source_path: PathBuf) -> ArtifactSource {
    ArtifactSource {
        path: record.path,
        source_path,
        semantic_kind: record.semantic_kind,
        class: record.class,
        redistributability: record.redistributability,
        run_ids: record.run_ids,
        observation_ids: record.observation_ids,
        derivation: record.derivation,
    }
}

pub(crate) fn v2_fixture_emission_plan(source_root: &Path) -> EmissionPlanV2 {
    fs::create_dir_all(source_root).unwrap();
    let source = source_root.join("input.bin");
    fs::write(&source, b"abc").unwrap();
    let manifest = valid_fixture_manifest();
    EmissionPlanV2 {
        schema_version: 2,
        payload: manifest.payload,
        dependencies: vec![],
        artifacts: manifest
            .artifacts
            .into_iter()
            .map(|artifact| artifact_source(artifact, source.clone()))
            .collect(),
    }
}

pub(crate) fn v2_observation_emission_plan(
    source_root: &Path,
    fixture: &super::ValidatedBundle,
    fixture_path: &Path,
) -> EmissionPlanV2 {
    fs::create_dir_all(source_root).unwrap();
    let raw = source_root.join("output.bin");
    let derived = source_root.join("output.txt");
    fs::write(&raw, b"abc").unwrap();
    fs::write(&derived, b"test").unwrap();
    let manifest = valid_observation_manifest(fixture.bundle_id());
    let sources = [raw, derived];
    EmissionPlanV2 {
        schema_version: 2,
        payload: manifest.payload,
        dependencies: manifest
            .dependencies
            .into_iter()
            .map(|requirement| DependencySource { requirement, source_path: fixture_path.to_owned() })
            .collect(),
        artifacts: manifest
            .artifacts
            .into_iter()
            .zip(sources)
            .map(|(artifact, source)| artifact_source(artifact, source))
            .collect(),
    }
}

fn tree_files(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![root.to_owned()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let relative = path.strip_prefix(root).unwrap().to_str().unwrap().replace('\\', "/");
                files.insert(relative, fs::read(path).unwrap());
            }
        }
    }
    files
}

#[test]
fn emit_round_trips_through_the_public_validator() {
    let temporary = tempfile::tempdir().unwrap();
    let plan = emission_plan(&temporary.path().join("sources"));
    let output = temporary.path().join("bundle");

    let emitted = emit_bundle(&plan, &output).unwrap();
    let validated = validate_bundle(&output).unwrap();
    assert_eq!(emitted.bundle_id(), validated.bundle_id());
    assert_eq!(emitted.manifest_sha256(), validated.manifest_sha256());
    assert_eq!(emitted.checksum_index_sha256(), validated.checksum_index_sha256());
}

#[test]
fn v2_emit_reuses_an_identical_fixture_without_mutation() {
    let temporary = tempfile::tempdir().unwrap();
    let plan = v2_fixture_emission_plan(&temporary.path().join("sources"));
    let output = temporary.path().join("fixture");

    let first = emit_bundle_v2(&plan, &output).unwrap();
    let first_tree = tree_files(&output);
    let second = emit_bundle_v2(&plan, &output).unwrap();
    assert_eq!(first.bundle_id(), second.bundle_id());
    assert_eq!(tree_files(&output), first_tree);
}

#[test]
fn v2_emit_observation_validates_fixture_graph_without_copying_it() {
    let temporary = tempfile::tempdir().unwrap();
    let fixture_path = temporary.path().join("fixture");
    let fixture =
        emit_bundle_v2(&v2_fixture_emission_plan(&temporary.path().join("fixture-source")), &fixture_path)
            .unwrap();
    let plan =
        v2_observation_emission_plan(&temporary.path().join("observation-source"), &fixture, &fixture_path);
    let output = temporary.path().join("observation");

    let observation = emit_bundle_v2(&plan, &output).unwrap();
    assert_ne!(observation.bundle_id(), fixture.bundle_id());
    assert_eq!(
        tree_files(&output).keys().cloned().collect::<Vec<_>>(),
        vec![
            "SHA256SUMS".to_owned(),
            "derived/output.txt".to_owned(),
            "manifest.json".to_owned(),
            "raw/output.bin".to_owned(),
        ]
    );
}

#[test]
fn v2_emit_dependency_failure_leaves_destination_absent() {
    let temporary = tempfile::tempdir().unwrap();
    let fixture_path = temporary.path().join("fixture");
    let fixture =
        emit_bundle_v2(&v2_fixture_emission_plan(&temporary.path().join("fixture-source")), &fixture_path)
            .unwrap();
    let mut plan =
        v2_observation_emission_plan(&temporary.path().join("observation-source"), &fixture, &fixture_path);
    plan.dependencies[0].source_path = temporary.path().join("missing");
    let output = temporary.path().join("observation");

    assert!(emit_bundle_v2(&plan, &output).is_err());
    assert!(!output.exists());
}

#[test]
fn v2_emit_refuses_to_replace_a_different_existing_bundle() {
    let temporary = tempfile::tempdir().unwrap();
    let output = temporary.path().join("fixture");
    let first_plan = v2_fixture_emission_plan(&temporary.path().join("first-source"));
    emit_bundle_v2(&first_plan, &output).unwrap();
    let original = tree_files(&output);

    let mut second_plan = v2_fixture_emission_plan(&temporary.path().join("second-source"));
    if let super::BundlePayload::Fixture(body) = &mut second_plan.payload {
        body.notes.push("different identity".into());
    }
    assert!(emit_bundle_v2(&second_plan, &output).is_err());
    assert_eq!(tree_files(&output), original);
}

#[test]
fn emit_is_path_and_authored_order_independent_without_source_leaks() {
    let first_temp = tempfile::tempdir().unwrap();
    let second_temp = tempfile::tempdir().unwrap();
    let first_plan = emission_plan(&first_temp.path().join("first-sources"));
    let mut second_plan = emission_plan(&second_temp.path().join("other-sources"));
    second_plan.artifacts.reverse();
    let first_output = first_temp.path().join("first-output");
    let second_output = second_temp.path().join("other-output");

    emit_bundle(&first_plan, &first_output).unwrap();
    emit_bundle(&second_plan, &second_output).unwrap();
    assert_eq!(tree_files(&first_output), tree_files(&second_output));

    let canonical_text = format!(
        "{}{}",
        fs::read_to_string(first_output.join("manifest.json")).unwrap(),
        fs::read_to_string(first_output.join("SHA256SUMS")).unwrap()
    );
    assert!(!canonical_text.contains(first_temp.path().to_str().unwrap()));
    assert!(!canonical_text.contains(second_temp.path().to_str().unwrap()));
}

#[test]
fn emit_artifact_byte_changes_change_bundle_identity() {
    let temporary = tempfile::tempdir().unwrap();
    let plan = emission_plan(&temporary.path().join("sources"));
    let first_output = temporary.path().join("first");
    let first = emit_bundle(&plan, &first_output).unwrap().bundle_id().to_owned();

    let raw_source = &plan
        .artifacts
        .iter()
        .find(|artifact| artifact.class == super::ArtifactClass::Raw)
        .unwrap()
        .source_path;
    fs::write(raw_source, b"abd").unwrap();
    let second_output = temporary.path().join("second");
    let second = emit_bundle(&plan, &second_output).unwrap().bundle_id().to_owned();
    assert_ne!(first, second);
}

#[test]
fn emit_never_mutates_an_existing_output() {
    let temporary = tempfile::tempdir().unwrap();
    let plan = emission_plan(&temporary.path().join("sources"));
    let output = temporary.path().join("bundle");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("sentinel"), b"keep").unwrap();

    assert!(emit_bundle(&plan, &output).is_err());
    assert_eq!(fs::read(output.join("sentinel")).unwrap(), b"keep");
    assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
}

#[test]
fn emit_rejects_symlink_and_non_regular_sources() {
    let symlink_temp = tempfile::tempdir().unwrap();
    let mut symlink_plan = emission_plan(&symlink_temp.path().join("sources"));
    let outside = symlink_temp.path().join("outside.bin");
    fs::write(&outside, b"abc").unwrap();
    let source = &mut symlink_plan.artifacts[0].source_path;
    fs::remove_file(&*source).unwrap();
    symlink(&outside, &*source).unwrap();
    let symlink_output = symlink_temp.path().join("bundle");
    assert!(emit_bundle(&symlink_plan, &symlink_output).is_err());
    assert!(!symlink_output.exists());

    let directory_temp = tempfile::tempdir().unwrap();
    let mut directory_plan = emission_plan(&directory_temp.path().join("sources"));
    let source = &mut directory_plan.artifacts[0].source_path;
    fs::remove_file(&*source).unwrap();
    fs::create_dir(&*source).unwrap();
    let directory_output = directory_temp.path().join("bundle");
    assert!(emit_bundle(&directory_plan, &directory_output).is_err());
    assert!(!directory_output.exists());
}

#[test]
fn emit_missing_source_fails_without_a_final_bundle() {
    let temporary = tempfile::tempdir().unwrap();
    let plan = emission_plan(&temporary.path().join("sources"));
    fs::remove_file(&plan.artifacts[0].source_path).unwrap();
    let output = temporary.path().join("bundle");

    assert!(emit_bundle(&plan, &output).is_err());
    assert!(!output.exists());
}

#[test]
fn emit_self_validation_failure_cannot_publish_staging() {
    let temporary = tempfile::tempdir().unwrap();
    let plan = emission_plan(&temporary.path().join("sources"));
    let output = temporary.path().join("bundle");

    let result = super::emitter::emit_bundle_with_test_mutation(&plan, &output, |staging| {
        fs::write(staging.join("unexpected"), b"fault")
    });
    assert!(result.is_err());
    assert!(!output.exists());
}
