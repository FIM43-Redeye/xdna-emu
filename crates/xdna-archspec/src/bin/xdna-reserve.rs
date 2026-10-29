use std::{
    env,
    ffi::OsString,
    fs,
    io::{self, Write},
    path::Path,
    process,
};
use xdna_archspec::{
    capture_bundle::{
        emit_bundle, emit_bundle_v2, parse_emission_plan_document, validate_bundle, validate_bundle_graph,
        EmissionPlanDocument,
    },
    research_reserve::{render_release_report, BundleLocationPlan, BundleLocationRoot, ReserveLedger},
};

fn main() {
    let args: Vec<OsString> = env::args_os().skip(1).collect();
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    process::exit(run(&args, &mut stdout, &mut stderr));
}

fn run(args: &[OsString], stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    match args {
        [command, plan_path, output] if command == "emit" => {
            let bytes = match fs::read(plan_path) {
                Ok(bytes) => bytes,
                Err(error) => return fail(stderr, format!("cannot read emission plan: {error}")),
            };
            let plan = match parse_emission_plan_document(&bytes) {
                Ok(plan) => plan,
                Err(error) => return fail(stderr, format!("invalid emission plan: {error}")),
            };
            let emitted = match &plan {
                EmissionPlanDocument::V1(plan) => emit_bundle(plan, Path::new(output)),
                EmissionPlanDocument::V2(plan) => emit_bundle_v2(plan, Path::new(output)),
            };
            match emitted {
                Ok(bundle) => {
                    if writeln!(stdout, "emitted {}", bundle.bundle_id()).is_err() {
                        return fail(stderr, "cannot write command output");
                    }
                    0
                }
                Err(error) => fail(stderr, format!("emission failed: {error}")),
            }
        }
        [command, bundle_path] if command == "validate" => {
            let bundle = match validate_bundle(Path::new(bundle_path)) {
                Ok(bundle) => bundle,
                Err(error) => return fail(stderr, format!("validation failed: {error}")),
            };
            if writeln!(stdout, "informational validation result; not a trusted receipt").is_err()
                || writeln!(stdout, "integrity: valid").is_err()
                || writeln!(stdout, "bundle_id: {}", bundle.bundle_id()).is_err()
                || writeln!(stdout, "manifest_sha256: {}", bundle.manifest_sha256()).is_err()
                || writeln!(stdout, "checksum_index_sha256: {}", bundle.checksum_index_sha256()).is_err()
            {
                return fail(stderr, "cannot write command output");
            }
            if bundle.is_promotion_eligible() {
                if writeln!(stdout, "promotion: eligible").is_err() {
                    return fail(stderr, "cannot write command output");
                }
                0
            } else {
                if writeln!(stdout, "promotion: blocked").is_err() {
                    return fail(stderr, "cannot write command output");
                }
                for blocker in bundle.promotion_blockers() {
                    if writeln!(stdout, "- {}: {}", blocker.path, blocker.message).is_err() {
                        return fail(stderr, "cannot write command output");
                    }
                }
                1
            }
        }
        [command, bundle_path, location_plan_path] if command == "validate-graph" => {
            let plan: BundleLocationPlan = match read_json(location_plan_path) {
                Ok(plan) => plan,
                Err(error) => return fail(stderr, format!("invalid location plan: {error}")),
            };
            let bundle_path = Path::new(bundle_path);
            let leaf = match validate_bundle(bundle_path) {
                Ok(bundle) => bundle,
                Err(error) => return fail(stderr, format!("leaf validation failed: {error}")),
            };
            let root = match select_location_root(&plan, bundle_path, leaf.bundle_id()) {
                Ok(root) => root,
                Err(error) => return fail(stderr, error),
            };
            let graph = match validate_bundle_graph(bundle_path, root) {
                Ok(graph) => graph,
                Err(error) => return fail(stderr, format!("graph validation failed: {error}")),
            };
            if writeln!(stdout, "informational graph validation result; not a trusted receipt").is_err()
                || writeln!(stdout, "root_bundle_id: {}", graph.root_bundle_id()).is_err()
                || writeln!(stdout, "bundle_count: {}", graph.bundle_count()).is_err()
            {
                return fail(stderr, "cannot write command output");
            }
            if graph.is_promotion_eligible() {
                if writeln!(stdout, "promotion: eligible").is_err() {
                    return fail(stderr, "cannot write command output");
                }
                0
            } else {
                if writeln!(stdout, "promotion: blocked").is_err() {
                    return fail(stderr, "cannot write command output");
                }
                for blocker in graph.promotion_blockers() {
                    if writeln!(stdout, "- {}: {}", blocker.path, blocker.message).is_err() {
                        return fail(stderr, "cannot write command output");
                    }
                }
                1
            }
        }
        [command, ledger_path, tuple_id, location_plan_path] if command == "audit" => {
            let ledger_text = match fs::read_to_string(ledger_path) {
                Ok(text) => text,
                Err(error) => return fail(stderr, format!("cannot read ledger: {error}")),
            };
            let ledger = match ReserveLedger::from_json(&ledger_text) {
                Ok(ledger) => ledger,
                Err(error) => return fail(stderr, format!("invalid ledger: {error}")),
            };
            let plan: BundleLocationPlan = match read_json(location_plan_path) {
                Ok(plan) => plan,
                Err(error) => return fail(stderr, format!("invalid location plan: {error}")),
            };
            let Some(tuple_id) = tuple_id.to_str() else {
                return fail(stderr, "tuple ID is not UTF-8");
            };
            let report = match ledger.clean_release_with_bundle_roots(tuple_id, &plan.roots) {
                Ok(report) => report,
                Err(error) => return fail(stderr, format!("audit failed: {error}")),
            };
            if writeln!(stdout, "informational audit result; not a trusted receipt").is_err()
                || write!(stdout, "{}", render_release_report(&ledger, &report)).is_err()
            {
                return fail(stderr, "cannot write command output");
            }
            i32::from(!report.is_clean)
        }
        _ => {
            let _ = writeln!(
                stderr,
                "usage: xdna-reserve emit <emission-plan.json> <output-bundle>\n       \
                 xdna-reserve validate <bundle>\n       \
                 xdna-reserve validate-graph <bundle> <location-plan.json>\n       \
                 xdna-reserve audit <ledger.json> <tuple-id> <location-plan.json>"
            );
            2
        }
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &OsString) -> Result<T, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read `{}`: {error}", Path::new(path).display()))?;
    serde_json::from_slice(&bytes).map_err(|error| error.to_string())
}

fn select_location_root<'a>(
    plan: &'a BundleLocationPlan,
    bundle_path: &Path,
    bundle_id: &str,
) -> Result<&'a BundleLocationRoot, String> {
    let requested = fs::canonicalize(bundle_path)
        .map_err(|error| format!("cannot resolve requested bundle path: {error}"))?;
    let matches: Vec<_> = plan
        .roots
        .iter()
        .filter(|root| {
            root.bundles.iter().any(|entry| {
                entry.bundle_id == bundle_id
                    && fs::canonicalize(root.path.join(&entry.relative_path))
                        .is_ok_and(|path| path == requested)
            })
        })
        .collect();
    match matches.as_slice() {
        [root] => Ok(*root),
        [] => Err(format!("location plan has no root mapping {bundle_id} to the requested path")),
        _ => Err(format!("location plan has multiple roots mapping {bundle_id} to the requested path")),
    }
}

fn fail(stderr: &mut dyn Write, message: impl std::fmt::Display) -> i32 {
    let _ = writeln!(stderr, "{message}");
    1
}

#[cfg(test)]
mod tests {
    use super::run;
    use std::{
        collections::BTreeMap,
        ffi::OsString,
        fs,
        path::{Path, PathBuf},
    };
    use xdna_archspec::{
        capture_bundle::{
            ArtifactClass, ArtifactSource, Availability, Campaign, CampaignOutcome, CommandStimulus,
            BundlePayload, ComponentPin, EmissionPlan, EmissionPlanV2, FixtureBody, ObservationRecord,
            PciIdentity, PlatformIdentity, Provenance, RunRecord, Stimulus, EMISSION_PLAN_SCHEMA_VERSION,
            EMISSION_PLAN_SCHEMA_VERSION_V2,
        },
        research_reserve::{ContentPin, Redistributability, RevisionPin},
        types::Architecture,
    };

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn known<T>(value: T) -> Availability<T> {
        Availability::Known { value }
    }

    fn content(name: &str) -> ContentPin {
        ContentPin { logical_name: name.into(), sha256: SHA_A.into() }
    }

    fn revision(name: &str) -> RevisionPin {
        RevisionPin {
            repository: format!("https://example.invalid/{name}"),
            commit: "commit.synthetic".into(),
        }
    }

    fn component(name: &str) -> ComponentPin {
        ComponentPin {
            name: name.into(),
            revision: "revision.synthetic".into(),
            sha256: known(SHA_A.into()),
        }
    }

    fn plan(source: PathBuf) -> EmissionPlan {
        EmissionPlan {
            schema_version: EMISSION_PLAN_SCHEMA_VERSION,
            campaign: Campaign {
                id: "campaign.synthetic.cli".into(),
                tuple_ids: vec!["tuple.synthetic.device".into()],
                inventory_ids: vec!["inventory.synthetic.firmware".into()],
                fact_ids: vec!["fact.synthetic.lifecycle".into()],
                evidence_ids: vec!["evidence.synthetic.cli".into()],
                provenance: Provenance::Current,
                risk_class: "non_destructive".into(),
                outcome: CampaignOutcome::Success,
                platform: PlatformIdentity {
                    architecture: Architecture::Aie2,
                    device_model_key: known("npu1".into()),
                    driver_platform_id: known("npu1".into()),
                    pci: known(PciIdentity {
                        vendor_id: "1022".into(),
                        device_id: "1502".into(),
                        subsystem_vendor_id: "1022".into(),
                        subsystem_device_id: "1502".into(),
                        revision_id: "00".into(),
                    }),
                    board_identity: known("integrated.reference".into()),
                    firmware: known(content("npu.sbin")),
                    host_kernel: known(component("linux")),
                    kernel_modules: vec![component("amdxdna")],
                    driver: known(revision("xdna-driver")),
                    xrt_components: vec![component("xrt-base")],
                    toolchain_components: vec![component("mlir-aie")],
                    compiler_mode: known("peano".into()),
                    execution_mode: known("direct".into()),
                    reset_state: known("cold".into()),
                    power_state: known("d0".into()),
                    clock_state: known("default".into()),
                    iommu_state: known("enabled".into()),
                    address_state: known("canonical".into()),
                },
                stimulus: Stimulus {
                    command: CommandStimulus {
                        argv: vec!["runner".into(), "--case".into(), "synthetic".into()],
                        environment: BTreeMap::from([("LANG".into(), "C".into())]),
                    },
                    source_revisions: vec![revision("fixture")],
                    build_recipe: known(content("build-recipe.json")),
                    inputs: vec![],
                    initial_state: vec!["memory.zeroed".into()],
                    external_events: vec![],
                },
                runs: vec![RunRecord {
                    id: "run.synthetic.0".into(),
                    ordinal: 0,
                    repetition: 0,
                    completion: "complete".into(),
                    output_artifact_paths: vec!["raw/output.bin".into()],
                    observations: vec![ObservationRecord {
                        id: "observation.synthetic.output".into(),
                        semantic_kind: "memory.snapshot".into(),
                        artifact_paths: vec!["raw/output.bin".into()],
                    }],
                    timing: vec![],
                    errors: vec![],
                    recovery_actions: vec![],
                    teardown: "clean".into(),
                    control_run_ids: vec![],
                }],
            },
            artifacts: vec![ArtifactSource {
                source_path: source,
                path: "raw/output.bin".into(),
                semantic_kind: "memory.snapshot".into(),
                class: ArtifactClass::Raw,
                redistributability: Redistributability::Redistributable,
                run_ids: vec!["run.synthetic.0".into()],
                observation_ids: vec!["observation.synthetic.output".into()],
                derivation: None,
            }],
        }
    }

    fn write_plan(root: &Path, blocked: bool) -> (PathBuf, PathBuf) {
        let source = root.join("source.bin");
        fs::write(&source, b"abc").unwrap();
        let mut plan = plan(source);
        if blocked {
            plan.campaign.platform.device_model_key =
                Availability::Unavailable { reason: "not recorded".into() };
        }
        let path = root.join("plan.json");
        fs::write(&path, serde_json::to_vec_pretty(&plan).unwrap()).unwrap();
        (path, root.join("bundle"))
    }

    fn write_v2_plan(root: &Path) -> (PathBuf, PathBuf) {
        let source = root.join("fixture.bin");
        fs::write(&source, b"abc").unwrap();
        let plan = EmissionPlanV2 {
            schema_version: EMISSION_PLAN_SCHEMA_VERSION_V2,
            payload: BundlePayload::Fixture(FixtureBody {
                id: "fixture.synthetic.cli".into(),
                semantic_kind: "npu_program".into(),
                provenance: Provenance::Current,
                source_revisions: vec![revision("fixture")],
                recipe: known(content("fixture-recipe.json")),
                notes: vec![],
            }),
            dependencies: vec![],
            artifacts: vec![ArtifactSource {
                source_path: source,
                path: "raw/fixture.bin".into(),
                semantic_kind: "input.binary".into(),
                class: ArtifactClass::Raw,
                redistributability: Redistributability::Redistributable,
                run_ids: vec![],
                observation_ids: vec![],
                derivation: None,
            }],
        };
        let path = root.join("plan-v2.json");
        fs::write(&path, serde_json::to_vec_pretty(&plan).unwrap()).unwrap();
        (path, root.join("bundle-v2"))
    }

    fn invoke(args: Vec<OsString>) -> (i32, String, String) {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let code = run(&args, &mut stdout, &mut stderr);
        (code, String::from_utf8(stdout).unwrap(), String::from_utf8(stderr).unwrap())
    }

    fn emit_args(plan: &Path, bundle: &Path) -> Vec<OsString> {
        vec!["emit".into(), plan.as_os_str().into(), bundle.as_os_str().into()]
    }

    fn validate_args(bundle: &Path) -> Vec<OsString> {
        vec!["validate".into(), bundle.as_os_str().into()]
    }

    fn location_plan(root: &Path, bundle: &Path) -> PathBuf {
        let validated = xdna_archspec::capture_bundle::validate_bundle(bundle).unwrap();
        let path = root.join("locations.json");
        let plan = xdna_archspec::research_reserve::BundleLocationPlan {
            roots: vec![xdna_archspec::research_reserve::BundleLocationRoot {
                alias: "synthetic".into(),
                path: root.to_owned(),
                failure_domain_id: "failure.synthetic".into(),
                bundles: vec![xdna_archspec::research_reserve::BundleLocationEntry {
                    bundle_id: validated.bundle_id().into(),
                    relative_path: bundle.strip_prefix(root).unwrap().to_str().unwrap().into(),
                }],
            }],
        };
        fs::write(&path, serde_json::to_vec_pretty(&plan).unwrap()).unwrap();
        path
    }

    #[test]
    fn rejects_missing_and_unknown_arguments() {
        assert_ne!(invoke(vec![]).0, 0);
        assert_ne!(invoke(vec!["unknown".into()]).0, 0);
    }

    #[test]
    fn emit_round_trips_a_synthetic_plan() {
        let temporary = tempfile::tempdir().unwrap();
        let (plan, bundle) = write_plan(temporary.path(), false);

        let (code, stdout, stderr) = invoke(emit_args(&plan, &bundle));
        assert_eq!(code, 0, "{stderr}");
        assert!(stdout.contains("bundle.sha256."));
        assert!(bundle.join("manifest.json").is_file());
    }

    #[test]
    fn emit_dispatches_a_v2_fixture_plan() {
        let temporary = tempfile::tempdir().unwrap();
        let (plan, bundle) = write_v2_plan(temporary.path());

        let (code, _, stderr) = invoke(emit_args(&plan, &bundle));
        assert_eq!(code, 0, "{stderr}");
        let manifest = fs::read_to_string(bundle.join("manifest.json")).unwrap();
        assert!(manifest.contains("\"schema_version\": 2"));
        assert!(manifest.contains("\"role\": \"fixture\""));
    }

    #[test]
    fn validate_prints_informational_identity_for_a_valid_bundle() {
        let temporary = tempfile::tempdir().unwrap();
        let (plan, bundle) = write_plan(temporary.path(), false);
        assert_eq!(invoke(emit_args(&plan, &bundle)).0, 0);

        let (code, stdout, stderr) = invoke(validate_args(&bundle));
        assert_eq!(code, 0, "{stderr}");
        for field in [
            "informational",
            "integrity: valid",
            "bundle_id: bundle.sha256.",
            "manifest_sha256:",
            "checksum_index_sha256:",
            "promotion: eligible",
        ] {
            assert!(stdout.contains(field), "{field}: {stdout}");
        }
    }

    #[test]
    fn validate_returns_nonzero_for_a_tampered_bundle() {
        let temporary = tempfile::tempdir().unwrap();
        let (plan, bundle) = write_plan(temporary.path(), false);
        assert_eq!(invoke(emit_args(&plan, &bundle)).0, 0);
        fs::write(bundle.join("raw/output.bin"), b"abd").unwrap();

        let (code, _, stderr) = invoke(validate_args(&bundle));
        assert_ne!(code, 0);
        assert!(stderr.contains("validation failed"));
    }

    #[test]
    fn validate_returns_nonzero_for_a_promotion_blocked_bundle() {
        let temporary = tempfile::tempdir().unwrap();
        let (plan, bundle) = write_plan(temporary.path(), true);
        assert_eq!(invoke(emit_args(&plan, &bundle)).0, 0);

        let (code, stdout, stderr) = invoke(validate_args(&bundle));
        assert_ne!(code, 0, "{stderr}");
        assert!(stdout.contains("integrity: valid"));
        assert!(stdout.contains("promotion: blocked"));
        assert!(stdout.contains("$.campaign.platform.device_model_key"));
    }

    #[test]
    fn validate_graph_prints_an_informational_node_report() {
        let temporary = tempfile::tempdir().unwrap();
        let (plan, bundle) = write_plan(temporary.path(), false);
        assert_eq!(invoke(emit_args(&plan, &bundle)).0, 0);
        let locations = location_plan(temporary.path(), &bundle);

        let (code, stdout, stderr) =
            invoke(vec!["validate-graph".into(), bundle.as_os_str().into(), locations.as_os_str().into()]);
        assert_eq!(code, 0, "{stderr}");
        assert!(stdout.contains("informational graph validation result"));
        assert!(stdout.contains("bundle_count: 1"));
    }

    #[test]
    fn audit_prints_a_blocked_report_and_returns_nonzero() {
        let temporary = tempfile::tempdir().unwrap();
        let ledger = temporary.path().join("ledger.json");
        let locations = temporary.path().join("locations.json");
        fs::write(&ledger, include_bytes!("../../data/research-reserve/npu1.json")).unwrap();
        fs::write(&locations, br#"{"roots":[]}"#).unwrap();

        let (code, stdout, stderr) = invoke(vec![
            "audit".into(),
            ledger.as_os_str().into(),
            "tuple.npu1.phoenix.fw-1_5_5_391".into(),
            locations.as_os_str().into(),
        ]);
        assert_ne!(code, 0, "{stderr}");
        assert!(stdout.contains("**Result: BLOCKED**"));
        assert!(stdout.contains("not a trusted receipt"));
    }
}
