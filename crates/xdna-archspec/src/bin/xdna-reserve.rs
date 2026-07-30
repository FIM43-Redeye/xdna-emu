use std::{
    env,
    ffi::OsString,
    fs,
    io::{self, Write},
    path::Path,
    process,
};
use xdna_archspec::capture_bundle::{emit_bundle, validate_bundle, EmissionPlan};

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
            let plan: EmissionPlan = match serde_json::from_slice(&bytes) {
                Ok(plan) => plan,
                Err(error) => return fail(stderr, format!("invalid emission plan: {error}")),
            };
            match emit_bundle(&plan, Path::new(output)) {
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
        _ => {
            let _ = writeln!(
                stderr,
                "usage: xdna-reserve emit <emission-plan.json> <output-bundle>\n       xdna-reserve validate <bundle>"
            );
            2
        }
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
            ComponentPin, EmissionPlan, ObservationRecord, PciIdentity, PlatformIdentity, Provenance,
            RunRecord, Stimulus, EMISSION_PLAN_SCHEMA_VERSION,
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
}
