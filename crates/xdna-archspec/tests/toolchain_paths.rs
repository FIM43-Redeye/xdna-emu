use std::collections::HashMap;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use tempfile::TempDir;
use xdna_archspec::toolchain_paths::ToolchainPaths;

fn write_file(path: &Path) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, "").unwrap();
}

fn write_executable(path: &Path) {
    write_file(path);
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}

fn create_toolchains(npu_work: &Path) {
    write_file(&npu_work.join("mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json"));
    write_file(&npu_work.join("llvm-aie/llvm/lib/Target/AIE/AIE2.td"));
    write_executable(&npu_work.join("llvm-aie/build/bin/llvm-config"));
    write_file(&npu_work.join("aie-rt/driver/src/global/xaiemlgbl_reginit.c"));
}

fn env(values: &[(&str, &Path)]) -> HashMap<String, OsString> {
    values
        .iter()
        .map(|(name, path)| ((*name).to_string(), path.as_os_str().to_owned()))
        .collect()
}

fn resolve(workspace: &Path, values: &HashMap<String, OsString>) -> Result<ToolchainPaths, String> {
    ToolchainPaths::resolve_with_env(workspace, &|name| values.get(name).cloned())
}

#[test]
fn discovers_toolchains_from_main_checkout_layout() {
    let temp = TempDir::new().unwrap();
    let npu_work = temp.path().join("npu-work");
    let workspace = npu_work.join("xdna-emu");
    fs::create_dir_all(&workspace).unwrap();
    create_toolchains(&npu_work);

    let paths = resolve(&workspace, &HashMap::new()).unwrap();

    assert_eq!(paths.mlir_aie, npu_work.join("mlir-aie").canonicalize().unwrap());
    assert_eq!(paths.llvm_aie, npu_work.join("llvm-aie").canonicalize().unwrap());
    assert_eq!(paths.aie_rt, npu_work.join("aie-rt/driver/src").canonicalize().unwrap());
}

#[test]
fn discovers_toolchains_from_nested_worktree_layout() {
    let temp = TempDir::new().unwrap();
    let npu_work = temp.path().join("npu-work");
    let workspace = npu_work.join("xdna-emu/.worktrees/firmware-priors");
    fs::create_dir_all(&workspace).unwrap();
    create_toolchains(&npu_work);

    let paths = resolve(&workspace, &HashMap::new()).unwrap();

    assert_eq!(paths.mlir_aie, npu_work.join("mlir-aie").canonicalize().unwrap());
    assert_eq!(paths.llvm_aie, npu_work.join("llvm-aie").canonicalize().unwrap());
    assert_eq!(paths.aie_rt, npu_work.join("aie-rt/driver/src").canonicalize().unwrap());
}

#[test]
fn primary_component_overrides_win_over_aliases_and_npu_work_dir() {
    let temp = TempDir::new().unwrap();
    let workspace = temp.path().join("workspace");
    let primary = temp.path().join("primary");
    let aliases = temp.path().join("aliases");
    let configured_npu_work = temp.path().join("configured");
    fs::create_dir_all(&workspace).unwrap();
    create_toolchains(&primary);
    create_toolchains(&aliases);
    create_toolchains(&configured_npu_work);

    let values = env(&[
        ("MLIR_AIE_PATH", &primary.join("mlir-aie")),
        ("MLIR_AIE_DIR", &aliases.join("mlir-aie")),
        ("LLVM_AIE_PATH", &primary.join("llvm-aie")),
        ("LLVM_AIE_DIR", &aliases.join("llvm-aie")),
        ("AIE_RT_PATH", &primary.join("aie-rt/driver/src")),
        ("NPU_WORK_DIR", &configured_npu_work),
    ]);

    let paths = resolve(&workspace, &values).unwrap();

    assert_eq!(paths.mlir_aie, primary.join("mlir-aie").canonicalize().unwrap());
    assert_eq!(paths.llvm_aie, primary.join("llvm-aie").canonicalize().unwrap());
    assert_eq!(paths.aie_rt, primary.join("aie-rt/driver/src").canonicalize().unwrap());
}

#[test]
fn activation_aliases_resolve_mlir_aie_and_llvm_aie() {
    let temp = TempDir::new().unwrap();
    let workspace = temp.path().join("workspace");
    let toolchains = temp.path().join("aliases");
    fs::create_dir_all(&workspace).unwrap();
    create_toolchains(&toolchains);

    let values = env(&[
        ("MLIR_AIE_DIR", &toolchains.join("mlir-aie")),
        ("LLVM_AIE_DIR", &toolchains.join("llvm-aie")),
        ("AIE_RT_PATH", &toolchains.join("aie-rt/driver/src")),
    ]);

    let paths = resolve(&workspace, &values).unwrap();

    assert_eq!(paths.mlir_aie, toolchains.join("mlir-aie").canonicalize().unwrap());
    assert_eq!(paths.llvm_aie, toolchains.join("llvm-aie").canonicalize().unwrap());
}

#[test]
fn npu_work_dir_resolves_all_standard_component_paths() {
    let temp = TempDir::new().unwrap();
    let workspace = temp.path().join("workspace");
    let npu_work = temp.path().join("configured");
    fs::create_dir_all(&workspace).unwrap();
    create_toolchains(&npu_work);

    let values = env(&[("NPU_WORK_DIR", &npu_work)]);
    let paths = resolve(&workspace, &values).unwrap();

    assert_eq!(paths.mlir_aie, npu_work.join("mlir-aie").canonicalize().unwrap());
    assert_eq!(paths.llvm_aie, npu_work.join("llvm-aie").canonicalize().unwrap());
    assert_eq!(paths.aie_rt, npu_work.join("aie-rt/driver/src").canonicalize().unwrap());
}

#[test]
fn invalid_npu_work_dir_fails_without_ancestor_fallback() {
    let temp = TempDir::new().unwrap();
    let valid_npu_work = temp.path().join("npu-work");
    let workspace = valid_npu_work.join("xdna-emu");
    let invalid_npu_work = temp.path().join("invalid");
    fs::create_dir_all(&workspace).unwrap();
    fs::create_dir_all(&invalid_npu_work).unwrap();
    create_toolchains(&valid_npu_work);

    let values = env(&[("NPU_WORK_DIR", &invalid_npu_work)]);
    let error = resolve(&workspace, &values).unwrap_err();

    assert!(error.contains("NPU_WORK_DIR"));
    assert!(error.contains("mlir-aie"));
}

#[test]
fn invalid_primary_override_fails_without_falling_back() {
    let temp = TempDir::new().unwrap();
    let npu_work = temp.path().join("npu-work");
    let workspace = npu_work.join("xdna-emu");
    let invalid = temp.path().join("invalid-mlir-aie");
    fs::create_dir_all(&workspace).unwrap();
    fs::create_dir_all(&invalid).unwrap();
    create_toolchains(&npu_work);

    let values = env(&[("MLIR_AIE_PATH", &invalid)]);
    let error = resolve(&workspace, &values).unwrap_err();

    assert!(error.contains("mlir-aie"));
    assert!(error.contains("MLIR_AIE_PATH"));
    assert!(error.contains("aie_registers_aie2.json"));
}

#[test]
fn non_executable_llvm_config_is_rejected() {
    let temp = TempDir::new().unwrap();
    let npu_work = temp.path().join("npu-work");
    let workspace = npu_work.join("xdna-emu");
    fs::create_dir_all(&workspace).unwrap();
    create_toolchains(&npu_work);

    let llvm_config = npu_work.join("llvm-aie/build/bin/llvm-config");
    let mut permissions = fs::metadata(&llvm_config).unwrap().permissions();
    permissions.set_mode(0o644);
    fs::set_permissions(&llvm_config, permissions).unwrap();

    let values = env(&[
        ("MLIR_AIE_PATH", &npu_work.join("mlir-aie")),
        ("LLVM_AIE_PATH", &npu_work.join("llvm-aie")),
        ("AIE_RT_PATH", &npu_work.join("aie-rt/driver/src")),
    ]);
    let error = resolve(&workspace, &values).unwrap_err();

    assert!(error.contains("LLVM_AIE_PATH"));
    assert!(error.contains("non-executable"));
    assert!(error.contains("build/bin/llvm-config"));
}

#[test]
fn blank_override_is_an_error() {
    let temp = TempDir::new().unwrap();
    let npu_work = temp.path().join("npu-work");
    let workspace = npu_work.join("xdna-emu");
    fs::create_dir_all(&workspace).unwrap();
    create_toolchains(&npu_work);

    let values = HashMap::from([("LLVM_AIE_PATH".to_string(), OsString::new())]);
    let error = resolve(&workspace, &values).unwrap_err();

    assert!(error.contains("LLVM_AIE_PATH"));
    assert!(error.contains("blank"));
}

#[test]
fn missing_component_sentinels_are_reported() {
    let cases = [
        ("mlir-aie", "lib/Dialect/AIE/Util/aie_registers_aie2.json"),
        ("llvm-aie", "llvm/lib/Target/AIE/AIE2.td"),
        ("llvm-aie", "build/bin/llvm-config"),
        ("aie-rt", "global/xaiemlgbl_reginit.c"),
    ];

    for (component, sentinel) in cases {
        let temp = TempDir::new().unwrap();
        let npu_work = temp.path().join("npu-work");
        let workspace = npu_work.join("xdna-emu");
        fs::create_dir_all(&workspace).unwrap();
        create_toolchains(&npu_work);

        let component_root = match component {
            "mlir-aie" => npu_work.join("mlir-aie"),
            "llvm-aie" => npu_work.join("llvm-aie"),
            "aie-rt" => npu_work.join("aie-rt/driver/src"),
            _ => unreachable!(),
        };
        fs::remove_file(component_root.join(sentinel)).unwrap();

        let values = env(&[(
            match component {
                "mlir-aie" => "MLIR_AIE_PATH",
                "llvm-aie" => "LLVM_AIE_PATH",
                "aie-rt" => "AIE_RT_PATH",
                _ => unreachable!(),
            },
            &component_root,
        )]);
        let error = resolve(&workspace, &values).unwrap_err();

        assert!(error.contains(component), "{error}");
        assert!(error.contains(sentinel), "{error}");
    }
}

#[test]
fn aie_rt_override_is_the_driver_src_directory() {
    let temp = TempDir::new().unwrap();
    let workspace = temp.path().join("workspace");
    let toolchains = temp.path().join("toolchains");
    fs::create_dir_all(&workspace).unwrap();
    create_toolchains(&toolchains);

    let driver_src = toolchains.join("aie-rt/driver/src");
    let values = env(&[
        ("MLIR_AIE_PATH", &toolchains.join("mlir-aie")),
        ("LLVM_AIE_PATH", &toolchains.join("llvm-aie")),
        ("AIE_RT_PATH", &driver_src),
    ]);
    let paths = resolve(&workspace, &values).unwrap();

    assert_eq!(paths.aie_rt, driver_src.canonicalize().unwrap());
}

#[test]
fn component_only_resolution_does_not_require_other_toolchains() {
    let temp = TempDir::new().unwrap();
    let workspace = temp.path().join("workspace");
    let mlir_aie = temp.path().join("mlir-aie");
    fs::create_dir_all(&workspace).unwrap();
    write_file(&mlir_aie.join("lib/Dialect/AIE/Util/aie_registers_aie2.json"));
    let values = env(&[("MLIR_AIE_PATH", &mlir_aie)]);

    let resolved =
        ToolchainPaths::resolve_mlir_aie_with_env(&workspace, &|name| values.get(name).cloned()).unwrap();

    assert_eq!(resolved, Some(mlir_aie.canonicalize().unwrap()));
}

#[test]
fn absent_component_only_resolution_returns_none() {
    let temp = TempDir::new().unwrap();
    let workspace = temp.path().join("workspace");
    fs::create_dir_all(&workspace).unwrap();

    let resolved = ToolchainPaths::resolve_mlir_aie_with_env(&workspace, &|_| None).unwrap();

    assert_eq!(resolved, None);
}

#[test]
fn resolved_paths_are_absolute() {
    let temp = TempDir::new().unwrap();
    let npu_work = temp.path().join("npu-work");
    let workspace = npu_work.join("xdna-emu/.worktrees/test");
    fs::create_dir_all(&workspace).unwrap();
    create_toolchains(&npu_work);

    let paths = resolve(&workspace, &HashMap::new()).unwrap();

    assert!(paths.mlir_aie.is_absolute());
    assert!(paths.llvm_aie.is_absolute());
    assert!(paths.aie_rt.is_absolute());
}
