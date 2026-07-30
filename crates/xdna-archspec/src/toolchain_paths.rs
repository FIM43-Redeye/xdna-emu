//! Provider-neutral local NPU toolchain path resolution.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

const MLIR_AIE_SENTINELS: &[Sentinel] = &[Sentinel::file("lib/Dialect/AIE/Util/aie_registers_aie2.json")];
const LLVM_AIE_SENTINELS: &[Sentinel] =
    &[Sentinel::file("llvm/lib/Target/AIE/AIE2.td"), Sentinel::executable("build/bin/llvm-config")];
const AIE_RT_SENTINELS: &[Sentinel] = &[Sentinel::file("global/xaiemlgbl_reginit.c")];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolchainPaths {
    pub mlir_aie: PathBuf,
    pub llvm_aie: PathBuf,
    pub aie_rt: PathBuf,
}

impl ToolchainPaths {
    pub fn resolve(workspace_root: &Path) -> Result<Self, String> {
        Self::resolve_with_env(workspace_root, &|name| std::env::var_os(name))
    }

    pub fn resolve_with_env<F>(workspace_root: &Path, env: &F) -> Result<Self, String>
    where
        F: Fn(&str) -> Option<OsString>,
    {
        let mlir_aie = Self::resolve_mlir_aie_with_env(workspace_root, env)?
            .ok_or_else(|| missing_component("mlir-aie", workspace_root, "mlir-aie", MLIR_AIE_SENTINELS))?;
        let llvm_aie = resolve_component(
            workspace_root,
            env,
            Component {
                name: "llvm-aie",
                primary_env: "LLVM_AIE_PATH",
                alias_env: Some("LLVM_AIE_DIR"),
                standard_path: "llvm-aie",
                sentinels: LLVM_AIE_SENTINELS,
            },
        )?
        .ok_or_else(|| missing_component("llvm-aie", workspace_root, "llvm-aie", LLVM_AIE_SENTINELS))?;
        let aie_rt = resolve_component(
            workspace_root,
            env,
            Component {
                name: "aie-rt",
                primary_env: "AIE_RT_PATH",
                alias_env: None,
                standard_path: "aie-rt/driver/src",
                sentinels: AIE_RT_SENTINELS,
            },
        )?
        .ok_or_else(|| missing_component("aie-rt", workspace_root, "aie-rt/driver/src", AIE_RT_SENTINELS))?;

        Ok(Self { mlir_aie, llvm_aie, aie_rt })
    }

    pub fn resolve_mlir_aie(workspace_root: &Path) -> Result<Option<PathBuf>, String> {
        Self::resolve_mlir_aie_with_env(workspace_root, &|name| std::env::var_os(name))
    }

    pub fn resolve_mlir_aie_with_env<F>(workspace_root: &Path, env: &F) -> Result<Option<PathBuf>, String>
    where
        F: Fn(&str) -> Option<OsString>,
    {
        resolve_component(
            workspace_root,
            env,
            Component {
                name: "mlir-aie",
                primary_env: "MLIR_AIE_PATH",
                alias_env: Some("MLIR_AIE_DIR"),
                standard_path: "mlir-aie",
                sentinels: MLIR_AIE_SENTINELS,
            },
        )
    }
}

#[derive(Clone, Copy)]
struct Component {
    name: &'static str,
    primary_env: &'static str,
    alias_env: Option<&'static str>,
    standard_path: &'static str,
    sentinels: &'static [Sentinel],
}

#[derive(Clone, Copy)]
struct Sentinel {
    path: &'static str,
    executable: bool,
}

impl Sentinel {
    const fn file(path: &'static str) -> Self {
        Self { path, executable: false }
    }

    const fn executable(path: &'static str) -> Self {
        Self { path, executable: true }
    }
}

fn resolve_component<F>(
    workspace_root: &Path,
    env: &F,
    component: Component,
) -> Result<Option<PathBuf>, String>
where
    F: Fn(&str) -> Option<OsString>,
{
    if let Some(path) = env(component.primary_env) {
        return validate_configured(component, component.primary_env, &path).map(Some);
    }

    if let Some(alias) = component.alias_env {
        if let Some(path) = env(alias) {
            return validate_configured(component, alias, &path).map(Some);
        }
    }

    if let Some(npu_work) = env("NPU_WORK_DIR") {
        if npu_work.is_empty() {
            return Err(format!("{} path selected by NPU_WORK_DIR is blank", component.name));
        }
        let candidate = PathBuf::from(npu_work).join(component.standard_path);
        return validate_candidate(component, "NPU_WORK_DIR", &candidate).map(Some);
    }

    for ancestor in workspace_root.ancestors() {
        let candidate = ancestor.join(component.standard_path);
        if candidate.exists() {
            let source = format!("ancestor discovery at {}", ancestor.display());
            return validate_candidate(component, &source, &candidate).map(Some);
        }
    }

    Ok(None)
}

fn validate_configured(component: Component, source: &str, value: &OsStr) -> Result<PathBuf, String> {
    if value.is_empty() {
        return Err(format!("{} path selected by {} is blank", component.name, source));
    }
    validate_candidate(component, source, Path::new(value))
}

fn validate_candidate(component: Component, source: &str, candidate: &Path) -> Result<PathBuf, String> {
    if !candidate.is_dir() {
        return Err(format!(
            "{} path selected by {} is not a directory: {}",
            component.name,
            source,
            candidate.display()
        ));
    }

    for sentinel in component.sentinels {
        let path = candidate.join(sentinel.path);
        if !path.is_file() {
            return Err(format!(
                "{} path selected by {} is missing required sentinel {}: {}",
                component.name,
                source,
                sentinel.path,
                path.display()
            ));
        }
        if sentinel.executable {
            let mode = fs::metadata(&path)
                .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?
                .permissions()
                .mode();
            if mode & 0o111 == 0 {
                return Err(format!(
                    "{} path selected by {} has non-executable sentinel {}: {}",
                    component.name,
                    source,
                    sentinel.path,
                    path.display()
                ));
            }
        }
    }

    candidate.canonicalize().map_err(|error| {
        format!(
            "cannot canonicalize {} path selected by {} at {}: {error}",
            component.name,
            source,
            candidate.display()
        )
    })
}

fn missing_component(
    name: &str,
    workspace_root: &Path,
    standard_path: &str,
    sentinels: &[Sentinel],
) -> String {
    let searched = workspace_root
        .ancestors()
        .map(|ancestor| ancestor.join(standard_path).display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let required = sentinels.iter().map(|sentinel| sentinel.path).collect::<Vec<_>>().join(", ");
    format!(
        "{name} was not found by ancestor discovery; required sentinels: {required}; searched: {searched}"
    )
}
