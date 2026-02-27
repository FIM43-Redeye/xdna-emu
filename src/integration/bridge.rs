//! Subprocess bridge to mlir-aie Python API.
//!
//! Invokes `tools/mlir-aie-bridge.py` and parses JSON output.
//! Each subcommand returns structured Rust types.
//!
//! Falls back gracefully when mlir-aie is not available (bridge script
//! missing, Python not found, import errors).

use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolved paths needed to invoke the bridge script.
pub struct BridgePath {
    pub script: PathBuf,
    pub python: PathBuf,
}

impl BridgePath {
    /// Discover the bridge script relative to the crate root.
    ///
    /// Looks for `tools/mlir-aie-bridge.py` in the crate directory.
    /// For Python, prefers mlir-aie's ironenv virtualenv, then falls
    /// back to system `python3`.
    pub fn discover() -> Option<Self> {
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let script = crate_dir.join("tools/mlir-aie-bridge.py");
        if !script.exists() {
            return None;
        }

        // Try ironenv python first (has mlir-aie bindings pre-installed).
        let npu_work = crate_dir.parent()?;
        let ironenv_python = npu_work.join("mlir-aie/ironenv/bin/python3");
        let python = if ironenv_python.exists() {
            ironenv_python
        } else {
            PathBuf::from("python3")
        };

        Some(Self { script, python })
    }
}

/// Invoke a bridge subcommand and return parsed JSON.
pub fn invoke_bridge(
    bridge: &BridgePath,
    subcommand: &str,
    args: &[&str],
) -> Result<serde_json::Value, String> {
    let mut cmd = Command::new(&bridge.python);
    cmd.arg(&bridge.script).arg(subcommand);
    for arg in args {
        cmd.arg(arg);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run bridge: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Bridge {} failed: {}", subcommand, stderr));
    }

    serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Bridge {} returned invalid JSON: {}", subcommand, e))
}

/// Detected platform capabilities from mlir-aie bridge.
#[derive(Debug)]
pub struct PlatformInfo {
    /// NPU model identifier ("npu1", "npu2"), if hardware detected.
    pub npu_model: Option<String>,
    /// Architecture name ("AIE2", "AIE2P"), if hardware detected.
    pub arch: Option<String>,
    /// Hardware generation name ("Phoenix", "Strix"), if detected.
    pub npu_generation: Option<String>,
    /// Feature set matching mlir-aie lit conventions.
    pub features: Vec<String>,
    /// Whether Peano compiler (llc with AIE target) is available.
    pub has_peano: bool,
    /// Whether Chess compiler (xchesscc) is available.
    pub has_chess: bool,
    /// Whether aiesimulator is available.
    pub has_aiesimulator: bool,
}

impl PlatformInfo {
    /// Query the bridge for platform detection results.
    pub fn from_bridge(bridge: &BridgePath) -> Result<Self, String> {
        let json = invoke_bridge(bridge, "platform-detect", &[])?;

        let hardware = &json["hardware"];
        let tools = &json["tools"];

        Ok(Self {
            npu_model: hardware["npu_model"].as_str().map(|s| s.to_string()),
            arch: hardware["arch"].as_str().map(|s| s.to_string()),
            npu_generation: hardware["npu_generation"]
                .as_str()
                .map(|s| s.to_string()),
            features: json["features"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            has_peano: tools["peano"]["found"].as_bool().unwrap_or(false),
            has_chess: tools["chess"]["found"].as_bool().unwrap_or(false),
            has_aiesimulator: tools["aiesimulator"]["found"]
                .as_bool()
                .unwrap_or(false),
        })
    }

    /// Check if the detected platform supports a given device target.
    pub fn supports_device(&self, target_device: &str) -> bool {
        match self.npu_model.as_deref() {
            Some("npu1") => target_device.starts_with("npu1"),
            Some("npu2") => target_device.starts_with("npu2"),
            _ => false,
        }
    }
}

/// Build feasibility information for a single test.
#[derive(Debug)]
pub struct BuildFeasibility {
    pub makefile_exists: bool,
    pub kernel_sources_exist: bool,
    pub python_generator_exists: bool,
    pub missing_dependencies: Vec<String>,
}

impl BuildFeasibility {
    /// Whether the test has enough prerequisites to attempt a build.
    pub fn is_buildable(&self) -> bool {
        self.makefile_exists
            && self.missing_dependencies.is_empty()
    }

    /// Human-readable reason why the test cannot be built.
    pub fn reason(&self) -> String {
        if self.is_buildable() {
            return "buildable".to_string();
        }
        if !self.makefile_exists {
            return "no Makefile".to_string();
        }
        format!("missing: {}", self.missing_dependencies.join(", "))
    }
}

/// Metadata for a single test extracted from the manifest.
#[derive(Debug)]
pub struct TestEntry {
    pub name: String,
    pub path: String,
    pub target_device: String,
    pub target_arch: String,
    pub requires: Vec<String>,
    pub compilers: Vec<String>,
    pub build_feasibility: BuildFeasibility,
    pub skip_reason: Option<String>,
}

/// Complete test manifest from the bridge.
#[derive(Debug)]
pub struct TestManifest {
    pub npu_xrt_tests: Vec<TestEntry>,
    pub examples: Vec<TestEntry>,
    pub total: usize,
    pub buildable: usize,
}

impl TestManifest {
    /// Query the bridge for a test manifest.
    pub fn from_bridge(
        bridge: &BridgePath,
        npu_xrt_dir: &Path,
        examples_dir: &Path,
    ) -> Result<Self, String> {
        let json = invoke_bridge(
            bridge,
            "test-manifest",
            &[
                "--npu-xrt-dir",
                &npu_xrt_dir.to_string_lossy(),
                "--examples-dir",
                &examples_dir.to_string_lossy(),
            ],
        )?;

        fn parse_tests(arr: &serde_json::Value) -> Vec<TestEntry> {
            arr.as_array()
                .map(|tests| {
                    tests
                        .iter()
                        .map(|t| {
                            let feasibility = &t["build_feasibility"];
                            TestEntry {
                                name: t["name"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string(),
                                path: t["path"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string(),
                                target_device: t["target_device"]
                                    .as_str()
                                    .unwrap_or("npu1")
                                    .to_string(),
                                target_arch: t["target_arch"]
                                    .as_str()
                                    .unwrap_or("AIE2")
                                    .to_string(),
                                requires: t["requires"]
                                    .as_array()
                                    .map(|a| {
                                        a.iter()
                                            .filter_map(|v| {
                                                v.as_str().map(|s| s.to_string())
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default(),
                                compilers: t["compilers"]
                                    .as_array()
                                    .map(|a| {
                                        a.iter()
                                            .filter_map(|v| {
                                                v.as_str().map(|s| s.to_string())
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default(),
                                build_feasibility: BuildFeasibility {
                                    makefile_exists: feasibility
                                        ["makefile_exists"]
                                        .as_bool()
                                        .unwrap_or(false),
                                    kernel_sources_exist: feasibility
                                        ["kernel_sources_exist"]
                                        .as_bool()
                                        .unwrap_or(false),
                                    python_generator_exists: feasibility
                                        ["python_generator_exists"]
                                        .as_bool()
                                        .unwrap_or(false),
                                    missing_dependencies: feasibility
                                        ["missing_dependencies"]
                                        .as_array()
                                        .map(|a| {
                                            a.iter()
                                                .filter_map(|v| {
                                                    v.as_str()
                                                        .map(|s| s.to_string())
                                                })
                                                .collect()
                                        })
                                        .unwrap_or_default(),
                                },
                                skip_reason: t["skip_reason"]
                                    .as_str()
                                    .map(|s| s.to_string()),
                            }
                        })
                        .collect()
                })
                .unwrap_or_default()
        }

        let npu_xrt_tests = parse_tests(&json["npu_xrt_tests"]);
        let examples = parse_tests(&json["examples"]);
        let summary = &json["summary"];

        Ok(Self {
            total: summary["total"].as_u64().unwrap_or(0) as usize,
            buildable: summary["buildable"].as_u64().unwrap_or(0) as usize,
            npu_xrt_tests,
            examples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_if_no_bridge() -> Option<BridgePath> {
        BridgePath::discover()
    }

    #[test]
    fn test_bridge_discovery() {
        // Bridge script should exist in our repo.
        let bridge = BridgePath::discover();
        assert!(bridge.is_some(), "bridge script not found");
        let bridge = bridge.unwrap();
        assert!(bridge.script.exists());
    }

    #[test]
    fn test_invoke_device_model() {
        let bridge = match skip_if_no_bridge() {
            Some(b) => b,
            None => return,
        };
        let result = invoke_bridge(&bridge, "device-model", &["--device", "npu1"]);
        assert!(result.is_ok(), "device-model failed: {:?}", result.err());
        let json = result.unwrap();
        let npu1 = &json["devices"]["npu1"];
        assert!(npu1["columns"].as_u64().unwrap() > 0);
        assert!(npu1["rows"].as_u64().unwrap() > 0);
        assert!(npu1["is_npu"].as_bool().unwrap());

        // Verify new fields exist.
        let tiles = npu1["tile_map"].as_array().unwrap();
        let core_tile = tiles.iter().find(|t| t["type"] == "core").unwrap();
        assert!(core_tile.get("is_internal").is_some());
        assert!(core_tile.get("edges").is_some());
        assert!(core_tile.get("mem_affinity").is_some());
    }

    #[test]
    fn test_invoke_platform_detect() {
        let bridge = match skip_if_no_bridge() {
            Some(b) => b,
            None => return,
        };
        let result = invoke_bridge(&bridge, "platform-detect", &[]);
        assert!(result.is_ok(), "platform-detect failed: {:?}", result.err());
        let json = result.unwrap();
        assert!(json["features"].is_array());
    }

    #[test]
    fn test_platform_info() {
        let bridge = match skip_if_no_bridge() {
            Some(b) => b,
            None => return,
        };
        let info = PlatformInfo::from_bridge(&bridge);
        assert!(info.is_ok(), "PlatformInfo failed: {:?}", info.err());
        let info = info.unwrap();
        // On our dev machine we should see npu1.
        if info.npu_model.is_some() {
            assert!(info.supports_device("npu1"));
            assert!(!info.supports_device("npu2"));
        }
    }

    #[test]
    fn test_invoke_trace_events() {
        let bridge = match skip_if_no_bridge() {
            Some(b) => b,
            None => return,
        };
        let result = invoke_bridge(&bridge, "trace-events", &[]);
        assert!(result.is_ok(), "trace-events failed: {:?}", result.err());
        let json = result.unwrap();
        let core = &json["enums"]["CoreEvent"];
        // INSTR_VECTOR should be defined.
        assert!(
            core["INSTR_VECTOR"].as_u64().is_some(),
            "missing INSTR_VECTOR"
        );
    }

    #[test]
    fn test_build_feasibility() {
        let f = BuildFeasibility {
            makefile_exists: true,
            kernel_sources_exist: true,
            python_generator_exists: false,
            missing_dependencies: vec![],
        };
        assert!(f.is_buildable());
        assert_eq!(f.reason(), "buildable");

        let f2 = BuildFeasibility {
            makefile_exists: false,
            kernel_sources_exist: true,
            python_generator_exists: false,
            missing_dependencies: vec!["Makefile".to_string()],
        };
        assert!(!f2.is_buildable());
        assert_eq!(f2.reason(), "no Makefile");
    }

    #[test]
    fn test_supports_device() {
        let info = PlatformInfo {
            npu_model: Some("npu1".to_string()),
            arch: Some("AIE2".to_string()),
            npu_generation: Some("Phoenix".to_string()),
            features: vec!["ryzen_ai".to_string(), "ryzen_ai_npu1".to_string()],
            has_peano: true,
            has_chess: true,
            has_aiesimulator: true,
        };

        assert!(info.supports_device("npu1"));
        assert!(info.supports_device("npu1_2col"));
        assert!(!info.supports_device("npu2"));
        assert!(!info.supports_device("npu2_3col"));
    }
}
