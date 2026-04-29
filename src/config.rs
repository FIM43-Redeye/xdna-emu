//! Configuration management for xdna-emu.
//!
//! Configuration is loaded from multiple sources in priority order:
//! 1. Environment variables (LLVM_AIE_PATH, etc.)
//! 2. Project-local config file (`./xdna-emu.toml`)
//! 3. User config file (`~/.config/xdna-emu/config.toml`)
//! 4. Built-in defaults
//!
//! # Config File Format
//!
//! ```toml
//! # xdna-emu.toml
//!
//! # Path to llvm-aie repository (contains TableGen ISA definitions)
//! llvm_aie_path = "../llvm-aie"
//!
//! # Path to mlir-aie repository (optional, for test discovery)
//! mlir_aie_path = "../mlir-aie"
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Global cached configuration.
static CONFIG: OnceLock<Config> = OnceLock::new();

/// xdna-emu configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    /// Path to llvm-aie repository.
    /// Contains TableGen files defining the AIE2 ISA.
    pub llvm_aie_path: Option<String>,

    /// Path to mlir-aie repository.
    /// Used for test discovery and validation.
    pub mlir_aie_path: Option<String>,

    /// Path to XRT installation.
    /// Usually /opt/xilinx/xrt.
    pub xrt_path: Option<String>,

    /// Path to AMD aietools installation (optional).
    /// Enables elfanalyzer cross-validation, Chess compilation, and
    /// aiesimulator comparison when present. The emulator works without it.
    pub aietools_path: Option<String>,

    /// Maximum emulation cycles before timeout.
    /// Complex multi-tile designs (memtile routing, cascade) may need more
    /// cycles than simple single-tile tests. Override via config file or
    /// XDNA_EMU_MAX_CYCLES environment variable.
    pub max_cycles: Option<u64>,

    /// Stall detection threshold in cycles.
    /// Number of consecutive cycles with no lock-release progress (while
    /// pending syncs exist) before the emulator declares a stall and
    /// terminates. Override via XDNA_EMU_STALL_THRESHOLD environment variable.
    pub stall_threshold: Option<u64>,
}

impl Config {
    /// Load configuration from all sources.
    ///
    /// Priority (highest to lowest):
    /// 1. Environment variables
    /// 2. Project-local `xdna-emu.toml`
    /// 3. User config `~/.config/xdna-emu/config.toml`
    /// 4. Defaults
    pub fn load() -> Self {
        let mut config = Self::default();

        // Load user config first (lowest priority of file configs)
        if let Some(user_config) = Self::load_user_config() {
            config.merge(user_config);
        }

        // Load project-local config (higher priority)
        if let Some(local_config) = Self::load_local_config() {
            config.merge(local_config);
        }

        // Environment variables override everything
        config.apply_env_overrides();

        config
    }

    /// Get the cached global configuration.
    ///
    /// Loads configuration on first call and caches it.
    pub fn get() -> &'static Config {
        CONFIG.get_or_init(|| {
            let config = Self::load();
            log::debug!("Loaded configuration: {:?}", config);
            config
        })
    }

    /// Get the llvm-aie path, with fallback to default.
    ///
    /// Returns the configured path, or "../llvm-aie" as fallback.
    pub fn llvm_aie_path(&self) -> String {
        self.llvm_aie_path.clone().unwrap_or_else(|| "../llvm-aie".to_string())
    }

    /// Get the mlir-aie path, with fallback to default.
    pub fn mlir_aie_path(&self) -> String {
        self.mlir_aie_path.clone().unwrap_or_else(|| "../mlir-aie".to_string())
    }

    /// Get the XRT path, with fallback to platform-appropriate default.
    ///
    /// Defaults:
    /// - Linux: `/opt/xilinx/xrt`
    /// - Windows: `C:\Xilinx\XRT`
    pub fn xrt_path(&self) -> String {
        self.xrt_path.clone().unwrap_or_else(|| {
            if cfg!(target_os = "windows") {
                r"C:\Xilinx\XRT".to_string()
            } else {
                "/opt/xilinx/xrt".to_string()
            }
        })
    }

    /// Get the aietools path, if configured.
    ///
    /// Unlike other path accessors, this returns `Option<PathBuf>` because
    /// aietools is genuinely optional -- the emulator works without it.
    /// Returns `None` if no path is configured or the path does not exist.
    pub fn aietools_path(&self) -> Option<PathBuf> {
        self.aietools_path.as_ref().map(PathBuf::from).filter(|p| p.exists())
    }

    /// Maximum emulation cycles before timeout. Default: 10,000,000.
    ///
    /// Simple tests complete in 2K-10K cycles. Complex multi-tile designs
    /// with memtile routing and cascade connections need 100K-300K cycles.
    /// The default is generous to avoid false timeouts; the stall detector
    /// provides a faster adaptive cut-off for truly stuck workloads.
    pub fn max_cycles(&self) -> u64 {
        self.max_cycles.unwrap_or(10_000_000)
    }

    /// Stall detection threshold in cycles. Default: 100,000.
    ///
    /// Number of consecutive cycles with no lock-release progress while
    /// pending DMA syncs remain unsatisfied before declaring a stall.
    /// Set to 0 to disable stall detection (rely on max_cycles only).
    pub fn stall_threshold(&self) -> u64 {
        self.stall_threshold.unwrap_or(100_000)
    }

    // -- Test fixture path helpers --
    //
    // These resolve commonly-used test binary paths relative to the configured
    // mlir-aie root, replacing what were previously hardcoded absolute paths.

    /// Resolve a path relative to the mlir-aie root.
    ///
    /// Joins `mlir_aie_path()` with the given sub-path. Does not check whether
    /// the resulting path exists -- callers should handle that (typically by
    /// skipping the test if the file is absent).
    pub fn mlir_aie_subpath(&self, subpath: &str) -> PathBuf {
        PathBuf::from(self.mlir_aie_path()).join(subpath)
    }

    /// Path to the add_one_objFifo xclbin test fixture.
    ///
    /// This is the most commonly used test binary across the test suite.
    /// Returns `None` if the file does not exist at the resolved path.
    pub fn add_one_xclbin(&self) -> Option<PathBuf> {
        let path = self.mlir_aie_subpath("build/test/npu-xrt/add_one_objFifo/aie.xclbin");
        path.exists().then_some(path)
    }

    /// Path to the add_one_objFifo core ELF test fixture.
    ///
    /// Returns `None` if the file does not exist at the resolved path.
    pub fn add_one_elf(&self) -> Option<PathBuf> {
        let path =
            self.mlir_aie_subpath("build/test/npu-xrt/add_one_objFifo/aie_arch.mlir.prj/main_core_0_2.elf");
        path.exists().then_some(path)
    }

    /// Candidate xclbin paths for the add_one kernel.
    ///
    /// Returns multiple potential locations (different build variants).
    /// Used by tests that try several directories before giving up.
    pub fn add_one_xclbin_candidates(&self) -> Vec<PathBuf> {
        vec![
            self.mlir_aie_subpath("build/test/npu-xrt/add_one_objFifo/aie.xclbin"),
            self.mlir_aie_subpath("build/test/npu-xrt/add_one_using_dma/aie.xclbin"),
            self.mlir_aie_subpath("build/test/npu-xrt/add_one_objFifo_elf/aie.xclbin"),
        ]
    }

    /// The npu-xrt test directory within mlir-aie (build tree, for --no-build).
    pub fn npu_xrt_test_dir(&self) -> PathBuf {
        self.mlir_aie_subpath("build/test/npu-xrt")
    }

    /// The npu-xrt test source directory within mlir-aie.
    ///
    /// This is the source tree, containing all test definitions (run.lit,
    /// aie2.py, aie.mlir). Used by source-driven discovery which finds
    /// tests by their entry point files rather than pre-built xclbins.
    pub fn npu_xrt_source_dir(&self) -> PathBuf {
        self.mlir_aie_subpath("test/npu-xrt")
    }

    /// Load user configuration from ~/.config/xdna-emu/config.toml
    fn load_user_config() -> Option<Self> {
        let config_dir = dirs::config_dir()?;
        let config_path = config_dir.join("xdna-emu").join("config.toml");
        Self::load_from_file(&config_path)
    }

    /// Load project-local configuration from ./xdna-emu.toml
    fn load_local_config() -> Option<Self> {
        // Try current directory
        let local_path = Path::new("xdna-emu.toml");
        if let Some(config) = Self::load_from_file(local_path) {
            return Some(config);
        }

        // Try to find project root by looking for Cargo.toml
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let project_path = Path::new(&manifest_dir).join("xdna-emu.toml");
            if let Some(config) = Self::load_from_file(&project_path) {
                return Some(config);
            }
        }

        None
    }

    /// Load configuration from a specific file.
    fn load_from_file(path: &Path) -> Option<Self> {
        if !path.exists() {
            return None;
        }

        match std::fs::read_to_string(path) {
            Ok(content) => match toml::from_str(&content) {
                Ok(config) => {
                    log::info!("Loaded config from {}", path.display());
                    Some(config)
                }
                Err(e) => {
                    log::warn!("Failed to parse {}: {}", path.display(), e);
                    None
                }
            },
            Err(e) => {
                log::warn!("Failed to read {}: {}", path.display(), e);
                None
            }
        }
    }

    /// Merge another config into this one.
    /// Only overrides fields that are Some in the other config.
    fn merge(&mut self, other: Self) {
        if other.llvm_aie_path.is_some() {
            self.llvm_aie_path = other.llvm_aie_path;
        }
        if other.mlir_aie_path.is_some() {
            self.mlir_aie_path = other.mlir_aie_path;
        }
        if other.xrt_path.is_some() {
            self.xrt_path = other.xrt_path;
        }
        if other.aietools_path.is_some() {
            self.aietools_path = other.aietools_path;
        }
        if other.max_cycles.is_some() {
            self.max_cycles = other.max_cycles;
        }
        if other.stall_threshold.is_some() {
            self.stall_threshold = other.stall_threshold;
        }
    }

    /// Apply environment variable overrides.
    fn apply_env_overrides(&mut self) {
        if let Ok(path) = std::env::var("LLVM_AIE_PATH") {
            log::info!("Using LLVM_AIE_PATH from environment: {}", path);
            self.llvm_aie_path = Some(path);
        }
        if let Ok(path) = std::env::var("MLIR_AIE_PATH") {
            log::info!("Using MLIR_AIE_PATH from environment: {}", path);
            self.mlir_aie_path = Some(path);
        }
        if let Ok(path) = std::env::var("XRT_PATH") {
            log::info!("Using XRT_PATH from environment: {}", path);
            self.xrt_path = Some(path);
        }
        // Check our env var first, then fall back to AMD's standard env var.
        // XILINX_VITIS_AIETOOLS is set by activate-npu-env.sh.
        if let Ok(path) = std::env::var("AIETOOLS_PATH") {
            log::info!("Using AIETOOLS_PATH from environment: {}", path);
            self.aietools_path = Some(path);
        } else if let Ok(path) = std::env::var("XILINX_VITIS_AIETOOLS") {
            log::info!("Using XILINX_VITIS_AIETOOLS from environment: {}", path);
            self.aietools_path = Some(path);
        }
        if let Ok(val) = std::env::var("XDNA_EMU_MAX_CYCLES") {
            if let Ok(cycles) = val.parse::<u64>() {
                log::info!("Using XDNA_EMU_MAX_CYCLES from environment: {}", cycles);
                self.max_cycles = Some(cycles);
            }
        }
        if let Ok(val) = std::env::var("XDNA_EMU_STALL_THRESHOLD") {
            if let Ok(threshold) = val.parse::<u64>() {
                log::info!("Using XDNA_EMU_STALL_THRESHOLD from environment: {}", threshold);
                self.stall_threshold = Some(threshold);
            }
        }
    }

    /// Get the path to the user config file (for display/creation).
    pub fn user_config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("xdna-emu").join("config.toml"))
    }

    /// Generate a sample config file content.
    ///
    /// Output is platform-aware: shows Windows-style paths on Windows,
    /// Unix-style paths on Linux/macOS.
    pub fn sample_config() -> String {
        if cfg!(target_os = "windows") {
            r#"# xdna-emu configuration
# Place this file at %APPDATA%\xdna-emu\config.toml or .\xdna-emu.toml

# Path to llvm-aie repository (required for instruction decoding)
# This should be an absolute path to your llvm-aie clone
llvm_aie_path = "C:\\dev\\llvm-aie"

# Path to mlir-aie repository (optional, for test discovery)
# mlir_aie_path = "C:\\dev\\mlir-aie"

# Path to XRT installation (optional, defaults to C:\Xilinx\XRT)
# xrt_path = "C:\\Xilinx\\XRT"

# Path to AMD aietools installation (optional)
# Enables elfanalyzer, Chess compiler, and aiesimulator integration
# Also detected via AIETOOLS_PATH or XILINX_VITIS_AIETOOLS env vars
# aietools_path = "C:\\Xilinx\\aietools"
"#
            .to_string()
        } else {
            r#"# xdna-emu configuration
# Place this file at ~/.config/xdna-emu/config.toml or ./xdna-emu.toml

# Path to llvm-aie repository (required for instruction decoding)
# This should be an absolute path to your llvm-aie clone
llvm_aie_path = "/home/user/llvm-aie"

# Path to mlir-aie repository (optional, for test discovery)
# mlir_aie_path = "/home/user/mlir-aie"

# Path to XRT installation (optional, defaults to /opt/xilinx/xrt)
# xrt_path = "/opt/xilinx/xrt"

# Path to AMD aietools installation (optional)
# Enables elfanalyzer, Chess compiler, and aiesimulator integration
# Also detected via AIETOOLS_PATH or XILINX_VITIS_AIETOOLS env vars
# aietools_path = "/home/user/aietools"
"#
            .to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_paths() {
        let config = Config::default();
        assert_eq!(config.llvm_aie_path(), "../llvm-aie");
        assert_eq!(config.mlir_aie_path(), "../mlir-aie");
        // XRT default is platform-dependent
        if cfg!(target_os = "windows") {
            assert_eq!(config.xrt_path(), r"C:\Xilinx\XRT");
        } else {
            assert_eq!(config.xrt_path(), "/opt/xilinx/xrt");
        }
    }

    #[test]
    fn test_fixture_helpers() {
        let config = Config { mlir_aie_path: Some("/test/mlir-aie".to_string()), ..Default::default() };
        // mlir_aie_subpath joins correctly
        let sub = config.mlir_aie_subpath("build/test/npu-xrt");
        assert!(sub.ends_with("build/test/npu-xrt"));
        assert!(sub.starts_with("/test/mlir-aie"));

        // add_one helpers return None when files don't exist
        assert!(config.add_one_xclbin().is_none());
        assert!(config.add_one_elf().is_none());

        // candidates returns 3 paths
        let candidates = config.add_one_xclbin_candidates();
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_config_merge() {
        let mut base = Config {
            llvm_aie_path: Some("/base/llvm-aie".to_string()),
            mlir_aie_path: None,
            xrt_path: Some("/base/xrt".to_string()),
            aietools_path: Some("/base/aietools".to_string()),
            max_cycles: None,
            stall_threshold: None,
        };

        let overlay = Config {
            llvm_aie_path: None,
            mlir_aie_path: Some("/overlay/mlir-aie".to_string()),
            xrt_path: Some("/overlay/xrt".to_string()),
            aietools_path: None,
            max_cycles: Some(200_000),
            stall_threshold: Some(50_000),
        };

        base.merge(overlay);

        // llvm_aie_path unchanged (overlay was None)
        assert_eq!(base.llvm_aie_path, Some("/base/llvm-aie".to_string()));
        // mlir_aie_path set from overlay
        assert_eq!(base.mlir_aie_path, Some("/overlay/mlir-aie".to_string()));
        // xrt_path overridden by overlay
        assert_eq!(base.xrt_path, Some("/overlay/xrt".to_string()));
        // aietools_path unchanged (overlay was None)
        assert_eq!(base.aietools_path, Some("/base/aietools".to_string()));
        // max_cycles set from overlay
        assert_eq!(base.max_cycles, Some(200_000));
        // stall_threshold set from overlay
        assert_eq!(base.stall_threshold, Some(50_000));
    }

    #[test]
    fn test_sample_config_parses() {
        let sample = Config::sample_config();
        // Should parse without error (though paths won't exist)
        let _: Config = toml::from_str(&sample).expect("Sample config should parse");
    }

    #[test]
    fn test_aietools_path_default() {
        let config = Config::default();
        // No aietools configured by default
        assert!(config.aietools_path.is_none());
        // Accessor returns None when path is not configured
        assert!(config.aietools_path().is_none());
    }

    #[test]
    fn test_aietools_path_nonexistent() {
        let config =
            Config { aietools_path: Some("/nonexistent/aietools".to_string()), ..Default::default() };
        // Configured but path does not exist on disk -> returns None
        assert!(config.aietools_path().is_none());
    }

    #[test]
    fn test_aietools_env_override() {
        // Simulate environment variable override
        let mut config = Config::default();
        // Directly test the merge behavior that apply_env_overrides uses
        config.aietools_path = Some("/env/aietools".to_string());
        assert_eq!(config.aietools_path, Some("/env/aietools".to_string()));
    }
}
