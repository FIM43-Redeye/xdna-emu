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
//! llvm_aie_path = "/home/user/llvm-aie"
//!
//! # Path to mlir-aie repository (optional, for test discovery)
//! mlir_aie_path = "/home/user/mlir-aie"
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
        self.llvm_aie_path
            .clone()
            .unwrap_or_else(|| "../llvm-aie".to_string())
    }

    /// Get the mlir-aie path, with fallback to default.
    pub fn mlir_aie_path(&self) -> String {
        self.mlir_aie_path
            .clone()
            .unwrap_or_else(|| "../mlir-aie".to_string())
    }

    /// Get the XRT path, with fallback to default.
    pub fn xrt_path(&self) -> String {
        self.xrt_path
            .clone()
            .unwrap_or_else(|| "/opt/xilinx/xrt".to_string())
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
    }

    /// Get the path to the user config file (for display/creation).
    pub fn user_config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("xdna-emu").join("config.toml"))
    }

    /// Generate a sample config file content.
    pub fn sample_config() -> String {
        r#"# xdna-emu configuration
# Place this file at ~/.config/xdna-emu/config.toml or ./xdna-emu.toml

# Path to llvm-aie repository (required for instruction decoding)
# This should be an absolute path to your llvm-aie clone
llvm_aie_path = "/home/user/llvm-aie"

# Path to mlir-aie repository (optional, for test discovery)
# mlir_aie_path = "/home/user/mlir-aie"

# Path to XRT installation (optional, defaults to /opt/xilinx/xrt)
# xrt_path = "/opt/xilinx/xrt"
"#
        .to_string()
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
        assert_eq!(config.xrt_path(), "/opt/xilinx/xrt");
    }

    #[test]
    fn test_config_merge() {
        let mut base = Config {
            llvm_aie_path: Some("/base/llvm-aie".to_string()),
            mlir_aie_path: None,
            xrt_path: Some("/base/xrt".to_string()),
        };

        let overlay = Config {
            llvm_aie_path: None,
            mlir_aie_path: Some("/overlay/mlir-aie".to_string()),
            xrt_path: Some("/overlay/xrt".to_string()),
        };

        base.merge(overlay);

        // llvm_aie_path unchanged (overlay was None)
        assert_eq!(base.llvm_aie_path, Some("/base/llvm-aie".to_string()));
        // mlir_aie_path set from overlay
        assert_eq!(base.mlir_aie_path, Some("/overlay/mlir-aie".to_string()));
        // xrt_path overridden by overlay
        assert_eq!(base.xrt_path, Some("/overlay/xrt".to_string()));
    }

    #[test]
    fn test_sample_config_parses() {
        let sample = Config::sample_config();
        // Should parse without error (though paths won't exist)
        let _: Config = toml::from_str(&sample).expect("Sample config should parse");
    }
}
