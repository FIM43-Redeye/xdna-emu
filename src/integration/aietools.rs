//! AMD aietools discovery and invocation.
//!
//! aietools is an optional dependency that provides:
//! - **elfanalyzer**: Static analysis of AIE ELF binaries
//! - **xchesscc**: Chess compiler (via xchesscc_wrapper)
//! - **aiesimulator**: Cycle-accurate simulation
//! - **hwanalyze**: Hardware trace analysis
//!
//! Discovery order:
//! 1. `Config::aietools_path` (from config file or `AIETOOLS_PATH` env)
//! 2. `XILINX_VITIS_AIETOOLS` environment variable (set by activate-npu-env.sh)
//! 3. Sibling directory `../aietools` (relative to this project)
//!
//! # LD_LIBRARY_PATH Warning
//!
//! aietools ships its own libstdc++ and other system libraries. These MUST be
//! appended (never prepended) to `LD_LIBRARY_PATH`, or they shadow the system
//! libs and break unrelated binaries including llvm-tblgen. The `command()`
//! method handles this automatically.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::Config;

/// Which aietools binary to invoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    Elfanalyzer,
    Xchesscc,
    Aiesimulator,
    Hwanalyze,
}

impl Tool {
    /// Binary name for this tool (platform-dependent).
    fn binary_name(&self) -> &'static str {
        if cfg!(target_os = "windows") {
            match self {
                Tool::Elfanalyzer => "elfanalyzer.exe",
                Tool::Xchesscc => "xchesscc.exe",
                Tool::Aiesimulator => "aiesimulator.exe",
                Tool::Hwanalyze => "hwanalyze.exe",
            }
        } else {
            match self {
                Tool::Elfanalyzer => "elfanalyzer",
                Tool::Xchesscc => "xchesscc",
                Tool::Aiesimulator => "aiesimulator",
                Tool::Hwanalyze => "hwanalyze",
            }
        }
    }

    /// Whether this tool requires a Xilinx license to run.
    pub fn requires_license(&self) -> bool {
        match self {
            // elfanalyzer is a static analysis tool, no license needed
            Tool::Elfanalyzer => false,
            // Chess compiler, simulator, and hardware analyzer need licenses
            Tool::Xchesscc => true,
            Tool::Aiesimulator => true,
            Tool::Hwanalyze => true,
        }
    }
}

/// Discovered aietools installation.
///
/// Each tool path is probed individually -- a partial installation (e.g.,
/// elfanalyzer present but aiesimulator absent) is handled gracefully.
#[derive(Debug, Clone)]
pub struct AieTools {
    /// Root directory of the aietools installation.
    pub(crate) root: PathBuf,
    /// Paths to individual tools (None if not found in this installation).
    pub elfanalyzer: Option<PathBuf>,
    pub xchesscc: Option<PathBuf>,
    pub aiesimulator: Option<PathBuf>,
    pub hwanalyze: Option<PathBuf>,
}

impl AieTools {
    /// Discover an aietools installation.
    ///
    /// Search order:
    /// 1. Config file / AIETOOLS_PATH / XILINX_VITIS_AIETOOLS (via Config)
    /// 2. `XILINX_VITIS_AIETOOLS` env var (direct check, in case Config
    ///    was loaded before the env was set)
    /// 3. `../aietools` relative to the working directory
    ///
    /// Returns `None` if no valid installation is found.
    pub fn discover(config: &Config) -> Option<Self> {
        // Try config first (already incorporates env var overrides)
        if let Some(path) = config.aietools_path() {
            if let Some(tools) = Self::probe(&path) {
                return Some(tools);
            }
        }

        // Direct env var check (covers cases where Config was cached
        // before the environment was fully set up)
        if let Ok(path) = std::env::var("XILINX_VITIS_AIETOOLS") {
            let p = PathBuf::from(&path);
            if p.exists() {
                if let Some(tools) = Self::probe(&p) {
                    return Some(tools);
                }
            }
        }

        // Sibling directory fallback
        let sibling = PathBuf::from("../aietools");
        if let Ok(canonical) = sibling.canonicalize() {
            if let Some(tools) = Self::probe(&canonical) {
                return Some(tools);
            }
        }

        log::debug!("No aietools installation found (this is OK -- aietools is optional)");
        None
    }

    /// Probe a directory to see if it's a valid aietools installation.
    ///
    /// Requires the `bin/` subdirectory to exist. Each tool is probed
    /// independently -- missing individual tools are tolerated.
    fn probe(root: &Path) -> Option<Self> {
        let bin_dir = root.join("bin");
        if !bin_dir.is_dir() {
            log::debug!("aietools probe: no bin/ directory at {}", root.display());
            return None;
        }

        let probe_tool = |tool: Tool| -> Option<PathBuf> {
            let path = bin_dir.join(tool.binary_name());
            if path.exists() {
                log::debug!("aietools: found {} at {}", tool.binary_name(), path.display());
                Some(path)
            } else {
                log::debug!("aietools: {} not found at {}", tool.binary_name(), path.display());
                None
            }
        };

        let tools = AieTools {
            root: root.to_path_buf(),
            elfanalyzer: probe_tool(Tool::Elfanalyzer),
            xchesscc: probe_tool(Tool::Xchesscc),
            aiesimulator: probe_tool(Tool::Aiesimulator),
            hwanalyze: probe_tool(Tool::Hwanalyze),
        };

        if tools.available() {
            log::info!("Discovered aietools at {}", root.display());
            Some(tools)
        } else {
            log::debug!("aietools at {} has no usable tools", root.display());
            None
        }
    }

    /// Whether at least one tool is usable.
    pub fn available(&self) -> bool {
        self.elfanalyzer.is_some()
            || self.xchesscc.is_some()
            || self.aiesimulator.is_some()
            || self.hwanalyze.is_some()
    }

    /// The root directory of this aietools installation.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Get the path for a specific tool, if available.
    pub fn tool_path(&self, tool: Tool) -> Option<&Path> {
        match tool {
            Tool::Elfanalyzer => self.elfanalyzer.as_deref(),
            Tool::Xchesscc => self.xchesscc.as_deref(),
            Tool::Aiesimulator => self.aiesimulator.as_deref(),
            Tool::Hwanalyze => self.hwanalyze.as_deref(),
        }
    }

    /// Construct a `Command` to invoke the given tool.
    ///
    /// Sets up the environment correctly:
    /// - `XILINX_VITIS_AIETOOLS` points to the installation root
    /// - `LD_LIBRARY_PATH` has aietools lib directories APPENDED (never
    ///   prepended -- aietools ships its own libstdc++ that would break
    ///   system binaries if it shadows the system version)
    ///
    /// Returns `None` if the tool is not available.
    pub fn command(&self, tool: Tool) -> Option<Command> {
        let tool_path = self.tool_path(tool)?;

        let mut cmd = Command::new(tool_path);
        cmd.env("XILINX_VITIS_AIETOOLS", &self.root);

        // Build LD_LIBRARY_PATH: system libs first, aietools libs appended.
        // This matches the pattern in activate-npu-env.sh.
        let aietools_lib = self.root.join("lib");
        let aietools_lnx64 = self.root.join("lnx64/lib");

        let mut ld_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        for lib_dir in [&aietools_lib, &aietools_lnx64] {
            if lib_dir.exists() {
                if !ld_path.is_empty() {
                    ld_path.push(':');
                }
                ld_path.push_str(&lib_dir.to_string_lossy());
            }
        }
        cmd.env("LD_LIBRARY_PATH", &ld_path);

        Some(cmd)
    }

    /// Summary of discovered tools for display.
    pub fn summary(&self) -> String {
        let mut tools = Vec::new();
        if self.elfanalyzer.is_some() {
            tools.push("elfanalyzer");
        }
        if self.xchesscc.is_some() {
            tools.push("xchesscc");
        }
        if self.aiesimulator.is_some() {
            tools.push("aiesimulator");
        }
        if self.hwanalyze.is_some() {
            tools.push("hwanalyze");
        }

        format!("aietools at {} ({})", self.root.display(), tools.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_missing() {
        // Probe of a nonexistent path should return None.
        // (We don't call discover() because it checks env vars that may
        // be set on the developer's machine.)
        assert!(AieTools::probe(Path::new("/nonexistent/aietools")).is_none());
    }

    #[test]
    fn test_tool_binary_names() {
        // Verify binary names are sensible
        assert!(Tool::Elfanalyzer.binary_name().contains("elfanalyzer"));
        assert!(Tool::Xchesscc.binary_name().contains("xchesscc"));
        assert!(Tool::Aiesimulator.binary_name().contains("aiesimulator"));
        assert!(Tool::Hwanalyze.binary_name().contains("hwanalyze"));
    }

    #[test]
    fn test_tool_license_requirements() {
        assert!(!Tool::Elfanalyzer.requires_license());
        assert!(Tool::Xchesscc.requires_license());
        assert!(Tool::Aiesimulator.requires_license());
        assert!(Tool::Hwanalyze.requires_license());
    }

    #[test]
    fn test_empty_tools_not_available() {
        let tools = AieTools {
            root: PathBuf::from("/fake"),
            elfanalyzer: None,
            xchesscc: None,
            aiesimulator: None,
            hwanalyze: None,
        };
        assert!(!tools.available());
    }

    #[test]
    fn test_partial_tools_available() {
        let tools = AieTools {
            root: PathBuf::from("/fake"),
            elfanalyzer: Some(PathBuf::from("/fake/bin/elfanalyzer")),
            xchesscc: None,
            aiesimulator: None,
            hwanalyze: None,
        };
        assert!(tools.available());
    }

    #[test]
    fn test_command_env() {
        // Build a command and verify environment setup (without actually running it)
        let tools = AieTools {
            root: PathBuf::from("/opt/aietools"),
            elfanalyzer: Some(PathBuf::from("/opt/aietools/bin/elfanalyzer")),
            xchesscc: None,
            aiesimulator: None,
            hwanalyze: None,
        };

        let cmd = tools.command(Tool::Elfanalyzer);
        assert!(cmd.is_some(), "should produce a command for available tool");

        let cmd = tools.command(Tool::Xchesscc);
        assert!(cmd.is_none(), "should return None for unavailable tool");
    }

    #[test]
    fn test_summary() {
        let tools = AieTools {
            root: PathBuf::from("/opt/aietools"),
            elfanalyzer: Some(PathBuf::from("/opt/aietools/bin/elfanalyzer")),
            xchesscc: Some(PathBuf::from("/opt/aietools/bin/xchesscc")),
            aiesimulator: None,
            hwanalyze: None,
        };
        let s = tools.summary();
        assert!(s.contains("/opt/aietools"));
        assert!(s.contains("elfanalyzer"));
        assert!(s.contains("xchesscc"));
        assert!(!s.contains("aiesimulator"));
    }
}
