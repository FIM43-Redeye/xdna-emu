//! The device-state file: the core state silicon carries across launches, kept across
//! emulator processes.
//!
//! Within one emulator instance the cores keep their carried state from launch to launch
//! always ([`CarriedCoreState`], `CoreState::reset`). Separate processes are separate
//! instances, so by default each starts from a fresh device. Setting [`STATE_DIR_ENV`] makes
//! the emulated device behave like silicon across processes too: the FFI loads the file when
//! the device opens and saves it when the device closes.
//!
//! The file is `<dir>/<arch words>-<cols>x<rows>.json` (NPU1: `aie2-npu1-phoenix-hawkpoint-5x6.json`), a [`DeviceStateFile`]. It is refused (and
//! the device starts fresh, with a warning) when its format version or device key differs
//! from this build's; an absent file is a fresh device. To reset the device, delete the file
//! or point the variable at an empty directory.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::InterpreterEngine;
use crate::interpreter::state::carried::CarriedCoreState;

/// Directory of the device-state file. Unset or empty = no cross-process state.
pub const STATE_DIR_ENV: &str = "XDNA_EMU_STATE_DIR";

/// Version of [`DeviceStateFile`]. Bump it whenever the schema or the meaning of a field
/// changes; an older file is then refused instead of misread.
pub const DEVICE_STATE_FORMAT: u32 = 1;

/// Which device a file belongs to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceKey {
    /// Architecture name (`AIE2`, `AIE2P`).
    pub arch: String,
    pub cols: usize,
    pub rows: usize,
}

impl DeviceKey {
    pub fn of(engine: &InterpreterEngine) -> Self {
        let d = engine.device();
        Self { arch: d.arch_name().to_string(), cols: d.cols(), rows: d.rows() }
    }

    /// The file name for this device.
    pub fn file_name(&self) -> String {
        // The architecture name is prose ("AIE2 (NPU1 - Phoenix/HawkPoint)"): keep its words.
        let mut slug = String::new();
        for w in self.arch.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()) {
            slug.push_str(&w.to_ascii_lowercase());
            slug.push('-');
        }
        format!("{slug}{}x{}.json", self.cols, self.rows)
    }
}

/// One compute core's saved state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedCore {
    pub col: u8,
    pub row: u8,
    pub state: CarriedCoreState,
}

/// The whole file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceStateFile {
    pub format: u32,
    pub device: DeviceKey,
    pub cores: Vec<SavedCore>,
}

/// Why a file was not loaded.
#[derive(Debug)]
pub enum LoadError {
    /// No file: a fresh device (not an error to report).
    Absent,
    Unreadable(String),
    Malformed(String),
    Format {
        found: u64,
    },
    Device {
        found: serde_json::Value,
    },
    /// The file names a core this device does not have.
    Cores(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Absent => write!(f, "no device-state file"),
            LoadError::Unreadable(e) => write!(f, "unreadable: {e}"),
            LoadError::Malformed(e) => write!(f, "malformed: {e}"),
            LoadError::Format { found } => {
                write!(f, "format {found}, this emulator reads format {DEVICE_STATE_FORMAT}")
            }
            LoadError::Device { found } => write!(f, "saved by another device ({found})"),
            LoadError::Cores(e) => write!(f, "{e}"),
        }
    }
}

/// The directory [`STATE_DIR_ENV`] names, if set.
pub fn state_dir_from_env() -> Option<PathBuf> {
    std::env::var_os(STATE_DIR_ENV).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// The file `engine`'s device uses in `dir`.
pub fn state_file(dir: &Path, engine: &InterpreterEngine) -> PathBuf {
    dir.join(DeviceKey::of(engine).file_name())
}

impl DeviceStateFile {
    /// The carried state of every compute core of `engine` (in-flight writes land first).
    pub fn capture(engine: &mut InterpreterEngine) -> Self {
        let device = DeviceKey::of(engine);
        let cores = engine
            .carried_core_states()
            .into_iter()
            .map(|((col, row), state)| SavedCore { col, row, state })
            .collect();
        Self { format: DEVICE_STATE_FORMAT, device, cores }
    }

    /// Parse `text`, checking the format version and then the device key before the body, so
    /// a file of another version is reported as such rather than as a schema error.
    pub fn parse(text: &str, expect: &DeviceKey) -> Result<Self, LoadError> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| LoadError::Malformed(e.to_string()))?;
        let found = v
            .get("format")
            .and_then(|f| f.as_u64())
            .ok_or_else(|| LoadError::Malformed("no format".into()))?;
        if found != DEVICE_STATE_FORMAT as u64 {
            return Err(LoadError::Format { found });
        }
        let device = v.get("device").cloned().unwrap_or(serde_json::Value::Null);
        if serde_json::from_value::<DeviceKey>(device.clone()).ok().as_ref() != Some(expect) {
            return Err(LoadError::Device { found: device });
        }
        serde_json::from_value(v).map_err(|e| LoadError::Malformed(e.to_string()))
    }

    /// Put the saved state into `engine`'s cores.
    pub fn apply(&self, engine: &mut InterpreterEngine) -> Result<(), LoadError> {
        let states: Vec<_> = self.cores.iter().map(|c| ((c.col, c.row), c.state.clone())).collect();
        engine.restore_carried_core_states(&states).map_err(LoadError::Cores)
    }
}

/// Load `dir`'s file for `engine`'s device into its cores. On any error the engine is left
/// as it was (a fresh device).
pub fn load(dir: &Path, engine: &mut InterpreterEngine) -> Result<PathBuf, LoadError> {
    let path = state_file(dir, engine);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(LoadError::Absent),
        Err(e) => return Err(LoadError::Unreadable(format!("{}: {e}", path.display()))),
    };
    DeviceStateFile::parse(&text, &DeviceKey::of(engine))?.apply(engine)?;
    Ok(path)
}

/// Save `engine`'s carried state to `dir` (created if missing), atomically: a reader sees the
/// old file or the new one, never half of one.
pub fn save(dir: &Path, engine: &mut InterpreterEngine) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = state_file(dir, engine);
    let text = serde_json::to_string(&DeviceStateFile::capture(engine)).map_err(std::io::Error::other)?;
    let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xdna-emu-devstate-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn marked_engine(v: u32) -> InterpreterEngine {
        let mut e = InterpreterEngine::new_npu1();
        // One launch first: the power-on sentinel fills whole words, a register write keeps
        // only the register's width, so a never-launched file would not compare equal to
        // itself loaded into a context.
        e.reset();
        e.core_context_mut(2, 4).unwrap().scalar.write(9, v);
        e.core_context_mut(2, 4).unwrap().vector.write(1, [v; 8]);
        e
    }

    #[test]
    fn round_trip_through_a_file() {
        let dir = scratch("rt");
        let mut a = marked_engine(0xfeed_beef);
        let path = save(&dir, &mut a).unwrap();
        assert_eq!(path.file_name().unwrap(), "aie2-npu1-phoenix-hawkpoint-5x6.json");
        let mut b = InterpreterEngine::new_npu1();
        assert_ne!(b.core_context(2, 4).unwrap().scalar.read(9), 0xfeed_beef);
        load(&dir, &mut b).unwrap();
        assert_eq!(b.core_context(2, 4).unwrap().scalar.read(9), 0xfeed_beef);
        assert_eq!(b.core_context(2, 4).unwrap().vector.read(1), [0xfeed_beef; 8]);
        assert_eq!(DeviceStateFile::capture(&mut a), DeviceStateFile::capture(&mut b));
        // every compute core is saved, and only those
        let f = DeviceStateFile::capture(&mut a);
        assert!(f.cores.iter().all(|c| c.row >= 2));
        assert_eq!(f.cores.len(), b.device().array.tiles.iter().filter(|t| t.is_compute()).count());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_absent_file_is_a_fresh_device() {
        let dir = scratch("absent");
        let mut e = InterpreterEngine::new_npu1();
        let before = DeviceStateFile::capture(&mut e);
        assert!(matches!(load(&dir, &mut e), Err(LoadError::Absent)));
        assert_eq!(DeviceStateFile::capture(&mut e), before);
    }

    fn write_variant(dir: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
        let mut a = marked_engine(1);
        let path = save(dir, &mut a).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        edit(&mut v);
        std::fs::write(&path, v.to_string()).unwrap();
    }

    #[test]
    fn another_format_version_is_refused_and_changes_nothing() {
        let dir = scratch("ver");
        write_variant(&dir, |v| v["format"] = serde_json::json!(DEVICE_STATE_FORMAT + 1));
        let mut e = InterpreterEngine::new_npu1();
        let before = DeviceStateFile::capture(&mut e);
        assert!(matches!(load(&dir, &mut e), Err(LoadError::Format { .. })));
        assert_eq!(DeviceStateFile::capture(&mut e), before);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn another_device_is_refused_and_changes_nothing() {
        let dir = scratch("dev");
        write_variant(&dir, |v| v["device"]["cols"] = serde_json::json!(8));
        let mut e = InterpreterEngine::new_npu1();
        let before = DeviceStateFile::capture(&mut e);
        assert!(matches!(load(&dir, &mut e), Err(LoadError::Device { .. })));
        assert_eq!(DeviceStateFile::capture(&mut e), before);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_naming_a_missing_core_is_refused_and_changes_nothing() {
        let dir = scratch("cores");
        write_variant(&dir, |v| v["cores"][0]["row"] = serde_json::json!(1)); // a memory tile
        let mut e = InterpreterEngine::new_npu1();
        let before = DeviceStateFile::capture(&mut e);
        assert!(matches!(load(&dir, &mut e), Err(LoadError::Cores(_))));
        assert_eq!(DeviceStateFile::capture(&mut e), before);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn garbage_is_refused() {
        let key = DeviceKey::of(&InterpreterEngine::new_npu1());
        assert!(matches!(DeviceStateFile::parse("not json", &key), Err(LoadError::Malformed(_))));
        assert!(matches!(DeviceStateFile::parse("{}", &key), Err(LoadError::Malformed(_))));
    }
}
