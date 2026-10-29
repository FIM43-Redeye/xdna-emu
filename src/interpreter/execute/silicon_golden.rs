//! Test support: loader for the NPU1 silicon goldens in `tools/golden/silicon/`.
//!
//! Each JSON file is one captured kernel (`kernel`, `class`, `n`, `inputs`,
//! `silicon`). The kernel's configuration (shift, rounding mode, saturation
//! mode) is not stored in the JSON; it is read from the kernel's C++ source
//! under `tests/vector-verify/<kernel>/`, which is what actually ran on silicon.

use std::path::PathBuf;

use xdna_archspec::aie2::rounding::RoundingMode;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// One captured kernel: the lane inputs and the lane outputs observed on silicon.
pub struct SiliconCase {
    pub kernel: String,
    pub inputs: Vec<i64>,
    pub silicon: Vec<i64>,
    source: String,
}

/// Kernel names (file stems) in `tools/golden/silicon/` starting with `prefix`, sorted.
pub fn kernels_with_prefix(prefix: &str) -> Vec<String> {
    let dir = repo_root().join("tools/golden/silicon");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            let stem = name.strip_suffix(".json")?;
            stem.starts_with(prefix).then(|| stem.to_string())
        })
        .collect();
    names.sort();
    names
}

impl SiliconCase {
    /// Load `tools/golden/silicon/<kernel>.json` plus the kernel's C++ source
    /// (`source_file` inside `tests/vector-verify/<kernel>/`).
    pub fn load(kernel: &str, source_file: &str) -> Self {
        let json_path = repo_root().join(format!("tools/golden/silicon/{kernel}.json"));
        let text = std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", json_path.display()));
        let doc: serde_json::Value = serde_json::from_str(&text).expect("golden JSON parses");
        let lanes = |v: &serde_json::Value| -> Vec<i64> {
            v.as_array()
                .expect("lane array")
                .iter()
                .map(|x| x.as_i64().expect("integer lane"))
                .collect()
        };
        let inputs = lanes(&doc["inputs"][0]);
        let silicon = lanes(&doc["silicon"]);
        assert_eq!(inputs.len(), silicon.len(), "{kernel}: lane count mismatch");
        assert_eq!(doc["n"].as_u64().unwrap() as usize, inputs.len(), "{kernel}: n");

        let src_path = repo_root().join(format!("tests/vector-verify/{kernel}/{source_file}"));
        let source =
            std::fs::read_to_string(&src_path).unwrap_or_else(|e| panic!("read {}: {e}", src_path.display()));
        Self { kernel: kernel.to_string(), inputs, silicon, source }
    }

    /// The `crRnd` index the kernel configured via `aie::set_rounding`.
    pub fn rounding_raw(&self) -> u8 {
        let after = self.source.split("rounding_mode").nth(1).expect("set_rounding present");
        // Either `rounding_mode::name` or `static_cast<aie::rounding_mode>(N)`.
        if let Some(num) = after.strip_prefix(">(") {
            let digits: String = num.chars().take_while(char::is_ascii_digit).collect();
            return digits.parse().expect("numeric rounding mode");
        }
        let name: String = after
            .trim_start_matches("::")
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        match name.as_str() {
            "floor" => 0,
            "ceil" => 1,
            "symmetric_floor" => 2,
            "symmetric_ceil" => 3,
            "negative_inf" => 8,
            "positive_inf" => 9,
            "symmetric_zero" => 10,
            "symmetric_inf" => 11,
            "conv_even" => 12,
            "conv_odd" => 13,
            other => panic!("{}: unknown rounding mode name {other}", self.kernel),
        }
    }

    pub fn rounding(&self) -> RoundingMode {
        RoundingMode::from_raw(self.rounding_raw()).expect("4-bit crRnd")
    }

    /// `(saturate, symmetric)` from `aie::set_saturation` (none / saturate / symmetric).
    pub fn saturation(&self) -> (bool, bool) {
        let after = self.source.split("saturation_mode::").nth(1).expect("set_saturation present");
        let name: String = after.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        match name.as_str() {
            "none" => (false, false),
            "saturate" => (true, false),
            "symmetric" => (true, true),
            other => panic!("{}: unknown saturation mode {other}", self.kernel),
        }
    }

    /// The `#define <name> N` constant from the kernel source.
    pub fn define(&self, name: &str) -> u32 {
        let pat = format!("#define {name} ");
        let rest = self
            .source
            .split(&pat)
            .nth(1)
            .unwrap_or_else(|| panic!("{} lacks {name}", self.kernel));
        rest.chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .unwrap()
    }
}
