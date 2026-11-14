//! The S2MM stream sidecar: every word the shim S2MMs took off their streams, per host buffer.
//!
//! A shim S2MM writes host memory in whole bursts, so a trace channel whose BD never completes
//! leaves its last partial burst unwritten, on silicon and in the emulator alike: `trace_raw.bin`
//! (the host's trace buffer) is the silicon-faithful capture. The trace gate scores the whole
//! stream instead, so that it stays exact: when `XDNA_TRACE_DIR` is set, the FFI turns on the
//! host memory's S2MM stream shadow and, after every run, writes `trace_stream.json` there (next
//! to the `trace_raw.bin` the host program writes). `tools/trace_stream.py` turns it into
//! `trace_stream.bin`, the buffer as it would be had every word been written.
//!
//! Format: `{"schema": 1, "buffers": [{"base", "size", "runs": [{"offset", "bytes"}]}]}`, one
//! entry per buffer that received any word, runs of contiguous words in address order, `bytes`
//! the run's memory image in hex (little-endian words, as in the buffer).

use std::path::{Path, PathBuf};

use xdna_emu_core::device::host_memory::HostMemory;

/// Environment variable naming the directory of the host program's trace capture.
pub const TRACE_DIR_ENV: &str = "XDNA_TRACE_DIR";

/// File name of the sidecar in that directory.
pub const SIDECAR_FILE: &str = "trace_stream.json";

/// The sidecar's path when a trace directory is set.
pub fn sidecar_path() -> Option<PathBuf> {
    std::env::var_os(TRACE_DIR_ENV).map(|dir| Path::new(&dir).join(SIDECAR_FILE))
}

/// The sidecar document for the shadow `memory` holds.
pub fn sidecar_json(memory: &HostMemory) -> serde_json::Value {
    let mut buffers = Vec::new();
    for (base, size) in memory.buffer_spans() {
        let words = memory.s2mm_stream_shadow_in(base, size);
        if words.is_empty() {
            continue;
        }
        let mut runs: Vec<(u64, Vec<u8>)> = Vec::new();
        for (addr, word) in words {
            let offset = addr - base;
            match runs.last_mut() {
                Some((start, bytes)) if *start + bytes.len() as u64 == offset => {
                    bytes.extend_from_slice(&word.to_le_bytes())
                }
                _ => runs.push((offset, word.to_le_bytes().to_vec())),
            }
        }
        let runs: Vec<serde_json::Value> = runs
            .into_iter()
            .map(|(offset, bytes)| {
                let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                serde_json::json!({ "offset": offset, "bytes": hex })
            })
            .collect();
        buffers.push(serde_json::json!({ "base": base, "size": size, "runs": runs }));
    }
    serde_json::json!({ "schema": 1, "buffers": buffers })
}

/// Write the sidecar when a trace directory is set. Best-effort: a write failure is logged and
/// never alters the run.
pub fn write_sidecar(memory: &HostMemory) {
    let Some(path) = sidecar_path() else {
        return;
    };
    let text = sidecar_json(memory).to_string();
    if let Err(e) = std::fs::write(&path, text) {
        log::error!("{TRACE_DIR_ENV}: failed to write {}: {e}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sidecar_lists_each_buffers_contiguous_runs() {
        let mut mem = HostMemory::new();
        mem.allocate_region("trace", 0x10_0000, 0x1000).unwrap();
        mem.allocate_region("out", 0x20_0000, 0x100).unwrap();
        mem.set_s2mm_stream_shadow(true);
        mem.shadow_s2mm_word(0x10_0000, 0x0403_0201);
        mem.shadow_s2mm_word(0x10_0004, 0x0807_0605);
        mem.shadow_s2mm_word(0x10_0040, 0xDDCC_BBAA);
        let doc = sidecar_json(&mem);
        assert_eq!(
            doc,
            serde_json::json!({"schema": 1, "buffers": [{"base": 0x10_0000, "size": 0x1000, "runs": [
                {"offset": 0, "bytes": "0102030405060708"},
                {"offset": 0x40, "bytes": "aabbccdd"},
            ]}]}),
            "a buffer with no word is left out"
        );
    }

    #[test]
    fn with_the_shadow_off_the_sidecar_is_empty() {
        let mut mem = HostMemory::new();
        mem.allocate_region("trace", 0x10_0000, 0x1000).unwrap();
        mem.shadow_s2mm_word(0x10_0000, 1);
        assert_eq!(sidecar_json(&mem), serde_json::json!({"schema": 1, "buffers": []}));
    }
}
