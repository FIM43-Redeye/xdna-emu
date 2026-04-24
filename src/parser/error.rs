//! Structured parser diagnostics.
//!
//! ParseError carries byte offset, expected vs got, and a hex context
//! window around the offending byte for every fallible boundary in
//! the parser. The Display impl renders this as a human-friendly
//! message suitable for CLI output; the Debug impl gives full
//! machine-readable context.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("unexpected magic at offset 0x{offset:x}: expected {expected}, got {got}")]
    BadMagic {
        offset: usize,
        expected: String,
        got: String,
        hex_context: Vec<u8>,
    },

    #[error("truncated at offset 0x{offset:x}: expected {expected_bytes} more bytes, have {available} ({context})")]
    Truncated {
        offset: usize,
        expected_bytes: usize,
        available: usize,
        context: &'static str,
    },

    #[error("unknown {kind} {value:#x} at offset 0x{offset:x} ({context})")]
    Unknown {
        offset: usize,
        kind: &'static str,
        value: u64,
        context: &'static str,
    },

    #[error("invalid {field} value {value:#x} at offset 0x{offset:x} ({reason})")]
    InvalidValue {
        offset: usize,
        field: &'static str,
        value: u64,
        reason: &'static str,
    },

    #[error("structural error parsing {context} at offset 0x{offset:x}: {source}")]
    Structural {
        offset: usize,
        context: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    #[error("error parsing {context} at offset 0x{offset:x}: {message}")]
    External {
        offset: usize,
        context: &'static str,
        message: String,
    },
}

impl ParseError {
    /// Format a hex-dump window (+/- `window` bytes) around `offset`,
    /// returning the raw bytes suitable for attaching to error
    /// messages. Saturates at data boundaries.
    pub fn hex_window(data: &[u8], offset: usize, window: usize) -> Vec<u8> {
        let start = offset.saturating_sub(window);
        let end = (offset + window).min(data.len());
        if start >= end {
            return Vec::new();
        }
        data[start..end].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_magic_renders_offset_in_hex() {
        let err = ParseError::BadMagic {
            offset: 0x1234,
            expected: "xclbin2\\0".to_string(),
            got: "deadbeef".to_string(),
            hex_context: vec![0xde, 0xad, 0xbe, 0xef],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0x1234"), "missing hex offset: {msg}");
        assert!(msg.contains("xclbin2"), "missing expected: {msg}");
        assert!(msg.contains("deadbeef"), "missing got: {msg}");
    }

    #[test]
    fn truncated_message_shows_shortfall() {
        let err = ParseError::Truncated {
            offset: 0x100,
            expected_bytes: 64,
            available: 12,
            context: "AIE Partition header",
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0x100"));
        assert!(msg.contains("64"));
        assert!(msg.contains("12"));
        assert!(msg.contains("AIE Partition"));
    }

    #[test]
    fn hex_window_extracts_bytes_around_offset() {
        let data: Vec<u8> = (0..=255).collect();
        let window = ParseError::hex_window(&data, 128, 4);
        // window of 4 around offset 128 -> [124..132]
        assert_eq!(window, vec![124, 125, 126, 127, 128, 129, 130, 131]);
    }

    #[test]
    fn external_renders_lower_level_message() {
        let err = ParseError::External {
            offset: 0x40,
            context: "XCLBIN header",
            message: "zerocopy: size mismatch".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0x40"), "missing hex offset: {msg}");
        assert!(msg.contains("XCLBIN header"), "missing context: {msg}");
        assert!(msg.contains("zerocopy"), "missing lower-level message: {msg}");
    }
}
