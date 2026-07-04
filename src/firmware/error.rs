//! Error type for the firmware loader/interpreter.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FirmwareError {
    #[error("bad firmware magic at offset {offset:#x}: found {found:02x?}")]
    BadMagic { offset: usize, found: [u8; 4] },

    #[error("firmware truncated at offset {offset:#x}: need {needed} bytes, have {got}")]
    Truncated {
        offset: usize,
        needed: usize,
        got: usize,
    },

    #[error("firmware size mismatch: header says {header:#x}, file is {file:#x}")]
    SizeMismatch { header: u32, file: usize },
}
