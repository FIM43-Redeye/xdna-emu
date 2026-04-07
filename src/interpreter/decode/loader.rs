//! Decoder construction and loading.
//!
//! Handles creating `InstructionDecoder` instances from various sources:
//! build-time generated constants, TableGen files, or raw encoding tables.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use crate::tablegen::{DecoderIndex, InstrEncoding, decoder_bytecode, decoder_ffi};

use super::decoder::InstructionDecoder;

/// Global cached decoder, loaded once on first use.
/// This avoids repeatedly parsing TableGen files for each core.
static CACHED_DECODER: OnceLock<InstructionDecoder> = OnceLock::new();

impl InstructionDecoder {
    /// Create an empty decoder (no encodings loaded).
    pub fn new() -> Self {
        Self {
            index: DecoderIndex::default(),
            format_table: None,
            instr_info: Vec::new(),
            decode_count: 0,
            unknown_count: 0,
        }
    }

    /// Get a cached decoder, loading it on first call.
    ///
    /// This is the preferred way to get a decoder - it loads once and reuses
    /// the cached instance for all subsequent calls. Each caller gets a clone
    /// with independent statistics.
    pub fn load_cached() -> Self {
        CACHED_DECODER.get_or_init(|| {
            log::info!("Initializing cached instruction decoder");
            Self::load_fresh()
        }).clone()
    }

    /// Load a fresh decoder (not cached).
    ///
    /// Use `load_cached()` instead unless you specifically need a fresh load.
    ///
    /// # Panics
    ///
    /// Panics if the TableGen parser fails. This is intentional - we want to
    /// fail fast rather than silently falling back to broken behavior.
    fn load_fresh() -> Self {
        Self::load_from_generated()
    }

    /// Load a decoder from build-time generated constants.
    ///
    /// All instruction encodings, decoder bytecode, and metadata were extracted
    /// from llvm-aie at compile time. No filesystem access required at runtime.
    fn load_from_generated() -> Self {
        let output = crate::tablegen::load_from_generated();

        // Build data-driven format table from composite format Inst fields
        let format_table = if output.composite_formats.iter().any(|f| !f.slot_maps.is_empty()) {
            let table = crate::interpreter::bundle::FormatTable::build(&output.composite_formats);
            log::info!(
                "Built data-driven format table: {} entries",
                table.total_entries(),
            );
            Some(table)
        } else {
            log::warn!("No Inst-derived format layouts; falling back to hand-coded extraction");
            None
        };

        let mut decoder = Self::from_tables_with_decoders(
            output.encodings_by_slot,
            output.decoder_tables,
        );
        decoder.format_table = format_table;

        // Populate per-opcode metadata from LLVM's MCInstrDesc + itinerary model.
        decoder.instr_info = decoder_ffi::query_all_instr_info();
        log::info!(
            "Loaded LLVM InstrInfo: {} opcodes, {} with latency, {} with flags",
            decoder.instr_info.len(),
            decoder.instr_info.iter().filter(|i| i.latency.is_some()).count(),
            decoder.instr_info.iter().filter(|i| i.flags != 0).count(),
        );

        decoder
    }

    /// Load a decoder from llvm-aie.
    ///
    /// Uses config file or environment variable to find llvm-aie path.
    /// Uses `llvm-tblgen` for accurate encodings.
    ///
    /// NOTE: Prefer `load_cached()` which avoids repeatedly parsing TableGen.
    ///
    /// # Panics
    ///
    /// Panics if llvm-aie is not found or TableGen parsing fails.
    pub fn load_default() -> Self {
        Self::load_cached()
    }

    /// Load a decoder using build-time generated data.
    ///
    /// The `llvm_aie_path` parameter is ignored -- all data is compiled in.
    /// This signature is kept for backward compatibility with existing tests.
    pub fn try_load_via_tblgen(_llvm_aie_path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        Ok(Self::load_from_generated())
    }

    /// Check if llvm-aie is available (checks config and env var).
    pub fn is_llvm_aie_available() -> bool {
        use crate::config::Config;
        Path::new(&Config::get().llvm_aie_path()).exists()
    }

    /// Create a decoder from encoding tables grouped by slot (no LLVM bytecode).
    ///
    /// Without bytecode tables, all slot decodes return `None` (unknown).
    /// Use `from_tables_with_decoders()` for the full decode path.
    pub fn from_tables(tables: HashMap<String, Vec<InstrEncoding>>) -> Self {
        Self::from_tables_with_decoders(tables, HashMap::new())
    }

    /// Create a decoder from encoding tables with LLVM decoder bytecode tables.
    ///
    /// This is the primary constructor. LLVM bytecode tables are the sole
    /// disambiguation mechanism, matching LLVM's own disassembler behavior.
    pub fn from_tables_with_decoders(
        tables: HashMap<String, Vec<InstrEncoding>>,
        decoder_tables: HashMap<String, decoder_bytecode::DecoderTable>,
    ) -> Self {
        let index = DecoderIndex::from_slot_encodings(tables, decoder_tables);

        Self {
            index,
            format_table: None,
            instr_info: Vec::new(),
            decode_count: 0,
            unknown_count: 0,
        }
    }

    /// Create a decoder from a pre-built DecoderIndex.
    pub fn from_index(index: DecoderIndex) -> Self {
        Self {
            index,
            format_table: None,
            instr_info: Vec::new(),
            decode_count: 0,
            unknown_count: 0,
        }
    }
}
