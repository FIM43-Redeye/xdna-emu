//! Cross-reference decoder output against llvm-objdump.
//!
//! Validates our TableGen-driven decoder by comparing its output against
//! llvm-objdump (Peano) for every instruction in an ELF binary. This finds
//! decoding discrepancies systematically rather than one-at-a-time via fuzz
//! seed analysis.
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::interpreter::decode::crossref::cross_reference_elf;
//! use xdna_emu::interpreter::decode::InstructionDecoder;
//! use std::path::Path;
//!
//! let decoder = InstructionDecoder::load_default();
//! let report = cross_reference_elf(
//!     Path::new("kernel.elf"),
//!     Path::new("../llvm-aie/build/bin/llvm-objdump"),
//!     &decoder,
//! ).unwrap();
//! eprintln!("{}", report);
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::process::Command;

use super::InstructionDecoder;
use crate::interpreter::bundle::Operation;
use crate::interpreter::traits::Decoder;
use crate::parser::elf::AieElf;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single instruction (or VLIW bundle) from llvm-objdump output.
#[derive(Debug, Clone)]
pub struct ObjdumpInstr {
    /// Program counter address.
    pub pc: u32,
    /// Raw instruction bytes.
    pub bytes: Vec<u8>,
    /// Per sub-instruction within the bundle: (mnemonic, operand_string).
    pub sub_instrs: Vec<(String, String)>,
}

/// Result of comparing one instruction between our decoder and llvm-objdump.
#[derive(Debug, Clone)]
pub enum InstrComparison {
    /// Both decoded, mnemonics match for all sub-instructions.
    Match { pc: u32 },
    /// Both decoded, but mnemonic(s) differ.
    MnemonicMismatch {
        pc: u32,
        objdump: Vec<String>,
        emu: Vec<String>,
    },
    /// Our decoder failed but objdump decoded it.
    DecodeFailed {
        pc: u32,
        objdump: Vec<String>,
        error: String,
    },
    /// Size disagreement (we advance by a different number of bytes).
    SizeMismatch {
        pc: u32,
        objdump_size: usize,
        emu_size: usize,
    },
    /// objdump had no entry at this PC (our decoder found something extra).
    ExtraInstruction { pc: u32, emu: Vec<String> },
}

/// Full cross-reference report for one ELF.
#[derive(Debug)]
pub struct CrossRefReport {
    pub elf_path: String,
    pub total_instructions: usize,
    pub matches: usize,
    pub mnemonic_mismatches: Vec<InstrComparison>,
    pub decode_failures: Vec<InstrComparison>,
    pub size_mismatches: Vec<InstrComparison>,
    pub extra_instructions: Vec<InstrComparison>,
}

impl CrossRefReport {
    /// Match rate as a percentage.
    pub fn match_rate(&self) -> f64 {
        if self.total_instructions == 0 {
            return 100.0;
        }
        (self.matches as f64 / self.total_instructions as f64) * 100.0
    }

    /// Total number of discrepancies.
    pub fn discrepancy_count(&self) -> usize {
        self.mnemonic_mismatches.len()
            + self.decode_failures.len()
            + self.size_mismatches.len()
    }
}

impl fmt::Display for CrossRefReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Cross-Reference: {} ===", self.elf_path)?;
        writeln!(f)?;
        writeln!(
            f,
            "Total: {} instructions, Match: {} ({:.1}%)",
            self.total_instructions,
            self.matches,
            self.match_rate()
        )?;
        if !self.mnemonic_mismatches.is_empty() {
            writeln!(
                f,
                "Mnemonic mismatch: {}",
                self.mnemonic_mismatches.len()
            )?;
        }
        if !self.decode_failures.is_empty() {
            writeln!(f, "Decode failure: {}", self.decode_failures.len())?;
        }
        if !self.size_mismatches.is_empty() {
            writeln!(f, "Size mismatch: {}", self.size_mismatches.len())?;
        }

        // Detail sections
        if !self.mnemonic_mismatches.is_empty() {
            writeln!(f)?;
            writeln!(f, "MNEMONIC MISMATCHES:")?;
            for comp in &self.mnemonic_mismatches {
                if let InstrComparison::MnemonicMismatch { pc, objdump, emu } = comp {
                    writeln!(
                        f,
                        "  PC 0x{:04X}: objdump=[{}] emu=[{}]",
                        pc,
                        objdump.join(", "),
                        emu.join(", ")
                    )?;
                }
            }
        }

        if !self.decode_failures.is_empty() {
            writeln!(f)?;
            writeln!(f, "DECODE FAILURES:")?;
            for comp in &self.decode_failures {
                if let InstrComparison::DecodeFailed { pc, objdump, error } = comp {
                    writeln!(
                        f,
                        "  PC 0x{:04X}: objdump=[{}] error={}",
                        pc,
                        objdump.join(", "),
                        error
                    )?;
                }
            }
        }

        if !self.size_mismatches.is_empty() {
            writeln!(f)?;
            writeln!(f, "SIZE MISMATCHES:")?;
            for comp in &self.size_mismatches {
                if let InstrComparison::SizeMismatch {
                    pc,
                    objdump_size,
                    emu_size,
                } = comp
                {
                    writeln!(
                        f,
                        "  PC 0x{:04X}: objdump={} bytes, emu={} bytes",
                        pc, objdump_size, emu_size
                    )?;
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// llvm-objdump output parser
// ---------------------------------------------------------------------------

/// Run llvm-objdump and parse its output into a PC-indexed map.
pub fn parse_objdump(
    elf_path: &Path,
    objdump_path: &Path,
) -> Result<BTreeMap<u32, ObjdumpInstr>, String> {
    let output = Command::new(objdump_path)
        .arg("-d")
        .arg(elf_path)
        .output()
        .map_err(|e| format!("Failed to run llvm-objdump: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("llvm-objdump failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_objdump_text(&stdout)
}

/// Parse the text output of llvm-objdump -d.
///
/// Each disassembly line has the format:
/// ```text
///   <addr>: <hex bytes>    <mnemonic>  <operands>
/// ```
/// VLIW bundles have multiple sub-instructions separated by `;\t\t`:
/// ```text
///   56: bb 10 00 ... 	mova	r16, #0x1;		movxm	p1, #0x70400
/// ```
fn parse_objdump_text(text: &str) -> Result<BTreeMap<u32, ObjdumpInstr>, String> {
    let mut result = BTreeMap::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Skip empty lines, section headers, symbol labels
        if trimmed.is_empty()
            || trimmed.ends_with(':')
            || trimmed.starts_with("Disassembly")
            || !trimmed.contains(':')
        {
            continue;
        }

        // Match lines like: "  0: 15 01 00 d8 00 00    	jl	#0x1b0"
        // The address ends at ':', then hex bytes, then a tab, then asm.
        let colon_pos = match trimmed.find(':') {
            Some(p) => p,
            None => continue,
        };

        // Parse address
        let addr_str = &trimmed[..colon_pos].trim();
        let pc = match u32::from_str_radix(addr_str, 16) {
            Ok(a) => a,
            Err(_) => continue, // Not an instruction line (e.g., section label)
        };

        // After the colon: " 15 01 00 d8 00 00    \tjl\t#0x1b0"
        let rest = &trimmed[colon_pos + 1..];

        // Split on the first tab character to separate bytes from asm.
        // The bytes section has hex pairs separated by spaces.
        // The asm section follows after one or more tabs.
        let tab_pos = match rest.find('\t') {
            Some(p) => p,
            None => continue, // No asm text (shouldn't happen)
        };

        let bytes_str = &rest[..tab_pos].trim();
        let asm_str = &rest[tab_pos..].trim();

        // Parse bytes
        let bytes: Vec<u8> = bytes_str
            .split_whitespace()
            .filter_map(|b| u8::from_str_radix(b, 16).ok())
            .collect();

        if bytes.is_empty() {
            continue;
        }

        // Parse sub-instructions (VLIW bundles use ";\t\t" or ";  " separators)
        let sub_instrs = parse_sub_instructions(asm_str);

        result.insert(
            pc,
            ObjdumpInstr {
                pc,
                bytes,
                sub_instrs,
            },
        );
    }

    Ok(result)
}

/// Parse the assembly text portion of an objdump line into sub-instructions.
///
/// For standalone instructions: `"jl\t#0x1b0"` -> `[("jl", "#0x1b0")]`
/// For VLIW bundles: `"mova\tr16, #0x1;\t\tmovxm\tp1, #0x70400"` ->
///   `[("mova", "r16, #0x1"), ("movxm", "p1, #0x70400")]`
fn parse_sub_instructions(asm: &str) -> Vec<(String, String)> {
    // Split on semicolons that separate VLIW sub-instructions.
    // The pattern is typically ";\t\t" but can vary.
    let parts: Vec<&str> = asm.split(';').collect();

    let mut result = Vec::new();
    for part in parts {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Split mnemonic from operands. The mnemonic is the first
        // whitespace-delimited token. Operands are everything after.
        let (mnemonic, operands) = match trimmed.split_once(|c: char| c == '\t' || c == ' ') {
            Some((m, ops)) => (m.trim().to_lowercase(), ops.trim().to_string()),
            None => (trimmed.to_lowercase(), String::new()),
        };

        if !mnemonic.is_empty() {
            result.push((mnemonic, operands));
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Mnemonic normalization
// ---------------------------------------------------------------------------

/// Normalize a mnemonic for comparison.
///
/// Handles known divergences between our decoder and llvm-objdump:
/// - Dot-suffixed variants: "st.s8" -> "st" (we strip the suffix)
/// - NOP variants: all nop* map to "nop"
/// - Case normalization
fn normalize_mnemonic(mnemonic: &str) -> String {
    let lower = mnemonic.to_lowercase();

    // All NOP variants are equivalent
    if lower.starts_with("nop") {
        return "nop".to_string();
    }

    // For dot-suffixed variants (st.s8, st.s16, etc.), keep the full
    // mnemonic but also provide the base for comparison. We compare
    // the base (before first dot) as a fallback if the full doesn't match.
    lower
}

/// Check if two mnemonics match, accounting for known normalization rules.
fn mnemonics_match(ours: &str, reference: &str) -> bool {
    let norm_ours = normalize_mnemonic(ours);
    let norm_ref = normalize_mnemonic(reference);

    if norm_ours == norm_ref {
        return true;
    }

    // Fallback: compare base mnemonic (before first dot)
    let base_ours = norm_ours.split('.').next().unwrap_or(&norm_ours);
    let base_ref = norm_ref.split('.').next().unwrap_or(&norm_ref);

    base_ours == base_ref
}

// ---------------------------------------------------------------------------
// Cross-reference engine
// ---------------------------------------------------------------------------

/// Cross-reference our decoder against llvm-objdump for an ELF binary.
///
/// Decodes every instruction in the ELF's .text section with both tools
/// and reports discrepancies.
pub fn cross_reference_elf(
    elf_path: &Path,
    objdump_path: &Path,
    decoder: &InstructionDecoder,
) -> Result<CrossRefReport, String> {
    // Step 1: Get reference disassembly from llvm-objdump
    let objdump_map = parse_objdump(elf_path, objdump_path)?;

    // Step 2: Load ELF and get .text bytes
    let elf_data = std::fs::read(elf_path)
        .map_err(|e| format!("Failed to read ELF: {}", e))?;
    let elf = AieElf::parse(&elf_data)
        .map_err(|e| format!("Failed to parse ELF: {}", e))?;
    let text = elf
        .text_section()
        .ok_or_else(|| "No .text section in ELF".to_string())?;

    // Step 3: Walk through .text, decode with our decoder, compare
    let mut report = CrossRefReport {
        elf_path: elf_path.display().to_string(),
        total_instructions: 0,
        matches: 0,
        mnemonic_mismatches: Vec::new(),
        decode_failures: Vec::new(),
        size_mismatches: Vec::new(),
        extra_instructions: Vec::new(),
    };

    let mut pc = 0u32;
    while (pc as usize) < text.len() {
        let bytes = &text[pc as usize..];
        if bytes.len() < 2 {
            break;
        }

        report.total_instructions += 1;

        // Get reference from objdump
        let reference = objdump_map.get(&pc);

        // Decode with our decoder
        match decoder.decode(bytes, pc) {
            Ok(bundle) => {
                let emu_size = bundle.size() as usize;

                // Extract non-NOP mnemonics from our decoder.
                // Filter out NOPs consistently to match the reference filtering.
                let is_nop_slot = |s: &&crate::interpreter::bundle::SlotOp| -> bool {
                    if matches!(s.op, Operation::Nop) {
                        return true;
                    }
                    if let Some(ref name) = s.encoding_name {
                        let lower = name.to_lowercase();
                        if lower.starts_with("nop") {
                            return true;
                        }
                    }
                    false
                };

                // Operation-derived mnemonics (fallback)
                let emu_mnemonics: Vec<String> = bundle
                    .active_slots()
                    .filter(|s| !is_nop_slot(s))
                    .map(|s| {
                        format!("{:?}", s.op)
                            .split('(')
                            .next()
                            .unwrap_or("unknown")
                            .to_lowercase()
                    })
                    .collect();

                // Raw encoding mnemonics (preferred)
                let emu_raw_mnemonics: Vec<String> = bundle
                    .active_slots()
                    .filter(|s| !is_nop_slot(s))
                    .filter_map(|s| s.encoding_name.as_deref())
                    .map(|n| n.to_lowercase())
                    .collect();

                match reference {
                    Some(ref_instr) => {
                        let ref_size = ref_instr.bytes.len();

                        // Size check
                        if emu_size != ref_size {
                            report.size_mismatches.push(InstrComparison::SizeMismatch {
                                pc,
                                objdump_size: ref_size,
                                emu_size,
                            });
                            // Use objdump's size to stay in sync
                            pc += ref_size as u32;
                            continue;
                        }

                        // Mnemonic check
                        let ref_mnemonics: Vec<&str> = ref_instr
                            .sub_instrs
                            .iter()
                            .filter(|(m, _)| !m.starts_with("nop"))
                            .map(|(m, _)| m.as_str())
                            .collect();

                        // Compare mnemonics -- try matching by position
                        let all_match = compare_mnemonic_sets(
                            &ref_mnemonics,
                            &emu_raw_mnemonics,
                            &emu_mnemonics,
                        );

                        if all_match {
                            report.matches += 1;
                        } else {
                            let ref_all: Vec<String> = ref_instr
                                .sub_instrs
                                .iter()
                                .map(|(m, _)| m.clone())
                                .collect();
                            let emu_all: Vec<String> = bundle
                                .active_slots()
                                .map(|s| slot_mnemonic(s))
                                .collect();
                            report
                                .mnemonic_mismatches
                                .push(InstrComparison::MnemonicMismatch {
                                    pc,
                                    objdump: ref_all,
                                    emu: emu_all,
                                });
                        }
                    }
                    None => {
                        // We decoded something but objdump didn't have it
                        // (shouldn't happen for well-formed ELFs)
                        let emu_all: Vec<String> = bundle
                            .active_slots()
                            .map(|s| slot_mnemonic(s))
                            .collect();
                        report
                            .extra_instructions
                            .push(InstrComparison::ExtraInstruction {
                                pc,
                                emu: emu_all,
                            });
                    }
                }

                pc += emu_size as u32;
            }
            Err(e) => {
                if let Some(ref_instr) = reference {
                    let ref_mnemonics: Vec<String> = ref_instr
                        .sub_instrs
                        .iter()
                        .map(|(m, _)| m.clone())
                        .collect();
                    report.decode_failures.push(InstrComparison::DecodeFailed {
                        pc,
                        objdump: ref_mnemonics,
                        error: format!("{:?}", e),
                    });
                    pc += ref_instr.bytes.len() as u32;
                } else {
                    // Neither decoded it -- skip 2 bytes (minimum instruction)
                    pc += 2;
                }
            }
        }
    }

    Ok(report)
}

/// Get a display mnemonic from a SlotOp.
fn slot_mnemonic(slot: &crate::interpreter::bundle::SlotOp) -> String {
    if matches!(slot.op, Operation::Nop) {
        return format!("nop({:?})", slot.slot).to_lowercase();
    }
    // Use encoding_name if available, otherwise fall back to Operation debug
    if let Some(ref name) = slot.encoding_name {
        name.to_lowercase()
    } else {
        let debug = format!("{:?}", slot.op);
        debug.split('(').next().unwrap_or("unknown").to_lowercase()
    }
}

/// Compare reference mnemonics against our decoded mnemonics.
///
/// Returns true if all non-NOP mnemonics match (after normalization).
/// Tries both the raw encoding mnemonic and the Operation-derived mnemonic.
fn compare_mnemonic_sets(
    reference: &[&str],
    emu_raw: &[String],
    emu_op: &[String],
) -> bool {
    // If reference has no non-NOP instructions, it's a NOP bundle -- always match
    if reference.is_empty() {
        return true;
    }

    // Try matching against raw encoding mnemonics first
    if reference.len() == emu_raw.len() {
        let all_match = reference
            .iter()
            .zip(emu_raw.iter())
            .all(|(r, e)| mnemonics_match(e, r));
        if all_match {
            return true;
        }
    }

    // Fall back to unordered matching (slots may come in different order)
    if reference.len() == emu_raw.len() {
        let mut matched = vec![false; emu_raw.len()];
        let all_found = reference.iter().all(|r| {
            for (i, e) in emu_raw.iter().enumerate() {
                if !matched[i] && mnemonics_match(e, r) {
                    matched[i] = true;
                    return true;
                }
            }
            false
        });
        if all_found {
            return true;
        }
    }

    // Try Operation-derived mnemonics as last resort
    if reference.len() == emu_op.len() {
        let all_match = reference
            .iter()
            .zip(emu_op.iter())
            .all(|(r, e)| mnemonics_match(e, r));
        if all_match {
            return true;
        }
    }

    false
}

/// Cross-reference all ELF files found under a directory.
///
/// Walks the directory tree, finds all `.elf` files, and produces a
/// cross-reference report for each.
pub fn cross_reference_directory(
    dir: &Path,
    objdump_path: &Path,
    decoder: &InstructionDecoder,
) -> Vec<CrossRefReport> {
    let mut reports = Vec::new();

    let walker = walkdir(dir);
    for elf_path in walker {
        match cross_reference_elf(&elf_path, objdump_path, decoder) {
            Ok(report) => reports.push(report),
            Err(e) => {
                log::warn!("Failed to cross-reference {}: {}", elf_path.display(), e);
            }
        }
    }

    reports
}

/// Walk a directory tree and return all .elf file paths.
fn walkdir(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut result = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                result.extend(walkdir(&path));
            } else if path.extension().map_or(false, |e| e == "elf") {
                result.push(path);
            }
        }
    }
    result
}

/// Print a consolidated summary for multiple reports.
pub fn print_summary(reports: &[CrossRefReport]) {
    let total_instrs: usize = reports.iter().map(|r| r.total_instructions).sum();
    let total_matches: usize = reports.iter().map(|r| r.matches).sum();
    let total_mismatch: usize = reports.iter().map(|r| r.mnemonic_mismatches.len()).sum();
    let total_failures: usize = reports.iter().map(|r| r.decode_failures.len()).sum();
    let total_size: usize = reports.iter().map(|r| r.size_mismatches.len()).sum();

    let rate = if total_instrs > 0 {
        (total_matches as f64 / total_instrs as f64) * 100.0
    } else {
        100.0
    };

    eprintln!("=== Cross-Reference Summary ({} ELFs) ===", reports.len());
    eprintln!(
        "Total: {} instructions, Match: {} ({:.1}%)",
        total_instrs, total_matches, rate
    );
    if total_mismatch > 0 {
        eprintln!("Mnemonic mismatches: {}", total_mismatch);
    }
    if total_failures > 0 {
        eprintln!("Decode failures: {}", total_failures);
    }
    if total_size > 0 {
        eprintln!("Size mismatches: {}", total_size);
    }

    // Show unique mnemonic mismatches across all reports
    if total_mismatch > 0 || total_failures > 0 {
        eprintln!();

        // Collect unique mismatch patterns
        let mut unique_mismatches: BTreeMap<String, usize> = BTreeMap::new();
        for report in reports {
            for comp in &report.mnemonic_mismatches {
                if let InstrComparison::MnemonicMismatch { objdump, emu, .. } = comp {
                    let key = format!("{} -> {}", objdump.join("+"), emu.join("+"));
                    *unique_mismatches.entry(key).or_insert(0) += 1;
                }
            }
        }
        if !unique_mismatches.is_empty() {
            eprintln!("Unique mismatch patterns:");
            for (pattern, count) in &unique_mismatches {
                eprintln!("  {} (x{})", pattern, count);
            }
        }

        // Collect unique decode failure mnemonics
        let mut unique_failures: BTreeMap<String, usize> = BTreeMap::new();
        for report in reports {
            for comp in &report.decode_failures {
                if let InstrComparison::DecodeFailed { objdump, .. } = comp {
                    let key = objdump.join("+");
                    *unique_failures.entry(key).or_insert(0) += 1;
                }
            }
        }
        if !unique_failures.is_empty() {
            eprintln!("Unique decode failures:");
            for (pattern, count) in &unique_failures {
                eprintln!("  {} (x{})", pattern, count);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_objdump_text_basic() {
        let text = r#"
build/test.elf:	file format elf32-aie

Disassembly of section .text:

00000000 <__start>:
       0: 15 01 00 d8 00 00    	jl	#0x1b0
       6: 55 00 e0 0c 07 00    	movxm	sp, #0x70000
       c: 01 00        	nop
"#;
        let map = parse_objdump_text(text).unwrap();

        assert_eq!(map.len(), 3);

        let instr0 = &map[&0];
        assert_eq!(instr0.pc, 0);
        assert_eq!(instr0.bytes, vec![0x15, 0x01, 0x00, 0xd8, 0x00, 0x00]);
        assert_eq!(instr0.sub_instrs.len(), 1);
        assert_eq!(instr0.sub_instrs[0].0, "jl");

        let instr6 = &map[&6];
        assert_eq!(instr6.sub_instrs[0].0, "movxm");
        assert_eq!(instr6.sub_instrs[0].1, "sp, #0x70000");

        let instrc = &map[&0xc];
        assert_eq!(instrc.bytes, vec![0x01, 0x00]);
        assert_eq!(instrc.sub_instrs[0].0, "nop");
    }

    #[test]
    fn test_parse_objdump_text_vliw_bundle() {
        let text = r#"
00000056 <main+0x36>:
      56: bb 10 00 9a c0 01 00 08 24 00	mova	r16, #0x1;		movxm	p1, #0x70400
"#;
        let map = parse_objdump_text(text).unwrap();

        let instr = &map[&0x56];
        assert_eq!(instr.bytes.len(), 10); // 80-bit bundle
        assert_eq!(instr.sub_instrs.len(), 2);
        assert_eq!(instr.sub_instrs[0].0, "mova");
        assert_eq!(instr.sub_instrs[1].0, "movxm");
    }

    #[test]
    fn test_parse_objdump_text_14byte_bundle() {
        let text = r#"
00000060 <main+0x40>:
      60: 7f 00 00 00 71 00 00 02 01 06 00 00 00 00    	nopa	;		nopb	;		rel	#0x30, r16;		nopm	;		nops
"#;
        let map = parse_objdump_text(text).unwrap();

        let instr = &map[&0x60];
        assert_eq!(instr.bytes.len(), 14); // 112-bit bundle
        assert_eq!(instr.sub_instrs.len(), 5);
        assert_eq!(instr.sub_instrs[0].0, "nopa");
        assert_eq!(instr.sub_instrs[1].0, "nopb");
        assert_eq!(instr.sub_instrs[2].0, "rel");
        assert_eq!(instr.sub_instrs[3].0, "nopm");
        assert_eq!(instr.sub_instrs[4].0, "nops");
    }

    #[test]
    fn test_normalize_mnemonic_nops() {
        assert_eq!(normalize_mnemonic("nop"), "nop");
        assert_eq!(normalize_mnemonic("nopa"), "nop");
        assert_eq!(normalize_mnemonic("nopb"), "nop");
        assert_eq!(normalize_mnemonic("nopx"), "nop");
        assert_eq!(normalize_mnemonic("nopm"), "nop");
        assert_eq!(normalize_mnemonic("nops"), "nop");
        assert_eq!(normalize_mnemonic("nopxm"), "nop");
    }

    #[test]
    fn test_mnemonics_match_exact() {
        assert!(mnemonics_match("jl", "jl"));
        assert!(mnemonics_match("paddb", "paddb"));
        assert!(mnemonics_match("movxm", "movxm"));
    }

    #[test]
    fn test_mnemonics_match_dot_suffix() {
        assert!(mnemonics_match("st", "st.s8"));
        assert!(mnemonics_match("st.s8", "st"));
        assert!(mnemonics_match("st.s16", "st.s8")); // both normalize to "st"
    }

    #[test]
    fn test_mnemonics_match_nop_variants() {
        assert!(mnemonics_match("nop", "nopa"));
        assert!(mnemonics_match("nopb", "nops"));
        assert!(mnemonics_match("nopxm", "nop"));
    }

    #[test]
    fn test_mnemonics_no_match() {
        assert!(!mnemonics_match("jl", "jnz"));
        assert!(!mnemonics_match("add", "sub"));
        assert!(!mnemonics_match("lda", "st"));
    }

    #[test]
    fn test_mov_vs_vge_disambiguation() {
        // MV slot bits 0x00713c should decode as "mov", not "vge.d8".
        // From seed_143 at PC 0xDE: 8-byte bundle with ALU+MV format.
        // objdump: "add r1, r1, #0x4; mov r0, #-0x77"
        // emu: "add r1, r1, #0x4; vge.d8 ..."
        use crate::interpreter::bundle::SlotType;

        let decoder = InstructionDecoder::load_default();
        let mv_bits: u64 = 0x00713c;

        // Dump all matching encodings for diagnostics
        let index = decoder.decoder_index();
        if let Some(slot_idx) = index.slot_index("mv") {
            let all_matches = slot_idx.all_matches(mv_bits);
            for enc in &all_matches {
                eprintln!(
                    "  candidate: name='{}' mnemonic='{}' fixed_mask=0x{:06x} fixed_bits=0x{:06x} sort_key={:?}",
                    enc.name, enc.mnemonic, enc.fixed_mask, enc.fixed_bits, enc.sort_key(),
                );
            }
        }

        if let Some(decoded) = decoder.decode_slot_bits(mv_bits, SlotType::Mv) {
            eprintln!(
                "MV 0x{:06x} -> name='{}' mnemonic='{}' sort_key={:?} fixed_mask=0x{:06x} fixed_bits=0x{:06x}",
                mv_bits,
                decoded.encoding.name,
                decoded.encoding.mnemonic,
                decoded.encoding.sort_key(),
                decoded.encoding.fixed_mask,
                decoded.encoding.fixed_bits,
            );
            // The correct match should be a MOV instruction, not VGE
            assert!(
                !decoded.encoding.mnemonic.starts_with("vge"),
                "MV bits 0x{:06x} decoded as '{}' but should be 'mov'",
                mv_bits,
                decoded.encoding.mnemonic,
            );
        } else {
            panic!("Failed to decode MV bits 0x{:06x}", mv_bits);
        }
    }

    #[test]
    fn test_crossref_seed_116() {
        let elf_path = Path::new("build/fuzz/seed_116/aie.mlir.prj/main_core_0_2.elf");
        if !elf_path.exists() {
            eprintln!("Skipping: seed 116 ELF not found");
            return;
        }

        let objdump = Path::new("../llvm-aie/build/bin/llvm-objdump");
        if !objdump.exists() {
            eprintln!("Skipping: llvm-objdump not found");
            return;
        }

        let decoder = InstructionDecoder::load_default();
        let report = cross_reference_elf(elf_path, objdump, &decoder)
            .expect("Cross-reference failed");

        eprintln!("{}", report);

        // We expect a high match rate but don't enforce 100% yet
        assert!(
            report.total_instructions > 0,
            "Should decode at least one instruction"
        );
    }

    #[test]
    fn test_crossref_all_fuzz_elfs() {
        let fuzz_dir = Path::new("build/fuzz");
        if !fuzz_dir.exists() {
            eprintln!("Skipping: build/fuzz not found");
            return;
        }

        let objdump = Path::new("../llvm-aie/build/bin/llvm-objdump");
        if !objdump.exists() {
            eprintln!("Skipping: llvm-objdump not found");
            return;
        }

        let decoder = InstructionDecoder::load_default();
        let reports = cross_reference_directory(fuzz_dir, objdump, &decoder);

        if reports.is_empty() {
            eprintln!("No ELF files found in build/fuzz/");
            return;
        }

        print_summary(&reports);

        // Print per-ELF details for any with failures
        for report in &reports {
            if report.discrepancy_count() > 0 {
                eprintln!();
                eprintln!("{}", report);
            }
        }
    }
}
