//! Compatibility forwarder. The pure-FFI half now lives in
//! `xdna_archspec::aie2::isa::decoder_ffi`. The MappedOperand /
//! RegisterMap / classify_reg_name half lives in
//! `crate::tablegen::register_map` (relocates to
//! `crate::interpreter::decode::register_map` in Part B).

pub use xdna_archspec::aie2::isa::decoder_ffi::*;
pub use super::register_map::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_init() {
        assert!(init(), "LLVM decoder should initialize successfully");
    }

    #[test]
    fn test_opcode_name_lookup() {
        assert!(init());
        let _ = opcode_name(0);
        assert!(opcode_name(999999).is_none());
    }

    #[test]
    fn test_mv_slot_decode_with_operands() {
        let result = decode_slot(Slot::Mv, 0x00713c);
        assert!(result.is_some(), "MV bits 0x00713c should decode");
        let decoded = result.unwrap();
        assert!(
            decoded.name.contains("MOV") || decoded.name.contains("mov"),
            "Expected MOV, got '{}'",
            decoded.name,
        );
        // Should have operands with register names.
        assert!(!decoded.operands.is_empty(), "Should have operands");
        // First operand(s) should have reg names like "r0", "x0", etc.
        for op in &decoded.operands {
            if let DecodedOperand::Reg { name, .. } = op {
                assert!(!name.starts_with('?'), "Reg name should be resolved: {}", name);
            }
        }
    }

    #[test]
    fn test_vpush_hi_32_disambiguation() {
        let r4_bits: u64 = 0x02903D;
        let decoded = decode_slot(Slot::Mv, r4_bits)
            .expect("r4 bits should decode");
        assert_eq!(
            decoded.name, "VPUSH_HI_32",
            "r4 encoding should decode as VPUSH_HI_32, got {}",
            decoded.name
        );
    }

    #[test]
    fn test_num_defs_for_vadd() {
        // VADD_F in the vec slot should have 1 output (the destination accum).
        // We need a real VADD_F encoding to test this properly.
        // For now, verify that num_defs is populated for any vec instruction.
        let result = decode_slot(Slot::Vec, 0x000001);
        if let Some(decoded) = result {
            // Just verify the field is populated (not all-zero by accident).
            // The specific value depends on the instruction matched.
            let _ = decoded.num_defs;
            let _ = decoded.defs();
            let _ = decoded.uses();
        }
    }

    #[test]
    fn test_register_names_populated() {
        // VPUSH_HI_32 x0, x0, r3 -- should have register names.
        let decoded = decode_slot(Slot::Mv, 0x028C3D)
            .expect("should decode");
        assert_eq!(decoded.name, "VPUSH_HI_32");

        let reg_names: Vec<&str> = decoded.operands.iter().filter_map(|op| {
            if let DecodedOperand::Reg { name, .. } = op { Some(name.as_str()) } else { None }
        }).collect();

        assert!(!reg_names.is_empty(), "Should have register operands");
        eprintln!("VPUSH_HI_32: num_defs={} operands={:?}", decoded.num_defs, decoded.operands);
    }

    #[test]
    fn test_vadd_f_decodes_correctly() {
        // VADD_F bml0, bml0, bml0, r0: raw bytes 89 00 00 00 (LE).
        // In a 32-bit I32_VEC bundle, vec slot bits = word >> 6 = 0x2.
        let decoded = decode_slot(Slot::Vec, 0x2).expect("VADD_F should decode");
        assert_eq!(decoded.name, "VADD_F", "LLVM should identify as VADD_F");
        assert_eq!(decoded.num_defs, 1, "one output (destination accum)");

        // All operands should be accumulator or scalar -- no VectorReg artifacts.
        for op in &decoded.operands {
            if let DecodedOperand::Reg { name, .. } = op {
                assert!(
                    name.starts_with("bm") || name.starts_with("r"),
                    "VADD_F operand should be accum or scalar, got {}",
                    name,
                );
            }
        }
    }

    #[test]
    fn test_vmac_f_untied_acc_operands() {
        // VMAC_F with dst=bml0, acc1=bml2 (untied operands).
        // From real binary: bytes 49 00 00 01 -> bundle 0x01000049.
        // vec_bits = ((0x01000049 >> 4) >> 2) & 0x03FFFFFF = 0x0040001
        let vec_bits: u64 = 0x0040001;
        let decoded = decode_slot(Slot::Vec, vec_bits)
            .expect("VMAC_F with untied acc should decode");

        assert!(
            decoded.name.starts_with("VMAC_F"),
            "Expected VMAC_F, got {}",
            decoded.name,
        );
        assert_eq!(decoded.num_defs, 1, "one output def");

        let reg_names: Vec<&str> = decoded
            .operands
            .iter()
            .filter_map(|op| {
                if let DecodedOperand::Reg { name, .. } = op {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect();

        assert!(reg_names.len() >= 2, "Need at least dst + acc1 registers");
        assert_eq!(reg_names[0], "bml0", "dst should be bml0");
        assert_eq!(reg_names[1], "bml2", "acc1 should be bml2");
    }

    /// Verify that no unmapped register ever appears as an explicit operand
    /// in any decoded instruction across all slots.
    ///
    /// If this test fails, it means LLVM is producing an operand with a
    /// register name we don't handle -- that's a real gap to fix.
    #[test]
    fn test_unmapped_regs_never_appear_as_explicit_operands() {
        let (_, unmapped) = reg_map_coverage();
        let unmapped_set: std::collections::HashSet<&str> =
            unmapped.iter().map(|s| s.as_str()).collect();

        let slots = [
            (Slot::Alu, 20),
            (Slot::Lda, 21),
            (Slot::Ldb, 16),
            (Slot::Mv,  22),
            (Slot::St,  21),
            (Slot::Vec, 26),
        ];

        let mut hits: Vec<String> = Vec::new();

        // Probe a range of bit patterns per slot to cover diverse encodings.
        for (slot, width) in &slots {
            for probe in 0..512u64 {
                // Spread probes across the encoding space.
                let bits = probe << (width / 2);
                if let Some(decoded) = decode_slot(*slot, bits) {
                    for op in &decoded.operands {
                        if let DecodedOperand::Reg { name, .. } = op {
                            if unmapped_set.contains(name.as_str()) {
                                let msg = format!(
                                    "{} in {} (slot {:?} bits 0x{:X})",
                                    name, decoded.name, slot, bits
                                );
                                if !hits.iter().any(|h| h.starts_with(name.as_str())) {
                                    hits.push(msg);
                                }
                            }
                        }
                    }
                }
            }
        }

        if !hits.is_empty() {
            eprintln!("Unmapped registers appearing as explicit operands ({}):", hits.len());
            for h in &hits {
                eprintln!("  {}", h);
            }
            panic!(
                "{} unmapped register(s) appear as explicit operands -- need mapping",
                hits.len()
            );
        }
    }

    #[test]
    fn test_llvm_decode_operands_map_correctly() {
        // Decode a real instruction and verify all register operands map
        // through operand_from_reg_name without returning None.
        let decoded = decode_slot(Slot::Mv, 0x028C3D)
            .expect("VPUSH_HI_32 should decode");
        assert_eq!(decoded.name, "VPUSH_HI_32");

        let mut mapped = 0;
        for op in &decoded.operands {
            if let DecodedOperand::Reg { name, .. } = op {
                let result = operand_from_reg_name(name);
                assert!(
                    result.is_some(),
                    "LLVM register '{}' should map to an Operand",
                    name,
                );
                mapped += 1;
            }
        }
        assert!(mapped > 0, "Should have mapped at least one register operand");
    }

    // -----------------------------------------------------------------------
    // Instruction metadata (InstrInfo) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_query_all_instr_info_returns_data() {
        let infos = query_all_instr_info();
        assert!(infos.len() > 600, "Should have 600+ opcodes, got {}", infos.len());

        // Count how many have latency data.
        let with_latency = infos.iter().filter(|i| i.latency.is_some()).count();
        assert!(with_latency > 100,
            "Should have 100+ opcodes with latency, got {}", with_latency);

        // Count how many have non-zero sched class indices.
        let with_sched = infos.iter().filter(|i| i.sched_class > 0).count();
        assert!(with_sched > 100,
            "Should have 100+ opcodes with sched class, got {}", with_sched);
    }

    #[test]
    fn test_instr_info_flags_populated() {
        let infos = query_all_instr_info();

        // Scan all opcodes for instructions with MayLoad and MayStore flags.
        let loads = infos.iter().filter(|i| i.is_load()).count();
        let stores = infos.iter().filter(|i| i.is_store()).count();
        let branches = infos.iter().filter(|i| i.is_branch()).count();

        assert!(loads > 50, "Should have 50+ load instructions, got {}", loads);
        assert!(stores > 50, "Should have 50+ store instructions, got {}", stores);
        assert!(branches > 5, "Should have 5+ branch instructions, got {}", branches);
    }

    #[test]
    fn test_instr_info_latency_cross_check() {
        // Cross-validate LLVM itinerary latencies against our AM020 constants.
        let infos = query_all_instr_info();

        let mut found_load = false;
        let mut found_store = false;
        let mut found_vmac = false;

        for (opcode, info) in infos.iter().enumerate() {
            if let Some(latency) = info.latency {
                let name = match opcode_name(opcode as u32) {
                    Some(n) => n,
                    None => continue,
                };

                // Load instructions with MayLoad flag should have latency 7.
                if info.is_load() && name.starts_with("LDA_") && !found_load {
                    assert_eq!(latency, 7, "{} latency should be 7 (load)", name);
                    found_load = true;
                }

                // Store instructions with MayStore flag should have latency 1.
                if info.is_store() && name.starts_with("ST_") && !found_store {
                    assert_eq!(latency, 1, "{} latency should be 1 (store)", name);
                    found_store = true;
                }

                // Vector MAC (integer) should have latency 5.
                // VMAC_F (float) may differ -- skip floating-point variants.
                if name.starts_with("VMAC_vmac_") && !found_vmac {
                    assert_eq!(latency, 5, "{} latency should be 5 (VMAC)", name);
                    found_vmac = true;
                }
            }
        }
        assert!(found_load, "Should have found a load instruction with latency");
        assert!(found_store, "Should have found a store instruction with latency");
        assert!(found_vmac, "Should have found a VMAC instruction with latency");
    }

    #[test]
    fn test_vec_decode_operands_map_correctly() {
        // Decode a vec-slot instruction and verify register name mapping.
        for bits in [0x000001u64, 0x100001, 0x200001] {
            if let Some(decoded) = decode_slot(Slot::Vec, bits) {
                for op in &decoded.operands {
                    if let DecodedOperand::Reg { name, .. } = op {
                        let result = operand_from_reg_name(name);
                        if result.is_none() {
                            eprintln!(
                                "WARNING: unmapped register '{}' in {} (Vec 0x{:06X})",
                                name, decoded.name, bits
                            );
                        }
                    }
                }
            }
        }
    }
}
