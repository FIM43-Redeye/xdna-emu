//! Chain -> C++ kernel lowering.
//!
//! [`lower_chain`] renders a [`Chain`] as a self-contained aie_api kernel
//! with the fuzz template's I/O contract:
//! `extern "C" void fuzz_kernel(int32_t* __restrict in, int32_t* __restrict out)`.
//! Pool slots are consecutive 64-byte slices at `in + 16*slot` (i32 words);
//! stage k stores its result at `out + 16*k`. Half-width (32-byte) stages load
//! and store with their half lane count, leaving the upper 32 bytes of the
//! slot untouched -- the runner zero-fills both emulator and silicon output
//! buffers, so the unwritten halves compare equal.

use super::chain::Chain;
use super::table::{table, VecType};

/// `aie::load_v` expression for `vt` at `in + words` (i32-word offset).
fn load_expr(vt: VecType, words: usize) -> String {
    format!("aie::load_v<{}>(({}*)(in + {}))", vt.lanes(), vt.ctype(), words)
}

/// Lower a chain to complete C++ kernel source.
pub fn lower_chain(chain: &Chain) -> String {
    let t = table();
    let mut s = String::new();
    s.push_str(&format!(
        "// Generated vector fuzz chain -- seed {}, target {}. DO NOT EDIT.\n",
        chain.seed, chain.target_key
    ));
    s.push_str("#include <stdint.h>\n");
    s.push_str("#include <aie_api/aie.hpp>\n");

    // Peano's GlobalISel crashes ("Register class not set") on chains of >=3
    // dependent bf16 element-wise ops at -O2 (e.g. three chained aie::add).
    // Independent ops and noinline-call-separated chains compile fine, so bf16
    // stages route through one noinline helper per stage. The helpers still
    // emit native vector code (vadd.f via fp32 conv, vmin_ge.bf16); verified
    // by the compile-clean test and objdump. Bf16 chains never mix types, so
    // a single helper signature over Bf16x32 covers every entry.
    let bf16_chain = chain.stages.iter().any(|st| t[st.entry_idx].out_type.is_float());
    if bf16_chain {
        s.push_str("using V = aie::vector<bfloat16, 32>;\n");
        for (k, st) in chain.stages.iter().enumerate() {
            let e = &t[st.entry_idx];
            let arity = e.in_types.len();
            let names: Vec<String> = ["a", "b"][..arity].iter().map(|n| n.to_string()).collect();
            let body = (e.emit)(&names, st.mode, e.out_type);
            let params = if arity == 2 { "V a, V b" } else { "V a" };
            s.push_str(&format!("__attribute__((noinline)) static V h{k}({params}) {{ return {body}; }}\n"));
        }
    }

    s.push_str("extern \"C\" void fuzz_kernel(int32_t* __restrict in, int32_t* __restrict out) {\n");

    for (k, st) in chain.stages.iter().enumerate() {
        let e = &t[st.entry_idx];
        let mut args = Vec::with_capacity(2);
        if k == 0 {
            let vt = e.in_types[0];
            s.push_str(&format!(
                "  aie::vector<{}, {}> v0_in = {};\n",
                vt.ctype(),
                vt.lanes(),
                load_expr(vt, 0)
            ));
            args.push("v0_in".to_string());
        } else {
            args.push(format!("v{}", k - 1));
        }
        if let Some(slot) = st.second_pool_slot {
            let vt = e.in_types[1];
            s.push_str(&format!(
                "  aie::vector<{}, {}> p{slot} = {};\n",
                vt.ctype(),
                vt.lanes(),
                load_expr(vt, 16 * slot)
            ));
            args.push(format!("p{slot}"));
        }
        let expr = if bf16_chain {
            format!("h{k}({})", args.join(", "))
        } else {
            (e.emit)(&args, st.mode, e.out_type)
        };
        s.push_str(&format!("  auto v{k} = {expr};\n"));
        s.push_str(&format!("  aie::store_v(({}*)(out + {}), v{k});\n", e.out_type.ctype(), 16 * k));
    }
    s.push_str("}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::vector::gen::{generate, Xorshift64};
    use crate::fuzzer::vector::table::universe_keys;

    /// Same deterministic key pick as gen's tests.
    fn key_for_seed(universe: &[String], seed: u64) -> &str {
        let mut rng = Xorshift64(seed.wrapping_mul(0x9E3779B97F4A7C15).max(1));
        &universe[(rng.next() % universe.len() as u64) as usize]
    }

    #[test]
    fn two_hundred_seeds_structural_invariants() {
        let t = table();
        let universe = universe_keys();
        for seed in 0..200u64 {
            let key = key_for_seed(&universe, seed);
            let chain = generate(seed, key);
            let src = lower_chain(&chain);

            // Balanced braces.
            let opens = src.matches('{').count();
            let closes = src.matches('}').count();
            assert_eq!(opens, closes, "seed {seed} key {key}: unbalanced braces");

            // Required boilerplate.
            assert!(src.contains("#include <aie_api/aie.hpp>"), "seed {seed}: missing include");
            assert!(
                src.contains(
                    "extern \"C\" void fuzz_kernel(int32_t* __restrict in, int32_t* __restrict out)"
                ),
                "seed {seed}: missing signature"
            );

            // One store per stage, at stride 16 words.
            assert_eq!(
                src.matches("aie::store_v(").count(),
                chain.stages.len(),
                "seed {seed} key {key}: store count"
            );
            for k in 0..chain.stages.len() {
                assert!(
                    src.contains(&format!("(out + {}), v{k})", 16 * k)),
                    "seed {seed} key {key}: stage {k} store offset"
                );
            }

            // Every binary stage loads its operand from in + 16*slot.
            for (k, st) in chain.stages.iter().enumerate() {
                if let Some(slot) = st.second_pool_slot {
                    let vt = t[st.entry_idx].in_types[1];
                    assert!(
                        src.contains(&format!(
                            "p{slot} = aie::load_v<{}>(({}*)(in + {}))",
                            vt.lanes(),
                            vt.ctype(),
                            16 * slot
                        )),
                        "seed {seed} key {key}: stage {k} slot {slot} load"
                    );
                }
            }

            // bf16 chains spell the element type.
            if key.contains("Bf16x32") {
                assert!(src.contains("bfloat16"), "seed {seed} key {key}: no bfloat16");
            }

            // No two declared variables share a name.
            let mut names = std::collections::HashSet::new();
            for line in src.lines() {
                let line = line.trim_start();
                let name = if let Some(rest) = line.strip_prefix("auto ") {
                    rest.split(' ').next()
                } else if line.starts_with("aie::vector<") {
                    line.split("> ").nth(1).and_then(|r| r.split(' ').next())
                } else {
                    None
                };
                if let Some(n) = name {
                    assert!(names.insert(n.to_string()), "seed {seed} key {key}: duplicate var {n}");
                }
            }
        }
    }

    #[test]
    fn golden_structure_add_i32x16() {
        let chain = generate(1, "add/I32x16/m0");
        let src = lower_chain(&chain);

        // Header and frame.
        assert!(
            src.starts_with("// Generated vector fuzz chain -- seed 1, target add/I32x16/m0. DO NOT EDIT.\n")
        );
        assert!(src.ends_with("}\n"));

        // Stage 0 is binary add on int32: exact decl, operand-2, and store lines.
        assert!(src.contains("  aie::vector<int32_t, 16> v0_in = aie::load_v<16>((int32_t*)(in + 0));\n"));
        assert!(src.contains("  aie::vector<int32_t, 16> p1 = aie::load_v<16>((int32_t*)(in + 16));\n"));
        assert!(src.contains("  auto v0 = "));
        assert!(src.contains("  aie::store_v((int32_t*)(out + 0), v0);\n"));
        assert!(src.contains("  aie::store_v((int32_t*)(out + 16), v1);\n"));
    }
}
