//! Chain AST for the vector fuzzer.
//!
//! A [`Chain`] is one fuzz case: 8-16 type-legal vector stages over the op
//! table in [`super::table`], plus the edge-weighted input pool the kernel
//! loads operands from. Stage k stores its result to output slice k (64 bytes
//! each, half-width results padded); two-operand stages take operand 2 from a
//! dedicated 64-byte pool slot, while operand 1 chains from the previous
//! stage's result (pool slot 0 for the first stage).

use super::table::table;

/// One stage of a chain: an op-table entry, its mode, and the pool slot for
/// the second operand if the entry is binary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    /// Index into [`table`].
    pub entry_idx: usize,
    /// Mode in `0..entry.modes`.
    pub mode: u8,
    /// Pool slot feeding the second operand; `None` for unary entries.
    pub second_pool_slot: Option<usize>,
}

/// One fuzz case: deterministic stages + input pool for `(seed, target_key)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain {
    pub seed: u64,
    pub target_key: String,
    pub stages: Vec<Stage>,
    pub pool: Vec<u8>,
}

impl Chain {
    /// Coverage-ledger key per stage (`{name}/{out_type:?}/m{mode}`).
    pub fn keys(&self) -> Vec<String> {
        let t = table();
        self.stages
            .iter()
            .map(|s| {
                let e = &t[s.entry_idx];
                format!("{}/{:?}/m{}", e.name, e.out_type, s.mode)
            })
            .collect()
    }

    /// Total output bytes: one 64-byte slice per stage, half-width padded.
    pub fn out_bytes(&self) -> usize {
        self.stages.len() * 64
    }

    /// Pool slot count: slot 0 for the first operand, plus one per binary stage.
    pub fn pool_slots(&self) -> usize {
        1 + self.stages.iter().filter(|s| s.second_pool_slot.is_some()).count()
    }
}
