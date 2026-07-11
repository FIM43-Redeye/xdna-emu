//! Boot-to-idle firmware test suite. Split by purpose; the shared symbol
//! helpers live here, each sub-module is `use super::*`.

use super::*;

mod guards;
mod static_tools;
mod runtime_tools;
mod idle_loop;
mod mmu;
mod external_stimulus;
mod external_observe;
mod coherence_mapper;

/// Nearest symbol at or below `pc`, formatted `name+0xNN` (or bare `name`
/// at the exact entry), for readable probe output. Empty when no symbol
/// lies within `MAX_SPAN` below `pc` -- so a gap between symbols reads as
/// blank rather than getting mislabeled as a distant earlier function.
/// Names live in `build/experiments/firmware-re/symbols.txt`; add semantic
/// names there as RE proceeds (e.g. `task_dispatcher`).
fn nearest_symbol(symbols: &std::collections::HashMap<u32, String>, pc: u32) -> String {
    const MAX_SPAN: u32 = 0x800;
    let mut best: Option<(u32, &str)> = None;
    for (&addr, name) in symbols {
        if addr <= pc && pc - addr < MAX_SPAN && best.map_or(true, |(b, _)| addr > b) {
            best = Some((addr, name.as_str()));
        }
    }
    match best {
        Some((addr, name)) if addr == pc => name.to_string(),
        Some((addr, name)) => format!("{name}+{:#x}", pc - addr),
        None => String::new(),
    }
}

/// Like [`nearest_symbol`] but returns the bucket KEY (the symbol entry
/// address, or the 4KB page when no symbol is near) plus a bare routine
/// label -- so cycle-accounting aggregates a whole routine into one bucket
/// instead of splitting it by offset. Every PC maps to exactly one bucket,
/// which is the "every cycle accounted" invariant the histogram relies on.
fn routine_bucket(symbols: &std::collections::HashMap<u32, String>, pc: u32) -> (u32, String) {
    const MAX_SPAN: u32 = 0x800;
    let mut best: Option<(u32, &str)> = None;
    for (&addr, name) in symbols {
        if addr <= pc && pc - addr < MAX_SPAN && best.map_or(true, |(b, _)| addr > b) {
            best = Some((addr, name.as_str()));
        }
    }
    match best {
        Some((addr, name)) => (addr, name.to_string()),
        None => {
            let page = pc & !0xfff;
            (page, format!("<no-sym {page:#x}>"))
        }
    }
}
