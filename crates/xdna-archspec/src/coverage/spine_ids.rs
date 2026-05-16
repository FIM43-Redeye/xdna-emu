//! Single source of truth for the capability-spine domain ids, as plain
//! `&'static str`s with ZERO `crate::` imports so build.rs can `#[path]`-
//! include it without dragging in `crate::aie2::isa` (the irreducible cycle,
//! Plan 2). `units::capability_spine()` builds the rich `CapabilityDomain`s
//! from this list; nothing else defines spine ids (spec Section 6: one
//! location).
pub const SPINE_DOMAIN_IDS: &[&str] = &[
    "core",
    "program_memory",
    "program_counter",
    "data_memory",
    "dma",
    "locks",
    "stream_switch",
    "events_trace",
    "performance_counters",
    "timer",
    "watchpoint",
    "debug_halt",
    "cascade",
    "interrupt",
    "noc",
    "shim_mux",
];
