//! Fuzzer domain tenants: each implements `core::Domain` and is driven by the
//! shared `core::engine::run_campaign`. The vector tenant came first (framework
//! Step 1); the scalar tenant (Step 2) is the second. Timing and trace are
//! planned as *modes* on these tenants, not separate domains.

pub mod vector;
