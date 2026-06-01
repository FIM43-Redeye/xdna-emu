//! aiesim backend: a `NpuBackend` that drives the closed AIE2 cluster ISS
//! in-process through the C++ bridge library `libxdna_aiesim_bridge.so`.
//!
//! Layering:
//! - `abi`     -- the C-ABI surface + the `BridgeAbi` trait + `MockBridge`.
//! - `bridge`  -- the real `dlopen`-backed `BridgeAbi` impl.
//! - `backend` -- `AiesimBackend`, which implements `NpuBackend` over a
//!                `Box<dyn BridgeAbi>`.
//!
//! Feature-gated behind `aiesim`; nothing here compiles in a default build.

pub(crate) mod abi;
pub(crate) mod backend;
pub(crate) mod bridge;
