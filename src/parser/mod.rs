//! Binary format parsers (XCLBIN, ELF, CDO)
//!
//! This module provides parsers for the binary formats used by AMD XDNA NPUs:
//!
//! - [`xclbin`] - Container format for NPU binaries
//! - [`aie_partition`] - AIE Partition section (contains PDI/CDO)
//! - [`cdo`] - Configuration Data Objects (tile setup commands)
//! - [`elf`] - AIE core ELF executables

pub mod xclbin;
pub mod aie_partition;
pub mod cdo;
pub mod elf;

pub use xclbin::Xclbin;
pub use aie_partition::AiePartition;
pub use cdo::Cdo;
pub use elf::AieElf;
