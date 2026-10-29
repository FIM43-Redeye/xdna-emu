//! The memory interface the Xtensa interpreter runs against.
//!
//! The core never knows what sits behind an address: a platform implements
//! [`XtensaBus`] and the [`Cpu`](super::interp::Cpu) is generic over it
//! (static dispatch; the interpreter has a hot fast path, so no `dyn`).
//!
//! # Contract
//!
//! Every `paddr` is a PHYSICAL address, produced by the CPU's own MMU
//! translation. The extra `vaddr`/`pc` arguments exist only for platforms whose
//! image views are keyed by virtual address (or that attribute accesses to a
//! program counter); a flat memory ignores them. Accesses are infallible: a
//! platform models unmapped memory however it likes (zeros, a stub, ...) and
//! the interpreter does not fault on it.

pub trait XtensaBus {
    /// Fetch one instruction byte at virtual address `vaddr`, already
    /// translated to `paddr`.
    fn fetch8(&mut self, vaddr: u32, paddr: u32) -> u8;

    /// Fetch one instruction byte by physical address alone, with no
    /// virtual-address view applied. Used by the fill-loop recognizer, which
    /// peeks at loop bodies by their translated address.
    fn inst_load8(&mut self, paddr: u32) -> u8;

    /// Read an L32R literal. `vaddr` is the literal's virtual address (some
    /// platforms keep literal pools in a view keyed by it); `paddr` is the
    /// translated address.
    fn literal_load32(&mut self, vaddr: u32, paddr: u32) -> u32;

    /// Little-endian 32-bit data load issued by the instruction at `pc`.
    fn data_load32(&mut self, pc: u32, paddr: u32) -> u32;

    /// Data load of one byte.
    fn data_load8(&mut self, paddr: u32) -> u8;

    /// Little-endian 32-bit data store.
    fn data_store32(&mut self, paddr: u32, v: u32);

    /// Data store of the low byte of `v`.
    fn data_store8(&mut self, paddr: u32, v: u32);

    /// Read a page-table entry for the MMU's autorefill walk. Distinct from
    /// [`XtensaBus::data_load32`] because a platform may treat a table walk as
    /// a hardware access rather than an instruction's load.
    fn pte_load32(&mut self, paddr: u32) -> u32;

    /// Bulk-fill `byte_len` bytes from `paddr` with `pattern` repeated from
    /// phase 0 (`pattern.len()` is 1, 2 or 4 and `paddr` is aligned to it).
    /// Must be byte-identical to `byte_len` successive
    /// [`XtensaBus::data_store8`] calls; platforms override it only to go
    /// faster.
    fn data_fill(&mut self, paddr: u32, pattern: &[u8], byte_len: usize) {
        for i in 0..byte_len {
            self.data_store8(paddr.wrapping_add(i as u32), pattern[i % pattern.len()] as u32);
        }
    }

    /// Mask of INTERRUPT-register bits the platform asserts since the last
    /// call. The CPU ORs it into INTERRUPT at the top of each step.
    fn take_interrupt_assertions(&mut self) -> u32 {
        0
    }
}
