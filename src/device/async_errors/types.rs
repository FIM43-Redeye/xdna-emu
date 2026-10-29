//! Wire-format types for async-error delivery.
//!
//! Direct mirrors of driver structs; bytes are what firmware would DMA into
//! the host async-event buffer. Compile-time size assertions pin the layout;
//! runtime tests verify offsets via byte round-trip.
//!
//! References:
//! - `aie_error`           -> xdna-driver/src/driver/amdxdna/aie2_error.c:56-64
//! - `aie_err_info`        -> xdna-driver/src/driver/amdxdna/aie2_error.c:66-71
//! - `amdxdna_async_error` -> xdna-driver/include/uapi/drm/amdxdna_accel.h:610-617

/// Mirrors driver `struct aie_error` (12 bytes). NOT packed -- driver
/// comment "Don't pack, unless XAIE side changed" (aie2_error.c:55).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct AieError {
    pub row: u8,
    pub col: u8,
    pub reserved_0: u16,
    pub mod_type: u32,
    pub event_id: u8,
    pub reserved_1: u8,
    pub reserved_2: u16,
}

const _: () = assert!(std::mem::size_of::<AieError>() == 12);

/// Header preceding `err_cnt` `AieError` entries in the ring.
/// Mirrors `struct aie_err_info` (aie2_error.c:66-71).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct AieErrInfoHeader {
    pub err_cnt: u32,
    pub ret_code: u32,
    pub rsvd: u32,
}

const _: () = assert!(std::mem::size_of::<AieErrInfoHeader>() == 12);

/// uapi async-error record returned by `DRM_AMDXDNA_HW_LAST_ASYNC_ERR`.
/// Mirrors `struct amdxdna_async_error` (amdxdna_accel.h:610-617).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct AmdxdnaAsyncError {
    pub err_code: u64,
    pub ts_us: u64,
    pub ex_err_code: u64,
}

const _: () = assert!(std::mem::size_of::<AmdxdnaAsyncError>() == 24);

/// 8 KB per driver `ASYNC_BUF_SIZE` (aie2_msg_priv.h:406, SZ_8K).
pub const ASYNC_BUF_SIZE: usize = 8 * 1024;

/// Max errors that fit after the header.
pub const MAX_ERRORS_PER_RING: usize =
    (ASYNC_BUF_SIZE - std::mem::size_of::<AieErrInfoHeader>()) / std::mem::size_of::<AieError>();

/// Emu-defined `ret_code` value set when a push overflows. Driver treats
/// any nonzero `ret_code` as an error (aie2_error.c worker handling).
pub const RET_CODE_OVERFLOW: u32 = 1;

/// Error type returned by `AsyncRing::push` when the ring is at capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overflow;

/// Aligned backing storage so casts to `AieErrInfoHeader` and `AieError` are
/// sound. Both wire-format types contain `u32` fields requiring align(4);
/// a bare `[u8; N]` only guarantees align(1).
#[repr(C, align(4))]
struct AlignedBuf([u8; ASYNC_BUF_SIZE]);

/// Per-column 8 KB ring in driver-wire format.
///
/// Layout: 12-byte `AieErrInfoHeader` at offset 0, followed by `err_cnt`
/// `AieError` records (12 bytes each) starting at offset 12. Byte-compatible
/// with what firmware would DMA into the driver's `dma_hdl` async-event
/// buffer.
pub struct AsyncRing {
    bytes: Box<AlignedBuf>,
}

impl AsyncRing {
    pub fn new() -> Self {
        Self { bytes: Box::new(AlignedBuf([0u8; ASYNC_BUF_SIZE])) }
    }

    pub fn header(&self) -> &AieErrInfoHeader {
        // SAFETY: AlignedBuf is #[repr(C, align(4))] so its bytes start
        // 4-aligned, matching AieErrInfoHeader's alignment. The struct is
        // 12 bytes of POD (no padding-sensitive fields) and lives within
        // the backing buffer.
        unsafe { &*(self.bytes.0.as_ptr() as *const AieErrInfoHeader) }
    }

    fn header_mut(&mut self) -> &mut AieErrInfoHeader {
        // SAFETY: AlignedBuf is #[repr(C, align(4))] so its bytes start
        // 4-aligned, matching AieErrInfoHeader's alignment. &mut self gives
        // exclusive access to the backing buffer.
        unsafe { &mut *(self.bytes.0.as_mut_ptr() as *mut AieErrInfoHeader) }
    }

    /// Append a record. Returns `Err(Overflow)` if the ring is at capacity;
    /// in that case the ring state is unchanged.
    pub fn push(&mut self, e: AieError) -> Result<(), Overflow> {
        let cnt = self.header().err_cnt as usize;
        if cnt >= MAX_ERRORS_PER_RING {
            return Err(Overflow);
        }
        let header_size = std::mem::size_of::<AieErrInfoHeader>();
        let rec_size = std::mem::size_of::<AieError>();
        let offset = header_size + cnt * rec_size;
        // SAFETY: bounds-checked: offset + rec_size <= ASYNC_BUF_SIZE iff
        // cnt < MAX_ERRORS_PER_RING (the gate above). AlignedBuf guarantees
        // align(4) at the base; write_unaligned handles any intra-record
        // alignment variation without assuming stricter alignment at offset.
        unsafe {
            let dst = self.bytes.0.as_mut_ptr().add(offset) as *mut AieError;
            std::ptr::write_unaligned(dst, e);
        }
        self.header_mut().err_cnt += 1;
        Ok(())
    }

    /// Read-only slice view of stored records.
    pub fn records(&self) -> &[AieError] {
        let cnt = self.header().err_cnt as usize;
        let header_size = std::mem::size_of::<AieErrInfoHeader>();
        // SAFETY: AlignedBuf guarantees align(4) at the base; AieError is
        // repr(C) with align(4), and the first record starts at offset 12
        // (a multiple of 4), so all record pointers are properly aligned.
        // cnt is bounded by MAX_ERRORS_PER_RING (push enforces).
        unsafe {
            let p = self.bytes.0.as_ptr().add(header_size) as *const AieError;
            std::slice::from_raw_parts(p, cnt)
        }
    }

    /// Copy header + valid records into `dst`. Returns the number of bytes
    /// copied (at most `dst.len()` and at most `12 + err_cnt * 12`).
    pub fn read_into(&self, dst: &mut [u8]) -> usize {
        let used = std::mem::size_of::<AieErrInfoHeader>()
            + (self.header().err_cnt as usize) * std::mem::size_of::<AieError>();
        let n = used.min(dst.len());
        dst[..n].copy_from_slice(&self.bytes.0[..n]);
        n
    }

    /// Set the ret_code field (used to signal Overflow to consumers).
    pub fn set_ret_code(&mut self, code: u32) {
        self.header_mut().ret_code = code;
    }

    /// Zero the entire buffer (header + payload area).
    pub fn clear(&mut self) {
        self.bytes.0.fill(0);
    }
}

impl Default for AsyncRing {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_starts_empty() {
        let ring = AsyncRing::new();
        assert_eq!(ring.header().err_cnt, 0);
        assert_eq!(ring.header().ret_code, 0);
    }

    #[test]
    fn push_increments_err_cnt_and_stores_record() {
        let mut ring = AsyncRing::new();
        let e = AieError { row: 2, col: 1, event_id: 69, mod_type: 1, ..Default::default() };
        ring.push(e).expect("first push must succeed");
        assert_eq!(ring.header().err_cnt, 1);
        assert_eq!(ring.records()[0], e);
        let e2 = AieError { row: 3, col: 2, event_id: 70, mod_type: 1, ..Default::default() };
        ring.push(e2).expect("second push must succeed");
        assert_eq!(ring.header().err_cnt, 2);
        assert_eq!(ring.records()[0], e);
        assert_eq!(ring.records()[1], e2);
    }

    #[test]
    fn push_at_capacity_returns_overflow() {
        let mut ring = AsyncRing::new();
        // Fill to capacity.
        for i in 0..MAX_ERRORS_PER_RING as u32 {
            ring.push(AieError {
                row: (i & 0xFF) as u8,
                col: ((i >> 8) & 0xFF) as u8,
                event_id: 69,
                mod_type: 1,
                ..Default::default()
            })
            .expect("push within capacity must succeed");
        }
        assert_eq!(ring.header().err_cnt as usize, MAX_ERRORS_PER_RING);
        // One more must overflow.
        let next = AieError { row: 0xFF, col: 0xFF, event_id: 69, mod_type: 1, ..Default::default() };
        assert!(matches!(ring.push(next), Err(Overflow)));
        // err_cnt unchanged.
        assert_eq!(ring.header().err_cnt as usize, MAX_ERRORS_PER_RING);
    }

    #[test]
    fn read_into_copies_header_then_records() {
        let mut ring = AsyncRing::new();
        ring.push(AieError { row: 5, col: 1, event_id: 69, mod_type: 1, ..Default::default() })
            .unwrap();
        let mut dst = vec![0u8; 64];
        let n = ring.read_into(&mut dst);
        // 12 byte header + 1 * 12 byte record = 24 bytes.
        assert_eq!(n, 24);
        // Header err_cnt = 1 (little-endian u32 at offset 0).
        assert_eq!(&dst[0..4], &1u32.to_le_bytes());
        // First record's row at offset 12.
        assert_eq!(dst[12], 5);
        assert_eq!(dst[13], 1);
    }

    #[test]
    fn read_into_returns_header_on_empty_ring() {
        let ring = AsyncRing::new();
        let mut dst = vec![0u8; 64];
        // Header is always present; an empty ring still copies the 12-byte header.
        let n = ring.read_into(&mut dst);
        assert_eq!(n, 12, "empty ring read returns just the header");
        assert_eq!(&dst[0..4], &0u32.to_le_bytes(), "err_cnt == 0");
    }

    #[test]
    fn read_into_truncates_to_dst_size() {
        let mut ring = AsyncRing::new();
        ring.push(AieError::default()).unwrap();
        ring.push(AieError::default()).unwrap();
        // dst smaller than the 36 bytes the ring contains: should copy only dst.len().
        let mut dst = vec![0u8; 20];
        assert_eq!(ring.read_into(&mut dst), 20);
    }

    #[test]
    fn clear_zeros_header_and_records() {
        let mut ring = AsyncRing::new();
        ring.push(AieError { row: 5, col: 1, event_id: 69, mod_type: 1, ..Default::default() })
            .unwrap();
        ring.set_ret_code(RET_CODE_OVERFLOW);
        ring.clear();
        assert_eq!(ring.header().err_cnt, 0);
        assert_eq!(ring.header().ret_code, 0);
        let mut dst = vec![0u8; ASYNC_BUF_SIZE];
        ring.read_into(&mut dst);
        // Only header (12B) is "used" -- all zero.
        assert!(dst[..12].iter().all(|&b| b == 0));
    }

    #[test]
    fn max_errors_per_ring_is_681() {
        // (8192 - 12) / 12 = 681 (integer division).
        assert_eq!(MAX_ERRORS_PER_RING, 681);
    }

    #[test]
    fn aie_error_field_offsets_via_byte_inspection() {
        // Construct a record with distinguishable values in each field,
        // transmute to bytes, and verify each field's position.
        let e = AieError {
            row: 0xA1,
            col: 0xB2,
            reserved_0: 0,
            mod_type: 0xDEAD_BEEF,
            event_id: 0xC3,
            reserved_1: 0,
            reserved_2: 0,
        };
        let bytes: [u8; 12] = unsafe { std::mem::transmute(e) };
        assert_eq!(bytes[0], 0xA1, "row at offset 0");
        assert_eq!(bytes[1], 0xB2, "col at offset 1");
        // bytes[2..4] is reserved_0
        // bytes[4..8] is mod_type, little-endian on x86_64
        assert_eq!(&bytes[4..8], &0xDEAD_BEEF_u32.to_le_bytes());
        assert_eq!(bytes[8], 0xC3, "event_id at offset 8");
    }

    #[test]
    fn amdxdna_async_error_field_offsets() {
        let r = AmdxdnaAsyncError {
            err_code: 0x1111_2222_3333_4444,
            ts_us: 0x5555_6666_7777_8888,
            ex_err_code: 0x9999_AAAA_BBBB_CCCC,
        };
        let bytes: [u8; 24] = unsafe { std::mem::transmute(r) };
        assert_eq!(&bytes[0..8], &r.err_code.to_le_bytes());
        assert_eq!(&bytes[8..16], &r.ts_us.to_le_bytes());
        assert_eq!(&bytes[16..24], &r.ex_err_code.to_le_bytes());
    }
}
