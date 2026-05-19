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

#[cfg(test)]
mod tests {
    use super::*;

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
