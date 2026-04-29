//! FFI wrapper for the kernarg classifier.
//!
//! Exposes `xdna_emu_classify_kernargs` as a C ABI symbol so
//! `bridge-trace-runner` (and any other external tool) can call into our
//! parsers to discover kernarg roles at runtime. The function takes the
//! xclbin bytes (for the CDO cross-check) and the raw NPU instruction
//! stream (for the actual classification), parses both via the in-tree
//! Rust implementations, and writes a fixed-size record per kernarg
//! into a caller-supplied buffer.
//!
//! # Safety
//!
//! The caller must uphold:
//! - `xclbin_data` points to `xclbin_len` readable bytes (or is null +
//!   `xclbin_len == 0`).
//! - `insts_data` points to `insts_len` readable bytes, non-null.
//! - `out` points to at least `out_cap * sizeof(XdnaEmuKernargRole)`
//!   writable bytes (or is null + `out_cap == 0` to query the required
//!   size).
//!
//! Error messages go through the thread-local LAST_ERROR; callers
//! retrieve them via `xdna_emu_get_error`.

use std::ffi::c_void;
use std::slice;

use xdna_emu_core::device::ops::DeviceOp;
use xdna_emu_core::npu::{classify_with_topology, KernargRole, NpuInstructionStream};
use xdna_emu_core::parser::aie_partition::AiePartition;
use xdna_emu_core::parser::cdo::framing::find_cdo_offset;
use xdna_emu_core::parser::cdo::semantics::lower;
use xdna_emu_core::parser::cdo::syntax::Cdo;
use xdna_emu_core::parser::stream_switch_topology::StreamSwitchTopology;
use xdna_emu_core::parser::xclbin::Xclbin;

use crate::set_last_error;

/// Kernarg role codes for the FFI. Stable wire format -- do not renumber.
/// Mirrors `xdna_emu_core::npu::KernargRole` but is `#[repr(u8)]` with
/// its own definition to keep the ABI explicit.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdnaEmuKernargRoleCode {
    DataMm2s = 0,
    DataS2mm = 1,
    Ctrlpkt = 2,
    Unknown = 255,
}

impl From<KernargRole> for XdnaEmuKernargRoleCode {
    fn from(r: KernargRole) -> Self {
        match r {
            KernargRole::DataMm2s => XdnaEmuKernargRoleCode::DataMm2s,
            KernargRole::DataS2mm => XdnaEmuKernargRoleCode::DataS2mm,
            KernargRole::Ctrlpkt => XdnaEmuKernargRoleCode::Ctrlpkt,
            KernargRole::Unknown => XdnaEmuKernargRoleCode::Unknown,
        }
    }
}

/// One classification record, flat for easy C consumption.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XdnaEmuKernargRole {
    /// arg_idx as it appears in the NPU instruction stream's DdrPatch
    /// ops. To convert to the XRT kernarg index, add the number of
    /// leading metadata kernargs (opcode, instr, ninstr = 3 in the
    /// mlir-aie convention).
    pub arg_idx: u8,
    pub role: u8,
    pub _pad: [u8; 2],
    /// Full (column-qualified) BD source register address first
    /// observed for this arg, for diagnostics.
    pub bd_reg_addr: u32,
}

/// Classify kernarg roles for an xclbin + NPU instruction stream.
///
/// Returns the number of roles that were (or would be) written, or a
/// negative value on error.
///
/// Pass `out = null` and `out_cap = 0` to query the required capacity
/// without materialising the results; the return value is the number
/// of roles the stream produces. A subsequent call with a buffer of
/// that size (or larger) will populate it.
///
/// If `xclbin_data` is null or `xclbin_len == 0`, the classifier runs
/// without the CDO cross-check (tiers 1 and 2 only). Supplying the
/// xclbin enables tier 3's disagreement warning but does not change
/// the classification itself.
///
/// # Safety
/// See module docs. All pointer preconditions must hold; violating
/// them is UB.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_classify_kernargs(
    xclbin_data: *const c_void,
    xclbin_len: u64,
    insts_data: *const c_void,
    insts_len: u64,
    out: *mut XdnaEmuKernargRole,
    out_cap: u64,
) -> i64 {
    if insts_data.is_null() || insts_len == 0 {
        set_last_error("xdna_emu_classify_kernargs: insts buffer is empty".to_string());
        return -1;
    }
    let insts = slice::from_raw_parts(insts_data as *const u8, insts_len as usize);
    let stream = match NpuInstructionStream::parse(insts) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("NPU instruction parse: {}", e));
            return -2;
        }
    };

    let topology = build_topology(xclbin_data, xclbin_len);
    let classification = match topology {
        Some(topo) => classify_with_topology(&stream, &topo),
        None => xdna_emu_core::npu::classify_kernargs(&stream),
    };

    let n = classification.len() as i64;
    if out.is_null() || out_cap == 0 {
        return n;
    }
    let max = classification.len().min(out_cap as usize);
    let out_slice = slice::from_raw_parts_mut(out, max);
    for (dst, src) in out_slice.iter_mut().zip(classification.iter().take(max)) {
        *dst = XdnaEmuKernargRole {
            arg_idx: src.arg_idx,
            role: XdnaEmuKernargRoleCode::from(src.role) as u8,
            _pad: [0; 2],
            bd_reg_addr: src.bd_reg_addr,
        };
    }
    n
}

fn build_topology(data: *const c_void, len: u64) -> Option<StreamSwitchTopology> {
    if data.is_null() || len == 0 {
        return None;
    }
    let bytes = unsafe { slice::from_raw_parts(data as *const u8, len as usize) };
    // Xclbin::from_file mmap's a path; here we've got bytes already. We
    // write them to a tempfile since Xclbin's public API uses mmap. This
    // is fine for a one-shot classification call: the file lives only
    // for the duration of this function.
    let mut tmp = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return None,
    };
    use std::io::Write;
    if tmp.write_all(bytes).is_err() {
        return None;
    }
    let path = tmp.path().to_path_buf();
    let xclbin = Xclbin::from_file(&path).ok()?;
    let aie = xclbin.aie_partition()?;
    let partition = AiePartition::parse(aie.data).ok()?;
    let mut ops: Vec<DeviceOp> = Vec::new();
    for pdi in partition.pdis() {
        let off = match find_cdo_offset(pdi.pdi_image) {
            Some(o) => o,
            None => continue,
        };
        let cdo = match Cdo::parse(&pdi.pdi_image[off..]) {
            Ok(c) => c,
            Err(_) => continue,
        };
        for raw in cdo.commands() {
            for op in lower(&raw) {
                ops.push(op);
            }
        }
    }
    Some(StreamSwitchTopology::from_device_ops(ops))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_insts_returns_error_code() {
        let rc = unsafe {
            xdna_emu_classify_kernargs(std::ptr::null(), 0, std::ptr::null(), 0, std::ptr::null_mut(), 0)
        };
        assert_eq!(rc, -1);
    }

    #[test]
    fn unparseable_insts_returns_error_code() {
        let bogus = [0u8; 16];
        let rc = unsafe {
            xdna_emu_classify_kernargs(
                std::ptr::null(),
                0,
                bogus.as_ptr() as *const c_void,
                bogus.len() as u64,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(rc, -2);
    }

    #[test]
    fn real_binary_roundtrip() {
        // add_one_cpp_aiecc is the smallest bridge-test binary; if it
        // parses we know the pipeline works end-to-end.
        let insts = match std::fs::read(
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_cpp_aiecc/insts.bin",
        ) {
            Ok(d) => d,
            Err(_) => return,
        };
        let rc_query = unsafe {
            xdna_emu_classify_kernargs(
                std::ptr::null(),
                0,
                insts.as_ptr() as *const c_void,
                insts.len() as u64,
                std::ptr::null_mut(),
                0,
            )
        };
        assert!(rc_query >= 0, "query returned {}", rc_query);
        let mut out =
            vec![XdnaEmuKernargRole { arg_idx: 0, role: 0, _pad: [0; 2], bd_reg_addr: 0 }; rc_query as usize];
        let rc_fill = unsafe {
            xdna_emu_classify_kernargs(
                std::ptr::null(),
                0,
                insts.as_ptr() as *const c_void,
                insts.len() as u64,
                out.as_mut_ptr(),
                out.len() as u64,
            )
        };
        assert_eq!(rc_fill, rc_query);
        assert!(out.iter().all(|r| r.role != XdnaEmuKernargRoleCode::Ctrlpkt as u8));
    }
}
