//! Real `BridgeAbi`: dlopen `libxdna_aiesim_bridge.so` and bind its C ABI.

use super::abi::{BridgeAbi, BridgeHalt, BridgeStatus};
use libloading::{Library, Symbol};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

type CreateFn = unsafe extern "C" fn(*const c_char, *const c_char) -> *mut c_void;
type LoadCdoFn = unsafe extern "C" fn(*mut c_void, *const u8, usize) -> c_int;
type ExecNpuFn = unsafe extern "C" fn(*mut c_void, *const u8, usize) -> c_int;
type WriteGmFn = unsafe extern "C" fn(*mut c_void, u64, *const u8, usize) -> c_int;
type ReadGmFn = unsafe extern "C" fn(*mut c_void, u64, *mut u8, usize) -> c_int;
type RunFn = unsafe extern "C" fn(*mut c_void, u64, *mut u64) -> c_int;
type ReadRegFn = unsafe extern "C" fn(*mut c_void, u64) -> u32;
type ResetFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type SetStartColFn = unsafe extern "C" fn(*mut c_void, u32) -> c_int;
type AddHostBufferFn = unsafe extern "C" fn(*mut c_void, u64, usize) -> c_int;
type ClearHostBuffersFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type DestroyFn = unsafe extern "C" fn(*mut c_void);

pub(crate) struct DlopenBridge {
    _lib: Library, // kept alive for the symbols' lifetime
    handle: *mut c_void,
    // resolved fn pointers (stored as raw to avoid lifetime gymnastics)
    load_cdo: LoadCdoFn,
    exec_npu: ExecNpuFn,
    write_gm: WriteGmFn,
    // read_gm / read_reg: tier-2 read-back, called from Part II (kept bound now).
    #[allow(dead_code)]
    read_gm: ReadGmFn,
    run: RunFn,
    #[allow(dead_code)]
    read_reg: ReadRegFn,
    reset: ResetFn,
    set_start_col: SetStartColFn,
    add_host_buffer: AddHostBufferFn,
    clear_host_buffers: ClearHostBuffersFn,
    destroy: DestroyFn,
}

impl DlopenBridge {
    /// Open the bridge and construct the cluster. Returns a clear error if the
    /// .so is missing (feature on but bridge not built) or a symbol is absent.
    pub(crate) fn open(arch: &str, device_json: &str) -> Result<Self, String> {
        let path =
            std::env::var("XDNA_AIESIM_BRIDGE").unwrap_or_else(|_| "libxdna_aiesim_bridge.so".to_string());
        // RTLD_GLOBAL is REQUIRED, not cosmetic: the bridge defines host globals
        // (sc_stop_at_end_of_main, plio_complete) that the cluster .so -- dlopened
        // later by the bridge -- resolves from the global symbol scope. The
        // default Library::new loads RTLD_LOCAL, hiding them, so the cluster
        // fails with "undefined symbol: sc_stop_at_end_of_main". Open with
        // RTLD_NOW | RTLD_GLOBAL via the unix-specific API.
        // SAFETY: loading a trusted, locally-built bridge.
        use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};
        let unix_lib = unsafe { UnixLibrary::open(Some(&path), RTLD_NOW | RTLD_GLOBAL) }.map_err(|e| {
            format!("aiesim: cannot load {path}: {e} (build it with scripts/build-aiesim-bridge.sh)")
        })?;
        let lib: Library = unix_lib.into();
        unsafe {
            let create: Symbol<CreateFn> = lib
                .get(b"aiesim_create\0")
                .map_err(|e| format!("aiesim: missing aiesim_create: {e}"))?;
            let c_arch = CString::new(arch).unwrap();
            let c_json = CString::new(device_json).unwrap();
            let handle = create(c_arch.as_ptr(), c_json.as_ptr());
            if handle.is_null() {
                return Err("aiesim: aiesim_create returned null".to_string());
            }
            // Resolve the rest; copy the fn pointers out so we don't hold Symbols.
            let load_cdo = *lib.get::<LoadCdoFn>(b"aiesim_load_cdo\0").map_err(sym_err)?;
            let exec_npu = *lib.get::<ExecNpuFn>(b"aiesim_exec_npu\0").map_err(sym_err)?;
            let write_gm = *lib.get::<WriteGmFn>(b"aiesim_write_gm\0").map_err(sym_err)?;
            let read_gm = *lib.get::<ReadGmFn>(b"aiesim_read_gm\0").map_err(sym_err)?;
            let run = *lib.get::<RunFn>(b"aiesim_run\0").map_err(sym_err)?;
            let read_reg = *lib.get::<ReadRegFn>(b"aiesim_read_reg\0").map_err(sym_err)?;
            let reset = *lib.get::<ResetFn>(b"aiesim_reset\0").map_err(sym_err)?;
            let set_start_col = *lib.get::<SetStartColFn>(b"aiesim_set_start_col\0").map_err(sym_err)?;
            let add_host_buffer =
                *lib.get::<AddHostBufferFn>(b"aiesim_add_host_buffer\0").map_err(sym_err)?;
            let clear_host_buffers =
                *lib.get::<ClearHostBuffersFn>(b"aiesim_clear_host_buffers\0").map_err(sym_err)?;
            let destroy = *lib.get::<DestroyFn>(b"aiesim_destroy\0").map_err(sym_err)?;
            Ok(Self {
                _lib: lib,
                handle,
                load_cdo,
                exec_npu,
                write_gm,
                read_gm,
                run,
                read_reg,
                reset,
                set_start_col,
                add_host_buffer,
                clear_host_buffers,
                destroy,
            })
        }
    }
}

fn sym_err(e: libloading::Error) -> String {
    format!("aiesim: missing bridge symbol: {e}")
}

impl Drop for DlopenBridge {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.handle) };
    }
}

impl BridgeAbi for DlopenBridge {
    fn create(&mut self, _arch: &str, _device_json: &str) -> BridgeStatus {
        // Construction happens in `open`; this is a no-op so the trait shape
        // matches the mock. (Always Ok once `open` succeeded.)
        BridgeStatus::Ok
    }
    fn load_cdo(&mut self, ops: &[u8]) -> BridgeStatus {
        let rc = unsafe { (self.load_cdo)(self.handle, ops.as_ptr(), ops.len()) };
        if rc == 0 {
            BridgeStatus::Ok
        } else {
            BridgeStatus::Error
        }
    }
    fn exec_npu(&mut self, ops: &[u8]) -> BridgeStatus {
        let rc = unsafe { (self.exec_npu)(self.handle, ops.as_ptr(), ops.len()) };
        if rc == 0 {
            BridgeStatus::Ok
        } else {
            BridgeStatus::Error
        }
    }
    fn add_host_buffer(&mut self, addr: u64, size: usize) -> BridgeStatus {
        let rc = unsafe { (self.add_host_buffer)(self.handle, addr, size) };
        if rc == 0 {
            BridgeStatus::Ok
        } else {
            BridgeStatus::Error
        }
    }
    fn clear_host_buffers(&mut self) -> BridgeStatus {
        let rc = unsafe { (self.clear_host_buffers)(self.handle) };
        if rc == 0 {
            BridgeStatus::Ok
        } else {
            BridgeStatus::Error
        }
    }
    fn write_gm(&mut self, addr: u64, data: &[u8]) -> BridgeStatus {
        let rc = unsafe { (self.write_gm)(self.handle, addr, data.as_ptr(), data.len()) };
        if rc == 0 {
            BridgeStatus::Ok
        } else {
            BridgeStatus::Error
        }
    }
    fn read_gm(&mut self, addr: u64, out: &mut [u8]) -> BridgeStatus {
        let rc = unsafe { (self.read_gm)(self.handle, addr, out.as_mut_ptr(), out.len()) };
        if rc == 0 {
            BridgeStatus::Ok
        } else {
            BridgeStatus::Error
        }
    }
    fn run(&mut self, budget: u64, cycles_out: &mut u64) -> BridgeHalt {
        let rc = unsafe { (self.run)(self.handle, budget, cycles_out as *mut u64) };
        match rc {
            0 => BridgeHalt::Completed,
            1 => BridgeHalt::Budget,
            _ => BridgeHalt::Error,
        }
    }
    fn read_reg(&mut self, addr: u64) -> u32 {
        unsafe { (self.read_reg)(self.handle, addr) }
    }
    fn reset(&mut self) -> BridgeStatus {
        let rc = unsafe { (self.reset)(self.handle) };
        if rc == 0 {
            BridgeStatus::Ok
        } else {
            BridgeStatus::Error
        }
    }
    fn set_start_col(&mut self, start_col: u8) -> BridgeStatus {
        let rc = unsafe { (self.set_start_col)(self.handle, start_col as u32) };
        if rc == 0 {
            BridgeStatus::Ok
        } else {
            BridgeStatus::Error
        }
    }
}

// SAFETY: the bridge is a process singleton driven from one thread at a time
// (the plugin serializes handle access; the SystemC kernel is global). The raw
// `handle` pointer is never shared across threads concurrently.
unsafe impl Send for DlopenBridge {}
