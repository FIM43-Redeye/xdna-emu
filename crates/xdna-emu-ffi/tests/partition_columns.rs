use xdna_emu::*;

#[test]
fn npu1_handle_reports_the_drivers_first_partition_column() {
    let handle = unsafe { xdna_emu_create() };
    assert!(!handle.is_null());
    assert_eq!(unsafe { xdna_emu_get_partition_column_start(handle) }, 1);
    unsafe { xdna_emu_destroy(handle) };
}
