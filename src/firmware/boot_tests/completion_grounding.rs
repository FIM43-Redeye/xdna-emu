use super::*;

/// One-shot RE sweep for the array-backed word touched by the post-waiti path.
/// Diagnostic only: no firmware/peripheral behavior is changed.
#[test]
fn re_probe_goalive_completion_values() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        return;
    }
    let path = firmware_path().expect("firmware binary");
    let raw = std::fs::read(path).expect("read firmware");

    for value in [0, 1, 0x400, 0x800, 0xc00, 0xffff, u32::MAX] {
        let img = FirmwareImage::parse(&raw).expect("parse firmware");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.enable_host_mailbox();
        let report = proc.boot_to_idle(200_000);
        assert!(report.reached_idle && proc.cpu.pc == 0x5645);

        let mut device = crate::device::DeviceState::new_npu1();
        device.write_tile_register(0, 0, 0x5078c, value);
        proc.bus.attach_device(device);
        proc.cpu.interrupt |= 1;

        eprintln!("\n=== array[0x0405078c]={value:#010x} ===");
        let mut stop = String::from("budget");
        for n in 0..800u64 {
            let pc = proc.cpu.pc;
            let phys = proc
                .cpu
                .translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch)
                .expect("fetch translate");
            let bytes: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            let op = decode::decode(&bytes, pc).op;
            if (0xd800..0xd900).contains(&(pc & 0x00ff_ffff))
                || (0x8c00..0x8d00).contains(&(pc & 0x00ff_ffff))
                || matches!(op, Op::Call8 { .. } | Op::Callx8 { .. })
            {
                eprintln!(
                    "n={n:>3} pc={pc:#010x} {op:?} a2={:#x} a3={:#x} a4={:#x} a10={:#x}",
                    proc.cpu.regs.read_ar(2),
                    proc.cpu.regs.read_ar(3),
                    proc.cpu.regs.read_ar(4),
                    proc.cpu.regs.read_ar(10),
                );
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => {}
                other => {
                    stop = format!("{other:?}");
                    break;
                }
            }
        }
        eprintln!("stop={stop} pc={:#010x}", proc.cpu.pc);
    }
}
