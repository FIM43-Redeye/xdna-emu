use super::{FirmwareProcessor, IdleReport};
use crate::interpreter::engine::{EngineStatus, InterpreterEngine};

// AM020 maps L2 output selectors 0..3 to device NPI interrupts 4..7.
// Pinned Phoenix firmware 5.5.391 maps selector 1 / NPI 5 -- aie-rt's
// XAIE_ERROR_NPI_INTR_ID -- to management-controller source 56.
const PHOENIX_AIE_NOC_SOURCE_OFFSET: u8 = 55;

/// Observable reason the functional firmware/array pump stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePumpStop {
    ResponseCompleted,
    ArrayIdleFirmwareWaiting,
    UnresolvedFirmwarePoll { address: u32 },
    UnknownFirmwareInstruction { pc: u32, word: u32 },
    EngineStalled,
    EngineError,
    NoProgressExhausted,
}

#[derive(Debug)]
pub struct RuntimePumpReport {
    pub stop: RuntimePumpStop,
    pub iterations: u64,
    pub firmware_instructions: u64,
    pub aie_cycles: u64,
    pub last_firmware: Option<IdleReport>,
}

impl FirmwareProcessor {
    pub(super) fn run_to_boundary_with_engine(
        &mut self,
        engine: &mut InterpreterEngine,
        max_instructions: u64,
    ) -> IdleReport {
        let (device, host_memory) = engine.device_and_host_memory();
        self.boot_to_idle_on(max_instructions, |cpu, bus| {
            cpu.step_with_device_and_host_memory(bus, device, host_memory)
        })
    }
}

fn advance_phoenix_tct_publication(firmware: &mut FirmwareProcessor, engine: &mut InterpreterEngine) -> bool {
    let (landed, pending) = {
        let array = &mut engine.device_mut().array;
        array.queue_phoenix_tct_packets();
        let landed = array.drain_phoenix_tct_egress();
        let pending = array.phoenix_tct_packets_in_flight() != 0;
        (landed, pending)
    };
    let published = !landed.is_empty();
    for (physical_col, transport_word) in landed {
        let completion_lane =
            usize::from(physical_col.checked_sub(1).expect("Phoenix physical column 0 has no shim tile"));
        let firmware_key = transport_word & !(1 << xdna_archspec::aie2::packet::PARITY_SHIFT);
        firmware.bus.publish_tct_word(completion_lane, firmware_key);
    }
    pending || published
}

fn advance_phoenix_l2_error_publication(
    firmware: &mut FirmwareProcessor,
    engine: &InterpreterEngine,
) -> bool {
    let mut pending = false;
    for col in 0..engine.device().array.cols() {
        let Some(l2) = engine.device().array.get(col, 0).and_then(|tile| tile.l2_irq.as_ref()) else {
            continue;
        };
        if l2.pending_host_interrupt() {
            pending = true;
            firmware
                .bus
                .assert_management_source(PHOENIX_AIE_NOC_SOURCE_OFFSET + l2.noc_interrupt());
        }
    }
    pending
}

/// Functionally interleave firmware boundaries with single AIE cycles.
///
/// The predicate observes a real response; it cannot mutate either side or
/// manufacture a completion edge.
pub fn pump_runtime(
    firmware: &mut FirmwareProcessor,
    engine: &mut InterpreterEngine,
    max_iterations: u64,
    firmware_budget: u64,
    mut response_complete: impl FnMut(&FirmwareProcessor, &InterpreterEngine) -> bool,
) -> RuntimePumpReport {
    let start_cycles = engine.total_cycles();
    let mut firmware_instructions = 0;
    let mut iterations = 0;
    let mut last_firmware = None;

    let stop = 'pump: {
        if response_complete(firmware, engine) {
            break 'pump RuntimePumpStop::ResponseCompleted;
        }
        match engine.status() {
            EngineStatus::Stalled => break 'pump RuntimePumpStop::EngineStalled,
            EngineStatus::Error => break 'pump RuntimePumpStop::EngineError,
            _ => {}
        }

        for iteration in 1..=max_iterations {
            iterations = iteration;
            let boundary = firmware.run_to_boundary_with_engine(engine, firmware_budget);
            firmware_instructions += boundary.instrs_executed;

            let boundary_stop = if response_complete(firmware, engine) {
                Some(RuntimePumpStop::ResponseCompleted)
            } else if let Some(address) = boundary.unresolved_spin {
                Some(RuntimePumpStop::UnresolvedFirmwarePoll { address })
            } else if let Some((pc, word)) = boundary.unknown_op {
                Some(RuntimePumpStop::UnknownFirmwareInstruction { pc, word })
            } else {
                None
            };
            if let Some(stop) = boundary_stop {
                last_firmware = Some(boundary);
                break 'pump stop;
            }

            engine.force_running();
            engine.step();
            let tct_work = advance_phoenix_tct_publication(firmware, engine);
            let error_work = advance_phoenix_l2_error_publication(firmware, engine);
            let engine_stop = match engine.status() {
                EngineStatus::Stalled => Some(RuntimePumpStop::EngineStalled),
                EngineStatus::Error => Some(RuntimePumpStop::EngineError),
                EngineStatus::Halted if boundary.reached_idle && !tct_work && !error_work => {
                    Some(RuntimePumpStop::ArrayIdleFirmwareWaiting)
                }
                _ => None,
            };
            last_firmware = Some(boundary);
            if let Some(stop) = engine_stop {
                break 'pump stop;
            }
        }

        RuntimePumpStop::NoProgressExhausted
    };

    RuntimePumpReport {
        stop,
        iterations,
        firmware_instructions,
        aie_cycles: engine.total_cycles() - start_cycles,
        last_firmware,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        advance_phoenix_l2_error_publication, advance_phoenix_tct_publication, pump_runtime, RuntimePumpStop,
    };
    use crate::firmware::host_mailbox::HostMailbox;
    use crate::firmware::xtensa::interp::{mapped_cpu, WaitReason};
    use crate::firmware::{Bus, FirmwareProcessor, SysStub};
    use crate::interpreter::engine::{EngineStatus, InterpreterEngine};
    use std::collections::HashMap;

    fn processor(rom: Vec<u8>) -> FirmwareProcessor {
        FirmwareProcessor {
            cpu: mapped_cpu(0),
            bus: Bus::new(rom),
            entry: 0,
            symbols: HashMap::new(),
            host_mailbox: HostMailbox::new(),
        }
    }

    #[test]
    fn routed_shim_tct_lands_as_parity_free_firmware_key() {
        use xdna_archspec::aie2::stream_switch::shim;

        let mut firmware = processor(vec![]);
        let mut engine = InterpreterEngine::new_npu1();
        engine.device_mut().array.clock_mut().ungate_all();
        let switch = &mut engine.device_mut().array.tile_mut(3, 0).stream_switch;
        let ctrl = switch.tile_ctrl_slave_port().unwrap();
        switch.slaves[ctrl].packet_enable = true;
        switch.configure_slave_slot(ctrl, 0, 0x1f << 16 | 1 << 8);
        switch.configure_master_packet(shim::SOUTH_MASTER_START as usize, 1 << 31 | 1 << 30 | 1 << 3);
        engine.device_mut().array.dma_engine_mut(3, 0).unwrap().issue_task_token(2, 0);

        assert!(advance_phoenix_tct_publication(&mut firmware, &mut engine));
        engine.force_running();
        engine.step();
        assert!(advance_phoenix_tct_publication(&mut firmware, &mut engine));
        assert_eq!(firmware.bus.data_load32(0xbd00_0000), 0x0060_6600);
    }

    #[test]
    fn pending_l2_error_remains_runtime_work_until_controller_accepts_it() {
        let mut firmware = processor(vec![]);
        let mut engine = InterpreterEngine::new_npu1();
        let l2 = engine.device_mut().array.tile_mut(1, 0).l2_irq.as_mut().unwrap();
        l2.write_enable(1);
        l2.write_register(crate::device::interrupts::L2_REG_INTERRUPT, 1);
        l2.signal_interrupt(0);

        assert!(
            advance_phoenix_l2_error_publication(&mut firmware, &engine),
            "a masked management source must not hide the pending L2 level",
        );

        firmware.bus.data_store32(0x2720_0304, 1 << 24);
        assert!(advance_phoenix_l2_error_publication(&mut firmware, &engine));
        assert_eq!(firmware.bus.data_load32(0x2720_03c4), 56);
    }

    #[test]
    fn waiting_firmware_yields_one_cycle_without_cloning_engine_state() {
        let mut firmware = processor(vec![0x00, 0x70, 0x00]); // waiti 0
        let mut engine = InterpreterEngine::new_npu1();
        let device = engine.device() as *const _;
        let host_memory = engine.host_memory() as *const _;

        let report = pump_runtime(&mut firmware, &mut engine, 1, 8, |_, _| false);

        assert_eq!(report.stop, RuntimePumpStop::ArrayIdleFirmwareWaiting);
        assert_eq!(report.iterations, 1);
        assert_eq!(report.firmware_instructions, 1);
        assert_eq!(report.aie_cycles, 1);
        assert_eq!(report.last_firmware.unwrap().wait_reason, Some(WaitReason::Waiti));
        assert_eq!(engine.device() as *const _, device);
        assert_eq!(engine.host_memory() as *const _, host_memory);
    }

    fn registered_host_poll() -> (FirmwareProcessor, InterpreterEngine) {
        const FLAG: u32 = 0x0400_9010;
        let mut firmware = processor(vec![0x48, 0x45, 0xf0, 0x20, 0x00]); // l32i.n; nop
        let page = FLAG & 0xffff_f000;
        firmware.cpu.mmu.write_tlb(true, page | 0x3, page | 0);
        firmware.cpu.regs.write_ar(5, FLAG - 16);
        firmware.cpu.regs.lbeg = 0;
        firmware.cpu.regs.lend = 2;
        firmware.cpu.regs.lcount = u32::MAX;

        let mut engine = InterpreterEngine::new_npu1();
        engine
            .host_memory_mut()
            .allocate_region("firmware poll flag", u64::from(FLAG), 4)
            .unwrap();
        (firmware, engine)
    }

    #[test]
    fn unchanged_registered_host_poll_yields_one_array_cycle() {
        const FLAG: u32 = 0x0400_9010;
        let (mut firmware, mut engine) = registered_host_poll();
        let budget = u64::from(SysStub::new_threshold()) + 1;

        let report = pump_runtime(&mut firmware, &mut engine, 1, budget, |_, _| false);

        assert_eq!(report.stop, RuntimePumpStop::ArrayIdleFirmwareWaiting);
        assert_eq!(report.iterations, 1);
        assert_eq!(report.firmware_instructions, budget);
        assert_eq!(report.aie_cycles, 1);
        assert_eq!(report.last_firmware.unwrap().wait_reason, Some(WaitReason::PollSpin { addr: FLAG }),);
    }

    #[test]
    fn changed_registered_host_value_starts_a_new_poll_streak() {
        const FLAG: u32 = 0x0400_9010;
        let (mut firmware, mut engine) = registered_host_poll();
        let threshold = u64::from(SysStub::new_threshold());

        let first = firmware.run_to_boundary_with_engine(&mut engine, threshold);
        assert!(!first.reached_idle);
        engine.host_memory_mut().write_u32(u64::from(FLAG), 1);

        let changed = firmware.run_to_boundary_with_engine(&mut engine, threshold + 1);

        assert!(changed.reached_idle);
        assert_eq!(changed.instrs_executed, threshold + 1);
        assert_eq!(changed.wait_reason, Some(WaitReason::PollSpin { addr: FLAG }));

        firmware.cpu.regs.lcount = 0;
        firmware.cpu.pc = 2;
        let following_nop = firmware.run_to_boundary_with_engine(&mut engine, 1);
        assert!(!following_nop.reached_idle, "a consumed poll edge must not re-yield without another load");
    }

    #[test]
    fn modeled_interrupt_resumes_waiting_firmware() {
        const VECTOR: usize = 0x200;
        let mut rom = vec![0; VECTOR + 6];
        rom[0..3].copy_from_slice(&[0x00, 0x70, 0x00]); // waiti 0
        rom[3..6].copy_from_slice(&[0xf0, 0x20, 0x00]); // nop
        rom[6..9].copy_from_slice(&[0x00, 0x70, 0x00]); // waiti 0
        rom[VECTOR..VECTOR + 3].copy_from_slice(&[0x20, 0xe3, 0x13]); // wsr.intclear a2
        rom[VECTOR + 3..VECTOR + 6].copy_from_slice(&[0x00, 0x30, 0x00]); // rfe

        let mut firmware = processor(rom);
        firmware.cpu.intenable = 1;
        firmware.cpu.regs.write_ar(2, 1);
        firmware.bus.data_store32(0x2720_0304, 1 << 14);
        let mut engine = InterpreterEngine::new_npu1();

        let idle = firmware.run_to_boundary_with_engine(&mut engine, 8);
        assert!(idle.reached_idle);
        assert_eq!(firmware.cpu.pc, 3);
        assert!(firmware.bus.assert_management_source(46));

        let report = pump_runtime(&mut firmware, &mut engine, 1, 16, |firmware, _| firmware.cpu.pc == 9);

        assert_eq!(report.stop, RuntimePumpStop::ResponseCompleted);
        assert_eq!(firmware.cpu.pc, 9);
        assert_eq!(firmware.cpu.interrupt, 0);
    }

    #[test]
    fn busy_firmware_stops_at_the_outer_budget() {
        let mut firmware = processor(vec![0x06, 0xff, 0xff]); // j 0
        let mut engine = InterpreterEngine::new_npu1();

        let report = pump_runtime(&mut firmware, &mut engine, 2, 4, |_, _| false);

        assert_eq!(report.stop, RuntimePumpStop::NoProgressExhausted);
        assert_eq!(report.iterations, 2);
        assert_eq!(report.firmware_instructions, 8);
        assert_eq!(report.aie_cycles, 2);
    }

    #[test]
    fn engine_error_is_not_pumped_past() {
        let mut firmware = processor(vec![0x06, 0xff, 0xff]); // j 0
        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();
        engine.enable_core(1, 2);
        engine.device_mut().tile_mut(1, 2).unwrap().write_program(0, &[0xff; 16]);

        let report = pump_runtime(&mut firmware, &mut engine, 2, 1, |_, _| false);

        assert_eq!(report.stop, RuntimePumpStop::EngineError);
        assert_eq!(engine.status(), EngineStatus::Error);
    }
}
