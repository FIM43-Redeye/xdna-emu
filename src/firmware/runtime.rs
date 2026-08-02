use super::{FirmwareProcessor, IdleReport};
use crate::interpreter::engine::{EngineStatus, InterpreterEngine};

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

fn phoenix_shim_tct_word(physical_col: u8, actor_id: u8, controller_id: u8) -> u32 {
    // The signed firmware matches this parity-free wait key. Row 0 occupies a
    // zero-valued field; Phoenix records use type 6.
    u32::from(physical_col) << 21 | 6 << 12 | u32::from(actor_id) << 8 | u32::from(controller_id)
}

fn phoenix_shim_dma_actor(channel: usize, s2mm_channels: usize) -> Option<u8> {
    // Derived from AIENpuToCert.cpp's shim DMA channel-to-actor tables.
    const S2MM: [u8; 4] = [0, 2, 3, 4];
    const MM2S: [u8; 8] = [6, 7, 8, 9, 10, 11, 12, 13];
    if channel < s2mm_channels {
        S2MM.get(channel).copied()
    } else {
        MM2S.get(channel - s2mm_channels).copied()
    }
}

fn publish_phoenix_shim_tct(firmware: &mut FirmwareProcessor, engine: &mut InterpreterEngine) -> bool {
    let completion = {
        let device = engine.device_mut();
        let mut completion = None;
        'columns: for physical_col in 0..device.array.cols() {
            let Some(dma) = device.array.dma_engine_mut(physical_col, 0) else {
                continue;
            };
            let s2mm_channels = dma.s2mm_channel_count();
            for channel in 0..dma.channel_count() {
                let Some(actor_id) = phoenix_shim_dma_actor(channel, s2mm_channels) else {
                    continue;
                };
                if let Some(token) = dma.pop_task_token_for_channel(channel as u8) {
                    completion = Some((physical_col, actor_id, token.controller_id));
                    break 'columns;
                }
            }
        }
        completion
    };
    if let Some((physical_col, actor_id, controller_id)) = completion {
        let word = phoenix_shim_tct_word(physical_col, actor_id, controller_id);
        let completion_lane =
            usize::from(physical_col.checked_sub(1).expect("Phoenix physical column 0 has no shim tile"));
        firmware.bus.publish_tct_word(completion_lane, word);
        return true;
    }
    false
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
            let published_tct = publish_phoenix_shim_tct(firmware, engine);
            let engine_stop = match engine.status() {
                EngineStatus::Stalled => Some(RuntimePumpStop::EngineStalled),
                EngineStatus::Error => Some(RuntimePumpStop::EngineError),
                EngineStatus::Halted if boundary.reached_idle && !published_tct => {
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
    use super::{phoenix_shim_tct_word, publish_phoenix_shim_tct, pump_runtime, RuntimePumpStop};
    use crate::firmware::host_mailbox::HostMailbox;
    use crate::firmware::xtensa::interp::{mapped_cpu, WaitReason};
    use crate::firmware::{Bus, FirmwareProcessor};
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
    fn phoenix_shim_s2mm0_tct_matches_frozen_aiesim_record() {
        assert_eq!(phoenix_shim_tct_word(1, 0, 15), 0x0020_600f);
    }

    #[test]
    fn phoenix_shim_tct_matches_signed_firmware_wait_key_without_transport_parity() {
        assert_eq!(phoenix_shim_tct_word(1, 0, 14), 0x0020_600e);
    }

    #[test]
    fn publishes_relocated_shim_channel_completions() {
        let mut firmware = processor(vec![]);
        let mut engine = InterpreterEngine::new_npu1();
        engine.device_mut().array.dma_engine_mut(3, 0).unwrap().issue_task_token(2, 0);

        assert!(publish_phoenix_shim_tct(&mut firmware, &mut engine));
        assert_eq!(firmware.bus.data_load32(0xbd00_0000), 0x0060_6600);

        engine.device_mut().array.dma_engine_mut(4, 0).unwrap().issue_task_token(1, 7);
        assert!(publish_phoenix_shim_tct(&mut firmware, &mut engine));
        assert_eq!(firmware.bus.data_load32(0xbd80_0000), 0x0080_6207);
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
