use xdna_emu::parser::{Xclbin, AiePartition, Cdo};
use xdna_emu::parser::xclbin::SectionKind;
use xdna_emu::parser::cdo::find_cdo_offset;
use xdna_emu::interpreter::engine::{InterpreterEngine, EngineStatus};
use xdna_emu::interpreter::bundle::{detect_format, extract_slots};

fn main() {
    let path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_314_using_dma_op/aie.xclbin";
    let elf_path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_314_using_dma_op/aie_arch.mlir.prj/main_core_0_2.elf";

    let xclbin = Xclbin::from_file(path).unwrap();
    let section = xclbin.find_section(SectionKind::AiePartition).unwrap();
    let partition = AiePartition::parse(section.data()).unwrap();
    let pdi = partition.primary_pdi().unwrap();
    let cdo_offset = find_cdo_offset(pdi.pdi_image).unwrap();
    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..]).unwrap();

    let mut engine = InterpreterEngine::new_npu1();
    engine.device_mut().apply_cdo(&cdo).unwrap();
    engine.sync_cores_from_device();

    let elf_data = std::fs::read(elf_path).unwrap();
    let entry = engine.load_elf_bytes(0, 2, &elf_data).unwrap();
    println!("Entry: 0x{:04X}", entry);

    for i in 0..50 {
        engine.step();
        
        if engine.status() == EngineStatus::Error {
            println!("Error at cycle {}", i+1);
            if let Some(ctx) = engine.core_context(0, 2) {
                let pc = ctx.pc();
                println!("PC=0x{:04X}", pc);
                
                if let Some(tile) = engine.device().tile(0, 2) {
                    if let Some(pm) = tile.program_memory() {
                        let pc_byte = pc as usize;
                        print!("PM[0x{:04X}]: ", pc);
                        for b in &pm[pc_byte..pc_byte.min(pm.len()-16)+16] {
                            print!("{:02X} ", b);
                        }
                        println!();
                        
                        let format = detect_format(&pm[pc_byte..]);
                        println!("Format: {:?} ({} bytes)", format, format.size_bytes());
                        
                        let extracted = extract_slots(&pm[pc_byte..]);
                        println!("Slots: {}", extracted.slots.len());
                        for slot in &extracted.slots {
                            println!("  {:?}: bits=0x{:X}", slot.slot_type, slot.bits);
                        }
                    }
                }
            }
            if let Some(bundle) = engine.core_last_bundle(0, 2) {
                println!("Last bundle:");
                for op in bundle.active_slots() {
                    println!("  {:?}: {:?}", op.slot, op.op);
                }
            }
            return;
        }
    }
    println!("Ran 50 cycles OK");
}
