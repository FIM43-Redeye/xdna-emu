//! xdna-emu: Open-source emulator for AMD XDNA NPUs

use std::env;
use std::collections::HashMap;
use xdna_emu::parser::{Xclbin, AiePartition, Cdo, AieElf};
use xdna_emu::parser::xclbin::SectionKind;
use xdna_emu::parser::cdo::{find_cdo_offset, CdoCommand};
use xdna_emu::device::{TileAddress, RegisterInfo, RegisterModule, DeviceState};
use xdna_emu::visual::EmulatorApp;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    // Check for GUI mode
    let gui_mode = args.iter().any(|a| a == "--gui" || a == "-g");
    let file_arg: Option<&str> = args.iter()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .map(|s| s.as_str());

    if gui_mode || args.len() < 2 {
        return run_gui(file_arg);
    }

    // Parse options for CLI mode
    let mut dump_state = false;
    let mut path = None;

    for arg in &args[1..] {
        if arg == "--dump-state" {
            dump_state = true;
        } else if !arg.starts_with('-') {
            path = Some(arg.as_str());
        }
    }

    let path = match path {
        Some(p) => p,
        None => {
            return run_gui(None);
        }
    };
    println!("Loading: {}", path);
    println!();

    // Detect format by extension or magic
    if path.ends_with(".elf") {
        return parse_elf(path);
    }

    let xclbin = Xclbin::from_file(path)?;
    xclbin.print_summary();

    // If there's AIE metadata, show a preview
    if let Some(metadata) = xclbin.aie_metadata() {
        println!();
        println!("AIE Metadata (first 500 chars):");
        println!("--------------------------------");
        let text = String::from_utf8_lossy(metadata.data());
        let preview: String = text.chars().take(500).collect();
        println!("{}", preview);
        if text.len() > 500 {
            println!("... ({} more bytes)", text.len() - 500);
        }
    }

    // Parse the AIE Partition section
    if let Some(partition_section) = xclbin.find_section(SectionKind::AiePartition) {
        println!();
        println!("AIE Partition Details");
        println!("=====================");
        println!("Section size: {} bytes", partition_section.size());

        match AiePartition::parse(partition_section.data()) {
            Ok(partition) => {
                if let Some(name) = partition.name() {
                    println!("Name: {}", name);
                }
                println!("Column width: {}", partition.column_width());

                let start_cols = partition.start_columns();
                if !start_cols.is_empty() {
                    println!("Start columns: {:?}", start_cols);
                }

                // Iterate over PDIs
                for (i, pdi) in partition.pdis().enumerate() {
                    println!();
                    println!("PDI #{}", i);
                    println!("  UUID: {}", pdi.uuid);
                    println!("  Type: {:?}", pdi.cdo_type);
                    println!("  Image size: {} bytes", pdi.pdi_image.len());

                    // Find and parse CDO from PDI image
                    // PDI is a container format; CDO is embedded inside
                    if let Some(cdo_offset) = find_cdo_offset(pdi.pdi_image) {
                        println!("  CDO found at offset 0x{:X} within PDI", cdo_offset);

                        match Cdo::parse(&pdi.pdi_image[cdo_offset..]) {
                            Ok(cdo) => {
                                println!();
                                cdo.print_summary();

                                // Analyze register usage
                                print_register_analysis(&cdo);

                                // If --dump-state, apply CDO and show state
                                if dump_state {
                                    println!();
                                    let mut state = DeviceState::new_npu1();
                                    if let Err(e) = state.apply_cdo(&cdo) {
                                        println!("Warning: Error applying CDO: {}", e);
                                    }
                                    state.print_summary();
                                } else {
                                    // Show all DMA_WRITE commands
                                    println!();
                                    println!("DMA_WRITE commands:");
                                    for (j, cmd) in cdo.commands().enumerate() {
                                        if let CdoCommand::DmaWrite { address, data } = &cmd {
                                            let addr = TileAddress::decode(*address);
                                            let module = addr.module();
                                            println!("  [{:2}] tile({},{}) offset=0x{:05X} {} {} bytes",
                                                j, addr.col, addr.row, addr.offset, module, data.len());
                                        }
                                    }

                                    // Show first few commands with register decode
                                    println!();
                                    println!("First 15 commands:");
                                    for (j, cmd) in cdo.commands().take(15).enumerate() {
                                        print_command(j, &cmd);
                                    }
                                }
                            }
                            Err(e) => {
                                println!("  Warning: Failed to parse CDO: {}", e);
                            }
                        }
                    } else {
                        println!("  Warning: No CDO magic found in PDI image");
                        // Show hex dump of first 64 bytes
                        println!("  First 64 bytes of PDI image:");
                        for (k, chunk) in pdi.pdi_image.chunks(16).take(4).enumerate() {
                            print!("    {:04X}: ", k * 16);
                            for b in chunk {
                                print!("{:02X} ", b);
                            }
                            println!();
                        }
                    }
                }
            }
            Err(e) => {
                println!("Warning: Failed to parse AIE partition: {}", e);
            }
        }
    }

    Ok(())
}

/// Parse and display an AIE ELF file
fn parse_elf(path: &str) -> anyhow::Result<()> {
    let data = std::fs::read(path)?;
    let elf = AieElf::parse(&data)?;

    elf.print_summary();

    Ok(())
}

/// Print register usage analysis for a CDO
fn print_register_analysis(cdo: &Cdo) {
    let mut module_counts: HashMap<RegisterModule, usize> = HashMap::new();
    let mut tile_counts: HashMap<(u8, u8), usize> = HashMap::new();
    let mut register_hits: HashMap<String, usize> = HashMap::new();

    for cmd in cdo.commands() {
        if let Some((col, row, offset)) = cmd.decode_aie_address() {
            let tile = TileAddress { col, row, offset };
            *module_counts.entry(tile.module()).or_insert(0) += 1;
            *tile_counts.entry((col, row)).or_insert(0) += 1;

            if let Some(info) = RegisterInfo::lookup_aie2(offset) {
                *register_hits.entry(info.name.to_string()).or_insert(0) += 1;
            }
        }
    }

    println!();
    println!("Register Analysis");
    println!("=================");

    // Show tiles accessed
    let mut tiles: Vec<_> = tile_counts.iter().collect();
    tiles.sort_by_key(|((c, r), _)| (*c, *r));
    print!("Tiles accessed: ");
    for ((col, row), count) in &tiles {
        print!("({},{}):{} ", col, row, count);
    }
    println!();

    // Show module breakdown
    println!();
    println!("By module:");
    let mut modules: Vec<_> = module_counts.iter().collect();
    modules.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    for (module, count) in modules {
        println!("  {:12} {:4} writes", format!("{}", module), count);
    }

    // Show top registers
    if !register_hits.is_empty() {
        println!();
        println!("Top registers:");
        let mut regs: Vec<_> = register_hits.iter().collect();
        regs.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (name, count) in regs.iter().take(10) {
            println!("  {:30} {:4}", name, count);
        }
    }
}

/// Print a single CDO command with register decode
fn print_command(idx: usize, cmd: &CdoCommand) {
    if let Some((col, row, offset)) = cmd.decode_aie_address() {
        let tile = TileAddress { col, row, offset };
        let reg_info = RegisterInfo::lookup_aie2(offset);

        let reg_name = reg_info
            .as_ref()
            .map(|r| r.name)
            .unwrap_or_else(|| match tile.module() {
                RegisterModule::Memory => "DATA",
                RegisterModule::ProgramMemory => "CODE",
                _ => "???",
            });

        // Format based on command type
        match cmd {
            CdoCommand::Write { value, .. } => {
                println!("  [{:2}] WRITE     tile({},{}) {} = 0x{:08X}",
                    idx, col, row, reg_name, value);
            }
            CdoCommand::MaskWrite { mask, value, .. } => {
                println!("  [{:2}] MASKWRITE tile({},{}) {} mask=0x{:X} val=0x{:X}",
                    idx, col, row, reg_name, mask, value);
            }
            CdoCommand::DmaWrite { data, .. } => {
                println!("  [{:2}] DMAWRITE  tile({},{}) {} {} bytes",
                    idx, col, row, reg_name, data.len());
            }
            _ => {
                println!("  [{:2}] {:?}", idx, cmd);
            }
        }
    } else {
        // Non-addressed commands (NOP, etc.)
        match cmd {
            CdoCommand::Nop { words } => {
                if *words > 0 {
                    println!("  [{:2}] NOP ({} words)", idx, words);
                } else {
                    println!("  [{:2}] NOP", idx);
                }
            }
            _ => {
                println!("  [{:2}] {:?}", idx, cmd);
            }
        }
    }
}

/// Run the GUI application.
fn run_gui(file_path: Option<&str>) -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("xdna-emu - AMD XDNA NPU Emulator"),
        ..Default::default()
    };

    eframe::run_native(
        "xdna-emu",
        options,
        Box::new(move |_cc| {
            let mut app = EmulatorApp::default();

            // Load file if provided
            if let Some(path) = file_path {
                if let Err(e) = app.load_xclbin(std::path::Path::new(path)) {
                    app.error_message = Some(format!("Failed to load: {}", e));
                }
            }

            Ok(Box::new(app))
        }),
    ).map_err(|e| anyhow::anyhow!("GUI error: {}", e))
}
