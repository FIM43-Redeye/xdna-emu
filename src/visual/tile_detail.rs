//! Tile detail panel showing internals of the selected tile.

use eframe::egui;

use crate::device::tile::TileType;

use super::app::EmulatorApp;
use super::memory_view;

/// Show the tile detail panel.
pub fn show_tile_detail(ui: &mut egui::Ui, app: &mut EmulatorApp) {
    let sel = match app.selected_tile {
        Some(s) => s,
        None => {
            ui.centered_and_justified(|ui| {
                ui.label("Select a tile to view details");
            });
            return;
        }
    };

    // Get tile type first (before borrowing for scroll area)
    let tile_info = app.engine.device().array.get(sel.col, sel.row).map(|t| {
        (t.tile_type, t.is_compute())
    });

    let (tile_type, is_compute) = match tile_info {
        Some((tt, ic)) => (tt, ic),
        None => {
            ui.label("Invalid tile selected");
            return;
        }
    };

    // Header
    let type_name = match tile_type {
        TileType::Shim => "Shim Tile",
        TileType::MemTile => "Memory Tile",
        TileType::Compute => "Compute Tile",
    };

    ui.heading(format!("{} ({}, {})", type_name, sel.col, sel.row));
    ui.separator();

    // Cache show flags before scroll area
    let want_registers = app.show_registers;
    let want_locks = app.show_locks;
    let want_dma = app.show_dma;
    let want_memory = app.show_memory;

    // Use scroll area for content
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // Core state (compute tiles only)
            if is_compute {
                show_core_state(ui, app, sel.col, sel.row);
            }

            // Registers (compute tiles only)
            if is_compute && want_registers {
                show_registers_panel(ui, app, sel.col, sel.row);
            }

            // Locks
            if want_locks {
                show_locks_panel(ui, app, sel.col, sel.row);
            }

            // DMA
            if want_dma {
                show_dma_panel(ui, app, sel.col, sel.row);
            }

            // Memory view
            if want_memory {
                show_memory_section(ui, app, sel.col, sel.row);
            }
        });
}

/// Show core state for a compute tile.
fn show_core_state(ui: &mut egui::Ui, app: &EmulatorApp, col: u8, row: u8) {
    egui::CollapsingHeader::new("Core State")
        .default_open(true)
        .show(ui, |ui| {
            if let Some(tile) = app.engine.device().array.get(col, row) {
                egui::Grid::new("core_state_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Enabled:");
                        ui.label(if tile.core.enabled { "Yes" } else { "No" });
                        ui.end_row();

                        ui.label("PC:");
                        ui.label(format!("0x{:04X}", tile.core.pc));
                        ui.end_row();

                        ui.label("SP:");
                        ui.label(format!("0x{:04X}", tile.core.sp));
                        ui.end_row();

                        ui.label("LR:");
                        ui.label(format!("0x{:04X}", tile.core.lr));
                        ui.end_row();

                        ui.label("Control:");
                        ui.label(format!("0x{:08X}", tile.core.control));
                        ui.end_row();

                        ui.label("Status:");
                        ui.label(format!("0x{:08X}", tile.core.status));
                        ui.end_row();
                    });

                // Interpreter state
                if let Some(ctx) = app.engine.core_context(col as usize, row as usize) {
                    ui.separator();
                    ui.label(format!("PC: 0x{:04X}", ctx.pc()));
                }
                if let Some(status) = app.engine.core_status(col as usize, row as usize) {
                    ui.label(format!("Status: {:?}", status));
                }
            }
        });
}

/// Show scalar registers for a compute tile.
fn show_registers_panel(ui: &mut egui::Ui, app: &EmulatorApp, col: u8, row: u8) {
    egui::CollapsingHeader::new("Registers")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(ctx) = app.engine.core_context(col as usize, row as usize) {
                egui::Grid::new("regs_grid")
                    .num_columns(4)
                    .spacing([10.0, 2.0])
                    .show(ui, |ui| {
                        for i in 0..16 {
                            ui.monospace(format!("r{:02}:", i));
                            ui.monospace(format!("0x{:08X}", ctx.scalar.read(i as u8)));

                            if i % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
            } else {
                ui.label("No context for this tile");
            }
        });
}

/// Show lock states.
fn show_locks_panel(ui: &mut egui::Ui, app: &EmulatorApp, col: u8, row: u8) {
    egui::CollapsingHeader::new("Locks")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(tile) = app.engine.device().array.get(col, row) {
                // Count non-zero locks
                let active_locks: Vec<_> = tile
                    .locks
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.value != 0)
                    .collect();

                if active_locks.is_empty() {
                    ui.label("All locks at 0");
                } else {
                    egui::Grid::new("locks_grid")
                        .num_columns(4)
                        .spacing([10.0, 2.0])
                        .show(ui, |ui| {
                            for (i, lock) in &active_locks {
                                ui.monospace(format!("L{:02}:", i));
                                ui.monospace(format!("{}", lock.value));

                                if active_locks.iter().position(|(idx, _)| idx == i).unwrap() % 2 == 1 {
                                    ui.end_row();
                                }
                            }
                        });
                }

                // Show all 64 locks in compact view
                ui.separator();
                ui.label("All locks (compact):");
                let mut lock_str = String::new();
                for (i, lock) in tile.locks.iter().enumerate() {
                    if i % 16 == 0 && i > 0 {
                        lock_str.push('\n');
                    }
                    lock_str.push_str(&format!("{:X}", lock.value.min(15)));
                }
                ui.monospace(lock_str);
            }
        });
}

/// Show DMA buffer descriptors and channels.
fn show_dma_panel(ui: &mut egui::Ui, app: &EmulatorApp, col: u8, row: u8) {
    egui::CollapsingHeader::new("DMA")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(tile) = app.engine.device().array.get(col, row) {
                // Channels
                ui.label("Channels:");
                egui::Grid::new("dma_channels_grid")
                    .num_columns(4)
                    .spacing([10.0, 2.0])
                    .show(ui, |ui| {
                        ui.label("Ch");
                        ui.label("Enabled");
                        ui.label("Running");
                        ui.label("BD");
                        ui.end_row();

                        for (i, ch) in tile.dma_channels.iter().enumerate() {
                            let name = match i {
                                0 => "S2MM_0",
                                1 => "S2MM_1",
                                2 => "MM2S_0",
                                3 => "MM2S_1",
                                _ => "???",
                            };
                            ui.label(name);
                            ui.label(if ch.is_enabled() { "Y" } else { "N" });
                            ui.label(if ch.running { "Y" } else { "N" });
                            ui.label(format!("{}", ch.current_bd));
                            ui.end_row();
                        }
                    });

                // Buffer descriptors
                ui.separator();
                ui.label("Buffer Descriptors:");

                let valid_bds: Vec<_> = tile
                    .dma_bds
                    .iter()
                    .enumerate()
                    .filter(|(_, bd)| bd.is_valid())
                    .collect();

                if valid_bds.is_empty() {
                    ui.label("No valid BDs");
                } else {
                    egui::Grid::new("dma_bds_grid")
                        .num_columns(4)
                        .spacing([10.0, 2.0])
                        .show(ui, |ui| {
                            ui.label("BD");
                            ui.label("Addr");
                            ui.label("Len");
                            ui.label("Next");
                            ui.end_row();

                            for (i, bd) in &valid_bds {
                                ui.label(format!("{}", i));
                                ui.monospace(format!("0x{:08X}", bd.address()));
                                ui.label(format!("{}", bd.length));
                                ui.label(
                                    bd.next_bd()
                                        .map(|n| format!("{}", n))
                                        .unwrap_or_else(|| "-".to_string()),
                                );
                                ui.end_row();
                            }
                        });
                }
            }
        });
}

/// Show memory view section.
fn show_memory_section(ui: &mut egui::Ui, app: &mut EmulatorApp, col: u8, row: u8) {
    egui::CollapsingHeader::new("Memory")
        .default_open(true)
        .show(ui, |ui| {
            // Memory type selector
            ui.horizontal(|ui| {
                ui.selectable_value(&mut app.show_program_memory, false, "Data");
                ui.selectable_value(&mut app.show_program_memory, true, "Program");
            });

            // Offset input
            ui.horizontal(|ui| {
                ui.label("Offset:");
                let mut offset_str = format!("0x{:X}", app.memory_offset);
                if ui.text_edit_singleline(&mut offset_str).changed() {
                    if let Some(stripped) = offset_str.strip_prefix("0x") {
                        if let Ok(val) = usize::from_str_radix(stripped, 16) {
                            app.memory_offset = val;
                        }
                    } else if let Ok(val) = offset_str.parse::<usize>() {
                        app.memory_offset = val;
                    }
                }
            });

            ui.separator();

            if let Some(tile) = app.engine.device().array.get(col, row) {
                if app.show_program_memory {
                    if let Some(pm) = tile.program_memory() {
                        memory_view::show_memory_view(ui, pm, app.memory_offset, "program_mem");
                    } else {
                        ui.label("No program memory (not a compute tile)");
                    }
                } else {
                    let dm = tile.data_memory();
                    if dm.is_empty() {
                        ui.label("No data memory (shim tile)");
                    } else {
                        memory_view::show_memory_view(ui, dm, app.memory_offset, "data_mem");
                    }
                }
            }
        });
}
