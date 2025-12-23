//! Hex memory view widget.

use eframe::egui;

/// Number of bytes per row in the hex view.
const BYTES_PER_ROW: usize = 16;
/// Number of rows to display.
const VISIBLE_ROWS: usize = 16;

/// Show a hex memory view.
pub fn show_memory_view(ui: &mut egui::Ui, memory: &[u8], offset: usize, id: &str) {
    if memory.is_empty() {
        ui.label("(empty)");
        return;
    }

    let max_offset = memory.len().saturating_sub(1);
    let start = offset.min(max_offset);
    let end = (start + BYTES_PER_ROW * VISIBLE_ROWS).min(memory.len());

    // Use monospace font for alignment
    egui::Grid::new(id)
        .num_columns(2 + BYTES_PER_ROW + 1)
        .spacing([2.0, 2.0])
        .show(ui, |ui| {
            // Header row
            ui.monospace("Offset");
            ui.monospace(" ");
            for i in 0..BYTES_PER_ROW {
                ui.monospace(format!("{:02X}", i));
            }
            ui.monospace("ASCII");
            ui.end_row();

            // Data rows
            let mut addr = start;
            while addr < end {
                // Offset column
                ui.monospace(format!("{:06X}", addr));
                ui.monospace(":");

                // Hex bytes
                let row_end = (addr + BYTES_PER_ROW).min(end);
                let row_data = &memory[addr..row_end];

                for byte in row_data.iter() {
                    let color = byte_color(*byte);
                    ui.colored_label(color, format!("{:02X}", byte));
                }

                // Pad if row is short
                for _ in row_data.len()..BYTES_PER_ROW {
                    ui.label("  ");
                }

                // ASCII column
                let ascii: String = row_data
                    .iter()
                    .map(|&b| {
                        if b.is_ascii_graphic() || b == b' ' {
                            b as char
                        } else {
                            '.'
                        }
                    })
                    .collect();
                ui.monospace(ascii);

                ui.end_row();
                addr += BYTES_PER_ROW;
            }
        });

    // Summary
    ui.separator();
    ui.label(format!(
        "Showing 0x{:X} - 0x{:X} of 0x{:X} bytes",
        start,
        end.saturating_sub(1),
        memory.len()
    ));

    // Check for non-zero regions
    let non_zero_count = memory.iter().filter(|&&b| b != 0).count();
    if non_zero_count > 0 {
        ui.label(format!("Non-zero bytes: {} ({:.1}%)",
            non_zero_count,
            (non_zero_count as f64 / memory.len() as f64) * 100.0
        ));
    } else {
        ui.label("Memory is all zeros");
    }
}

/// Get color for a byte value (to highlight patterns).
fn byte_color(byte: u8) -> egui::Color32 {
    if byte == 0 {
        egui::Color32::DARK_GRAY
    } else if byte == 0xFF {
        egui::Color32::from_rgb(200, 100, 100)
    } else if byte.is_ascii_alphanumeric() {
        egui::Color32::from_rgb(100, 200, 100)
    } else {
        egui::Color32::WHITE
    }
}
