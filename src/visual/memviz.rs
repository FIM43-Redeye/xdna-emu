//! Data-memory-to-texture mapping for the selected tile.

use std::collections::HashMap;

use eframe::egui;

pub const FLOOR: u8 = 96;

pub fn word_to_pixel(bytes: [u8; 4], floor: u8) -> egui::Color32 {
    let [b0, b1, b2, b3] = bytes;
    if b0 | b1 | b2 | b3 == 0 {
        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0)
    } else {
        egui::Color32::from_rgba_unmultiplied(b0, b1, b2, b3.max(floor))
    }
}

pub fn memory_dims(pixel_count: usize) -> (usize, usize) {
    if pixel_count == 0 {
        return (0, 0);
    }

    let width = pixel_count.isqrt().next_power_of_two();
    (width, pixel_count.div_ceil(width))
}

pub fn build_image(mem: &[u8]) -> egui::ColorImage {
    let pixel_count = mem.len() / 4;
    let (width, height) = memory_dims(pixel_count);
    let mut pixels = mem
        .chunks_exact(4)
        .map(|word| word_to_pixel(word.try_into().expect("four-byte chunk"), FLOOR))
        .collect::<Vec<_>>();
    pixels.resize(width * height, egui::Color32::TRANSPARENT);
    egui::ColorImage { size: [width, height], pixels }
}

fn texture_needs_rebuild(cached_gen: Option<u64>, gen: u64) -> bool {
    cached_gen != Some(gen)
}

#[derive(Default)]
pub struct MemoryTextures {
    textures: HashMap<(u8, u8), (u64, egui::TextureHandle)>,
}

impl MemoryTextures {
    pub fn texture(
        &mut self,
        ctx: &egui::Context,
        col: u8,
        row: u8,
        mem: &[u8],
        gen: u64,
    ) -> Option<egui::TextureId> {
        if mem.is_empty() {
            return None;
        }

        let key = (col, row);
        if let Some((cached_gen, texture)) = self.textures.get(&key) {
            if !texture_needs_rebuild(Some(*cached_gen), gen) {
                return Some(texture.id());
            }
        }

        let texture =
            ctx.load_texture(format!("mem-{col}-{row}"), build_image(mem), egui::TextureOptions::NEAREST);
        let id = texture.id();
        self.textures.insert(key, (gen, texture));
        Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_to_pixel_preserves_bytes_and_floors_nonzero_alpha() {
        assert_eq!(word_to_pixel([0, 0, 0, 0], FLOOR).a(), 0);
        assert_eq!(word_to_pixel([0xff, 0, 0, 0], FLOOR).to_srgba_unmultiplied(), [0xff, 0, 0, FLOOR]);
        assert_eq!(
            word_to_pixel([1, 2, 3, 0xc0], FLOOR),
            egui::Color32::from_rgba_unmultiplied(1, 2, 3, 0xc0)
        );
        assert_eq!(word_to_pixel([1, 2, 3, 0x10], FLOOR).a(), FLOOR);
    }

    #[test]
    fn memory_dims_match_compute_and_mem_tile_sizes() {
        assert_eq!(memory_dims(16_384), (128, 128));
        assert_eq!(memory_dims(131_072), (512, 256));
        assert_eq!(memory_dims(0), (0, 0));

        for (width, height) in [memory_dims(16_384), memory_dims(131_072)] {
            assert!(width.is_power_of_two());
            assert!(height.is_power_of_two());
        }
    }

    #[test]
    fn build_image_maps_words_in_address_order() {
        let first = [1, 2, 3, 4];
        let mut mem = vec![0; 16];
        mem[..4].copy_from_slice(&first);

        let image = build_image(&mem);

        let (width, height) = memory_dims(mem.len() / 4);
        assert_eq!(image.size, [width, height]);
        assert_eq!(image.pixels[0], word_to_pixel(first, FLOOR));
        assert_eq!(image.pixels[1], egui::Color32::TRANSPARENT);
    }

    #[test]
    fn texture_rebuild_decision_depends_only_on_generation() {
        assert!(texture_needs_rebuild(None, 7));
        assert!(!texture_needs_rebuild(Some(7), 7));
        assert!(texture_needs_rebuild(Some(6), 7));
    }
}
