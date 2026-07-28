//! Named, swappable visual-debugger palettes.

use eframe::egui::Color32;

use crate::debugger::model::TileState;

const fn rgb(hex: u32) -> Color32 {
    Color32::from_rgb((hex >> 16) as u8, (hex >> 8) as u8, hex as u8)
}

#[derive(Clone, Copy, Debug)]
pub struct Palette {
    pub bg: Color32,
    pub text: Color32,
    pub kind_shim: Color32,
    pub kind_mem: Color32,
    pub kind_core: Color32,
    pub band_grey: Color32,
    pub band_pale: Color32,
    pub band_green: Color32,
    pub band_amber: Color32,
    pub band_blue: Color32,
    pub band_red: Color32,
    pub selected: Color32,
    pub route_moving: Color32,
    pub route_idle: Color32,
    pub route_stalled: Color32,
    pub dma_starved: Color32,
    pub err_border: Color32,
}

impl Palette {
    // Candidate bands; tuned by eye. AA-verified against `text` -- keep any
    // change above 4.5:1. All tunable color literals live in these palettes.
    pub const fn dark() -> Self {
        Self {
            bg: rgb(0x181a1e),
            text: rgb(0xededed),
            kind_shim: rgb(0x6f5592),
            kind_mem: rgb(0x397187),
            kind_core: rgb(0x477a54),
            band_grey: rgb(0x44464a),
            band_pale: rgb(0x51545a),
            band_green: rgb(0x2e6b3e),
            band_amber: rgb(0x8a5a18),
            band_blue: rgb(0x2a4e7a),
            band_red: rgb(0x7a2530),
            selected: rgb(0xe6c85a),
            route_moving: rgb(0x78d890),
            route_idle: rgb(0x59616b),
            route_stalled: rgb(0xdc8c5a),
            dma_starved: rgb(0x633a7a),
            err_border: rgb(0xff6b7a),
        }
    }

    pub const fn light() -> Self {
        Self {
            bg: rgb(0xf7f7f8),
            text: rgb(0x1a1a1a),
            kind_shim: rgb(0x7b5aa6),
            kind_mem: rgb(0x39788d),
            kind_core: rgb(0x477a54),
            band_grey: rgb(0xd6d7da),
            band_pale: rgb(0xf0f1f3),
            band_green: rgb(0xa9d7b4),
            band_amber: rgb(0xf4ce8a),
            band_blue: rgb(0xafc9ea),
            band_red: rgb(0xe7aeb5),
            selected: rgb(0x8a6a00),
            route_moving: rgb(0x128a32),
            route_idle: rgb(0x89919a),
            route_stalled: rgb(0xb55b16),
            dma_starved: rgb(0xd8b6e8),
            err_border: rgb(0x9e1026),
        }
    }

    pub const fn high_contrast() -> Self {
        Self {
            bg: Color32::BLACK,
            text: Color32::WHITE,
            kind_shim: rgb(0xd49cff),
            kind_mem: rgb(0x55ddff),
            kind_core: rgb(0x66ee88),
            band_grey: rgb(0x202020),
            band_pale: rgb(0x303030),
            band_green: rgb(0x005a1f),
            band_amber: rgb(0x6a3e00),
            band_blue: rgb(0x003d73),
            band_red: rgb(0x650018),
            selected: rgb(0xffff00),
            route_moving: Color32::WHITE,
            route_idle: rgb(0x888888),
            route_stalled: rgb(0xffa500),
            dma_starved: rgb(0x7b2cbf),
            err_border: rgb(0xff2d55),
        }
    }

    pub fn band(&self, state: &TileState) -> Color32 {
        match state {
            TileState::NotEnabled => self.band_grey,
            TileState::Ready | TileState::Idle => self.band_pale,
            TileState::Running | TileState::Dma | TileState::Stream => self.band_green,
            TileState::WaitLock(_)
            | TileState::WaitDma(_)
            | TileState::WaitStream(_)
            | TileState::WaitBank => self.band_amber,
            TileState::Done => self.band_blue,
            TileState::Error => self.band_red,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_palette_uses_the_candidate_band_tokens() {
        let palette = Palette::dark();

        assert_eq!(palette.text, Color32::from_rgb(0xed, 0xed, 0xed));
        assert_eq!(palette.band_grey, Color32::from_rgb(0x44, 0x46, 0x4a));
        assert_eq!(palette.band_green, Color32::from_rgb(0x2e, 0x6b, 0x3e));
        assert_eq!(palette.band_amber, Color32::from_rgb(0x8a, 0x5a, 0x18));
        assert_eq!(palette.band_blue, Color32::from_rgb(0x2a, 0x4e, 0x7a));
        assert_eq!(palette.band_red, Color32::from_rgb(0x7a, 0x25, 0x30));
    }

    #[test]
    fn band_maps_every_tile_state_by_meaning() {
        let palette = Palette::dark();

        assert_eq!(palette.band(&TileState::NotEnabled), palette.band_grey);
        assert_eq!(palette.band(&TileState::Ready), palette.band_pale);
        assert_eq!(palette.band(&TileState::Running), palette.band_green);
        assert_eq!(palette.band(&TileState::WaitLock(1)), palette.band_amber);
        assert_eq!(palette.band(&TileState::WaitDma(1)), palette.band_amber);
        assert_eq!(palette.band(&TileState::WaitStream(1)), palette.band_amber);
        assert_eq!(palette.band(&TileState::WaitBank), palette.band_amber);
        assert_eq!(palette.band(&TileState::Done), palette.band_blue);
        assert_eq!(palette.band(&TileState::Error), palette.band_red);
        assert_eq!(palette.band(&TileState::Dma), palette.band_green);
        assert_eq!(palette.band(&TileState::Stream), palette.band_green);
        assert_eq!(palette.band(&TileState::Idle), palette.band_pale);
    }

    #[test]
    fn palettes_provide_distinct_theme_grounds() {
        let dark = Palette::dark();
        let light = Palette::light();
        let high_contrast = Palette::high_contrast();

        assert_ne!(dark.bg, light.bg);
        assert_ne!(dark.text, light.text);
        assert_eq!(high_contrast.bg, Color32::BLACK);
        assert_eq!(high_contrast.text, Color32::WHITE);
    }

    #[test]
    fn dma_starvation_is_distinct_from_other_stalls_in_every_palette() {
        for palette in [Palette::dark(), Palette::light(), Palette::high_contrast()] {
            assert_ne!(palette.dma_starved, palette.band_amber);
            assert_ne!(palette.dma_starved, palette.route_stalled);
        }
    }
}
