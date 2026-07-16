//! Stream-route overlay for the architecture view.

use eframe::egui::{Painter, Pos2, Stroke};

use crate::visual::theme::Palette;

#[derive(Clone, Copy, Debug)]
pub struct RouteLine {
    pub from: Pos2,
    pub to: Pos2,
    pub moving: bool,
    pub stalled: bool,
}

fn route_stroke(route: &RouteLine, palette: &Palette) -> Stroke {
    if route.stalled {
        Stroke::new(2.5_f32, palette.route_stalled)
    } else if route.moving {
        Stroke::new(2.0_f32, palette.route_moving)
    } else {
        Stroke::new(0.5_f32, palette.route_idle)
    }
}

pub fn draw_routes(painter: &Painter, routes: &[RouteLine], palette: &Palette) {
    for route in routes {
        painter.line_segment([route.from, route.to], route_stroke(route, palette));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visual::theme::Palette;
    use eframe::egui::pos2;

    #[test]
    fn route_strokes_follow_activity_salience() {
        let palette = Palette::dark();
        let idle = RouteLine { from: pos2(0.0, 0.0), to: pos2(1.0, 1.0), moving: false, stalled: false };
        let moving = RouteLine { moving: true, ..idle };
        let stalled = RouteLine { stalled: true, ..moving };

        let idle_stroke = route_stroke(&idle, &palette);
        let moving_stroke = route_stroke(&moving, &palette);
        let stalled_stroke = route_stroke(&stalled, &palette);

        assert_eq!(idle_stroke.color, palette.route_idle);
        assert_eq!(moving_stroke.color, palette.route_moving);
        assert_eq!(stalled_stroke.color, palette.route_stalled);
        assert!(idle_stroke.width < moving_stroke.width);
    }
}
