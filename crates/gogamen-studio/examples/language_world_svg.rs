//! Language World a SVG (`cargo run -p gogamen-studio --example language_world_svg -- [agent|evaluator] [true|false]`).
//! Aviso humano (verdadero/falso) sobre la celda (2,1) y 3 ticks de rollout offline. SVG por stdout.

use gogamen_core::{Permission, PermissionSet};
use gogamen_studio::language_world_lab::{
    ASK_PERMISSION, CannedBehavior, LanguageWorldLabConfig, LanguageWorldRollout,
    OfflineAskProvider, ViewMode, WARNING_TEXT, render_svg, scenario_map,
};
use std::time::Duration;

fn main() {
    let mut a = std::env::args().skip(1);
    let mode = match a.next().as_deref() {
        Some("evaluator") => ViewMode::Evaluator,
        _ => ViewMode::Agent,
    };
    let obstacle = !matches!(a.next().as_deref(), Some("false"));
    let provider = OfflineAskProvider::new(CannedBehavior::Answer {
        obstacle,
        confidence: 0.9,
        latency: Duration::from_millis(200),
    });
    let perms = PermissionSet::empty().with_permission(Permission::new(ASK_PERMISSION));
    let mut r = LanguageWorldRollout::new(
        LanguageWorldLabConfig::new(scenario_map(obstacle)),
        provider,
        perms,
    )
    .expect("rollout");
    r.submit_human_notice(WARNING_TEXT).expect("notice");
    for _ in 0..3 {
        r.tick().expect("tick");
    }
    println!("{}", render_svg(&r.scene(mode), 900.0, 700.0));
}
