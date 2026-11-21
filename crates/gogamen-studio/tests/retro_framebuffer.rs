//! GOG-111: framebuffer retro + overlays (AC1..AC3), sin ROM comercial ni UI.

use gogamen_lab_retro::{AdapterConfig, fnv1a};
use gogamen_projection::ImageViewport;
use gogamen_projection::{LayerId, Texture};
use gogamen_studio::retro_session::{
    CommandOutcome, CommandReject, EmuCommand, FREEPLAY_PALETTE, HOMEBREW_LICENSE, RetroHost,
    detect_index_boxes,
};
use gogamen_studio::retro_view::{
    CommandCounters, FrameCounters, OverlayItem, PRIVILEGED_LAYER, PerceptionMode,
    PerceptionSource, RetroFrameView, RetroViewError, project_retro_scene, render_retro_svg,
};

fn cfg() -> AdapterConfig {
    AdapterConfig {
        frame_skip: 2,
        goal_score: u16::MAX,
        max_steps: 10_000,
        ..AdapterConfig::default()
    }
}

fn run(host: &mut RetroHost, n: usize) {
    for i in 0..n {
        if i % 5 == 0 {
            host.submit(EmuCommand::Input((i * 3 + 1) % 9)).unwrap();
        }
        host.tick().unwrap();
    }
}

fn item(source: PerceptionSource, label: &str, x: u32, y: u32) -> OverlayItem {
    OverlayItem {
        source,
        label: label.into(),
        x,
        y,
        w: 1,
        h: 1,
    }
}

// ---------- AC1 ----------

#[test]
fn frame_bytes_pass_through_unmodified_and_colors_follow_palette() {
    let mut host = RetroHost::headless(cfg()).unwrap();
    run(&mut host, 20);
    let tex = host.texture().unwrap();
    assert_eq!(tex.pixels(), host.adapter().framebuffer());
    assert_eq!(tex.pixel_hash(), host.adapter().frame_hash());
    assert_eq!(tex.pixel_hash(), fnv1a(host.adapter().framebuffer()));
    // el pixel del jugador (RAM 0,1) es índice 1 y su color sale de la paleta
    let p = &host.privileged_items()[0];
    assert_eq!(tex.pixels()[(p.y * 32 + p.x) as usize], 1);
    assert_eq!(tex.rgba_at(p.x, p.y), Some(FREEPLAY_PALETTE[1]));
    // el frame cambia al avanzar
    run(&mut host, 7);
    assert_ne!(host.texture().unwrap().pixel_hash(), tex.pixel_hash());
    // y la vista conserva el hash
    let view = host.current_view(false).unwrap();
    assert_eq!(view.texture().pixel_hash(), host.adapter().frame_hash());
}

#[test]
fn mapping_is_exact_and_invertible_across_resizes() {
    let sizes = [
        (32, 32),
        (64, 64),
        (100, 70),
        (70, 100),
        (321, 200),
        (640, 480),
        (1920, 1080),
        (33, 95),
    ];
    for (vw, vh) in sizes {
        let vp = ImageViewport::fit(32, 32, vw, vh).unwrap();
        assert!(vp.is_integer_scale(), "{vw}x{vh}");
        let s = vp.scale();
        let (ox, oy) = vp.offset();
        assert_eq!(s, ((vw / 32).min(vh / 32)) as f64);
        // letterbox centrado (diferencia <= 1 px) y dentro de la ventana
        let (rx, ry, rw, rh) = vp.image_screen_rect();
        assert!(rx >= 0.0 && ry >= 0.0 && rx + rw <= vw as f64 && ry + rh <= vh as f64);
        assert!((((vw as f64) - (rx + rw)) - rx).abs() <= 1.0);
        assert!((((vh as f64) - (ry + rh)) - ry).abs() <= 1.0);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let (sx, sy) = vp.image_to_screen(x as f64, y as f64);
                assert_eq!((sx, sy), (ox + x as f64 * s, oy + y as f64 * s));
                // imagen -> pantalla -> imagen exacto
                assert_eq!(vp.screen_to_image(sx, sy), (x as f64, y as f64));
                // el centro del píxel cae en el píxel correcto; esquina también
                assert_eq!(vp.screen_to_pixel(sx, sy), Some((x, y)));
                assert_eq!(vp.screen_to_pixel(sx + s / 2.0, sy + s / 2.0), Some((x, y)));
            }
        }
        // pantalla -> imagen -> pantalla exacto en la rejilla de píxeles
        for k in [0.0, 1.0, 7.0, 31.0, 32.0] {
            let (sx, sy) = (ox + k * s, oy + k * s);
            let (ix, iy) = vp.screen_to_image(sx, sy);
            assert_eq!(vp.image_to_screen(ix, iy), (sx, sy));
        }
        // barras: fuera de la imagen no hay píxel
        if ox > 0.0 {
            assert_eq!(vp.screen_to_pixel(ox - 1.0, oy), None);
        }
        assert_eq!(vp.screen_to_pixel(rx + rw, ry), None);
    }
}

#[test]
fn rgba_and_non_square_images_map_correctly_and_small_windows_stay_invertible() {
    let vp = ImageViewport::fit(160, 144, 800, 600).unwrap();
    assert_eq!(vp.scale(), 4.0);
    assert_eq!(vp.offset(), (80.0, 12.0));
    assert_eq!(vp.image_to_screen(10.0, 20.0), (120.0, 92.0));
    let small = ImageViewport::fit(160, 144, 100, 90).unwrap();
    assert!(!small.is_integer_scale());
    for (x, y) in [(0.0, 0.0), (13.0, 77.0), (159.0, 143.0)] {
        let (sx, sy) = small.image_to_screen(x, y);
        let (bx, by) = small.screen_to_image(sx, sy);
        assert!((bx - x).abs() < 1e-9 && (by - y).abs() < 1e-9);
    }
    let rgba = Texture::rgba8(2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    assert_eq!(rgba.rgba_at(1, 0), Some([5, 6, 7, 8]));
}

#[test]
fn overlay_lands_on_expected_screen_position_in_svg() {
    let mut host = RetroHost::headless(cfg()).unwrap();
    run(&mut host, 3);
    let tex = host.texture().unwrap();
    let view = RetroFrameView::new(
        1,
        tex,
        vec![item(PerceptionSource::Privileged, "p", 5, 7)],
        vec![],
        HOMEBREW_LICENSE,
        FrameCounters::default(),
        CommandCounters::default(),
    )
    .unwrap();
    // 100x70 -> escala 2, offset (18, 3): (5,7) -> (28, 17), tamaño 2x2
    let svg = render_retro_svg(&view, PerceptionMode::Privileged, 100, 70).unwrap();
    assert!(svg.contains(
        "<rect x=\"28.000\" y=\"17.000\" width=\"2.000\" height=\"2.000\" fill=\"none\" stroke=\"#00c853\""
    ));
    // tras redimensionar: 640x480 -> escala 15, offset (80, 0): (5,7) -> (155, 105)
    let svg = render_retro_svg(&view, PerceptionMode::Privileged, 640, 480).unwrap();
    assert!(svg.contains("<rect x=\"155.000\" y=\"105.000\" width=\"15.000\" height=\"15.000\""));
    assert_eq!(
        svg,
        render_retro_svg(&view, PerceptionMode::Privileged, 640, 480).unwrap()
    );
}

// ---------- AC2 ----------

fn view_with(detect: bool) -> RetroFrameView {
    let mut host = RetroHost::headless(cfg()).unwrap();
    run(&mut host, 10);
    host.current_view(detect).unwrap()
}

#[test]
fn pixel_only_without_detector_shows_no_detections_and_no_privileged_data() {
    let view = view_with(false);
    assert!(view.visible_items(PerceptionMode::PixelOnly).is_empty());
    let scene = project_retro_scene(&view, PerceptionMode::PixelOnly).unwrap();
    for layer in scene.layers() {
        let n = layer.nodes().len();
        if layer.id() == LayerId::Environment {
            assert_eq!(n, 1); // sólo la imagen
        } else {
            assert_eq!(
                n,
                0,
                "{:?} must be empty in pixel-only/no-detector",
                layer.id()
            );
        }
    }
    let svg = render_retro_svg(&view, PerceptionMode::PixelOnly, 330, 360).unwrap();
    assert!(!svg.contains("[RAM]") && !svg.contains("[PX]") && !svg.contains("dasharray"));
    // en modo privilegiado sí hay items [RAM]
    assert_eq!(view.visible_items(PerceptionMode::Privileged).len(), 3);
}

#[test]
fn privileged_never_leaks_into_pixel_only_even_with_detector() {
    let view = view_with(true);
    let items = view.visible_items(PerceptionMode::PixelOnly);
    assert!(!items.is_empty());
    assert!(
        items
            .iter()
            .all(|i| i.source == PerceptionSource::PixelOnly)
    );
    let scene = project_retro_scene(&view, PerceptionMode::PixelOnly).unwrap();
    let privileged_layer = scene
        .layers()
        .iter()
        .find(|l| l.id() == PRIVILEGED_LAYER)
        .unwrap();
    assert!(privileged_layer.nodes().is_empty());
    let svg = render_retro_svg(&view, PerceptionMode::PixelOnly, 330, 360).unwrap();
    assert!(svg.contains("[PX] ") && !svg.contains("[RAM]") && !svg.contains("dasharray"));
    // modo privilegiado: ambos, con estilo distinto
    let full = render_retro_svg(&view, PerceptionMode::Privileged, 330, 360).unwrap();
    assert!(full.contains("[RAM] ") && full.contains("[PX] "));
    assert!(full.contains("stroke=\"#00c853\" stroke-dasharray=\"4 2\""));
    assert!(full.contains("stroke=\"#ff6d00\"/>"));
}

#[test]
fn detector_does_not_invent_and_mistagged_items_are_rejected() {
    let blank = Texture::indexed8(4, 4, FREEPLAY_PALETTE.to_vec(), vec![0; 16]).unwrap();
    assert!(detect_index_boxes(&blank, &[(1, "player"), (2, "coin"), (3, "enemy")]).is_empty());
    let tex = Texture::indexed8(4, 4, FREEPLAY_PALETTE.to_vec(), vec![0; 16]).unwrap();
    let wrong = RetroFrameView::new(
        0,
        tex.clone(),
        vec![item(PerceptionSource::PixelOnly, "x", 0, 0)],
        vec![],
        "lic",
        FrameCounters::default(),
        CommandCounters::default(),
    );
    assert_eq!(wrong.unwrap_err(), RetroViewError::WrongSource);
    let smuggled = RetroFrameView::new(
        0,
        tex.clone(),
        vec![],
        vec![item(PerceptionSource::Privileged, "x", 0, 0)],
        "lic",
        FrameCounters::default(),
        CommandCounters::default(),
    );
    assert_eq!(smuggled.unwrap_err(), RetroViewError::WrongSource);
    let oob = RetroFrameView::new(
        0,
        tex,
        vec![item(PerceptionSource::Privileged, "x", 4, 0)],
        vec![],
        "lic",
        FrameCounters::default(),
        CommandCounters::default(),
    );
    assert_eq!(oob.unwrap_err(), RetroViewError::ItemOutOfBounds);
}

// ---------- AC3 ----------

#[test]
fn slow_or_disconnected_viewer_does_not_affect_emulator_or_commands() {
    const N: usize = 40;
    let mut headless = RetroHost::headless(cfg()).unwrap();
    let mut slow = RetroHost::with_viewer(cfg(), 2).unwrap();
    let mut gone = RetroHost::with_viewer(cfg(), 2).unwrap();
    gone.set_viewer_connected(false);
    for h in [&mut headless, &mut slow, &mut gone] {
        run(h, N);
    }
    // emulador y comandos idénticos
    assert_eq!(
        headless.adapter().snapshot_bytes(),
        slow.adapter().snapshot_bytes()
    );
    assert_eq!(
        headless.adapter().snapshot_bytes(),
        gone.adapter().snapshot_bytes()
    );
    assert_eq!(headless.command_log(), slow.command_log());
    assert_eq!(headless.command_log(), gone.command_log());
    assert_eq!(headless.command_counters(), slow.command_counters());
    assert_eq!(headless.command_counters(), gone.command_counters());
    // frames separados
    let n = N as u64;
    let f = headless.frame_counters();
    assert_eq!(
        (
            f.produced,
            f.published,
            f.dropped_slow,
            f.dropped_disconnected
        ),
        (n, 0, 0, 0)
    );
    let f = slow.frame_counters();
    assert_eq!(
        (
            f.produced,
            f.published,
            f.dropped_slow,
            f.dropped_disconnected
        ),
        (n, n, n - 2, 0)
    );
    assert_eq!(slow.queued_frames(), 2);
    assert_eq!(f.produced, f.published + f.dropped_disconnected);
    assert_eq!(
        f.published,
        f.consumed + f.dropped_slow + slow.queued_frames() as u64
    );
    let f = gone.frame_counters();
    assert_eq!(
        (
            f.produced,
            f.published,
            f.dropped_slow,
            f.dropped_disconnected
        ),
        (n, 0, 0, n)
    );
    // el visor lento consume lo último disponible
    let p = slow.take_frame().unwrap();
    assert_eq!(p.index, n - 1);
    assert_eq!(slow.frame_counters().consumed, 1);
}

#[test]
fn emulator_commands_are_recorded_separately_from_frames() {
    let mut h = RetroHost::with_viewer(cfg(), 1).unwrap();
    assert_eq!(
        h.submit(EmuCommand::Step).unwrap(),
        CommandOutcome::Rejected(CommandReject::NotPaused)
    );
    assert_eq!(
        h.submit(EmuCommand::Input(99)).unwrap(),
        CommandOutcome::Rejected(CommandReject::InvalidAction)
    );
    assert_eq!(
        h.submit(EmuCommand::Pause).unwrap(),
        CommandOutcome::Applied
    );
    assert!(!h.tick().unwrap()); // en pausa no hay frames
    assert_eq!(h.frame_counters().produced, 0);
    assert_eq!(h.submit(EmuCommand::Step).unwrap(), CommandOutcome::Applied);
    assert_eq!(h.submit(EmuCommand::Step).unwrap(), CommandOutcome::Applied);
    assert_eq!(h.frame_counters().produced, 2);
    assert_eq!(h.frame_counters().dropped_slow, 1); // visor de capacidad 1
    let c = h.command_counters();
    assert_eq!((c.received, c.applied, c.rejected), (5, 3, 2));
    let seqs: Vec<u64> = h.command_log().iter().map(|r| r.seq).collect();
    assert_eq!(seqs, vec![1, 2, 3, 4, 5]);
    // los contadores aparecen separados en el SVG
    let svg = render_retro_svg(
        &h.current_view(false).unwrap(),
        PerceptionMode::Privileged,
        330,
        360,
    )
    .unwrap();
    assert!(svg.contains("dropped_slow=1"));
    assert!(svg.contains("commands received=5 applied=3 rejected=2"));
}

#[test]
fn license_round_trips_through_view_svg_and_readme() {
    let view = view_with(false);
    assert_eq!(view.license(), HOMEBREW_LICENSE);
    let svg = render_retro_svg(&view, PerceptionMode::PixelOnly, 330, 360).unwrap();
    assert!(svg.contains(&format!("license: {HOMEBREW_LICENSE}")));
    let readme = include_str!("../../../labs/retro/README.md");
    assert!(readme.contains(HOMEBREW_LICENSE));
    let tex = Texture::indexed8(1, 1, vec![[0; 4]], vec![0]).unwrap();
    let empty = RetroFrameView::new(
        0,
        tex,
        vec![],
        vec![],
        "  ",
        FrameCounters::default(),
        CommandCounters::default(),
    );
    assert_eq!(empty.unwrap_err(), RetroViewError::MissingLicense);
}
