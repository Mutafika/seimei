//! Standalone live VRM viewer, built on **Sabitori's `SceneApp`**.
//!
//!   vrm-viewer /path/to/model.vrm
//!
//! `render_scene()` draws the avatar with seimei straight into Sabitori's frame
//! (shared wgpu device, no readback), and the controls are a Sabitori declarative
//! UI overlay. All avatar motion lives in `vrm_seimei::AvatarController`; this file
//! is only the shell. This is also the pattern mearie's persona uses to embed the
//! avatar in its Sabitori UI.
//!
//! Buttons sit at the bottom. Drag the avatar area = orbit. Type in "Say:" + Speak
//! to lip-sync.

use std::time::Instant;

use glam::Mat4;
use sabitori::element::*;
use sabitori::*;
use seimei::{Camera, InstanceData, Light, Point3, Renderer};
use vrm_seimei::{AvatarController, ExpressionPreset, VrmAvatar};

/// Fixed animation step (~60fps; SceneApp drives a continuous redraw loop).
const DT: f32 = 1.0 / 60.0;

/// Emotion/vowel buttons: (element id, label, preset). Only those the model
/// actually defines are shown.
const EXPRS: [(&str, &str, ExpressionPreset); 9] = [
    ("joy", "Joy", ExpressionPreset::Happy),
    ("angry", "Angry", ExpressionPreset::Angry),
    ("sorrow", "Sorrow", ExpressionPreset::Sad),
    ("fun", "Fun", ExpressionPreset::Relaxed),
    ("a", "A", ExpressionPreset::Aa),
    ("i", "I", ExpressionPreset::Ih),
    ("u", "U", ExpressionPreset::Ou),
    ("e", "E", ExpressionPreset::Ee),
    ("o", "O", ExpressionPreset::Oh),
];

fn onoff(b: bool) -> &'static str {
    if b { "on" } else { "off" }
}

/// The whole live viewer: a seimei renderer + a vrm-seimei controller, plus orbit
/// camera and the text-box state. GPU objects are created in `setup` (once the
/// Sabitori device exists), so they start as `None`.
struct ViewerApp {
    vrm_path: String,
    renderer: Option<Renderer>,
    ctrl: Option<AvatarController>,
    instances: Vec<(String, InstanceData)>,
    opaque_count: usize,
    n_prims: usize,
    // orbit camera (spherical around `center`)
    cam: Camera,
    center: [f64; 3],
    dist: f64,
    azim: f64,
    elev: f64,
    azim_front: f64,
    dragging: bool,
    last_cursor: Option<(f32, f32)>,
    // "Say:" text box
    say: TextInputState,
    // perf instrumentation
    frame_count: u32,
    fps_timer: Instant,
    skin_accum_us: u128,
}

impl ViewerApp {
    fn new(vrm_path: String) -> Self {
        Self {
            vrm_path,
            renderer: None,
            ctrl: None,
            instances: Vec::new(),
            opaque_count: 0,
            n_prims: 0,
            cam: Camera::new(),
            center: [0.0; 3],
            dist: 1.0,
            azim: 0.0,
            elev: 0.0,
            azim_front: 0.0,
            dragging: false,
            last_cursor: None,
            say: TextInputState::new("type Japanese or romaji…"),
            frame_count: 0,
            fps_timer: Instant::now(),
            skin_accum_us: 0,
        }
    }

    /// Place the orbit camera from (azim, elev, dist) around `center`.
    fn place_camera(&mut self) {
        let a = self.azim + self.azim_front;
        let ce = self.elev.cos();
        self.cam.position = Point3::new(
            self.center[0] + self.dist * ce * a.sin(),
            self.center[1] + self.dist * ce * a.cos(),
            self.center[2] + self.dist * self.elev.sin(),
        );
        self.cam.target = Point3::new(self.center[0], self.center[1], self.center[2]);
    }

    /// A labelled control button with an id.
    fn btn(id: &str, label: impl Into<String>) -> Element {
        button(label).id(id)
    }

    /// Build the bottom control bar (declarative UI, painted over the 3D scene).
    fn control_bar(&self) -> Element {
        let surface = Color::from_hex("#1a1b26");
        let text_c = Color::from_hex("#c0caf5");
        let field_bg = Color::from_hex("#414868");
        let border_c = Color::from_hex("#565f89");
        let primary = Color::from_hex("#7aa2f7");

        let (paused, spinning, spring, blink) = match self.ctrl.as_ref() {
            Some(c) => (c.is_paused(), c.is_spinning(), c.avatar().spring_enabled(), c.auto_blink()),
            None => (false, false, true, true),
        };

        // Row 1: locomotion + view.
        let row1 = div().flex_row().gap(6.0).children([
            Self::btn("bind", "Bind"),
            Self::btn("idle", "Idle"),
            Self::btn("walk", "Walk"),
            Self::btn("run", "Run"),
            Self::btn("pause", if paused { "Resume" } else { "Pause" }),
            Self::btn("spin", format!("Spin: {}", onoff(spinning))),
            Self::btn("spring", format!("Spring: {}", onoff(spring))),
            Self::btn("turnl", "Turn <"),
            Self::btn("turnr", "Turn >"),
        ]);

        // Row 2: face (only presets the model has).
        let mut face: Vec<Element> = vec![text("Face:").font_size(13.0).color(text_c)];
        if let Some(c) = self.ctrl.as_ref() {
            for (id, label, preset) in EXPRS {
                if c.avatar().has_expression(preset) {
                    face.push(Self::btn(id, label));
                }
            }
        }
        face.push(Self::btn("neutral", "Neutral"));
        face.push(Self::btn("blink", format!("Blink: {}", onoff(blink))));
        let row2 = div().flex_row().gap(6.0).items_center().children(face);

        // Row 3: lip-sync text box + Speak.
        let say_text = if self.say.text.is_empty() {
            "type Japanese or romaji…".to_string()
        } else {
            self.say.text.clone()
        };
        let say_color = if self.say.text.is_empty() { border_c } else { text_c };
        let row3 = div().flex_row().gap(6.0).items_center().children([
            text("Say:").font_size(13.0).color(text_c),
            div()
                .id("say")
                .w(Px(280.0))
                .h(Px(28.0))
                .bg(field_bg)
                .rounded_px(4.0)
                .px_pad(Px(8.0))
                .items_center()
                .flex_row()
                .border(1.0, border_c)
                .children([text(say_text).font_size(13.0).color(say_color)]),
            Self::btn("speak", "Speak"),
        ]);

        div()
            .flex_col()
            .gap(6.0)
            .p_px(10.0)
            .bg(surface)
            .children([row1, row2, row3])
    }
}

impl DeclarativeApp for ViewerApp {
    fn title(&self) -> &str {
        "vrm viewer"
    }
    fn size(&self) -> (f32, f32) {
        (1100.0, 820.0)
    }

    fn view(&self, ctx: &ViewContext) -> Element {
        // Root has NO background → transparent → the 3D scene shows through; only
        // the bottom bar is opaque. justify_end pins the bar to the bottom.
        div()
            .w(Px(ctx.width))
            .h(Px(ctx.height))
            .flex_col()
            .justify_end()
            .children([self.control_bar()])
    }

    fn on_click(&mut self, id: &str) {
        // Camera actions are local; everything else goes to the controller.
        match id {
            "turnl" => {
                self.azim -= 0.35;
                return;
            }
            "turnr" => {
                self.azim += 0.35;
                return;
            }
            _ => {}
        }
        let Some(c) = self.ctrl.as_mut() else { return };
        match id {
            "bind" => c.bind(),
            "idle" => c.play_idle(),
            "walk" => c.play_walk(),
            "run" => c.play_run(),
            "pause" => c.toggle_pause(),
            "spin" => c.toggle_spin(),
            "spring" => {
                let on = !c.avatar().spring_enabled();
                c.avatar_mut().set_spring_enabled(on);
            }
            "neutral" => c.clear_face(),
            "blink" => c.toggle_auto_blink(),
            "speak" => c.say(&self.say.text),
            other => {
                if let Some((_, _, p)) = EXPRS.iter().find(|(i, _, _)| *i == other) {
                    c.set_emotion(*p);
                }
            }
        }
    }

    fn on_focused_input(&mut self, id: &str, event: &InputEvent) -> bool {
        if id != "say" {
            return false;
        }
        match event {
            InputEvent::CharInput(ch) => {
                self.say.on_char(*ch);
                true
            }
            InputEvent::KeyInput { key, pressed: true, modifiers } => self.say.on_key(*key, *modifiers),
            InputEvent::ImePreedit { text, cursor } => {
                self.say.on_ime_preedit(text.clone(), *cursor);
                true
            }
            InputEvent::ImeCommit { text } => {
                self.say.on_ime_commit(text);
                true
            }
            _ => false,
        }
    }

    fn on_input(&mut self, event: &InputEvent) -> bool {
        // Press starts a drag; movement/release arrive via on_pointer_move/up below
        // (SceneApp does NOT deliver cursor movement as a PointerMoved InputEvent).
        match event {
            InputEvent::PointerPressed { button: Some(MouseButton::Left), position, .. } => {
                self.dragging = true;
                self.last_cursor = Some((position.x, position.y));
            }
            InputEvent::PointerReleased { .. } => self.dragging = false,
            _ => {}
        }
        false
    }

    fn on_pointer_move(&mut self, x: f32, y: f32) {
        if !self.dragging {
            return;
        }
        if let Some((lx, ly)) = self.last_cursor {
            self.azim += (x - lx) as f64 * 0.01;
            self.elev = (self.elev + (y - ly) as f64 * 0.01).clamp(-1.3, 1.3);
        }
        self.last_cursor = Some((x, y));
    }

    fn on_pointer_up(&mut self) {
        self.dragging = false;
    }
}

impl SceneApp for ViewerApp {
    fn setup(&mut self, ctx: &GpuContext) {
        let (w, h) = (ctx.surface_width, ctx.surface_height);
        let bytes = std::fs::read(&self.vrm_path).expect("read vrm");
        let avatar = VrmAvatar::load(&bytes).expect("load vrm");
        eprintln!(
            "[viewer] spring joints (揺れもの): {}  colliders: {}",
            avatar.spring_joints(),
            avatar.spring_colliders()
        );
        eprintln!(
            "[viewer] expression presets ({}): {:?}",
            avatar.available_presets().len(),
            avatar.available_presets()
        );

        let meshes = avatar.skin(&[]); // bind pose to seed the GPU meshes
        let prims = avatar.primitives();
        self.n_prims = prims.len();

        let mut renderer = Renderer::new(ctx.device.clone(), ctx.queue.clone(), ctx.surface_format, w, h)
            .expect("renderer");
        renderer.set_clear_rgba(wgpu::Color { r: 0.08, g: 0.08, b: 0.10, a: 1.0 });
        renderer.update_lights(
            [0.62, 0.62, 0.66],
            &[
                Light::directional([0.3, -0.6, 0.5], [1.0, 1.0, 1.0], 0.85),
                Light::directional([-0.4, 0.5, 0.25], [0.6, 0.6, 0.7], 0.4),
            ],
            false,
        );

        // AABB (seimei space: Z-up, mm) for camera framing.
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for m in &meshes {
            for v in &m.vertices {
                let p = [v.position.x as f32, v.position.y as f32, v.position.z as f32];
                for k in 0..3 {
                    min[k] = min[k].min(p[k]);
                    max[k] = max[k].max(p[k]);
                }
            }
        }
        let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
        let radius = 0.5 * (max[0] - min[0]).max(max[2] - min[2]).max(1.0);

        // Register textures + meshes; opaque first, blend last.
        let mut opaque: Vec<(String, InstanceData)> = Vec::new();
        let mut blend: Vec<(String, InstanceData)> = Vec::new();
        for (i, (mesh, prim)) in meshes.iter().zip(prims).enumerate() {
            let mesh_id = format!("vrm{i}");
            let tex_id = prim.texture.as_ref().map(|t| {
                let tid = format!("vtex{i}");
                renderer.register_texture_rgba(&tid, t.width, t.height, &t.rgba);
                tid
            });
            renderer.add_mesh(&mesh_id, mesh, tex_id.clone());
            let bc = prim.base_color;
            let inst = InstanceData {
                model: Mat4::IDENTITY.to_cols_array_2d(),
                color: if tex_id.is_some() { [1.0, 1.0, 1.0, bc[3]] } else { bc },
                material: [prim.metallic, prim.roughness.max(0.5), 0.0, 0.0],
            };
            if prim.alpha_blend {
                blend.push((mesh_id, inst));
            } else {
                opaque.push((mesh_id, inst));
            }
        }
        self.opaque_count = opaque.len();
        let mut instances = opaque;
        instances.extend(blend);
        self.instances = instances;

        // Orbit camera. azim=0 faces front: VRM 0.x is +Y, VRM 1.0 is -Y (π offset).
        let fov = 32.0_f64;
        self.dist = (radius as f64) / (fov.to_radians() * 0.5).tan() * 1.15;
        self.azim_front = if avatar.is_vrm0() { 0.0 } else { std::f64::consts::PI };
        self.center = [center[0] as f64, center[1] as f64, center[2] as f64];
        self.cam.fov = fov;
        self.cam.aspect = w as f64 / h as f64;
        self.cam.near = (self.dist * 0.05).max(1.0);
        self.cam.far = self.dist * 4.0 + radius as f64 * 4.0;

        self.renderer = Some(renderer);
        self.ctrl = Some(AvatarController::new(avatar));
    }

    fn on_resize(&mut self, ctx: &GpuContext) {
        self.cam.aspect = ctx.surface_width as f64 / ctx.surface_height.max(1) as f64;
    }

    fn render_scene(&mut self, ctx: &mut SceneRenderContext) {
        // Advance avatar motion and push the new meshes to the GPU.
        if let Some(c) = self.ctrl.as_mut() {
            let t0 = Instant::now();
            let meshes = c.update(DT); // CPU skinning + spring + morphs
            self.skin_accum_us += t0.elapsed().as_micros();
            if let Some(r) = self.renderer.as_mut() {
                for (i, m) in meshes.iter().enumerate().take(self.n_prims) {
                    r.update_mesh_vertices(&format!("vrm{i}"), m);
                }
            }
        }
        // FPS + average CPU-skin time, printed to stderr once a second.
        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            let fps = self.frame_count as f32 / elapsed;
            let skin_ms = self.skin_accum_us as f32 / self.frame_count as f32 / 1000.0;
            eprintln!("[viewer] fps: {fps:.1}  skin: {skin_ms:.2}ms/frame  prims: {}", self.n_prims);
            self.frame_count = 0;
            self.skin_accum_us = 0;
            self.fps_timer = Instant::now();
        }
        self.place_camera();
        if let Some(r) = self.renderer.as_mut() {
            r.render_to_view(
                ctx.encoder,
                ctx.surface_view,
                ctx.depth_view,
                &self.cam,
                &self.instances,
                self.opaque_count,
            );
        }
    }
}

fn main() {
    let vrm_path = std::env::args().nth(1).expect("usage: vrm-viewer <file.vrm>");
    eprintln!("buttons at the bottom; drag the avatar to orbit.");
    sabitori::run_scene(ViewerApp::new(vrm_path));
}
