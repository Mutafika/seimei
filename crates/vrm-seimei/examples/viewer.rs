//! Live VRM viewer — the dynamic-render prototype.
//!
//!   cargo run --release -p vrm-seimei --example viewer -- /path/to/model.vrm
//!
//! Opens a real window and renders the avatar **per frame** (not once). Two motion
//! channels are live here:
//!   * channel 1 (root transform): a turntable spin — no skinning, no LBS tear.
//!   * channel 2 (skeletal clips): an `AnimationPlayer` drives bone rotations;
//!     each frame we `sample() → skin() → update_mesh_vertices`.
//!
//! This is the iteration surface for motion: tune a clip, relaunch, watch. The
//! built-in `walk_cycle`/`run_cycle` were authored for a Mixamo/RPM rig, so their
//! swing axes may not match a VRM rig exactly — seeing that and tuning is the point.
//!
//! Keys: 0 bind · 1 idle · 2 walk · 3 run · Space pause · R spin · Esc quit.

use std::sync::Arc;

use glam::Mat4;
use seimei::{Camera, InstanceData, Light, Point3, Renderer};
use vrm_anatomy::animation::{AnimationClip, AnimationPlayer, BoneTrack, Easing, Keyframe, LoopMode};
use vrm_seimei::VrmAvatar;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use egui_wgpu::ScreenDescriptor;

/// Radians added to the spin each frame (~0.6°/frame ≈ a slow turntable at 60fps).
const SPIN_PER_FRAME: f32 = 0.01;
/// Fixed animation step. We drive a continuous redraw loop, so a constant ~60fps
/// dt is fine for a viewer (no wall clock needed).
const DT: f32 = 1.0 / 60.0;
/// How far to lower the arms from the bind T-pose toward the sides (0 = T-pose,
/// 1 = straight down). The earlier "mangled" arms-down was the multi-skin bug, not
/// LBS — with skinning correct this should hang cleanly. Tune by eye.
const ARM_DOWN: f32 = 0.72;

/// A control action — emitted by both the egui buttons and the keyboard.
enum Act {
    Bind,
    Idle,
    Walk,
    Run,
    Pause,
    Spin,
    Spring,
    TurnL,
    TurnR,
}

fn onoff(b: bool) -> &'static str {
    if b { "on" } else { "off" }
}

/// The built-in clips use Mixamo/RPM bone names; our `skin()` keys on VRM humanoid
/// names. Remap the bones the locomotion clips touch; pass anything else through,
/// so a clip already authored in humanoid names (idle) is unaffected.
fn remap(name: &str) -> &str {
    match name {
        "Hips" => "hips",
        "Spine" => "spine",
        "Spine1" | "Spine2" => "chest",
        "LeftUpLeg" => "leftUpperLeg",
        "RightUpLeg" => "rightUpperLeg",
        "LeftLeg" => "leftLowerLeg",
        "RightLeg" => "rightLowerLeg",
        "LeftFoot" => "leftFoot",
        "RightFoot" => "rightFoot",
        "LeftArm" => "leftUpperArm",
        "RightArm" => "rightUpperArm",
        "LeftForeArm" => "leftLowerArm",
        "RightForeArm" => "rightLowerArm",
        other => other,
    }
}

/// A gentle "alive" idle, authored in VRM humanoid bone names: a slow breathing
/// nod on spine/chest plus a small hip sway. PingPong so it loops seamlessly.
/// Rotations are `[roll(X), pitch(Y), yaw(Z)]` rad; amplitudes are tiny on
/// purpose. If the sway tilts the wrong way, flip an axis index — that's the kind
/// of one-liner the viewer is for.
fn idle_clip() -> AnimationClip {
    let kf = |t: f32, r: [f32; 3]| Keyframe {
        time: t,
        rotation: r,
        easing: Easing::EaseInOut,
        stiffness: None,
        damping: None,
    };
    let track = |bone: &str, a: [f32; 3], b: [f32; 3]| BoneTrack {
        bone_name: bone.to_string(),
        keyframes: vec![kf(0.0, a), kf(1.5, b)],
    };
    AnimationClip {
        name: "idle".into(),
        duration: 1.5,
        loop_mode: LoopMode::PingPong,
        // Upper body only — NO hips. hips is the root, so any rotation there tilts
        // the whole figure rigidly (reads as leaning, not breathing).
        tracks: vec![
            track("spine", [0.0, 0.0, 0.0], [0.016, 0.0, 0.0]), // breathe (subtle nod)
            track("chest", [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]),
        ],
    }
}

/// A locomotion cycle (walk or run) authored in VRM humanoid bone names, tuned
/// for this rig: hips swing the legs front/back on X, knees flex on **−X** (the
/// natural direction for this VRM — +X hyperextends, per the offline probe). Same
/// sin/knee shape as vrm-anatomy's built-in cycles, with the corrected knee sign
/// and humanoid names (so the viewer's remap leaves it untouched). `dur` controls
/// cadence; `hip`/`knee`/`foot` are the swing amplitudes (knee negative = −X).
fn locomotion_clip(name: &str, dur: f32, hip: f32, knee: f32, foot: f32) -> AnimationClip {
    let n = 8;
    let two_pi = 2.0 * std::f32::consts::PI;
    let sin_track = |bone: &str, axis: usize, amp: f32, phase: f32| {
        let kfs = (0..=n)
            .map(|i| {
                let t = dur * i as f32 / n as f32;
                let mut rot = [0.0f32; 3];
                rot[axis] = amp * (two_pi * (t / dur + phase)).sin();
                Keyframe { time: t, rotation: rot, easing: Easing::Linear, stiffness: None, damping: None }
            })
            .collect();
        BoneTrack { bone_name: bone.into(), keyframes: kfs }
    };
    // One-directional flex (bend ≥ 0) during the swing phase, on −X.
    let knee_track = |bone: &str, amp: f32, phase: f32| {
        let kfs = (0..=n)
            .map(|i| {
                let t = dur * i as f32 / n as f32;
                let p = two_pi * (t / dur + phase) + std::f32::consts::FRAC_PI_2;
                let bend = amp * p.sin().max(0.0);
                Keyframe { time: t, rotation: [bend, 0.0, 0.0], easing: Easing::Linear, stiffness: None, damping: None }
            })
            .collect();
        BoneTrack { bone_name: bone.into(), keyframes: kfs }
    };
    // Body sway, scaled to the gait: the pelvis twists on Y (this also swings the
    // skirt springs), and the upper body counter-twists (swinging the hair). hips Y
    // is the vertical-yaw axis (per the offline probe).
    let pelvis_yaw = 0.25 * hip;
    let spine_yaw = -0.15 * hip;
    AnimationClip {
        name: name.into(),
        duration: dur,
        loop_mode: LoopMode::Loop,
        tracks: vec![
            sin_track("leftUpperLeg", 0, hip, 0.0),
            sin_track("rightUpperLeg", 0, hip, 0.5),
            knee_track("leftLowerLeg", knee, 0.0), // knee < 0 → −X = natural flex
            knee_track("rightLowerLeg", knee, 0.5),
            sin_track("leftFoot", 0, foot, 0.25),
            sin_track("rightFoot", 0, foot, 0.75),
            sin_track("hips", 1, pelvis_yaw, 0.0),  // pelvis twist (sways the skirt)
            sin_track("spine", 1, spine_yaw, 0.0),  // counter-twist (sways the hair)
        ],
    }
}

fn make_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer-depth"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

/// Everything tied to the live window: surface, seimei renderer, the avatar (kept
/// so we can re-skin each frame), the clip player, and view state.
struct Gpu {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    depth: wgpu::TextureView,
    renderer: Renderer,
    avatar: VrmAvatar,
    n_prims: usize,
    instances: Vec<(String, InstanceData)>,
    opaque_count: usize,
    // Orbit camera: spherical around `center`. azim=0 faces the avatar's front.
    cam: Camera,
    center: [f64; 3],
    dist: f64,
    azim: f64,
    elev: f64,
    azim_front: f64, // 0 for VRM 0.x (+Y front), π for VRM 1.0 (-Y front)
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    player: AnimationPlayer,
    paused: bool,
    spinning: bool,
    body_yaw: f32,    // accumulated hips-Y rotation when "Spin" turns the body
    gait_period: f32, // current locomotion period (s); 0 = no walk → no arm swing
    // egui button bar overlay
    window: Arc<Window>,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

impl Gpu {
    async fn new(window: Arc<Window>, vrm_path: &str) -> Gpu {
        let size = window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no adapter");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("viewer"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("no device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // seimei writes already-lit linear colour, so pick a non-sRGB surface format
        // (an sRGB surface would gamma-encode it and wash the avatar out).
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| matches!(f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let depth = make_depth(&device, w, h);

        // --- load the avatar; skin the bind pose once to seed the GPU meshes ---
        let bytes = std::fs::read(vrm_path).expect("read vrm");
        let avatar = VrmAvatar::load(&bytes).expect("load vrm");
        eprintln!(
            "[viewer] spring joints (揺れもの): {}  colliders: {}",
            avatar.spring_joints(),
            avatar.spring_colliders()
        );
        let meshes = avatar.skin(&[]); // bind pose to start
        let prims = avatar.primitives();
        let n_prims = prims.len();

        let mut renderer =
            Renderer::new(device.clone(), queue.clone(), format, w, h).expect("renderer");
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
        let opaque_count = opaque.len();
        let mut instances = opaque;
        instances.extend(blend);

        // Orbit camera. azim=0 faces the front: VRM 0.x front is +Y, VRM 1.0 is -Y
        // (π offset). Up is +Z. Position is recomputed each frame in place_camera.
        let fov = 32.0_f64;
        let dist = (radius as f64) / (fov.to_radians() * 0.5).tan() * 1.15;
        let azim_front = if avatar.is_vrm0() { 0.0 } else { std::f64::consts::PI };
        let mut cam = Camera::new();
        cam.fov = fov;
        cam.aspect = w as f64 / h as f64;
        cam.near = (dist * 0.05).max(1.0);
        cam.far = dist * 4.0 + radius as f64 * 4.0;

        // Start in idle so the window opens already "alive".
        let mut player = AnimationPlayer::new();
        player.play(idle_clip());

        // egui setup (button bar over the 3D render).
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        Gpu {
            surface,
            device,
            queue,
            config,
            depth,
            renderer,
            avatar,
            n_prims,
            instances,
            opaque_count,
            cam,
            center: [center[0] as f64, center[1] as f64, center[2] as f64],
            dist,
            azim: 0.0,
            elev: 0.0,
            azim_front,
            dragging: false,
            last_cursor: None,
            player,
            paused: false,
            spinning: false,
            body_yaw: 0.0,
            gait_period: 0.0,
            window,
            egui_ctx,
            egui_state,
            egui_renderer,
        }
    }

    /// Apply a control action (from a button or a key).
    fn apply(&mut self, a: Act) {
        match a {
            Act::Bind => {
                self.player.stop();
                self.gait_period = 0.0;
            }
            Act::Idle => {
                self.player.play(idle_clip());
                self.paused = false;
                self.gait_period = 0.0;
            }
            Act::Walk => {
                self.player.play(locomotion_clip("walk", 1.0, 0.45, -0.8, -0.15));
                self.paused = false;
                self.gait_period = 1.0;
            }
            Act::Run => {
                self.player.play(locomotion_clip("run", 0.6, 0.7, -1.2, -0.2));
                self.paused = false;
                self.gait_period = 0.6;
            }
            Act::Pause => self.paused = !self.paused,
            Act::Spin => self.spinning = !self.spinning,
            Act::Spring => {
                let on = !self.avatar.spring_enabled();
                self.avatar.set_spring_enabled(on);
            }
            Act::TurnL => self.azim -= 0.35,
            Act::TurnR => self.azim += 0.35,
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

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth = make_depth(&self.device, w, h);
        self.cam.aspect = w as f64 / h as f64;
    }

    fn render(&mut self) {
        // --- channel 2: advance the clip and re-skin the meshes ---
        if !self.paused {
            self.player.update(DT);
        }
        // "Spin" turns the BODY (a hips-Y / vertical yaw), not the camera — this
        // moves the hair/skirt anchors so the spring sim actually has something to
        // react to (the springs only sway when their anchor moves). Camera orbit
        // stays on mouse drag.
        if self.spinning {
            self.body_yaw += SPIN_PER_FRAME;
        }
        let raw = self.player.sample(); // Vec<(&str,[f32;3])>; empty when stopped → bind
        let mut pose: Vec<(&str, [f32; 3])> = raw.iter().map(|(n, r)| (remap(n), *r)).collect();
        // Add the Spin body yaw to the pelvis, merging with any clip pelvis sway.
        if let Some(h) = pose.iter_mut().find(|(n, _)| *n == "hips") {
            h.1[1] += self.body_yaw;
        } else {
            pose.push(("hips", [0.0, self.body_yaw, 0.0]));
        }
        // Lower the arms out of the T-pose, and swing them front/back (opposite
        // each other) when walking — the clips don't touch the arms.
        let (ls, rs) = if self.gait_period > 0.0 {
            let phase = self.player.current_time() / self.gait_period * std::f32::consts::TAU;
            let s = 0.45 * phase.sin();
            (-s, s) // arms opposite each other (and counter to the legs)
        } else {
            (0.0, 0.0)
        };
        pose.extend(self.avatar.arms_pose(ARM_DOWN, ls, rs));
        // skin_dynamic advances the spring-bone (揺れもの) sim by DT each frame.
        let meshes = self.avatar.skin_dynamic(&pose, DT);
        for (i, m) in meshes.iter().enumerate().take(self.n_prims) {
            self.renderer.update_mesh_vertices(&format!("vrm{i}"), m);
        }

        self.place_camera();

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("viewer") });
        self.renderer.render_to_view(
            &mut encoder,
            &view,
            &self.depth,
            &self.cam,
            &self.instances,
            self.opaque_count,
        );

        // --- egui button bar (drawn over the 3D render) ---
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let (spinning, paused, spring_on) = (self.spinning, self.paused, self.avatar.spring_enabled());
        let mut act: Option<Act> = None;
        let full = self.egui_ctx.run(raw_input, |ctx| {
            egui::TopBottomPanel::bottom("controls").show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Bind").clicked() { act = Some(Act::Bind); }
                    if ui.button("Idle").clicked() { act = Some(Act::Idle); }
                    if ui.button("Walk").clicked() { act = Some(Act::Walk); }
                    if ui.button("Run").clicked() { act = Some(Act::Run); }
                    if ui.button(if paused { "Resume" } else { "Pause" }).clicked() { act = Some(Act::Pause); }
                    if ui.button(format!("Spin: {}", onoff(spinning))).clicked() { act = Some(Act::Spin); }
                    if ui.button(format!("Spring: {}", onoff(spring_on))).clicked() { act = Some(Act::Spring); }
                    if ui.button("Turn <").clicked() { act = Some(Act::TurnL); }
                    if ui.button("Turn >").clicked() { act = Some(Act::TurnR); }
                });
            });
        });
        self.egui_state.handle_platform_output(&self.window, full.platform_output);
        let ppp = self.egui_ctx.pixels_per_point();
        let tris = self.egui_ctx.tessellate(full.shapes, ppp);
        for (id, delta) in &full.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
        }
        let screen = ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ppp,
        };
        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &tris, &screen);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, &tris, &screen);
        }
        for id in &full.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        if let Some(a) = act {
            self.apply(a);
        }
    }
}

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    vrm_path: String,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("vrm viewer"))
                .expect("create window"),
        );
        self.gpu = Some(pollster::block_on(Gpu::new(window.clone(), &self.vrm_path)));
        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Let egui see the event first; if it consumed it (a button/hover), skip
        // the orbit/keyboard handling below.
        let mut consumed = false;
        if let Some(g) = &mut self.gpu {
            consumed = g.egui_state.on_window_event(g.window.as_ref(), &event).consumed;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = &mut self.gpu {
                    g.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() && !consumed => {
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Space) => {
                        if let Some(g) = &mut self.gpu {
                            g.apply(Act::Pause);
                        }
                    }
                    Key::Character(ref s) => {
                        if let Some(g) = &mut self.gpu {
                            match s.as_str() {
                                "0" => g.apply(Act::Bind),
                                "1" => g.apply(Act::Idle),
                                "2" => g.apply(Act::Walk),
                                "3" => g.apply(Act::Run),
                                "r" | "R" => g.apply(Act::Spin),
                                "p" | "P" => g.apply(Act::Spring),
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    if let Some(g) = &mut self.gpu {
                        // Only start a drag if the press wasn't on the egui bar.
                        g.dragging = state == ElementState::Pressed && !consumed;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(g) = &mut self.gpu {
                    if g.dragging {
                        if let Some((lx, ly)) = g.last_cursor {
                            g.azim += (position.x - lx) * 0.01;
                            g.elev = (g.elev + (position.y - ly) * 0.01).clamp(-1.3, 1.3);
                        }
                    }
                    g.last_cursor = Some((position.x, position.y));
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(g) = &mut self.gpu {
                    g.render();
                }
                if let Some(w) = &self.window {
                    w.request_redraw(); // drive a continuous animation loop
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let vrm_path = std::env::args().nth(1).expect("usage: viewer <file.vrm>");
    eprintln!("buttons at the bottom; keys also work. drag=orbit · Esc quit");
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { vrm_path, ..Default::default() };
    event_loop.run_app(&mut app).expect("run");
}
