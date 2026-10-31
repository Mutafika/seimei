//! 影の深度パスを前のフレームのまま使う（描き直さない）とき、絵が描き直したときと同じになるか。
//!
//! 1 つのレンダラーで状態を順に変えながら描いた絵と、その状態を新しいレンダラーで一から描いた絵を、画素の
//! バイトで比べる（新しいレンダラーは必ず影の深度を描くので真値になる）。変えるのは、部材を動かす・形を
//! 差し替える・色だけ変える（影は描き直さなくてよい）・影から外す・形を消す・影の箱を動かす。
//! アダプタが取れない環境では何もせずに通す。

use seimei::math::{Point3, Vec3D};
use seimei::{Camera, InstanceData, Light, MsaaSamples, QualitySettings, Renderer, ShadowQuality};
use std::sync::Arc;

const W: u32 = 192;
const H: u32 = 128;

fn device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

/// 1 つの状態（シーンの中身と影の設定）
#[derive(Clone)]
struct State {
    /// 箱の位置
    box_at: [f32; 3],
    /// 箱の高さ（形そのものを差し替える）
    box_height: f32,
    /// 箱の色（影の深度には効かない）
    box_color: [f32; 4],
    /// 箱を影から外す
    exclude_box: bool,
    /// 箱の形を登録していない（インスタンスは残る＝描かれない）
    box_removed: bool,
    /// 影の箱の半幅
    light_half: f32,
}

fn box_mesh(height: f32) -> seimei::RenderMesh {
    let mut m = seimei::procedural::cube(1.0);
    for v in &mut m.vertices {
        v.position = Point3::new(v.position.x * 400.0, v.position.y * 400.0, (v.position.z + 0.5) * height as f64);
    }
    m
}

fn camera() -> Camera {
    let mut c = Camera::new();
    c.aspect = W as f64 / H as f64;
    c.position = Point3::new(-2500.0, -3500.0, 2500.0);
    c.target = Point3::new(0.0, 0.0, 0.0);
    c.up = Vec3D::new(0.0, 0.0, 1.0);
    c.fov = 45.0;
    c.near = 10.0;
    c.far = 50_000.0;
    c
}

fn light_vp(half: f32) -> [[f32; 4]; 4] {
    let eye = glam::Vec3::new(3000.0, -4000.0, 8500.0);
    let view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Z);
    let proj = glam::Mat4::orthographic_rh(-half, half, -half, half, 1.0, 30_000.0);
    (proj * view).to_cols_array_2d()
}

fn new_renderer(dev: &(Arc<wgpu::Device>, Arc<wgpu::Queue>)) -> Renderer {
    let mut r = Renderer::new(dev.0.clone(), dev.1.clone(), wgpu::TextureFormat::Rgba8Unorm, W, H).expect("renderer");
    r.set_quality(QualitySettings {
        msaa: MsaaSamples::Off,
        shadow: ShadowQuality::Low,
        ssao: false,
        bloom: false,
        ssr: false,
        dof: false,
        edge_bevel: false,
        ..QualitySettings::default()
    })
    .expect("quality");
    r.update_lights([0.2, 0.2, 0.2], &[Light::directional([0.3, -0.4, 0.85], [1.0, 1.0, 1.0], 3.0)], false);
    r.add_mesh("ground", &seimei::procedural::cube(1.0), None);
    r
}

/// 状態をレンダラーに当てる（`prev` から変わった所だけ。一から作るときは `prev = None`）
fn apply(r: &mut Renderer, s: &State, prev: Option<&State>) {
    if prev.is_none_or(|p| p.box_height != s.box_height || p.box_removed != s.box_removed) {
        if s.box_removed {
            r.remove_mesh("box");
        } else if prev.is_some_and(|p| !p.box_removed) {
            r.update_mesh_vertices("box", &box_mesh(s.box_height));
        } else {
            r.add_mesh("box", &box_mesh(s.box_height), None);
        }
    }
    if prev.is_none_or(|p| p.exclude_box != s.exclude_box) {
        r.set_shadow_exclude(if s.exclude_box { vec!["box"] } else { Vec::new() });
    }
    r.update_shadow_matrix(light_vp(s.light_half));
}

fn draw(r: &mut Renderer, s: &State) -> Vec<u8> {
    let ground = InstanceData::from_transform([0.0, 0.0, -50.0], 0.0, [6000.0, 6000.0, 100.0], [0.8, 0.8, 0.8, 1.0]);
    let bx = InstanceData::from_transform(s.box_at, 0.0, [1.0, 1.0, 1.0], s.box_color);
    let inst = vec![("ground".to_string(), ground), ("box".to_string(), bx)];
    r.render_offscreen(&camera(), &inst, inst.len(), W, H).expect("render")
}

#[test]
fn reused_shadow_depth_matches_a_fresh_render() {
    let Some(dev) = device() else {
        eprintln!("GPU が取れない");
        return;
    };
    let s0 = State {
        box_at: [0.0, 0.0, 0.0],
        box_height: 1200.0,
        box_color: [0.9, 0.3, 0.2, 1.0],
        exclude_box: false,
        box_removed: false,
        light_half: 4000.0,
    };
    let steps: Vec<(&str, State)> = vec![
        ("最初", s0.clone()),
        ("同じまま（影を描き直さない）", s0.clone()),
        ("箱を動かす", State { box_at: [700.0, 500.0, 0.0], ..s0.clone() }),
        ("箱の形を差し替える", State { box_at: [700.0, 500.0, 0.0], box_height: 2500.0, ..s0.clone() }),
        ("色だけ変える", State { box_at: [700.0, 500.0, 0.0], box_height: 2500.0, box_color: [0.2, 0.4, 0.9, 1.0], ..s0.clone() }),
        ("影から外す", State { box_at: [700.0, 500.0, 0.0], box_height: 2500.0, exclude_box: true, ..s0.clone() }),
        ("影に戻す", State { box_at: [700.0, 500.0, 0.0], box_height: 2500.0, ..s0.clone() }),
        ("影の箱を動かす", State { box_at: [700.0, 500.0, 0.0], box_height: 2500.0, light_half: 5000.0, ..s0.clone() }),
        ("形を消す", State { box_at: [700.0, 500.0, 0.0], box_height: 2500.0, box_removed: true, light_half: 5000.0, ..s0.clone() }),
        ("形を戻す", State { box_at: [700.0, 500.0, 0.0], box_height: 2500.0, light_half: 5000.0, ..s0.clone() }),
    ];
    let mut reused = new_renderer(&dev);
    let mut prev: Option<State> = None;
    let mut images: Vec<Vec<u8>> = Vec::new();
    for (name, s) in &steps {
        apply(&mut reused, s, prev.as_ref());
        let got = draw(&mut reused, s);
        let mut fresh = new_renderer(&dev);
        apply(&mut fresh, s, None);
        let want = draw(&mut fresh, s);
        let differ = got.chunks_exact(4).zip(want.chunks_exact(4)).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "{name}: 前の影の深度を使った絵が、一から描いた絵と {differ} 画素違う");
        images.push(got);
        prev = Some(s.clone());
    }
    // 影が実際に写っている（影から外すと絵が変わる＝この試験が影の深度を見ている）
    let shadowed = &images[4];
    let excluded = &images[5];
    let changed = shadowed.chunks_exact(4).zip(excluded.chunks_exact(4)).filter(|(a, b)| a != b).count();
    assert!(changed > 200, "箱の影が写っていない（影から外しても {changed} 画素しか変わらない）");
}
