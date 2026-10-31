//! 一括描画（`multi_draw_indexed_indirect`）の絵が、1 メッシュ 1 命令で描いた絵と画素のバイトで一致するか。
//!
//! 形が全部違う物・同じ形が続く物（インスタンシング）・画面の外の物（視錐台で間引く）を混ぜ、影（4 カスケード）を
//! 落として、`set_multi_draw(true)` と `(false)` で同じ場面を描いて比べる。デバイスに一括描画の機能が無い環境
//! （アダプタが取れない・機能が無い）では何もせずに通す。

use seimei::math::{Point3, Vec3D};
use seimei::{Camera, InstanceData, Light, MsaaSamples, QualitySettings, Renderer, ShadowQuality, MULTI_DRAW_FEATURES};
use std::sync::Arc;

const W: u32 = 320;
const H: u32 = 200;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    if !adapter.features().contains(MULTI_DRAW_FEATURES) {
        eprintln!("一括描画の機能が無い");
        return None;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor { required_features: MULTI_DRAW_FEATURES, ..Default::default() },
        None,
    ))
    .ok()?;
    let mut r = Renderer::new(Arc::new(device), Arc::new(queue), wgpu::TextureFormat::Rgba8Unorm, W, H).ok()?;
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
    .ok()?;
    r.update_lights([0.25, 0.25, 0.25], &[Light::directional([0.3, -0.4, 0.85], [1.0, 1.0, 1.0], 3.0)], false);
    Some(r)
}

#[test]
fn multi_draw_matches_one_draw_per_mesh() {
    let Some(mut r) = renderer() else {
        return;
    };
    assert!(r.multi_draw(), "機能のあるデバイスでは既定で一括描画");
    r.add_mesh("ground", &seimei::procedural::cube(1.0), None);
    let mut inst: Vec<(String, InstanceData)> = vec![(
        "ground".to_string(),
        InstanceData::from_transform([0.0, 0.0, -50.0], 0.0, [20_000.0, 20_000.0, 100.0], [0.8, 0.8, 0.8, 1.0]),
    )];
    // 形が全部違う箱（大きさと高さを変える）・同じ形が 5 個ずつ続く柱・画面の外の箱
    for i in 0..300u32 {
        let id = format!("box{i}");
        let mut m = seimei::procedural::cube(1.0);
        let (sx, sz) = (100.0 + (i % 13) as f64 * 20.0, 200.0 + (i % 17) as f64 * 60.0);
        for v in &mut m.vertices {
            v.position = Point3::new(v.position.x * sx, v.position.y * sx, (v.position.z + 0.5) * sz);
        }
        r.add_mesh(&id, &m, None);
        let (x, y) = ((i % 20) as f32 * 450.0 - 4500.0, (i / 20) as f32 * 450.0 - 3000.0);
        let c = [(i % 5) as f32 / 5.0, (i % 7) as f32 / 7.0, (i % 3) as f32 / 3.0, 1.0];
        inst.push((id, InstanceData::from_transform([x, y, 0.0], (i % 4) as f32 * 0.3, [1.0, 1.0, 1.0], c)));
    }
    r.add_mesh("post", &seimei::procedural::cylinder(60.0, 1500.0, 12), None);
    for k in 0..40u32 {
        let (x, y) = ((k % 8) as f32 * 700.0 - 2800.0, (k / 8) as f32 * 900.0 + 3600.0);
        inst.push(("post".to_string(), InstanceData::from_transform([x, y, 0.0], 0.0, [1.0, 1.0, 1.0], [0.4, 0.3, 0.2, 1.0])));
    }
    for k in 0..30u32 {
        // 画面の外（カメラの後ろ）
        inst.push((format!("box{}", k * 7), InstanceData::from_transform([0.0, -60_000.0 - k as f32 * 500.0, 0.0], 0.0, [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0])));
    }
    let mut cam = Camera::new();
    cam.aspect = W as f64 / H as f64;
    cam.position = Point3::new(-2000.0, -9000.0, 6000.0);
    cam.target = Point3::new(0.0, 0.0, 0.0);
    cam.up = Vec3D::new(0.0, 0.0, 1.0);
    cam.fov = 50.0;
    cam.near = 10.0;
    cam.far = 100_000.0;
    let eye = glam::Vec3::new(3000.0, -4000.0, 8500.0);
    let view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Z);
    let vps: Vec<[[f32; 4]; 4]> = [4000.0_f32, 6000.0, 9000.0, 14000.0]
        .iter()
        .map(|h| (glam::Mat4::orthographic_rh(-h, *h, -h, *h, 1.0, 40_000.0) * view).to_cols_array_2d())
        .collect();
    r.update_shadow_cascades(&vps, [3000.0, 6000.0, 12000.0, f32::MAX], [0.0015; 4]);

    let opaque = inst.len();
    let multi = r.render_offscreen(&cam, &inst, opaque, W, H).expect("一括");
    r.set_multi_draw(false);
    assert!(!r.multi_draw());
    // 影の深度パスを前のフレームのまま使わないよう、影の箱を一度動かしてから戻す（どちらの経路でも影を描かせる）
    r.update_shadow_cascades(&vps[..3], [3000.0, 6000.0, f32::MAX, f32::MAX], [0.0015; 4]);
    let _ = r.render_offscreen(&cam, &inst, opaque, W, H).expect("影を描き直させる");
    r.update_shadow_cascades(&vps, [3000.0, 6000.0, 12000.0, f32::MAX], [0.0015; 4]);
    let single = r.render_offscreen(&cam, &inst, opaque, W, H).expect("1 メッシュ 1 命令");
    let differ = multi.chunks_exact(4).zip(single.chunks_exact(4)).filter(|(a, b)| a != b).count();
    assert_eq!(differ, 0, "一括描画の絵が 1 メッシュ 1 命令の絵と {differ} 画素違う");
    let bg = &single[0..4];
    let drawn = single.chunks_exact(4).filter(|p| p != &bg).count();
    assert!(drawn > (W * H / 4) as usize, "ほとんど写っていない（{drawn} 画素）");
    let (vis, total) = r.cull_stats();
    assert!(vis < total, "画面の外の物が間引かれていない（{vis}/{total}）");
}
