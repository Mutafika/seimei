//! 形が全部違うメッシュをたくさん描くときの 1 フレームの CPU の時間（計測用・`#[ignore]`）。
//!
//! IFC から取り込んだ建物は部材ごとに形が違う（メッシュを共有しない）ので、部材の数だけ描画命令が出る。
//! 2.4 万部材で 1 フレームの描画命令に約 100ms かかった（bamiri・2026-10-31）。ここではそれを seimei だけで
//! 再現し、描画命令を記録する時間（`render_to_view`）・submit・GPU の完了待ちを分けて測る。
//! 比べるのは「全部違う形」と「全部同じ形（インスタンシングで 1 命令）」、影なし・影 4 カスケード（止まっている＝
//! 影の深度パスを前のフレームのまま使える）・影 4 カスケードで影の箱が毎フレーム動く（オービット＝描き直す）。
//!
//! 実行: `cargo test --release --test many_meshes_bench -- --ignored --nocapture`
//! （部材の数は `SEIMEI_BENCH_N`・既定 24000）

use seimei::math::{Point3, Vec3D};
use seimei::{Camera, InstanceData, Light, MsaaSamples, QualitySettings, Renderer, ShadowQuality};
use std::sync::Arc;
use std::time::Instant;

const W: u32 = 1280;
const H: u32 = 800;

struct Gpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()?;
    Some(Gpu { device: Arc::new(device), queue: Arc::new(queue) })
}

fn target(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[derive(Clone, Copy, PartialEq)]
enum Shadow {
    Off,
    Still,
    Moving,
}

/// 1 つの組み合わせを測る: (記録 ms, submit ms, GPU 待ち ms) の中央値
fn measure(g: &Gpu, n: usize, distinct: bool, shadow: Shadow) -> (f64, f64, f64) {
    let mut r = Renderer::new(g.device.clone(), g.queue.clone(), wgpu::TextureFormat::Rgba8Unorm, W, H).expect("renderer");
    r.set_quality(QualitySettings {
        msaa: MsaaSamples::Off,
        shadow: if shadow == Shadow::Off { ShadowQuality::Off } else { ShadowQuality::Low },
        ssao: false,
        bloom: false,
        ssr: false,
        dof: false,
        edge_bevel: false,
        ..QualitySettings::default()
    })
    .expect("quality");
    let side = (n as f64).sqrt().ceil() as usize;
    let spacing = 300.0_f32;
    let extent = side as f32 * spacing;
    // 形は部材ごとに少しずつ違う箱（共有しない）か、全部同じ箱
    let mut instances = Vec::with_capacity(n);
    for i in 0..n {
        let id = if distinct { format!("m{i}") } else { "m0".to_string() };
        if distinct || i == 0 {
            r.add_mesh(&id, &seimei::procedural::cube(100.0 + (i % 50) as f32), None);
        }
        let (x, y) = ((i % side) as f32 * spacing, (i / side) as f32 * spacing);
        instances.push((id, InstanceData::from_transform([x, y, 0.0], 0.0, [1.0, 1.0, 1.0], [0.8, 0.8, 0.8, 1.0])));
    }
    let c = extent / 2.0;
    let mut cam = Camera::new();
    cam.aspect = W as f64 / H as f64;
    cam.position = Point3::new(c as f64, -(extent as f64) * 0.3, extent as f64 * 0.9);
    cam.target = Point3::new(c as f64, c as f64, 0.0);
    cam.up = Vec3D::new(0.0, 0.0, 1.0);
    cam.fov = 50.0;
    cam.near = 10.0;
    cam.far = extent as f64 * 4.0;
    // 4 カスケードとも全体を覆う（全部材が 4 回とも影の深度パスに入る＝いちばん重い場合）。
    // 動く場合は箱の大きさをフレームごとに少し変える（カメラに合わせて箱が動くオービットの代わり）
    let cascades = |frame: usize| {
        let eye = glam::Vec3::new(c + 0.3 * extent, c - 0.4 * extent, 0.85 * extent);
        let view = glam::Mat4::look_at_rh(eye, glam::Vec3::new(c, c, 0.0), glam::Vec3::Z);
        let half = extent * (0.75 + 0.001 * frame as f32);
        let proj = glam::Mat4::orthographic_rh(-half, half, -half, half, 1.0, extent * 3.0);
        [(proj * view).to_cols_array_2d(); 4]
    };
    if shadow != Shadow::Off {
        r.update_lights([0.3, 0.3, 0.3], &[Light::directional([0.3, -0.4, 0.85], [1.0, 1.0, 1.0], 3.0)], false);
        r.update_shadow_cascades(&cascades(0), [f32::MAX; 4], [0.0015; 4]);
    }
    let color = target(&g.device, wgpu::TextureFormat::Rgba8Unorm);
    let depth = target(&g.device, wgpu::TextureFormat::Depth32Float);
    let (mut rec, mut sub, mut wait) = (Vec::new(), Vec::new(), Vec::new());
    let frames: usize = std::env::var("SEIMEI_BENCH_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    for frame in 0..frames {
        if shadow == Shadow::Moving {
            r.update_shadow_cascades(&cascades(frame), [f32::MAX; 4], [0.0015; 4]);
        }
        let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let t0 = Instant::now();
        r.render_to_view(&mut enc, &color, &depth, &cam, &instances, instances.len());
        let t1 = Instant::now();
        g.queue.submit(std::iter::once(enc.finish()));
        let t2 = Instant::now();
        g.device.poll(wgpu::Maintain::Wait);
        let t3 = Instant::now();
        if frame >= 2 {
            rec.push((t1 - t0).as_secs_f64() * 1000.0);
            sub.push((t2 - t1).as_secs_f64() * 1000.0);
            wait.push((t3 - t2).as_secs_f64() * 1000.0);
        }
    }
    let (drawn, total) = r.cull_stats();
    assert_eq!(drawn, total, "全部が画面に入っている（視錐台で間引かれていない）");
    (median(rec), median(sub), median(wait))
}

#[test]
#[ignore = "計測用。cargo test --release --test many_meshes_bench -- --ignored --nocapture"]
fn many_distinct_meshes_frame_cost() {
    let Some(g) = gpu() else {
        eprintln!("GPU が取れない");
        return;
    };
    let n: usize = std::env::var("SEIMEI_BENCH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(24_000);
    println!("部材 {n}・{W}×{H}・MSAA なし（ms・10 フレームの中央値）");
    println!("| 形 | 影 | 記録（render_to_view） | submit | GPU 待ち |");
    println!("|---|---|---:|---:|---:|");
    for distinct in [true, false] {
        for shadow in [Shadow::Off, Shadow::Still, Shadow::Moving] {
            // 1 つだけ測る（呼び出しの木を採るとき）: SEIMEI_BENCH_ONLY=still 等
            if let Ok(only) = std::env::var("SEIMEI_BENCH_ONLY") {
                let name = match shadow {
                    Shadow::Off => "off",
                    Shadow::Still => "still",
                    Shadow::Moving => "moving",
                };
                if !distinct || only != name {
                    continue;
                }
            }
            let (rec, sub, wait) = measure(&g, n, distinct, shadow);
            println!(
                "| {} | {} | {rec:.1} | {sub:.1} | {wait:.1} |",
                if distinct { "全部違う" } else { "全部同じ" },
                match shadow {
                    Shadow::Off => "なし",
                    Shadow::Still => "4 カスケード・止まっている",
                    Shadow::Moving => "4 カスケード・毎フレーム動く",
                }
            );
        }
    }
}
