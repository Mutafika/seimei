//! メッシュを共有のページに詰めても（v0.5.0）、足す・消す・大きさの違う形に差し替えるを繰り返した後の絵が、
//! 最後に残った形だけを新しいレンダラーに足して描いた絵と、画素のバイトで一致するか。
//!
//! 空きの使い回し（消した範囲・縮んだ形の書き直し）や、ページをまたぐ割り当てで、ほかのメッシュの頂点を
//! 踏むと絵が崩れる。形は箱を何個か重ねたもの（頂点の数がばらばら）で、全部で最初のページ（1 万 6 千頂点）を超える。
//! アダプタが取れない環境では何もせずに通す。

use seimei::math::{Point3, Vec3D};
use seimei::{Camera, InstanceData, MsaaSamples, QualitySettings, RenderMesh, Renderer, ShadowQuality};
use std::collections::BTreeMap;
use std::sync::Arc;

const W: u32 = 256;
const H: u32 = 256;

fn device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

fn new_renderer(dev: &(Arc<wgpu::Device>, Arc<wgpu::Queue>)) -> Renderer {
    let mut r = Renderer::new(dev.0.clone(), dev.1.clone(), wgpu::TextureFormat::Rgba8Unorm, W, H).expect("renderer");
    r.set_quality(QualitySettings {
        msaa: MsaaSamples::Off,
        shadow: ShadowQuality::Off,
        ssao: false,
        bloom: false,
        ssr: false,
        dof: false,
        edge_bevel: false,
        ..QualitySettings::default()
    })
    .expect("quality");
    r
}

/// 箱を `k` 個、斜めにずらして重ねた形（頂点 24k 個）。`seed` で大きさを変える
fn stack(k: u32, seed: u32) -> RenderMesh {
    let mut m = RenderMesh::new();
    for i in 0..k {
        let mut c = seimei::procedural::cube(10.0 + ((seed + i) % 7) as f32 * 3.0);
        for v in &mut c.vertices {
            v.position = Point3::new(v.position.x + i as f64 * 4.0, v.position.y + i as f64 * 2.0, v.position.z + i as f64 * 5.0);
        }
        m.merge(&c);
    }
    m
}

fn draw(r: &mut Renderer, meshes: &BTreeMap<String, (u32, u32)>) -> Vec<u8> {
    let mut cam = Camera::new();
    cam.aspect = 1.0;
    cam.position = Point3::new(800.0, 800.0, 3000.0);
    cam.target = Point3::new(800.0, 800.0, 0.0);
    cam.up = Vec3D::new(0.0, 1.0, 0.0);
    cam.is_orthographic = true;
    cam.ortho_width = 1800.0;
    cam.near = 1.0;
    cam.far = 10_000.0;
    // 形ごとに格子の 1 マス・色も形ごと
    let inst: Vec<(String, InstanceData)> = meshes
        .keys()
        .enumerate()
        .map(|(n, id)| {
            let (x, y) = ((n % 20) as f32 * 85.0, (n / 20) as f32 * 85.0);
            let c = [(n % 5) as f32 / 5.0, (n % 7) as f32 / 7.0, (n % 3) as f32 / 3.0, 1.0];
            (id.clone(), InstanceData::from_transform([x, y, 0.0], 0.0, [1.0, 1.0, 1.0], c))
        })
        .collect();
    r.render_offscreen(&cam, &inst, inst.len(), W, H).expect("render")
}

#[test]
fn meshes_packed_into_pages_survive_add_remove_and_resize() {
    let Some(dev) = device() else {
        eprintln!("GPU が取れない");
        return;
    };
    let mut seed = 7u64;
    let mut rnd = |n: u32| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as u32) % n
    };
    let mut r = new_renderer(&dev);
    // id → (箱の数, 大きさの種)
    let mut live: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    let mut next = 0u32;
    let mut total_vertices = 0u32;
    for _ in 0..1200 {
        match rnd(4) {
            // 足す
            0 | 1 => {
                let (k, s) = (1 + rnd(40), rnd(100));
                let id = format!("m{next}");
                next += 1;
                r.add_mesh(&id, &stack(k, s), None);
                total_vertices += 24 * k;
                live.insert(id, (k, s));
            }
            // 消す
            2 if !live.is_empty() => {
                let id = live.keys().nth(rnd(live.len() as u32) as usize).cloned().unwrap();
                r.remove_mesh(&id);
                live.remove(&id);
            }
            // 大きさの違う形に差し替える（大きくなればページの別の所へ・小さくなればその場で書き直す）
            _ if !live.is_empty() => {
                let id = live.keys().nth(rnd(live.len() as u32) as usize).cloned().unwrap();
                let (k, s) = (1 + rnd(40), rnd(100));
                r.update_mesh_vertices(&id, &stack(k, s));
                total_vertices += 24 * k;
                live.insert(id, (k, s));
            }
            _ => {}
        }
    }
    // 画面に収まる数まで消す（格子 20×20）
    while live.len() > 400 {
        let id = live.keys().next().cloned().unwrap();
        r.remove_mesh(&id);
        live.remove(&id);
    }
    assert!(total_vertices > 16 * 1024 * 4, "最初のページを何枚分も超えて割り当てた（{total_vertices} 頂点）");
    let got = draw(&mut r, &live);

    let mut fresh = new_renderer(&dev);
    for (id, (k, s)) in &live {
        fresh.add_mesh(id, &stack(*k, *s), None);
    }
    let want = draw(&mut fresh, &live);
    let differ = got.chunks_exact(4).zip(want.chunks_exact(4)).filter(|(a, b)| a != b).count();
    assert_eq!(differ, 0, "足す・消す・差し替えを繰り返したレンダラーの絵が、一から足した絵と {differ} 画素違う");
    let bg = &want[0..4];
    let drawn = want.chunks_exact(4).filter(|p| p != &bg).count();
    assert!(drawn > (W * H / 10) as usize, "形がほとんど写っていない（{drawn} 画素）");
}
