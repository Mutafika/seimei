//! 追加の描画ビュー（`RenderView`）を実 GPU で描いて画素を数える。
//!
//! - 既定のビューと追加のビュー 2 つを **1 回の submit** で描いても、それぞれ自分のカメラの絵になる
//!   （共有のカメラ uniform だと全部が最後に書いたカメラで描かれる）
//! - 追加のビューを描いても、既定のビューの絵と `cull_stats` は変わらない
//! - 追加のビューは影を落とさない（影の深度は既定のビューのカメラに合わせて焼いてある）
//! - 線チャンクの範囲・プレビュー線の有無をビューごとに選べる
//! - ビューだけの線（`update_view_lines`）はそのビューにだけ出て、面より手前・プレビュー線より下
//!
//! MSAA なし・4x・ポストプロセス（描画先が Rgba16Float になる）の全経路で縛る。
//! アダプタが取れない環境では何もせずに通す。

use seimei::math::{Point3, Vec3D};
use seimei::vertex::MODEL_BAKED;
use seimei::{
    Camera, InstanceData, Light, LineVertex, MsaaSamples, QualitySettings, RenderViewOptions, Renderer,
};
use std::sync::Arc;

const N: u32 = 160;
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()?;
    let mut r = Renderer::new(Arc::new(device), Arc::new(queue), wgpu::TextureFormat::Rgba8Unorm, N, N).ok()?;
    r.set_clear_color(0.0, 0.0, 0.0);
    r.add_mesh("cube", &seimei::procedural::cube(100.0), None);
    Some(r)
}

fn quality_paths() -> Vec<(&'static str, QualitySettings)> {
    let q = |msaa| QualitySettings {
        msaa,
        ssao: false,
        bloom: false,
        ssr: false,
        dof: false,
        edge_bevel: false,
        ..QualitySettings::default()
    };
    vec![
        ("MSAAなし", q(MsaaSamples::Off)),
        ("MSAA4x", q(MsaaSamples::X4)),
        ("ポストプロセス", QualitySettings { bloom: true, ..q(MsaaSamples::Off) }),
    ]
}

/// 真上から見下ろす正投影カメラ（Z が上・画面の上が +Y）
fn top(x: f64, y: f64, width: f64) -> Camera {
    let mut c = Camera::new();
    c.aspect = 1.0;
    c.position = Point3::new(x, y, 1000.0);
    c.target = Point3::new(x, y, 0.0);
    c.up = Vec3D::new(0.0, 1.0, 0.0);
    c.is_orthographic = true;
    c.ortho_width = width;
    c.near = 1.0;
    c.far = 5000.0;
    c
}

fn baked(pos: [f32; 3], scale: [f32; 3], color: [f32; 4]) -> (String, InstanceData) {
    let mut d = InstanceData::from_transform(pos, 0.0, scale, color);
    d.model_id = MODEL_BAKED;
    ("cube".to_string(), d)
}

/// (赤, 緑, 青) の画素数
fn count(img: &[u8]) -> (usize, usize, usize) {
    let mut n = (0, 0, 0);
    for p in img.chunks(4) {
        let (r, g, b) = (p[0], p[1], p[2]);
        if r > 150 && g < 100 && b < 100 {
            n.0 += 1;
        }
        if g > 150 && r < 100 && b < 100 {
            n.1 += 1;
        }
        if b > 150 && r < 100 && g < 100 {
            n.2 += 1;
        }
    }
    n
}

/// 既定のビューを呼び出し側の encoder で描くための描画先（`render_to_view` の外部 color/depth）
struct DefaultTarget {
    color: wgpu::Texture,
    depth: wgpu::Texture,
}

impl DefaultTarget {
    fn new(r: &Renderer) -> Self {
        let size = wgpu::Extent3d { width: N, height: N, depth_or_array_layers: 1 };
        let color = r.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test default color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = r.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test default depth"),
            size,
            mip_level_count: 1,
            sample_count: r.scene_sample_count(),
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self { color, depth }
    }

    fn read(&self, r: &Renderer) -> Vec<u8> {
        let padded = (N * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buf = r.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded * N) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(N) },
            },
            wgpu::Extent3d { width: N, height: N, depth_or_array_layers: 1 },
        );
        r.queue().submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        r.device().poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        (0..N as usize)
            .flat_map(|row| data[row * padded as usize..row * padded as usize + (N * 4) as usize].to_vec())
            .collect()
    }
}

/// 赤い箱（+X 側）と緑の箱（−X 側）。既定のビューは両方、ビュー A は赤だけ、ビュー B は緑だけを写す。
/// 3 つを 1 回の submit に積み、順番を入れ替えても、それぞれが自分のカメラの絵になる。
#[test]
fn views_in_one_submit_keep_their_own_cameras() {
    let Some(mut r) = renderer() else {
        eprintln!("GPU アダプタが無いのでスキップ");
        return;
    };
    let instances = [baked([300.0, 0.0, 0.0], [1.0; 3], RED), baked([-300.0, 0.0, 0.0], [1.0; 3], GREEN)];
    for (path, q) in quality_paths() {
        r.set_quality(q).expect("quality");
        let mut view_a = r.create_view(N, N);
        let mut view_b = r.create_view(N / 2, N);
        let opts = RenderViewOptions::default();
        for default_first in [true, false] {
            let target = DefaultTarget::new(&r);
            let color = target.color.create_view(&Default::default());
            let depth = target.depth.create_view(&Default::default());
            let mut enc = r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let wide = top(0.0, 0.0, 1000.0);
            if default_first {
                r.render_to_view(&mut enc, &color, &depth, &wide, &instances, instances.len());
            }
            r.render_view(&mut enc, &mut view_a, &top(300.0, 0.0, 300.0), &instances, instances.len(), &opts);
            let mut cam_b = top(-300.0, 0.0, 300.0);
            cam_b.aspect = 0.5;
            r.render_view(&mut enc, &mut view_b, &cam_b, &instances, instances.len(), &opts);
            if !default_first {
                r.render_to_view(&mut enc, &color, &depth, &wide, &instances, instances.len());
            }
            r.queue().submit(std::iter::once(enc.finish()));

            let label = format!("{path} 既定を{}", if default_first { "先" } else { "後" });
            let (dr, dg, _) = count(&target.read(&r));
            let (ar, ag, _) = count(&r.read_view_rgba(&view_a).unwrap());
            let (br, bg, _) = count(&r.read_view_rgba(&view_b).unwrap());
            eprintln!("{label}: 既定 赤{dr} 緑{dg} / A 赤{ar} 緑{ag} / B 赤{br} 緑{bg}");
            assert!(dr > 0 && dg > 0, "{label}: 既定のビューに両方の箱が写っていない");
            assert!(ar > 0 && ag == 0, "{label}: ビュー A は赤い箱だけのはず");
            assert!(bg > 0 && br == 0, "{label}: ビュー B は緑の箱だけのはず");
            // 箱の一辺 100 は幅 300 の視野の 1/3。正投影なので画素数は寸法から決まる
            let expect_a = (N as f64 / 3.0).powi(2);
            assert!((ar as f64 - expect_a).abs() < expect_a * 0.1, "{label}: A の赤 {ar} ≠ 約{expect_a:.0}");
            // B は横 N/2 px に 300、縦 N px に 600（aspect 0.5）を写す＝箱 100 は (N/2)/3 × N/6
            let expect_b = (N as f64 / 2.0 / 3.0) * (N as f64 / 6.0);
            assert!((bg as f64 - expect_b).abs() < expect_b * 0.1, "{label}: B の緑 {bg} ≠ 約{expect_b:.0}");
        }
    }
}

/// 追加のビューを描いても、既定のビューの絵と視錐台カリングの計測値は変わらない
#[test]
fn views_leave_the_default_view_alone() {
    let Some(mut r) = renderer() else {
        eprintln!("GPU アダプタが無いのでスキップ");
        return;
    };
    // 遠くの青い箱は既定のビューのカリングで落ちる（境界箱には 500 の余裕が付くので十分遠くに置く）
    let instances = [
        baked([300.0, 0.0, 0.0], [1.0; 3], RED),
        baked([-300.0, 0.0, 0.0], [1.0; 3], GREEN),
        baked([-20000.0, 0.0, 0.0], [1.0; 3], BLUE),
    ];
    for (path, q) in quality_paths() {
        r.set_quality(q).expect("quality");
        let cam = top(300.0, 0.0, 300.0);
        let before = r.render_offscreen(&cam, &instances, instances.len(), N, N).unwrap();
        let stats = r.cull_stats();
        assert!(stats.0 < stats.1, "{path}: 前提（既定のビューで遠くの箱が落ちる）{stats:?}");

        // 全部が見えるビューを描く
        let mut view = r.create_view(N, N);
        let mut enc = r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        r.render_view(&mut enc, &mut view, &top(-10000.0, 0.0, 30000.0), &instances, instances.len(), &Default::default());
        r.queue().submit(std::iter::once(enc.finish()));
        assert_eq!(r.cull_stats(), stats, "{path}: 追加のビューがカリングの計測値を上書きした");

        let after = r.render_offscreen(&cam, &instances, instances.len(), N, N).unwrap();
        assert!(before == after, "{path}: 追加のビューのあと既定のビューの絵が変わった");
    }
}

/// 既定のビューは影を落とし、追加のビューは落とさない
#[test]
fn views_do_not_cast_shadows() {
    let Some(mut r) = renderer() else {
        eprintln!("GPU アダプタが無いのでスキップ");
        return;
    };
    r.set_quality(quality_paths().remove(0).1).expect("quality");
    // 白い床（照明あり）と、その上に浮いた青い箱（照明なし）。光は −X 側の斜め上から＝影は箱の +X 側にずれる
    let floor = InstanceData::from_transform_with_material([0.0, 0.0, -50.0], 0.0, [10.0, 10.0, 0.05], [1.0; 4], 0.0, 1.0);
    let instances = [("cube".to_string(), floor), baked([0.0, 0.0, 150.0], [1.0; 3], BLUE)];
    // 方向光の向きは「光源の方へ」（pbr の n·l）
    let to_light = glam::Vec3::new(-1.0, 0.0, 1.0).normalize();
    r.update_lights([0.15; 3], &[Light::directional(to_light.to_array(), [1.0; 3], 3.0)], false);
    r.setup_shadow_map().expect("shadow");
    let eye = to_light * 800.0;
    let light_vp = glam::Mat4::orthographic_rh(-600.0, 600.0, -600.0, 600.0, 1.0, 2000.0)
        * glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Z);
    r.update_shadow_matrix(light_vp.to_cols_array_2d());

    let cam = top(0.0, 0.0, 800.0);
    // 影 = 床のうち暗い灰色（照明ありの床の 7 割未満の明るさ）の画素
    let dark_floor = |img: &[u8]| {
        let lum = |p: &[u8]| (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3;
        let grey = |p: &[u8]| p[0].abs_diff(p[1]) < 20 && p[1].abs_diff(p[2]) < 20;
        let lit = img.chunks(4).filter(|p| grey(p)).map(lum).max().unwrap_or(0);
        img.chunks(4).filter(|p| grey(p) && lum(p) > 0 && lum(p) * 10 < lit * 7).count()
    };
    let shadowed = dark_floor(&r.render_offscreen(&cam, &instances, instances.len(), N, N).unwrap());

    let mut view = r.create_view(N, N);
    let mut enc = r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    r.render_view(&mut enc, &mut view, &cam, &instances, instances.len(), &Default::default());
    r.queue().submit(std::iter::once(enc.finish()));
    let in_view = dark_floor(&r.read_view_rgba(&view).unwrap());
    eprintln!("影の画素: 既定 {shadowed} / 追加のビュー {in_view}");
    assert!(shadowed > 200, "前提: 既定のビューに影が出ていない（{shadowed}）");
    assert_eq!(in_view, 0, "追加のビューに影が出た");
}

/// 線チャンクの範囲・プレビュー線の有無はビューごとに選べる（平面と 3D で線の組を分ける用）
#[test]
fn views_pick_their_own_lines() {
    let Some(mut r) = renderer() else {
        eprintln!("GPU アダプタが無いのでスキップ");
        return;
    };
    let seg = |y: f32, color| [LineVertex::with_width([-100.0, y, 0.0], color, 3.0), LineVertex::with_width([100.0, y, 0.0], color, 3.0)];
    r.update_line_chunk(10, &seg(-50.0, RED)); // 「平面の組」
    r.update_line_chunk(20, &seg(0.0, GREEN)); // 「3D の組」
    r.update_preview_lines(&seg(50.0, BLUE));
    let cam = top(0.0, 0.0, 300.0);
    for (path, q) in quality_paths() {
        r.set_quality(q).expect("quality");
        let mut view = r.create_view(N, N);
        let mut shot = |opts: RenderViewOptions| {
            let mut enc = r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            r.render_view(&mut enc, &mut view, &cam, &[], 0, &opts);
            r.queue().submit(std::iter::once(enc.finish()));
            count(&r.read_view_rgba(&view).unwrap())
        };
        let all = shot(RenderViewOptions::default());
        let plan = shot(RenderViewOptions { line_chunks: Some(0..20), preview_lines: false, ..Default::default() });
        let model = shot(RenderViewOptions { line_chunks: Some(20..30), ..Default::default() });
        eprintln!("{path}: 全部 {all:?} / 平面の組 {plan:?} / 3D の組 {model:?}");
        assert!(all.0 > 0 && all.1 > 0 && all.2 > 0, "{path}: 既定では全部の線が出る");
        assert!(plan.0 > 0 && plan.1 == 0 && plan.2 == 0, "{path}: 平面の組は赤だけ（プレビューなし）");
        assert!(model.0 == 0 && model.1 > 0 && model.2 > 0, "{path}: 3D の組は緑とプレビュー");
    }
}

/// 寸法を変えても、品質を切り替えても（形式・サンプル数が変わる）描ける
#[test]
fn views_follow_resize_and_quality_changes() {
    let Some(mut r) = renderer() else {
        eprintln!("GPU アダプタが無いのでスキップ");
        return;
    };
    let instances = [baked([0.0; 3], [1.0; 3], RED)];
    let mut view = r.create_view(10, 10);
    for (path, q) in quality_paths() {
        r.set_quality(q).expect("quality");
        for (w, h) in [(N, N), (N / 2, N), (N, N)] {
            r.resize_view(&mut view, w, h);
            let mut cam = top(0.0, 0.0, 300.0);
            cam.aspect = w as f64 / h as f64;
            let mut enc = r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            r.render_view(&mut enc, &mut view, &cam, &instances, 1, &Default::default());
            r.queue().submit(std::iter::once(enc.finish()));
            let img = r.read_view_rgba(&view).unwrap();
            assert_eq!(img.len(), (w * h * 4) as usize, "{path} {w}x{h}: 寸法");
            assert_eq!(view.size(), (w, h));
            let (red, _, _) = count(&img);
            let expect = (w as f64 / 3.0) * (w as f64 / 3.0);
            assert!((red as f64 - expect).abs() < expect * 0.1, "{path} {w}x{h}: 赤 {red} ≠ 約{expect:.0}");
        }
    }
}

/// 既定のビューが描く線チャンクを範囲で絞れる（追加のビューだけが使う線の組を既定のビューに出さない）
#[test]
fn default_view_draws_only_its_line_chunk_range() {
    let Some(mut r) = renderer() else {
        eprintln!("GPU アダプタが無いのでスキップ");
        return;
    };
    let seg = |y: f32, color| [LineVertex::with_width([-100.0, y, 0.0], color, 3.0), LineVertex::with_width([100.0, y, 0.0], color, 3.0)];
    r.update_line_chunk(1, &seg(-50.0, RED)); // 既定のビューの組
    r.update_line_chunk(1_000_001, &seg(50.0, GREEN)); // 追加のビューだけの組
    let cam = top(0.0, 0.0, 300.0);
    for (path, q) in quality_paths() {
        r.set_quality(q).expect("quality");
        r.set_default_line_chunks(None);
        let all = count(&r.render_offscreen(&cam, &[], 0, N, N).unwrap());
        r.set_default_line_chunks(Some(0..1_000_000));
        let main = count(&r.render_offscreen(&cam, &[], 0, N, N).unwrap());
        // 追加のビューは既定のビューの絞り込みと関係なく自分の範囲を描く
        let mut view = r.create_view(N, N);
        let mut enc = r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let opts = RenderViewOptions { line_chunks: Some(1_000_000..2_000_000), ..Default::default() };
        r.render_view(&mut enc, &mut view, &cam, &[], 0, &opts);
        r.queue().submit(std::iter::once(enc.finish()));
        let pane = count(&r.read_view_rgba(&view).unwrap());
        eprintln!("{path}: 全部 {all:?} / 既定を絞る {main:?} / 追加のビュー {pane:?}");
        assert!(all.0 > 0 && all.1 > 0, "{path}: 絞らなければ両方の組が出る（従来どおり）");
        assert!(main.0 > 0 && main.1 == 0, "{path}: 既定のビューは自分の範囲だけ");
        assert!(pane.0 == 0 && pane.1 > 0, "{path}: 追加のビューは自分の範囲だけ");
    }
    r.set_default_line_chunks(None);
}

/// ビューだけの線: そのビューにだけ出る（ほかのビュー・既定のビューには出ない）。面に隠れず、
/// プレビュー線とは重なった所でプレビュー線が勝つ。空で消える
#[test]
fn view_lines_belong_to_one_view_and_sit_under_the_preview() {
    let Some(mut r) = renderer() else {
        eprintln!("GPU アダプタが無いのでスキップ");
        return;
    };
    let seg = |y: f32, z: f32, color| [LineVertex::with_width([-100.0, y, z], color, 3.0), LineVertex::with_width([100.0, y, z], color, 3.0)];
    let cam = top(0.0, 0.0, 300.0);
    // 赤い箱（上面 z=50）の下（z=-200）を通る緑の線＝面に隠れないか。y=40 はプレビュー線（青）と同じ所
    let instances = [baked([0.0; 3], [1.0; 3], RED)];
    let mut green = seg(-20.0, -200.0, GREEN).to_vec();
    green.extend(seg(40.0, 0.0, GREEN));
    r.update_preview_lines(&seg(40.0, 0.0, BLUE));
    for (path, q) in quality_paths() {
        r.set_quality(q).expect("quality");
        r.set_lines_depth_tested(true);
        let mut a = r.create_view(N, N);
        let mut b = r.create_view(N, N);
        r.update_view_lines(&mut a, &green);
        let mut enc = r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let opts = RenderViewOptions { lines_depth_tested: true, ..Default::default() };
        r.render_view(&mut enc, &mut a, &cam, &instances, 1, &opts);
        r.render_view(&mut enc, &mut b, &cam, &instances, 1, &opts);
        r.queue().submit(std::iter::once(enc.finish()));
        let img_a = r.read_view_rgba(&a).unwrap();
        let img_b = r.read_view_rgba(&b).unwrap();
        let default = count(&r.render_offscreen(&cam, &instances, 1, N, N).unwrap());
        // 画面の y（上が +Y・幅 300 を 160px）
        let row = |img: &[u8], wy: f32| -> (usize, usize, usize) {
            let py = ((150.0 - wy) / 300.0 * N as f32) as usize;
            count(&img[py * N as usize * 4..(py + 1) * N as usize * 4])
        };
        let (a_low, a_prev) = (row(&img_a, -20.0), row(&img_a, 40.0));
        eprintln!("{path}: A 全体 {:?} / 箱の下の線の行 {a_low:?} / プレビューと重なる行 {a_prev:?} / B {:?} / 既定 {default:?}", count(&img_a), count(&img_b));
        // 線は幅 200（画面の 2/3 = 約 106px）。箱（幅 100 = 約 53px）の上を横切っても欠けない
        let len = N as usize * 2 / 3;
        assert!(a_low.1 >= len - 2, "{path}: 箱の下を通るビューの線が、箱を横切って全部出る（面に隠れない）: {a_low:?}");
        assert!(a_prev.2 >= len - 2 && a_prev.1 == 0, "{path}: 重なった所はプレビュー線が上: {a_prev:?}");
        assert_eq!(count(&img_b).1, 0, "{path}: ほかのビューには出ない");
        assert_eq!(default.1, 0, "{path}: 既定のビューには出ない");
        r.update_view_lines(&mut a, &[]);
        let mut enc = r.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        r.render_view(&mut enc, &mut a, &cam, &instances, 1, &opts);
        r.queue().submit(std::iter::once(enc.finish()));
        assert_eq!(count(&r.read_view_rgba(&a).unwrap()).1, 0, "{path}: 空で消える");
    }
    r.set_lines_depth_tested(false);
    r.update_preview_lines(&[]);
}
