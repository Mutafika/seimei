//! 太線（LineVertex::width > 1）を実 GPU で描いて画素を数える（#14）。
//!
//! 正投影・画面 200px・表示幅 200 なので 1 ワールド単位 = 1px。線幅は px 指定なので、
//! 横線なら縦方向に、縦線なら横方向に width 画素ぶん塗られるはず。
//! アダプタが取れない環境（GPU の無い CI 等）では何もせずに通す。

use seimei::{Camera, LineVertex, MsaaSamples, QualitySettings, Renderer};
use std::sync::Arc;

const N: u32 = 200;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()?;
    let mut r = Renderer::new(Arc::new(device), Arc::new(queue), wgpu::TextureFormat::Rgba8Unorm, N, N).ok()?;
    r.set_clear_color(0.0, 0.0, 0.0);
    Some(r)
}

fn camera() -> Camera {
    let mut cam = Camera::new();
    cam.is_orthographic = true;
    cam.ortho_width = N as f64;
    cam.aspect = 1.0;
    cam.position = seimei::math::Point3::new(0.0, 0.0, 1000.0);
    cam.target = seimei::math::Point3::new(0.0, 0.0, 0.0);
    cam.up = seimei::math::Vec3D::new(0.0, 1.0, 0.0);
    cam
}

fn seg(a: [f32; 3], b: [f32; 3], color: [f32; 4], width: f32) -> [LineVertex; 2] {
    [LineVertex::with_width(a, color, width), LineVertex::with_width(b, color, width)]
}

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// 画素 (x, y) のチャンネル c が半分以上か
fn lit(img: &[u8], x: u32, y: u32, c: usize) -> bool {
    img[((y * N + x) * 4) as usize + c] > 127
}

fn column_run(img: &[u8], x: u32, c: usize) -> u32 {
    (0..N).filter(|&y| lit(img, x, y, c)).count() as u32
}

fn row_run(img: &[u8], y: u32, c: usize) -> u32 {
    (0..N).filter(|&x| lit(img, x, y, c)).count() as u32
}

fn draw_and_measure(r: &mut Renderer) -> (u32, u32, u32, u32) {
    // 本体バッファ: 横線 width 9（y=+40）と 1px の横線（y=+70）
    let mut main = seg([-80.0, 40.0, 0.0], [80.0, 40.0, 0.0], RED, 9.0).to_vec();
    main.extend(seg([-80.0, 70.0, 0.0], [80.0, 70.0, 0.0], RED, 1.0));
    r.update_lines(&main);
    // 静的線チャンク: 縦線 width 5（x=-40）
    r.update_line_chunk(7, &seg([-40.0, -30.0, 0.0], [-40.0, 30.0, 0.0], GREEN, 5.0));
    // プレビュー線: 横線 width 3（y=-40）
    r.update_preview_lines(&seg([-80.0, -40.0, 0.0], [80.0, -40.0, 0.0], BLUE, 3.0));

    let img = r.render_offscreen(&camera(), &[], 0, N, N).expect("render");
    assert_eq!(img.len(), (N * N * 4) as usize);
    // 横線は列 x=140 で、縦線は行 y=100 で幅を数える。
    // 画面の行 = 100 - ワールド y なので、赤の太線と細線は別の行に出る。
    let red_thick = (40..80).filter(|&y| lit(&img, 140, y, 0)).count() as u32;
    let red_thin = (20..40).filter(|&y| lit(&img, 140, y, 0)).count() as u32;
    (red_thick, red_thin, row_run(&img, 100, 1), column_run(&img, 140, 2))
}

fn check(label: &str, (thick, thin, green, blue): (u32, u32, u32, u32)) {
    eprintln!("{label}: 9→{thick}px 1→{thin}px 5→{green}px 3→{blue}px");
    assert!((8..=10).contains(&thick), "{label}: width 9 の横線が {thick}px");
    assert!((1..=2).contains(&thin), "{label}: width 1 の横線が {thin}px");
    assert!((4..=6).contains(&green), "{label}: チャンクの width 5 の縦線が {green}px");
    assert!((2..=4).contains(&blue), "{label}: プレビューの width 3 の横線が {blue}px");
}

#[test]
fn wide_lines_have_their_pixel_width_on_every_path() {
    let Some(mut r) = renderer() else {
        eprintln!("GPU アダプタが無いのでスキップ");
        return;
    };
    let mut q = QualitySettings { msaa: MsaaSamples::Off, ..QualitySettings::default() };
    // ポストプロセスが入ると描画先が変わるので、線の経路だけを見るために切る
    q.ssao = false;
    q.bloom = false;
    q.ssr = false;
    q.dof = false;
    q.edge_bevel = false;
    r.set_quality(q.clone()).expect("quality off");
    check("MSAA なし", draw_and_measure(&mut r));

    q.msaa = MsaaSamples::X4;
    r.set_quality(q).expect("quality msaa");
    check("MSAA 4x", draw_and_measure(&mut r));
}

/// 透視で片端がカメラの後ろにある線分。near 面で切らないと後ろの端が画面の反対側へ
/// 投影され、四角形が画面を横切る。切れていれば、地面の線は画面の下半分に幅一定で出る。
#[test]
fn wide_line_crossing_behind_camera_is_clipped() {
    let Some(mut r) = renderer() else {
        eprintln!("GPU アダプタが無いのでスキップ");
        return;
    };
    let mut cam = Camera::new();
    cam.aspect = 1.0;
    cam.position = seimei::math::Point3::new(0.0, 0.0, 50.0);
    cam.target = seimei::math::Point3::new(0.0, 100.0, 50.0);
    cam.up = seimei::math::Vec3D::new(0.0, 0.0, 1.0);
    // 目の高さ 50 の下、後ろ (y=-100) から前 (y=3000) へ伸びる地面の線
    r.update_lines(&seg([0.0, -100.0, 0.0], [0.0, 3000.0, 0.0], RED, 5.0));
    let img = r.render_offscreen(&cam, &[], 0, N, N).expect("render");

    let upper: u32 = (0..N / 2 - 2).map(|y| row_run(&img, y, 0)).sum();
    assert_eq!(upper, 0, "地平線より上に線が出た（後ろの端が反転している）");
    // 遠い端 (y=3000) は地平線の約 4px 下で終わるので、その下から数える
    let rows: Vec<u32> = (N / 2 + 6..N).map(|y| row_run(&img, y, 0)).collect();
    assert!(rows.iter().all(|&n| (4..=6).contains(&n)), "下半分の各行で幅 5px のはず: {rows:?}");
}
