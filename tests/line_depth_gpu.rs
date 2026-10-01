//! 深さ判定ありの静的線（#22）を実 GPU で描いて画素を数える。
//!
//! 一辺 100 の不透明な立方体（原点中心・照明なしの MODEL_BAKED で青一色）に対して
//! - 赤: 立方体の**中**の十字 → 判定ありなら完全に隠れる、なしなら見える
//! - 緑: 立方体の**上面と同一平面**の × → 判定ありでも欠けない（なしと同じ画素数）
//!
//! を、透視・正投影・真上、1px・太線、MSAA なし・4x の全組み合わせで縛る。
//! アダプタが取れない環境では何もせずに通す。

use seimei::math::{Point3, Vec3D};
use seimei::vertex::MODEL_BAKED;
use seimei::{Camera, InstanceData, LineVertex, MsaaSamples, QualitySettings, Renderer};
use std::sync::Arc;

const N: u32 = 200;
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

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

fn quality(msaa: MsaaSamples) -> QualitySettings {
    QualitySettings {
        msaa,
        ssao: false,
        bloom: false,
        ssr: false,
        dof: false,
        edge_bevel: false,
        ..QualitySettings::default()
    }
}

fn cameras() -> Vec<(&'static str, Camera)> {
    let look = |pos: [f64; 3], up: [f64; 3], ortho: bool| {
        let mut c = Camera::new();
        c.aspect = 1.0;
        c.position = Point3::new(pos[0], pos[1], pos[2]);
        c.target = Point3::new(0.0, 0.0, 0.0);
        c.up = Vec3D::new(up[0], up[1], up[2]);
        c.is_orthographic = ortho;
        c.ortho_width = 300.0;
        c
    };
    vec![
        ("透視・斜め上", look([250.0, -300.0, 280.0], [0.0, 0.0, 1.0], false)),
        ("正投影・斜め上", look([250.0, -300.0, 280.0], [0.0, 0.0, 1.0], true)),
        ("透視・真上", look([0.0, 0.0, 400.0], [0.0, 1.0, 0.0], false)),
    ]
}

fn seg(a: [f32; 3], b: [f32; 3], color: [f32; 4], width: f32) -> [LineVertex; 2] {
    [LineVertex::with_width(a, color, width), LineVertex::with_width(b, color, width)]
}

/// (赤の画素数, 緑の画素数)
fn count(img: &[u8]) -> (usize, usize) {
    let px = img.chunks(4);
    let red = px.clone().filter(|p| p[0] > 150 && p[1] < 100 && p[2] < 100).count();
    let green = px.filter(|p| p[1] > 150 && p[0] < 100 && p[2] < 100).count();
    (red, green)
}

#[test]
fn depth_tested_lines_hide_behind_faces_but_not_on_them() {
    let Some(mut r) = renderer() else {
        eprintln!("GPU アダプタが無いのでスキップ");
        return;
    };
    let mut cube = InstanceData::from_transform([0.0; 3], 0.0, [1.0; 3], [0.0, 0.0, 1.0, 1.0]);
    cube.model_id = MODEL_BAKED;
    let instances = [("cube".to_string(), cube)];

    // ポストプロセスあり（bloom）は描画が 2 パス・HDR の別の深度へ行く経路
    let bloom = QualitySettings { bloom: true, ..quality(MsaaSamples::Off) };
    for (path, q) in [("MSAAなし", quality(MsaaSamples::Off)), ("MSAA4x", quality(MsaaSamples::X4)), ("ポストプロセス", bloom)] {
        r.set_quality(q).expect("quality");
        for width in [1.0f32, 4.0] {
            let mut lines = Vec::new();
            // 中の十字（赤）は本体バッファ、上面の ×（緑）は線チャンク — 両方の経路を通す
            // 0.7 ずらすのは、真上から見たとき画素の境目に乗って MSAA で半被覆にならないように
            lines.extend(seg([-30.0, 0.7, 0.0], [30.0, 0.7, 0.0], RED, width));
            lines.extend(seg([0.7, -30.0, 0.0], [0.7, 30.0, 0.0], RED, width));
            r.update_lines(&lines);
            let mut on_top = seg([-30.0, -30.0, 50.0], [30.0, 30.0, 50.0], GREEN, width).to_vec();
            on_top.extend(seg([-30.0, 30.0, 50.0], [30.0, -30.0, 50.0], GREEN, width));
            r.update_line_chunk(1, &on_top);

            for (view, cam) in cameras() {
                let label = format!("{view} width={width} {path}");
                r.set_lines_depth_tested(false);
                let img0 = r.render_offscreen(&cam, &instances, 1, N, N).unwrap();
                let (red0, green0) = count(&img0);
                r.set_lines_depth_tested(true);
                let img1 = r.render_offscreen(&cam, &instances, 1, N, N).unwrap();
                let (red1, green1) = count(&img1);
                eprintln!("{label}: 判定なし 赤{red0} 緑{green0} / あり 赤{red1} 緑{green1}");

                assert!(red0 > 0 && green0 > 0, "{label}: 判定なしで線が見えていない（テストの前提が崩れた）");
                assert_eq!(red1, 0, "{label}: 立方体の中の線が透けた");
                assert!(green1 * 100 >= green0 * 97, "{label}: 上面の線が欠けた {green1}/{green0}");
            }
        }
    }
}

/// プレビュー線は判定ありにしても常に手前
#[test]
fn preview_lines_stay_on_top() {
    let Some(mut r) = renderer() else {
        eprintln!("GPU アダプタが無いのでスキップ");
        return;
    };
    r.set_quality(quality(MsaaSamples::Off)).expect("quality");
    let mut cube = InstanceData::from_transform([0.0; 3], 0.0, [1.0; 3], [0.0, 0.0, 1.0, 1.0]);
    cube.model_id = MODEL_BAKED;
    let instances = [("cube".to_string(), cube)];
    r.update_preview_lines(&seg([-30.0, 0.0, 0.0], [30.0, 0.0, 0.0], RED, 3.0));
    r.set_lines_depth_tested(true);
    let (_, cam) = cameras().remove(0);
    let (red, _) = count(&r.render_offscreen(&cam, &instances, 1, N, N).unwrap());
    assert!(red > 0, "プレビュー線が隠れた");
}
