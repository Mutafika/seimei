//! レンダーパイプライン

use crate::{GpuVertex, InstanceData, LineVertex};
use thiserror::Error;

/// パイプラインエラー
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("シェーダーコンパイルエラー: {0}")]
    ShaderCompilation(String),

    #[error("パイプライン作成エラー: {0}")]
    PipelineCreation(String),
}

/// メインPBRシェーダーソース
pub const SHADER_SOURCE: &str = include_str!("../shaders/pbr.wgsl");

/// スクリーンスペース屈折シェーダソース
pub const SHADER_REFRACTION_SOURCE: &str = include_str!("../shaders/refraction.wgsl");

/// シャドウマップの解像度
pub const SHADOW_MAP_SIZE: u32 = 2048;

// ── パイプライン生成関数 ──

/// メインレンダーパイプライン
pub fn create_main_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    paint_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_main_pipeline_impl(device, format, camera_bind_group_layout, light_bind_group_layout, texture_bind_group_layout, paint_bind_group_layout,true, "Main Pipeline", 1)
}

/// 半透明用パイプライン（深度書き込みOFF）
pub fn create_transparent_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    paint_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_main_pipeline_impl(device, format, camera_bind_group_layout, light_bind_group_layout, texture_bind_group_layout, paint_bind_group_layout,false, "Transparent Pipeline", 1)
}

/// 深度プリパス用パイプライン（深度のみ書き込み・色出力なし）。
/// 透過(see-through)メッシュを半透明色パスの前にこれで深度だけ先書きし、後ろの髪等を遮蔽して
/// 半透明ソートのオーラを消す。vs_main は camera(group0) のみ使うので layout は camera だけ。
pub fn create_depth_prepass_pipeline(
    device: &wgpu::Device,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Depth Prepass"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Depth Prepass Layout"),
        bind_group_layouts: &[camera_bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Depth Prepass Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[GpuVertex::layout(), InstanceData::layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: None, // 色出力なし＝深度だけ書く
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back), // 前面のみで深度を書く
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });
    Ok(pipeline)
}

/// MSAA対応メインパイプライン
pub fn create_main_pipeline_msaa(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    paint_bind_group_layout: &wgpu::BindGroupLayout,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_main_pipeline_impl(device, format, camera_bind_group_layout, light_bind_group_layout, texture_bind_group_layout, paint_bind_group_layout,true, "Main Pipeline MSAA", msaa_samples)
}

/// MSAA対応半透明パイプライン
pub fn create_transparent_pipeline_msaa(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    paint_bind_group_layout: &wgpu::BindGroupLayout,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_main_pipeline_impl(device, format, camera_bind_group_layout, light_bind_group_layout, texture_bind_group_layout, paint_bind_group_layout,false, "Transparent Pipeline MSAA", msaa_samples)
}

/// 線分用パイプライン
pub fn create_line_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_line_or_point_pipeline(device, format, camera_bind_group_layout, wgpu::PrimitiveTopology::LineList, false, "Line Pipeline", 1)
}

/// ポイント描画用パイプライン
pub fn create_point_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_line_or_point_pipeline(device, format, camera_bind_group_layout, wgpu::PrimitiveTopology::PointList, true, "Point Pipeline", 1)
}

/// MSAA対応ライン用パイプライン
pub fn create_line_pipeline_msaa(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_line_or_point_pipeline(device, format, camera_bind_group_layout, wgpu::PrimitiveTopology::LineList, false, "Line Pipeline MSAA", msaa_samples)
}

/// MSAA対応ポイント用パイプライン
pub fn create_point_pipeline_msaa(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_line_or_point_pipeline(device, format, camera_bind_group_layout, wgpu::PrimitiveTopology::PointList, true, "Point Pipeline MSAA", msaa_samples)
}

/// 太線パイプライン（width > 1 の線分を画面上の四角形に展開して描く）
pub fn create_wide_line_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_wide_line_pipeline_impl(device, format, camera_bind_group_layout, false, "Wide Line Pipeline", 1)
}

/// 太線パイプライン（MSAA）
pub fn create_wide_line_pipeline_msaa(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_wide_line_pipeline_impl(device, format, camera_bind_group_layout, false, "Wide Line Pipeline MSAA", msaa_samples)
}

/// 深さ判定ありの線パイプライン（1px と太線の組）。手前の不透明メッシュに隠れる。
/// `msaa_samples` = 1 で MSAA なし。
pub fn create_depth_tested_line_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    msaa_samples: u32,
) -> Result<(wgpu::RenderPipeline, wgpu::RenderPipeline), PipelineError> {
    let thin = create_line_pipeline_impl(
        device, format, camera_bind_group_layout, wgpu::PrimitiveTopology::LineList,
        false, wgpu::CompareFunction::LessEqual, "vs_main_depth_tested",
        "Depth-Tested Line Pipeline", msaa_samples,
    )?;
    let wide = create_wide_line_pipeline_impl(
        device, format, camera_bind_group_layout, true,
        "Depth-Tested Wide Line Pipeline", msaa_samples,
    )?;
    Ok((thin, wide))
}

// シャドウ専用の別パイプライン（旧 create_main_pipeline_with_shadow*）は撤去した。
// 影を点けると pbr.wgsl の劣化複製(pbr_shadow.wgsl＝塗布/濡れ/SSS/clearcoat/リム無し)へ
// 切り替わる構造で、影と材質表現がトレードオフになっていたため。影は group 4 として
// メイン/半透明パイプラインへ常設した（create_main_pipeline_impl）。

/// Gaussian Splatting パイプライン
pub fn create_splat_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    splat_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Splat Shader"),
        source: wgpu::ShaderSource::Wgsl(crate::splat::SPLAT_SHADER_SOURCE.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Splat Pipeline Layout"),
        bind_group_layouts: &[camera_bind_group_layout, splat_bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Splat Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    Ok(pipeline)
}

/// スクリーンスペース屈折パイプライン（水専用）。
/// group0=camera, group1=light, group2=scene_color(tex+sampler) の3グループのみ。
/// 半透明と同じブレンド・深度設定（depth test有/write無）で HDR ターゲットに描く。
/// MSAA は屈折ON時(has_pp==true)には使われないので msaa_samples=1 固定でよいが、
/// 呼び出し側の都合で引数化しておく。
pub fn create_refraction_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    scene_color_bind_group_layout: &wgpu::BindGroupLayout,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Refraction Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_REFRACTION_SOURCE.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Refraction Pipeline Layout"),
        bind_group_layouts: &[
            camera_bind_group_layout,
            light_bind_group_layout,
            scene_color_bind_group_layout,
        ],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Refraction Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[GpuVertex::layout(), InstanceData::layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // 半透明と同じく両面
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false, // 半透明と同じく深度書き込みなし
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: -2,
                slope_scale: -1.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState {
            count: msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    Ok(pipeline)
}

// ── 内部実装 ──

/// 深さ判定ありの線を面より少し手前へ寄せる WGSL（1px と太線のシェーダで共有）。
/// 面と同一平面の線（床の記号・箱の上面の外周）が面に負けて欠けないようにする。
///
/// 線の画素は本来の線から最大で半 px（太線なら幅/2 + 半 px）ずれた位置にあり、斜めに
/// 見た面はその分だけ画素ごとに深さが違う。なので寄せる量は距離ではなく**画素の大きさ**で
/// 決める: その点での 1px のワールド長 ×（LINE_DEPTH_PULL_PX + 線幅）。画面上の位置は
/// 変えず、視線に沿って寄せる（透視は視点へ向かう光線上、正投影は視線方向）。
/// 1px の線なら 3px 分＝画面 1000px・画角 45° で 10m 先なら約 25mm。これより薄い壁の
/// 向こうの線は透ける。面を真横近く（約 15° 未満）から見ると同一平面の線は欠け始める。
/// 深度バイアス（DepthBiasState）は WebGPU が LineList に許さないので使わない。
macro_rules! line_depth_pull_wgsl {
    () => {
        r#"
const LINE_DEPTH_PULL_PX: f32 = 2.0;
fn pull_toward_eye(p: vec3<f32>, width_px: f32) -> vec3<f32> {
    let vp = camera.view_proj;
    // 透視なら w がワールド座標に依存する（正投影はクリップ w の行が (0, 0, 0, 1)）
    let perspective = abs(vp[0][3]) + abs(vp[1][3]) + abs(vp[2][3]) > 0.0;
    let eye = camera.position.xyz;
    let a = vp * vec4<f32>(p, 1.0);
    if (perspective && a.w <= 0.0) {
        return p; // 視点の後ろ。どうせ near 面で切られる
    }
    // その点での 1px のワールド長（視線に垂直な画面上方向で測る。レンズシフトは差で消える）
    let up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    let b = vp * vec4<f32>(p + up, 1.0);
    let px_per_unit = abs(b.y / b.w - a.y / a.w) * max(camera.resolution.y, 1.0) * 0.5;
    var pull = (LINE_DEPTH_PULL_PX + max(width_px, 1.0)) / max(px_per_unit, 1e-6);
    if (perspective) {
        let to_p = p - eye;
        let dist = length(to_p);
        // 視点を越えて反対側へ出ないように
        pull = min(pull, dist * 0.5);
        return p - to_p / dist * pull;
    }
    let fwd = -vec3<f32>(camera.view[0][2], camera.view[1][2], camera.view[2][2]);
    return p - fwd * pull;
}
"#
    };
}

/// ラインシェーダー（1px の線と点）。`vs_main_depth_tested` は深さ判定ありの線用
pub const LINE_SHADER_SOURCE: &str = concat!(r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec4<f32>,
    clip_min: vec4<f32>,
    clip_max: vec4<f32>,
    resolution: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn project(position: vec3<f32>, color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = camera.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> VertexOutput {
    return project(position, color);
}

@vertex
fn vs_main_depth_tested(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> VertexOutput {
    return project(pull_toward_eye(position, 1.0), color);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#, line_depth_pull_wgsl!());

/// 太線シェーダー。1 インスタンス = 1 線分（始点 a・終点 b）、頂点 4 つの三角形ストリップ
/// （0: a 右, 1: a 左, 2: b 右, 3: b 左）。両端をクリップ空間へ投影し、画面 px で線分の
/// 法線方向に width/2 ずつ押し出す。線端はバット（角）、継ぎ目は処理しない。
pub const WIDE_LINE_SHADER_SOURCE: &str = concat!(r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec4<f32>,
    clip_min: vec4<f32>,
    clip_max: vec4<f32>,
    resolution: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) p0: vec3<f32>, @location(1) c0: vec4<f32>, @location(2) w0: f32,
    @location(3) p1: vec3<f32>, @location(4) c1: vec4<f32>, @location(5) w1: f32,
) -> VertexOutput {
    return expand(vi, p0, c0, w0, p1, c1, w1);
}

@vertex
fn vs_main_depth_tested(
    @builtin(vertex_index) vi: u32,
    @location(0) p0: vec3<f32>, @location(1) c0: vec4<f32>, @location(2) w0: f32,
    @location(3) p1: vec3<f32>, @location(4) c1: vec4<f32>, @location(5) w1: f32,
) -> VertexOutput {
    return expand(vi, pull_toward_eye(p0, w0), c0, w0, pull_toward_eye(p1, w1), c1, w1);
}

fn expand(
    vi: u32,
    p0: vec3<f32>, c0: vec4<f32>, w0: f32,
    p1: vec3<f32>, c1: vec4<f32>, w1: f32,
) -> VertexOutput {
    var out: VertexOutput;
    var a = camera.view_proj * vec4<f32>(p0, 1.0);
    var b = camera.view_proj * vec4<f32>(p1, 1.0);
    var ca = c0;
    var cb = c1;

    // near 面（wgpu のクリップ空間 z >= 0）で切る。w <= 0 の端を割ると画面の反対側へ飛ぶため、
    // 押し出しの前にクリップ空間で済ませる。両端とも手前なら潰して描かない。
    if (a.z < 0.0 && b.z < 0.0) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }
    if (a.z < 0.0) {
        let t = a.z / (a.z - b.z);
        a = mix(a, b, t);
        ca = mix(ca, cb, t);
    } else if (b.z < 0.0) {
        let t = b.z / (b.z - a.z);
        b = mix(b, a, t);
        cb = mix(cb, ca, t);
    }

    // 解像度が未設定（0）でも割り算で壊れないよう 1px 以上とみなす
    let res = max(camera.resolution.xy, vec2<f32>(1.0));
    // 画面 px での向き（NDC の差に解像度を掛けると縦横比が戻る）
    var dir = (b.xy / b.w - a.xy / a.w) * res;
    let len = length(dir);
    if (len > 1e-6) {
        dir = dir / len;
    } else {
        dir = vec2<f32>(1.0, 0.0);
    }
    let normal = vec2<f32>(-dir.y, dir.x);

    let at_b = (vi >> 1u) == 1u;
    let side = select(-1.0, 1.0, (vi & 1u) == 1u);
    let base = select(a, b, at_b);
    let width = max(select(w0, w1, at_b), 1.0);
    // 半幅 (width/2) px → NDC は ×2/res。クリップ空間に戻すため w を掛ける。
    let offset = normal * side * width / res;
    out.position = vec4<f32>(base.xy + offset * base.w, base.z, base.w);
    out.color = select(ca, cb, at_b);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#, line_depth_pull_wgsl!());

fn create_wide_line_pipeline_impl(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    depth_tested: bool,
    label: &str,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(WIDE_LINE_SHADER_SOURCE.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[camera_bind_group_layout],
        push_constant_ranges: &[],
    });

    // 深度は 1px の線と揃える: 書かない。判定なしは常に通す・ありは LessEqual
    let (entry_point, depth_compare) = if depth_tested {
        ("vs_main_depth_tested", wgpu::CompareFunction::LessEqual)
    } else {
        ("vs_main", wgpu::CompareFunction::Always)
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(entry_point),
            buffers: &[LineVertex::wide_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    Ok(pipeline)
}

fn create_main_pipeline_impl(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    paint_bind_group_layout: &wgpu::BindGroupLayout,
    depth_write: bool,
    label: &str,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
    });

    // group 3 = 体表塗布マップ（テクスチャと同じ layout を流用＝texture+sampler）。
    // 影は group を増やさず **group 1(light) に同居**させている（max_bind_groups=4 の制限）。
    // ＝影ONで別シェーダへ切り替える旧方式をやめた（旧方式は pbr.wgsl の劣化複製を使っており、
    // 影を点けると塗布/濡れ/SSS/clearcoat がごっそり落ちていた）。
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[
            camera_bind_group_layout,
            light_bind_group_layout, // group 1 = ライト uniform ＋ 影(深度tex/比較sampler/ライトVP)
            texture_bind_group_layout,
            paint_bind_group_layout, // group 3 = 体表塗布(色+被覆 / 塗布時法線 の2tex束)
        ],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[GpuVertex::layout(), InstanceData::layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: if depth_write { Some(wgpu::Face::Back) } else { None },
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: depth_write,
            // LessEqual: 同深度で後勝ち。VRMのBody→服のように同位置で重なる
            // メッシュを描画順で上書きできるようにする。
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: if depth_write {
                wgpu::DepthBiasState::default()
            } else {
                // 半透明パス: depth biasで手前に引き出してZファイティング防止
                wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: -1.0,
                    clamp: 0.0,
                }
            },
        }),
        multisample: wgpu::MultisampleState {
            count: msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    Ok(pipeline)
}

fn create_line_or_point_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    topology: wgpu::PrimitiveTopology,
    depth_write: bool,
    label: &str,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    let depth_compare = if depth_write {
        wgpu::CompareFunction::Less
    } else {
        wgpu::CompareFunction::Always
    };
    create_line_pipeline_impl(
        device, format, camera_bind_group_layout, topology, depth_write, depth_compare,
        "vs_main", label, msaa_samples,
    )
}

#[allow(clippy::too_many_arguments)]
fn create_line_pipeline_impl(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    topology: wgpu::PrimitiveTopology,
    depth_write: bool,
    depth_compare: wgpu::CompareFunction,
    entry_point: &str,
    label: &str,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(LINE_SHADER_SOURCE.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[camera_bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(entry_point),
            buffers: &[LineVertex::layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: depth_write,
            depth_compare,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    Ok(pipeline)
}
