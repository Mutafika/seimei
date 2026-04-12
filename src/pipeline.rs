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

/// シャドウ付きPBRシェーダーソース
pub const SHADER_WITH_SHADOW_SOURCE: &str = include_str!("../shaders/pbr_shadow.wgsl");

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
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_main_pipeline_impl(device, format, camera_bind_group_layout, light_bind_group_layout, texture_bind_group_layout, true, "Main Pipeline", 1)
}

/// 半透明用パイプライン（深度書き込みOFF）
pub fn create_transparent_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_main_pipeline_impl(device, format, camera_bind_group_layout, light_bind_group_layout, texture_bind_group_layout, false, "Transparent Pipeline", 1)
}

/// MSAA対応メインパイプライン
pub fn create_main_pipeline_msaa(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_main_pipeline_impl(device, format, camera_bind_group_layout, light_bind_group_layout, texture_bind_group_layout, true, "Main Pipeline MSAA", msaa_samples)
}

/// MSAA対応半透明パイプライン
pub fn create_transparent_pipeline_msaa(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_main_pipeline_impl(device, format, camera_bind_group_layout, light_bind_group_layout, texture_bind_group_layout, false, "Transparent Pipeline MSAA", msaa_samples)
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

/// シャドウマップ対応メインパイプライン（Group 3追加）
pub fn create_main_pipeline_with_shadow(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    shadow_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_shadow_pipeline_impl(device, format, camera_bind_group_layout, light_bind_group_layout, texture_bind_group_layout, shadow_bind_group_layout, "Main Pipeline with Shadow", 1)
}

/// MSAA対応シャドウ付きメインパイプライン
pub fn create_main_pipeline_with_shadow_msaa(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    shadow_bind_group_layout: &wgpu::BindGroupLayout,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    create_shadow_pipeline_impl(device, format, camera_bind_group_layout, light_bind_group_layout, texture_bind_group_layout, shadow_bind_group_layout, "Main Pipeline with Shadow MSAA", msaa_samples)
}

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

// ── 内部実装 ──

/// ラインシェーダー
const LINE_SHADER_SOURCE: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec4<f32>,
    clip_min: vec4<f32>,
    clip_max: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = camera.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

fn create_main_pipeline_impl(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    depth_write: bool,
    label: &str,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[camera_bind_group_layout, light_bind_group_layout, texture_bind_group_layout],
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
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: depth_write,
            depth_compare: wgpu::CompareFunction::Less,
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

fn create_shadow_pipeline_impl(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    light_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    shadow_bind_group_layout: &wgpu::BindGroupLayout,
    label: &str,
    msaa_samples: u32,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(SHADER_WITH_SHADOW_SOURCE.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[
            camera_bind_group_layout,
            light_bind_group_layout,
            texture_bind_group_layout,
            shadow_bind_group_layout,
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
            cull_mode: None,
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

fn create_line_or_point_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    topology: wgpu::PrimitiveTopology,
    depth_write: bool,
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

    let depth_compare = if depth_write {
        wgpu::CompareFunction::Less
    } else {
        wgpu::CompareFunction::Always
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
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
