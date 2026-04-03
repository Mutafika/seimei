//! シャドウマップ

use crate::{GpuVertex, InstanceData};
use crate::pipeline::PipelineError;

/// シャドウマップの解像度
pub const SHADOW_MAP_SIZE: u32 = 2048;

/// ポイントライトシャドウアトラスのサイズ
pub const POINT_SHADOW_ATLAS_SIZE: u32 = 4096;
/// 各ライトのタイルサイズ
pub const POINT_SHADOW_TILE_SIZE: u32 = 512;
/// 最大ポイントライトシャドウキャスター数
pub const MAX_POINT_SHADOW_CASTERS: usize = 4;

/// シャドウ深度パス用シェーダー
const SHADOW_SHADER_SOURCE: &str = r#"
struct LightVP {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> light: LightVP;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(9) tangent: vec4<f32>,
    @location(10) vertex_color: vec4<f32>,
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) color: vec4<f32>,
    @location(8) material: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let world_pos = model * vec4<f32>(in.position, 1.0);
    return light.view_proj * world_pos;
}
"#;

/// シャドウ深度パス用パイプライン
pub fn create_shadow_pipeline(
    device: &wgpu::Device,
    light_bind_group_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::RenderPipeline, PipelineError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shadow Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER_SOURCE.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Shadow Pipeline Layout"),
        bind_group_layouts: &[light_bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Shadow Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[GpuVertex::layout(), InstanceData::layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: None,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Front),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    Ok(pipeline)
}
