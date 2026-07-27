//! シャドウマップ

use crate::{GpuVertex, InstanceData};
use crate::pipeline::PipelineError;

/// シャドウマップ（カスケードアトラス）の解像度。2x2 に区切って各カスケードが1タイルを使う。
pub const SHADOW_MAP_SIZE: u32 = 4096;

/// カスケード数。近距離ほど小さい箱を密に、遠距離は大きい箱を粗く覆う＝1枚では両立できない
/// 「足元の輪郭の鋭さ」と「建物1棟ぶんの射程」を同時に満たす。2x2 アトラス前提なので 4 固定。
pub const SHADOW_CASCADES: usize = 4;
/// アトラス1タイルの辺（＝カスケード1枚の実効解像度）。
pub const SHADOW_TILE_SIZE: u32 = SHADOW_MAP_SIZE / 2;

/// カスケードの GPU 表現（本描画のサンプル用・group1 binding3）。
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniform {
    /// 各カスケードのライトVP（world → ライトクリップ）。
    pub view_proj: [[[f32; 4]; 4]; SHADOW_CASCADES],
    /// 各カスケードの遠端（カメラからの距離）。手前から順に増える。
    pub splits: [f32; 4],
    /// 各カスケードの深度バイアス（箱が大きいほど1テクセルが覆う実距離が伸びるので大きくする）。
    pub biases: [f32; 4],
    /// x=有効カスケード数(0=影なし), y=アトラス1テクセル(1/SHADOW_MAP_SIZE), z/w=予備。
    pub params: [f32; 4],
}

impl ShadowUniform {
    /// 影なし（カスケード数0）。ライトVPは単位行列。
    pub fn disabled() -> Self {
        let ident = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self {
            view_proj: [ident; SHADOW_CASCADES],
            splits: [0.0; 4],
            biases: [0.0; 4],
            params: [0.0, 1.0 / SHADOW_MAP_SIZE as f32, 0.0, 0.0],
        }
    }
}

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
