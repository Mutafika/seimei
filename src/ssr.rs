//! スクリーンスペースリフレクション (SSR)
//!
//! G-Bufferの深度+法線を使い、スクリーンスペースでレイマーチして反射を計算する。

/// SSRパス
#[allow(dead_code)]
pub struct SsrPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
}

/// SSRシェーダー
const SSR_SHADER: &str = r#"
struct Params {
    inv_proj: mat4x4<f32>,
    proj: mat4x4<f32>,
    screen_size: vec2<f32>,
    max_steps: f32,
    step_size: f32,
};

@group(0) @binding(0)
var<uniform> params: Params;
@group(0) @binding(1)
var color_tex: texture_2d<f32>;
@group(0) @binding(2)
var normal_tex: texture_2d<f32>;
@group(0) @binding(3)
var depth_tex: texture_depth_2d;
@group(0) @binding(4)
var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(color_tex, tex_sampler, in.uv);
    let normal_roughness = textureSample(normal_tex, tex_sampler, in.uv);
    let roughness = normal_roughness.w;

    // roughness が高い面はSSRをスキップ
    if (roughness > 0.5) {
        return color;
    }

    let depth = textureSample(depth_tex, tex_sampler, in.uv);

    // 深度からビュー空間位置を再構築
    let ndc = vec3<f32>(in.uv * 2.0 - 1.0, depth);
    let view_pos = params.inv_proj * vec4<f32>(ndc.x, -ndc.y, ndc.z, 1.0);
    let pos = view_pos.xyz / view_pos.w;

    // 法線をビュー空間に変換済みと仮定
    let normal = normalize(normal_roughness.xyz * 2.0 - 1.0);

    // ビュー方向
    let view_dir = normalize(pos);
    let reflect_dir = reflect(view_dir, normal);

    // レイマーチ
    var ray_pos = pos;
    let max_steps = i32(params.max_steps);
    let step = params.step_size;

    for (var i = 0; i < max_steps; i++) {
        ray_pos = ray_pos + reflect_dir * step;

        // スクリーン座標に投影
        let proj_pos = params.proj * vec4<f32>(ray_pos, 1.0);
        let screen_uv = (proj_pos.xy / proj_pos.w) * vec2<f32>(0.5, -0.5) + 0.5;

        if (screen_uv.x < 0.0 || screen_uv.x > 1.0 || screen_uv.y < 0.0 || screen_uv.y > 1.0) {
            break;
        }

        let sample_depth = textureSample(depth_tex, tex_sampler, screen_uv);
        let ray_depth = proj_pos.z / proj_pos.w;

        if (ray_depth > sample_depth && ray_depth - sample_depth < 0.01) {
            let reflected_color = textureSample(color_tex, tex_sampler, screen_uv);
            let fresnel = pow(1.0 - max(dot(-view_dir, normal), 0.0), 5.0);
            let blend = fresnel * (1.0 - roughness);
            return mix(color, reflected_color, blend * 0.5);
        }
    }

    return color;
}
"#;

impl SsrPass {
    /// 新しいSSRパスを作成
    pub fn new(device: &wgpu::Device, width: u32, height: u32, output_format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSR Shader"),
            source: wgpu::ShaderSource::Wgsl(SSR_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSR Pipeline"),
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
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let (output_texture, output_view) = Self::create_output_texture(device, width, height, output_format);

        Self {
            pipeline,
            bind_group_layout,
            output_texture,
            output_view,
        }
    }

    fn create_output_texture(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR Output"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// リサイズ
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) {
        let (t, v) = Self::create_output_texture(device, width, height, format);
        self.output_texture = t;
        self.output_view = v;
    }

    /// 出力テクスチャビューを取得
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// バインドグループレイアウトを取得
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
