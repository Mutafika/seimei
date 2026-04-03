//! 被写界深度 (Depth of Field)
//!
//! CoC (Circle of Confusion) ベースのボケ効果。
//! 焦点距離とF値からピンボケを計算する。

/// DOFパラメータ
#[derive(Debug, Clone, Copy)]
pub struct DofParams {
    /// 焦点距離 (mm, ワールド空間)
    pub focus_distance: f32,
    /// F値 (f-stop)
    pub f_stop: f32,
    /// センサーサイズ (mm, デフォルト36mm = フルフレーム)
    pub sensor_size: f32,
    /// 最大CoC半径 (pixels)
    pub max_coc: f32,
}

impl Default for DofParams {
    fn default() -> Self {
        Self {
            focus_distance: 5000.0, // 5m
            f_stop: 2.8,
            sensor_size: 36.0,
            max_coc: 10.0,
        }
    }
}

/// DOFパス
#[allow(dead_code)]
pub struct DofPass {
    coc_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    coc_texture: wgpu::Texture,
    coc_view: wgpu::TextureView,
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
}

/// CoC計算シェーダー
const COC_SHADER: &str = r#"
struct DofUniforms {
    focus_distance: f32,
    f_stop: f32,
    sensor_size: f32,
    max_coc: f32,
    near: f32,
    far: f32,
    focal_length: f32,
    _padding: f32,
};

@group(0) @binding(0)
var<uniform> dof: DofUniforms;
@group(0) @binding(1)
var depth_tex: texture_depth_2d;
@group(0) @binding(2)
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

fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, tex_sampler, in.uv);
    let linear_depth = linearize_depth(depth, dof.near, dof.far);

    // CoC = |S1 - S2| * f^2 / (S2 * (S1 - f) * N)
    // ここで S1 = focus_distance, S2 = pixel_depth, f = focal_length, N = f_stop
    let s1 = dof.focus_distance;
    let s2 = max(linear_depth, 0.001);
    let f = dof.focal_length;
    let coc = abs(s1 - s2) * f * f / (s2 * (s1 - f) * dof.f_stop);
    let coc_clamped = clamp(coc, 0.0, dof.max_coc);

    // 符号付きCoC: 前ボケ(負)、後ボケ(正)
    let signed_coc = select(coc_clamped, -coc_clamped, linear_depth < s1);

    return vec4<f32>(signed_coc, 0.0, 0.0, 1.0);
}
"#;

/// ボケブラーシェーダー
const BOKEH_BLUR_SHADER: &str = r#"
@group(0) @binding(0)
var color_tex: texture_2d<f32>;
@group(0) @binding(1)
var coc_tex: texture_2d<f32>;
@group(0) @binding(2)
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
    let center_color = textureSample(color_tex, tex_sampler, in.uv);
    let center_coc = abs(textureSample(coc_tex, tex_sampler, in.uv).r);

    if (center_coc < 0.5) {
        return center_color;
    }

    // ディスクサンプリング（簡易Bokeh）
    let tex_size = vec2<f32>(textureDimensions(color_tex));
    let pixel_size = 1.0 / tex_size;

    var color_sum = center_color.rgb;
    var weight_sum = 1.0;

    // 8方向 × 最大coc半径のサンプリング
    let samples = 8;
    let golden_angle = 2.399963;

    for (var i = 1; i <= samples; i++) {
        let fi = f32(i);
        let radius = center_coc * fi / f32(samples);
        let angle = fi * golden_angle;
        let offset = vec2<f32>(cos(angle), sin(angle)) * radius * pixel_size;

        let sample_color = textureSample(color_tex, tex_sampler, in.uv + offset);
        let sample_coc = abs(textureSample(coc_tex, tex_sampler, in.uv + offset).r);

        let w = smoothstep(0.0, 1.0, sample_coc);
        color_sum += sample_color.rgb * w;
        weight_sum += w;
    }

    return vec4<f32>(color_sum / weight_sum, center_color.a);
}
"#;

impl DofPass {
    /// 新しいDOFパスを作成
    pub fn new(device: &wgpu::Device, width: u32, height: u32, color_format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DOF Bind Group Layout"),
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
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // CoC計算パイプライン
        let coc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CoC Shader"),
            source: wgpu::ShaderSource::Wgsl(COC_SHADER.into()),
        });
        let coc_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CoC Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let coc_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CoC Pipeline"),
            layout: Some(&coc_layout),
            vertex: wgpu::VertexState {
                module: &coc_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &coc_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
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

        // ボケブラーパイプライン
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bokeh Blur BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bokeh Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(BOKEH_BLUR_SHADER.into()),
        });
        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bokeh Blur Pipeline Layout"),
            bind_group_layouts: &[&blur_bgl],
            push_constant_ranges: &[],
        });
        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bokeh Blur Pipeline"),
            layout: Some(&blur_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
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

        let (coc_texture, coc_view) = Self::create_coc_texture(device, width, height);
        let (output_texture, output_view) = Self::create_output_texture(device, width, height, color_format);

        Self {
            coc_pipeline,
            blur_pipeline,
            bind_group_layout,
            coc_texture,
            coc_view,
            output_texture,
            output_view,
        }
    }

    fn create_coc_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CoC Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_output_texture(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DOF Output"),
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
        let (ct, cv) = Self::create_coc_texture(device, width, height);
        self.coc_texture = ct;
        self.coc_view = cv;
        let (ot, ov) = Self::create_output_texture(device, width, height, format);
        self.output_texture = ot;
        self.output_view = ov;
    }

    /// 出力テクスチャビューを取得
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }
}
