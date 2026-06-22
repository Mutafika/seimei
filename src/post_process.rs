//! ポストプロセスパイプライン（SSAO, Bloom, SSR, DOF）

use crate::quality::QualitySettings;

/// G-Buffer（法線+粗さ+深度をオフスクリーンRTに出力）
pub struct GBuffer {
    /// 法線+粗さ (Rgba16Float)
    pub normal_roughness_texture: wgpu::Texture,
    pub normal_roughness_view: wgpu::TextureView,
    /// 深度テクスチャ（メインパスと共有可能）
    pub depth_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl GBuffer {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, depth_view: wgpu::TextureView) -> Self {
        let normal_roughness_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GBuffer Normal+Roughness"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let normal_roughness_view =
            normal_roughness_texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            normal_roughness_texture,
            normal_roughness_view,
            depth_view,
            width,
            height,
        }
    }
}

/// フルスクリーン三角形用の頂点シェーダー（共通）
pub const FULLSCREEN_TRIANGLE_VS: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // フルスクリーン三角形（3頂点で画面全体をカバー）
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}
"#;

/// SSAOパス
#[allow(dead_code)]
pub struct SsaoPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    noise_texture: wgpu::Texture,
    noise_view: wgpu::TextureView,
    kernel_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// SSAOの中間テクスチャ（R8Unorm）
    pub ssao_texture: wgpu::Texture,
    pub ssao_view: wgpu::TextureView,
    /// ブラー用パイプライン
    blur_pipeline: wgpu::RenderPipeline,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    pub blurred_texture: wgpu::Texture,
    pub blurred_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl SsaoPass {
    /// SSAOパスを作成
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // SSAOカーネル（64サンプル、半球分布）
        let mut kernel = Vec::with_capacity(64 * 4);
        for i in 0..64u32 {
            let scale = (i as f32) / 64.0;
            let scale = 0.1 + scale * scale * 0.9;
            // 簡易ハルトン列で擬似ランダム
            let x = halton(i + 1, 2) * 2.0 - 1.0;
            let y = halton(i + 1, 3) * 2.0 - 1.0;
            let z = halton(i + 1, 5);
            let len = (x * x + y * y + z * z).sqrt().max(0.001);
            kernel.push(x / len * scale);
            kernel.push(y / len * scale);
            kernel.push(z / len * scale);
            kernel.push(0.0);
        }

        let kernel_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SSAO Kernel"),
                contents: bytemuck::cast_slice(&kernel),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // 4x4ノイズテクスチャ
        let mut noise_data = Vec::with_capacity(16 * 4);
        for i in 0..16u32 {
            let x = halton(i + 1, 2) * 2.0 - 1.0;
            let y = halton(i + 1, 3) * 2.0 - 1.0;
            let len = (x * x + y * y).sqrt().max(0.001);
            noise_data.push(((x / len * 0.5 + 0.5) * 255.0) as u8);
            noise_data.push(((y / len * 0.5 + 0.5) * 255.0) as u8);
            noise_data.push(0u8);
            noise_data.push(255u8);
        }
        let noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Noise"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &noise_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &noise_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
        let noise_view = noise_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // SSAOバインドグループレイアウト
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSAO Bind Group Layout"),
                entries: &[
                    // 法線+粗さテクスチャ
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
                    // 深度テクスチャ
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
                    // ノイズテクスチャ
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
                    // サンプラー
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // カーネルバッファ
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // SSAOシェーダー
        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSAO Shader"),
            source: wgpu::ShaderSource::Wgsl(SSAO_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSAO Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAO Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &ssao_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_shader,
                entry_point: Some("fs_ssao"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
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
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // SSAOテクスチャ
        let (ssao_texture, ssao_view) = create_r8_texture(device, width, height, "SSAO");

        // ブラーパス
        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSAO Blur Bind Group Layout"),
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSAO Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(BLUR_SHADER.into()),
        });

        let blur_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSAO Blur Pipeline Layout"),
                bind_group_layouts: &[&blur_bind_group_layout],
                push_constant_ranges: &[],
            });

        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAO Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: Some("fs_blur"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
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
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        let (blurred_texture, blurred_view) =
            create_r8_texture(device, width, height, "SSAO Blurred");

        Self {
            pipeline,
            bind_group_layout,
            noise_texture,
            noise_view,
            kernel_buffer,
            ssao_texture,
            ssao_view,
            blur_pipeline,
            blur_bind_group_layout,
            blurred_texture,
            blurred_view,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("SSAO Sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            width,
            height,
        }
    }

    /// SSAOテクスチャをリサイズ
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (t, v) = create_r8_texture(device, width, height, "SSAO");
        self.ssao_texture = t;
        self.ssao_view = v;
        let (t, v) = create_r8_texture(device, width, height, "SSAO Blurred");
        self.blurred_texture = t;
        self.blurred_view = v;
        self.width = width;
        self.height = height;
    }

    /// SSAOパスを実行
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_bind_group: &wgpu::BindGroup,
        gbuffer: &GBuffer,
        device: &wgpu::Device,
    ) {
        let ssao_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSAO Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.normal_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.kernel_buffer.as_entire_binding(),
                },
            ],
        });

        // SSAOパス
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAO Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, camera_bind_group, &[]);
            pass.set_bind_group(1, &ssao_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // ブラーパス
        let blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSAO Blur Bind Group"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAO Blur Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.blurred_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &blur_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// エッジベベルパス（G-Buffer法線+深度からエッジ検出→ハイライト）
pub struct EdgeBevelPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pub bevel_texture: wgpu::Texture,
    pub bevel_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl EdgeBevelPass {
    /// エッジベベルパスを作成
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Edge Bevel Bind Group Layout"),
                entries: &[
                    // 法線+粗さテクスチャ
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
                    // 深度テクスチャ
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
                    // サンプラー
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Edge Bevel Shader"),
            source: wgpu::ShaderSource::Wgsl(EDGE_BEVEL_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Edge Bevel Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Edge Bevel Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_edge_bevel"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
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
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        let (bevel_texture, bevel_view) = create_r8_texture(device, width, height, "Edge Bevel");

        Self {
            pipeline,
            bind_group_layout,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("EdgeBevel Sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            bevel_texture,
            bevel_view,
            width,
            height,
        }
    }

    /// リサイズ
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (t, v) = create_r8_texture(device, width, height, "Edge Bevel");
        self.bevel_texture = t;
        self.bevel_view = v;
        self.width = width;
        self.height = height;
    }

    /// エッジベベルパスを実行
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        device: &wgpu::Device,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Edge Bevel Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.normal_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Edge Bevel Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.bevel_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Bloomパス
pub struct BloomPass {
    /// 輝度抽出パイプライン
    extract_pipeline: wgpu::RenderPipeline,
    extract_bind_group_layout: wgpu::BindGroupLayout,
    /// ダウンサンプル/ブラーパイプライン
    downsample_pipeline: wgpu::RenderPipeline,
    downsample_bind_group_layout: wgpu::BindGroupLayout,
    /// アップサンプル加算パイプライン
    upsample_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    /// ミップチェーンテクスチャ（1/2, 1/4, 1/8, 1/16）
    pub mip_textures: Vec<(wgpu::Texture, wgpu::TextureView)>,
    pub width: u32,
    pub height: u32,
}

impl BloomPass {
    /// Bloomパスを作成
    pub fn new(device: &wgpu::Device, width: u32, height: u32, _surface_format: wgpu::TextureFormat) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Bind Group Layout"),
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Shader"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bloom Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // 使用するフォーマット: Rgba16Float(HDR)
        let bloom_format = wgpu::TextureFormat::Rgba16Float;

        let extract_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Extract Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &bloom_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: Some("fs_extract"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: bloom_format,
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
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        let downsample_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Downsample Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &bloom_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: Some("fs_downsample"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: bloom_format,
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
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        let upsample_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Upsample Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &bloom_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: Some("fs_upsample"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: bloom_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // ミップチェーン生成（4段）
        let mip_textures = Self::create_mip_textures(device, width, height, bloom_format);

        Self {
            extract_pipeline,
            extract_bind_group_layout: bind_group_layout.clone(),
            downsample_pipeline,
            downsample_bind_group_layout: bind_group_layout,
            upsample_pipeline,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Bloom Sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            mip_textures,
            width,
            height,
        }
    }

    fn create_mip_textures(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Vec<(wgpu::Texture, wgpu::TextureView)> {
        let mut result = Vec::new();
        let mut w = width / 2;
        let mut h = height / 2;
        for i in 0..4 {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Mip {}", i)),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            result.push((tex, view));
            w /= 2;
            h /= 2;
        }
        result
    }

    /// リサイズ
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.mip_textures =
            Self::create_mip_textures(device, width, height, wgpu::TextureFormat::Rgba16Float);
        self.width = width;
        self.height = height;
    }

    /// Bloomパスを実行（シーンカラーテクスチャビューを入力）
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_color_view: &wgpu::TextureView,
        device: &wgpu::Device,
    ) {
        if self.mip_textures.is_empty() {
            return;
        }

        // 1. 輝度抽出 → mip[0]
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom Extract BG"),
                layout: &self.extract_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Extract"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mip_textures[0].1,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.extract_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 2. ダウンサンプル mip[0] → mip[1] → mip[2] → mip[3]
        for i in 1..self.mip_textures.len() {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Bloom Down {}", i)),
                layout: &self.downsample_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.mip_textures[i - 1].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("Bloom Downsample {}", i)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mip_textures[i].1,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 3. アップサンプル mip[3] → mip[2] → mip[1] → mip[0]（加算合成）
        for i in (0..self.mip_textures.len() - 1).rev() {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Bloom Up {}", i)),
                layout: &self.downsample_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.mip_textures[i + 1].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("Bloom Upsample {}", i)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mip_textures[i].1,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // 加算
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.upsample_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// ドット絵（ピクセルアート）ポストプロセスのパラメータ（uniform）
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PixelArtParams {
    /// ドットセルの大きさ(px)。大きいほど粗い
    pub cell_px: f32,
    /// 0=ポスタライズ, 1=gameboy, 2=pico8, 3=famicom風
    pub palette_id: f32,
    /// Bayerディザ (0/1)
    pub dither: f32,
    /// 輪郭線の濃さ (0..1)
    pub outline: f32,
    /// ポスタライズ階調数（palette_id=0 のとき有効）
    pub levels: f32,
    /// 彩度ブースト (1.0=変更なし)
    pub sat: f32,
    /// #2 減色方式: 0=RGB最近傍 / 1=Oklab最近傍＋2色オーダードディザ
    pub color_mode: f32,
    /// #3 輪郭方式: 0=輝度差 / 1=深度(シルエット)
    pub outline_mode: f32,
}

impl Default for PixelArtParams {
    fn default() -> Self {
        Self {
            cell_px: 8.0,
            palette_id: 2.0, // pico8
            dither: 1.0,
            outline: 0.5,
            levels: 32.0, // 色数=フル（下げるほど減色）。palette/posterize 両モードで効く
            sat: 1.3,
            color_mode: 0.0,
            outline_mode: 0.0,
        }
    }
}

/// ドット絵パス: 合成済みLDR画像をセル単位でドット化して output へ書き出す（FXAA代替）。
/// 通常は full-res 単パス。#1 低解像度2パス ON 時は lores へ低解像度描画→nearest 拡大。
pub struct PixelArtPass {
    pipeline: wgpu::RenderPipeline,    // full-res 単パス（fs_pixel_art）
    pipeline_lo: wgpu::RenderPipeline, // #1 Pass A（fs_dot_lo）lores へ
    pipeline_up: wgpu::RenderPipeline, // #1 Pass B（fs_dot_up）lores→output
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    surface_format: wgpu::TextureFormat,
    /// #1 低解像度2パスの中間RT（フルサイズ確保し viewport で部分使用）
    lores_texture: wgpu::Texture,
    lores_view: wgpu::TextureView,
    /// 深度なし（gbuffer 無効）時のダミー深度
    _dummy_depth_texture: wgpu::Texture,
    dummy_depth_view: wgpu::TextureView,
    width: u32,
    height: u32,
    /// CPU 側に持つ現在のセル幅（Pass A の viewport 計算用）。update_params で同期。
    cell_px: f32,
    /// #1 低解像度2パス ON/OFF
    lowres: bool,
}

impl PixelArtPass {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        params: PixelArtParams,
        width: u32,
        height: u32,
    ) -> Self {
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PixelArt Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PixelArt Bind Group Layout"),
                entries: &[
                    // 入力（合成済みLDR or lores）
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // パラメータ
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 深度（#3 アウトライン用。textureLoad なのでサンプラ不要）
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
                ],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PixelArt Shader"),
            source: wgpu::ShaderSource::Wgsl(PIXEL_ART_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PixelArt Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // 3エントリポイント（fs_pixel_art / fs_dot_lo / fs_dot_up）を同レイアウト・同フォーマットで。
        let make_pipeline = |entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("PixelArt Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
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
                multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
                multiview: None,
                cache: None,
            })
        };
        let pipeline = make_pipeline("fs_pixel_art");
        let pipeline_lo = make_pipeline("fs_dot_lo");
        let pipeline_up = make_pipeline("fs_dot_up");

        // ニアレストでもよいが、セル中心を Linear で拾い軽く均す
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PixelArt Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // #1 低解像度2パス用の中間RT（フルサイズ。viewport で部分使用）
        let (lores_texture, lores_view) =
            create_ldr_texture(device, width.max(1), height.max(1), surface_format, "PixelArt Lores");

        // gbuffer 無効時のダミー深度（1x1）
        let dummy_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PixelArt Dummy Depth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_depth_view = dummy_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            pipeline,
            pipeline_lo,
            pipeline_up,
            bind_group_layout,
            params_buffer,
            sampler,
            surface_format,
            lores_texture,
            lores_view,
            _dummy_depth_texture: dummy_depth_texture,
            dummy_depth_view,
            width: width.max(1),
            height: height.max(1),
            cell_px: params.cell_px,
            lowres: false,
        }
    }

    /// パラメータバッファを書き換え（毎フレーム呼んでもよい軽量更新）。CPU側 cell も同期。
    pub fn update_params(&mut self, queue: &wgpu::Queue, params: PixelArtParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        self.cell_px = params.cell_px;
    }

    /// #1 低解像度2パスの ON/OFF
    pub fn set_lowres(&mut self, on: bool) {
        self.lowres = on;
    }

    /// 中間RTをリサイズ（ウィンドウサイズ変更時）。
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (t, v) =
            create_ldr_texture(device, width.max(1), height.max(1), self.surface_format, "PixelArt Lores");
        self.lores_texture = t;
        self.lores_view = v;
        self.width = width.max(1);
        self.height = height.max(1);
    }

    fn make_bind_group(
        &self,
        device: &wgpu::Device,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PixelArt Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
            ],
        })
    }

    /// input_view（合成済みLDR）をドット化して output_view へ。depth は #3 用（無ければダミー）。
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
    ) {
        let depth = depth_view.unwrap_or(&self.dummy_depth_view);

        if self.lowres {
            // === Pass A: input → lores（低解像度ビューポート。1画素=1セル）===
            let cell = self.cell_px.max(1.0);
            let vw = ((self.width as f32) / cell).ceil().max(1.0);
            let vh = ((self.height as f32) / cell).ceil().max(1.0);
            {
                let bg = self.make_bind_group(device, input_view, depth);
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("PixelArt Pass A (lores)"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.lores_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.pipeline_lo);
                pass.set_viewport(0.0, 0.0, vw, vh, 0.0, 1.0);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            // === Pass B: lores → output（full-res, nearest 拡大＋輪郭）===
            {
                let bg = self.make_bind_group(device, &self.lores_view, depth);
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("PixelArt Pass B (upscale)"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.pipeline_up);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
        } else {
            // === full-res 単パス ===
            let bg = self.make_bind_group(device, input_view, depth);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PixelArt Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// ポストプロセスパイプライン全体
pub struct PostProcessPipeline {
    pub ssao: Option<SsaoPass>,
    pub bloom: Option<BloomPass>,
    pub edge_bevel: Option<EdgeBevelPass>,
    /// 最終合成パイプライン
    pub composite_pipeline: wgpu::RenderPipeline,
    pub composite_bind_group_layout: wgpu::BindGroupLayout,
    /// シーンカラーHDRテクスチャ（ポストプロセス入力）
    pub scene_color_texture: wgpu::Texture,
    pub scene_color_view: wgpu::TextureView,
    pub gbuffer: Option<GBuffer>,
    pub width: u32,
    pub height: u32,
    /// FXAA最終アンチエイリアス用
    pub surface_format: wgpu::TextureFormat,
    pub fxaa_texture: wgpu::Texture,
    pub fxaa_view: wgpu::TextureView,
    pub fxaa_pipeline: wgpu::RenderPipeline,
    pub fxaa_bind_group_layout: wgpu::BindGroupLayout,
    pub fxaa_sampler: wgpu::Sampler,
    /// ドット絵（ピクセルアート）パス。ON時は FXAA を置換して最終出力をドット化する
    pub pixel_art: PixelArtPass,
    pub pixel_art_enabled: bool,
}

impl PostProcessPipeline {
    /// ポストプロセスパイプラインを作成
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        settings: &QualitySettings,
    ) -> Self {
        // シーンカラーHDRテクスチャ
        let (scene_color_texture, scene_color_view) =
            create_hdr_texture(device, width, height, "Scene Color HDR");

        // G-Buffer（SSAO or SSR で必要）
        let gbuffer = if settings.needs_gbuffer() {
            let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("PostProcess Depth Copy"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
            Some(GBuffer::new(device, width, height, depth_view))
        } else {
            None
        };

        let ssao = if settings.ssao {
            Some(SsaoPass::new(device, queue, width, height, camera_bind_group_layout))
        } else {
            None
        };

        let bloom = if settings.bloom {
            Some(BloomPass::new(device, width, height, surface_format))
        } else {
            None
        };

        let edge_bevel = if settings.edge_bevel {
            Some(EdgeBevelPass::new(device, width, height))
        } else {
            None
        };

        // 最終合成パイプライン
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Bind Group Layout"),
                entries: &[
                    // シーンカラー
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
                    // Bloomテクスチャ（使わない場合もバインド必要）
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
                    // SSAOテクスチャ
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
                    // サンプラー
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // エッジベベルテクスチャ
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
        });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
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
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // === FXAA 最終アンチエイリアスパス ===
        // composite はこの中間LDRテクスチャへ描画し、FXAA がそれをサンプルして output_view へ
        let (fxaa_texture, fxaa_view) =
            create_ldr_texture(device, width, height, surface_format, "FXAA Intermediate");

        let fxaa_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FXAA Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let fxaa_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("FXAA Bind Group Layout"),
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let fxaa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FXAA Shader"),
            source: wgpu::ShaderSource::Wgsl(FXAA_SHADER.into()),
        });

        let fxaa_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("FXAA Pipeline Layout"),
                bind_group_layouts: &[&fxaa_bind_group_layout],
                push_constant_ranges: &[],
            });

        let fxaa_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FXAA Pipeline"),
            layout: Some(&fxaa_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &fxaa_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fxaa_shader,
                entry_point: Some("fs_fxaa"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
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
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        Self {
            ssao,
            bloom,
            edge_bevel,
            composite_pipeline,
            composite_bind_group_layout,
            scene_color_texture,
            scene_color_view,
            gbuffer,
            width,
            height,
            surface_format,
            fxaa_texture,
            fxaa_view,
            fxaa_pipeline,
            fxaa_bind_group_layout,
            fxaa_sampler,
            pixel_art: PixelArtPass::new(device, surface_format, PixelArtParams::default(), width, height),
            pixel_art_enabled: false,
        }
    }

    /// ドット絵パスの ON/OFF
    pub fn set_pixel_art_enabled(&mut self, on: bool) {
        self.pixel_art_enabled = on;
    }

    /// #1 低解像度2パスの ON/OFF
    pub fn set_pixel_art_lowres(&mut self, on: bool) {
        self.pixel_art.set_lowres(on);
    }

    /// ドット絵パラメータをライブ更新（パイプライン再構築なし）。CPU側 cell も同期するため &mut。
    pub fn set_pixel_art_params(&mut self, queue: &wgpu::Queue, params: PixelArtParams) {
        self.pixel_art.update_params(queue, params);
    }

    /// シーンカラーHDRテクスチャビューを取得（3Dシーンの描画先）
    pub fn scene_color_view(&self) -> &wgpu::TextureView {
        &self.scene_color_view
    }

    /// ポストプロセス全体を実行し、最終結果を output_view に合成
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_bind_group: &wgpu::BindGroup,
        device: &wgpu::Device,
        output_view: &wgpu::TextureView,
    ) {
        // SSAO
        if let (Some(ssao), Some(gbuffer)) = (&self.ssao, &self.gbuffer) {
            ssao.execute(encoder, camera_bind_group, gbuffer, device);
        }

        // Edge Bevel
        if let (Some(edge_bevel), Some(gbuffer)) = (&self.edge_bevel, &self.gbuffer) {
            edge_bevel.execute(encoder, gbuffer, device);
        }

        // Bloom
        if let Some(bloom) = &self.bloom {
            bloom.execute(encoder, &self.scene_color_view, device);
        }

        // 最終合成 → output_view
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Composite Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // フォールバック用 1x1 白テクスチャビュー（無効パスのダミー — 1.0で初期化）
        let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PP Dummy"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        // Clear dummy to white via a tiny render pass
        {
            let dummy_view = dummy_tex.create_view(&wgpu::TextureViewDescriptor::default());
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PP Dummy Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dummy_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        let dummy_view = dummy_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Bloom 無効時のフォールバックは「黒(0)」でなければならない（合成は color += bloom の加算。
        // 白ダミーを使うと R8 を .rgb で読んで (1,0,0)=赤を全画面に加算しピンクに washout する）。
        let black_dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PP Bloom Dummy (black)"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        {
            let bv = black_dummy_tex.create_view(&wgpu::TextureViewDescriptor::default());
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PP Bloom Dummy Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &bv,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        let black_dummy_view = black_dummy_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Bloom テクスチャ（mip[0] or 黒ダミー）
        let bloom_view = self.bloom.as_ref()
            .and_then(|b| b.mip_textures.first())
            .map(|(_, v)| v)
            .unwrap_or(&black_dummy_view);

        // SSAO テクスチャ（blurred or dummy）
        let ssao_view = self.ssao.as_ref()
            .map(|s| &s.blurred_view)
            .unwrap_or(&dummy_view);

        // EdgeBevel テクスチャ
        let bevel_view = self.edge_bevel.as_ref()
            .map(|e| &e.bevel_view)
            .unwrap_or(&dummy_view);

        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Bind Group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(bevel_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    // FXAA中間テクスチャへ描画（最終出力はFXAAパスが行う）
                    view: &self.fxaa_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ドット絵ON時は合成済みLDR(fxaa_view)をドット化して output へ。
        // ピクセルアートにアンチエイリアスは逆効果なので FXAA は通さない。
        // 深度(#3 輪郭用)は gbuffer があれば渡す（無ければパス側でダミー）。
        if self.pixel_art_enabled {
            let depth = self.gbuffer.as_ref().map(|g| &g.depth_view);
            self.pixel_art.execute(encoder, device, &self.fxaa_view, output_view, depth);
            return;
        }

        // === FXAA パス: 中間テクスチャをサンプルして output_view へ ===
        let fxaa_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FXAA Bind Group"),
            layout: &self.fxaa_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.fxaa_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.fxaa_sampler),
                },
            ],
        });

        let mut fxaa_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("FXAA Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        fxaa_pass.set_pipeline(&self.fxaa_pipeline);
        fxaa_pass.set_bind_group(0, &fxaa_bg, &[]);
        fxaa_pass.draw(0..3, 0..1);
    }

    /// リサイズ
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, depth_view: wgpu::TextureView) {
        let (t, v) = create_hdr_texture(device, width, height, "Scene Color HDR");
        self.scene_color_texture = t;
        self.scene_color_view = v;

        if let Some(ssao) = &mut self.ssao {
            ssao.resize(device, width, height);
        }
        if let Some(bloom) = &mut self.bloom {
            bloom.resize(device, width, height);
        }
        if let Some(edge_bevel) = &mut self.edge_bevel {
            edge_bevel.resize(device, width, height);
        }
        // G-Buffer再作成（サイズ変更時にテクスチャを再構築）
        if self.gbuffer.is_some() {
            self.gbuffer = Some(GBuffer::new(device, width, height, depth_view));
        }
        // FXAA中間テクスチャ再作成
        let (ft, fv) =
            create_ldr_texture(device, width, height, self.surface_format, "FXAA Intermediate");
        self.fxaa_texture = ft;
        self.fxaa_view = fv;
        // ドット絵 低解像度2パス(#1)の中間RTも再作成
        self.pixel_art.resize(device, width, height);
        self.width = width;
        self.height = height;
    }
}

// === ヘルパー関数 ===

fn create_r8_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// FXAA中間用のLDRテクスチャ（surface_format, RENDER_ATTACHMENT | TEXTURE_BINDING）
fn create_ldr_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_hdr_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        // COPY_SRC: スクリーンスペース屈折で HDR シーンカラーを scene_copy へ
        // copy_texture_to_texture するために必須。
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// ハルトン列（擬似ランダムサンプリング用）
fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let inv_base = 1.0 / base as f32;
    while index > 0 {
        f *= inv_base;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

use wgpu::util::DeviceExt;

// === シェーダー定義 ===

const EDGE_BEVEL_SHADER: &str = r#"
@group(0) @binding(0) var normal_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_edge_bevel(in: VertexOutput) -> @location(0) f32 {
    let dims = vec2<f32>(textureDimensions(normal_tex));
    let texel = 1.0 / dims;

    // 中心の法線
    let n_c = textureSample(normal_tex, samp, in.uv).xyz * 2.0 - 1.0;

    // 4方向サンプリング（Sobel法線差分）
    let n_l = textureSample(normal_tex, samp, in.uv + vec2(-texel.x, 0.0)).xyz * 2.0 - 1.0;
    let n_r = textureSample(normal_tex, samp, in.uv + vec2(texel.x, 0.0)).xyz * 2.0 - 1.0;
    let n_u = textureSample(normal_tex, samp, in.uv + vec2(0.0, -texel.y)).xyz * 2.0 - 1.0;
    let n_d = textureSample(normal_tex, samp, in.uv + vec2(0.0, texel.y)).xyz * 2.0 - 1.0;

    // 法線の変化量
    let normal_diff = (
        length(n_c - n_l) + length(n_c - n_r) +
        length(n_c - n_u) + length(n_c - n_d)
    ) * 0.25;

    // 深度サンプリング
    let d_c = textureSample(depth_tex, samp, in.uv);
    let d_l = textureSample(depth_tex, samp, in.uv + vec2(-texel.x, 0.0));
    let d_r = textureSample(depth_tex, samp, in.uv + vec2(texel.x, 0.0));
    let d_u = textureSample(depth_tex, samp, in.uv + vec2(0.0, -texel.y));
    let d_d = textureSample(depth_tex, samp, in.uv + vec2(0.0, texel.y));

    // 深度不連続検出
    let depth_diff = (
        abs(d_c - d_l) + abs(d_c - d_r) +
        abs(d_c - d_u) + abs(d_c - d_d)
    );

    // 法線の急変はハイライト（エッジベベル）
    // 深度不連続はシルエットエッジ（ハイライトなし）
    let is_depth_edge = step(0.01, depth_diff);
    let edge_strength = smoothstep(0.1, 0.5, normal_diff) * (1.0 - is_depth_edge);

    // 1.0=変化なし、1.0+α=ハイライト追加
    return 1.0 + edge_strength * 0.3;
}
"#;

const SSAO_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec4<f32>,
    clip_min: vec4<f32>,
    clip_max: vec4<f32>,
    resolution: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0) var normal_tex: texture_2d<f32>;
@group(1) @binding(1) var depth_tex: texture_depth_2d;
@group(1) @binding(2) var noise_tex: texture_2d<f32>;
@group(1) @binding(3) var samp: sampler;
@group(1) @binding(4) var<uniform> kernel: array<vec4<f32>, 64>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_ssao(in: VertexOutput) -> @location(0) f32 {
    let normal_sample = textureSample(normal_tex, samp, in.uv);
    let normal = normal_sample.xyz * 2.0 - 1.0;
    let depth = textureSample(depth_tex, samp, in.uv);

    // 深度から視点空間位置を復元（簡易近似）
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, depth, 1.0);

    let dims = vec2<f32>(textureDimensions(normal_tex));
    let noise_uv = in.uv * dims / 4.0;
    let noise = textureSample(noise_tex, samp, noise_uv).xy * 2.0 - 1.0;

    // TBN構築
    let tangent = normalize(vec3<f32>(noise.x, noise.y, 0.0) - normal * dot(vec3<f32>(noise.x, noise.y, 0.0), normal));
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, normal);

    let radius = 0.5;
    var occlusion: f32 = 0.0;
    let sample_count = 32u;

    for (var i: u32 = 0u; i < sample_count; i++) {
        let sample_dir = tbn * kernel[i].xyz;
        let sample_uv = in.uv + sample_dir.xy * radius / dims;

        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        let sample_depth = textureSample(depth_tex, samp, sample_uv);
        let range_check = smoothstep(0.0, 1.0, radius / abs(depth - sample_depth + 0.001));

        if (sample_depth < depth - 0.001) {
            occlusion += range_check;
        }
    }

    return 1.0 - occlusion / f32(sample_count);
}
"#;

const BLUR_SHADER: &str = r#"
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_blur(in: VertexOutput) -> @location(0) f32 {
    let dims = vec2<f32>(textureDimensions(input_tex));
    let texel = 1.0 / dims;
    var result: f32 = 0.0;
    for (var x: i32 = -2; x <= 2; x++) {
        for (var y: i32 = -2; y <= 2; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            result += textureSample(input_tex, samp, in.uv + offset).r;
        }
    }
    return result / 25.0;
}
"#;

const BLOOM_SHADER: &str = r#"
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// 輝度抽出（閾値以上の明るい部分のみ）
@fragment
fn fs_extract(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(input_tex, samp, in.uv).rgb;
    let brightness = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let threshold = 1.0;
    if (brightness > threshold) {
        return vec4<f32>(color * (brightness - threshold), 1.0);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

// ダウンサンプル（バイリニアフィルタ）
@fragment
fn fs_downsample(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(input_tex));
    let texel = 1.0 / dims;

    // 13サンプルダウンサンプルフィルタ
    var color = textureSample(input_tex, samp, in.uv) * 4.0;
    color += textureSample(input_tex, samp, in.uv + vec2(-texel.x, 0.0));
    color += textureSample(input_tex, samp, in.uv + vec2(texel.x, 0.0));
    color += textureSample(input_tex, samp, in.uv + vec2(0.0, -texel.y));
    color += textureSample(input_tex, samp, in.uv + vec2(0.0, texel.y));
    return color / 8.0;
}

// アップサンプル（テントフィルタ + 加算合成用）
@fragment
fn fs_upsample(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(input_tex));
    let texel = 1.0 / dims;

    var color = textureSample(input_tex, samp, in.uv) * 4.0;
    color += textureSample(input_tex, samp, in.uv + vec2(-texel.x, texel.y));
    color += textureSample(input_tex, samp, in.uv + vec2(texel.x, texel.y));
    color += textureSample(input_tex, samp, in.uv + vec2(-texel.x, -texel.y));
    color += textureSample(input_tex, samp, in.uv + vec2(texel.x, -texel.y));
    return color / 8.0 * 0.3; // 強度0.3
}
"#;

const COMPOSITE_SHADER: &str = r#"
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var ssao_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var bevel_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// ACES トーンマッピング
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(scene_tex, samp, in.uv).rgb;

    // Bloom加算
    let bloom = textureSample(bloom_tex, samp, in.uv).rgb;
    color += bloom;

    // SSAO乗算
    let ao = textureSample(ssao_tex, samp, in.uv).r;
    color *= ao;

    // エッジベベル（ハイライト乗算）
    let bevel = textureSample(bevel_tex, samp, in.uv).r;
    color *= bevel;

    // トーンマッピング + ガンマ補正
    let mapped = aces_tonemap(color);
    let gamma_corrected = pow(mapped, vec3(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, 1.0);
}
"#;

const FXAA_SHADER: &str = r#"
// Fullscreen triangle (matches composite vs_main pattern)
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.uv = vec2<f32>(x, y);
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var s_color: sampler;

fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.299, 0.587, 0.114)); }

@fragment
fn fs_fxaa(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(t_color));
    let rcp = 1.0 / dims;
    let uv = in.uv;

    let EDGE_MIN: f32 = 1.0 / 24.0;
    let EDGE_MAX: f32 = 1.0 / 8.0;
    let SPAN_MAX: f32 = 8.0;
    let REDUCE_MUL: f32 = 1.0 / 8.0;
    let REDUCE_MIN: f32 = 1.0 / 128.0;

    let rgbM  = textureSampleLevel(t_color, s_color, uv, 0.0).rgb;
    let rgbNW = textureSampleLevel(t_color, s_color, uv + vec2<f32>(-1.0, -1.0) * rcp, 0.0).rgb;
    let rgbNE = textureSampleLevel(t_color, s_color, uv + vec2<f32>( 1.0, -1.0) * rcp, 0.0).rgb;
    let rgbSW = textureSampleLevel(t_color, s_color, uv + vec2<f32>(-1.0,  1.0) * rcp, 0.0).rgb;
    let rgbSE = textureSampleLevel(t_color, s_color, uv + vec2<f32>( 1.0,  1.0) * rcp, 0.0).rgb;

    let lM  = luma(rgbM);
    let lNW = luma(rgbNW);
    let lNE = luma(rgbNE);
    let lSW = luma(rgbSW);
    let lSE = luma(rgbSE);

    let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));

    // Skip non-edges (cheap early-ish out via mix at the end)
    var dir: vec2<f32>;
    dir.x = -((lNW + lNE) - (lSW + lSE));
    dir.y =  ((lNW + lSW) - (lNE + lSE));

    let dirReduce = max((lNW + lNE + lSW + lSE) * 0.25 * REDUCE_MUL, REDUCE_MIN);
    let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2<f32>(-SPAN_MAX), vec2<f32>(SPAN_MAX)) * rcp;

    let rgbA = 0.5 * (
        textureSampleLevel(t_color, s_color, uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb +
        textureSampleLevel(t_color, s_color, uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb);
    let rgbB = rgbA * 0.5 + 0.25 * (
        textureSampleLevel(t_color, s_color, uv + dir * -0.5, 0.0).rgb +
        textureSampleLevel(t_color, s_color, uv + dir *  0.5, 0.0).rgb);

    let lB = luma(rgbB);
    var result = rgbB;
    if (lB < lMin || lB > lMax) { result = rgbA; }

    // If contrast is below threshold, keep original (no AA) to avoid over-blurring flat areas.
    let contrast = lMax - lMin;
    if (contrast < max(EDGE_MIN, lMax * EDGE_MAX)) { result = rgbM; }

    return vec4<f32>(result, 1.0);
}
"#;

const PIXEL_ART_SHADER: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct Params {
    cell_px: f32,
    palette_id: f32,
    dither: f32,
    outline: f32,
    levels: f32,
    sat: f32,
    color_mode: f32,   // #2: 0=RGB最近傍 / 1=Oklab最近傍＋2色オーダードディザ
    outline_mode: f32, // #3: 0=輝度差 / 1=深度(シルエット)
};
@group(0) @binding(2) var<uniform> P: Params;
@group(0) @binding(3) var depth_tex: texture_depth_2d;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.299, 0.587, 0.114)); }

// Bayer 4x4 オーダードディザ（戻り値 -0.5..0.5）
fn bayer4(p: vec2<i32>) -> f32 {
    var m = array<f32, 16>(
        0.0,  8.0,  2.0, 10.0,
       12.0,  4.0, 14.0,  6.0,
        3.0, 11.0,  1.0,  9.0,
       15.0,  7.0, 13.0,  5.0
    );
    let idx = (p.y & 3) * 4 + (p.x & 3);
    return m[idx] / 16.0 - 0.5;
}

fn posterize(c: vec3<f32>, levels: f32) -> vec3<f32> {
    let l = max(levels, 2.0);
    return floor(c * l + 0.5) / l;
}

// sRGB(γ2近似)→Oklab。減色を知覚距離で行うと肌などの色転びが減る(#2)。
fn rgb_to_oklab(crgb: vec3<f32>) -> vec3<f32> {
    let c = crgb * crgb; // sRGB->linear 近似
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    let l_ = pow(max(l, 0.0), 0.3333333);
    let m_ = pow(max(m, 0.0), 0.3333333);
    let s_ = pow(max(s, 0.0), 0.3333333);
    return vec3<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    );
}

// 固定パレットへ最近傍マッピング。cap で使用色数を制限する。
// bv は Bayer 値(-0.5..0.5)。color_mode=1 のとき 2色オーダードディザに使う。
fn nearest_palette(c: vec3<f32>, id: f32, cap: f32, bv: f32) -> vec3<f32> {
    var n: i32 = 0;
    var pal: array<vec3<f32>, 32>;
    if (id < 1.5) {
        // gameboy 4色
        pal[0] = vec3<f32>(15.0, 56.0, 15.0) / 255.0;
        pal[1] = vec3<f32>(48.0, 98.0, 48.0) / 255.0;
        pal[2] = vec3<f32>(139.0, 172.0, 15.0) / 255.0;
        pal[3] = vec3<f32>(155.0, 188.0, 15.0) / 255.0;
        n = 4;
    } else if (id < 2.5) {
        // pico8 16色
        pal[0]  = vec3<f32>(0.0, 0.0, 0.0) / 255.0;
        pal[1]  = vec3<f32>(29.0, 43.0, 83.0) / 255.0;
        pal[2]  = vec3<f32>(126.0, 37.0, 83.0) / 255.0;
        pal[3]  = vec3<f32>(0.0, 135.0, 81.0) / 255.0;
        pal[4]  = vec3<f32>(171.0, 82.0, 54.0) / 255.0;
        pal[5]  = vec3<f32>(95.0, 87.0, 79.0) / 255.0;
        pal[6]  = vec3<f32>(194.0, 195.0, 199.0) / 255.0;
        pal[7]  = vec3<f32>(255.0, 241.0, 232.0) / 255.0;
        pal[8]  = vec3<f32>(255.0, 0.0, 77.0) / 255.0;
        pal[9]  = vec3<f32>(255.0, 163.0, 0.0) / 255.0;
        pal[10] = vec3<f32>(255.0, 236.0, 39.0) / 255.0;
        pal[11] = vec3<f32>(0.0, 228.0, 54.0) / 255.0;
        pal[12] = vec3<f32>(41.0, 173.0, 255.0) / 255.0;
        pal[13] = vec3<f32>(131.0, 118.0, 156.0) / 255.0;
        pal[14] = vec3<f32>(255.0, 119.0, 168.0) / 255.0;
        pal[15] = vec3<f32>(255.0, 204.0, 170.0) / 255.0;
        n = 16;
    } else if (id < 3.5) {
        // famicom風 16色
        pal[0]  = vec3<f32>(0.0, 0.0, 0.0) / 255.0;
        pal[1]  = vec3<f32>(252.0, 252.0, 252.0) / 255.0;
        pal[2]  = vec3<f32>(188.0, 188.0, 188.0) / 255.0;
        pal[3]  = vec3<f32>(124.0, 124.0, 124.0) / 255.0;
        pal[4]  = vec3<f32>(0.0, 120.0, 248.0) / 255.0;
        pal[5]  = vec3<f32>(0.0, 0.0, 252.0) / 255.0;
        pal[6]  = vec3<f32>(104.0, 68.0, 252.0) / 255.0;
        pal[7]  = vec3<f32>(216.0, 0.0, 204.0) / 255.0;
        pal[8]  = vec3<f32>(228.0, 0.0, 88.0) / 255.0;
        pal[9]  = vec3<f32>(248.0, 56.0, 0.0) / 255.0;
        pal[10] = vec3<f32>(228.0, 92.0, 16.0) / 255.0;
        pal[11] = vec3<f32>(172.0, 124.0, 0.0) / 255.0;
        pal[12] = vec3<f32>(0.0, 184.0, 0.0) / 255.0;
        pal[13] = vec3<f32>(0.0, 168.0, 68.0) / 255.0;
        pal[14] = vec3<f32>(0.0, 136.0, 136.0) / 255.0;
        pal[15] = vec3<f32>(248.0, 216.0, 120.0) / 255.0;
        n = 16;
    } else if (id < 4.5) {
        // Endesga 32（暖色〜寒色の豊富な32色。実写が一番"映える"）
        pal[0]  = vec3<f32>(190.0, 74.0, 47.0) / 255.0;
        pal[1]  = vec3<f32>(215.0, 118.0, 67.0) / 255.0;
        pal[2]  = vec3<f32>(234.0, 212.0, 170.0) / 255.0;
        pal[3]  = vec3<f32>(228.0, 166.0, 114.0) / 255.0;
        pal[4]  = vec3<f32>(184.0, 111.0, 80.0) / 255.0;
        pal[5]  = vec3<f32>(115.0, 62.0, 57.0) / 255.0;
        pal[6]  = vec3<f32>(62.0, 39.0, 49.0) / 255.0;
        pal[7]  = vec3<f32>(162.0, 38.0, 51.0) / 255.0;
        pal[8]  = vec3<f32>(228.0, 59.0, 68.0) / 255.0;
        pal[9]  = vec3<f32>(247.0, 118.0, 34.0) / 255.0;
        pal[10] = vec3<f32>(254.0, 174.0, 52.0) / 255.0;
        pal[11] = vec3<f32>(254.0, 231.0, 97.0) / 255.0;
        pal[12] = vec3<f32>(99.0, 199.0, 77.0) / 255.0;
        pal[13] = vec3<f32>(62.0, 137.0, 72.0) / 255.0;
        pal[14] = vec3<f32>(38.0, 92.0, 66.0) / 255.0;
        pal[15] = vec3<f32>(25.0, 60.0, 62.0) / 255.0;
        pal[16] = vec3<f32>(18.0, 78.0, 137.0) / 255.0;
        pal[17] = vec3<f32>(0.0, 153.0, 219.0) / 255.0;
        pal[18] = vec3<f32>(44.0, 232.0, 245.0) / 255.0;
        pal[19] = vec3<f32>(192.0, 203.0, 220.0) / 255.0;
        pal[20] = vec3<f32>(139.0, 155.0, 180.0) / 255.0;
        pal[21] = vec3<f32>(90.0, 105.0, 136.0) / 255.0;
        pal[22] = vec3<f32>(58.0, 68.0, 102.0) / 255.0;
        pal[23] = vec3<f32>(38.0, 43.0, 68.0) / 255.0;
        pal[24] = vec3<f32>(24.0, 20.0, 37.0) / 255.0;
        pal[25] = vec3<f32>(255.0, 0.0, 68.0) / 255.0;
        pal[26] = vec3<f32>(104.0, 56.0, 108.0) / 255.0;
        pal[27] = vec3<f32>(181.0, 80.0, 136.0) / 255.0;
        pal[28] = vec3<f32>(246.0, 117.0, 122.0) / 255.0;
        pal[29] = vec3<f32>(232.0, 183.0, 150.0) / 255.0;
        pal[30] = vec3<f32>(194.0, 133.0, 105.0) / 255.0;
        pal[31] = vec3<f32>(143.0, 86.0, 59.0) / 255.0;
        n = 32;
    } else if (id < 5.5) {
        // Sweetie 16（鮮やかでポップな16色）
        pal[0]  = vec3<f32>(26.0, 28.0, 44.0) / 255.0;
        pal[1]  = vec3<f32>(93.0, 39.0, 93.0) / 255.0;
        pal[2]  = vec3<f32>(177.0, 62.0, 83.0) / 255.0;
        pal[3]  = vec3<f32>(239.0, 125.0, 87.0) / 255.0;
        pal[4]  = vec3<f32>(255.0, 205.0, 117.0) / 255.0;
        pal[5]  = vec3<f32>(167.0, 240.0, 112.0) / 255.0;
        pal[6]  = vec3<f32>(56.0, 183.0, 100.0) / 255.0;
        pal[7]  = vec3<f32>(37.0, 113.0, 121.0) / 255.0;
        pal[8]  = vec3<f32>(41.0, 54.0, 111.0) / 255.0;
        pal[9]  = vec3<f32>(59.0, 93.0, 201.0) / 255.0;
        pal[10] = vec3<f32>(65.0, 166.0, 246.0) / 255.0;
        pal[11] = vec3<f32>(115.0, 239.0, 247.0) / 255.0;
        pal[12] = vec3<f32>(244.0, 244.0, 244.0) / 255.0;
        pal[13] = vec3<f32>(148.0, 176.0, 194.0) / 255.0;
        pal[14] = vec3<f32>(86.0, 108.0, 134.0) / 255.0;
        pal[15] = vec3<f32>(51.0, 60.0, 87.0) / 255.0;
        n = 16;
    } else if (id < 6.5) {
        // CGA（レトロPC 4色：黒/シアン/マゼンタ/白）
        pal[0] = vec3<f32>(0.0, 0.0, 0.0) / 255.0;
        pal[1] = vec3<f32>(85.0, 255.0, 255.0) / 255.0;
        pal[2] = vec3<f32>(255.0, 85.0, 255.0) / 255.0;
        pal[3] = vec3<f32>(255.0, 255.0, 255.0) / 255.0;
        n = 4;
    } else {
        // 1-bit（2色：ほぼ白黒。一番"キモく"なる極端モード）
        pal[0] = vec3<f32>(18.0, 16.0, 28.0) / 255.0;
        pal[1] = vec3<f32>(237.0, 237.0, 230.0) / 255.0;
        n = 2;
    }
    // 使用色数を cap に制限。先頭N個だと暗色などに偏るのでパレット全体から
    // ストライドで均等に間引いて選ぶ。cap>=パレット色数なら全色を使う。
    let lim = clamp(i32(cap + 0.5), 2, n);
    if (P.color_mode > 0.5) {
        // #2: Oklab で最近傍2色を求め、距離比に応じて Bayer で振り分け（オーダードディザ）。
        let co = rgb_to_oklab(c);
        var b1 = pal[0]; var d1 = 1e9;
        var b2 = pal[0]; var d2 = 1e9;
        for (var i: i32 = 0; i < lim; i = i + 1) {
            let idx = (i * n) / lim;
            let d = distance(co, rgb_to_oklab(pal[idx]));
            if (d < d1) { d2 = d1; b2 = b1; d1 = d; b1 = pal[idx]; }
            else if (d < d2) { d2 = d; b2 = pal[idx]; }
        }
        if (P.dither > 0.5) {
            let t = d1 / (d1 + d2 + 1e-6); // 0(=b1ぴったり)..0.5(=中間)
            if ((bv + 0.5) < t) { return b2; }
        }
        return b1;
    } else {
        // 従来: RGB 最近傍。
        var best = pal[0];
        var bestd = 1e9;
        for (var i: i32 = 0; i < lim; i = i + 1) {
            let idx = (i * n) / lim;
            let d = distance(c, pal[idx]);
            if (d < bestd) { bestd = d; best = pal[idx]; }
        }
        return best;
    }
}

fn quantize(c: vec3<f32>, bv: f32) -> vec3<f32> {
    // 色数コントロール（levels=色数）。Posterizeモードは階調数、
    // パレットモードは「使うパレット色数」としてそのまま効く。
    if (P.palette_id < 0.5) {
        return posterize(c, P.levels);
    }
    return nearest_palette(c, P.palette_id, P.levels, bv);
}

// セルの代表色をエリア平均で求める。中心1点サンプルだと細部ノイズが
// 残って "低解像度の写真=モザイク" に見えるため、3x3 平均でフラットなセル色を作る。
fn cell_color(cell_idx: vec2<f32>, cell: f32, dims: vec2<f32>) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    let base = cell_idx * cell;
    for (var sy: i32 = 0; sy < 2; sy = sy + 1) {
        for (var sx: i32 = 0; sx < 2; sx = sx + 1) {
            let sub = (vec2<f32>(f32(sx), f32(sy)) + vec2<f32>(0.5)) / 2.0;
            let p = (base + sub * cell) / dims;
            acc = acc + textureSampleLevel(src_tex, samp, p, 0.0).rgb;
        }
    }
    return acc / 4.0;
}

// 1セルぶんの最終色（サンプル→彩度/コントラスト→量子化）。輪郭は含めない。
// full-res パスと低解像度パス(#1)で共有する。
fn shade_cell(cell_idx: vec2<f32>, cell: f32, dims: vec2<f32>) -> vec3<f32> {
    var col = cell_color(cell_idx, cell, dims);

    // 彩度ブースト＋コントラスト押し（限られたパレットに色を"寄せ"て写真っぽさを除去）
    let l0 = luma(col);
    col = mix(vec3<f32>(l0), col, P.sat);
    col = (col - vec3<f32>(0.5)) * 1.18 + vec3<f32>(0.5);

    let bv = bayer4(vec2<i32>(cell_idx));
    // 従来モードのみ量子化前に加算ディザ。Oklabモード(#2)は nearest 内で2色ディザ。
    if (P.dither > 0.5 && P.color_mode < 0.5) {
        let amt = select(0.18, 1.0 / max(P.levels, 2.0), P.palette_id < 0.5);
        col = col + vec3<f32>(bv * amt);
    }
    col = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
    return quantize(col, bv);
}

// 深度テクスチャの安全読み出し（範囲外はクランプ）。
fn depth_at(px: vec2<f32>, ddims: vec2<f32>) -> f32 {
    let ci = clamp(vec2<i32>(px), vec2<i32>(0, 0), vec2<i32>(ddims) - vec2<i32>(1, 1));
    return textureLoad(depth_tex, ci, 0);
}

// #3: 深度の不連続(シルエット)を 0..1 のエッジ強度で返す。深度は非線形なので相対差で評価。
fn depth_edge(center_px: vec2<f32>, cell: f32) -> f32 {
    let ddims = vec2<f32>(textureDimensions(depth_tex));
    let dc = depth_at(center_px, ddims);
    let dl = depth_at(center_px - vec2<f32>(cell, 0.0), ddims);
    let dr = depth_at(center_px + vec2<f32>(cell, 0.0), ddims);
    let du = depth_at(center_px - vec2<f32>(0.0, cell), ddims);
    let dd = depth_at(center_px + vec2<f32>(0.0, cell), ddims);
    let g = abs(dc - dl) + abs(dc - dr) + abs(dc - du) + abs(dc - dd);
    let rel = g / (dc + 0.001);
    return smoothstep(0.01, 0.05, rel);
}

// === full-res 単パス（#1 OFF）===
@fragment
fn fs_pixel_art(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_tex));
    let cell = max(P.cell_px, 1.0);
    let cell_idx = floor(in.position.xy / cell);
    var q = shade_cell(cell_idx, cell, dims);

    if (P.outline > 0.01) {
        let center_px = (cell_idx + vec2<f32>(0.5)) * cell;
        var edge = 0.0;
        if (P.outline_mode > 0.5) {
            edge = depth_edge(center_px, cell);
        } else {
            let here = center_px / dims;
            let texel = cell / dims;
            let ln = luma(textureSampleLevel(src_tex, samp, here - vec2<f32>(texel.x, 0.0), 0.0).rgb);
            let lu = luma(textureSampleLevel(src_tex, samp, here - vec2<f32>(0.0, texel.y), 0.0).rgb);
            let lc = luma(textureSampleLevel(src_tex, samp, here, 0.0).rgb);
            edge = smoothstep(0.12, 0.30, max(abs(lc - ln), abs(lc - lu)));
        }
        let ostr = select(0.8, 2.0, P.outline_mode > 0.5); // 深度=黒インク濃く / 輝度=控えめ
        q = q * (1.0 - clamp(edge * P.outline * ostr, 0.0, 1.0));
    }
    return vec4<f32>(q, 1.0);
}

// === #1 低解像度2パス: Pass A（1画素=1セル。低解像度ビューポートへ描く）===
@fragment
fn fs_dot_lo(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_tex)); // 入力=フル解像度
    let cell = max(P.cell_px, 1.0);
    let cell_idx = floor(in.position.xy); // ビューポートの1画素がそのまま1セル
    return vec4<f32>(shade_cell(cell_idx, cell, dims), 1.0);
}

// === #1 低解像度2パス: Pass B（lores を nearest 拡大＋深度/輝度輪郭）===
@fragment
fn fs_dot_up(in: VertexOutput) -> @location(0) vec4<f32> {
    let cell = max(P.cell_px, 1.0);
    let cell_idx = floor(in.position.xy / cell);
    let ci = vec2<i32>(cell_idx);
    var q = textureLoad(src_tex, ci, 0).rgb; // src_tex=lores（セル色がそのまま入っている）

    if (P.outline > 0.01) {
        let center_px = (cell_idx + vec2<f32>(0.5)) * cell;
        var edge = 0.0;
        if (P.outline_mode > 0.5) {
            edge = depth_edge(center_px, cell);
        } else {
            let lc = luma(q);
            let ln = luma(textureLoad(src_tex, ci - vec2<i32>(1, 0), 0).rgb);
            let lu = luma(textureLoad(src_tex, ci - vec2<i32>(0, 1), 0).rgb);
            edge = smoothstep(0.12, 0.30, max(abs(lc - ln), abs(lc - lu)));
        }
        let ostr = select(0.8, 2.0, P.outline_mode > 0.5); // 深度=黒インク濃く / 輝度=控えめ
        q = q * (1.0 - clamp(edge * P.outline * ostr, 0.0, 1.0));
    }
    return vec4<f32>(q, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halton_index_zero() {
        assert!((halton(0, 2)).abs() < 1e-6, "halton(0, base) = 0");
        assert!((halton(0, 3)).abs() < 1e-6);
    }

    #[test]
    fn test_halton_base2_known_values() {
        // halton(1,2) = 0.5, halton(2,2) = 0.25, halton(3,2) = 0.75
        assert!((halton(1, 2) - 0.5).abs() < 1e-6);
        assert!((halton(2, 2) - 0.25).abs() < 1e-6);
        assert!((halton(3, 2) - 0.75).abs() < 1e-6);
        assert!((halton(4, 2) - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_halton_base3_known_values() {
        // halton(1,3) = 1/3, halton(2,3) = 2/3, halton(3,3) = 1/9
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-5);
        assert!((halton(2, 3) - 2.0 / 3.0).abs() < 1e-5);
        assert!((halton(3, 3) - 1.0 / 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_halton_range() {
        // 全値が[0,1)の範囲内
        for i in 0..100 {
            let h2 = halton(i, 2);
            let h3 = halton(i, 3);
            assert!(h2 >= 0.0 && h2 < 1.0, "halton({i},2)={h2} out of range");
            assert!(h3 >= 0.0 && h3 < 1.0, "halton({i},3)={h3} out of range");
        }
    }

    #[test]
    fn test_halton_uniqueness() {
        // 最初の16サンプルがすべて異なる
        let vals: Vec<f32> = (1..17).map(|i| halton(i, 2)).collect();
        for i in 0..vals.len() {
            for j in (i + 1)..vals.len() {
                assert!((vals[i] - vals[j]).abs() > 1e-6,
                    "halton({},{}) == halton({},{}) = {}", i + 1, 2, j + 1, 2, vals[i]);
            }
        }
    }
}
