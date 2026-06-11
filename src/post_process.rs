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
        }
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

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Composite Pass"),
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
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &composite_bg, &[]);
        pass.draw(0..3, 0..1);
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
