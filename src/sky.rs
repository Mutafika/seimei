//! 空＋体積雲の背景パス（`shaders/sky_cloud.wgsl`）。
//!
//! シーンの**手前**に何も無い画素を埋める背景。不透明メッシュより先に、深度を書かず・読まずに
//! 全画面へ描く＝後から描くメッシュが普通に上へ乗る。
//!
//! 2パス構成（原本 unkai の設計をそのまま踏襲）:
//!   1. `fs_cloud` … 雲のレイマーチを**半解像度**でオフスクリーンへ。出力は色ではなく
//!      `vec4(散乱, 透過率)`。★ここをフル解像度で回すと一気に重くなる（原本で実測済み）。
//!   2. `fs_bg`   … 空をフル解像度で描き、1 の結果を bilinear 拡大して合成。
//!      輪郭が硬い物（太陽の縁）はフル解像度側に残る。
//!
//! ★出力は**線形HDR**。トーンマップは seimei のポストプロセスが一括で行う。

use wgpu::util::DeviceExt;

/// 雲パスの解像度比（1.0=フル）。0.5 で 1/4 の画素数。
const CLOUD_SCALE: f32 = 0.5;

/// シェーダへ渡す値。フィールドの並びは `sky_cloud.wgsl` の `struct Un` と一致させること。
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniform {
    pub time: f32,
    pub aspect: f32,
    /// 太陽高度 -1..1（0=地平）。空の色味と星の出方。sun と整合させること。
    pub sun_alt: f32,
    /// 雲量バイアス（+で曇り / -で晴れ）。
    pub cover_bias: f32,
    /// カメラ位置（m・Y-up）。w 未使用。
    pub cam: [f32; 4],
    /// カメラ前方。w = 焦点距離（呼び元の fov と合わせる。tan(fov/2) の逆数）。
    pub fwd: [f32; 4],
    pub rgt: [f32; 4],
    pub up: [f32; 4],
    /// 太陽へ向かう単位ベクトル（Y-up）。
    pub sun: [f32; 4],
}

impl Default for SkyUniform {
    fn default() -> Self {
        Self {
            time: 0.0,
            aspect: 1.0,
            sun_alt: 0.6,
            cover_bias: 0.0,
            cam: [0.0, 1000.0, 0.0, 0.0],
            fwd: [0.0, 0.0, 1.0, 1.15],
            rgt: [1.0, 0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0, 0.0],
            sun: [0.35, 0.82, 0.45, 0.0],
        }
    }
}

/// 空＋雲の背景パス。
pub struct SkyPass {
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    cloud_pipeline: wgpu::RenderPipeline,
    bg_pipeline: wgpu::RenderPipeline,
    cloud_bgl: wgpu::BindGroupLayout,
    cloud_view: wgpu::TextureView,
    cloud_bind_group: wgpu::BindGroup,
    cloud_size: (u32, u32),
    sampler: wgpu::Sampler,
    /// 雲テクスチャのフォーマット（合成先と別＝常に HDR で持つ）。
    cloud_format: wgpu::TextureFormat,
}

impl SkyPass {
    /// `target_format` = 合成先（シーンカラー）のフォーマット。`(w,h)` = そのサイズ。
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Cloud Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky_cloud.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sky Uniform"),
            contents: bytemuck::bytes_of(&SkyUniform::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky Uniform BG"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let cloud_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky Cloud Tex BGL"),
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

        // 雲は HDR で持つ（散乱値が 1.0 を超える。8bit に落とすと太陽側の銀縁が潰れる）。
        let cloud_format = wgpu::TextureFormat::Rgba16Float;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sky Cloud Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let cloud_pipeline = Self::make_pipeline(
            device, &shader, &[&uniform_bgl], cloud_format, "fs_cloud", "Sky Cloud Pipeline",
        );
        let bg_pipeline = Self::make_pipeline(
            device, &shader, &[&uniform_bgl, &cloud_bgl], target_format, "fs_bg", "Sky BG Pipeline",
        );

        let (cloud_view, cloud_bind_group, cloud_size) =
            Self::make_cloud_target(device, &cloud_bgl, &sampler, cloud_format, width, height);

        Self {
            uniform_buffer,
            uniform_bind_group,
            cloud_pipeline,
            bg_pipeline,
            cloud_bgl,
            cloud_view,
            cloud_bind_group,
            cloud_size,
            sampler,
            cloud_format,
        }
    }

    fn make_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layouts: &[&wgpu::BindGroupLayout],
        format: wgpu::TextureFormat,
        fs_entry: &str,
        label: &str,
    ) -> wgpu::RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: layouts,
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_bg"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // 背景なので深度は持たない（テストも書き込みもしない）。後段のメッシュが素直に上へ乗る。
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn make_cloud_target(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::TextureView, wgpu::BindGroup, (u32, u32)) {
        let w = ((width as f32 * CLOUD_SCALE) as u32).max(8);
        let h = ((height as f32 * CLOUD_SCALE) as u32).max(8);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Sky Cloud Target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky Cloud Tex BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        });
        (view, bg, (w, h))
    }

    /// 画面サイズが変わったら雲ターゲットを作り直す（毎フレーム呼んでよい＝同サイズなら何もしない）。
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let want = (
            ((width as f32 * CLOUD_SCALE) as u32).max(8),
            ((height as f32 * CLOUD_SCALE) as u32).max(8),
        );
        if want == self.cloud_size {
            return;
        }
        let (v, bg, size) = Self::make_cloud_target(
            device, &self.cloud_bgl, &self.sampler, self.cloud_format, width, height,
        );
        self.cloud_view = v;
        self.cloud_bind_group = bg;
        self.cloud_size = size;
    }

    pub fn update(&self, queue: &wgpu::Queue, u: &SkyUniform) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(u));
    }

    /// 背景を `color_view` へ描く。**不透明メッシュより先に**呼ぶこと。
    /// 内部で雲パス（半解像度）→合成パス（フル解像度）の2パスを積む。
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, color_view: &wgpu::TextureView) {
        {
            let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sky Cloud Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.cloud_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            p.set_pipeline(&self.cloud_pipeline);
            p.set_bind_group(0, &self.uniform_bind_group, &[]);
            p.draw(0..6, 0..1);
        }
        {
            let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sky BG Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            p.set_pipeline(&self.bg_pipeline);
            p.set_bind_group(0, &self.uniform_bind_group, &[]);
            p.set_bind_group(1, &self.cloud_bind_group, &[]);
            p.draw(0..6, 0..1);
        }
    }
}
