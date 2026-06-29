//! メインレンダラー
//!
//! UI非依存の3Dレンダラー。surface/windowは呼び出し側が管理する。
//! `render_to_view()` で外部のカラー/深度ビューに描画。

use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tracing::info;
use wgpu::util::DeviceExt;

use crate::{
    Camera, CameraUniform, GpuLight, GpuVertex, InstanceData,
    Light, LightHeader, LightKind, LightStorageData, LightUniform, LineVertex, TextureManager,
    RenderMesh, GaussianCloud,
    pipeline,
    shadow::{SHADOW_MAP_SIZE, POINT_SHADOW_ATLAS_SIZE, POINT_SHADOW_TILE_SIZE, MAX_POINT_SHADOW_CASTERS},
    splat::SplatCloudData,
    quality::QualitySettings,
    post_process::PostProcessPipeline,
};

/// レンダラーエラー
#[derive(Debug, Error)]
pub enum RendererError {
    #[error("GPUデバイスの取得に失敗: {0}")]
    DeviceCreation(String),

    #[error("サーフェス設定エラー: {0}")]
    SurfaceConfiguration(String),

    #[error("レンダリングエラー: {0}")]
    Rendering(String),

    #[error("パイプラインエラー: {0}")]
    Pipeline(#[from] pipeline::PipelineError),
}

/// メッシュインスタンス（GPUバッファ参照）
pub struct MeshInstance {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub texture_id: Option<String>,
    /// 体表塗布マップの色id（group 3 で合成）。この id で「色+被覆 / 塗布時法線」を束ねた
    /// paint バインドグループを引く。None なら透明(__paint_none__)。
    pub paint_texture_id: Option<String>,
}

/// Splatクラウドのレンダリングインスタンス
#[allow(dead_code)]
struct SplatInstance {
    splat_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    data: SplatCloudData,
}

/// ポイントライトシャドウキャスター情報
#[derive(Clone)]
pub struct PointShadowCaster {
    pub position: [f32; 3],
    pub view_projs: [[[f32; 4]; 4]; 6],
    pub atlas_offset: [u32; 2],
    pub tile_size: u32,
}

/// レンダラー
pub struct Renderer {
    // GPU resources
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    // Depth
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    // Group 0: Camera
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    // Group 1: Lights
    light_uniform_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    pub light_bind_group_layout: wgpu::BindGroupLayout,
    // Group 2: Textures
    pub texture_manager: TextureManager,
    // Pipelines
    main_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    // Buffers
    instance_buffer: wgpu::Buffer,
    line_vertex_buffer: wgpu::Buffer,
    line_vertex_count: u32,
    point_vertex_buffer: wgpu::Buffer,
    point_vertex_count: u32,
    // Meshes
    meshes: HashMap<String, MeshInstance>,
    // State
    clear_color: wgpu::Color,
    /// trueの場合、render_to_viewでClearをスキップ（外部で事前描画済みの場合）
    skip_clear: bool,
    clip_min: [f32; 4],
    clip_max: [f32; 4],
    // === Shadow Map ===
    shadow_enabled: bool,
    shadow_depth_texture: Option<wgpu::Texture>,
    shadow_depth_view: Option<wgpu::TextureView>,
    shadow_sampler: Option<wgpu::Sampler>,
    shadow_pipeline: Option<wgpu::RenderPipeline>,
    shadow_bind_group: Option<wgpu::BindGroup>,
    shadow_bind_group_layout: Option<wgpu::BindGroupLayout>,
    shadow_light_vp_buffer: Option<wgpu::Buffer>,
    shadow_light_vp_bind_group: Option<wgpu::BindGroup>,
    light_view_proj: [[f32; 4]; 4],
    main_pipeline_with_shadow: Option<wgpu::RenderPipeline>,
    // === Point Light Shadow Atlas ===
    point_shadow_atlas: Option<wgpu::Texture>,
    point_shadow_atlas_view: Option<wgpu::TextureView>,
    point_shadow_casters: Vec<PointShadowCaster>,
    // === Gaussian Splatting ===
    splat_pipeline: Option<wgpu::RenderPipeline>,
    splat_bind_group_layout: Option<wgpu::BindGroupLayout>,
    splat_clouds: HashMap<String, SplatInstance>,
    // === Quality Settings ===
    quality_settings: QualitySettings,
    msaa_texture: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    // === Post Process ===
    post_process: Option<PostProcessPipeline>,
    // === Screen-Space Refraction (水専用) ===
    // 不透明描画後の HDR シーンカラーを copy_texture_to_texture でここへ複製し、
    // 半透明の屈折パスから group2 でサンプルする。屈折ON(has_pp==true)時のみ使う。
    scene_copy_texture: wgpu::Texture,
    scene_copy_view: wgpu::TextureView,
    scene_copy_sampler: wgpu::Sampler,
    scene_color_bind_group_layout: wgpu::BindGroupLayout,
    scene_color_bind_group: wgpu::BindGroup,
    refraction_pipeline: wgpu::RenderPipeline,
}

impl Renderer {
    /// 新しいレンダラーを作成
    ///
    /// surface/windowは呼び出し側が管理する。ここではパイプラインとバッファの初期化のみ。
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<Self, RendererError> {
        let (depth_texture, depth_view) = Self::create_depth_texture_impl(&device, width, height, 1);

        // === Group 0: Camera ===
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // === Group 1: Lights (Uniform, WebGL2互換) ===
        let initial_light = LightUniform::default_lighting();
        let light_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Uniform Buffer"),
            size: std::mem::size_of::<LightUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&light_uniform_buffer, 0, bytemuck::bytes_of(&initial_light));

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Light Bind Group Layout"),
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
                ],
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light Bind Group"),
            layout: &light_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // === Group 2: Textures ===
        let texture_manager = TextureManager::new(&device, &queue);

        // === Pipelines ===
        let main_pipeline = pipeline::create_main_pipeline(
            &device, surface_format,
            &camera_bind_group_layout, &light_bind_group_layout, &texture_manager.bind_group_layout,
            &texture_manager.paint_bind_group_layout,
        )?;
        let transparent_pipeline = pipeline::create_transparent_pipeline(
            &device, surface_format,
            &camera_bind_group_layout, &light_bind_group_layout, &texture_manager.bind_group_layout,
            &texture_manager.paint_bind_group_layout,
        )?;
        let line_pipeline = pipeline::create_line_pipeline(
            &device, surface_format, &camera_bind_group_layout,
        )?;
        let point_pipeline = pipeline::create_point_pipeline(
            &device, surface_format, &camera_bind_group_layout,
        )?;

        // === Gaussian Splatting (WebGL2非対応のためWASMではスキップ) ===
        #[cfg(not(target_arch = "wasm32"))]
        let (splat_bind_group_layout, splat_pipeline) = {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Splat Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let pipe = pipeline::create_splat_pipeline(
                &device, surface_format, &camera_bind_group_layout, &layout,
            )?;
            (Some(layout), Some(pipe))
        };
        #[cfg(target_arch = "wasm32")]
        let (splat_bind_group_layout, splat_pipeline): (Option<wgpu::BindGroupLayout>, Option<wgpu::RenderPipeline>) = (None, None);

        // === Buffers ===
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: std::mem::size_of::<InstanceData>() as u64 * 1000,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let line_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Line Vertex Buffer"),
            size: std::mem::size_of::<LineVertex>() as u64 * 10000,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let point_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Point Vertex Buffer"),
            size: std::mem::size_of::<LineVertex>() as u64 * 10000,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // === Screen-Space Refraction ===
        // scene_copy は HDR(Rgba16Float)と同 format/同 size。usage=TEXTURE_BINDING|COPY_DST。
        let scene_color_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Scene Color Bind Group Layout"),
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
        let scene_copy_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Scene Copy Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let (scene_copy_texture, scene_copy_view) =
            Self::create_scene_copy_texture_impl(&device, width, height);
        let scene_color_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene Color Bind Group"),
            layout: &scene_color_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_copy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&scene_copy_sampler),
                },
            ],
        });
        let refraction_pipeline = pipeline::create_refraction_pipeline(
            &device, surface_format,
            &camera_bind_group_layout, &light_bind_group_layout,
            &scene_color_bind_group_layout, 1,
        )?;

        Ok(Self {
            device,
            queue,
            surface_format,
            width,
            height,
            depth_texture,
            depth_view,
            camera_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            light_uniform_buffer,
            light_bind_group,
            light_bind_group_layout,
            texture_manager,
            main_pipeline,
            transparent_pipeline,
            line_pipeline,
            point_pipeline,
            instance_buffer,
            line_vertex_buffer,
            line_vertex_count: 0,
            point_vertex_buffer,
            point_vertex_count: 0,
            meshes: HashMap::new(),
            clear_color: wgpu::Color { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
            skip_clear: false,
            clip_min: [0.0; 4],
            clip_max: [0.0; 4],
            shadow_enabled: false,
            shadow_depth_texture: None,
            shadow_depth_view: None,
            shadow_sampler: None,
            shadow_pipeline: None,
            shadow_bind_group: None,
            shadow_bind_group_layout: None,
            shadow_light_vp_buffer: None,
            shadow_light_vp_bind_group: None,
            light_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            main_pipeline_with_shadow: None,
            point_shadow_atlas: None,
            point_shadow_atlas_view: None,
            point_shadow_casters: Vec::new(),
            splat_pipeline,
            splat_bind_group_layout,
            splat_clouds: HashMap::new(),
            quality_settings: QualitySettings::default(),
            msaa_texture: None,
            msaa_view: None,
            post_process: None,
            scene_copy_texture,
            scene_copy_view,
            scene_copy_sampler,
            scene_color_bind_group_layout,
            scene_color_bind_group,
            refraction_pipeline,
        })
    }

    /// 屈折用 scene_copy テクスチャ生成（HDR と同 format/size、TEXTURE_BINDING|COPY_DST）。
    fn create_scene_copy_texture_impl(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Copy (Refraction)"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    /// scene_copy テクスチャと bind group を再構築（解像度変更時）。
    fn rebuild_scene_copy(&mut self, width: u32, height: u32) {
        let (t, v) = Self::create_scene_copy_texture_impl(&self.device, width, height);
        self.scene_copy_texture = t;
        self.scene_copy_view = v;
        self.scene_color_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene Color Bind Group"),
            layout: &self.scene_color_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.scene_copy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.scene_copy_sampler),
                },
            ],
        });
    }

    // ── メッシュ管理 ──

    /// メッシュを追加
    pub fn add_mesh(&mut self, id: &str, mesh: &RenderMesh, texture_id: Option<String>) {
        if mesh.is_empty() {
            return;
        }

        let gpu_vertices: Vec<GpuVertex> = mesh.gpu_vertices();

        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Vertex Buffer {}", id)),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Index Buffer {}", id)),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        self.meshes.insert(id.to_string(), MeshInstance {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
            texture_id,
            paint_texture_id: None,
        });
    }

    /// メッシュが登録済みか確認
    pub fn has_mesh(&self, id: &str) -> bool {
        self.meshes.contains_key(id)
    }

    /// メッシュインスタンスを取得（GPUバッファへの参照）
    pub fn get_mesh(&self, id: &str) -> Option<&MeshInstance> {
        self.meshes.get(id)
    }

    /// `render_to_view` の Clear 色をアルファ付きで設定する。アルファ 0 の透明
    /// クリアにすると、オフスクリーンに描いたジオメトリだけを後段で合成できる
    /// （ホスト UI の背景レイヤーとして埋め込む用途）。既存の `set_clear_color`
    /// は a=1.0 固定なので、透明にしたい場合はこちらを使う。
    pub fn set_clear_rgba(&mut self, color: wgpu::Color) {
        self.clear_color = color;
    }

    /// メッシュのテクスチャIDを設定/変更
    pub fn set_mesh_texture(&mut self, mesh_id: &str, texture_id: Option<String>) {
        if let Some(instance) = self.meshes.get_mut(mesh_id) {
            instance.texture_id = texture_id;
        }
    }

    /// メッシュの体表塗布マップを設定/変更。色テクスチャ(色+被覆)と法線テクスチャ(塗布時法線)を
    /// group 3 用の1バインドグループに束ねて color_id をキーに登録し、インスタンスへ割当てる。
    /// color_id が None なら透明（塗布なし）へ戻す。両テクスチャは事前に register 済みであること。
    pub fn set_mesh_paint(
        &mut self,
        mesh_id: &str,
        color_id: Option<String>,
        normal_id: Option<String>,
    ) {
        if let (Some(cid), Some(nid)) = (color_id.as_deref(), normal_id.as_deref()) {
            self.texture_manager.build_paint_bind_group(&self.device, cid, cid, nid);
        }
        if let Some(instance) = self.meshes.get_mut(mesh_id) {
            instance.paint_texture_id = color_id;
        }
    }

    /// 既存テクスチャの中身を差し替える（塗布マップの毎フレーム更新用。バインドグループは保持）。
    pub fn update_texture_rgba(&self, id: &str, width: u32, height: u32, rgba: &[u8]) {
        self.texture_manager.update_rgba(&self.queue, id, width, height, rgba);
    }

    /// メッシュの頂点データを更新
    pub fn update_mesh_vertices(&mut self, id: &str, mesh: &RenderMesh) {
        self.update_mesh_vertices_colored(id, mesh, None);
    }

    /// 頂点カラー付きでメッシュ頂点を更新
    pub fn update_mesh_vertices_colored(
        &mut self,
        id: &str,
        mesh: &RenderMesh,
        vertex_colors: Option<&[[f32; 4]]>,
    ) {
        if mesh.is_empty() {
            return;
        }
        let gpu_vertices: Vec<GpuVertex> = mesh.vertices.iter().enumerate()
            .map(|(i, v)| GpuVertex {
                position: [v.position.x as f32, v.position.y as f32, v.position.z as f32],
                normal: [v.normal.x as f32, v.normal.y as f32, v.normal.z as f32],
                uv: v.uv,
                tangent: [1.0, 0.0, 0.0, 1.0],
                vertex_color: vertex_colors
                    .and_then(|c| c.get(i).copied())
                    .unwrap_or([1.0, 1.0, 1.0, 1.0]),
            })
            .collect();
        let byte_size = (std::mem::size_of::<GpuVertex>() * gpu_vertices.len()) as u64;
        let idx_size = (std::mem::size_of::<u32>() * mesh.indices.len()) as u64;

        // 既存バッファに頂点・インデックスとも収まるなら再利用（GPU 確保を避ける）。
        // 重要: 縮小時はインデックスと index_count も更新しないと、旧トポロジで古い頂点を
        // 描き続けてメッシュが消えない（潰しメッシュで隠せない）。
        let reuse = self
            .meshes
            .get(id)
            .map_or(false, |e| byte_size <= e.vertex_buffer.size() && idx_size <= e.index_buffer.size());
        if reuse {
            {
                let e = self.meshes.get(id).unwrap();
                self.queue.write_buffer(&e.vertex_buffer, 0, bytemuck::cast_slice(&gpu_vertices));
                self.queue.write_buffer(&e.index_buffer, 0, bytemuck::cast_slice(&mesh.indices));
            }
            self.meshes.get_mut(id).unwrap().index_count = mesh.indices.len() as u32;
            return;
        }
        let texture_id = self.meshes.get(id).and_then(|m| m.texture_id.clone());
        self.meshes.remove(id);
        self.add_mesh(id, mesh, texture_id);
    }

    /// メッシュを削除
    pub fn remove_mesh(&mut self, id: &str) {
        self.meshes.remove(id);
    }

    /// 全メッシュをクリア
    pub fn clear_meshes(&mut self) {
        self.meshes.clear();
    }

    // ── カメラ ──

    /// カメラユニフォームを更新
    pub fn update_camera(&self, camera: &Camera) {
        let mut uniform = CameraUniform::from_camera(camera);
        uniform.clip_min = self.clip_min;
        uniform.clip_max = self.clip_max;
        // スクリーンスペース屈折/HDRモードのフラグ。z>=0.5 = ポストプロセス有効(HDRテクスチャへ描画)。
        // このとき各シェーダは tonemap/gamma を行わず線形のまま出力し、合成シェーダが一括で
        // tonemap+gamma する（二重処理＝ピンクのwashoutを防ぐ）。屈折のシーンカラーtexも有効。
        let hdr_flag = if self.post_process.is_some() { 1.0 } else { 0.0 };
        uniform.resolution = [self.width as f32, self.height as f32, hdr_flag, 0.0];
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// クリップボックスを設定
    pub fn set_clip_box(&mut self, min: [f32; 3], max: [f32; 3]) {
        self.clip_min = [min[0], min[1], min[2], 1.0];
        self.clip_max = [max[0], max[1], max[2], 0.0];
    }

    /// クリップボックスを解除
    pub fn clear_clip_box(&mut self) {
        self.clip_min = [0.0; 4];
        self.clip_max = [0.0; 4];
    }

    // ── ライト ──

    /// ライトを更新（Uniform版、最大8ライト）
    pub fn update_lights(&mut self, ambient_color: [f32; 3], lights: &[Light], _dark_room: bool) {
        let uniform = LightUniform::from_lights(ambient_color, lights);
        self.queue.write_buffer(&self.light_uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    // ── シャドウマップ ──

    /// シャドウマップを有効化
    pub fn setup_shadow_map(&mut self) -> Result<(), RendererError> {
        let shadow_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let shadow_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let light_vp_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Light VP Buffer"),
            size: std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // シャドウパス用バインドグループ
        let shadow_pass_bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Pass Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shadow_pass_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Pass Bind Group"),
            layout: &shadow_pass_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_vp_buffer.as_entire_binding(),
            }],
        });

        let shadow_pipeline = crate::create_shadow_pipeline(&self.device, &shadow_pass_bgl)?;

        // メインパス用 Group 3
        let shadow_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
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
            ],
        });

        let shadow_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Bind Group"),
            layout: &shadow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: light_vp_buffer.as_entire_binding(),
                },
            ],
        });

        let main_pipeline_with_shadow = pipeline::create_main_pipeline_with_shadow(
            &self.device,
            self.surface_format,
            &self.camera_bind_group_layout,
            &self.light_bind_group_layout,
            &self.texture_manager.bind_group_layout,
            &shadow_bind_group_layout,
        )?;

        self.shadow_depth_texture = Some(shadow_texture);
        self.shadow_depth_view = Some(shadow_view);
        self.shadow_sampler = Some(shadow_sampler);
        self.shadow_pipeline = Some(shadow_pipeline);
        self.shadow_bind_group = Some(shadow_bind_group);
        self.shadow_bind_group_layout = Some(shadow_bind_group_layout);
        self.shadow_light_vp_buffer = Some(light_vp_buffer);
        self.shadow_light_vp_bind_group = Some(shadow_pass_bg);
        self.main_pipeline_with_shadow = Some(main_pipeline_with_shadow);
        self.shadow_enabled = true;

        info!("Shadow map enabled ({}x{})", SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
        Ok(())
    }

    /// シャドウマップを無効化
    pub fn disable_shadow_map(&mut self) {
        self.shadow_enabled = false;
        self.shadow_depth_texture = None;
        self.shadow_depth_view = None;
        self.shadow_sampler = None;
        self.shadow_pipeline = None;
        self.shadow_bind_group = None;
        self.shadow_bind_group_layout = None;
        self.shadow_light_vp_buffer = None;
        self.shadow_light_vp_bind_group = None;
        self.main_pipeline_with_shadow = None;
    }

    /// シャドウマップが有効か
    pub fn is_shadow_enabled(&self) -> bool {
        self.shadow_enabled
    }

    /// シャドウマップ用のライトVP行列を更新
    pub fn update_shadow_matrix(&mut self, light_view_proj: [[f32; 4]; 4]) {
        self.light_view_proj = light_view_proj;
        if let Some(ref buffer) = self.shadow_light_vp_buffer {
            self.queue.write_buffer(buffer, 0, bytemuck::bytes_of(&light_view_proj));
        }
    }

    /// ポイントライトシャドウアトラスを初期化
    pub fn setup_point_shadow_atlas(&mut self) {
        if self.point_shadow_atlas.is_some() {
            return;
        }

        let atlas = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Point Shadow Atlas"),
            size: wgpu::Extent3d {
                width: POINT_SHADOW_ATLAS_SIZE,
                height: POINT_SHADOW_ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());

        self.point_shadow_atlas = Some(atlas);
        self.point_shadow_atlas_view = Some(atlas_view);
    }

    /// ポイントライトシャドウアトラスを無効化
    pub fn disable_point_shadow_atlas(&mut self) {
        self.point_shadow_atlas = None;
        self.point_shadow_atlas_view = None;
        self.point_shadow_casters.clear();
    }

    /// ポイントライトシャドウキャスターを更新
    pub fn update_point_shadow_casters(&mut self, lights: &[Light], camera_pos: [f32; 3]) {
        self.point_shadow_casters.clear();

        let mut scored: Vec<(usize, f32)> = lights.iter().enumerate()
            .filter(|(_, l)| l.kind == LightKind::Point)
            .map(|(i, l)| {
                let dx = l.direction_or_position[0] - camera_pos[0];
                let dy = l.direction_or_position[1] - camera_pos[1];
                let dz = l.direction_or_position[2] - camera_pos[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let score = l.intensity / (1.0 + dist_sq * 0.000001);
                (i, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let tile = POINT_SHADOW_TILE_SIZE;
        for (slot, &(light_idx, _)) in scored.iter().take(MAX_POINT_SHADOW_CASTERS).enumerate() {
            let light = &lights[light_idx];
            let pos = light.direction_or_position;
            let near: f32 = 10.0;
            let far: f32 = if light.range > 0.0 { light.range } else { 20000.0 };
            let view_projs = Self::compute_cube_view_projs(pos, near, far);

            self.point_shadow_casters.push(PointShadowCaster {
                position: pos,
                view_projs,
                atlas_offset: [0, slot as u32 * tile],
                tile_size: tile,
            });
        }
    }

    // ── 線分 / ポイント ──

    /// 線分の頂点データを更新
    pub fn update_lines(&mut self, vertices: &[LineVertex]) {
        self.line_vertex_count = vertices.len() as u32;
        Self::update_vertex_buffer(
            &self.device, &self.queue, &mut self.line_vertex_buffer,
            "Line Vertex Buffer", vertices,
        );
    }

    /// ポイントの頂点データを更新
    pub fn update_points(&mut self, vertices: &[LineVertex]) {
        self.point_vertex_count = vertices.len() as u32;
        Self::update_vertex_buffer(
            &self.device, &self.queue, &mut self.point_vertex_buffer,
            "Point Vertex Buffer", vertices,
        );
    }

    fn update_vertex_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &mut wgpu::Buffer,
        label: &str,
        vertices: &[LineVertex],
    ) {
        if vertices.is_empty() {
            return;
        }
        let required_size = std::mem::size_of_val(vertices) as u64;
        if required_size > buffer.size() {
            *buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: required_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(vertices));
    }

    // ── Gaussian Splatting ──

    /// Gaussian Splat CloudをGPUにアップロード
    pub fn upload_splat_cloud(&mut self, id: &str, cloud: &GaussianCloud) {
        let data = SplatCloudData::from_cloud(cloud);
        if data.count == 0 {
            return;
        }

        let splat_bind_group_layout = match &self.splat_bind_group_layout {
            Some(layout) => layout,
            None => return,
        };

        let splat_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Splat Buffer {}", id)),
            contents: bytemuck::cast_slice(&data.splats),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Splat Index Buffer {}", id)),
            contents: bytemuck::cast_slice(&data.sorted_indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Splat Bind Group {}", id)),
            layout: splat_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: splat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index_buffer.as_entire_binding(),
                },
            ],
        });

        self.splat_clouds.insert(id.to_string(), SplatInstance {
            splat_buffer,
            index_buffer,
            bind_group,
            data,
        });
    }

    /// Splat Cloudの深度ソートを更新
    pub fn update_splat_sort(&mut self, id: &str, camera: &Camera) {
        let cam_pos = &camera.position;
        let cam_pos_arr = [cam_pos.x as f32, cam_pos.y as f32, cam_pos.z as f32];

        if let Some(instance) = self.splat_clouds.get_mut(id) {
            instance.data.sort_by_depth(cam_pos_arr);
            self.queue.write_buffer(
                &instance.index_buffer,
                0,
                bytemuck::cast_slice(&instance.data.sorted_indices),
            );
        }
    }

    /// Splat Cloudを削除
    pub fn remove_splat_cloud(&mut self, id: &str) {
        self.splat_clouds.remove(id);
    }

    // ── 描画 ──

    /// 外部encoder/viewに3Dシーンを描画
    ///
    /// instances は `[0..opaque_count)` が不透明、`[opaque_count..)` が半透明。
    /// シャドウパスは内部で別エンコーダを作成しsubmitする。
    pub fn render_to_view(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &Camera,
        instances: &[(String, InstanceData)],
        opaque_count: usize,
    ) {
        self.update_camera(camera);

        // インスタンスデータ書き込み
        let instance_size = std::mem::size_of::<InstanceData>() as u64;
        self.upload_instances(instances);

        // シャドウパス（有効時のみ、別エンコーダでsubmit）
        if self.shadow_enabled {
            self.render_shadow_pass(instances, instance_size);
        }

        // ポストプロセス有効時は HDR テクスチャに描画
        let has_pp = self.post_process.is_some();
        let scene_target = if has_pp {
            self.post_process.as_ref().unwrap().scene_color_view()
        } else {
            color_view
        };

        // MSAA対応（ポストプロセス時は MSAA 無効 — HDR テクスチャは sample_count=1）
        let (render_view, resolve) = if !has_pp {
            if let Some(ref msaa_view) = self.msaa_view {
                (msaa_view as &wgpu::TextureView, Some(scene_target))
            } else {
                (scene_target, None)
            }
        } else {
            (scene_target, None)
        };

        // メインパス
        let use_shadow_pipeline = self.shadow_enabled
            && self.main_pipeline_with_shadow.is_some()
            && self.shadow_bind_group.is_some();

        let color_load = if self.skip_clear {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(self.clear_color)
        };
        let depth_load = if self.skip_clear {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(1.0)
        };

        if has_pp {
            // === スクリーンスペース屈折 2パス描画 ===
            // パスA: 不透明 + splat/point/line を HDR に描く（半透明は描かない）。
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Main Render Pass (Opaque)"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: render_view,
                        resolve_target: resolve,
                        ops: wgpu::Operations { load: color_load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations { load: depth_load, store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if use_shadow_pipeline {
                    self.draw_opaque_meshes_with_shadow(&mut pass, instances, opaque_count, instance_size);
                } else {
                    self.draw_opaque_meshes(&mut pass, instances, opaque_count, instance_size);
                }
                self.draw_splats_points_lines(&mut pass);
            }

            // HDR シーンカラー → scene_copy へコピー（屈折で背景としてサンプルする）。
            {
                let src = self.post_process.as_ref().unwrap();
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &src.scene_color_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.scene_copy_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
                );
            }

            // パスB: 半透明（屈折ON）。HDR/深度は Load（clearしない）。
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Main Render Pass (Transparent/Refraction)"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: render_view,
                        resolve_target: resolve,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                self.draw_transparent_meshes(&mut pass, instances, opaque_count, instance_size, true);
            }
        } else {
            // === 従来の単一パス描画（屈折なし、MSAA直描き可） ===
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: render_view,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: color_load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: depth_load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if use_shadow_pipeline {
                self.draw_meshes_with_shadow(&mut render_pass, instances, opaque_count, instance_size);
            } else {
                self.draw_meshes(&mut render_pass, instances, opaque_count, instance_size);
            }
            self.draw_splats_points_lines(&mut render_pass);
        }

        // ポストプロセス
        if let Some(pp) = &self.post_process {
            pp.execute(encoder, &self.camera_bind_group, &self.device, color_view);
        }
    }

    /// オフスクリーンレンダリング — RGBA バイト列を返す
    pub fn render_offscreen(
        &mut self,
        camera: &Camera,
        instances: &[(String, InstanceData)],
        opaque_count: usize,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, RendererError> {
        self.update_camera(camera);

        let color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Color"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Offscreen Encoder"),
        });

        self.render_to_view(&mut encoder, &color_view, &depth_view, camera, instances, opaque_count);

        // テクスチャ → バッファ転送
        let bytes_per_pixel = 4u32;
        let unpadded_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded_row.div_ceil(align) * align;
        let buffer_size = (padded_row * height) as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Offscreen Output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| RendererError::Rendering(format!("Buffer map recv error: {e}")))?
            .map_err(|e| RendererError::Rendering(format!("Buffer map error: {e}")))?;

        let data = slice.get_mapped_range();
        let color_format = self.surface_format;
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (row * padded_row) as usize;
            let end = start + unpadded_row as usize;
            let row_data = &data[start..end];
            let needs_swap = matches!(color_format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
            for pixel in row_data.chunks_exact(4) {
                if needs_swap {
                    rgba.push(pixel[2]);
                    rgba.push(pixel[1]);
                    rgba.push(pixel[0]);
                } else {
                    rgba.push(pixel[0]);
                    rgba.push(pixel[1]);
                    rgba.push(pixel[2]);
                }
                rgba.push(pixel[3]);
            }
        }
        drop(data);
        output_buffer.unmap();

        Ok(rgba)
    }

    // ── 品質設定 ──

    /// 品質設定を変更
    pub fn set_quality(&mut self, settings: QualitySettings) -> Result<(), RendererError> {
        let msaa_changed = self.quality_settings.msaa.count() != settings.msaa.count();
        let post_changed = self.quality_settings.needs_post_process() != settings.needs_post_process()
            || self.quality_settings.ssao != settings.ssao
            || self.quality_settings.bloom != settings.bloom
            || self.quality_settings.edge_bevel != settings.edge_bevel;

        self.quality_settings = settings.clone();

        // ポストプロセス再構築（パイプラインの対象フォーマット判定より先に行う＝scene_render_format が
        // post_process の有無で分岐するため）。
        if post_changed {
            if settings.needs_post_process() {
                self.post_process = Some(PostProcessPipeline::new(
                    &self.device,
                    &self.queue,
                    self.width,
                    self.height,
                    self.surface_format,
                    &self.camera_bind_group_layout,
                    &settings,
                ));
            } else {
                self.post_process = None;
            }
        }

        // MSAA かポストプロセスの切替でシーン描画ターゲット（フォーマット/サンプル数）が変わる。
        // MSAA色テクスチャ・深度テクスチャを実効サンプル数で作り直し、全パイプラインを再構築する。
        // ポストプロセス有効時は HDR(Rgba16Float, 単一サンプル)へ描くので、ここを誤ると
        // 「Render pipeline targets are incompatible with render pass」で落ちる。
        if msaa_changed || post_changed {
            let count = self.scene_sample_count();
            if !self.quality_settings.needs_post_process() && count > 1 {
                let (tex, view) = Self::create_msaa_texture_impl(
                    &self.device, self.width, self.height, self.surface_format, count,
                );
                self.msaa_texture = Some(tex);
                self.msaa_view = Some(view);
            } else {
                self.msaa_texture = None;
                self.msaa_view = None;
            }
            let (dt, dv) = Self::create_depth_texture_impl(
                &self.device, self.width, self.height, count,
            );
            self.depth_texture = dt;
            self.depth_view = dv;

            self.rebuild_pipelines()?;
        }

        info!("品質設定変更: {:?}", settings.preset);
        Ok(())
    }

    /// 現在の品質設定
    pub fn quality_settings(&self) -> &QualitySettings {
        &self.quality_settings
    }

    // ── リサイズ ──

    /// サイズ変更時に深度/MSAA/ポストプロセステクスチャを再作成
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;

        let msaa_count = self.quality_settings.msaa.count();
        let (dt, dv) = Self::create_depth_texture_impl(&self.device, width, height, msaa_count);
        self.depth_texture = dt;
        self.depth_view = dv;

        if msaa_count > 1 {
            let (mt, mv) = Self::create_msaa_texture_impl(
                &self.device, width, height, self.surface_format, msaa_count,
            );
            self.msaa_texture = Some(mt);
            self.msaa_view = Some(mv);
        }

        if let Some(pp) = &mut self.post_process {
            let pp_depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            pp.resize(&self.device, width, height, pp_depth_view);
        }

        // 屈折用 scene_copy も同サイズで作り直す（copy_texture_to_texture のサイズ一致のため）。
        self.rebuild_scene_copy(width, height);
    }

    // ── アクセサ ──

    pub fn set_clear_color(&mut self, r: f64, g: f64, b: f64) {
        self.clear_color = wgpu::Color { r, g, b, a: 1.0 };
    }

    /// Clearをスキップするか設定（外部で事前パスを描画済みの場合にtrue）
    pub fn set_skip_clear(&mut self, skip: bool) {
        self.skip_clear = skip;
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        self.device.clone()
    }

    pub fn queue_arc(&self) -> Arc<wgpu::Queue> {
        self.queue.clone()
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    pub fn texture_manager(&self) -> &TextureManager {
        &self.texture_manager
    }

    pub fn texture_manager_mut(&mut self) -> &mut TextureManager {
        &mut self.texture_manager
    }

    // ── 外部パス向けアクセサ ──

    /// カメラ bind group（Group 0）
    pub fn camera_bind_group(&self) -> &wgpu::BindGroup {
        &self.camera_bind_group
    }

    /// カメラ uniform バッファ
    pub fn camera_buffer(&self) -> &wgpu::Buffer {
        &self.camera_buffer
    }

    /// ライト bind group（Group 1）
    pub fn light_bind_group(&self) -> &wgpu::BindGroup {
        &self.light_bind_group
    }

    /// シャドウ bind group（Group 3, 有効時のみ）
    pub fn shadow_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.shadow_bind_group.as_ref()
    }

    /// シャドウ bind group layout（有効時のみ）
    pub fn shadow_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.shadow_bind_group_layout.as_ref()
    }

    /// テクスチャ読み込みヘルパー
    #[cfg(feature = "gltf")]
    pub fn load_texture_from_file(
        &mut self,
        id: &str,
        path: &std::path::Path,
    ) -> Result<(), crate::TextureError> {
        self.texture_manager.load_from_file(&self.device, &self.queue, id, path)
    }

    /// RGBAデータからテクスチャを登録
    pub fn register_texture_rgba(&mut self, id: &str, width: u32, height: u32, rgba: &[u8]) {
        self.texture_manager.create_from_rgba(&self.device, &self.queue, id, width, height, rgba);
    }

    /// RGBAデータからリニア(非sRGB)テクスチャを登録（法線マップ等の値テクスチャ用）。
    pub fn register_texture_rgba_linear(&mut self, id: &str, width: u32, height: u32, rgba: &[u8]) {
        self.texture_manager.create_from_rgba_linear(&self.device, &self.queue, id, width, height, rgba);
    }

    // ── 内部ヘルパー ──

    /// インスタンスデータをGPUに書き込み
    fn upload_instances(&mut self, instances: &[(String, InstanceData)]) {
        if instances.is_empty() {
            return;
        }
        let mut all_data: Vec<u8> = Vec::with_capacity(
            instances.len() * std::mem::size_of::<InstanceData>()
        );
        for (_, inst_data) in instances {
            all_data.extend_from_slice(bytemuck::bytes_of(inst_data));
        }
        let required = all_data.len() as u64;
        if required > self.instance_buffer.size() {
            self.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Instance Buffer"),
                size: required,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        self.queue.write_buffer(&self.instance_buffer, 0, &all_data);
    }

    /// シャドウ深度パスを実行
    fn render_shadow_pass(&self, instances: &[(String, InstanceData)], instance_size: u64) {
        if let (Some(shadow_depth_view), Some(shadow_pipeline), Some(shadow_light_vp_bg)) = (
            &self.shadow_depth_view,
            &self.shadow_pipeline,
            &self.shadow_light_vp_bind_group,
        ) {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Shadow Render Encoder"),
            });

            {
                let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Shadow Depth Pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: shadow_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                shadow_pass.set_pipeline(shadow_pipeline);
                shadow_pass.set_bind_group(0, shadow_light_vp_bg, &[]);

                for (idx, (mesh_id, _)) in instances.iter().enumerate() {
                    if let Some(mesh) = self.meshes.get(mesh_id) {
                        let offset = idx as u64 * instance_size;
                        shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        shadow_pass.set_vertex_buffer(1, self.instance_buffer.slice(offset..));
                        shadow_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        shadow_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }
    }

    /// メッシュ描画（シャドウなし）
    fn draw_meshes<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        instances: &[(String, InstanceData)],
        opaque_count: usize,
        instance_size: u64,
    ) {
        self.draw_opaque_meshes(render_pass, instances, opaque_count, instance_size);
        self.draw_transparent_meshes(render_pass, instances, opaque_count, instance_size, false);
    }

    /// Gaussian Splat / ポイント / 線分を描画（不透明寄りの補助ジオメトリ）。
    /// 屈折2パス描画ではパスA（不透明）側で描く＝屈折の背景に含める。
    fn draw_splats_points_lines<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if let Some(splat_pipeline) = &self.splat_pipeline {
            if !self.splat_clouds.is_empty() {
                render_pass.set_pipeline(splat_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                for instance in self.splat_clouds.values() {
                    render_pass.set_bind_group(1, &instance.bind_group, &[]);
                    render_pass.draw(0..4, 0..instance.data.count);
                }
            }
        }
        if self.point_vertex_count > 0 {
            render_pass.set_pipeline(&self.point_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.point_vertex_buffer.slice(..));
            render_pass.draw(0..self.point_vertex_count, 0..1);
        }
        if self.line_vertex_count > 0 {
            render_pass.set_pipeline(&self.line_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.line_vertex_buffer.slice(..));
            render_pass.draw(0..self.line_vertex_count, 0..1);
        }
    }

    /// 不透明メッシュ `[0..opaque_count)` のみを main_pipeline で描画。
    fn draw_opaque_meshes<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        instances: &[(String, InstanceData)],
        opaque_count: usize,
        instance_size: u64,
    ) {
        render_pass.set_pipeline(&self.main_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.light_bind_group, &[]);

        for (idx, (mesh_id, _)) in instances[..opaque_count].iter().enumerate() {
            if let Some(mesh) = self.meshes.get(mesh_id) {
                let tex_bind_group = self.texture_manager.get_bind_group(mesh.texture_id.as_deref());
                render_pass.set_bind_group(2, tex_bind_group, &[]);
                // group 3 = 体表塗布(色+被覆 / 塗布時法線 を束ねた1グループ)。無ければ透明。
                let paint_bg = self.texture_manager.get_paint_bind_group(mesh.paint_texture_id.as_deref());
                render_pass.set_bind_group(3, paint_bg, &[]);
                let offset = idx as u64 * instance_size;
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, self.instance_buffer.slice(offset..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
    }

    /// 半透明メッシュ `[opaque_count..)` を描画。
    /// `use_refraction==true` のとき、material[3] > 5.0 のインスタンスは屈折パイプライン
    /// (group2=scene_copy)で、それ以外は従来 transparent_pipeline で描く。
    /// `use_refraction==false` のときは全て従来 transparent_pipeline（屈折なし）。
    fn draw_transparent_meshes<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        instances: &[(String, InstanceData)],
        opaque_count: usize,
        instance_size: u64,
        use_refraction: bool,
    ) {
        if opaque_count >= instances.len() {
            return;
        }
        // パイプライン切り替えコストを抑えるため、直前のモードを覚えておく。
        // 0=未設定 / 1=transparent / 2=refraction
        let mut cur_mode = 0u8;

        for (i, (mesh_id, _)) in instances[opaque_count..].iter().enumerate() {
            let idx = opaque_count + i;
            if let Some(mesh) = self.meshes.get(mesh_id) {
                let is_refr = use_refraction && instances[idx].1.material[3] > 5.0;
                let want_mode = if is_refr { 2u8 } else { 1u8 };
                if want_mode != cur_mode {
                    if is_refr {
                        render_pass.set_pipeline(&self.refraction_pipeline);
                        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                        render_pass.set_bind_group(1, &self.light_bind_group, &[]);
                        // group2 = 不透明描画済みシーンカラーのコピー。
                        render_pass.set_bind_group(2, &self.scene_color_bind_group, &[]);
                    } else {
                        render_pass.set_pipeline(&self.transparent_pipeline);
                        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                        render_pass.set_bind_group(1, &self.light_bind_group, &[]);
                    }
                    cur_mode = want_mode;
                }

                if is_refr {
                    // 屈折パイプラインは group2 を scene_color に占有済み。group3 は使わない。
                    let offset = idx as u64 * instance_size;
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(offset..));
                    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                } else {
                    let tex_bind_group = self.texture_manager.get_bind_group(mesh.texture_id.as_deref());
                    render_pass.set_bind_group(2, tex_bind_group, &[]);
                    let paint_bg = self.texture_manager.get_paint_bind_group(mesh.paint_texture_id.as_deref());
                    render_pass.set_bind_group(3, paint_bg, &[]);
                    let offset = idx as u64 * instance_size;
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(offset..));
                    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }
    }

    /// メッシュ描画（シャドウ対応）
    fn draw_meshes_with_shadow<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        instances: &[(String, InstanceData)],
        opaque_count: usize,
        instance_size: u64,
    ) {
        let pipeline = self.main_pipeline_with_shadow.as_ref().unwrap();
        let shadow_bg = self.shadow_bind_group.as_ref().unwrap();

        // 不透明パス（シャドウ付き）
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.light_bind_group, &[]);
        render_pass.set_bind_group(3, shadow_bg, &[]);

        for (idx, (mesh_id, _)) in instances[..opaque_count].iter().enumerate() {
            if let Some(mesh) = self.meshes.get(mesh_id) {
                let tex_bind_group = self.texture_manager.get_bind_group(mesh.texture_id.as_deref());
                render_pass.set_bind_group(2, tex_bind_group, &[]);
                let offset = idx as u64 * instance_size;
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, self.instance_buffer.slice(offset..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        // 半透明パス（シャドウなし）。屈折なし版（単一パス描画なので scene_copy は未確定）。
        self.draw_transparent_meshes(render_pass, instances, opaque_count, instance_size, false);
    }

    /// 不透明メッシュ（シャドウ付き）のみを描画。屈折の2パス描画でパスAに使う。
    fn draw_opaque_meshes_with_shadow<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        instances: &[(String, InstanceData)],
        opaque_count: usize,
        instance_size: u64,
    ) {
        let pipeline = self.main_pipeline_with_shadow.as_ref().unwrap();
        let shadow_bg = self.shadow_bind_group.as_ref().unwrap();
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.light_bind_group, &[]);
        render_pass.set_bind_group(3, shadow_bg, &[]);

        for (idx, (mesh_id, _)) in instances[..opaque_count].iter().enumerate() {
            if let Some(mesh) = self.meshes.get(mesh_id) {
                let tex_bind_group = self.texture_manager.get_bind_group(mesh.texture_id.as_deref());
                render_pass.set_bind_group(2, tex_bind_group, &[]);
                let offset = idx as u64 * instance_size;
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, self.instance_buffer.slice(offset..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
    }

    /// パイプラインを再構築（MSAA変更時）
    /// シーン描画パイプラインの対象カラーフォーマット。ポストプロセス有効時は HDR テクスチャ
    /// (Rgba16Float) へ描くのでそれに合わせる。無効時はサーフェスフォーマット直描き。
    fn scene_render_format(&self) -> wgpu::TextureFormat {
        if self.quality_settings.needs_post_process() {
            wgpu::TextureFormat::Rgba16Float
        } else {
            self.surface_format
        }
    }

    /// シーン描画の MSAA サンプル数。ポストプロセス有効時は HDR テクスチャが単一サンプルなので 1。
    ///
    /// これは深度テクスチャ・MSAA カラーテクスチャ・全シーンパイプラインを作る基準値
    /// （`set_quality` 参照）。外部パス（例: bamiri の FinishPass）をシーン深度と共有して
    /// 重ねる場合、そのパイプラインのサンプル数をこの値に合わせる必要がある。
    pub fn scene_sample_count(&self) -> u32 {
        if self.quality_settings.needs_post_process() {
            1
        } else {
            self.quality_settings.msaa.count()
        }
    }

    /// MSAA 有効（かつポストプロセス無効）時のシーン用マルチサンプルカラービュー。
    /// シーンはここへ描き surface へ resolve される（`StoreOp::Store` で保持済み）。
    /// 外部パスを resolve 前に重ねたい場合、このビューへ `LoadOp::Load` で描き
    /// resolve_target に surface を指定する。MSAA 無効/ポストプロセス時は `None`。
    pub fn msaa_view(&self) -> Option<&wgpu::TextureView> {
        self.msaa_view.as_ref()
    }

    fn rebuild_pipelines(&mut self) -> Result<(), RendererError> {
        let fmt = self.scene_render_format();
        let msaa_samples = self.scene_sample_count();
        self.main_pipeline = pipeline::create_main_pipeline_msaa(
            &self.device, fmt,
            &self.camera_bind_group_layout, &self.light_bind_group_layout,
            &self.texture_manager.bind_group_layout,
            &self.texture_manager.paint_bind_group_layout, msaa_samples,
        )?;

        self.transparent_pipeline = pipeline::create_transparent_pipeline_msaa(
            &self.device, fmt,
            &self.camera_bind_group_layout, &self.light_bind_group_layout,
            &self.texture_manager.bind_group_layout,
            &self.texture_manager.paint_bind_group_layout, msaa_samples,
        )?;

        // 屈折パイプラインもシーンと同 format/同サンプル数で再構築（HDR時は Rgba16Float）。
        self.refraction_pipeline = pipeline::create_refraction_pipeline(
            &self.device, fmt,
            &self.camera_bind_group_layout, &self.light_bind_group_layout,
            &self.scene_color_bind_group_layout, msaa_samples,
        )?;

        self.line_pipeline = pipeline::create_line_pipeline_msaa(
            &self.device, fmt,
            &self.camera_bind_group_layout, msaa_samples,
        )?;

        self.point_pipeline = pipeline::create_point_pipeline_msaa(
            &self.device, fmt,
            &self.camera_bind_group_layout, msaa_samples,
        )?;

        if self.shadow_enabled {
            if let Some(ref shadow_bgl) = self.shadow_bind_group_layout {
                self.main_pipeline_with_shadow = Some(pipeline::create_main_pipeline_with_shadow_msaa(
                    &self.device, fmt,
                    &self.camera_bind_group_layout, &self.light_bind_group_layout,
                    &self.texture_manager.bind_group_layout, shadow_bgl, msaa_samples,
                )?);
            }
        }

        info!("パイプライン再構築完了 (format: {:?}, MSAA: {}x)", fmt, msaa_samples);
        Ok(())
    }

    // ── テクスチャ生成ヘルパー ──

    fn create_depth_texture_impl(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_texture_impl(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    // ── シャドウ数学ヘルパー ──

    /// キューブマップの6面のビュー射影行列を計算
    fn compute_cube_view_projs(pos: [f32; 3], near: f32, far: f32) -> [[[f32; 4]; 4]; 6] {
        let faces: [([f32; 3], [f32; 3]); 6] = [
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ];

        let mut result = [[[0.0f32; 4]; 4]; 6];
        let proj = Self::perspective_90(near, far);

        for (i, (dir, up)) in faces.iter().enumerate() {
            let target = [pos[0] + dir[0], pos[1] + dir[1], pos[2] + dir[2]];
            let view = Self::look_at(pos, target, *up);
            result[i] = Self::mat4_mul(&proj, &view);
        }

        result
    }

    /// 90° FOV 正方形 perspective 行列
    fn perspective_90(near: f32, far: f32) -> [[f32; 4]; 4] {
        let r = far - near;
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -far / r, -1.0],
            [0.0, 0.0, -far * near / r, 0.0],
        ]
    }

    /// look_at 行列 (column-major)
    fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
        let fx = target[0] - eye[0];
        let fy = target[1] - eye[1];
        let fz = target[2] - eye[2];
        let fl = (fx * fx + fy * fy + fz * fz).sqrt();
        if fl < 1e-10 {
            return [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]];
        }
        let (fx, fy, fz) = (fx / fl, fy / fl, fz / fl);

        let rx = fy * up[2] - fz * up[1];
        let ry = fz * up[0] - fx * up[2];
        let rz = fx * up[1] - fy * up[0];
        let rl = (rx * rx + ry * ry + rz * rz).sqrt();
        if rl < 1e-10 {
            return [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]];
        }
        let (rx, ry, rz) = (rx / rl, ry / rl, rz / rl);

        let ux = ry * fz - rz * fy;
        let uy = rz * fx - rx * fz;
        let uz = rx * fy - ry * fx;

        [
            [rx, ux, -fx, 0.0],
            [ry, uy, -fy, 0.0],
            [rz, uz, -fz, 0.0],
            [
                -(rx * eye[0] + ry * eye[1] + rz * eye[2]),
                -(ux * eye[0] + uy * eye[1] + uz * eye[2]),
                fx * eye[0] + fy * eye[1] + fz * eye[2],
                1.0,
            ],
        ]
    }

    /// 4x4 行列乗算 (column-major)
    fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut result = [[0.0f32; 4]; 4];
        for c in 0..4 {
            for r in 0..4 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += a[k][r] * b[c][k];
                }
                result[c][r] = sum;
            }
        }
        result
    }
}

#[cfg(test)]
mod shadow_math_tests {
    use super::*;

    #[test]
    fn test_perspective_90_identity_at_fov90() {
        let proj = Renderer::perspective_90(10.0, 20000.0);
        assert!((proj[0][0] - 1.0).abs() < 1e-6);
        assert!((proj[1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_perspective_90_clip_w() {
        let proj = Renderer::perspective_90(1.0, 100.0);
        assert!((proj[2][3] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_look_at_no_nan() {
        let view = Renderer::look_at([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
        for col in &view {
            for &val in col {
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_look_at_orthogonality() {
        let view = Renderer::look_at([100.0, 200.0, 300.0], [500.0, 0.0, 100.0], [0.0, 0.0, 1.0]);
        let dot01 = view[0][0] * view[1][0] + view[0][1] * view[1][1] + view[0][2] * view[1][2];
        let dot02 = view[0][0] * view[2][0] + view[0][1] * view[2][1] + view[0][2] * view[2][2];
        let dot12 = view[1][0] * view[2][0] + view[1][1] * view[2][1] + view[1][2] * view[2][2];
        assert!(dot01.abs() < 1e-4);
        assert!(dot02.abs() < 1e-4);
        assert!(dot12.abs() < 1e-4);
    }

    #[test]
    fn test_mat4_mul_identity() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let a = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
        ];
        let result = Renderer::mat4_mul(&a, &identity);
        for c in 0..4 {
            for r in 0..4 {
                assert!((result[c][r] - a[c][r]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_cube_view_projs_6_faces() {
        let vps = Renderer::compute_cube_view_projs([0.0, 0.0, 0.0], 10.0, 20000.0);
        assert_eq!(vps.len(), 6);
    }

    #[test]
    fn test_cube_view_projs_all_different() {
        let vps = Renderer::compute_cube_view_projs([0.0, 0.0, 0.0], 10.0, 20000.0);
        for i in 0..6 {
            for j in (i + 1)..6 {
                let mut same = true;
                'outer: for c in 0..4 {
                    for r in 0..4 {
                        if (vps[i][c][r] - vps[j][c][r]).abs() > 1e-6 {
                            same = false;
                            break 'outer;
                        }
                    }
                }
                assert!(!same, "VP[{i}] and VP[{j}] should be different");
            }
        }
    }
}
