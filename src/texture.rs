//! テクスチャ管理

use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, warn};

/// テクスチャエラー
#[derive(Debug, Error)]
pub enum TextureError {
    #[error("画像読み込みエラー: {0}")]
    ImageLoad(String),

    #[error("テクスチャ作成エラー: {0}")]
    Creation(String),
}

/// デフォルト白テクスチャのID
pub const DEFAULT_TEXTURE_ID: &str = "__default_white__";

/// 「塗布なし」の既定テクスチャID（透明 RGBA=0）。塗布マップ(group 3)を持たないメッシュ用。
/// 白(DEFAULT)だと alpha=1 で全面が塗られてしまうため、塗布のフォールバックは透明にする。
pub const PAINT_NONE_ID: &str = "__paint_none__";

/// GPU上のテクスチャとバインドグループ
pub struct GpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
}

/// テクスチャマネージャー
pub struct TextureManager {
    textures: HashMap<String, GpuTexture>,
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// 体表塗布(group 3)用レイアウト: binding 0/1=色+被覆テクスチャ, 2/3=塗布時法線テクスチャ。
    /// テクスチャ枚数を増やすのにバインドグループ「数」を増やさない（max_bind_groups=4 制限回避）。
    pub paint_bind_group_layout: wgpu::BindGroupLayout,
    /// 色id -> 合成済み paint バインドグループ（色tex＋法線tex を1グループに束ねたもの）。
    paint_bind_groups: HashMap<String, wgpu::BindGroup>,
    sampler: wgpu::Sampler,
}

impl TextureManager {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Texture Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
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

        // 体表塗布 group 3 用: 色+被覆(0/1) と 塗布時法線(2/3) の2テクスチャを1グループに束ねる。
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
            count: None,
        };
        let samp_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let paint_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Paint Bind Group Layout"),
                entries: &[tex_entry(0), samp_entry(1), tex_entry(2), samp_entry(3)],
            });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mut manager = Self {
            textures: HashMap::new(),
            bind_group_layout,
            paint_bind_group_layout,
            paint_bind_groups: HashMap::new(),
            sampler,
        };

        manager.create_from_rgba(device, queue, DEFAULT_TEXTURE_ID, 1, 1, &[255, 255, 255, 255]);
        manager.create_from_rgba(device, queue, PAINT_NONE_ID, 1, 1, &[0, 0, 0, 0]);
        // 塗布なしメッシュ用の既定 paint グループ（色=透明 / 法線=透明）。
        manager.build_paint_bind_group(device, PAINT_NONE_ID, PAINT_NONE_ID, PAINT_NONE_ID);
        manager
    }

    /// 色テクスチャと法線テクスチャを1つの paint バインドグループに束ねて `key` で登録する。
    /// 両テクスチャは事前に create_from_rgba 済みであること。group 3 に丸ごとバインドして使う。
    pub fn build_paint_bind_group(
        &mut self,
        device: &wgpu::Device,
        key: &str,
        color_id: &str,
        normal_id: &str,
    ) {
        let (Some(color), Some(normal)) =
            (self.textures.get(color_id), self.textures.get(normal_id))
        else {
            return;
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Paint Bind Group {}", key)),
            layout: &self.paint_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&color.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal.view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.paint_bind_groups.insert(key.to_string(), bg);
    }

    /// group 3 にバインドする paint グループを引く。未登録（塗布なし）は既定の透明グループ。
    pub fn get_paint_bind_group(&self, key: Option<&str>) -> &wgpu::BindGroup {
        let k = key.unwrap_or(PAINT_NONE_ID);
        self.paint_bind_groups
            .get(k)
            .unwrap_or_else(|| &self.paint_bind_groups[PAINT_NONE_ID])
    }

    /// 既存テクスチャの中身だけを差し替える（バインドグループは作り直さない＝塗布マップの
    /// 毎フレーム更新を安価に）。サイズは作成時と同じである前提。未作成 id なら何もしない。
    pub fn update_rgba(&self, queue: &wgpu::Queue, id: &str, width: u32, height: u32, rgba: &[u8]) {
        let Some(tex) = self.textures.get(id) else { return };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
    }

    pub fn create_from_rgba(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) {
        self.create_from_rgba_fmt(device, queue, id, width, height, rgba, wgpu::TextureFormat::Rgba8UnormSrgb);
    }

    /// リニア(非sRGB)テクスチャとして登録。法線マップ等、値をそのまま使うデータ用
    /// （sRGB だとガンマ変換で encode(n*0.5+0.5) が歪み、表裏判定が壊れる）。
    pub fn create_from_rgba_linear(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) {
        self.create_from_rgba_fmt(device, queue, id, width, height, rgba, wgpu::TextureFormat::Rgba8Unorm);
    }

    fn create_from_rgba_fmt(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
        format: wgpu::TextureFormat,
    ) {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Texture {}", id)),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Texture Bind Group {}", id)),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.textures.insert(
            id.to_string(),
            GpuTexture { texture, view, bind_group },
        );

        debug!("テクスチャ作成: {} ({}x{})", id, width, height);
    }

    #[cfg(feature = "gltf")]
    pub fn load_from_file(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: &str,
        path: &std::path::Path,
    ) -> Result<(), TextureError> {
        let img = image::open(path)
            .map_err(|e| TextureError::ImageLoad(format!("{}: {}", path.display(), e)))?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        self.create_from_rgba(device, queue, id, width, height, &rgba);
        Ok(())
    }

    #[cfg(feature = "gltf")]
    pub fn load_from_bytes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: &str,
        bytes: &[u8],
    ) -> Result<(), TextureError> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| TextureError::ImageLoad(e.to_string()))?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        self.create_from_rgba(device, queue, id, width, height, &rgba);
        Ok(())
    }

    pub fn get_bind_group(&self, id: Option<&str>) -> &wgpu::BindGroup {
        let key = id.unwrap_or(DEFAULT_TEXTURE_ID);
        match self.textures.get(key) {
            Some(tex) => &tex.bind_group,
            None => {
                if key != DEFAULT_TEXTURE_ID {
                    warn!("テクスチャ未発見: {}, デフォルト白を使用", key);
                }
                &self.textures[DEFAULT_TEXTURE_ID].bind_group
            }
        }
    }

    pub fn remove(&mut self, id: &str) {
        if id == DEFAULT_TEXTURE_ID {
            warn!("デフォルトテクスチャは削除できません");
            return;
        }
        self.textures.remove(id);
    }

    pub fn clear(&mut self) {
        self.textures.retain(|k, _| k == DEFAULT_TEXTURE_ID);
    }

    pub fn contains(&self, id: &str) -> bool {
        self.textures.contains_key(id)
    }

    /// テクスチャのwgpu::Textureへの参照を取得（copy_texture_to_texture等で使用）
    pub fn get_texture(&self, id: &str) -> Option<&wgpu::Texture> {
        self.textures.get(id).map(|t| &t.texture)
    }
}
