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
            sampler,
        };

        manager.create_from_rgba(device, queue, DEFAULT_TEXTURE_ID, 1, 1, &[255, 255, 255, 255]);
        manager
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
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
