//! 追加の描画ビュー（同じシーンを別のアングルで同時に出すペイン用）。
//!
//! 既定のビュー（[`Renderer::render_to_view`]）のカメラ uniform とインスタンスのバッファは
//! Renderer に 1 個ずつしか無い。`queue.write_buffer` は次の submit の先頭で効くので、
//! 1 回の submit の中で別のカメラを書いても、全部のパスが最後に書いた値で描かれる。
//! そこでカメラ uniform・インスタンス・描画先をビューごとに持たせる。メッシュ・パイプライン・
//! ライト・線・点は共有なので、どのビューにも同じ物が（同じフレームで）写る。
//!
//! 図面のペイン向けに、追加のビューは影・空・ポストプロセスを使わない。影の深度は既定のビューの
//! カメラに合わせて焼いてあるので、別のカメラから引くと合わない（シェーダへは
//! `resolution.w` で伝える）。色はシェーダが tonemap と gamma までかけた値で、既定のビューの
//! ポストプロセスなしの経路と同じ出方になる。

use super::*;

/// 追加のビューの描き方。
#[derive(Clone, Debug)]
pub struct RenderViewOptions {
    /// 背景色
    pub clear_color: wgpu::Color,
    /// 静的線（`update_lines` と線チャンク）を深さ判定ありで描くか。プレビュー線は常に手前
    pub lines_depth_tested: bool,
    /// クリップボックス（min, max）。None = 切らない
    pub clip_box: Option<([f32; 3], [f32; 3])>,
    /// 描く線チャンクの id の範囲。None = 全部（平面と 3D で線の組を分けるときに使う）
    pub line_chunks: Option<std::ops::Range<u32>>,
    /// `update_lines` の線を描くか
    pub static_lines: bool,
    /// プレビュー線（作図中のラバーバンド等）を描くか
    pub preview_lines: bool,
    /// 点（`update_points`）を描くか
    pub points: bool,
    /// Gaussian Splat を描くか
    pub splats: bool,
}

impl Default for RenderViewOptions {
    fn default() -> Self {
        Self {
            clear_color: wgpu::Color::BLACK,
            lines_depth_tested: false,
            clip_box: None,
            line_chunks: None,
            static_lines: true,
            preview_lines: true,
            points: true,
            splats: true,
        }
    }
}

/// 追加の描画ビュー。[`Renderer::create_view`] で作り、[`Renderer::render_view`] で描く。
///
/// 描き上がった色は [`RenderView::color_view`]（単一サンプル・`TEXTURE_BINDING` 付き）に残るので、
/// 呼び出し側の合成パスで画面の好きな位置へ貼れる。GPU から読み戻さない。
pub struct RenderView {
    width: u32,
    height: u32,
    /// 描画先の形式とサンプル数。作った時点の Renderer の品質設定に合わせてある
    /// （パイプラインと食い違うと validation で落ちる）。品質が変わったら `render_view` が作り直す。
    format: wgpu::TextureFormat,
    samples: u32,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    targets: ViewTargets,
}

struct ViewTargets {
    color_texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    #[allow(dead_code)] // view の裏付け（drop されるとアタッチメントが無効になる）
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// MSAA 有効時だけ。ここへ描いて `color_view` へ resolve する
    msaa: Option<(wgpu::Texture, wgpu::TextureView)>,
}

impl RenderView {
    /// 描画先の寸法（px）
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// 描き上がった色（単一サンプル・`TEXTURE_BINDING | COPY_SRC` 付き）
    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.targets.color_view
    }

    pub fn color_texture(&self) -> &wgpu::Texture {
        &self.targets.color_texture
    }

    /// 色の形式。ポストプロセスなしなら surface と同じ、ありなら `Rgba16Float`
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }
}

impl Renderer {
    /// 追加の描画ビューを作る。寸法は 1px 未満にならないよう丸める。
    pub fn create_view(&self, width: u32, height: u32) -> RenderView {
        let (width, height) = (width.max(1), height.max(1));
        let camera_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("View Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // 既定のビューと同じレイアウトなので、パイプラインはそのまま使える
        let camera_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("View Camera Bind Group"),
            layout: &self.camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
        });
        let instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("View Instance Buffer"),
            size: std::mem::size_of::<InstanceData>() as u64 * 64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (format, samples) = (self.scene_render_format(), self.scene_sample_count());
        RenderView {
            width,
            height,
            format,
            samples,
            camera_buffer,
            camera_bind_group,
            instance_buffer,
            targets: Self::view_targets(&self.device, width, height, format, samples),
        }
    }

    /// ビューの寸法を変える。同じ寸法なら何もしない。
    pub fn resize_view(&self, view: &mut RenderView, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (view.width, view.height) {
            return;
        }
        view.width = width;
        view.height = height;
        view.targets = Self::view_targets(&self.device, width, height, view.format, view.samples);
    }

    fn view_targets(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        samples: u32,
    ) -> ViewTargets {
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("View Color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let (depth_texture, depth_view) = Self::create_depth_texture_impl(device, width, height, samples);
        let msaa = (samples > 1).then(|| Self::create_msaa_texture_impl(device, width, height, format, samples));
        ViewTargets { color_texture, color_view, depth_texture, depth_view, msaa }
    }

    /// 追加のビューに描く。`encoder` は呼び出し側が submit する。
    ///
    /// 既定のビュー（`render_to_view`）や他の追加ビューと**同じ submit の中で**描いてよい
    /// （カメラとインスタンスはビューごとのバッファに書く）。影の深度パスは走らせず、
    /// `cull_stats` も書き換えない（既定のビューの計測値のまま）。
    pub fn render_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &mut RenderView,
        camera: &Camera,
        instances: &[(String, InstanceData)],
        opaque_count: usize,
        options: &RenderViewOptions,
    ) {
        // 品質の切り替え（MSAA・ポストプロセス）で形式やサンプル数が変わったら作り直す
        let (format, samples) = (self.scene_render_format(), self.scene_sample_count());
        if (format, samples) != (view.format, view.samples) {
            view.format = format;
            view.samples = samples;
            view.targets = Self::view_targets(&self.device, view.width, view.height, format, samples);
        }

        let clip = match options.clip_box {
            Some((min, max)) => ([min[0], min[1], min[2], 1.0], [max[0], max[1], max[2], 0.0]),
            None => ([0.0; 4], [0.0; 4]),
        };
        // HDR なし（シェーダが tonemap/gamma までかける）・影なし
        let uniform = self.camera_uniform(camera, clip, [view.width as f32, view.height as f32, 0.0, 1.0]);
        self.queue.write_buffer(&view.camera_buffer, 0, bytemuck::bytes_of(&uniform));
        Self::write_instances(&self.device, &self.queue, &mut view.instance_buffer, instances);

        let vis = self.visibility_unrecorded(
            instances,
            glam::Mat4::from_cols_array_2d(&crate::camera::dmat4_to_f32(camera.view_projection_matrix())),
        );
        let view: &RenderView = view;
        let b = ViewBinding {
            camera_bind_group: &view.camera_bind_group,
            instance_buffer: &view.instance_buffer,
            lines_depth_tested: options.lines_depth_tested,
            line_chunks: options.line_chunks.clone(),
            static_lines: options.static_lines,
            preview_lines: options.preview_lines,
            points: options.points,
            splats: options.splats,
        };
        let t = &view.targets;
        let (target, resolve) = match &t.msaa {
            Some((_, msaa_view)) => (msaa_view, Some(&t.color_view)),
            None => (&t.color_view, None),
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("View Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: resolve,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(options.clear_color), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &t.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        let instance_size = std::mem::size_of::<InstanceData>() as u64;
        self.draw_meshes(&mut pass, &b, instances, opaque_count, instance_size, &vis);
        self.draw_splats_points_lines(&mut pass, &b);
    }

    /// ビューの色を RGBA8 で読み戻す（試験・書き出し用。毎フレームの表示には使わない）。
    ///
    /// 画面に貼ったときと同じ値になるように、`Rgba16Float` のビューを sRGB の surface へ貼る構成なら
    /// sRGB に符号化してから返す（sRGB 形式のビューは格納値そのもの＝`render_offscreen` と同じ）。
    pub fn read_view_rgba(&self, view: &RenderView) -> Result<Vec<u8>, RendererError> {
        let (width, height) = (view.width, view.height);
        let bpp: u32 = match view.format {
            wgpu::TextureFormat::Rgba16Float => 8,
            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb => 4,
            other => {
                return Err(RendererError::Rendering(format!("read_view_rgba: 未対応の形式 {other:?}")));
            }
        };
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (width * bpp).div_ceil(align) * align;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("View Readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("View Readback Encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &view.targets.color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| RendererError::Rendering(format!("Buffer map recv error: {e}")))?
            .map_err(|e| RendererError::Rendering(format!("Buffer map error: {e}")))?;

        let data = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        let encode_srgb = self.surface_format.is_srgb();
        for row in 0..height {
            let start = (row * padded_row) as usize;
            let row_data = &data[start..start + (width * bpp) as usize];
            match view.format {
                wgpu::TextureFormat::Rgba16Float => {
                    for px in row_data.chunks_exact(8) {
                        for c in 0..4 {
                            let v = half::f16::from_le_bytes([px[c * 2], px[c * 2 + 1]]).to_f32().clamp(0.0, 1.0);
                            let v = if encode_srgb && c < 3 { linear_to_srgb(v) } else { v };
                            rgba.push((v * 255.0).round() as u8);
                        }
                    }
                }
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                    for px in row_data.chunks_exact(4) {
                        rgba.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                    }
                }
                _ => rgba.extend_from_slice(row_data),
            }
        }
        drop(data);
        buffer.unmap();
        Ok(rgba)
    }

    /// インスタンスを `buffer` へ書く（足りなければ作り直す）。既定のビューと追加のビューで共用。
    pub(super) fn write_instances(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &mut wgpu::Buffer,
        instances: &[(String, InstanceData)],
    ) {
        if instances.is_empty() {
            return;
        }
        let mut all_data: Vec<u8> = Vec::with_capacity(instances.len() * std::mem::size_of::<InstanceData>());
        for (_, inst_data) in instances {
            all_data.extend_from_slice(bytemuck::bytes_of(inst_data));
        }
        let required = all_data.len() as u64;
        if required > buffer.size() {
            *buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Instance Buffer"),
                size: required,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(buffer, 0, &all_data);
    }
}

/// 線形 → sRGB の符号化（0..1）
fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}
