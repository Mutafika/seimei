//! GPU 頂点データ定義

use bytemuck::{Pod, Zeroable};

/// GPU頂点データ
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// タンジェント (xyz=tangent direction, w=bitangent sign)
    pub tangent: [f32; 4],
    /// 頂点カラー (RGBA) — デフォルト白
    pub vertex_color: [f32; 4],
}

impl GpuVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// インスタンスデータ（要素ごとの変換・色・マテリアル）
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct InstanceData {
    /// モデル行列
    pub model: [[f32; 4]; 4],
    /// 色（RGBA）
    pub color: [f32; 4],
    /// マテリアル [metallic, roughness, sss_or_transmission, emissive]
    pub material: [f32; 4],
}

impl InstanceData {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[[f32; 4]; 4]>() + std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }

    pub fn identity(color: [f32; 4]) -> Self {
        Self {
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            color,
            material: [0.0, 0.5, 0.0, 0.0],
        }
    }

    pub fn identity_with_material(color: [f32; 4], metallic: f32, roughness: f32) -> Self {
        Self {
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            color,
            material: [metallic, roughness, 0.0, 0.0],
        }
    }

    pub fn from_transform(
        position: [f32; 3],
        rotation_deg: f32,
        scale: [f32; 3],
        color: [f32; 4],
    ) -> Self {
        let r = rotation_deg.to_radians();
        let (sin_r, cos_r) = r.sin_cos();
        let [sx, sy, sz] = scale;
        let [tx, ty, tz] = position;
        Self {
            model: [
                [sx * cos_r,  sx * sin_r,  0.0, 0.0],
                [-sy * sin_r, sy * cos_r,  0.0, 0.0],
                [0.0,         0.0,         sz,  0.0],
                [tx,          ty,          tz,  1.0],
            ],
            color,
            material: [0.0, 0.5, 0.0, 0.0],
        }
    }

    pub fn from_transform_with_material(
        position: [f32; 3],
        rotation_deg: f32,
        scale: [f32; 3],
        color: [f32; 4],
        metallic: f32,
        roughness: f32,
    ) -> Self {
        let mut inst = Self::from_transform(position, rotation_deg, scale, color);
        inst.material = [metallic, roughness, 0.0, 0.0];
        inst
    }
}

/// 線分用GPU頂点データ
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl LineVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}
