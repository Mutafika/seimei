//! メッシュデータ構造

use crate::math::{BoundingBox, Point3, Vec3D};
use serde::{Deserialize, Serialize};

/// 頂点データ（CPU側）
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vertex {
    pub position: Point3,
    pub normal: Vec3D,
    pub uv: [f32; 2],
}

impl Vertex {
    pub fn new(position: Point3, normal: Vec3D) -> Self {
        Self {
            position,
            normal,
            uv: [0.0, 0.0],
        }
    }

    pub fn with_uv(position: Point3, normal: Vec3D, uv: [f32; 2]) -> Self {
        Self { position, normal, uv }
    }
}

/// 三角形メッシュ
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl RenderMesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn from_data(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    pub fn add_vertex(&mut self, vertex: Vertex) -> u32 {
        let index = self.vertices.len() as u32;
        self.vertices.push(vertex);
        index
    }

    pub fn add_triangle(&mut self, i0: u32, i1: u32, i2: u32) {
        self.indices.push(i0);
        self.indices.push(i1);
        self.indices.push(i2);
    }

    pub fn add_quad(&mut self, i0: u32, i1: u32, i2: u32, i3: u32) {
        self.add_triangle(i0, i1, i2);
        self.add_triangle(i0, i2, i3);
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn bounding_box(&self) -> BoundingBox {
        let mut bb = BoundingBox::empty();
        for vertex in &self.vertices {
            bb.extend_point(&vertex.position);
        }
        bb
    }

    pub fn recalculate_normals(&mut self) {
        for vertex in &mut self.vertices {
            vertex.normal = Vec3D::zero();
        }

        for chunk in self.indices.chunks(3) {
            if chunk.len() < 3 {
                continue;
            }
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;

            let p0 = &self.vertices[i0].position;
            let p1 = &self.vertices[i1].position;
            let p2 = &self.vertices[i2].position;

            let e1 = Vec3D::new(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
            let e2 = Vec3D::new(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
            let normal = e1.cross(&e2);

            for &idx in &[i0, i1, i2] {
                let n = &mut self.vertices[idx].normal;
                *n = Vec3D::new(n.x + normal.x, n.y + normal.y, n.z + normal.z);
            }
        }

        for vertex in &mut self.vertices {
            vertex.normal = vertex.normal.normalize();
        }
    }

    pub fn merge(&mut self, other: &RenderMesh) {
        let offset = self.vertices.len() as u32;
        self.vertices.extend(other.vertices.iter().cloned());
        self.indices.extend(other.indices.iter().map(|i| i + offset));
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// GPU用頂点配列を取得
    pub fn gpu_vertices(&self) -> Vec<crate::vertex::GpuVertex> {
        self.vertices
            .iter()
            .map(|v| crate::vertex::GpuVertex {
                position: [v.position.x as f32, v.position.y as f32, v.position.z as f32],
                normal: [v.normal.x as f32, v.normal.y as f32, v.normal.z as f32],
                uv: v.uv,
                tangent: [0.0; 4],
                vertex_color: [1.0, 1.0, 1.0, 1.0],
            })
            .collect()
    }
}
