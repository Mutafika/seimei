//! プロシージャルメッシュ生成
//!
//! icosphere, cube, torus, plane, cylinder をコードで生成する。

use crate::math::{Point3, Vec3D};
use crate::mesh::{RenderMesh, Vertex};
use std::f32::consts::PI;

/// UVスフィア — 緯度経度で分割した球体
pub fn sphere(radius: f32, segments: u32, rings: u32) -> RenderMesh {
    let mut mesh = RenderMesh::new();

    for j in 0..=rings {
        let v = j as f32 / rings as f32;
        let phi = v * PI;
        let sp = phi.sin();
        let cp = phi.cos();

        for i in 0..=segments {
            let u = i as f32 / segments as f32;
            let theta = u * 2.0 * PI;
            let st = theta.sin();
            let ct = theta.cos();

            let nx = st * sp;
            let ny = cp;
            let nz = ct * sp;

            mesh.add_vertex(Vertex::with_uv(
                Point3::new_f32(nx * radius, ny * radius, nz * radius),
                Vec3D::new_f32(nx, ny, nz),
                [u, v],
            ));
        }
    }

    for j in 0..rings {
        for i in 0..segments {
            let a = j * (segments + 1) + i;
            let b = a + 1;
            let c = (j + 1) * (segments + 1) + i;
            let d = c + 1;

            if j != 0 {
                mesh.add_triangle(a, c, b);
            }
            if j != rings - 1 {
                mesh.add_triangle(b, c, d);
            }
        }
    }

    mesh
}

/// Icosphere — 正二十面体を再帰分割した球体
pub fn icosphere(radius: f32, subdivisions: u32) -> RenderMesh {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;

    let mut positions: Vec<[f32; 3]> = vec![
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ];

    // Normalize to unit sphere
    for p in &mut positions {
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        p[0] /= len;
        p[1] /= len;
        p[2] /= len;
    }

    let mut indices: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

    // Subdivide
    for _ in 0..subdivisions {
        let mut new_indices = Vec::new();
        let mut midpoint_cache = std::collections::HashMap::new();

        for tri in &indices {
            let a = get_midpoint(&mut positions, &mut midpoint_cache, tri[0], tri[1]);
            let b = get_midpoint(&mut positions, &mut midpoint_cache, tri[1], tri[2]);
            let c = get_midpoint(&mut positions, &mut midpoint_cache, tri[2], tri[0]);

            new_indices.push([tri[0], a, c]);
            new_indices.push([tri[1], b, a]);
            new_indices.push([tri[2], c, b]);
            new_indices.push([a, b, c]);
        }
        indices = new_indices;
    }

    let mut mesh = RenderMesh::new();
    for p in &positions {
        let n = Vec3D::new_f32(p[0], p[1], p[2]);
        // Spherical UV
        let u = 0.5 + p[2].atan2(p[0]) / (2.0 * PI);
        let v = 0.5 - p[1].asin() / PI;
        mesh.add_vertex(Vertex::with_uv(
            Point3::new_f32(p[0] * radius, p[1] * radius, p[2] * radius),
            n,
            [u, v],
        ));
    }
    for tri in &indices {
        mesh.add_triangle(tri[0], tri[1], tri[2]);
    }

    mesh
}

fn get_midpoint(
    positions: &mut Vec<[f32; 3]>,
    cache: &mut std::collections::HashMap<(u32, u32), u32>,
    a: u32, b: u32,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let pa = positions[a as usize];
    let pb = positions[b as usize];
    let mut mid = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5];
    let len = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
    mid[0] /= len;
    mid[1] /= len;
    mid[2] /= len;
    let idx = positions.len() as u32;
    positions.push(mid);
    cache.insert(key, idx);
    idx
}

/// 立方体
pub fn cube(size: f32) -> RenderMesh {
    let h = size * 0.5;
    let mut mesh = RenderMesh::new();

    // 6面、各面4頂点
    let faces: &[([f32; 3], [[f32; 3]; 4], [[f32; 2]; 4])] = &[
        // normal, positions, uvs
        ([0.0, 0.0, 1.0], [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]], [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]]),      // front
        ([0.0, 0.0, -1.0], [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]], [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]]),  // back
        ([0.0, 1.0, 0.0], [[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]], [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]]),       // top
        ([0.0, -1.0, 0.0], [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]], [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]]),  // bottom
        ([1.0, 0.0, 0.0], [[h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h]], [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]]),       // right
        ([-1.0, 0.0, 0.0], [[-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h]], [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]]),  // left
    ];

    for (normal, positions, uvs) in faces {
        let n = Vec3D::new_f32(normal[0], normal[1], normal[2]);
        let base = mesh.vertices.len() as u32;
        for (i, pos) in positions.iter().enumerate() {
            mesh.add_vertex(Vertex::with_uv(
                Point3::new_f32(pos[0], pos[1], pos[2]),
                n,
                uvs[i],
            ));
        }
        mesh.add_quad(base, base + 1, base + 2, base + 3);
    }

    mesh
}

/// トーラス（ドーナツ）
pub fn torus(major_radius: f32, minor_radius: f32, major_segments: u32, minor_segments: u32) -> RenderMesh {
    let mut mesh = RenderMesh::new();

    for j in 0..=major_segments {
        let u = j as f32 / major_segments as f32;
        let theta = u * 2.0 * PI;
        let ct = theta.cos();
        let st = theta.sin();

        for i in 0..=minor_segments {
            let v = i as f32 / minor_segments as f32;
            let phi = v * 2.0 * PI;
            let cp = phi.cos();
            let sp = phi.sin();

            let x = (major_radius + minor_radius * cp) * ct;
            let y = minor_radius * sp;
            let z = (major_radius + minor_radius * cp) * st;

            let nx = cp * ct;
            let ny = sp;
            let nz = cp * st;

            mesh.add_vertex(Vertex::with_uv(
                Point3::new_f32(x, y, z),
                Vec3D::new_f32(nx, ny, nz),
                [u, v],
            ));
        }
    }

    for j in 0..major_segments {
        for i in 0..minor_segments {
            let a = j * (minor_segments + 1) + i;
            let b = a + 1;
            let c = (j + 1) * (minor_segments + 1) + i;
            let d = c + 1;
            mesh.add_triangle(a, c, b);
            mesh.add_triangle(b, c, d);
        }
    }

    mesh
}

/// 平面
pub fn plane(width: f32, depth: f32, segments_x: u32, segments_z: u32) -> RenderMesh {
    let mut mesh = RenderMesh::new();
    let hw = width * 0.5;
    let hd = depth * 0.5;

    for j in 0..=segments_z {
        let v = j as f32 / segments_z as f32;
        let z = -hd + v * depth;
        for i in 0..=segments_x {
            let u = i as f32 / segments_x as f32;
            let x = -hw + u * width;
            mesh.add_vertex(Vertex::with_uv(
                Point3::new_f32(x, 0.0, z),
                Vec3D::new_f32(0.0, 1.0, 0.0),
                [u, v],
            ));
        }
    }

    for j in 0..segments_z {
        for i in 0..segments_x {
            let a = j * (segments_x + 1) + i;
            let b = a + 1;
            let c = (j + 1) * (segments_x + 1) + i;
            let d = c + 1;
            mesh.add_triangle(a, c, b);
            mesh.add_triangle(b, c, d);
        }
    }

    mesh
}

/// 円柱
pub fn cylinder(radius: f32, height: f32, segments: u32) -> RenderMesh {
    let mut mesh = RenderMesh::new();
    let hh = height * 0.5;

    // Side
    for i in 0..=segments {
        let u = i as f32 / segments as f32;
        let theta = u * 2.0 * PI;
        let ct = theta.cos();
        let st = theta.sin();

        // Bottom
        mesh.add_vertex(Vertex::with_uv(
            Point3::new_f32(ct * radius, -hh, st * radius),
            Vec3D::new_f32(ct, 0.0, st),
            [u, 1.0],
        ));
        // Top
        mesh.add_vertex(Vertex::with_uv(
            Point3::new_f32(ct * radius, hh, st * radius),
            Vec3D::new_f32(ct, 0.0, st),
            [u, 0.0],
        ));
    }

    for i in 0..segments {
        let a = i * 2;
        let b = a + 1;
        let c = a + 2;
        let d = a + 3;
        mesh.add_triangle(a, c, b);
        mesh.add_triangle(b, c, d);
    }

    // Top cap
    let top_center = mesh.add_vertex(Vertex::with_uv(
        Point3::new_f32(0.0, hh, 0.0),
        Vec3D::new_f32(0.0, 1.0, 0.0),
        [0.5, 0.5],
    ));
    let base = mesh.vertices.len() as u32;
    for i in 0..=segments {
        let theta = (i as f32 / segments as f32) * 2.0 * PI;
        let ct = theta.cos();
        let st = theta.sin();
        mesh.add_vertex(Vertex::with_uv(
            Point3::new_f32(ct * radius, hh, st * radius),
            Vec3D::new_f32(0.0, 1.0, 0.0),
            [ct * 0.5 + 0.5, st * 0.5 + 0.5],
        ));
    }
    for i in 0..segments {
        mesh.add_triangle(top_center, base + i, base + i + 1);
    }

    // Bottom cap
    let bot_center = mesh.add_vertex(Vertex::with_uv(
        Point3::new_f32(0.0, -hh, 0.0),
        Vec3D::new_f32(0.0, -1.0, 0.0),
        [0.5, 0.5],
    ));
    let base2 = mesh.vertices.len() as u32;
    for i in 0..=segments {
        let theta = (i as f32 / segments as f32) * 2.0 * PI;
        let ct = theta.cos();
        let st = theta.sin();
        mesh.add_vertex(Vertex::with_uv(
            Point3::new_f32(ct * radius, -hh, st * radius),
            Vec3D::new_f32(0.0, -1.0, 0.0),
            [ct * 0.5 + 0.5, st * 0.5 + 0.5],
        ));
    }
    for i in 0..segments {
        mesh.add_triangle(bot_center, base2 + i + 1, base2 + i);
    }

    mesh
}
