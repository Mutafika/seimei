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

// ── Simple hash-based 3D noise for mesh displacement ──

fn hash3(p: [f32; 3]) -> f32 {
    let h = p[0] * 127.1 + p[1] * 311.7 + p[2] * 74.7;
    (h.sin() * 43758.5453).fract()
}

fn smooth_noise_3d(x: f32, y: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let fz = z - z.floor();
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let uz = fz * fz * (3.0 - 2.0 * fz);

    let v = |dx: i32, dy: i32, dz: i32| -> f32 {
        hash3([(ix + dx) as f32, (iy + dy) as f32, (iz + dz) as f32])
    };

    let x0 = v(0,0,0) * (1.0 - ux) + v(1,0,0) * ux;
    let x1 = v(0,1,0) * (1.0 - ux) + v(1,1,0) * ux;
    let x2 = v(0,0,1) * (1.0 - ux) + v(1,0,1) * ux;
    let x3 = v(0,1,1) * (1.0 - ux) + v(1,1,1) * ux;
    let y0 = x0 * (1.0 - uy) + x1 * uy;
    let y1 = x2 * (1.0 - uy) + x3 * uy;
    y0 * (1.0 - uz) + y1 * uz
}

fn fbm_3d(x: f32, y: f32, z: f32, octaves: u32) -> f32 {
    let mut val = 0.0;
    let mut amp = 0.5;
    let (mut px, mut py, mut pz) = (x, y, z);
    for _ in 0..octaves {
        val += smooth_noise_3d(px, py, pz) * amp;
        amp *= 0.5;
        px *= 2.0;
        py *= 2.0;
        pz *= 2.0;
    }
    val
}

/// 不規則な岩メッシュ — icosphereの頂点をFBMノイズでずらす
pub fn rock(base_radius: f32, subdivisions: u32, roughness: f32, seed: f32) -> RenderMesh {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;

    let mut positions: Vec<[f32; 3]> = vec![
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ];
    for p in &mut positions {
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        p[0] /= len; p[1] /= len; p[2] /= len;
    }

    let mut indices: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

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

    // Displace vertices — angular, rocky shapes (not smooth blobs)
    for p in &mut positions {
        let nx = p[0]; let ny = p[1]; let nz = p[2];

        // Large-scale angular shape: few big flat faces
        let big = smooth_noise_3d(nx * 1.2 + seed, ny * 1.2 + seed * 0.7, nz * 1.2 + seed * 1.3);
        // Quantize to create flat facets
        let faceted = (big * 4.0).round() / 4.0;

        // Ridge noise: sharp creases between flat areas
        let ridge1 = 1.0 - (smooth_noise_3d(nx * 2.5 + seed + 5.0, ny * 2.5 + 3.0, nz * 2.5 + 7.0) * 2.0 - 1.0).abs();
        let ridge2 = 1.0 - (smooth_noise_3d(nx * 5.0 + seed + 11.0, ny * 5.0 + 8.0, nz * 5.0 + 2.0) * 2.0 - 1.0).abs();

        // Small chips and dents
        let chip = smooth_noise_3d(nx * 8.0 + seed + 20.0, ny * 8.0 + 15.0, nz * 8.0 + 30.0);
        let dent = (chip * 3.0).fract().min(1.0).max(0.0);

        let total = (faceted * 0.5 - 0.25
            + ridge1 * 0.15
            + ridge2 * 0.05
            - dent * 0.08) * roughness;

        let r = base_radius * (1.0 + total);
        p[0] = nx * r; p[1] = ny * r; p[2] = nz * r;
    }

    // Recalculate normals from geometry
    let mut normals = vec![[0.0f32; 3]; positions.len()];
    for tri in &indices {
        let p0 = positions[tri[0] as usize];
        let p1 = positions[tri[1] as usize];
        let p2 = positions[tri[2] as usize];
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let face_n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for &idx in tri {
            let n = &mut normals[idx as usize];
            n[0] += face_n[0]; n[1] += face_n[1]; n[2] += face_n[2];
        }
    }
    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-8 { n[0] /= len; n[1] /= len; n[2] /= len; }
    }

    let mut mesh = RenderMesh::new();
    for (i, p) in positions.iter().enumerate() {
        let nm = normals[i];
        let u = 0.5 + p[2].atan2(p[0]) / (2.0 * PI);
        let v = 0.5 - (p[1] / base_radius).clamp(-1.0, 1.0).asin() / PI;
        mesh.add_vertex(Vertex::with_uv(
            Point3::new_f32(p[0], p[1], p[2]),
            Vec3D::new_f32(nm[0], nm[1], nm[2]),
            [u, v],
        ));
    }
    for tri in &indices {
        mesh.add_triangle(tri[0], tri[1], tri[2]);
    }

    mesh
}

/// 地形メッシュ — ノイズベースの起伏 + 中央に池の窪み
pub fn terrain(size: f32, resolution: u32, height_scale: f32) -> RenderMesh {
    let mut mesh = RenderMesh::new();
    let res = resolution as usize;
    let half = size * 0.5;
    let step = size / resolution as f32;

    let mut heights = vec![vec![0.0f32; res + 1]; res + 1];
    for iz in 0..=res {
        for ix in 0..=res {
            let x = ix as f32 / res as f32 * 2.0 - 1.0;
            let z = iz as f32 / res as f32 * 2.0 - 1.0;

            let wx = x * 3.0;
            let wz = z * 3.0;
            let h1 = smooth_noise_3d(wx * 0.5 + 5.0, 0.0, wz * 0.5 + 3.0);
            let h2 = smooth_noise_3d(wx * 1.0 + 2.0, 0.5, wz * 1.0 + 7.0) * 0.5;
            let h3 = smooth_noise_3d(wx * 2.0 + 8.0, 1.0, wz * 2.0 + 1.0) * 0.25;
            let h4 = smooth_noise_3d(wx * 4.0 + 3.0, 1.5, wz * 4.0 + 9.0) * 0.12;
            let terrain_h = (h1 + h2 + h3 + h4 - 0.4) * height_scale;

            // Pond basin
            let dx = x - 0.1;
            let dz = z - 0.05;
            let pond_dist = (dx * dx + dz * dz).sqrt();
            let pond_radius = 0.3;
            let pond = if pond_dist < pond_radius {
                let t = 1.0 - pond_dist / pond_radius;
                -0.4 * height_scale * t * t * (3.0 - 2.0 * t)
            } else {
                0.0
            };

            let edge_dist = x.abs().max(z.abs());
            let edge = 1.0 - ((edge_dist - 0.7) / 0.3).clamp(0.0, 1.0);
            heights[iz][ix] = (terrain_h + pond) * edge;
        }
    }

    for iz in 0..=res {
        for ix in 0..=res {
            let x = -half + ix as f32 * step;
            let z = -half + iz as f32 * step;
            let y = heights[iz][ix];

            let hx_l = if ix > 0 { heights[iz][ix - 1] } else { y };
            let hx_r = if ix < res { heights[iz][ix + 1] } else { y };
            let hz_b = if iz > 0 { heights[iz - 1][ix] } else { y };
            let hz_f = if iz < res { heights[iz + 1][ix] } else { y };
            let nx = (hx_l - hx_r) / (2.0 * step);
            let nz = (hz_b - hz_f) / (2.0 * step);
            let len = (nx * nx + 1.0 + nz * nz).sqrt();

            mesh.add_vertex(Vertex::with_uv(
                Point3::new_f32(x, y, z),
                Vec3D::new_f32(nx / len, 1.0 / len, nz / len),
                [ix as f32 / res as f32, iz as f32 / res as f32],
            ));
        }
    }

    let w = (res + 1) as u32;
    for iz in 0..res as u32 {
        for ix in 0..res as u32 {
            let i = iz * w + ix;
            mesh.add_triangle(i, i + w, i + 1);
            mesh.add_triangle(i + 1, i + w, i + w + 1);
        }
    }

    mesh
}

/// 水面メッシュ — 指定した高さの平面
pub fn water_plane(size: f32, y_level: f32, resolution: u32) -> RenderMesh {
    let mut mesh = RenderMesh::new();
    let res = resolution as usize;
    let half = size * 0.5;
    let step = size / resolution as f32;

    for iz in 0..=res {
        for ix in 0..=res {
            let x = -half + ix as f32 * step;
            let z = -half + iz as f32 * step;
            mesh.add_vertex(Vertex::with_uv(
                Point3::new_f32(x, y_level, z),
                Vec3D::new_f32(0.0, 1.0, 0.0),
                [ix as f32 / res as f32, iz as f32 / res as f32],
            ));
        }
    }

    let w = (res + 1) as u32;
    for iz in 0..res as u32 {
        for ix in 0..res as u32 {
            let i = iz * w + ix;
            mesh.add_triangle(i, i + w, i + 1);
            mesh.add_triangle(i + 1, i + w, i + w + 1);
        }
    }

    mesh
}
