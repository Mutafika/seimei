//! Gaussian Splatting レンダリング

use crate::ply::GaussianCloud;
use bytemuck::{Pod, Zeroable};

/// GPUに送る1スプラットのデータ（64 bytes）
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuSplat {
    pub position: [f32; 4],
    pub scale_opacity: [f32; 4],
    pub rotation: [f32; 4],
    pub color: [f32; 4],
}

/// Splatクラウドのデータ管理
pub struct SplatCloudData {
    pub splats: Vec<GpuSplat>,
    pub sorted_indices: Vec<u32>,
    pub count: u32,
}

impl SplatCloudData {
    pub fn from_cloud(cloud: &GaussianCloud) -> Self {
        let splats: Vec<GpuSplat> = cloud
            .points
            .iter()
            .map(|p| GpuSplat {
                position: [p.position.x as f32, p.position.y as f32, p.position.z as f32, 0.0],
                scale_opacity: [p.scale[0], p.scale[1], p.scale[2], p.opacity],
                rotation: p.rotation,
                color: [p.color[0], p.color[1], p.color[2], p.opacity],
            })
            .collect();

        let count = splats.len() as u32;
        let sorted_indices: Vec<u32> = (0..count).collect();

        Self { splats, sorted_indices, count }
    }

    /// カメラ位置でback-to-frontソート
    pub fn sort_by_depth(&mut self, camera_pos: [f32; 3]) {
        let splats = &self.splats;
        self.sorted_indices.sort_unstable_by(|&a, &b| {
            let pa = &splats[a as usize].position;
            let pb = &splats[b as usize].position;
            let da = (pa[0] - camera_pos[0]).powi(2)
                + (pa[1] - camera_pos[1]).powi(2)
                + (pa[2] - camera_pos[2]).powi(2);
            let db = (pb[0] - camera_pos[0]).powi(2)
                + (pb[1] - camera_pos[1]).powi(2)
                + (pb[2] - camera_pos[2]).powi(2);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

/// Splatレンダリング用WGSLシェーダー
pub const SPLAT_SHADER_SOURCE: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec4<f32>,
    clip_min: vec4<f32>,
    clip_max: vec4<f32>,
    resolution: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct SplatData {
    position: vec4<f32>,
    scale_opacity: vec4<f32>,
    rotation: vec4<f32>,
    color: vec4<f32>,
};

@group(1) @binding(0) var<storage, read> splats: array<SplatData>;
@group(1) @binding(1) var<storage, read> sorted_indices: array<u32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) cov2d_a: vec3<f32>,
};

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    let x2 = x + x; let y2 = y + y; let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;
    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy)),
    );
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let splat_idx = sorted_indices[instance_index];
    let splat = splats[splat_idx];

    let pos3 = splat.position.xyz;
    let scale = splat.scale_opacity.xyz;
    let opacity = splat.scale_opacity.w;
    let color = splat.color;

    let quad_uv = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    let uv = quad_uv[vertex_index];
    out.uv = uv;

    let R = quat_to_mat3(splat.rotation);
    let S = mat3x3<f32>(
        vec3<f32>(scale.x, 0.0, 0.0),
        vec3<f32>(0.0, scale.y, 0.0),
        vec3<f32>(0.0, 0.0, scale.z),
    );
    let M = R * S;
    let Sigma = M * transpose(M);

    let view_pos = camera.view * vec4<f32>(pos3, 1.0);
    let t = view_pos.xyz;
    let focal_x = camera.view_proj[0][0] * 0.5;
    let focal_y = camera.view_proj[1][1] * 0.5;
    let tz = max(t.z, 0.001);
    let tz2 = tz * tz;

    let J = mat3x3<f32>(
        vec3<f32>(focal_x / tz, 0.0, 0.0),
        vec3<f32>(0.0, focal_y / tz, 0.0),
        vec3<f32>(-focal_x * t.x / tz2, -focal_y * t.y / tz2, 0.0),
    );
    let W = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );
    let T = J * W;
    let cov2d = T * Sigma * transpose(T);

    let a = cov2d[0][0] + 0.3;
    let b = cov2d[0][1];
    let c = cov2d[1][1] + 0.3;

    let det = a * c - b * b;
    let det_safe = max(det, 0.0001);
    let trace = a + c;
    let disc = max(trace * trace * 0.25 - det_safe, 0.0);
    let sqrt_disc = sqrt(disc);
    let lambda1 = trace * 0.5 + sqrt_disc;
    let lambda2 = max(trace * 0.5 - sqrt_disc, 0.1);
    let radius = 3.0 * sqrt(max(lambda1, lambda2));

    let clip_pos = camera.view_proj * vec4<f32>(pos3, 1.0);
    let ndc = clip_pos.xy / clip_pos.w;

    let inv_det = 1.0 / det_safe;
    out.cov2d_a = vec3<f32>(c * inv_det, -b * inv_det, a * inv_det);

    let viewport_scale = vec2<f32>(1.0 / camera.view_proj[0][0], 1.0 / camera.view_proj[1][1]);
    let offset = uv * radius * viewport_scale * 2.0;
    out.clip_position = vec4<f32>((ndc + offset) * clip_pos.w, clip_pos.z, clip_pos.w);

    out.color = vec4<f32>(color.rgb, opacity);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let u = in.uv;
    let power = -0.5 * (in.cov2d_a.x * u.x * u.x + 2.0 * in.cov2d_a.y * u.x * u.y + in.cov2d_a.z * u.y * u.y);

    if (power > 0.0) { discard; }
    let alpha = min(in.color.a * exp(power), 0.99);
    if (alpha < 1.0 / 255.0) { discard; }

    return vec4<f32>(in.color.rgb * alpha, alpha);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_splat_size() {
        assert_eq!(std::mem::size_of::<GpuSplat>(), 64);
    }
}
