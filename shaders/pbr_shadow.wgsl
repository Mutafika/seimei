// === Group 0: Camera ===
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec4<f32>,
    clip_min: vec4<f32>,
    clip_max: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// === Group 1: Lights (header uniform + storage buffer) ===
struct GpuLight {
    direction_or_position_and_type: vec4<f32>,
    color_and_intensity: vec4<f32>,
    extra: vec4<f32>,
};

struct LightHeader {
    ambient_and_count: vec4<f32>,
    mode_flags: vec4<f32>,
};

@group(1) @binding(0)
var<uniform> light_header: LightHeader;
@group(1) @binding(1)
var<storage, read> lights: array<GpuLight>;

// === Group 2: Texture ===
@group(2) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(2) @binding(1)
var s_diffuse: sampler;

// === Group 3: Shadow Map ===
@group(3) @binding(0)
var shadow_map: texture_depth_2d;
@group(3) @binding(1)
var shadow_sampler: sampler_comparison;
@group(3) @binding(2)
var<uniform> light_view_proj: mat4x4<f32>;

const PI: f32 = 3.14159265359;
const SHADOW_MAP_SIZE: f32 = 2048.0;
const SHADOW_BIAS: f32 = 0.005;

// === PBR Functions ===
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

// PCF shadow sampling
fn calculate_shadow(world_pos: vec3<f32>) -> f32 {
    let light_pos = light_view_proj * vec4<f32>(world_pos, 1.0);
    let proj_coords = light_pos.xyz / light_pos.w;
    let uv = proj_coords.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj_coords.z > 1.0) {
        return 1.0;
    }
    let depth = proj_coords.z - SHADOW_BIAS;
    var shadow: f32 = 0.0;
    let texel_size = 1.0 / SHADOW_MAP_SIZE;
    for (var x: i32 = -1; x <= 1; x++) {
        for (var y: i32 = -1; y <= 1; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
        }
    }
    return shadow / 9.0;
}

// === Vertex/Instance Input ===
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(9) tangent: vec4<f32>,
    @location(10) vertex_color: vec4<f32>,
};

struct InstanceInput {
    @location(3) model_matrix_0: vec4<f32>,
    @location(4) model_matrix_1: vec4<f32>,
    @location(5) model_matrix_2: vec4<f32>,
    @location(6) model_matrix_3: vec4<f32>,
    @location(7) color: vec4<f32>,
    @location(8) material: vec4<f32>,  // [metallic, roughness, sss_or_transmission, emissive]
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) material: vec4<f32>,
    @location(5) world_tangent: vec3<f32>,
    @location(6) world_bitangent: vec3<f32>,
};

// IOR -> F0
fn ior_to_f0(ior: f32) -> f32 {
    let r = (ior - 1.0) / (ior + 1.0);
    return r * r;
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var out: VertexOutput;
    let world_pos = model_matrix * vec4<f32>(vertex.position, 1.0);
    out.clip_position = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;

    let normal_matrix = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz,
    );
    out.world_normal = normalize(normal_matrix * vertex.normal);
    out.world_tangent = normalize(normal_matrix * vertex.tangent.xyz);
    out.world_bitangent = cross(out.world_normal, out.world_tangent) * vertex.tangent.w;

    out.uv = vertex.uv;
    out.color = instance.color * vertex.vertex_color;
    out.material = instance.material;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // clip box
    if (camera.clip_min.w > 0.5) {
        let p = in.world_position;
        if (p.x < camera.clip_min.x || p.y < camera.clip_min.y || p.z < camera.clip_min.z ||
            p.x > camera.clip_max.x || p.y > camera.clip_max.y || p.z > camera.clip_max.z) {
            discard;
        }
    }

    let tex_color = textureSample(t_diffuse, s_diffuse, in.uv);

    let metallic = clamp(in.material.x, 0.0, 1.0);
    let roughness = clamp(in.material.y, 0.04, 1.0);
    let mat_z = in.material.z;
    let is_transmission = mat_z < 0.0;
    let sss = select(clamp(mat_z, 0.0, 1.0), 0.0, is_transmission);
    let transmission = select(0.0, clamp(-mat_z, 0.0, 1.0), is_transmission);
    let emissive_strength = max(in.material.w, 0.0);
    let albedo = in.color.rgb * tex_color.rgb;

    let glass_f0 = ior_to_f0(1.52);
    let f0_dielectric = select(vec3(0.04), vec3(glass_f0), is_transmission);
    let f0 = mix(f0_dielectric, albedo, metallic);

    let n = normalize(in.world_normal);
    let v = normalize(camera.position.xyz - in.world_position);
    let n_dot_v = max(dot(n, v), 0.001);

    let light_count = i32(light_header.ambient_and_count.a);
    let is_dark_room = light_header.mode_flags.x > 0.5;

    let f_ambient = fresnel_schlick(n_dot_v, f0);
    let kd_ambient = (1.0 - f_ambient) * (1.0 - metallic);
    var ambient_base = light_header.ambient_and_count.rgb;
    if (is_dark_room) {
        ambient_base = ambient_base * 0.02;
    }
    let ambient = ambient_base * (kd_ambient * albedo + f_ambient * 0.1);

    let shadow = calculate_shadow(in.world_position);

    var lo = vec3(0.0);

    for (var i = 0; i < light_count; i = i + 1) {
        let light = lights[i];
        let light_type = light.direction_or_position_and_type.w;
        let light_color = light.color_and_intensity.rgb;
        let intensity = light.color_and_intensity.a;
        let range = light.extra.x;
        let spot_half_angle = light.extra.z;

        var l: vec3<f32>;
        var attenuation: f32 = 1.0;

        if (light_type < 0.5) {
            l = normalize(light.direction_or_position_and_type.xyz);
        } else {
            let light_vec = light.direction_or_position_and_type.xyz - in.world_position;
            let dist = length(light_vec);
            l = light_vec / max(dist, 0.001);

            if (is_dark_room) {
                let dist_m = dist / 1000.0;
                attenuation = 1.0 / max(dist_m * dist_m, 0.001);
                if (range > 0.0 && dist > range) {
                    attenuation = 0.0;
                }
            } else {
                attenuation = 1.0 / (1.0 + 0.0001 * dist * dist);
            }

            if (spot_half_angle > 0.0) {
                let spot_dir = vec3(0.0, 0.0, -1.0);
                let cos_angle = dot(-l, spot_dir);
                let cos_half = cos(spot_half_angle);
                let cos_outer = cos(spot_half_angle * 1.2);
                attenuation = attenuation * smoothstep(cos_outer, cos_half, cos_angle);
            }
        }

        let h = normalize(v + l);
        let raw_n_dot_l = dot(n, l);
        let n_dot_h = max(dot(n, h), 0.0);
        let h_dot_v = max(dot(h, v), 0.0);

        let wrap = 0.3 * sss;
        let n_dot_l_wrap = (raw_n_dot_l + wrap) / (1.0 + wrap);
        let n_dot_l = max(select(raw_n_dot_l, n_dot_l_wrap, sss > 0.0), 0.0);

        let ndf = distribution_ggx(n_dot_h, roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let f = fresnel_schlick(h_dot_v, f0);

        let numerator = ndf * g * f;
        let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
        let specular = numerator / denominator;

        let kd = (1.0 - f) * (1.0 - metallic);
        let diffuse = kd * albedo / PI;

        let radiance = light_color * intensity * attenuation;

        let scatter = max(0.0, -raw_n_dot_l) * sss * 0.3;
        let scatter_color = vec3(1.0, 0.4, 0.2);
        let sss_contrib = scatter * scatter_color * albedo * radiance;

        var shadow_factor: f32 = 1.0;
        if (i == 0 && light_type < 0.5) {
            shadow_factor = shadow;
        }
        lo = lo + (diffuse + specular) * radiance * n_dot_l * shadow_factor + sss_contrib;
    }

    // Emissive
    let emissive = albedo * emissive_strength;

    let color = ambient + lo + emissive;
    let mapped = aces_tonemap(color);
    let gamma_corrected = pow(mapped, vec3(1.0 / 2.2));

    // Glass: Fresnel-based transmission alpha
    var alpha = in.color.a * tex_color.a;
    if (is_transmission) {
        let fresnel = fresnel_schlick(n_dot_v, f0);
        let avg_fresnel = (fresnel.x + fresnel.y + fresnel.z) / 3.0;
        let transmittance = transmission * (1.0 - avg_fresnel);
        alpha = 1.0 - transmittance;
    }

    return vec4<f32>(gamma_corrected, alpha);
}
