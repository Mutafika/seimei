// === スクリーンスペース屈折 専用シェーダ ===
// pbr.wgsl の water_shade をベースに、group2 に「不透明描画済みシーンカラー」テクスチャを
// 束ねて、refr_uv で実際にサンプルする真のスクリーンスペース屈折を行う。
// 頂点シェーダ・VertexInput/InstanceInput は pbr.wgsl と完全同一レイアウト
// （同じ vertex/instance buffer を流用するため）。
// バインドグループ: group0=camera, group1=light, group2=scene_color(tex+sampler) の3つのみ。

// === Group 0: Camera ===
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec4<f32>,
    clip_min: vec4<f32>,
    clip_max: vec4<f32>,
    resolution: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// === Group 1: Lights ===
struct GpuLight {
    direction_or_position_and_type: vec4<f32>,
    color_and_intensity: vec4<f32>,
    extra: vec4<f32>,
};

struct LightUniform {
    ambient_and_count: vec4<f32>,
    lights: array<GpuLight, 8>,
};

@group(1) @binding(0)
var<uniform> light_data: LightUniform;

// === Group 2: Scene Color (不透明描画済みのHDRシーンカラーのコピー) ===
@group(2) @binding(0)
var scene_color: texture_2d<f32>;
@group(2) @binding(1)
var scene_samp: sampler;

const PI: f32 = 3.14159265359;

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

// === Vertex/Instance Input (pbr.wgsl と同一レイアウト) ===
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(9) tangent: vec4<f32>,
    @location(10) vertex_color: vec4<f32>,
};

// シェーディングモデルID（seimei vertex.rs の MODEL_* / pbr.wgsl の M_* と一致）
const M_FLUID: i32 = 5;

struct InstanceInput {
    @location(3) model_matrix_0: vec4<f32>,
    @location(4) model_matrix_1: vec4<f32>,
    @location(5) model_matrix_2: vec4<f32>,
    @location(6) model_matrix_3: vec4<f32>,
    @location(7) color: vec4<f32>,
    @location(8) material: vec4<f32>,
    @location(11) model_id: f32,
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
    @location(7) @interpolate(flat) model_id: i32,
};

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
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
    out.model_id = i32(instance.model_id + 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // material.x = 粘性(0=サラサラ水, 1=どろどろ)。
    let visc = clamp(in.material.x, 0.0, 1.0);
    let n0 = normalize(in.world_normal);
    let v = normalize(camera.position.xyz - in.world_position);
    let wp = in.world_position;
    var up = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(n0.z) > 0.9) { up = vec3<f32>(1.0, 0.0, 0.0); }
    let t = normalize(cross(up, n0));
    let b = cross(n0, t);
    let ph = wp.z * 0.10;
    let w1 = sin(ph + dot(wp, t) * 0.6);
    let w2 = sin(ph * 1.9 + dot(wp, b) * 0.4 + 2.0);
    let bump = 0.12 * (1.0 - 0.7 * visc);
    let n = normalize(n0 + t * (w1 * bump) + b * (w2 * bump));
    let ndv = max(dot(n, v), 1e-3);
    let fres = clamp(0.03 + 0.35 * pow(1.0 - ndv, 4.0), 0.0, 0.32);
    let shin = mix(70.0, 220.0, visc);
    var spec = 0.0;
    let lc = i32(light_data.ambient_and_count.a);
    for (var i = 0; i < lc; i = i + 1) {
        let lt = light_data.lights[i];
        var ld: vec3<f32>;
        if (lt.direction_or_position_and_type.w < 0.5) {
            ld = normalize(lt.direction_or_position_and_type.xyz);
        } else {
            ld = normalize(lt.direction_or_position_and_type.xyz - wp);
        }
        let h = normalize(ld + v);
        spec = spec + pow(max(dot(n, h), 0.0), shin) * lt.color_and_intensity.a;
    }
    spec = clamp(spec, 0.0, 1.5) * mix(0.12, 0.65, visc);
    let tint = in.color.rgb;
    let sky = vec3<f32>(0.62, 0.76, 0.96);

    // === 真のスクリーンスペース屈折 ===
    // フラグメントのスクリーンUV = @builtin(position).xy / 解像度。
    // 表面法線の view空間XY成分に比例してUVをずらし、不透明描画済みのシーンカラー
    // (scene_color)をサンプル＝背景が歪んだ屈折像になる。粘性で歪み量を変調。
    let res = max(camera.resolution.xy, vec2<f32>(1.0, 1.0));
    let screen_uv = in.clip_position.xy / res;
    let n_view = (camera.view * vec4<f32>(n, 0.0)).xyz;
    // 歪み量(スクリーンUV比)。大きすぎると像が崩壊するので控えめ＝最大~3%程度に。
    let distort = (0.012 + 0.020 * visc) * (0.5 + 0.5 * fres);
    let refr_uv = clamp(screen_uv + n_view.xy * distort, vec2<f32>(0.0), vec2<f32>(1.0));
    // 屈折色 = 背景シーンカラーを refr_uv でサンプル。地色(tint)で軽く色付け。
    let scene = textureSample(scene_color, scene_samp, refr_uv).rgb;
    let refraction = mix(scene, scene * tint * 1.4, 0.25 + 0.35 * visc);

    // 反射(空)とフレネルでブレンド。中心ほど屈折(背景透過)、縁ほど反射。
    let skymix = fres * (1.0 - 0.85 * visc);
    var col = mix(refraction, sky, skymix) + vec3<f32>(spec);
    col = mix(col, tint * 1.04, visc * 0.3);
    // 屈折パイプラインはHDR(pp)モードでのみ動く。tonemap/gamma は合成シェーダが一括で行うので
    // ここは線形のまま出力（二重に掛けると washout する）。非pp時のフォールバックのみ自前処理。
    if (camera.resolution.z < 0.5) {
        col = aces_tonemap(col);
        col = pow(col, vec3<f32>(1.0 / 2.2));
    }

    // 透過度。濃い不透明な液(Fluid)はスライダー直結、それ以外(Water)は視線依存。
    let water_a = clamp(0.04 + fres * 0.45 + spec * 0.15, 0.0, 0.40);
    let thick_a = clamp(in.color.a * (0.85 + 0.15 * fres) + spec * 0.2, 0.0, 1.0);
    var alpha = mix(water_a, thick_a, visc);
    if (in.model_id == M_FLUID) {
        alpha = clamp(in.color.a + spec * 0.15, 0.0, 1.0);
    }
    return vec4<f32>(col, alpha);
}
