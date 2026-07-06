// === Group 0: Camera ===
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec4<f32>,
    clip_min: vec4<f32>,
    clip_max: vec4<f32>,
    // xy=描画解像度(px), z=屈折フラグ(1=シーンカラーtex有効), w=予備。
    // スクリーンスペース屈折のUV計算に使う。
    resolution: vec4<f32>,
    // 肌パラメータ: x=全身濡れfloor(0..1), y=濡れ新方式A/B(1=新/0=旧), z=SSS倍率, w=SSS新A/B(1/0)。
    skin_params: vec4<f32>,
    // レンダFX: x=髪異方性強度(0=OFF), y=瞳角膜艶倍率(1=通常), z=グレード強度(composite用), w=予備。
    fx_params: vec4<f32>,
    // レンダFX2: x=リムライト強度(0=OFF), y=DoF量(composite用), z/w=予備。
    fx_params2: vec4<f32>,
    // 溶解の焼き点: xyz=world座標の溶解中心, w=熱(0..1)。この点の近傍だけ溶け縁を焦がし光らせる。
    melt: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// === Group 1: Lights (uniform, WebGL2互換) ===
struct GpuLight {
    direction_or_position_and_type: vec4<f32>,
    color_and_intensity: vec4<f32>,
    extra: vec4<f32>,
};

struct LightUniform {
    // rgb = ambient color, a = light_count
    ambient_and_count: vec4<f32>,
    lights: array<GpuLight, 8>,
};

@group(1) @binding(0)
var<uniform> light_data: LightUniform;

// === Group 2: Texture ===
@group(2) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(2) @binding(1)
var s_diffuse: sampler;

// === Group 3: Paint map (体表塗布: 液の付着を UV 空間に塗り込み、面に合成) ===
// rgb=塗布色, a=被覆率(0=素肌/1=塗り潰し)。塗布なしメッシュは透明(__paint_none__)が入る。
@group(3) @binding(0)
var t_paint: texture_2d<f32>;
@group(3) @binding(1)
var s_paint: sampler;

// 塗布時の面法線マップ（同 group 3 の binding 2/3。rgb=encode(n*0.5+0.5)）。
// 表裏/左右でUVを共有するモデル（VRoid等）で、塗った時と逆を向く面の塗布を弾く＝裏に滲ませない。
// rgb=(0,0,0) は「法線未記録」＝弾かない（後方互換）。塗布なしメッシュは透明(__paint_none__)。
// バインドグループを増やさず1グループに2tex束ねる（max_bind_groups=4 制限のため）。
@group(3) @binding(2)
var t_paintn: texture_2d<f32>;
@group(3) @binding(3)
var s_paintn: sampler;

const PI: f32 = 3.14159265359;

// 2D→1D ハッシュ（濡れ表面の微小rough揺らぎ用。world座標で安定＝カメラで泳がない）。
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// 3D→1D ハッシュ（溶解ノイズのセル値用）。
fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// 3D value noise（三線形補間で連続）。生ハッシュを座標に直接使うと1画素ごとに乱数が変わり
// 塩胡椒ノイズ(砂/ラメ)になるため、格子点でハッシュ→補間して「なめらかな」場を作る。溶解の穴を
// コヒーレントに繋げるのに必須。
fn vnoise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// === PBR Functions ===

// GGX/Trowbridge-Reitz 法線分布関数
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Smith GGX 幾何減衰関数
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

// Schlick フレネル近似
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ACES トーンマッピング
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

// === シェーディングモデルID（seimei vertex.rs の MODEL_* と一致） ===
// 旧来は material.w(発光と髪=3/瞳=4/水>5タグの兼用) と material.z の符号(肌/透過) を値で推定して
// いたが、標準材質(革/金属)が誤って特殊分岐に巻き込まれる事故源だった。明示IDで分岐する。
const M_STANDARD: i32 = 0;
const M_SKIN: i32 = 1;
const M_HAIR: i32 = 2;
const M_EYE: i32 = 3;
const M_WATER: i32 = 4;
const M_FLUID: i32 = 5;
const M_GLASS: i32 = 6;
const M_JELLY: i32 = 7;

// === Vertex/Instance Input ===
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(9) tangent: vec4<f32>,  // xyz=tangent, w=bitangent sign
    @location(10) vertex_color: vec4<f32>,
};

struct InstanceInput {
    @location(3) model_matrix_0: vec4<f32>,
    @location(4) model_matrix_1: vec4<f32>,
    @location(5) model_matrix_2: vec4<f32>,
    @location(6) model_matrix_3: vec4<f32>,
    @location(7) color: vec4<f32>,
    @location(8) material: vec4<f32>,  // [metallic, roughness, sss(肌/ゼリー)又はtransmission(ガラス), emissive]
    @location(11) model_id: f32,       // シェーディングモデル（M_* 定数）
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

    // 法線変換（スケールを無視するため正規化）
    let normal_matrix = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz,
    );
    out.world_normal = normalize(normal_matrix * vertex.normal);

    // TBN行列用のタンジェント・バイタンジェント
    out.world_tangent = normalize(normal_matrix * vertex.tangent.xyz);
    out.world_bitangent = cross(out.world_normal, out.world_tangent) * vertex.tangent.w;

    out.uv = vertex.uv;
    out.color = instance.color * vertex.vertex_color;
    out.material = instance.material;
    out.model_id = i32(instance.model_id + 0.5);
    return out;
}

// IOR → F0 (Schlick近似のF0パラメータ)
fn ior_to_f0(ior: f32) -> f32 {
    let r = (ior - 1.0) / (ior + 1.0);
    return r * r;
}

// === 水(water): material[3] > 5.0 をフラグに専用シェーディングへ分岐。屈折用シーンカラーが無い
// ので核は (1)ワールド座標駆動のリップル法線＝ジオメトリが毎フレーム動くので流れて見える,
// (2)視線依存の透明(中心は透け縁は反射), (3)控えめなスペキュラ。emissive 経路の白飛びを回避し、
// 「青く透ける水」を作る（白く塗り潰さない）。
fn water_shade(in: VertexOutput) -> vec4<f32> {
    // material.x = 粘性(0=サラサラ水, 1=どろどろ)。水↔粘液を 1 本のシェーダでブレンド。
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
    let bump = 0.12 * (1.0 - 0.7 * visc); // 粘い液ほど波立たず滑らかな表面
    let n = normalize(n0 + t * (w1 * bump) + b * (w2 * bump));
    let ndv = max(dot(n, v), 1e-3);
    let fres = clamp(0.03 + 0.35 * pow(1.0 - ndv, 4.0), 0.0, 0.32);
    let shin = mix(70.0, 220.0, visc); // 粘い液は鋭くテラテラのハイライト
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
    spec = clamp(spec, 0.0, 1.5) * mix(0.12, 0.65, visc); // 水(低粘性)は白いハイライトを抑えて透明に
    let tint = in.color.rgb;
    let sky = vec3<f32>(0.62, 0.76, 0.96);

    // === スクリーンスペース屈折 ===
    // フラグメントのスクリーンUV = @builtin(position).xy / 解像度。
    // 表面法線の「view空間XY成分」に比例してUVをずらし、背景(シーンカラー)を
    // サンプルすると歪んだ屈折像になる。粘性ほど歪み・色付けを強める。
    // resolution.z(屈折フラグ) >= 0.5 でシーンカラーtexが配線されたら真の屈折へ昇格。
    // 現状(<0.5)は tex 未配線なので「空ベースの屈折色」をフォールバックに使い、見た目を壊さない。
    let res = max(camera.resolution.xy, vec2<f32>(1.0, 1.0));
    let screen_uv = in.clip_position.xy / res;
    // view空間法線（オフセット方向）。歪み量は粘性とフレネルで変調。
    let n_view = (camera.view * vec4<f32>(n, 0.0)).xyz;
    let distort = (0.03 + 0.05 * visc) * (0.5 + 0.5 * fres);
    let refr_uv = clamp(screen_uv + n_view.xy * distort, vec2<f32>(0.0), vec2<f32>(1.0));
    // 屈折色: シーンカラーtex が無い間は「地色×空」を近似背景として使う。
    // （tex 配線後はここを textureSample(scene_color, samp, refr_uv).rgb に差し替える）
    let bg_approx = mix(tint, sky, 0.25 + 0.35 * visc);
    let refraction = bg_approx;

    // 粘い液は空の反射が乏しく地色(tint)が主役。水は反射で青空が乗る。
    let skymix = fres * (1.0 - 0.85 * visc);
    // 屈折(背景)と反射(空)をフレネルでブレンド。中心ほど屈折、縁ほど反射。
    var col = mix(refraction, sky, skymix) + vec3<f32>(spec);
    col = mix(col, tint * 1.04, visc * 0.3); // 粘液の地色をわずかに持ち上げ（濃い液の色を保つ）
    col = aces_tonemap(col);
    col = pow(col, vec3<f32>(1.0 / 2.2));
    // 透過度の決め方は flag で分岐。
    //  flag>=7.5 (濃い不透明な液): tint.a(=透過度スライダー)を不透明度として直接使う＝粘性とは無関係。
    //    粘性は色/艶(spec/bump/skymix)だけに効かせ、透け具合はスライダーで独立制御する。
    //  flag in (5,7.5) (水等): 従来どおり視線依存の薄い透過↔濁りを粘性でブレンド。
    let water_a = clamp(0.04 + fres * 0.45 + spec * 0.15, 0.0, 0.40); // 水は薄く＝背景が透けて濁らない
    let thick_a = clamp(in.color.a * (0.85 + 0.15 * fres) + spec * 0.2, 0.0, 1.0);
    var alpha = mix(water_a, thick_a, visc);
    if (in.model_id == M_FLUID) {
        alpha = clamp(in.color.a + spec * 0.15, 0.0, 1.0); // 濃い液=透過度スライダー直結＋ハイライトだけ僅かに
    }
    return vec4<f32>(col, alpha);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let model = in.model_id;
    // 水/濃い液は専用シェーディングへ（テクスチャ判定前に分岐）。
    if (model == M_WATER || model == M_FLUID) {
        return water_shade(in);
    }
    // クリップボックス判定
    if (camera.clip_min.w > 0.5) {
        let p = in.world_position;
        if (p.x < camera.clip_min.x || p.y < camera.clip_min.y || p.z < camera.clip_min.z ||
            p.x > camera.clip_max.x || p.y > camera.clip_max.y || p.z > camera.clip_max.z) {
            discard;
        }
    }

    // テクスチャサンプリング
    let tex_color = textureSample(t_diffuse, s_diffuse, in.uv);

    // アルファテスト
    if (tex_color.a < 0.01) {
        discard;
    }

    // === 溶解(dissolve/melt): 表面を局所的に溶かして穴を空ける汎用機能 ===
    // 溶解フラグ material.z>0.5 を立てた標準材(=服)のみ対象。in.color.a を頂点ごとの「溶け量」として
    // 読む(不透明clothは instance.a=1 なので in.color.a==頂点値・a==1→dissolve=0で無反応)。
    // world座標の有機ノイズで縁がレース状に溶ける。out.color.a は不透明パスでは出力に効かない。
    // ★ゲート必須: 髪/肌/ガラス/霧等の「元々半透明(alpha<1)」な材を巻き込んで砂状に消さないため、
    //   model==M_STANDARD かつ material.z>0.5(服が溶解中の時だけ ticsim が立てる)に限定する。
    let dissolve = select(
        0.0,
        clamp(1.0 - in.color.a, 0.0, 1.0),
        model == M_STANDARD && in.material.z > 0.5);
    // 溶解の「熱」＝現在の焼き点(camera.melt.xyz)の近傍のみ camera.melt.w で光る。
    // ＝当てている場所だけ縁が焦げ光り、離れた古い穴は光らない（局所化）。半径は狭く絞る＝
    // 近接した別の穴(例: 胸と腹は≈200mm)を巻き込まない。焼き点直近だけ(≈35mmで満・75mmで消)。
    let melt_heat = camera.melt.w * (1.0 - smoothstep(35.0, 75.0, distance(in.world_position, camera.melt.xyz)));
    var melt_edge = 0.0;
    if (dissolve > 0.002) {
        let wp = in.world_position;
        // なめらか value noise 2オクターブ(mm単位)。粗(≈11mm)で穴の塊、細(≈4mm)で縁のレース。
        // 連続なので穴が繋がり、白ノイズの砂ザラつきにならない。
        let nz = vnoise3(wp * 0.09) * 0.62 + vnoise3(wp * 0.24) * 0.38;
        // 溶け量がノイズを越えた画素を捨てる＝穴が有機的に広がる。a=0(完全溶解)は nz<1 が常真で確実消滅。
        if (nz < dissolve) {
            discard;
        }
        // 消滅境界の細い帯＝溶け縁。発光/焦がしはこの帯 × 熱でのみ出す(熱ゼロ＝ただの綺麗な穴)。
        melt_edge = (1.0 - smoothstep(0.0, 0.09, nz - dissolve)) * melt_heat;
    }

    // 体表塗布マップ（付着した液）。被覆率 paint.a で素肌色↔塗布色を混ぜ、濡れて滑らかに（roughness↓）。
    let paint = textureSample(t_paint, s_paint, in.uv);
    // 塗布時法線で表裏を判定: 記録法線と現在の面法線が逆向き(dot<0)なら、表のUVを共有する
    // 裏面なので塗布を消す。
    // 重要: 法線mapの未塗布texelは rgb=a=0。線形補間で塗りの縁に混ざると rgb→0＝decode で
    // (-1,-1,-1) の「裏向き」に誤判定し、疎な塗り(顔等)が縁だらけで丸ごと消える。そこで
    // 「塗られた寄与だけ」を alpha で割って復元（未塗布は rgb/a とも 0 寄与で打ち消える）。
    let pn_tex = textureSample(t_paintn, s_paintn, in.uv);
    let pn_a = pn_tex.a;
    let has_n = pn_a > 0.02; // 塗布寄与がほぼ無い texel は法線未記録＝判定しない
    let pn_raw = (pn_tex.xyz / max(pn_a, 0.001)) * 2.0 - 1.0;
    let side = select(1.0, dot(normalize(in.world_normal), normalize(pn_raw)), has_n);
    let side_mask = smoothstep(-0.05, 0.35, side); // 表=1 / 裏=0 へ滑らかに
    let paint_a = clamp(paint.a, 0.0, 1.0) * side_mask;
    // 「濡れ(wet)」と「不透明度(opaque)」を分離して液種を両立する:
    //  - wet    = 被覆を厚みに見立てた濡れ量。艶(roughness↓)/clearcoat/凹凸に効く＝全液種共通。
    //  - opaque = wet × 塗布色の白さ。アルベド白化＆SSS に効く＝白く塗った塗布だけ。暗く塗ると ≈0
    //    （paint_white→0 となり、肌色を変えずテラテラの濡れだけ乗る＝透明な液の定着）。
    // sqrt で中被覆を持ち上げ＝塊が埋まる。極低被覆の縁は薄く残りメニスカス（縁透け）を維持。
    // 評価/全身汗用の「全身濡れ floor」(skin_params.x)。塗布が無くても全身を濡らせる。
    // 全身濡れ floor は「肌(SSS>0)」のみに効かせる。これをしないと global uniform が
    // 全メッシュ共通のため、レンガ壁等の環境メッシュ(SSS=0)まで roughness↓＋clearcoat が
    // 乗って世界中が specular firefly でビカビカになる。塗布由来の濡れ(wet_p)は全メッシュ可。
    let gw_gate = select(0.0, 1.0, model == M_SKIN); // 全身濡れfloorは肌モデルのみ
    let gw0 = clamp(camera.skin_params.x, 0.0, 1.0) * gw_gate;
    // A/B(skin_params.y): 新=水膜フレネル＋筋ムラ＋環境反射 / 旧=均一クリアコート。
    let wet_new = camera.skin_params.y > 0.5;
    // 汗筋ムラ: 全身濡れを均一な水膜にしない。縦長ノイズ(Y低周波×XZ中周波)で「流れた筋/
    // たまり」の濃淡を付ける＝均一テカリのプラ人形感を割る本丸。塗布(wet_p)は被覆勾配で
    // 既に構造があるので触らない。world空間なので大移動では泳ぐが溶解ノイズと同じ割り切り。
    let streak_n = vnoise3(vec3<f32>(in.world_position.x * 0.035, in.world_position.y * 0.005, in.world_position.z * 0.035));
    // 濡れデバッグ(fx_params2.z): 1=筋ムラOFF / 2=環境反射OFF / 3=旧rough / 4=旧水膜倍率。
    // アーティファクト(境界線等)の犯人切り分け用。0で全効果ON。
    let wet_dbg = camera.fx_params2.z;
    let dbg_no_streak = wet_dbg > 0.5 && wet_dbg < 1.5;
    let dbg_no_env = wet_dbg > 1.5 && wet_dbg < 2.5;
    let dbg_old_rough = wet_dbg > 2.5 && wet_dbg < 3.5;
    let dbg_old_cc = wet_dbg > 3.5 && wet_dbg < 4.5;
    let streak = select(select(1.0, mix(0.55, 1.3, streak_n), wet_new), 1.0, dbg_no_streak);
    let gw = clamp(gw0 * streak, 0.0, 1.0);
    let wet_p = sqrt(paint_a);     // 塗布由来の濡れ
    let wet = max(wet_p, gw);      // 実効濡れ（roughness等に効く）
    let paint_luma = dot(paint.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let paint_white = smoothstep(0.55, 0.85, paint_luma); // 白く焼かれた塗り=1 / 暗く焼かれた塗り=0
    let opaque_body = wet_p * paint_white; // 白化は塗布のみ（全身濡れは透明として扱う）
    // 立体感: 被覆率を「厚み」とみなし、その勾配で法線を起伏させる＝粘液が盛り上がって見え、
    // 縁が丸く光る（平らな塗料でなく濡れた塊に）。近傍4点の差分で UV 勾配→TBN で世界法線へ。
    let psz = vec2<f32>(textureDimensions(t_paint));
    let ptx = 1.0 / max(psz, vec2<f32>(1.0));
    let h_l = textureSample(t_paint, s_paint, in.uv - vec2<f32>(ptx.x, 0.0)).a;
    let h_r = textureSample(t_paint, s_paint, in.uv + vec2<f32>(ptx.x, 0.0)).a;
    let h_d = textureSample(t_paint, s_paint, in.uv - vec2<f32>(0.0, ptx.y)).a;
    let h_u = textureSample(t_paint, s_paint, in.uv + vec2<f32>(0.0, ptx.y)).a;
    let paint_grad = vec2<f32>(h_r - h_l, h_u - h_d);

    // マテリアルパラメータ
    let metallic = clamp(in.material.x, 0.0, 1.0);
    // シェーディングモデルは model_id で明示分岐（値域推定の sentinel を撤去＝標準材が誤って
    // 髪/瞳/肌の特殊分岐に巻き込まれる事故を構造的に排除）。
    let is_hair = model == M_HAIR;
    let is_eye = model == M_EYE;
    let is_sss = model == M_SKIN || model == M_JELLY; // SSSを持つモデル
    let is_transmission = model == M_GLASS;
    // 塗られた所は濡れて滑らか＝鋭いハイライト（液体の艶）。白い/暗い塗布問わず wet で艶を出す。
    // 瞳は角膜=常にツルツル＝低 rough で鋭い反射（濡れに依らず固定）。
    let rough_wet = select(select(0.14, 0.09, wet_new), 0.14, dbg_old_rough); // 新方式は水膜らしく更に鋭く
    let roughness = select(mix(clamp(in.material.y, 0.04, 1.0), rough_wet, wet), 0.035, is_eye);
    let mat_z = in.material.z;
    // material[2]: 肌/ゼリー=SSS強度 / ガラス=transmission量（共に正値）
    let sss = select(0.0, clamp(mat_z, 0.0, 1.0), is_sss);
    let transmission = select(0.0, clamp(mat_z, 0.0, 1.0), is_transmission);
    let emissive_strength = max(in.material.w, 0.0); // 発光は純粋に発光のみ（タグ兼用を撤去）

    // 透明濡れ = wet のうち白くない分（暗く塗られた塗布＝肌色を変えない濡れ）。汗テカリと同様、
    // 肌色を保ったまま「少し暗化＋艶」だけ乗せる。白い塗布(paint_white≈1)では 0＝不透明ボディが支配。
    let wet_clear = max(wet_p * (1.0 - paint_white), gw); // 全身濡れは透明濡れとして加算
    // 透明ジェルのレンズ屈折: ジェル表面の勾配(paint_grad=被覆の傾き)で下の肌テクスチャの UV を
    // ズラして再サンプル＝水滴/ジェル越しに肌が歪んで見える。平らなジェル(勾配0)は歪まず、縁や
    // 盛り上がりの所だけ曲がる＝物理的に正しい。塗布は一切動かさず陰影サンプルだけ曲げる。
    let lens_uv = in.uv - paint_grad * (wet_clear * 0.045);
    let skin_tex = mix(tex_color.rgb, textureSample(t_diffuse, s_diffuse, lens_uv).rgb, clamp(wet_clear, 0.0, 1.0));
    // ベースカラー = インスタンスカラー * (レンズで歪めた)テクスチャ。塗布があれば不透明ボディ被覆で
    // 塗布色へ寄せ、さらに白へ少し持ち上げる＝透明ガラスでなく白く濁った不透明な液の地色に。
    let albedo0 = mix(in.color.rgb * skin_tex, paint.rgb, opaque_body);
    let albedo1 = mix(albedo0, vec3<f32>(0.96, 0.95, 0.93), opaque_body * 0.22);
    // 濡れると暗くなる（水が光を吸う）＝濡れ感の核。A/B は上で定義済みの wet_new。
    // 濡れの暗化は控えめに。暗いダンジョンはアンビエントが弱く、albedo暗化が拡散/アンビ/SSS
    // 全てに効くため強いと肌が黒潰れする。「しっとり艶」は clearcoat 側で出す。
    // 筋ムラ(streak)が wet_clear に乗るので、暗化も筋状に濃淡が付く＝流れた汗の跡。
    let dark_amt = select(0.18 * wet_clear, 0.14 * smoothstep(0.0, 1.0, wet_clear), wet_new);
    let albedo = albedo1 * (1.0 - dark_amt);

    // 誘電体の基本反射率 (F0)
    // ガラスの場合: IOR 1.52 → F0 ≈ 0.0425
    let glass_f0 = ior_to_f0(1.52);
    let f0_dielectric = select(vec3(0.04), vec3(glass_f0), is_transmission);
    let f0 = mix(f0_dielectric, albedo, metallic);

    // 塗布の厚み勾配で法線を起伏（盛り上がり）。被覆率(=厚み)の勾配で縁が丸く盛り上がる＝
    // 濡れた塊の艶。bump は控えめにして「粒ごとの黒い縁取り/網目」を出さず、滑らかな盛りに。
    let n_geo = normalize(in.world_normal);
    // 無塗布(paint_a==0)では塗布由来の法線起伏を一切計算しない。乗算ゲート(... * paint_a)は、
    // paint_a==0 でも接線項(タンジェント未生成メッシュ等で normalize(0)=NaN になり得る)を
    // 巻き込むと 0×NaN=NaN となり陰影法線 n を破壊し、塗布の無い汎用メッシュ(壁等)が脱色・
    // 消失する回帰を生む(#2)。分岐ゲートにして未塗布は n_geo を素通しさせ、塗布導入前の描画と
    // bit 一致させる。塗布合成(roughness/albedo/clearcoat/opaque_floor)は paint_a==0 で有限値
    // に収束する(mix(x,_,0)=x, _*0=0)ため変更不要＝NaN源は法線起伏のみ。
    var n = n_geo;
    if (paint_a > 0.0) {
        let tb_t = normalize(in.world_tangent);
        let tb_b = normalize(in.world_bitangent);
        // 起伏: 白い塗布(濃い不透明な液)は厚い塊なので強く盛り上げる。透明濡れは薄い膜＝ほぼ平ら
        // （汗テカリと同じく面で均一にヌメッと光らせる）ので paint_white で起伏量を絞る。
        let paint_bump = 6.0 * mix(0.18, 1.0, paint_white);
        n = normalize(n_geo - paint_bump * paint_a * (paint_grad.x * tb_t + paint_grad.y * tb_b));
    }
    let v = normalize(camera.position.xyz - in.world_position);
    let n_dot_v = max(dot(n, v), 0.001);
    // スペキュラAA（法線分散→roughness床上げ・Kaplanyan NDFフィルタの簡易版）。
    // 焼き込みメッシュの局所サブディブ境界/クリース/シルエットは法線が画素間で急変し、
    // 低roughの鏡面（濡れ水膜）がそこを「境界線」や白ブロブとして露出させる。画素間の
    // 法線分散ぶんだけ局所的に粗くして、滑らかな面のシャープさは保ったまま急変部だけ鈍らせる。
    let n_var = dot(dpdx(n), dpdx(n)) + dot(dpdy(n), dpdy(n));
    let spec_aa = min(0.20, 2.0 * n_var);
    let rough_s = min(1.0, sqrt(roughness * roughness + spec_aa));

    let light_count = i32(light_data.ambient_and_count.a);
    let is_dark_room = false;

    // アンビエント (簡易IBL近似)
    // 暗室モード: アンビエントを大幅減衰（微小環境光のみ）
    let f_ambient = fresnel_schlick(n_dot_v, f0);
    let kd_ambient = (1.0 - f_ambient) * (1.0 - metallic);
    var ambient_base = light_data.ambient_and_count.rgb;
    if (is_dark_room) {
        ambient_base = ambient_base * 0.02;
    }
    // 濡れ環境反射(擬似IBL): 水膜は解析ライトの向きに依らず環境光を鏡面で拾う。視線の浅い
    // 体側/輪郭ほど強い水膜フレネル＝ライト正面以外も「しっとり」見える濡れ感の主役。
    // 全世界ビカビカ(firefly)防止で肌系のみ・強度は wet(筋ムラ込み)に比例。
    let wet_env_f = 0.02 + 0.98 * pow(1.0 - n_dot_v, 3.0);
    let wet_env = ambient_base * wet_env_f * wet * 1.5 * select(0.0, 1.0, wet_new && is_sss && !dbg_no_env);
    let ambient = ambient_base * (kd_ambient * albedo + f_ambient * 0.1) + wet_env;

    var lo = vec3(0.0);

    // 各ライトからのPBR寄与を累積
    for (var i = 0; i < light_count; i = i + 1) {
        let light = light_data.lights[i];
        let light_type = light.direction_or_position_and_type.w;
        let light_color = light.color_and_intensity.rgb;
        let intensity = light.color_and_intensity.a;
        let range = light.extra.x;
        let spot_half_angle = light.extra.z;

        var l: vec3<f32>;
        var attenuation: f32 = 1.0;

        if (light_type < 0.5) {
            // Directional light
            l = normalize(light.direction_or_position_and_type.xyz);
        } else {
            // Point light
            let light_vec = light.direction_or_position_and_type.xyz - in.world_position;
            let dist = length(light_vec);
            l = light_vec / max(dist, 0.001);

            if (is_dark_room) {
                // 物理ベース逆二乗減衰 (mm→m変換)
                let dist_m = dist / 1000.0;
                attenuation = 1.0 / max(dist_m * dist_m, 0.001);
                // range制限
                if (range > 0.0 && dist > range) {
                    attenuation = 0.0;
                }
            } else {
                // 通常モード: 既存の減衰式（見た目の互換性維持）
                attenuation = 1.0 / (1.0 + 0.0001 * dist * dist);
            }

            // スポットライト コーン減衰
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

        // SSS ラップライティング（光をターミネーター越しに拡張）。S1 A/B(skin_params.w):
        //  新 = per-channel 赤方シフト（Rを一番遠くまで回す＝影の境目が赤くにじむ肌の核）＋強度倍率
        //  旧 = 単一wrap
        let sss_b = camera.skin_params.z;             // SSS強度倍率
        let sss_use_new = camera.skin_params.w > 0.5;
        let sss_e = sss * sss_b;                      // 実効SSS
        let wrap = 0.3 * sss_e;
        let wrap_rgb = wrap * select(vec3(1.0), vec3(1.0, 0.5, 0.32), sss_use_new); // Rを深く=滑らかな赤方シフト
        let nl_rgb = max((vec3(raw_n_dot_l) + wrap_rgb) / (vec3(1.0) + wrap_rgb), vec3(0.0));
        let n_dot_l = max(select(raw_n_dot_l, (raw_n_dot_l + wrap) / (1.0 + wrap), sss_e > 0.0), 0.0);
        // 拡散用の per-channel 重み（SSS新時のみ赤方シフト、それ以外はスカラー n_dot_l）
        let diff_w = select(vec3<f32>(n_dot_l), nl_rgb, sss_e > 0.0 && sss_use_new);

        // Cook-Torrance BRDF（rough_s=スペキュラAA込みroughness）
        let ndf = distribution_ggx(n_dot_h, rough_s);
        let g = geometry_smith(n_dot_v, n_dot_l, rough_s);
        let f = fresnel_schlick(h_dot_v, f0);

        let numerator = ndf * g * f;
        let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
        let specular = numerator / denominator;

        // 拡散反射 (金属は拡散反射なし)
        let kd = (1.0 - f) * (1.0 - metallic);
        let diffuse = kd * albedo / PI;

        let radiance = light_color * intensity * attenuation;

        // SSS: 背面散乱（影側の暖色透過光）。倍率反映＋新方式は少し強め。
        let scatter = max(0.0, -raw_n_dot_l) * sss_e * select(0.3, 0.45, sss_use_new);
        let scatter_color = vec3(1.0, 0.4, 0.2); // 暖色系（血液透過色）
        let sss_contrib = scatter * scatter_color * albedo * radiance;

        // 濡れた水膜の鏡面（薄い水層）。一律テカリ→「斜め/縁で強く光る本物の濡れ」へ:
        //  - 水のF0≈0.02、微小roughで完全鏡面のプラ感を割る
        //  - グレージング角(視線が浅い所)ほど強く＝水膜のフレネル
        //  - 被覆の縁でビーズ(表面張力の粒)を一段明るく
        // A/B(skin_params.y): 新=水膜フレネル＋グレージング＋縁ビーズ＋微小rough / 旧=均一クリアコート。
        // wet_amt=塗布 or 全身濡れ。全身濡れでも水膜艶が乗る。
        let wet_amt = max(paint_a, gw);
        // micro: 水膜のごく僅かな粗さ揺らぎ。大きいと per-pixel に rough が振れて firefly（ギラつき）に
        // なるので極小に。cc_rough 下限も上げて極小鏡面の aliasing(チラ)を抑える。
        let micro = select(0.0, (hash21(in.world_position.xy * 0.6 + in.world_position.zz * 0.6) - 0.5) * 0.006, wet_new);
        let cc_rough0 = select(0.05, clamp(0.07 + micro, 0.05, 0.12), wet_new);
        let cc_rough = min(0.6, sqrt(cc_rough0 * cc_rough0 + spec_aa)); // スペキュラAA込み
        let cc_ndf = distribution_ggx(n_dot_h, cc_rough);
        let cc_g = geometry_smith(n_dot_v, n_dot_l, cc_rough);
        let cc_f = fresnel_schlick(h_dot_v, select(vec3(0.04), vec3(0.02), wet_new));
        let cc_graze = select(1.0, mix(1.0, 2.4, pow(1.0 - n_dot_v, 3.0)), wet_new); // 縁の水膜リム
        let cc_bead = select(1.0, 1.0 + smoothstep(0.04, 0.22, wet_amt) * (1.0 - smoothstep(0.22, 0.6, wet_amt)) * 1.4, wet_new);
        // 新方式の水膜倍率: F0=0.02(物理値)のままだと解析ライト強度では鈍く「しょぼい」の
        // 主因だったため増強。筋ムラ(streak入り gw→wet_amt)で濃淡が付くので一様には飛ばない。
        let cc_mult = select(select(2.5, 3.4, wet_new), 2.0, dbg_old_cc);
        let clearcoat = (cc_ndf * cc_g * cc_f) / (4.0 * n_dot_v * n_dot_l + 0.0001);

        // 髪: アニメ的な「天使の輪」帯。tangent を法線方向へずらして帯位置を作り(主=上/副=下)、
        // 高指数で帯を細くする＝一様な面光りでなく帯に。色は髪色寄り(tint0.7)＋髪色でクランプ＝
        // 強めても白飛びせず“ブライトな髪色の艶”に飽和する（白光り回避）。
        let t_hair = normalize(in.world_tangent);
        let t1 = normalize(t_hair + n * 0.20);
        let t2 = normalize(t_hair - n * 0.30);
        let s1 = sqrt(max(0.0, 1.0 - dot(t1, h) * dot(t1, h)));
        let s2 = sqrt(max(0.0, 1.0 - dot(t2, h) * dot(t2, h)));
        let hair_spec = pow(s1, 140.0) + pow(s2, 50.0) * 0.25;
        let hair_tint = mix(vec3<f32>(1.0), albedo, 0.7); // 髪色寄りの艶（白飛び回避）
        // 鏡面項: 既定=GGX / 髪=天使の輪(tint・髪色クランプ) / 瞳=GGX増強(濡れた角膜の鋭い反射)。
        // 強度は実機操作盤(fx_params): x=髪, y=瞳。0で各々OFF/通常。
        let spec_ggx = specular * radiance * n_dot_l * select(1.0, camera.fx_params.y, is_eye);
        let spec_hair = min(hair_tint * hair_spec * radiance * n_dot_l * camera.fx_params.x, albedo * 1.5 + vec3<f32>(0.1));
        let spec_term = select(spec_ggx, spec_hair, is_hair);

        // 拡散は per-channel（SSS赤方シフト）、鏡面は上の spec_term（髪/瞳で切替）。
        // 濡れclearcoat(白い水膜艶)も肌/肉のみ。標準材(革/金具/布)には乗せない。
        // シルエット(n_dot_v→0)で Cook-Torrance 分母が発散し 100 倍級の firefly になり、
        // bloom 低解像バッファに乗って四角い白ブロブと化す→寄与ごと上限クランプで遮断。
        let cc_gate = select(0.0, 1.0, is_sss);
        let cc_term = min(clearcoat * radiance * n_dot_l * wet_amt * cc_mult * cc_graze * cc_bead * cc_gate,
                          vec3<f32>(2.5));
        lo = lo + diffuse * radiance * diff_w + spec_term + sss_contrib + cc_term;
    }

    // Emissive（発光）
    let emissive = albedo * emissive_strength;

    // 塗布部に淡い白の底上げ＝影側でも透けて暗くならず「白く濁った不透明な液」に見える(擬似SSS)。
    let opaque_floor = opaque_body * vec3<f32>(0.20, 0.19, 0.18);

    // リムライト（縁光）: 暗い背景からシルエットを起こす暖色の縁。fx_params2.x=強度(0=OFF)。
    // 真っ白化対策＝(1) 既に明るい面には乗せない（暗い縁だけ埋める rim_dark ゲート。松明で
    // 明るい石の縁まで足して白飛び→ACES色転び＝緑被り、を断つ）、(2) 細い縁(高指数)、(3) 色は
    // やや沈めた琥珀＋アルベド混ぜで白飛び回避。瞳は自前の角膜艶があるので除外。
    let lit_lum = dot(ambient + lo + emissive + opaque_floor, vec3<f32>(0.2126, 0.7152, 0.0722));
    let rim_dark = 1.0 - smoothstep(0.12, 0.6, lit_lum); // 明所=0で乗らない／暗い縁だけ埋める
    let rim_w = pow(1.0 - n_dot_v, 4.0) * camera.fx_params2.x * rim_dark;
    let rim_col = vec3<f32>(0.95, 0.82, 0.62) * mix(vec3<f32>(1.0), albedo, 0.35);
    // リムは肌/肉(is_sss)のシルエット用のみ。革帯/金具/布/壁等の標準材に乗せると、暗い面が
    // グレージング角で暖色白に飛ぶ（口枷帯が拘束ポーズで白く見える事故の元）。model_idで遮断。
    let rim = select(0.0, rim_w, is_sss) * rim_col;

    // 溶け縁の焦げ光(琥珀)＝溶かしてる最中(heat>0)だけ。境界の帯を軽く焦がし、内側を淡く発光。
    let melt_glow = melt_edge * vec3<f32>(1.7, 0.55, 0.12);
    let char_dark = 1.0 - melt_edge * 0.45;
    let color = (ambient + lo + emissive + opaque_floor + rim) * char_dark + melt_glow;

    // ACES トーンマッピング
    let mapped = aces_tonemap(color);

    // ガンマ補正 (linear → sRGB)
    let gamma_corrected = pow(mapped, vec3(1.0 / 2.2));

    // ガラス: Fresnelベースの透過アルファ
    // 溶解中は in.color.a を溶け量に転用しているので、生き残った画素は不透明として出す
    // (穴は discard で表現済み。alpha を下げると blend パスで薄まってしまう)。
    var alpha = select(in.color.a, 1.0, dissolve > 0.002) * tex_color.a;
    if (is_transmission) {
        let fresnel = fresnel_schlick(n_dot_v, f0);
        let avg_fresnel = (fresnel.x + fresnel.y + fresnel.z) / 3.0;
        // 透過量 = transmission * (1 - fresnel反射率)
        let transmittance = transmission * (1.0 - avg_fresnel);
        alpha = 1.0 - transmittance;
    }

    // HDR/ポストプロセスモード(resolution.z>=0.5)では tonemap/gamma を合成シェーダに任せ、
    // ここは線形HDRのまま出力する（合成側で二重に tonemap+gamma すると washout=ピンクになる）。
    if (camera.resolution.z >= 0.5) {
        return vec4<f32>(color, alpha);
    }
    return vec4<f32>(gamma_corrected, alpha);
}
