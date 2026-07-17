// noise.wgsl — 値ノイズと fbm。
//
// 手続き的な模様（石・雲・煙・水面・打ち肌）はだいたいこれで作る。
// 単体でコンパイルできる＝依存は無い。
//
// 使い方は `seimei::shader_lib` を参照。ここは他のどのモジュールより先に置く。

// ── ハッシュ ────────────────────────────────────────
//
// sin を使った定番の擬似乱数。統計的にはよくないが、GPU で速く、
// どのドライバでも同じ絵になるという一点で実用的。
// ⚠ 引数が大きい（1e4 以上）と sin の精度が落ちて縞が出る。
//   座標を大きく飛ばすときは fract してから渡すこと。

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn hash31(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453123);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let x = dot(p, vec2<f32>(127.1, 311.7));
    let y = dot(p, vec2<f32>(269.5, 183.3));
    return fract(sin(vec2<f32>(x, y)) * 43758.5453123);
}

// ── 値ノイズ ────────────────────────────────────────

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // 三次のエルミート補間。線形補間だと格子が見える。
    let s = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

/// 3D 値ノイズ。立体を歪ませるとき（溶けた金属・雲）に要る。
fn vnoise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, s.x);
    let x10 = mix(n010, n110, s.x);
    let x01 = mix(n001, n101, s.x);
    let x11 = mix(n011, n111, s.x);
    return mix(mix(x00, x10, s.y), mix(x01, x11, s.y), s.z);
}

// ── fbm ─────────────────────────────────────────────
//
// オクターブを重ねて自然な粗さを作る。返り値はおよそ 0..1（厳密ではない）。
//
// ⚠ 周波数の倍率を 2.0 ちょうどにしないこと。格子が全オクターブで揃って
//   縞や十字が出る。2.02 のように少しずらす。

fn fbm_oct(p: vec2<f32>, octaves: i32) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < octaves; i = i + 1) {
        v = v + amp * vnoise(q);
        q = q * 2.02;
        amp = amp * 0.5;
    }
    return v;
}

fn fbm3_oct(p: vec3<f32>, octaves: i32) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < octaves; i = i + 1) {
        v = v + amp * vnoise3(q);
        q = q * 2.03;
        amp = amp * 0.5;
    }
    return v;
}

fn fbm(p: vec2<f32>) -> f32 {
    return fbm_oct(p, 5);
}

/// 3D の fbm。オクターブが 4 なのは、距離場を歪ませる用途では
/// 5 以上にしてもレイマーチの繰り返し回数に埋もれて見えないため。
fn fbm3(p: vec3<f32>) -> f32 {
    return fbm3_oct(p, 4);
}

/// 尾根状の fbm。山や岩の稜線に。折り返すので谷でなく峰が立つ。
fn fbm_ridged(p: vec2<f32>, octaves: i32) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < octaves; i = i + 1) {
        v = v + amp * (1.0 - abs(vnoise(q) * 2.0 - 1.0));
        q = q * 2.02;
        amp = amp * 0.5;
    }
    return v;
}

// ── 合成 ────────────────────────────────────────────

/// 前を上に重ねる（アルファ合成のカバレッジ版）。
fn over(a: f32, b: f32) -> f32 {
    return a + b * (1.0 - a);
}
