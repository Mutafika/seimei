// sdf.wgsl — 符号付き距離関数のプリミティブと合成。
//
// 返り値は「その点から形の表面までの距離」で、外が正・中が負。
// 単体でコンパイルできる＝依存は無い。
//
// ⚠ ここの関数はどれも「厳密な距離」を返す。厳密であるうちは
//   レイマーチで距離ぶん丸ごと進んでよい。ノイズを足した瞬間に
//   距離は過大評価になるので、march の step_scale を落とすこと。

// ── プリミティブ ────────────────────────────────────

fn sd_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

/// 直方体。`b` は各軸の半径（中心から面まで）。
fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    // 外側は角までの距離、内側は一番近い面までの距離。
    return length(max(q, vec3<f32>(0.0, 0.0, 0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_round_box(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    return sd_box(p, b - vec3<f32>(r, r, r)) - r;
}

fn sd_box_2d(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0);
}

/// 平面。`n` は正規化された法線、`h` は原点からの距離。
fn sd_plane(p: vec3<f32>, n: vec3<f32>, h: f32) -> f32 {
    return dot(p, n) + h;
}

/// トーラス。`t.x` が輪の半径、`t.y` が管の半径。
fn sd_torus(p: vec3<f32>, t: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

/// 線分 a-b を半径 r で太らせたもの。
fn sd_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

/// y 軸に沿った円柱。`h` は半分の高さ。
fn sd_cylinder(p: vec3<f32>, h: f32, r: f32) -> f32 {
    let d = vec2<f32>(length(p.xz) - r, abs(p.y) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0, 0.0)));
}

/// 矩形の筒の内側から見た、壁までの距離。
///
/// 吹き抜け・部屋・煙突など「囲われた空間の中に居る」形はこれ。
/// `half_inner` は内法の半径、`top` は壁の天端（これより上は開いている）。
///
/// ⚠ 天端で切らないと（＝無限に高い筒にすると）真上からの光しか入らない。
///   斜めの光を入れたい形では必ず切ること。
fn sd_shaft_interior(p: vec3<f32>, half_inner: vec2<f32>, top: f32) -> f32 {
    // x: 内法面までの水平距離（内側で正）、y: 天端からの高さ（上で正）。
    // どちらも「その半空間の外にどれだけ居るか」なので sd_box と同じ式で合成できる。
    let q = vec2<f32>(
        min(half_inner.x - abs(p.x), half_inner.y - abs(p.z)),
        p.y - top,
    );
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0);
}

// ── 合成 ────────────────────────────────────────────

fn op_union(a: f32, b: f32) -> f32 {
    return min(a, b);
}

/// a から b を抜く。
fn op_sub(a: f32, b: f32) -> f32 {
    return max(a, -b);
}

fn op_intersect(a: f32, b: f32) -> f32 {
    return max(a, b);
}

/// 滑らかに繋ぐ。`k` が大きいほど水滴のように融ける。
fn op_smooth_union(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn op_smooth_sub(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (b + a) / k, 0.0, 1.0);
    return mix(a, -b, h) + k * h * (1.0 - h);
}

fn op_smooth_intersect(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) + k * h * (1.0 - h);
}

// ── 変換 ────────────────────────────────────────────
//
// 距離場は「点を動かす」のでなく「空間を逆に動かして」形を配置する。
// 回すときは形を回すのでなく、p に逆回転を掛ける。

fn rot_x(a: f32) -> mat3x3<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, c, s),
        vec3<f32>(0.0, -s, c),
    );
}

fn rot_y(a: f32) -> mat3x3<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat3x3<f32>(
        vec3<f32>(c, 0.0, -s),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(s, 0.0, c),
    );
}

fn rot_z(a: f32) -> mat3x3<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat3x3<f32>(
        vec3<f32>(c, s, 0.0),
        vec3<f32>(-s, c, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
}

/// 空間を繰り返す。1つの形が無限に並ぶ。
///
/// ⚠ これを通した距離場は厳密ではなくなる（隣のセルの形が近いことがある）。
///   繰り返しを使ったら march の step_scale を落とすこと。
fn op_repeat(p: vec3<f32>, period: vec3<f32>) -> vec3<f32> {
    return p - period * round(p / period);
}
