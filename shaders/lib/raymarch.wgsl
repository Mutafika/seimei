// raymarch.wgsl — 距離場のレイマーチ、法線、ソフトシャドウ、AO。
//
// ⚠ これ単体ではコンパイルできない。使う側が同じモジュールのどこかで
//   map を定義しておくこと：
//
//       fn map(p: vec3<f32>) -> f32 { ... }
//
//   置く場所は前でも後ろでもよい。WGSL のモジュールスコープの宣言は
//   プログラム全体がスコープなので、前方参照が合法（禁じられているのは再帰だけ）。
//   `compose()` が順序を保つのは可読性とエラー行の追跡のためであって、
//   依存解決のためではない。
//
// ⚠ map に時刻や温度のような引数を足したくなったら、引数を増やすのではなく
//   `var<private>` か uniform に逃がすこと。march からは `map(p)` としか呼べない
//   （呼び出し側の署名を lib が知る方法が無い）。
//
//       var<private> g_time: f32;
//       fn map(p: vec3<f32>) -> f32 { return sd_sphere(p, 1.0) + sin(g_time); }

struct MarchOpts {
    /// 開始距離。カメラが面の近くに居るときに食い込みを避ける。
    tmin: f32,
    /// これを越えたら当たらなかったことにする。
    tmax: f32,
    /// 表面と見なす距離。小さいほど正確だが、遠景でステップを食う。
    eps: f32,
    max_steps: i32,
    /// 踏み込み係数 0..1。
    ///
    /// ⚠ 1.0 でよいのは map が厳密な距離を返すときだけ。
    ///   fbm を足した距離場や op_repeat を通した距離場は距離を過大評価するので、
    ///   そのまま進むと面を突き抜けて穴が開く。歪ませたら 0.5〜0.7 へ落とすこと。
    step_scale: f32,
}

struct Hit {
    /// 表面までの距離。当たらなければ -1。
    dist: f32,
    /// 使ったステップ数。max_steps に張り付いていたら形が重すぎる合図。
    steps: i32,
}

/// 素直な距離場（プリミティブの合成だけ）向けの既定値。
fn march_defaults() -> MarchOpts {
    return MarchOpts(0.02, 60.0, 0.0015, 96, 1.0);
}

/// ノイズで歪めた距離場向けの既定値。踏み込みを落としてある。
fn march_defaults_warped() -> MarchOpts {
    return MarchOpts(0.02, 60.0, 0.0015, 96, 0.6);
}

fn march(ro: vec3<f32>, rd: vec3<f32>, o: MarchOpts) -> Hit {
    var t = o.tmin;
    for (var i = 0; i < o.max_steps; i = i + 1) {
        let d = map(ro + rd * t);
        if (d < o.eps) {
            return Hit(t, i);
        }
        t = t + d * o.step_scale;
        if (t > o.tmax) {
            return Hit(-1.0, i);
        }
    }
    return Hit(-1.0, o.max_steps);
}

/// 法線。距離場の勾配＝一番急に遠ざかる向き。
///
/// `eps` は形のいちばん細かい特徴より小さく、かつ float の精度より大きく。
/// 大きすぎると角が丸まり、小さすぎると面がざらつく。1e-3 前後が無難。
fn calc_normal(p: vec3<f32>, eps: f32) -> vec3<f32> {
    let e = vec2<f32>(eps, 0.0);
    return normalize(vec3<f32>(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx),
    ));
}

/// ソフトシャドウ。光源へ向けて撃ち返し、途中でどれだけ形に近づいたかで陰らせる。
///
/// `k` が大きいほど硬い影。太陽なら 8〜16、面光源なら 2〜4。
///
/// ⚠ `ro` は面から法線方向へ少し浮かせて渡すこと（`p + n * eps * 8.0` 程度）。
///   面の上から撃つと自分自身に当たって、全面が影になる。
fn soft_shadow(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32, k: f32) -> f32 {
    var res = 1.0;
    var t = tmin;
    for (var i = 0; i < 48; i = i + 1) {
        let d = map(ro + rd * t);
        if (d < 0.0015) {
            return 0.0;
        }
        // d / t ＝ その地点の形を見込む角。小さいほど半影が濃い。
        res = min(res, k * d / t);
        t = t + clamp(d, 0.02, 0.45);
        if (t > tmax) {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

/// 遮蔽。法線方向へ数歩ぶん出て、形がどれだけ near にあるかを見る。
/// 隅を落とすためのもので、これが無いと面がただ平らに並んで立体に見えない。
///
/// `radius` は「どこまでを隣と見なすか」。形の大きさに合わせる。
fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>, radius: f32) -> f32 {
    var occ = 0.0;
    var sca = 1.0;
    for (var i = 0; i < 5; i = i + 1) {
        let h = 0.02 * radius + radius * 0.22 * f32(i);
        occ = occ + (h - map(p + n * h)) * sca;
        sca = sca * 0.72;
    }
    return clamp(1.0 - 2.2 * occ, 0.0, 1.0);
}

/// 全画面を覆う三角形2枚ぶんの頂点。板ポリ1枚に距離場を彫るときの土台。
/// `@builtin(vertex_index)` を 0..6 で描く。
var<private> FULLSCREEN_TRIS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
);

/// 画面座標 → レイの向き。
///
/// `uv` は中心が原点で、y が上、x が ±aspect の範囲。
/// `fov_z` が大きいほど望遠（画角が狭い）。
fn ray_dir(uv: vec2<f32>, fov_z: f32) -> vec3<f32> {
    return normalize(vec3<f32>(uv.x, uv.y, fov_z));
}

/// 注視点を向くレイ。`ro` から `at` を見る。
///
/// ⚠ 引数名に `target` は使えない（WGSL の予約語）。
fn ray_dir_look_at(uv: vec2<f32>, ro: vec3<f32>, at: vec3<f32>, fov_z: f32) -> vec3<f32> {
    let fwd = normalize(at - ro);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up = cross(fwd, right);
    return normalize(fwd * fov_z + right * uv.x + up * uv.y);
}
