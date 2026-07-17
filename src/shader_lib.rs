//! WGSL の部品置き場 — ノイズ・距離場・レイマーチ。
//!
//! メッシュを積まずに GPU だけで絵を作るとき（背景・煙・溶けた金属・地形）、
//! 毎回おなじものを手で書くことになる部分をここへ集めてある。
//!
//! # なぜ文字列なのか
//!
//! WGSL には `#include` が無い。モジュールを跨いで関数を共有する方法が
//! 言語側に無いので、ソースを連結してから `create_shader_module` に渡すしかない。
//! [`compose`] がその連結を、依存の順序を守った上でやる。
//!
//! # 順序は依存ではない
//!
//! [`RAYMARCH`] は利用者の `map()` を呼ぶが、`map()` を先に書く必要は無い。
//! WGSL のモジュールスコープの宣言はプログラム全体がスコープなので、前方参照は
//! 合法（[仕様][spec]。禁じられているのは再帰だけ）。[`compose`] が順序を保つのは
//! 可読性と、naga のエラー行を [`locate`] で引き直せるようにするためであって、
//! 依存解決のためではない。
//!
//! [spec]: https://www.w3.org/TR/WGSL/#declaration-and-scope
//!
//! ```no_run
//! use seimei::shader_lib::{self, NOISE, SDF, RAYMARCH};
//!
//! const MY_WORLD: &str = r#"
//!     var<private> g_time: f32;
//!     fn map(p: vec3<f32>) -> f32 {
//!         return sd_sphere(p, 1.0) + (fbm3(p * 2.0) - 0.5) * 0.3;
//!     }
//! "#;
//!
//! let src = shader_lib::compose(&[NOISE, SDF, MY_WORLD, RAYMARCH]);
//! ```
//!
//! # 落とし穴
//!
//! - `map()` に引数を足せない。[`RAYMARCH`] は `map(p)` としか呼べないので、
//!   時刻などは `var<private>` か uniform に逃がす。
//! - ノイズで歪めた距離場は距離を過大評価する。`step_scale` を 1.0 のままにすると
//!   面を突き抜けて穴が開く。`march_defaults_warped()` を使うこと。
//! - エラーの行番号は **連結後** のものになる。[`locate`] で元のファイルへ引き直せる。

/// ハッシュ・値ノイズ・fbm。依存なし。
///
/// 提供する関数: `hash21` `hash31` `hash22` `vnoise` `vnoise3`
/// `fbm` `fbm3` `fbm_oct` `fbm3_oct` `fbm_ridged` `over`
pub const NOISE: &str = include_str!("../shaders/lib/noise.wgsl");

/// 符号付き距離関数のプリミティブ・合成・変換。依存なし。
///
/// 提供する関数: `sd_sphere` `sd_box` `sd_round_box` `sd_box_2d` `sd_plane`
/// `sd_torus` `sd_capsule` `sd_cylinder` `sd_shaft_interior`
/// `op_union` `op_sub` `op_intersect` `op_smooth_union` `op_smooth_sub`
/// `op_smooth_intersect` `op_repeat` `rot_x` `rot_y` `rot_z`
pub const SDF: &str = include_str!("../shaders/lib/sdf.wgsl");

/// レイマーチ・法線・ソフトシャドウ・AO。
///
/// **利用者の `fn map(p: vec3<f32>) -> f32` より後ろに置くこと。**
///
/// 提供する関数: `march` `march_defaults` `march_defaults_warped`
/// `calc_normal` `soft_shadow` `ambient_occlusion` `ray_dir` `ray_dir_look_at`
/// と `FULLSCREEN_TRIS`
pub const RAYMARCH: &str = include_str!("../shaders/lib/raymarch.wgsl");

/// 連結後のソースで、ある行がどの断片の何行目だったかを示す。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLoc {
    /// [`compose`] に渡した断片の番号。
    pub part: usize,
    /// その断片の中での行番号（1始まり）。
    pub line: usize,
}

/// WGSL の断片を、依存の順に連結する。
///
/// 各断片の境目にコメントの見出しを挟むので、naga のエラーを目で追える。
/// 行番号から元の位置を引くには [`locate`] を使う。
///
/// 順序は呼び出し側の責任。`map()` を使う断片（[`RAYMARCH`]）は、
/// `map()` を定義する断片より後ろに置くこと。
pub fn compose(parts: &[&str]) -> String {
    let mut out = String::new();
    for (i, part) in parts.iter().enumerate() {
        out.push_str(&format!("// ── seimei::shader_lib part {i} ──\n"));
        out.push_str(part);
        // 断片が改行で終わっていないと、次の断片の1行目と繋がって壊れる。
        if !part.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

/// [`compose`] した結果の行番号 → 元の断片と行番号。
///
/// naga のエラーは連結後の行を指すので、そのままでは元のファイルを探せない。
/// `compose` と同じ `parts` を渡すこと。範囲外なら `None`。
pub fn locate(parts: &[&str], composed_line: usize) -> Option<SourceLoc> {
    // compose は断片ごとに見出しを1行足している。
    let mut cursor = 0usize;
    for (i, part) in parts.iter().enumerate() {
        cursor += 1; // 見出し
        let mut n = part.lines().count();
        if !part.ends_with('\n') {
            // 末尾に補った改行は行を増やさない（最終行がそのまま1行）。
        } else if part.is_empty() {
            n = 0;
        }
        if composed_line > cursor && composed_line <= cursor + n {
            return Some(SourceLoc { part: i, line: composed_line - cursor });
        }
        cursor += n;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_keeps_order_and_separates_parts() {
        let s = compose(&["fn a() {}", "fn b() {}"]);
        let ai = s.find("fn a()").unwrap();
        let bi = s.find("fn b()").unwrap();
        assert!(ai < bi, "渡した順が保たれていない");
        assert!(s.contains("part 0") && s.contains("part 1"));
    }

    #[test]
    fn compose_inserts_newline_between_parts() {
        // 改行で終わらない断片を繋いだとき、次の断片の1行目と融合してはいけない。
        let s = compose(&["fn a() {}", "fn b() {}"]);
        assert!(!s.contains("fn a() {}//"), "断片が改行なしで連結された");
        assert!(s.contains("fn a() {}\n"));
    }

    #[test]
    fn locate_maps_composed_line_back_to_part() {
        let parts = ["one\ntwo\n", "three\nfour\n"];
        let composed = compose(&parts);
        // 目で数えた位置と一致するか。
        let lines: Vec<&str> = composed.lines().collect();
        let idx = lines.iter().position(|l| *l == "three").unwrap() + 1;
        assert_eq!(locate(&parts, idx), Some(SourceLoc { part: 1, line: 1 }));

        let idx2 = lines.iter().position(|l| *l == "two").unwrap() + 1;
        assert_eq!(locate(&parts, idx2), Some(SourceLoc { part: 0, line: 2 }));
    }

    #[test]
    fn locate_rejects_headers_and_out_of_range() {
        let parts = ["one\n"];
        // 1行目は見出しなので、どの断片にも属さない。
        assert_eq!(locate(&parts, 1), None);
        assert_eq!(locate(&parts, 2), Some(SourceLoc { part: 0, line: 1 }));
        assert_eq!(locate(&parts, 999), None);
    }

    #[test]
    fn shipped_parts_are_not_empty() {
        for (name, src) in [("NOISE", NOISE), ("SDF", SDF), ("RAYMARCH", RAYMARCH)] {
            assert!(!src.trim().is_empty(), "{name} が空");
        }
    }

    /// noise と sdf は単体で閉じていること（map に依存しない）。
    /// raymarch だけが map を呼ぶ＝順序制約の根拠なので、そこを固定しておく。
    #[test]
    fn only_raymarch_depends_on_user_map() {
        assert!(!NOISE.contains("map("), "noise が map に依存している");
        assert!(!SDF.contains("map("), "sdf が map に依存している");
        assert!(RAYMARCH.contains("map("), "raymarch が map を呼んでいない");
    }

    // ── 本番と同じ経路で通るか ──────────────────────
    //
    // 「文字列が空でない」だけでは何も保証できない。断片が実際に naga を通り、
    // 型検査まで抜けることを見る。ここが緑でなければこのモジュールは無価値。

    /// naga で構文解析と検証をかける。エラーはそのまま返す。
    fn validate(src: &str) -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(src).map_err(|e| e.to_string())?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
    }

    #[test]
    fn noise_compiles_standalone() {
        validate(NOISE).expect("noise.wgsl 単体が通らない");
    }

    #[test]
    fn sdf_compiles_standalone() {
        validate(SDF).expect("sdf.wgsl 単体が通らない");
    }

    /// 説明どおりの順（noise → sdf → map → raymarch）で通ること。
    #[test]
    fn documented_compose_order_compiles() {
        const USER: &str = r#"
            var<private> g_time: f32;
            fn map(p: vec3<f32>) -> f32 {
                let d = sd_sphere(p, 1.0);
                return d + (fbm3(p * 2.0) - 0.5) * 0.3 + g_time * 0.0;
            }
        "#;
        let src = compose(&[NOISE, SDF, USER, RAYMARCH]);
        validate(&src).expect("説明どおりの順で通らない");
    }

    /// 順序は依存ではない。map を raymarch より後ろに書いても通る。
    ///
    /// WGSL のモジュールスコープの宣言はプログラム全体がスコープで、前方参照は合法
    /// （https://www.w3.org/TR/WGSL/#declaration-and-scope）。
    /// 「呼ぶ側より上に定義が要る」は C の直感で、WGSL には当てはまらない。
    /// doc がそう書いている根拠がこれ。
    #[test]
    fn compose_order_does_not_matter() {
        const USER: &str = "fn map(p: vec3<f32>) -> f32 { return sd_sphere(p, 1.0); }";
        validate(&compose(&[NOISE, SDF, USER, RAYMARCH])).expect("map が先の順で落ちた");
        validate(&compose(&[NOISE, SDF, RAYMARCH, USER])).expect("map が後の順で落ちた");
    }

    /// raymarch は map 無しでは成立しない（単体では落ちる）。
    #[test]
    fn raymarch_alone_fails_without_map() {
        assert!(
            validate(RAYMARCH).is_err(),
            "raymarch が map 無しで通ってしまった"
        );
    }

    /// locate が naga のエラー行を元の断片へ引き直せること。
    /// エラー行番号は連結後のものなので、これができないと debug に使えない。
    #[test]
    fn locate_resolves_a_real_naga_error() {
        const BROKEN: &str = "fn oops() -> f32 { return notdefined; }\n";
        let parts = [NOISE, BROKEN];
        let src = compose(&parts);
        let err = naga::front::wgsl::parse_str(&src).expect_err("壊れた断片が通ってしまった");
        // naga のエラーは連結後の位置を指す。行へ直してから locate に渡す。
        let (loc, _) = err.location(&src).map(|l| (l.line_number as usize, l)).unwrap();
        let found = locate(&parts, loc).expect("連結後の行を元の断片へ引けない");
        assert_eq!(found.part, 1, "壊れているのは2つ目の断片のはず");
        assert_eq!(found.line, 1);
    }
}
