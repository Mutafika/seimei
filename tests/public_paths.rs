//! math / ray を `seimei-math` に切り出したあとも、外部クレートから見える
//! パスが変わっていないことを固定する。
//!
//! ⚠ 再エクスポートの取りこぼしは、クレート内部からは絶対に見つからない。
//!   `src/` の中では `crate::math::Point3` が常に解決してしまうので、
//!   `pub use` を1本落としても本体のビルドもユニットテストも通る。壊れるのは
//!   `seimei::math::Point3` と書いている下流（bamiri / everrusting）だけで、
//!   しかもそれは seimei をリリースしたあとに初めて分かる。
//!
//! ここが落ちたら、切り出しではなく `src/lib.rs` の再エクスポートを疑うこと。

use seimei::math::{BoundingBox, Point3, Transform, Vec3D};
use seimei::ray::{Ray, RayHit};

/// bamiri-core / everrusting が実際に書いているパス（`seimei::math::*`）。
#[test]
fn math_module_path_still_resolves() {
    let p = Point3::new(1.0, 2.0, 3.0);
    let v = Vec3D::new(0.0, 0.0, 1.0);
    let t = Transform::default();
    let b = BoundingBox::default();

    // 型が別クレートへ移っても値として同じように使えること
    assert_eq!(p.x, 1.0);
    assert_eq!(v.z, 1.0);
    let _ = (t, b);
}

/// bamiri-renderer が書いているパス（`seimei::ray::*`）。
#[test]
fn ray_module_path_still_resolves() {
    let r = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3D::new(1.0, 0.0, 0.0));
    assert_eq!(r.origin.x, 0.0);
    let _: Option<RayHit> = None;
}

/// ルート直下の再エクスポート（`seimei::Point3` など）。
#[test]
fn root_reexports_still_resolve() {
    let _: seimei::Point3 = seimei::Point3::new(0.0, 0.0, 0.0);
    let _: seimei::BoundingBox = seimei::BoundingBox::default();
    let _: seimei::Transform = seimei::Transform::default();
    let _: seimei::Ray = seimei::Ray::new(seimei::Point3::new(0.0, 0.0, 0.0), Vec3D::new(1.0, 0.0, 0.0));
}

/// マテリアル ID がルートからも `vertex` からも引けること。
/// `MODEL_BAKED` は #15 でルート再エクスポートに入れ忘れていた。
#[test]
fn material_ids_are_exported_from_the_root() {
    assert_eq!(seimei::MODEL_BAKED, seimei::vertex::MODEL_BAKED);
    assert_eq!(seimei::MODEL_GEL, 8.0);
    assert_eq!(seimei::MODEL_IRIDESCENT, 9.0);
    assert_eq!(seimei::MODEL_BAKED, 10.0);
}
