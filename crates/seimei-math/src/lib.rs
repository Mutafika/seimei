//! seimei-math — seimei の幾何プリミティブ
//!
//! 点 / ベクトル / 変換行列 / バウンディングボックス / レイ。依存は glam と
//! （任意で）serde だけで、**wgpu には一切触れない**。
//!
//! seimei 本体から切り出してあるのは、型だけを使う利用者に描画スタック一式を
//! 引かせないため。BIM の書き出しや干渉判定のように画面を持たない処理が、
//! シェーダーコンパイラや GPU バックエンドをビルドせずに済む。
//!
//! `seimei` 側は `seimei::math` / `seimei::ray` として再エクスポートしているので、
//! 既存のパスはそのまま使える。

pub mod math;
pub mod ray;

pub use math::{BoundingBox, Point3, Transform, Vec3D};
pub use ray::{Ray, RayHit};
