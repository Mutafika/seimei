//! 同梱シェーダが WGSL として通ることを検証する。
//!
//! ⚠ シェーダは実行時までコンパイルされない。このテストが無いと、WGSL の構文エラーや
//!   型の食い違いは「起動したら画面が真っ黒」という形でしか出てこないので、
//!   原因がシェーダなのか、パイプラインなのか、データなのか分からなくなる。
//!
//! ここが落ちたら、パイプラインを疑う前にまずシェーダを疑うこと。

use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};

fn check(name: &str, src: &str) {
    let module = match wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => panic!("{name}: WGSL の構文が通らない:\n{}", e.emit_to_string(src)),
    };
    let mut v = Validator::new(ValidationFlags::all(), Capabilities::all());
    if let Err(e) = v.validate(&module) {
        panic!("{name}: WGSL の検証に落ちた:\n{e:?}");
    }
}

#[test]
fn pbr_compiles() {
    check("pbr.wgsl", seimei::SHADER_SOURCE);
}

/// 空＋体積雲の背景パス。**このテストが無いと、シェーダの構文ミスは起動時 panic でしか
/// 分からない**（実際に `if` の後ろへ return を足して「instructions after return」で落ちた）。
#[test]
fn sky_cloud_compiles() {
    check("sky_cloud.wgsl", include_str!("../shaders/sky_cloud.wgsl"));
}

#[test]
fn refraction_compiles() {
    check("refraction.wgsl", seimei::pipeline::SHADER_REFRACTION_SOURCE);
}
