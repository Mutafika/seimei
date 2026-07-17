# seimei

UI-agnostic 3D rendering library — wgpu + glam.

特定の UI フレームワーク（egui / winit など）に依存しない PBR レンダラー。

## Status

🚧 Work in progress — API は予告なく変更される可能性があります。

## Features

- PBR レンダリングパイプライン（シャドウマッピング、MSAA 対応）
- glTF メッシュロード（`gltf` feature、デフォルト有効）
- プロシージャルメッシュ生成（terrain / water plane / rock）
- ポストプロセス（SSAO / Bloom / DOF / SSR / EdgeBevel）
- Gaussian Splatting（PLY ロード対応、`splat` feature）
- 任意の egui 統合（`egui` feature）

## Usage

`Cargo.toml` に git 依存として追加してください:

```toml
[dependencies]
seimei = { git = "https://github.com/Mutafika/seimei", branch = "master" }
```

再現性を確保するなら `rev` を固定:

```toml
[dependencies]
seimei = { git = "https://github.com/Mutafika/seimei", rev = "<commit-hash>" }
```

### Features

```toml
seimei = { git = "...", default-features = false, features = ["gltf", "egui"] }
```

| Feature        | Default | 内容                                     |
| -------------- | ------- | ---------------------------------------- |
| `gltf`         | ✅      | glTF メッシュロード                      |
| `splat`        |         | Gaussian Splatting レンダリング          |
| `post-process` |         | ポストプロセスパス                       |
| `egui`         |         | egui-wgpu 統合                           |

## Requirements

- Rust edition 2024
- wgpu 24

## License

MIT
