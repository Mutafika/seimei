//! Seimei — UI非依存 3D レンダリングライブラリ
//!
//! wgpu + glam ベースの PBR レンダラー。
//! egui / sabitori など特定の UI フレームワークに依存しない。

pub mod math;
pub mod vertex;
pub mod light;
pub mod mesh;
pub mod ray;
pub mod camera;
pub mod texture;
pub mod pipeline;
pub mod renderer;
pub mod quality;
pub mod ply;
pub mod splat;
pub mod shadow;
pub mod post_process;
pub mod dof;
pub mod ssr;
#[cfg(feature = "gltf")]
pub mod gltf;

// re-exports
pub use math::{Point3, BoundingBox, Transform};
pub use vertex::{GpuVertex, InstanceData, LineVertex};
pub use vertex::{
    MODEL_STANDARD, MODEL_SKIN, MODEL_HAIR, MODEL_EYE, MODEL_WATER, MODEL_FLUID, MODEL_GLASS,
    MODEL_JELLY,
};
pub use light::{Light, LightKind, GpuLight, LightHeader, LightStorageData, LightUniform, MAX_LIGHTS};
pub use mesh::{Vertex, RenderMesh};
pub use ray::{Ray, RayHit};
pub use camera::{Camera, CameraUniform};
pub use texture::{TextureManager, GpuTexture, TextureError, DEFAULT_TEXTURE_ID};
pub use pipeline::{SHADER_SOURCE, SHADER_WITH_SHADOW_SOURCE};
pub use pipeline::{
    create_main_pipeline_with_shadow, create_main_pipeline_with_shadow_msaa,
    create_splat_pipeline,
};
pub use renderer::{Renderer, RendererError, MeshInstance, PointShadowCaster};
pub use quality::{QualityPreset, QualitySettings, ShadowQuality, MsaaSamples};
pub use ply::{PlyLoader, PlyPointCloud, GaussianCloud, GaussianPoint, PlyError};
pub use splat::{GpuSplat, SplatCloudData, SPLAT_SHADER_SOURCE};
pub use shadow::{create_shadow_pipeline, SHADOW_MAP_SIZE};
pub use post_process::{PostProcessPipeline, GBuffer, SsaoPass, BloomPass, EdgeBevelPass, PixelArtPass, PixelArtParams};
pub use dof::{DofPass, DofParams};
pub use ssr::SsrPass;
pub mod procedural;
pub mod shader_lib;
#[cfg(feature = "gltf")]
pub use gltf::{load_gltf, load_gltf_from_bytes, GltfScene, GltfPrimitive, GltfMaterial, GltfError};
