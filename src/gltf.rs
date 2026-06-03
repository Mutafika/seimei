//! glTFインポーター
//!
//! Y-up → Z-up 座標系変換、メートル → ミリメートル スケール変換

use std::collections::HashMap;

use crate::math::{Point3, Vec3D};
use crate::mesh::{RenderMesh, Vertex};
use thiserror::Error;
use tracing::{debug, info, warn};

#[derive(Debug, Error)]
pub enum GltfError {
    #[error("glTFファイルの読み込みに失敗: {0}")]
    FileLoad(String),

    #[error("glTFパースエラー: {0}")]
    Parse(String),

    #[error("メッシュデータが見つかりません")]
    NoMeshData,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GltfAlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone, Debug)]
pub struct GltfMaterial {
    pub name: Option<String>,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub base_color_texture: Option<GltfTextureData>,
    pub alpha_mode: GltfAlphaMode,
}

#[derive(Clone, Debug)]
pub struct GltfTextureData {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct GltfSkinBinding {
    pub joint_names: Vec<String>,
    /// glTF node index of each joint (same order as `joint_names` / IBMs). Needed
    /// to rebuild the skeleton hierarchy — node indices are the unambiguous key
    /// (names can collide), e.g. for CPU skinning in the `vrm-seimei` bridge.
    pub joint_nodes: Vec<usize>,
    pub joints_per_vertex: Vec<[u32; 4]>,
    pub weights_per_vertex: Vec<[f32; 4]>,
    pub inverse_bind_matrices: Vec<[[f32; 4]; 4]>,
}

#[derive(Clone, Debug)]
pub struct GltfMorphTarget {
    pub name: String,
    pub position_deltas: Vec<[f32; 3]>,
    pub normal_deltas: Vec<[f32; 3]>,
}

#[derive(Clone, Debug)]
pub struct GltfPrimitive {
    pub mesh: RenderMesh,
    pub material: GltfMaterial,
    pub morph_targets: Vec<GltfMorphTarget>,
    pub default_morph_weights: Vec<f32>,
    pub skin: Option<GltfSkinBinding>,
}

#[derive(Clone, Debug)]
pub struct GltfNodeInfo {
    pub name: String,
    pub index: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct GltfScene {
    pub primitives: Vec<GltfPrimitive>,
    pub extensions_json: Option<serde_json::Value>,
    pub nodes: Vec<GltfNodeInfo>,
}

/// Y-up → Z-up: (x, y, z) → (x, -z, y)
fn yup_to_zup(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    (x, -z, y)
}

fn yup_to_zup_f32(v: [f32; 3], scale: f32) -> [f32; 3] {
    [v[0] * scale, -v[2] * scale, v[1] * scale]
}

const METER_TO_MM: f64 = 1000.0;
const METER_TO_MM_F32: f32 = 1000.0;

/// glTFファイルからシーンを読み込み
pub fn load_gltf(path: &std::path::Path) -> Result<GltfScene, GltfError> {
    let (document, buffers, images) = gltf::import(path)
        .map_err(|e| GltfError::FileLoad(format!("{}: {}", path.display(), e)))?;
    info!("glTF読み込み: {} (メッシュ数: {})", path.display(), document.meshes().count());
    load_from_document(&document, &buffers, &images)
}

/// glTFバイト列からシーンを読み込み
pub fn load_gltf_from_bytes(bytes: &[u8]) -> Result<GltfScene, GltfError> {
    let (document, buffers, images) = gltf::import_slice(bytes)
        .map_err(|e| GltfError::Parse(e.to_string()))?;
    load_from_document(&document, &buffers, &images)
}

fn load_from_document(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
) -> Result<GltfScene, GltfError> {
    let mut primitives = Vec::new();

    let mesh_skin = build_mesh_skin_map(document);

    for mesh in document.meshes() {
        let mesh_name = mesh.name().unwrap_or("unnamed");
        debug!("メッシュ処理: {} (プリミティブ数: {})", mesh_name, mesh.primitives().count());

        // Each mesh is skinned by the skin on the node that references it. VRM
        // exports one skin per mesh chunk, so a single global skin would interpret
        // every other chunk's JOINTS_0 against the wrong joint list (scrambling).
        let (skin_joint_names, skin_joint_nodes, skin_ibms) = match mesh_skin.get(&mesh.index()) {
            Some(skin) => {
                let (n, ni, ib) = extract_one_skin(skin, buffers);
                (Some(n), Some(ni), Some(ib))
            }
            None => (None, None, None),
        };

        let target_names = extract_target_names(&mesh);
        let default_weights: Vec<f32> = mesh.weights().unwrap_or(&[]).to_vec();

        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                warn!("三角形以外のプリミティブはスキップ: {:?}", primitive.mode());
                continue;
            }

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => { warn!("頂点位置データなし、スキップ"); continue; }
            };

            let normals: Vec<[f32; 3]> = reader.read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            let indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            let skin_binding = read_skin_binding(&primitive, buffers, &positions, &skin_joint_names, &skin_joint_nodes, &skin_ibms);

            let vertices: Vec<Vertex> = positions.iter()
                .zip(normals.iter())
                .zip(uvs.iter())
                .map(|((pos, norm), uv)| {
                    let (px, py, pz) = yup_to_zup(
                        pos[0] as f64 * METER_TO_MM,
                        pos[1] as f64 * METER_TO_MM,
                        pos[2] as f64 * METER_TO_MM,
                    );
                    let (nx, ny, nz) = yup_to_zup(norm[0] as f64, norm[1] as f64, norm[2] as f64);
                    Vertex::with_uv(Point3::new(px, py, pz), Vec3D::new(nx, ny, nz), *uv)
                })
                .collect();

            let render_mesh = RenderMesh { vertices, indices };

            let pbr = primitive.material().pbr_metallic_roughness();
            let base_color_texture = pbr.base_color_texture().and_then(|tex_info| {
                let img_index = tex_info.texture().source().index();
                images.get(img_index).and_then(convert_image_to_rgba)
            });

            let alpha_mode = match primitive.material().alpha_mode() {
                gltf::material::AlphaMode::Opaque => GltfAlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => GltfAlphaMode::Mask,
                gltf::material::AlphaMode::Blend => GltfAlphaMode::Blend,
            };

            let material = GltfMaterial {
                name: primitive.material().name().map(|s| s.to_string()),
                base_color: pbr.base_color_factor(),
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                base_color_texture,
                alpha_mode,
            };

            let morph_targets = read_morph_targets(&primitive, buffers, &target_names);

            primitives.push(GltfPrimitive {
                mesh: render_mesh,
                material,
                morph_targets,
                default_morph_weights: default_weights.clone(),
                skin: skin_binding,
            });
        }
    }

    let nodes = extract_nodes(document);
    let extensions_json = document.extensions().map(|ext| serde_json::Value::Object(ext.clone()));

    info!("glTFプリミティブ読み込み完了: {}個", primitives.len());
    Ok(GltfScene { primitives, extensions_json, nodes })
}

/// Map each mesh index to the skin of a node that references it. VRM exports one
/// skin per mesh chunk; first writer wins if a mesh is instanced by many nodes.
fn build_mesh_skin_map(document: &gltf::Document) -> HashMap<usize, gltf::Skin<'_>> {
    let mut map = HashMap::new();
    for node in document.nodes() {
        if let (Some(mesh), Some(skin)) = (node.mesh(), node.skin()) {
            map.entry(mesh.index()).or_insert(skin);
        }
    }
    map
}

/// Extract one skin's joint names, joint node indices, and inverse-bind matrices.
fn extract_one_skin(
    skin: &gltf::Skin,
    buffers: &[gltf::buffer::Data],
) -> (Vec<String>, Vec<usize>, Vec<[[f32; 4]; 4]>) {
    let names: Vec<String> = skin
        .joints()
        .map(|node| node.name().unwrap_or("unnamed_joint").to_string())
        .collect();
    let node_indices: Vec<usize> = skin.joints().map(|node| node.index()).collect();
    let ibms = skin
        .reader(|buf| Some(&buffers[buf.index()]))
        .read_inverse_bind_matrices()
        .map(|iter| iter.collect())
        .unwrap_or_else(|| {
            vec![
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
                node_indices.len()
            ]
        });
    (names, node_indices, ibms)
}

fn extract_target_names(mesh: &gltf::Mesh) -> Vec<String> {
    mesh.extras().as_ref()
        .and_then(|extras| serde_json::from_str::<serde_json::Value>(extras.get()).ok())
        .and_then(|val| {
            val.get("targetNames").and_then(|tn| tn.as_array().map(|arr| {
                arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
            }))
        })
        .unwrap_or_default()
}

fn read_skin_binding(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    positions: &[[f32; 3]],
    skin_joint_names: &Option<Vec<String>>,
    skin_joint_nodes: &Option<Vec<usize>>,
    skin_ibms: &Option<Vec<[[f32; 4]; 4]>>,
) -> Option<GltfSkinBinding> {
    let joints_opt = primitive.reader(|b| Some(&buffers[b.index()])).read_joints(0);
    let weights_opt = primitive.reader(|b| Some(&buffers[b.index()])).read_weights(0);

    match (joints_opt, weights_opt, skin_joint_names) {
        (Some(joints_iter), Some(weights_iter), Some(joint_names)) => {
            let joints: Vec<[u32; 4]> = joints_iter.into_u16()
                .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
                .collect();
            let weights: Vec<[f32; 4]> = weights_iter.into_f32().collect();

            if joints.len() == positions.len() && weights.len() == positions.len() {
                Some(GltfSkinBinding {
                    joint_names: joint_names.clone(),
                    joint_nodes: skin_joint_nodes.clone().unwrap_or_default(),
                    joints_per_vertex: joints,
                    weights_per_vertex: weights,
                    inverse_bind_matrices: skin_ibms.clone().unwrap_or_default(),
                })
            } else { None }
        }
        _ => None,
    }
}

fn read_morph_targets(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    target_names: &[String],
) -> Vec<GltfMorphTarget> {
    let morph_reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    morph_reader.read_morph_targets()
        .enumerate()
        .map(|(i, (pos_deltas, norm_deltas, _))| {
            let position_deltas: Vec<[f32; 3]> = pos_deltas
                .map(|iter| iter.map(|p| yup_to_zup_f32(p, METER_TO_MM_F32)).collect())
                .unwrap_or_default();
            let normal_deltas: Vec<[f32; 3]> = norm_deltas
                .map(|iter| iter.map(|n| yup_to_zup_f32(n, 1.0)).collect())
                .unwrap_or_default();
            let name = target_names.get(i).cloned().unwrap_or_else(|| format!("morph_{}", i));
            GltfMorphTarget { name, position_deltas, normal_deltas }
        })
        .collect()
}

fn extract_nodes(document: &gltf::Document) -> Vec<GltfNodeInfo> {
    let node_count = document.nodes().count();
    let mut parent_table: Vec<Option<usize>> = vec![None; node_count];
    for node in document.nodes() {
        for child in node.children() {
            parent_table[child.index()] = Some(node.index());
        }
    }

    let mut nodes: Vec<GltfNodeInfo> = document.nodes().map(|node| {
        let (translation, rotation, scale) = match node.transform() {
            gltf::scene::Transform::Decomposed { translation, rotation, scale } => {
                (translation, rotation, scale)
            }
            gltf::scene::Transform::Matrix { matrix } => {
                ([matrix[3][0], matrix[3][1], matrix[3][2]], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0])
            }
        };
        GltfNodeInfo {
            name: node.name().unwrap_or("").to_string(),
            index: node.index(),
            parent: parent_table[node.index()],
            children: node.children().map(|c| c.index()).collect(),
            translation,
            rotation,
            scale,
        }
    }).collect();

    nodes.sort_by_key(|n| n.index);
    nodes
}

fn convert_image_to_rgba(img_data: &gltf::image::Data) -> Option<GltfTextureData> {
    let width = img_data.width;
    let height = img_data.height;
    let rgba = match img_data.format {
        gltf::image::Format::R8G8B8A8 => img_data.pixels.clone(),
        gltf::image::Format::R8G8B8 => {
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for chunk in img_data.pixels.chunks(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            rgba
        }
        gltf::image::Format::R8 => {
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for &val in &img_data.pixels { rgba.extend_from_slice(&[val, val, val, 255]); }
            rgba
        }
        gltf::image::Format::R8G8 => {
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for chunk in img_data.pixels.chunks(2) { rgba.extend_from_slice(&[chunk[0], chunk[1], 0, 255]); }
            rgba
        }
        _ => {
            warn!("未対応の画像フォーマット: {:?}", img_data.format);
            return None;
        }
    };
    Some(GltfTextureData { width, height, rgba })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yup_to_zup() {
        let (x, y, z) = yup_to_zup(0.0, 1.0, 0.0);
        assert!((z - 1.0).abs() < 1e-10);
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
    }

    #[test]
    fn test_meter_to_mm_scale() {
        assert!((1.0 * METER_TO_MM - 1000.0).abs() < 1e-10);
    }
}
