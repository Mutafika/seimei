//! vrm-seimei — VRM × seimei の統合層
//!
//! - seimei の glTF ローダで VRM バイト列を読み込み
//! - vrm-anatomy でヒューマノイドスケルトン構築・スキニング
//! - 座標系変換（Z-up/mm ↔ Y-up/m）を吸収
//! - Mixamo / RPM のボーン名 → VRM ヒューマノイド名のマップを提供
//!
//! 上層（ev-client / transer-renderer-vrm 等）は本クレートを consume するだけで
//! VRM レンダリングに必要な骨アニメ & スキニングを得る。
//!
//! 座標系メモ:
//!   seimei は描画時に Y-up/m → Z-up/mm 変換を頂点に適用済み。
//!   IBM / ノード変換は Y-up/m のまま。スキニングは Y-up/m 空間で行い、
//!   結果を Z-up/mm に変換して seimei に渡す。

use seimei::gltf::{GltfScene, load_gltf_from_bytes};
use seimei::math::{Point3, Vec3D};
use seimei::{RenderMesh, Vertex};
use std::collections::HashMap;
use tracing::info;
use vrm_anatomy::vrm::{parse_vrm0, parse_vrmc_vrm};
use vrm_anatomy::{
    GltfNodeInfo, Skeleton,
    skinning::{SkinMesh, SkinVertex, SkinnedVertex},
};

pub type Mat4 = [[f32; 4]; 4];

/// VRM 読み込み結果。
pub struct VrmModel {
    pub parts: Vec<VrmPart>,
    pub skeleton: Option<Skeleton>,
    /// バインドポーズ頂点（Y-up/m の元座標で保持）
    pub skin_meshes: Vec<SkinMesh>,
    /// ノードのローカル変換（Y-up/m、全ノード）
    pub node_translations: Vec<[f32; 3]>,
    pub node_rotations: Vec<[f32; 4]>,
    pub parents: Vec<Option<usize>>,
    /// IBM（ジョイント順、Y-up/m、行優先に転置済み）
    pub inverse_bind_matrices: Vec<Mat4>,
    /// ジョイントインデックス → ノードインデックス
    pub joint_node_indices: Vec<usize>,
    /// ボーン名 → ジョイントインデックス
    pub bone_name_to_joint: Vec<(String, usize)>,
    /// VRM ヒューマノイドボーン名 → ノードインデックス
    pub vrm_bone_to_node: HashMap<String, usize>,
}

pub struct VrmPart {
    pub name: String,
    pub mesh: RenderMesh,
    pub texture: Option<VrmTexture>,
    pub base_color: [f32; 4],
    pub is_mask: bool,
}

pub struct VrmTexture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

// === 座標変換 ===

/// Z-up/mm → Y-up/m（seimei 変換の逆）。
pub fn zup_mm_to_yup_m(p: [f32; 3]) -> [f32; 3] {
    [p[0] / 1000.0, p[2] / 1000.0, -p[1] / 1000.0]
}

pub fn zup_mm_to_yup_m_normal(n: [f32; 3]) -> [f32; 3] {
    [n[0], n[2], -n[1]]
}

pub fn yup_m_to_zup_mm(p: [f32; 3]) -> [f32; 3] {
    [p[0] * 1000.0, -p[2] * 1000.0, p[1] * 1000.0]
}

pub fn yup_m_to_zup_mm_normal(n: [f32; 3]) -> [f32; 3] {
    [n[0], -n[2], n[1]]
}

// === 行列演算 ===

pub fn mat4_identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut o = [[0.0_f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            o[i][j] = a[i][0] * b[0][j]
                + a[i][1] * b[1][j]
                + a[i][2] * b[2][j]
                + a[i][3] * b[3][j];
        }
    }
    o
}

pub fn mat4_transform_point(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
    ]
}

pub fn mat4_transform_normal(m: &Mat4, n: [f32; 3]) -> [f32; 3] {
    let o = [
        m[0][0] * n[0] + m[0][1] * n[1] + m[0][2] * n[2],
        m[1][0] * n[0] + m[1][1] * n[1] + m[1][2] * n[2],
        m[2][0] * n[0] + m[2][1] * n[1] + m[2][2] * n[2],
    ];
    let len = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
    if len > 1e-8 { [o[0] / len, o[1] / len, o[2] / len] } else { o }
}

pub fn quat_to_mat4(q: [f32; 4]) -> Mat4 {
    let [x, y, z, w] = q;
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    [
        [1.0 - (yy + zz), xy - wz, xz + wy, 0.0],
        [xy + wz, 1.0 - (xx + zz), yz - wx, 0.0],
        [xz - wy, yz + wx, 1.0 - (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn euler_to_mat4(e: [f32; 3]) -> Mat4 {
    let (sr, cr) = e[0].sin_cos();
    let (sp, cp) = e[1].sin_cos();
    let (sy, cy) = e[2].sin_cos();
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
        [-sp, cp * sr, cp * cr, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn translation_mat4(t: [f32; 3]) -> Mat4 {
    [
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// glTF の列優先行列を行優先に転置。
pub fn mat4_transpose(m: Mat4) -> Mat4 {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

// === ノード変換 ===

fn find_node_index(nodes: &[GltfNodeInfo], name: &str) -> Option<usize> {
    nodes.iter().position(|n| n.name == name)
}

fn convert_nodes(seimei_nodes: &[seimei::gltf::GltfNodeInfo]) -> Vec<GltfNodeInfo> {
    seimei_nodes
        .iter()
        .map(|n| GltfNodeInfo {
            name: n.name.clone(),
            index: n.index,
            parent: n.parent,
            children: n.children.clone(),
            translation: n.translation,
            rotation: n.rotation,
            scale: n.scale,
        })
        .collect()
}

// === ロード ===

/// VRM バイト列から `VrmModel` を構築。
pub fn load_vrm_from_bytes(bytes: &[u8]) -> Result<VrmModel, String> {
    info!("VRM 読み込み開始 ({} bytes)", bytes.len());

    let scene: GltfScene =
        load_gltf_from_bytes(bytes).map_err(|e| format!("VRM 読み込み失敗: {e}"))?;

    info!(
        "VRM プリミティブ数: {}, ノード数: {}",
        scene.primitives.len(),
        scene.nodes.len()
    );

    let nodes = convert_nodes(&scene.nodes);

    // VRM 拡張パース → スケルトン構築 + VRM ボーン → ノードマップ
    let mut vrm_bone_to_node: HashMap<String, usize> = HashMap::new();
    let skeleton = build_skeleton(&scene, &nodes, &mut vrm_bone_to_node);

    if let Some(ref skel) = skeleton {
        info!(
            "スケルトン構築完了: {} ボーン, VRM マップ = {}",
            skel.bone_count(),
            vrm_bone_to_node.len()
        );
    }

    let node_translations: Vec<[f32; 3]> = nodes.iter().map(|n| n.translation).collect();
    let node_rotations: Vec<[f32; 4]> = nodes.iter().map(|n| n.rotation).collect();
    let parents: Vec<Option<usize>> = nodes.iter().map(|n| n.parent).collect();

    let skin = scene.primitives.iter().find_map(|p| p.skin.as_ref());

    let inverse_bind_matrices: Vec<Mat4> = skin
        .map(|s| {
            s.inverse_bind_matrices
                .iter()
                .map(|m| mat4_transpose(*m))
                .collect()
        })
        .unwrap_or_default();

    let joint_node_indices: Vec<usize> = skin
        .map(|s| {
            s.joint_names
                .iter()
                .map(|name| find_node_index(&nodes, name).unwrap_or(0))
                .collect()
        })
        .unwrap_or_default();

    info!(
        "ジョイント数: {}, ノード数: {}",
        joint_node_indices.len(),
        nodes.len()
    );

    let bone_name_to_joint: Vec<(String, usize)> = skin
        .map(|s| {
            s.joint_names
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect()
        })
        .unwrap_or_default();

    // パーツ + SkinMesh
    let mut parts = Vec::new();
    let mut skin_meshes = Vec::new();

    for (i, prim) in scene.primitives.iter().enumerate() {
        let name = prim
            .material
            .name
            .clone()
            .unwrap_or_else(|| format!("vrm_part_{i}"));

        let texture = prim.material.base_color_texture.as_ref().map(|tex| VrmTexture {
            width: tex.width,
            height: tex.height,
            rgba: tex.rgba.clone(),
        });

        let skin_mesh = if let Some(skin_data) = &prim.skin {
            let vertices: Vec<SkinVertex> = prim
                .mesh
                .vertices
                .iter()
                .enumerate()
                .map(|(vi, v)| {
                    let pos_zup_mm = [v.position.x as f32, v.position.y as f32, v.position.z as f32];
                    let norm_zup = [v.normal.x as f32, v.normal.y as f32, v.normal.z as f32];
                    SkinVertex {
                        position: zup_mm_to_yup_m(pos_zup_mm),
                        normal: zup_mm_to_yup_m_normal(norm_zup),
                        uv: v.uv,
                        joints: if vi < skin_data.joints_per_vertex.len() {
                            skin_data.joints_per_vertex[vi]
                        } else {
                            [0; 4]
                        },
                        weights: if vi < skin_data.weights_per_vertex.len() {
                            skin_data.weights_per_vertex[vi]
                        } else {
                            [1.0, 0.0, 0.0, 0.0]
                        },
                    }
                })
                .collect();
            Some(SkinMesh {
                vertices,
                indices: prim.mesh.indices.clone(),
            })
        } else {
            None
        };

        skin_meshes.push(skin_mesh.unwrap_or_else(|| SkinMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        }));

        let is_mask = matches!(prim.material.alpha_mode, seimei::gltf::GltfAlphaMode::Mask);

        parts.push(VrmPart {
            name,
            mesh: prim.mesh.clone(),
            texture,
            base_color: prim.material.base_color,
            is_mask,
        });
    }

    info!(
        "VRM 読み込み完了: {} パーツ, スケルトン = {}",
        parts.len(),
        skeleton.is_some()
    );
    Ok(VrmModel {
        parts,
        skeleton,
        skin_meshes,
        node_translations,
        node_rotations,
        parents,
        inverse_bind_matrices,
        joint_node_indices,
        bone_name_to_joint,
        vrm_bone_to_node,
    })
}

fn build_skeleton(
    scene: &GltfScene,
    nodes: &[GltfNodeInfo],
    vrm_bone_to_node: &mut HashMap<String, usize>,
) -> Option<Skeleton> {
    let ext = scene.extensions_json.as_ref()?;
    if let Some(vrmc) = parse_vrmc_vrm(ext) {
        info!("VRM 1.0 検出: {} ボーン", vrmc.human_bones.len());
        for (name, bone) in &vrmc.human_bones {
            vrm_bone_to_node.insert(name.clone(), bone.node);
        }
        return Some(Skeleton::from_vrm(&vrmc, nodes));
    }
    if let Some(vrm0) = parse_vrm0(ext) {
        info!("VRM 0.x 検出: {} ボーン", vrm0.human_bones.len());
        for bone in &vrm0.human_bones {
            vrm_bone_to_node.insert(bone.bone.clone(), bone.node);
        }
        return Some(Skeleton::from_vrm0(&vrm0, nodes));
    }
    scene
        .primitives
        .first()
        .and_then(|p| p.skin.as_ref())
        .map(|skin| Skeleton::from_gltf_skin_joints(nodes, &skin.joint_names))
}

// === ジョイント行列 ===

/// ノード階層をトポロジカル順にトラバースし、ジョイント行列を計算。
///
/// `anim_rotations_per_node[i]` が `Some(euler)` ならそのノードのアニメーション回転、
/// `None` ならバインドポーズの回転をそのまま使う。
pub fn compute_vrm_joint_matrices(
    anim_rotations_per_node: &[Option<[f32; 3]>],
    node_translations: &[[f32; 3]],
    node_rotations: &[[f32; 4]],
    parents: &[Option<usize>],
    inverse_bind_matrices: &[Mat4],
    joint_node_indices: &[usize],
) -> Vec<Mat4> {
    let node_count = node_translations.len();
    let mut world_matrices = vec![mat4_identity(); node_count];
    let mut computed = vec![false; node_count];

    let local_matrices: Vec<Mat4> = (0..node_count)
        .map(|i| {
            let t = translation_mat4(node_translations[i]);
            let r_bind = quat_to_mat4(node_rotations[i]);
            if let Some(Some(euler)) = anim_rotations_per_node.get(i) {
                let r_anim = euler_to_mat4(*euler);
                mat4_mul(&t, &mat4_mul(&r_anim, &r_bind))
            } else {
                mat4_mul(&t, &r_bind)
            }
        })
        .collect();

    fn compute_world(
        i: usize,
        parents: &[Option<usize>],
        local_matrices: &[Mat4],
        world_matrices: &mut [Mat4],
        computed: &mut [bool],
    ) {
        if computed[i] {
            return;
        }
        if let Some(p) = parents[i] {
            if p < local_matrices.len() && !computed[p] {
                compute_world(p, parents, local_matrices, world_matrices, computed);
            }
            if p < local_matrices.len() && computed[p] {
                world_matrices[i] = mat4_mul(&world_matrices[p], &local_matrices[i]);
            } else {
                world_matrices[i] = local_matrices[i];
            }
        } else {
            world_matrices[i] = local_matrices[i];
        }
        computed[i] = true;
    }

    for i in 0..node_count {
        compute_world(i, parents, &local_matrices, &mut world_matrices, &mut computed);
    }

    joint_node_indices
        .iter()
        .enumerate()
        .map(|(j, &node_idx)| {
            let world = if node_idx < node_count {
                &world_matrices[node_idx]
            } else {
                &world_matrices[0]
            };
            let ibm = if j < inverse_bind_matrices.len() {
                &inverse_bind_matrices[j]
            } else {
                return mat4_identity();
            };
            mat4_mul(world, ibm)
        })
        .collect()
}

// === Mixamo / RPM → VRM ボーン名 ===

/// Mixamo / RPM のボーン名を VRM ヒューマノイドボーン名にマップ。
pub fn mixamo_to_vrm_name(name: &str) -> Option<&'static str> {
    match name {
        "Hips" => Some("hips"),
        "Spine" => Some("spine"),
        "Spine1" => Some("chest"),
        "Spine2" => Some("upperChest"),
        "Neck" => Some("neck"),
        "Head" => Some("head"),
        "LeftShoulder" => Some("leftShoulder"),
        "LeftArm" => Some("leftUpperArm"),
        "LeftForeArm" => Some("leftLowerArm"),
        "LeftHand" => Some("leftHand"),
        "RightShoulder" => Some("rightShoulder"),
        "RightArm" => Some("rightUpperArm"),
        "RightForeArm" => Some("rightLowerArm"),
        "RightHand" => Some("rightHand"),
        "LeftUpLeg" => Some("leftUpperLeg"),
        "LeftLeg" => Some("leftLowerLeg"),
        "LeftFoot" => Some("leftFoot"),
        "LeftToeBase" => Some("leftToes"),
        "RightUpLeg" => Some("rightUpperLeg"),
        "RightLeg" => Some("rightLowerLeg"),
        "RightFoot" => Some("rightFoot"),
        "RightToeBase" => Some("rightToes"),
        _ => None,
    }
}

pub fn is_upper_leg(vrm_name: &str) -> bool {
    matches!(vrm_name, "leftUpperLeg" | "rightUpperLeg")
}

pub fn is_arm_bone(vrm_name: &str) -> bool {
    matches!(vrm_name, "leftUpperArm" | "rightUpperArm")
}

/// アニメーションサンプル（ボーン名 + Euler）を VRM ノードインデックス配列にマップ。
///
/// ボーン向きごとの回転軸変換を適用：
///   - 上腿: X 回転を反転（前方スイング）
///   - 上腕: X 回転を Y 回転に置換、右腕は符号反転
///
/// `default_pose` は呼出側が任意に与える「未指定ノードの初期姿勢」。
/// 例えばアームを下げたい場合は `[("leftUpperArm", [0, 0, -1.35])]` を渡す。
pub fn apply_mixamo_animation(
    samples: &[(&str, [f32; 3])],
    vrm_bone_to_node: &HashMap<String, usize>,
    node_count: usize,
    default_pose: &[(&str, [f32; 3])],
) -> Vec<Option<[f32; 3]>> {
    let mut rotations: Vec<Option<[f32; 3]>> = vec![None; node_count];

    for (bone_name, euler) in samples {
        let vrm_name_opt = mixamo_to_vrm_name(bone_name);
        let node_idx = vrm_name_opt
            .and_then(|vn| vrm_bone_to_node.get(vn).copied())
            .or_else(|| vrm_bone_to_node.get(*bone_name).copied());

        if let Some(idx) = node_idx
            && idx < node_count
        {
            let vn = vrm_name_opt.unwrap_or("");
            let converted = if is_upper_leg(vn) {
                [-euler[0], euler[1], euler[2]]
            } else if is_arm_bone(vn) {
                let swing = if vn.starts_with("right") { -euler[0] } else { euler[0] };
                [0.0, swing, 0.0]
            } else {
                *euler
            };
            rotations[idx] = Some(converted);
        }
    }

    for (vn, pose) in default_pose {
        if let Some(&idx) = vrm_bone_to_node.get(*vn)
            && idx < node_count
            && rotations[idx].is_none()
        {
            rotations[idx] = Some(*pose);
        }
    }

    rotations
}

// === スキニング ===

/// CPU スキニング（Y-up/m 空間で実行）→ Z-up/mm 結果を返す。
pub fn skin_and_convert(mesh: &SkinMesh, joint_matrices: &[Mat4]) -> Vec<SkinnedVertex> {
    mesh.vertices
        .iter()
        .map(|v| {
            let mut pos = [0.0_f32; 3];
            let mut norm = [0.0_f32; 3];

            for k in 0..4 {
                let w = v.weights[k];
                if w < 1e-6 {
                    continue;
                }
                let ji = v.joints[k] as usize;
                if ji >= joint_matrices.len() {
                    continue;
                }
                let m = &joint_matrices[ji];
                let p = mat4_transform_point(m, v.position);
                let n = mat4_transform_normal(m, v.normal);
                pos[0] += p[0] * w;
                pos[1] += p[1] * w;
                pos[2] += p[2] * w;
                norm[0] += n[0] * w;
                norm[1] += n[1] * w;
                norm[2] += n[2] * w;
            }

            let pos_zup = yup_m_to_zup_mm(pos);
            let norm_zup = yup_m_to_zup_mm_normal(norm);

            let nlen = (norm_zup[0] * norm_zup[0]
                + norm_zup[1] * norm_zup[1]
                + norm_zup[2] * norm_zup[2])
                .sqrt();
            let norm_final = if nlen > 1e-8 {
                [norm_zup[0] / nlen, norm_zup[1] / nlen, norm_zup[2] / nlen]
            } else {
                norm_zup
            };

            SkinnedVertex {
                position: pos_zup,
                normal: norm_final,
                uv: v.uv,
            }
        })
        .collect()
}

/// スキニング結果を seimei の `RenderMesh` に変換。
pub fn skinned_to_render_mesh(skinned: &[SkinnedVertex], indices: &[u32]) -> RenderMesh {
    let vertices = skinned
        .iter()
        .map(|v| {
            Vertex::with_uv(
                Point3::new(
                    v.position[0] as f64,
                    v.position[1] as f64,
                    v.position[2] as f64,
                ),
                Vec3D::new(
                    v.normal[0] as f64,
                    v.normal[1] as f64,
                    v.normal[2] as f64,
                ),
                v.uv,
            )
        })
        .collect();
    RenderMesh {
        vertices,
        indices: indices.to_vec(),
    }
}
