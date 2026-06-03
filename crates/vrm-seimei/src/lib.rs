//! vrm-seimei — the bridge between `vrm-anatomy` (renderer-agnostic VRM SDK / CPU
//! skinning) and `seimei` (wgpu PBR renderer).
//!
//! It loads a `.vrm` through seimei's glTF loader, rebuilds the skeleton from the
//! glTF node hierarchy, applies a pose, runs linear-blend skinning, and hands back
//! posed `seimei::RenderMesh`es ready to render. The GPU stays out of here — the
//! caller (e.g. mearie) registers the textures and renders.
//!
//! # Coordinate spaces
//! seimei's glTF loader bakes Y-up/metres → **Z-up/millimetres** into mesh
//! positions, but leaves node TRS and inverse-bind matrices in glTF-native
//! (Y-up/metres). All skinning runs in **native** space — we un-convert seimei's
//! baked vertices once on load — and convert the skinned result back at the end.
//!
//! # Skeleton
//! World transforms are computed over **every** node (joint or not), because a
//! skin's joint list is often a subset of the skeleton (fingers parented through
//! non-joint `palm.*` nodes). Matrix math uses glam; LBS uses vrm-anatomy.

use std::collections::HashMap;

use glam::{EulerRot, Mat4, Quat, Vec3};
use seimei::gltf::GltfAlphaMode;
use seimei::math::Vec3D;
use seimei::{Point3, RenderMesh, Vertex};
use vrm_anatomy::{cpu_skin_lbs, parse_vrm0, parse_vrmc_vrm, SkinMesh, SkinVertex};
use vrm_anatomy::expression::{
    Expressions, MorphAddressing, parse_vrm0_blend_shapes, parse_vrmc_expressions, resolve_weights,
};
pub use vrm_anatomy::expression::ExpressionPreset;

mod spring;
pub use spring::SpringSystem;

/// Row-major 4×4 as `cpu_skin_lbs` wants it (it reads `m[row][col]`, translation
/// at `m[i][3]`). glam is column-major, so we transpose on the way out.
type RowMat = [[f32; 4]; 4];

// --- coordinate conversions (seimei Z-up/mm <-> glTF-native Y-up/m) ----------
// seimei = (x_n, -z_n, y_n) * 1000  (positions; *1 for directions)

#[inline]
fn pos_seimei_to_native(p: [f32; 3]) -> [f32; 3] {
    [p[0] / 1000.0, p[2] / 1000.0, -p[1] / 1000.0]
}
#[inline]
fn pos_native_to_seimei(p: [f32; 3]) -> [f32; 3] {
    [p[0] * 1000.0, -p[2] * 1000.0, p[1] * 1000.0]
}
#[inline]
fn dir_seimei_to_native(d: [f32; 3]) -> [f32; 3] {
    [d[0], d[2], -d[1]]
}
#[inline]
fn dir_native_to_seimei(d: [f32; 3]) -> [f32; 3] {
    [d[0], -d[2], d[1]]
}

/// glam (column-major) → row-major array for `cpu_skin_lbs`.
fn to_row_major(m: Mat4) -> RowMat {
    let c = m.to_cols_array_2d(); // c[col][row]
    let mut r = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] = c[j][i];
        }
    }
    r
}

/// Base-colour texture bytes for a primitive (tight RGBA8).
pub struct TexData {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// One renderable piece of the avatar. `skin` is in native space, with vertex
/// joint indices in the skin's own order; material data is for the caller.
pub struct VrmPrimitive {
    skin: SkinMesh,
    /// This primitive's own skin joints: glTF node index per joint, the
    /// inverse-bind world matrix per joint, and joint names — all in the
    /// primitive's skin order (matches the vertex joint indices). VRM exports one
    /// skin per mesh chunk, so these are per-primitive, not global.
    joint_nodes: Vec<usize>,
    inv_bind: Vec<Mat4>,
    joint_names: Vec<String>,
    /// glTF mesh this primitive belongs to — expression binds address morphs by
    /// `(mesh, target_index)` (0.x) or `(node→mesh, target_index)` (1.0).
    mesh_index: usize,
    /// Per morph-target vertex position deltas, in **native** space (raw glTF —
    /// same space as `skin` positions, so they add directly). Index = morph target
    /// index, matching the expression bind's `index`.
    morph_deltas: Vec<Vec<[f32; 3]>>,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub alpha_blend: bool,
    pub texture: Option<TexData>,
}

/// A loaded VRM avatar: renderable primitives + the full-node skeleton needed to
/// pose them. Skeleton arrays are indexed by glTF node index; the joint map
/// connects skin-joint order (vertex indices, IBMs) to nodes.
pub struct VrmAvatar {
    primitives: Vec<VrmPrimitive>,

    // Full node skeleton (indexed by glTF node index).
    nodes_t: Vec<Vec3>,
    nodes_r: Vec<Quat>,
    nodes_s: Vec<Vec3>,
    nodes_parent: Vec<Option<usize>>,
    nodes_name: Vec<String>,
    node_order: Vec<usize>, // topological (parent-first) order of node indices

    // Inverse-bind matrices live per-primitive (see VrmPrimitive); they're
    // computed from OUR node chain (not the file's IBMs), guaranteeing
    // `jm = world_posed · inv_bind = I` at bind regardless of file IBM quirks.
    humanoid: HashMap<String, usize>, // VRM humanoid bone name -> glTF node index
    is_vrm0: bool,                    // VRM 0.x faces +Y in seimei space (vs -Y for 1.0)
    spring: Option<SpringSystem>,     // 揺れもの (hair/skirt) secondary motion, if any

    // --- 表情 (facial expressions / blendshapes) ---
    expressions: Expressions,                  // model-authored presets (0.x or 1.0)
    node_mesh: Vec<Option<usize>>,             // glTF node index -> mesh index (for 1.0 binds)
    expr_weights: HashMap<ExpressionPreset, f32>, // currently requested preset weights
}

impl VrmAvatar {
    /// Parse a `.vrm` (glb) and build the avatar. Errors on missing skin/parse.
    pub fn load(bytes: &[u8]) -> Result<VrmAvatar, String> {
        let scene = seimei::load_gltf_from_bytes(bytes).map_err(|e| e.to_string())?;

        // Full node skeleton, indexed by node index (glTF nodes are 0..N).
        let nn = scene.nodes.iter().map(|n| n.index + 1).max().unwrap_or(0);
        let mut nodes_t = vec![Vec3::ZERO; nn];
        let mut nodes_r = vec![Quat::IDENTITY; nn];
        let mut nodes_s = vec![Vec3::ONE; nn];
        let mut nodes_parent = vec![None; nn];
        let mut nodes_name = vec![String::new(); nn];
        for nd in &scene.nodes {
            nodes_t[nd.index] = Vec3::from_array(nd.translation);
            nodes_r[nd.index] = Quat::from_array(nd.rotation); // glTF [x,y,z,w]
            nodes_s[nd.index] = Vec3::from_array(nd.scale);
            nodes_parent[nd.index] = nd.parent;
            nodes_name[nd.index] = nd.name.clone();
        }
        let node_order = topo_order(&nodes_parent);

        // Inverse-bind from OUR node chain (bind = no anim), computed once; each
        // primitive selects the nodes its own skin uses.
        let no_anim = vec![Quat::IDENTITY; nn];
        let world_bind = compute_world(&nodes_t, &nodes_r, &nodes_s, &nodes_parent, &node_order, &no_anim);

        // Humanoid bone map (VRM 1.0 then 0.x): bone name -> glTF node index.
        let (humanoid, is_vrm0) = scene
            .extensions_json
            .as_ref()
            .map(humanoid_bone_nodes)
            .unwrap_or_default();

        // Native-space skin meshes for every skinned primitive, each carrying its
        // OWN skin's joint list (VRM uses one skin per mesh chunk — sharing one
        // global skin scrambles the joint indices of every other chunk).
        let mut primitives = Vec::new();
        for p in &scene.primitives {
            let Some(b) = p.skin.as_ref() else { continue };
            if b.joint_nodes.is_empty() {
                continue;
            }
            let vertices: Vec<SkinVertex> = p
                .mesh
                .vertices
                .iter()
                .enumerate()
                .map(|(i, v)| SkinVertex {
                    position: pos_seimei_to_native([
                        v.position.x as f32,
                        v.position.y as f32,
                        v.position.z as f32,
                    ]),
                    normal: dir_seimei_to_native([
                        v.normal.x as f32,
                        v.normal.y as f32,
                        v.normal.z as f32,
                    ]),
                    uv: v.uv,
                    joints: b.joints_per_vertex[i],
                    weights: normalize_weights(b.weights_per_vertex[i]),
                })
                .collect();
            let joint_nodes = b.joint_nodes.clone();
            let inv_bind: Vec<Mat4> =
                joint_nodes.iter().map(|&node| world_bind[node].inverse()).collect();
            primitives.push(VrmPrimitive {
                skin: SkinMesh { vertices, indices: p.mesh.indices.clone() },
                joint_nodes,
                inv_bind,
                joint_names: b.joint_names.clone(),
                // seimei BAKES morph position deltas into its space (yup_to_zup ×1000),
                // exactly like vertex positions. Un-bake each delta back to native
                // (m, Y-up) with the inverse linear map so it adds onto the native
                // skin positions at the right scale and axis. Per-vertex order matches.
                mesh_index: p.mesh_index,
                morph_deltas: p
                    .morph_targets
                    .iter()
                    .map(|mt| mt.position_deltas.iter().map(|d| pos_seimei_to_native(*d)).collect())
                    .collect(),
                base_color: p.material.base_color,
                metallic: p.material.metallic,
                roughness: p.material.roughness,
                alpha_blend: matches!(p.material.alpha_mode, GltfAlphaMode::Blend),
                texture: p.material.base_color_texture.as_ref().map(|t| TexData {
                    width: t.width,
                    height: t.height,
                    rgba: t.rgba.clone(),
                }),
            });
        }
        if primitives.is_empty() {
            return Err("vrm has no skinned primitive".into());
        }

        // VRM 0.x spring bones (揺れもの), if the model defines any.
        let spring = scene
            .extensions_json
            .as_ref()
            .and_then(|ext| {
                SpringSystem::from_vrm0(ext, &nodes_t, &nodes_r, &nodes_s, &nodes_parent, &world_bind)
            });

        // 表情: normalize VRM 1.0 expressions or 0.x blendShapeMaster into one model.
        let expressions = scene
            .extensions_json
            .as_ref()
            .and_then(|ext| parse_vrmc_expressions(ext).or_else(|| parse_vrm0_blend_shapes(ext)))
            .unwrap_or_default();
        // node index -> mesh index, for VRM 1.0 binds that address morphs by node.
        let mut node_mesh = vec![None; nn];
        for nd in &scene.nodes {
            node_mesh[nd.index] = nd.mesh;
        }

        Ok(VrmAvatar {
            primitives,
            nodes_t,
            nodes_r,
            nodes_s,
            nodes_parent,
            nodes_name,
            node_order,
            humanoid,
            is_vrm0,
            spring,
            expressions,
            node_mesh,
            expr_weights: HashMap::new(),
        })
    }

    /// The renderable primitives (material + texture). Same order/count as `skin`.
    pub fn primitives(&self) -> &[VrmPrimitive] {
        &self.primitives
    }

    /// Skin every primitive for `pose` (a list of VRM humanoid bone name → local
    /// euler `[roll(X), pitch(Y), yaw(Z)]` radians). Empty pose = bind pose.
    /// Returns posed `RenderMesh`es in **seimei** space, in `primitives()` order.
    pub fn skin(&self, pose: &[(&str, [f32; 3])]) -> Vec<RenderMesh> {
        let world = self.world_for_pose(pose);
        self.skin_with_world(&world)
    }

    /// Like `skin`, but also advances the spring-bone (揺れもの) simulation by `dt`
    /// seconds so hair/skirt lag and sway with the body's motion. Stateful — call
    /// once per frame. Falls back to `skin` if the model has no spring config.
    pub fn skin_dynamic(&mut self, pose: &[(&str, [f32; 3])], dt: f32) -> Vec<RenderMesh> {
        let mut world = self.world_for_pose(pose);
        if let Some(spring) = self.spring.as_mut() {
            spring.step(&mut world, dt);
        }
        self.skin_with_world(&world)
    }

    // --- 表情 (facial expressions) -------------------------------------------

    /// The expression presets this model actually defines (a subset of the VRM
    /// standard set — e.g. happy/angry/sad/relaxed, the aa/ih/ou/ee/oh vowels,
    /// blink). Drive them with [`set_expression`].
    pub fn available_presets(&self) -> Vec<ExpressionPreset> {
        self.expressions.presets.keys().copied().collect()
    }

    /// Whether the model defines `preset`.
    pub fn has_expression(&self, preset: ExpressionPreset) -> bool {
        self.expressions.presets.contains_key(&preset)
    }

    /// Set a preset's weight in `0..=1`. `0` clears it. Presets overlay additively
    /// (e.g. a blink over a smile); the next `skin`/`skin_dynamic` blends the morph
    /// deltas in. No-op if the model lacks the preset.
    pub fn set_expression(&mut self, preset: ExpressionPreset, weight: f32) {
        let w = weight.clamp(0.0, 1.0);
        if w <= 0.0 {
            self.expr_weights.remove(&preset);
        } else {
            self.expr_weights.insert(preset, w);
        }
    }

    /// Clear all active expressions (back to the neutral mesh).
    pub fn clear_expressions(&mut self) {
        self.expr_weights.clear();
    }

    /// Accumulate the currently-active presets into per-mesh `(morph_index, weight)`
    /// lists. Empty when no expression is active (the common case → no morph work).
    fn active_morphs(&self) -> HashMap<usize, Vec<(usize, f32)>> {
        let mut out: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        if self.expr_weights.is_empty() {
            return out;
        }
        for (preset, w) in resolve_weights(&self.expressions, &self.expr_weights) {
            if w <= 0.0 {
                continue;
            }
            let Some(expr) = self.expressions.get_preset(preset) else { continue };
            for bind in &expr.morph_target_binds {
                let mesh = match bind.addressing {
                    MorphAddressing::MeshIndex => Some(bind.target),
                    MorphAddressing::NodeIndex => {
                        self.node_mesh.get(bind.target).copied().flatten()
                    }
                };
                let Some(mesh) = mesh else { continue };
                let weight = w * bind.weight;
                if weight != 0.0 {
                    out.entry(mesh).or_default().push((bind.index, weight));
                }
            }
        }
        out
    }

    /// Skin every primitive against an already-computed node world-transform array.
    fn skin_with_world(&self, world: &[Mat4]) -> Vec<RenderMesh> {
        let morphs = self.active_morphs();
        self.primitives
            .iter()
            .map(|prim| {
                // Per-primitive joint matrices: this primitive's skin order.
                let jm: Vec<RowMat> = prim
                    .joint_nodes
                    .iter()
                    .zip(&prim.inv_bind)
                    .map(|(&node, ib)| to_row_major(world[node] * *ib))
                    .collect();
                // Apply active morph deltas to the bind positions (native space)
                // before skinning. No active morph on this mesh → skin as-is.
                let morphed: Option<SkinMesh> = morphs
                    .get(&prim.mesh_index)
                    .filter(|a| !a.is_empty())
                    .map(|active| apply_morphs(&prim.skin, &prim.morph_deltas, active));
                let skin_ref = morphed.as_ref().unwrap_or(&prim.skin);
                let skinned = cpu_skin_lbs(skin_ref, &jm);
                let vertices: Vec<Vertex> = skinned
                    .iter()
                    .map(|sv| {
                        let sp = pos_native_to_seimei(sv.position);
                        let sn = dir_native_to_seimei(sv.normal);
                        Vertex::with_uv(
                            Point3::new(sp[0] as f64, sp[1] as f64, sp[2] as f64),
                            Vec3D::new(sn[0] as f64, sn[1] as f64, sn[2] as f64),
                            sv.uv,
                        )
                    })
                    .collect();
                RenderMesh { vertices, indices: prim.skin.indices.clone() }
            })
            .collect()
    }

    /// VRM humanoid bone names present in this model (for building poses).
    pub fn bone_names(&self) -> impl Iterator<Item = &str> {
        self.humanoid.keys().map(|s| s.as_str())
    }

    /// True for VRM 0.x. In seimei space a VRM 0.x avatar faces **+Y** (toward the
    /// back of where a 1.0 model faces), so the camera must sit on the +Y side.
    pub fn is_vrm0(&self) -> bool {
        self.is_vrm0
    }

    /// Number of simulated spring-bone (揺れもの) joints (0 if the model has none).
    pub fn spring_joints(&self) -> usize {
        self.spring.as_ref().map(|s| s.joint_count()).unwrap_or(0)
    }

    /// Number of spring-bone colliders (sphere count).
    pub fn spring_colliders(&self) -> usize {
        self.spring.as_ref().map(|s| s.collider_count()).unwrap_or(0)
    }

    pub fn spring_enabled(&self) -> bool {
        self.spring.as_ref().map(|s| s.enabled).unwrap_or(false)
    }

    /// Toggle the 揺れもの simulation (no-op if the model has no spring config).
    pub fn set_spring_enabled(&mut self, on: bool) {
        if let Some(s) = self.spring.as_mut() {
            s.enabled = on;
        }
    }

    /// (node index, node name, is-a-skin-joint in any primitive) for a humanoid
    /// bone — debug.
    pub fn debug_bone_info(&self, bone: &str) -> Option<(usize, String, bool)> {
        let node = *self.humanoid.get(bone)?;
        let is_joint = self.primitives.iter().any(|p| p.joint_nodes.contains(&node));
        Some((node, self.nodes_name.get(node).cloned().unwrap_or_default(), is_joint))
    }

    /// A pose that swings each upper arm from its bind direction toward straight
    /// **down** by `fraction` (0 = bind/T-pose, 1 = fully down). Derived from the
    /// actual bone geometry (the arm's bind world direction → down), then mapped
    /// back into the bone's local frame — so it's correct for any rig, no
    /// hand-tuned euler guessing. `optional_extra` lets the caller append more.
    pub fn arms_down_pose(&self, fraction: f32) -> Vec<(&'static str, [f32; 3])> {
        self.arms_pose(fraction, 0.0, 0.0)
    }

    /// Lower the arms toward the sides by `fraction`, then swing each upper arm
    /// front/back by `left_swing` / `right_swing` radians (positive = one way,
    /// negative = the other — opposite arms for a walk). The target is straight
    /// **down** rotated about the left-right (native X) axis by the swing, mapped
    /// back into each bone's local frame.
    pub fn arms_pose(&self, fraction: f32, left_swing: f32, right_swing: f32) -> Vec<(&'static str, [f32; 3])> {
        let no_anim = vec![Quat::IDENTITY; self.nodes_t.len()];
        let world = compute_world(
            &self.nodes_t,
            &self.nodes_r,
            &self.nodes_s,
            &self.nodes_parent,
            &self.node_order,
            &no_anim,
        );
        let mut pose = Vec::new();
        for (upper, lower, swing) in [
            ("leftUpperArm", "leftLowerArm", left_swing),
            ("rightUpperArm", "rightLowerArm", right_swing),
        ] {
            let (Some(&un), Some(&ln)) =
                (self.humanoid.get(upper), self.humanoid.get(lower))
            else {
                continue;
            };
            // Arm direction in world (native Y-up): joint → its child.
            let (_, m, jpos) = world[un].to_scale_rotation_translation();
            let cpos = world[ln].to_scale_rotation_translation().2;
            let arm = cpos - jpos;
            if arm.length_squared() < 1e-9 {
                continue;
            }
            // Target = straight down, swung front/back about the left-right (X) axis.
            let target = Quat::from_axis_angle(Vec3::X, swing) * Vec3::NEG_Y;
            // World rotation arm→target, then expressed in the joint's LOCAL frame:
            // the arm vector is rotated by M (the joint's world orientation), so a
            // world rotation R needs local R_anim = M⁻¹ · R · M.
            let r_world = Quat::IDENTITY
                .slerp(Quat::from_rotation_arc(arm.normalize(), target), fraction.clamp(0.0, 1.0));
            let r_anim = m.inverse() * r_world * m;
            let (yaw, pitch, roll) = r_anim.to_euler(EulerRot::ZYX);
            pose.push((upper, [roll, pitch, yaw]));
        }
        pose
    }

    /// World transform of every node for `pose` (humanoid bone name → local euler
    /// `[roll(X), pitch(Y), yaw(Z)]`, applied Z·Y·X). Unknown bone names are
    /// ignored. Per-primitive `jm[j] = world[joint_nodes[j]] · inv_bind[j]`; at
    /// bind (empty pose) `world = world_bind` so every `jm = I`.
    fn world_for_pose(&self, pose: &[(&str, [f32; 3])]) -> Vec<Mat4> {
        let nn = self.nodes_t.len();
        let mut anim: Vec<Quat> = vec![Quat::IDENTITY; nn];
        for (bone, e) in pose {
            if let Some(&node) = self.humanoid.get(*bone) {
                anim[node] = Quat::from_euler(EulerRot::ZYX, e[2], e[1], e[0]);
            }
        }
        compute_world(
            &self.nodes_t,
            &self.nodes_r,
            &self.nodes_s,
            &self.nodes_parent,
            &self.node_order,
            &anim,
        )
    }

    // --- diagnostics (used by the parity example) ---------------------------

    pub fn debug_n_joints(&self) -> usize {
        self.primitives.iter().map(|p| p.joint_nodes.len()).max().unwrap_or(0)
    }

    /// At bind (zero) pose every joint matrix should be the identity. Returns
    /// `(joint name, translation magnitude [native m], rotation deviation)` over
    /// all primitives, sorted by rotation deviation worst-first.
    pub fn debug_joint_diag(&self) -> Vec<(String, f32, f32)> {
        let world = self.world_for_pose(&[]);
        let mut out: Vec<(String, f32, f32)> = Vec::new();
        for prim in &self.primitives {
            for (j, (&node, ib)) in prim.joint_nodes.iter().zip(&prim.inv_bind).enumerate() {
                let c = (world[node] * *ib).to_cols_array_2d();
                let t = (c[3][0] * c[3][0] + c[3][1] * c[3][1] + c[3][2] * c[3][2]).sqrt();
                let mut rot = 0.0f32;
                for col in 0..3 {
                    for row in 0..3 {
                        let id = if col == row { 1.0 } else { 0.0 };
                        rot = rot.max((c[col][row] - id).abs());
                    }
                }
                out.push((prim.joint_names.get(j).cloned().unwrap_or_default(), t, rot));
            }
        }
        out.sort_by(|a, b| b.2.total_cmp(&a.2));
        out
    }

    /// The (primitive, vertex) most heavily weighted to humanoid `bone` — a good
    /// proxy to track that body part through a pose (e.g. follow the hand).
    pub fn debug_heaviest_vertex(&self, bone: &str) -> Option<(usize, usize)> {
        let node = *self.humanoid.get(bone)?;
        let mut best = None;
        let mut bestw = 0.0f32;
        for (pi, p) in self.primitives.iter().enumerate() {
            let Some(j) = p.joint_nodes.iter().position(|&n| n == node) else { continue };
            for (vi, v) in p.skin.vertices.iter().enumerate() {
                for k in 0..4 {
                    if v.joints[k] as usize == j && v.weights[k] > bestw {
                        bestw = v.weights[k];
                        best = Some((pi, vi));
                    }
                }
            }
        }
        best
    }

    /// Bone names + weights driving primitive `pi`'s vertex `vi`.
    pub fn debug_vertex_bones(&self, pi: usize, vi: usize) -> Vec<(String, f32)> {
        let prim = &self.primitives[pi];
        let v = &prim.skin.vertices[vi];
        (0..4)
            .filter(|&k| v.weights[k] > 1e-6)
            .map(|k| {
                (prim.joint_names.get(v.joints[k] as usize).cloned().unwrap_or_default(), v.weights[k])
            })
            .collect()
    }
}

/// World transform of every node for a given per-node anim rotation. Nodes must
/// be visited parent-first (`order`). Local = `T · (R_bind · R_anim) · S`.
fn compute_world(
    t: &[Vec3],
    r: &[Quat],
    s: &[Vec3],
    parent: &[Option<usize>],
    order: &[usize],
    anim: &[Quat],
) -> Vec<Mat4> {
    let nn = t.len();
    let mut world = vec![Mat4::IDENTITY; nn];
    for &i in order {
        let local = Mat4::from_scale_rotation_translation(s[i], r[i] * anim[i], t[i]);
        world[i] = match parent[i] {
            Some(p) => world[p] * local,
            None => local,
        };
    }
    world
}

/// Pull the humanoid `bone name -> glTF node index` map from VRM extensions,
/// trying VRM 1.0 (`VRMC_vrm`) then VRM 0.x (`VRM`). Returns `(map, is_vrm0)`.
fn humanoid_bone_nodes(ext: &serde_json::Value) -> (HashMap<String, usize>, bool) {
    if let Some(h) = parse_vrmc_vrm(ext) {
        return (h.human_bones.into_iter().map(|(k, v)| (k, v.node)).collect(), false);
    }
    if let Some(h) = parse_vrm0(ext) {
        return (h.human_bones.into_iter().map(|b| (b.bone, b.node)).collect(), true);
    }
    (HashMap::new(), false)
}

/// Normalize the 4 skin weights to sum to 1. `cpu_skin_lbs` blends `Σ wₖ·Mₖ·v`
/// without renormalizing, so an un-normalized vertex collapses toward the origin.
fn normalize_weights(w: [f32; 4]) -> [f32; 4] {
    let s = w[0] + w[1] + w[2] + w[3];
    if s > 1e-6 {
        [w[0] / s, w[1] / s, w[2] / s, w[3] / s]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// Build a morphed copy of `base` by adding each active target's per-vertex
/// position deltas (scaled by weight). `active` is `(morph_index, weight)`.
fn apply_morphs(base: &SkinMesh, deltas: &[Vec<[f32; 3]>], active: &[(usize, f32)]) -> SkinMesh {
    let mut vertices = base.vertices.clone();
    for &(ti, w) in active {
        let Some(td) = deltas.get(ti) else { continue };
        for (v, d) in vertices.iter_mut().zip(td.iter()) {
            v.position[0] += d[0] * w;
            v.position[1] += d[1] * w;
            v.position[2] += d[2] * w;
        }
    }
    SkinMesh { vertices, indices: base.indices.clone() }
}

/// Topologically order node indices so every parent precedes its children (sort
/// by depth — a parent is always strictly shallower).
fn topo_order(parents: &[Option<usize>]) -> Vec<usize> {
    let n = parents.len();
    fn depth_of(parents: &[Option<usize>], j: usize, memo: &mut [Option<usize>]) -> usize {
        if let Some(d) = memo[j] {
            return d;
        }
        memo[j] = Some(0); // cycle guard
        let d = match parents[j] {
            Some(p) => depth_of(parents, p, memo) + 1,
            None => 0,
        };
        memo[j] = Some(d);
        d
    }
    let mut memo = vec![None; n];
    let mut depths = vec![0usize; n];
    for j in 0..n {
        depths[j] = depth_of(parents, j, &mut memo);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&j| depths[j]);
    order
}
