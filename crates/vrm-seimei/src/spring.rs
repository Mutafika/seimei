//! VRM spring-bone (揺れもの) simulator — secondary motion for hair / skirt / etc.
//!
//! Parses VRM 0.x `VRM.secondaryAnimation` and runs the UniVRM verlet step every
//! frame in glTF-**native** space (Y-up, metres — where the bridge's skeleton
//! lives), updating the world transforms of spring joints so the skinning that
//! follows makes them sway. Stateful: each joint's previous tail position persists
//! across frames, which is what produces lag/overshoot.
//!
//! One simplification vs. full UniVRM: every spring joint collides with *all*
//! colliders (the per-group collider assignment is ignored). Fine for a first cut.

use std::collections::HashSet;

use glam::{Mat4, Quat, Vec3};

/// Multiplier on authored collider radii. Kept at 1.0: inflating uniformly pushes
/// the hair off the (already-tuned) head sphere — the front bangs float. If skirt
/// clipping needs help, inflate only the lower-body colliders, not the head.
const COLLIDER_INFLATE: f32 = 1.0;

/// Spring "drama" tuning (global multipliers on the model's authored values).
/// Lower stiffness → wider swings; lower drag → more momentum / overshoot. Too low
/// on either gets floppy or unstable.
const STIFFNESS_SCALE: f32 = 0.45;
const DRAG_SCALE: f32 = 0.8;

/// A simulated spring joint. Order in `SpringSystem::joints` is parent-before-child
/// so a joint's parent world is already finalized when we reach it.
#[derive(Clone)]
struct Joint {
    node: usize,        // glTF node index (also a skin joint of the hair/skirt mesh)
    parent: usize,      // node index of the parent (anchor or a prior spring joint)
    bone_axis: Vec3,    // unit local direction joint -> tail (rest)
    length: f32,        // metres, joint -> tail
    rest_local_rot: Quat,
    local_t: Vec3,
    local_s: Vec3,
    stiffness: f32,
    gravity_dir: Vec3,
    gravity_power: f32,
    drag: f32,
    radius: f32,        // hit radius (metres)
    prev_tail: Vec3,    // world (native)
    cur_tail: Vec3,     // world (native)
}

#[derive(Clone)]
struct Collider {
    node: usize,
    offset: Vec3, // local offset in `node` space (native, metres)
    radius: f32,
}

/// Parsed-and-built spring-bone system for one avatar.
pub struct SpringSystem {
    joints: Vec<Joint>,
    colliders: Vec<Collider>,
    pub enabled: bool,
}

struct RawGroup {
    stiffness: f32,
    gravity_power: f32,
    gravity_dir: Vec3,
    drag: f32,
    hit_radius: f32,
    bones: Vec<usize>,
}

impl SpringSystem {
    /// Build from VRM 0.x `secondaryAnimation` in the glTF extensions JSON. Returns
    /// `None` if there's no spring config (or it's empty).
    pub fn from_vrm0(
        ext: &serde_json::Value,
        nodes_t: &[Vec3],
        nodes_r: &[Quat],
        nodes_s: &[Vec3],
        nodes_parent: &[Option<usize>],
        world_bind: &[Mat4],
    ) -> Option<SpringSystem> {
        let sa = ext.get("VRM")?.get("secondaryAnimation")?;

        // --- bone groups ---
        let groups: Vec<RawGroup> = sa
            .get("boneGroups")?
            .as_array()?
            .iter()
            .filter_map(|g| {
                let f = |k: &str, d: f32| g.get(k).and_then(|v| v.as_f64()).unwrap_or(d as f64) as f32;
                let gd = g.get("gravityDir");
                let gv = |k: &str, d: f32| {
                    gd.and_then(|o| o.get(k)).and_then(|v| v.as_f64()).unwrap_or(d as f64) as f32
                };
                let gravity_dir = Vec3::new(gv("x", 0.0), gv("y", -1.0), gv("z", 0.0));
                let bones: Vec<usize> = g
                    .get("bones")?
                    .as_array()?
                    .iter()
                    .filter_map(|b| b.as_u64().map(|n| n as usize))
                    .collect();
                Some(RawGroup {
                    stiffness: f("stiffiness", 1.0), // NB: VRM 0.x spec misspells it "stiffiness"
                    gravity_power: f("gravityPower", 0.0),
                    gravity_dir: gravity_dir.normalize_or(Vec3::NEG_Y),
                    drag: f("dragForce", 0.4),
                    hit_radius: f("hitRadius", 0.02),
                    bones,
                })
            })
            .collect();

        // --- collider groups (flattened; every joint collides with all) ---
        let mut colliders = Vec::new();
        if let Some(cgs) = sa.get("colliderGroups").and_then(|c| c.as_array()) {
            for cg in cgs {
                let Some(node) = cg.get("node").and_then(|v| v.as_u64()).map(|n| n as usize) else {
                    continue;
                };
                if let Some(cs) = cg.get("colliders").and_then(|c| c.as_array()) {
                    for c in cs {
                        let o = c.get("offset");
                        let ov = |k: &str| o.and_then(|d| d.get(k)).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                        let radius = c.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                        colliders.push(Collider { node, offset: Vec3::new(ov("x"), ov("y"), ov("z")), radius });
                    }
                }
            }
        }

        // --- children adjacency, then build joints per chain (parent-before-child) ---
        let nn = nodes_t.len();
        let mut children = vec![Vec::new(); nn];
        for (n, p) in nodes_parent.iter().enumerate() {
            if let Some(p) = p {
                children[*p].push(n);
            }
        }

        let mut joints = Vec::new();
        let mut seen = HashSet::new();
        for g in &groups {
            for &root in &g.bones {
                if root >= nn {
                    continue;
                }
                // pre-order DFS so a parent is always pushed before its descendants
                let mut stack = vec![root];
                while let Some(node) = stack.pop() {
                    if !seen.insert(node) {
                        continue;
                    }
                    let parent = nodes_parent[node].unwrap_or(node);
                    // tail = first child if any, else continue this bone's own direction
                    let (bone_axis, length) = match children[node].first() {
                        Some(&c) => {
                            let tl = nodes_t[c];
                            let l = tl.length();
                            if l > 1e-6 { (tl / l, l) } else { (Vec3::NEG_Y, 0.07) }
                        }
                        None => {
                            let tl = nodes_t[node];
                            let l = tl.length();
                            if l > 1e-6 { (tl / l, 0.07) } else { (Vec3::NEG_Y, 0.07) }
                        }
                    };
                    let tail_world = world_bind[node].transform_point3(bone_axis * length);
                    joints.push(Joint {
                        node,
                        parent,
                        bone_axis,
                        length,
                        rest_local_rot: nodes_r[node],
                        local_t: nodes_t[node],
                        local_s: nodes_s[node],
                        stiffness: g.stiffness * STIFFNESS_SCALE,
                        gravity_dir: g.gravity_dir,
                        gravity_power: g.gravity_power,
                        drag: (g.drag * DRAG_SCALE).clamp(0.0, 1.0),
                        radius: g.hit_radius,
                        prev_tail: tail_world,
                        cur_tail: tail_world,
                    });
                    for &c in &children[node] {
                        stack.push(c);
                    }
                }
            }
        }

        if joints.is_empty() {
            return None;
        }
        Some(SpringSystem { joints, colliders, enabled: true })
    }

    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    pub fn collider_count(&self) -> usize {
        self.colliders.len()
    }

    /// Advance one step and overwrite `world[node]` for every spring joint. `world`
    /// must be the animation-posed skeleton (native space); the head/skirt anchors
    /// (non-spring parents) are read from it, so the springs follow the body.
    pub fn step(&mut self, world: &mut [Mat4], dt: f32) {
        if !self.enabled || dt <= 0.0 {
            return;
        }
        // collider world centres for this frame
        let centers: Vec<(Vec3, f32)> = self
            .colliders
            .iter()
            .map(|c| (world[c.node].transform_point3(c.offset), c.radius))
            .collect();

        for j in &mut self.joints {
            let parent_world = world[j.parent];
            let (_, parent_rot, _) = parent_world.to_scale_rotation_translation();
            let world_pos = parent_world.transform_point3(j.local_t);
            let rest_world_rot = parent_rot * j.rest_local_rot;
            let rest_dir = rest_world_rot * j.bone_axis;

            // verlet integrate the tail
            let inertia = (j.cur_tail - j.prev_tail) * (1.0 - j.drag);
            let stiff = rest_dir * (j.stiffness * dt);
            let ext = j.gravity_dir * (j.gravity_power * dt);
            let mut next = j.cur_tail + inertia + stiff + ext;
            next = world_pos + (next - world_pos).normalize_or_zero() * j.length;

            // collisions: push the tail out of each sphere, then re-constrain length
            for (center, cr) in &centers {
                let r = cr * COLLIDER_INFLATE + j.radius;
                let d = next - *center;
                let dl = d.length();
                if dl > 1e-9 && dl < r {
                    next = *center + d / dl * r;
                }
            }
            next = world_pos + (next - world_pos).normalize_or_zero() * j.length;

            j.prev_tail = j.cur_tail;
            j.cur_tail = next;

            // turn the tail direction into the joint's world rotation, then write
            // back the joint's world matrix so its children read the updated pose.
            let aim = (next - world_pos).normalize_or_zero();
            let rot_world = if aim.length_squared() > 1e-12 {
                Quat::from_rotation_arc(rest_dir, aim) * rest_world_rot
            } else {
                rest_world_rot
            };
            let local_rot = parent_rot.inverse() * rot_world;
            world[j.node] = parent_world * Mat4::from_scale_rotation_translation(j.local_s, local_rot, j.local_t);
        }
    }
}
