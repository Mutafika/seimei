//! VRM spring-bone (揺れもの) simulator — secondary motion for hair / skirt / etc.
//!
//! Parses both VRM **0.x** (`VRM.secondaryAnimation`) and VRM **1.0**
//! (`VRMC_springBone`) into one internal joint/collider model, then runs the UniVRM
//! verlet step every frame in glTF-**native** space (Y-up, metres — where the
//! bridge's skeleton lives), updating the world transforms of spring joints so the
//! skinning that follows makes them sway. Stateful: each joint's previous tail
//! position persists across frames, which is what produces lag/overshoot.
//!
//! Differences the parsers normalize away: 0.x groups bones into chains via the
//! node hierarchy with one shared param set, while 1.0 lists each chain explicitly
//! with per-joint params; 0.x colliders are spheres only, 1.0 adds capsules.
//!
//! One simplification vs. full UniVRM: every spring joint collides with *all*
//! colliders (the per-group / per-spring collider assignment is ignored). Fine for
//! a first cut.

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

/// Wind turbulence. The steady wind is modulated by a non-repeating gust envelope
/// (layered incommensurate sines — no RNG, so the sim stays deterministic) that
/// ranges from near-calm (`WIND_LULL`) up to full gust, while the heading slowly
/// wanders by ±`WIND_WANDER` rad around vertical. Each strand is phase-shifted by
/// `WIND_PHASE_STEP * index` so gusts hit them at slightly different moments.
const WIND_LULL: f32 = 0.15; // envelope floor (0 = dead calm between gusts)
const WIND_WANDER: f32 = 0.5; // heading wander amplitude (rad)
const WIND_PHASE_STEP: f32 = 0.35;

/// Non-repeating gust envelope in ~[0,1] from layered incommensurate sines. The
/// frequencies (≈7s/2.7s/1.2s periods) don't share a common multiple, so the sum
/// never visibly repeats — reads as natural gustiness rather than a pulse.
fn gust_env(t: f32) -> f32 {
    let n = 0.5 * (t * 0.9).sin() + 0.3 * (t * 2.3 + 1.7).sin() + 0.2 * (t * 5.1 + 4.2).sin();
    (n * 0.5 + 0.5).clamp(0.0, 1.0)
}

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
    group: u32,         // index into SpringSystem::group_names (the owning chain)
    prev_tail: Vec3,    // world (native)
    cur_tail: Vec3,     // world (native)
}

/// Collision shape carried in the collider's local `node` space (native, metres).
/// 0.x only emits `Sphere`; 1.0 adds `Capsule` (a sphere swept along a segment).
#[derive(Clone)]
enum ColliderShape {
    Sphere { offset: Vec3, radius: f32 },
    Capsule { offset: Vec3, tail: Vec3, radius: f32 },
}

#[derive(Clone)]
struct Collider {
    node: usize,
    shape: ColliderShape,
}

/// Closest point on segment `a`–`b` to `p` (used to collapse a capsule to the
/// sphere nearest the tail being constrained).
fn closest_on_segment(a: Vec3, b: Vec3, p: Vec3) -> Vec3 {
    let ab = b - a;
    let t = (p - a).dot(ab) / ab.length_squared().max(1e-12);
    a + ab * t.clamp(0.0, 1.0)
}

/// Parsed-and-built spring-bone system for one avatar.
pub struct SpringSystem {
    joints: Vec<Joint>,
    colliders: Vec<Collider>,
    pub enabled: bool,
    /// Steady wind direction (native space, unit). Force = `wind_dir * wind_strength`,
    /// gusted per joint. `wind_strength == 0` means no wind.
    wind_dir: Vec3,
    wind_strength: f32,
    /// Extra steady downward gravity (native -Y) added to every joint on top of the
    /// per-joint `gravity_power`. Many VRM models author `gravityPower≈0` (hair held
    /// only by stiffness), so this lets a host force the strands to hang under
    /// gravity (e.g. a limp/hanging body). 0 = no extra gravity.
    gravity_boost: f32,
    /// Names of the spring chains (vrm1 `springs[].name` / vrm0 group comment), indexed
    /// by `Joint::group`. Lets a host address one labelled chain by name.
    group_names: Vec<String>,
    /// Per-group steady external force (native space) added to every joint in that group,
    /// on top of gravity/wind. Lets a host drive one chain (e.g. a labelled secondary
    /// bone) without disturbing the others. Parallel to `group_names`; ZERO = none.
    group_forces: Vec<Vec3>,
    /// Per-group drag override (parallel to `group_names`). `< 0` = no override (use each
    /// joint's own `drag`). Lets a host temporarily over-damp one chain so it follows a
    /// driving force smoothly instead of whipping/resonating (a soft, low-drag chain
    /// resonates under a steady oscillating force). `< 0` restores the authored drag.
    group_drags: Vec<f32>,
    /// Per-group stiffness override (parallel to `group_names`). `< 0` = no override (use
    /// each joint's own `stiffness`). Lets a host temporarily stiffen one chain so it
    /// resists a driving force elastically (small, proportional deflection + a snappy
    /// return) instead of a soft chain slumping to its length limit. `< 0` restores.
    group_stiffs: Vec<f32>,
    /// Accumulated sim time, for gust phase.
    time: f32,
}

struct RawGroup {
    name: String,
    stiffness: f32,
    gravity_power: f32,
    gravity_dir: Vec3,
    drag: f32,
    hit_radius: f32,
    bones: Vec<usize>,
}

impl SpringSystem {
    /// Wrap parsed joints/colliders, defaulting wind off. Shared by both parsers.
    fn build(joints: Vec<Joint>, colliders: Vec<Collider>, group_names: Vec<String>) -> SpringSystem {
        let group_forces = vec![Vec3::ZERO; group_names.len()];
        let group_drags = vec![-1.0; group_names.len()];
        let group_stiffs = vec![-1.0; group_names.len()];
        SpringSystem {
            joints,
            colliders,
            enabled: true,
            wind_dir: Vec3::X,
            wind_strength: 0.0,
            gravity_boost: 0.0,
            group_names,
            group_drags,
            group_stiffs,
            group_forces,
            time: 0.0,
        }
    }

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
                    name: g.get("comment").and_then(|v| v.as_str()).unwrap_or("").to_string(),
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
                        let shape = ColliderShape::Sphere { offset: Vec3::new(ov("x"), ov("y"), ov("z")), radius };
                        colliders.push(Collider { node, shape });
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
        let group_names: Vec<String> = groups.iter().map(|g| g.name.clone()).collect();
        for (gi, g) in groups.iter().enumerate() {
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
                        group: gi as u32,
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
        Some(SpringSystem::build(joints, colliders, group_names))
    }

    /// Build from VRM 1.0 `VRMC_springBone`. Unlike 0.x, every chain is listed
    /// explicitly (`springs[].joints[]`, root-first) and each joint carries its own
    /// params; colliders may be spheres or capsules. Returns `None` if absent/empty.
    pub fn from_vrm1(
        ext: &serde_json::Value,
        nodes_t: &[Vec3],
        nodes_r: &[Quat],
        nodes_s: &[Vec3],
        nodes_parent: &[Option<usize>],
        world_bind: &[Mat4],
    ) -> Option<SpringSystem> {
        let sb = ext.get("VRMC_springBone")?;
        let nn = nodes_t.len();

        // glTF arrays store vectors as [x, y, z] (vs 0.x's {x, y, z} objects).
        let arr3 = |v: Option<&serde_json::Value>, dflt: Vec3| -> Vec3 {
            v.and_then(|a| a.as_array())
                .map(|a| {
                    let g = |i: usize| a.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                    Vec3::new(g(0), g(1), g(2))
                })
                .unwrap_or(dflt)
        };

        // --- colliders (sphere or capsule, flattened; every joint collides with all) ---
        let mut colliders = Vec::new();
        if let Some(cs) = sb.get("colliders").and_then(|c| c.as_array()) {
            for c in cs {
                let Some(node) = c.get("node").and_then(|v| v.as_u64()).map(|n| n as usize) else {
                    continue;
                };
                let Some(shape) = c.get("shape") else { continue };
                let r = |s: &serde_json::Value| s.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                if let Some(sp) = shape.get("sphere") {
                    colliders.push(Collider {
                        node,
                        shape: ColliderShape::Sphere { offset: arr3(sp.get("offset"), Vec3::ZERO), radius: r(sp) },
                    });
                } else if let Some(cap) = shape.get("capsule") {
                    colliders.push(Collider {
                        node,
                        shape: ColliderShape::Capsule {
                            offset: arr3(cap.get("offset"), Vec3::ZERO),
                            tail: arr3(cap.get("tail"), Vec3::ZERO),
                            radius: r(cap),
                        },
                    });
                }
            }
        }

        // --- springs: each is an explicit, root-first joint chain ---
        let mut joints = Vec::new();
        let mut group_names = Vec::new();
        for spring in sb.get("springs")?.as_array()? {
            let Some(jarr) = spring.get("joints").and_then(|j| j.as_array()) else { continue };
            let gi = group_names.len() as u32;
            group_names.push(spring.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string());
            // (node, stiffness, gravity_power, gravity_dir, drag, hit_radius), in order
            let chain: Vec<(usize, f32, f32, Vec3, f32, f32)> = jarr
                .iter()
                .filter_map(|jj| {
                    let node = jj.get("node")?.as_u64()? as usize;
                    let f = |k: &str, d: f32| jj.get(k).and_then(|v| v.as_f64()).unwrap_or(d as f64) as f32;
                    let gdir = arr3(jj.get("gravityDir"), Vec3::NEG_Y).normalize_or(Vec3::NEG_Y);
                    Some((node, f("stiffness", 1.0), f("gravityPower", 0.0), gdir, f("dragForce", 0.4), f("hitRadius", 0.02)))
                })
                .collect();

            // Each chain node becomes a simulated joint; its tail is the next node in
            // the chain (last joint extends its own bone, matching the 0.x leaf rule).
            for i in 0..chain.len() {
                let (node, stiffness, gravity_power, gravity_dir, drag, hit_radius) = chain[i];
                if node >= nn {
                    continue;
                }
                let parent = nodes_parent[node].unwrap_or(node);
                let (bone_axis, length) = match chain.get(i + 1) {
                    Some(&(child, ..)) if child < nn => {
                        let tl = nodes_t[child]; // child's local translation = node->child offset
                        let l = tl.length();
                        if l > 1e-6 { (tl / l, l) } else { (Vec3::NEG_Y, 0.07) }
                    }
                    _ => {
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
                    stiffness: stiffness * STIFFNESS_SCALE,
                    gravity_dir,
                    gravity_power,
                    drag: (drag * DRAG_SCALE).clamp(0.0, 1.0),
                    radius: hit_radius,
                    group: gi,
                    prev_tail: tail_world,
                    cur_tail: tail_world,
                });
            }
        }

        if joints.is_empty() {
            return None;
        }
        Some(SpringSystem::build(joints, colliders, group_names))
    }

    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    pub fn collider_count(&self) -> usize {
        self.colliders.len()
    }

    /// Set the steady wind (native space). `strength` 0 disables it; the direction is
    /// normalized. Gusts and per-strand desync are applied internally in `step`.
    pub fn set_wind(&mut self, dir: Vec3, strength: f32) {
        self.wind_dir = dir.normalize_or(Vec3::X);
        self.wind_strength = strength.max(0.0);
    }

    pub fn wind_strength(&self) -> f32 {
        self.wind_strength
    }

    /// Extra steady downward (native -Y) gravity added to every joint, on top of the
    /// model's per-joint `gravityPower`. Lets a host force the strands to hang under
    /// gravity even when the model authored ~0 gravity. 0 disables.
    pub fn set_gravity_boost(&mut self, power: f32) {
        self.gravity_boost = power.max(0.0);
    }

    pub fn gravity_boost(&self) -> f32 {
        self.gravity_boost
    }

    /// Apply a steady external force (native space) to every joint whose chain name
    /// contains `name` (case-insensitive), on top of gravity/wind. Lets a host drive
    /// one labelled chain — e.g. orbit a force to slosh a specific secondary bone —
    /// without touching the others. `Vec3::ZERO` clears it. Returns true if any chain
    /// matched.
    pub fn set_group_force(&mut self, name: &str, force: Vec3) -> bool {
        let key = name.to_ascii_lowercase();
        let mut hit = false;
        for (gn, gf) in self.group_names.iter().zip(self.group_forces.iter_mut()) {
            if gn.to_ascii_lowercase().contains(&key) {
                *gf = force;
                hit = true;
            }
        }
        hit
    }

    /// Override the drag (0..1) of every joint whose chain name contains `name`
    /// (case-insensitive). `Some(d)` over-damps the chain so it follows a driving force
    /// smoothly instead of resonating/whipping; `None` restores the authored per-joint
    /// drag. Returns true if any chain matched.
    pub fn set_group_drag(&mut self, name: &str, drag: Option<f32>) -> bool {
        let key = name.to_ascii_lowercase();
        let val = drag.map(|d| d.clamp(0.0, 1.0)).unwrap_or(-1.0);
        let mut hit = false;
        for (gn, gd) in self.group_names.iter().zip(self.group_drags.iter_mut()) {
            if gn.to_ascii_lowercase().contains(&key) {
                *gd = val;
                hit = true;
            }
        }
        hit
    }

    /// Override the stiffness of every joint whose chain name contains `name`
    /// (case-insensitive). `Some(s)` stiffens the chain so it resists a driving force
    /// elastically (smaller, springier deflection + a snappy return); `None` restores
    /// the authored per-joint stiffness. Returns true if any chain matched.
    pub fn set_group_stiffness(&mut self, name: &str, stiffness: Option<f32>) -> bool {
        let key = name.to_ascii_lowercase();
        let val = stiffness.map(|s| s.max(0.0)).unwrap_or(-1.0);
        let mut hit = false;
        for (gn, gs) in self.group_names.iter().zip(self.group_stiffs.iter_mut()) {
            if gn.to_ascii_lowercase().contains(&key) {
                *gs = val;
                hit = true;
            }
        }
        hit
    }

    /// Advance one step and overwrite `world[node]` for every spring joint. `world`
    /// must be the animation-posed skeleton (native space); the head/skirt anchors
    /// (non-spring parents) are read from it, so the springs follow the body.
    pub fn step(&mut self, world: &mut [Mat4], dt: f32) {
        if !self.enabled || dt <= 0.0 {
            return;
        }
        // Wind for this frame (copied out so the &mut self.joints loop can read them).
        let (wind_dir, wind_strength) = (self.wind_dir, self.wind_strength);
        let gravity_boost = self.gravity_boost;
        let group_forces = self.group_forces.clone();
        let group_drags = self.group_drags.clone();
        let group_stiffs = self.group_stiffs.clone();
        self.time += dt;
        let time = self.time;
        // Heading wanders slowly around vertical so it's not a fixed vector.
        let gust_dir = if wind_strength > 0.0 {
            let yaw = WIND_WANDER * ((time * 0.27).sin() + 0.5 * (time * 0.6 + 2.0).sin());
            Quat::from_rotation_y(yaw) * wind_dir
        } else {
            wind_dir
        };
        // collider world geometry for this frame. Spheres collapse to (centre, _, r);
        // capsules keep both endpoints so each joint can pick its nearest point.
        let worlds: Vec<(Vec3, Option<Vec3>, f32)> = self
            .colliders
            .iter()
            .map(|c| {
                let w = world[c.node];
                match &c.shape {
                    ColliderShape::Sphere { offset, radius } => (w.transform_point3(*offset), None, *radius),
                    ColliderShape::Capsule { offset, tail, radius } => {
                        (w.transform_point3(*offset), Some(w.transform_point3(*tail)), *radius)
                    }
                }
            })
            .collect();

        for (idx, j) in self.joints.iter_mut().enumerate() {
            let parent_world = world[j.parent];
            let (_, parent_rot, _) = parent_world.to_scale_rotation_translation();
            let world_pos = parent_world.transform_point3(j.local_t);
            let rest_world_rot = parent_rot * j.rest_local_rot;
            let rest_dir = rest_world_rot * j.bone_axis;

            // verlet integrate the tail. drag は group 上書きがあればそれを使う（揉み等で
            // 一時的に過減衰させ、駆動力へ滑らかに追従させるため）。
            let drag = match group_drags.get(j.group as usize) {
                Some(&gd) if gd >= 0.0 => gd,
                _ => j.drag,
            };
            let inertia = (j.cur_tail - j.prev_tail) * (1.0 - drag);
            // stiffness も group 上書き可（掴み/揉み中だけ硬くして弾力＝抵抗＋速い戻りを出す）。
            let stiffness = match group_stiffs.get(j.group as usize) {
                Some(&gs) if gs >= 0.0 => gs,
                _ => j.stiffness,
            };
            let stiff = rest_dir * (stiffness * dt);
            // per-joint gravity ＋ ホスト指定の追加重力（native -Y, steady）＋ グループ外力。
            let group_force = group_forces.get(j.group as usize).copied().unwrap_or(Vec3::ZERO);
            let ext = j.gravity_dir * (j.gravity_power * dt)
                + Vec3::NEG_Y * (gravity_boost * dt)
                + group_force * dt;
            // wind: gusty envelope (calm↔gust) + wandering heading; each strand is
            // time-offset so they don't all surge together.
            let wind = if wind_strength > 0.0 {
                let env = gust_env(time + idx as f32 * WIND_PHASE_STEP);
                let speed = wind_strength * (WIND_LULL + (1.0 - WIND_LULL) * env);
                gust_dir * (speed * dt)
            } else {
                Vec3::ZERO
            };
            let mut next = j.cur_tail + inertia + stiff + ext + wind;
            next = world_pos + (next - world_pos).normalize_or_zero() * j.length;

            // collisions: push the tail out of each collider, then re-constrain length.
            // A capsule reduces to the sphere at its nearest point to the tail.
            for (a, b, cr) in &worlds {
                let center = match b {
                    None => *a,
                    Some(b) => closest_on_segment(*a, *b, next),
                };
                let r = cr * COLLIDER_INFLATE + j.radius;
                let d = next - center;
                let dl = d.length();
                if dl > 1e-9 && dl < r {
                    next = center + d / dl * r;
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A 3-node skeleton: head(0) → hairRoot(1) → hairTip(2), each child offset 0.1m
    /// down in its parent's space. Returns the (t, r, s, parent, world_bind) arrays.
    fn rig() -> (Vec<Vec3>, Vec<Quat>, Vec<Vec3>, Vec<Option<usize>>, Vec<Mat4>) {
        let t = vec![Vec3::new(0.0, 1.5, 0.0), Vec3::new(0.0, -0.1, 0.0), Vec3::new(0.0, -0.1, 0.0)];
        let r = vec![Quat::IDENTITY; 3];
        let s = vec![Vec3::ONE; 3];
        let parent = vec![None, Some(0), Some(1)];
        // world_bind = parent_world * local TRS, walked root-first
        let mut world = vec![Mat4::IDENTITY; 3];
        for i in 0..3 {
            let local = Mat4::from_scale_rotation_translation(s[i], r[i], t[i]);
            world[i] = match parent[i] {
                Some(p) => world[p] * local,
                None => local,
            };
        }
        (t, r, s, parent, world)
    }

    fn springbone_ext() -> serde_json::Value {
        json!({
            "VRMC_springBone": {
                "specVersion": "1.0",
                "colliders": [
                    { "node": 0, "shape": { "sphere": { "offset": [0.0, 0.0, 0.0], "radius": 0.08 } } },
                    { "node": 0, "shape": { "capsule": { "offset": [0.0, 0.0, 0.0], "tail": [0.0, -0.2, 0.0], "radius": 0.05 } } }
                ],
                "colliderGroups": [{ "name": "head", "colliders": [0, 1] }],
                "springs": [{
                    "name": "hair",
                    "joints": [
                        { "node": 1, "hitRadius": 0.02, "stiffness": 1.0, "gravityPower": 0.5, "gravityDir": [0.0, -1.0, 0.0], "dragForce": 0.4 },
                        { "node": 2, "hitRadius": 0.02, "stiffness": 1.0, "gravityPower": 0.5, "gravityDir": [0.0, -1.0, 0.0], "dragForce": 0.4 }
                    ],
                    "colliderGroups": [0]
                }]
            }
        })
    }

    #[test]
    fn parses_vrm1_springs_and_both_collider_shapes() {
        let (t, r, s, parent, world) = rig();
        let sys = SpringSystem::from_vrm1(&springbone_ext(), &t, &r, &s, &parent, &world)
            .expect("VRMC_springBone should build a system");
        // both chain nodes become joints; sphere + capsule both parsed
        assert_eq!(sys.joint_count(), 2);
        assert_eq!(sys.collider_count(), 2);
        // chain wiring: joint 0 anchors to the head (node 0), joint 1 to its parent (node 1)
        assert_eq!(sys.joints[0].parent, 0);
        assert_eq!(sys.joints[1].parent, 1);
        // STIFFNESS_SCALE / DRAG_SCALE applied from the per-joint params
        assert!((sys.joints[0].stiffness - STIFFNESS_SCALE).abs() < 1e-6);
        assert!((sys.joints[0].drag - (0.4 * DRAG_SCALE)).abs() < 1e-6);
    }

    #[test]
    fn from_vrm1_returns_none_without_extension() {
        let (t, r, s, parent, world) = rig();
        assert!(SpringSystem::from_vrm1(&json!({ "VRM": {} }), &t, &r, &s, &parent, &world).is_none());
    }

    #[test]
    fn sideways_gravity_swings_tail() {
        // Hair hangs straight down; a *sideways* gravity must bend it toward +X
        // (vertical gravity on a vertical strand is a degenerate equilibrium).
        let (t, r, s, parent, world_bind) = rig();
        let mut ext = springbone_ext();
        for j in ext["VRMC_springBone"]["springs"][0]["joints"].as_array_mut().unwrap() {
            j["gravityDir"] = json!([1.0, 0.0, 0.0]);
            j["gravityPower"] = json!(1.0);
        }
        let mut sys = SpringSystem::from_vrm1(&ext, &t, &r, &s, &parent, &world_bind).unwrap();
        let tail0 = sys.joints[1].cur_tail;
        let mut world = world_bind.clone();
        for _ in 0..60 {
            sys.step(&mut world, 1.0 / 60.0);
        }
        let tail1 = sys.joints[1].cur_tail;
        assert!(tail1.is_finite());
        assert!(tail1.x > tail0.x + 1e-3, "tail should swing toward +X under sideways gravity");
    }

    #[test]
    fn wind_blows_strands_and_toggles_off() {
        // Gravity-free strand: with no wind it stays put; +X wind blows it toward +X.
        let (t, r, s, parent, world_bind) = rig();
        let mut ext = springbone_ext();
        for j in ext["VRMC_springBone"]["springs"][0]["joints"].as_array_mut().unwrap() {
            j["gravityPower"] = json!(0.0);
        }
        let step60 = |sys: &mut SpringSystem| {
            let mut world = world_bind.clone();
            for _ in 0..60 {
                sys.step(&mut world, 1.0 / 60.0);
            }
        };

        // no wind → essentially static
        let mut calm = SpringSystem::from_vrm1(&ext, &t, &r, &s, &parent, &world_bind).unwrap();
        assert_eq!(calm.wind_strength(), 0.0);
        let tip0 = calm.joints[1].cur_tail;
        step60(&mut calm);
        assert!((calm.joints[1].cur_tail - tip0).length() < 1e-4, "no wind, no gravity → no drift");

        // +X wind → tip is pushed toward +X
        let mut windy = SpringSystem::from_vrm1(&ext, &t, &r, &s, &parent, &world_bind).unwrap();
        windy.set_wind(Vec3::X, 2.0);
        assert!(windy.wind_strength() > 0.0);
        step60(&mut windy);
        let tip = windy.joints[1].cur_tail;
        assert!(tip.is_finite());
        assert!(tip.x > tip0.x + 1e-3, "wind should blow the tip toward +X");
    }

    #[test]
    fn group_force_drives_named_chain_and_clears() {
        // Gravity-free "hair" chain: a +X group force on "hair" pushes the tip toward
        // +X; a non-matching name is a no-op; ZERO restores rest.
        let (t, r, s, parent, world_bind) = rig();
        let mut ext = springbone_ext();
        for j in ext["VRMC_springBone"]["springs"][0]["joints"].as_array_mut().unwrap() {
            j["gravityPower"] = json!(0.0);
        }
        let step60 = |sys: &mut SpringSystem| {
            let mut world = world_bind.clone();
            for _ in 0..60 {
                sys.step(&mut world, 1.0 / 60.0);
            }
        };

        let mut sys = SpringSystem::from_vrm1(&ext, &t, &r, &s, &parent, &world_bind).unwrap();
        let tip0 = sys.joints[1].cur_tail;

        // unknown name → no match, no motion
        assert!(!sys.set_group_force("skirt", Vec3::X * 5.0));
        step60(&mut sys);
        assert!((sys.joints[1].cur_tail - tip0).length() < 1e-4, "non-matching name must not move anything");

        // matching name (case-insensitive) → tip pushed toward +X
        assert!(sys.set_group_force("HAIR", Vec3::X * 5.0));
        step60(&mut sys);
        let tip = sys.joints[1].cur_tail;
        assert!(tip.is_finite() && tip.x > tip0.x + 1e-3, "group force should push the named chain toward +X");

        // ZERO clears → no further +X drift accumulates (force is truly gone)
        assert!(sys.set_group_force("hair", Vec3::ZERO));
        let x_after_clear = sys.joints[1].cur_tail.x;
        step60(&mut sys);
        assert!(
            sys.joints[1].cur_tail.x <= x_after_clear + 1e-4,
            "after clearing, the tip must not keep being pushed toward +X"
        );
    }

    #[test]
    fn group_drag_override_matches_and_clears() {
        let (t, r, s, parent, world_bind) = rig();
        let mut sys = SpringSystem::from_vrm1(&springbone_ext(), &t, &r, &s, &parent, &world_bind).unwrap();
        let authored = sys.joints[0].drag;
        // unknown name → no match, authored drag is used
        assert!(!sys.set_group_drag("skirt", Some(0.9)));
        assert!((effective_drag(&sys, 0) - authored).abs() < 1e-6);
        // matching name → override is used
        assert!(sys.set_group_drag("hair", Some(0.9)));
        assert!((effective_drag(&sys, 0) - 0.9).abs() < 1e-6);
        // None restores the authored per-joint drag
        assert!(sys.set_group_drag("hair", None));
        assert!((effective_drag(&sys, 0) - authored).abs() < 1e-6);
    }

    // Mirror the solver's drag selection so the override is testable without stepping.
    fn effective_drag(sys: &SpringSystem, joint: usize) -> f32 {
        let j = &sys.joints[joint];
        match sys.group_drags.get(j.group as usize) {
            Some(&gd) if gd >= 0.0 => gd,
            _ => j.drag,
        }
    }

    #[test]
    fn group_stiffness_override_matches_and_clears() {
        let (t, r, s, parent, world_bind) = rig();
        let mut sys = SpringSystem::from_vrm1(&springbone_ext(), &t, &r, &s, &parent, &world_bind).unwrap();
        let authored = sys.joints[0].stiffness;
        // unknown name → no match, authored stiffness is used
        assert!(!sys.set_group_stiffness("skirt", Some(2.0)));
        assert!((effective_stiffness(&sys, 0) - authored).abs() < 1e-6);
        // matching name → override is used
        assert!(sys.set_group_stiffness("hair", Some(2.0)));
        assert!((effective_stiffness(&sys, 0) - 2.0).abs() < 1e-6);
        // None restores the authored per-joint stiffness
        assert!(sys.set_group_stiffness("hair", None));
        assert!((effective_stiffness(&sys, 0) - authored).abs() < 1e-6);
    }

    fn effective_stiffness(sys: &SpringSystem, joint: usize) -> f32 {
        let j = &sys.joints[joint];
        match sys.group_stiffs.get(j.group as usize) {
            Some(&gs) if gs >= 0.0 => gs,
            _ => j.stiffness,
        }
    }
}
