//! 2a gate + pose exploration. Run:
//!   cargo run -p vrm-seimei --example parity -- /path/to/model.vrm
//!
//! 1. Zero-pose skinning must reproduce the bind pose (proves the pipeline +
//!    coordinate reconciliation are correct, numerically — no eyeballs needed).
//! 2. Brute-forces which local axis lowers each upper arm, so we can author a
//!    natural rest pose offline.

use seimei::RenderMesh;
use vrm_seimei::VrmAvatar;

fn max_delta(a: &[RenderMesh], b: &[&RenderMesh]) -> f64 {
    let mut maxd = 0.0f64;
    for (ma, mb) in a.iter().zip(b) {
        for (va, vb) in ma.vertices.iter().zip(&mb.vertices) {
            let dx = va.position.x - vb.position.x;
            let dy = va.position.y - vb.position.y;
            let dz = va.position.z - vb.position.z;
            maxd = maxd.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    maxd
}

/// AABB over all meshes (seimei space: Z-up, mm).
fn aabb(meshes: &[RenderMesh]) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for m in meshes {
        for v in &m.vertices {
            let p = [v.position.x, v.position.y, v.position.z];
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
    }
    (min, max)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: parity <file.vrm>");
    let bytes = std::fs::read(&path).expect("read vrm");

    let scene = seimei::load_gltf_from_bytes(&bytes).expect("seimei parse");
    let avatar = VrmAvatar::load(&bytes).expect("vrm-seimei load");

    // How many distinct skins? (Multiple skins break the single-binding bridge.)
    println!("primitives: {}", scene.primitives.len());
    let mut sigs: Vec<(usize, usize)> = Vec::new();
    for (i, p) in scene.primitives.iter().enumerate() {
        match &p.skin {
            Some(s) => {
                let n = s.joint_nodes.len();
                let head: Vec<usize> = s.joint_nodes.iter().take(4).copied().collect();
                println!("  prim {i}: skin joints {n}  head {head:?}");
                sigs.push((n, s.joint_nodes.first().copied().unwrap_or(0)));
            }
            None => println!("  prim {i}: no skin"),
        }
    }
    sigs.sort();
    sigs.dedup();
    println!("  => distinct (len, first-node) skin signatures: {}", sigs.len());

    println!("joints: {}  vrm0: {}", avatar.debug_n_joints(), avatar.is_vrm0());
    let mut bones: Vec<&str> = avatar.bone_names().collect();
    bones.sort();
    println!("humanoid bones ({}): {bones:?}", bones.len());

    // --- gate 1: zero-pose parity ---
    let bind = avatar.skin(&[]);
    let expected: Vec<&RenderMesh> = scene
        .primitives
        .iter()
        .filter(|p| p.skin.is_some())
        .map(|p| &p.mesh)
        .collect();
    assert_eq!(bind.len(), expected.len(), "primitive count mismatch");
    let d = max_delta(&bind, &expected);
    println!("\n[gate 1] zero-pose max vertex delta vs bind: {d:.4} mm");
    println!("  => {}", if d < 1.0 { "PASS (pipeline correct)" } else { "FAIL" });

    // Which joints are non-identity at bind pose (the parity breakers)?
    let diag = avatar.debug_joint_diag();
    println!("  worst joints at bind (rot dev / trans mm / name):");
    for (name, t, rot) in diag.iter().take(12) {
        println!("    rot {rot:7.4}  trans {t:8.2}mm  {name}");
    }

    // Trace the single worst vertex back to its bones.
    let mut worst = (0usize, 0usize, 0.0f64);
    for (mi, (ma, mb)) in bind.iter().zip(&expected).enumerate() {
        for (vi, (va, vb)) in ma.vertices.iter().zip(&mb.vertices).enumerate() {
            let dd = ((va.position.x - vb.position.x).powi(2)
                + (va.position.y - vb.position.y).powi(2)
                + (va.position.z - vb.position.z).powi(2))
            .sqrt();
            if dd > worst.2 {
                worst = (mi, vi, dd);
            }
        }
    }
    println!(
        "  worst vertex: prim {} vert {} delta {:.1}mm  bones: {:?}",
        worst.0,
        worst.1,
        worst.2,
        avatar.debug_vertex_bones(worst.0, worst.1)
    );

    let (bmin, bmax) = aabb(&bind);
    println!("  bind AABB min {bmin:?} max {bmax:?}");

    // --- which humanoid bones actually DEFORM the mesh? (control/deform split) ---
    println!("\n[deform probe] max vertex delta when each bone is rotated 1 rad:");
    for bone in [
        "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
        "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
        "leftUpperLeg", "leftLowerLeg", "rightUpperLeg", "rightLowerLeg",
    ] {
        let posed = avatar.skin(&[(bone, [0.0, 1.0, 0.0])]);
        let d = max_delta(&posed, &expected);
        let flag = if d > 1.0 { "deforms" } else { "NO EFFECT" };
        println!("    {bone:14} Δ {d:8.1} mm  {flag}");
    }

    // Why do legs do nothing? Dump the leg/arm bone→node mapping, and rotate the
    // leg bones on ALL THREE axes (the probe above only tried Y — that could be
    // the bone's twist axis, which barely moves the mesh).
    println!("\n[bone mapping] bone -> (node, name, is-skin-joint):");
    for b in ["leftUpperArm", "leftUpperLeg", "leftLowerLeg", "leftFoot", "hips", "spine"] {
        println!("    {b:14} {:?}", avatar.debug_bone_info(b));
    }
    println!("[leg axis check] rotate each leg bone 1 rad on each local axis:");
    for b in ["leftUpperLeg", "leftLowerLeg"] {
        for (ax, l) in [([1.0f32, 0., 0.], "X"), ([0., 1., 0.], "Y"), ([0., 0., 1.], "Z")] {
            let d = max_delta(&avatar.skin(&[(b, ax)]), &expected);
            println!("    {b:14} {l}  Δ {d:8.1} mm");
        }
    }

    // What ARE the legs weighted to? Find the lowest (foot) and mid-low (knee)
    // vertices in the bind mesh and print their bones — reveals the real leg
    // deform bone names (vs the humanoid map's unused LeftUpLeg).
    {
        let mut lowest = (0usize, 0usize, f64::MAX);
        let mut knee = (0usize, 0usize, f64::MAX, 380.0f64); // nearest to z≈380mm
        for (mi, m) in bind.iter().enumerate() {
            for (vi, v) in m.vertices.iter().enumerate() {
                if v.position.z < lowest.2 {
                    lowest = (mi, vi, v.position.z);
                }
                if (v.position.z - knee.3).abs() < (knee.2 - knee.3).abs() {
                    knee = (mi, vi, v.position.z, knee.3);
                }
            }
        }
        println!("\n[leg weights] lowest vert z={:.0}: {:?}", lowest.2, avatar.debug_vertex_bones(lowest.0, lowest.1));
        println!("[leg weights] knee-ish vert z={:.0}: {:?}", knee.2, avatar.debug_vertex_bones(knee.0, knee.1));
    }

    // Does the built-in walk actually move the mesh, with the viewer's remap?
    {
        use vrm_anatomy::animation::AnimationClip;
        let walk = AnimationClip::walk_cycle();
        let remap = |n: &str| -> &'static str {
            match n {
                "Hips" => "hips",
                "LeftUpLeg" => "leftUpperLeg",
                "RightUpLeg" => "rightUpperLeg",
                "LeftLeg" => "leftLowerLeg",
                "RightLeg" => "rightLowerLeg",
                "LeftFoot" => "leftFoot",
                "RightFoot" => "rightFoot",
                "LeftArm" => "leftUpperArm",
                "RightArm" => "rightUpperArm",
                "LeftForeArm" => "leftLowerArm",
                "RightForeArm" => "rightLowerArm",
                _ => "__unmapped__",
            }
        };
        println!("\n[walk check] max vertex delta vs bind at each phase:");
        for t in [0.0f32, 0.25, 0.5, 0.75] {
            let raw = walk.sample(t);
            let pose: Vec<(&str, [f32; 3])> = raw.iter().map(|(n, r)| (remap(n), *r)).collect();
            let d = max_delta(&avatar.skin(&pose), &expected);
            println!("    t={t:.2}  Δ {d:8.1} mm");
        }
    }

    // Knee flex direction: rotate leftLowerLeg on each axis/sign and see where the
    // foot goes. A natural knee flex lifts the foot BACK and UP (heel toward hip).
    // Avatar faces +Y (vrm0), so back = -Y, up = +Z.
    {
        let mut foot = (0usize, 0usize);
        let mut minz = f64::MAX;
        for (mi, m) in bind.iter().enumerate() {
            for (vi, v) in m.vertices.iter().enumerate() {
                if v.position.z < minz {
                    minz = v.position.z;
                    foot = (mi, vi);
                }
            }
        }
        let fb = vertex_pos(&bind, foot);
        println!("\n[knee probe] foot bind [{:.0},{:.0},{:.0}] (seimei x,y,z), forward=+Y vrm0={}", fb[0], fb[1], fb[2], avatar.is_vrm0());
        for (ax, l) in [([0.8f32, 0., 0.], "+X"), ([-0.8, 0., 0.], "-X"), ([0., 0., 0.8], "+Z"), ([0., 0., -0.8], "-Z")] {
            let p = vertex_pos(&avatar.skin(&[("leftLowerLeg", ax)]), foot);
            println!("    leftLowerLeg {l}: foot dY {:+5.0}  dZ {:+5.0}", p[1] - fb[1], p[2] - fb[2]);
        }
    }

    // Spring excitation: does skin_dynamic actually move the hair when the head
    // rocks? Compares spring-on (lagging) vs spring-off (rigid) at the SAME pose.
    {
        let mut av = VrmAvatar::load(&bytes).expect("reload");
        let mut sprung = Vec::new();
        for f in 0..90 {
            let ang = 0.6 * ((f as f32) * 0.25).sin();
            sprung = av.skin_dynamic(&[("neck", [0.0, 0.0, ang])], 1.0 / 60.0);
        }
        let ang = 0.6 * (89.0_f32 * 0.25).sin();
        av.set_spring_enabled(false);
        let rigid = av.skin(&[("neck", [0.0, 0.0, ang])]);
        let rigid_ref: Vec<&RenderMesh> = rigid.iter().collect();
        let d = max_delta(&sprung, &rigid_ref);
        println!("\n[spring excite] joints={}  max delta on(lag) vs off(rigid) same pose: {d:.1} mm", av.spring_joints());
        println!("  => {}", if d > 1.0 { "spring MOVES the mesh (issue = no body motion to excite it)" } else { "spring has NO effect (deeper bug)" });
    }

    // Hips turntable axis: which rotation TURNS the body about vertical (head
    // moves horizontally, height ~unchanged) vs TILTS it (height drops)?
    {
        let mut top = (0usize, 0usize);
        let mut maxz = f64::MIN;
        for (mi, m) in bind.iter().enumerate() {
            for (vi, v) in m.vertices.iter().enumerate() {
                if v.position.z > maxz {
                    maxz = v.position.z;
                    top = (mi, vi);
                }
            }
        }
        let tb = vertex_pos(&bind, top);
        println!("\n[hips axis] head-top bind [{:.0},{:.0},{:.0}] (seimei x,y,z)", tb[0], tb[1], tb[2]);
        for (ax, l) in [([0.6f32, 0., 0.], "X"), ([0., 0.6, 0.], "Y"), ([0., 0., 0.6], "Z")] {
            let p = vertex_pos(&avatar.skin(&[("hips", ax)]), top);
            println!("    hips {l}: head-top d[{:+5.0},{:+5.0},{:+5.0}]", p[0] - tb[0], p[1] - tb[1], p[2] - tb[2]);
        }
    }

    // arms-down via reliable global metric: which (arm, axis, sign) shrinks the
    // model's X-width the most = arm swung down to the side.
    let (bn, bx) = aabb(&bind);
    let bind_w = bx[0] - bn[0];
    let _ = (bind_w, &bn, &bx);
    // The bind max-X vertex IS the left fingertip (geometric, reliable). Track it.
    let mut tip = (0usize, 0usize);
    let mut maxx = f64::MIN;
    for (mi, m) in bind.iter().enumerate() {
        for (vi, v) in m.vertices.iter().enumerate() {
            if v.position.x > maxx {
                maxx = v.position.x;
                tip = (mi, vi);
            }
        }
    }
    let mut tip2 = (0usize, 0usize);
    let mut minx = f64::MAX;
    for (mi, m) in bind.iter().enumerate() {
        for (vi, v) in m.vertices.iter().enumerate() {
            if v.position.x < minx {
                minx = v.position.x;
                tip2 = (mi, vi);
            }
        }
    }
    let (bp, bp2) = (vertex_pos(&bind, tip), vertex_pos(&bind, tip2));
    println!("\n[arms-down] both fingertips: +X bind Z {:.0}  -X bind Z {:.0}", bp[2], bp2[2]);
    for frac in [0.3f32, 0.5, 0.7] {
        let posed = avatar.skin(&avatar.arms_down_pose(frac));
        let (p, p2) = (vertex_pos(&posed, tip), vertex_pos(&posed, tip2));
        println!(
            "  frac {frac:.1}: +X [{:5.0},{:5.0}] dZ{:+5.0}   -X [{:5.0},{:5.0}] dZ{:+5.0}",
            p[0], p[2], p[2] - bp[2], p2[0], p2[2], p2[2] - bp2[2]
        );
    }
}

fn vertex_pos(meshes: &[RenderMesh], id: (usize, usize)) -> [f64; 3] {
    let v = &meshes[id.0].vertices[id.1];
    [v.position.x, v.position.y, v.position.z]
}
