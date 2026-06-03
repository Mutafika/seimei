//! Throwaway probe: can seimei parse a real .vrm, and what does it extract?
//! Run: cargo run --example vrm_probe -- /path/to/model.vrm

fn main() {
    let path = std::env::args().nth(1).expect("usage: vrm_probe <file.vrm>");
    let bytes = std::fs::read(&path).expect("read vrm");
    println!("loaded {} bytes from {path}", bytes.len());

    let scene = seimei::load_gltf_from_bytes(&bytes).expect("parse vrm");

    println!("primitives: {}", scene.primitives.len());
    let mut total_v = 0;
    let mut total_i = 0;
    let mut skinned = 0;
    let mut textured = 0;
    let mut morphed = 0;
    for (i, p) in scene.primitives.iter().enumerate() {
        total_v += p.mesh.vertices.len();
        total_i += p.mesh.indices.len();
        if p.skin.is_some() {
            skinned += 1;
        }
        if p.material.base_color_texture.is_some() {
            textured += 1;
        }
        if !p.morph_targets.is_empty() {
            morphed += 1;
        }
        if i < 6 {
            let sk = p
                .skin
                .as_ref()
                .map(|s| format!("skin[{} joints]", s.joint_names.len()))
                .unwrap_or_else(|| "no-skin".into());
            let tex = p
                .material
                .base_color_texture
                .as_ref()
                .map(|t| format!("tex {}x{}", t.width, t.height))
                .unwrap_or_else(|| "flat".into());
            println!(
                "  prim {i}: {} verts, {} tris, {sk}, {tex}, {} morphs, mat={:?}",
                p.mesh.vertices.len(),
                p.mesh.indices.len() / 3,
                p.morph_targets.len(),
                p.material.name,
            );
        }
    }
    println!("totals: {total_v} verts, {} tris", total_i / 3);
    println!("skinned prims: {skinned}, textured: {textured}, morphed: {morphed}");
    println!("nodes: {}", scene.nodes.len());

    // VRM version detection via extensions
    match &scene.extensions_json {
        Some(ext) => {
            let keys: Vec<&String> = ext.as_object().map(|o| o.keys().collect()).unwrap_or_default();
            println!("extensions: {keys:?}");
            if ext.get("VRMC_vrm").is_some() {
                println!("=> VRM 1.0 (VRMC_vrm)");
                if let Some(hb) = ext
                    .get("VRMC_vrm")
                    .and_then(|v| v.get("humanoid"))
                    .and_then(|h| h.get("humanBones"))
                    .and_then(|b| b.as_object())
                {
                    println!("   humanBones: {}", hb.len());
                }
            } else if ext.get("VRM").is_some() {
                println!("=> VRM 0.x (VRM)");
            } else {
                println!("=> no VRM humanoid extension found");
            }
        }
        None => println!("no extensions_json"),
    }

    // a few skin joint names from the first skinned primitive
    if let Some(p) = scene.primitives.iter().find(|p| p.skin.is_some()) {
        let s = p.skin.as_ref().unwrap();
        let sample: Vec<&String> = s.joint_names.iter().take(12).collect();
        println!("first skin joints (12): {sample:?}");
        println!("IBMs: {}", s.inverse_bind_matrices.len());
    }

    // mesh AABB (in seimei space: Z-up, mm)
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for p in &scene.primitives {
        for v in &p.mesh.vertices {
            let pos = [v.position.x as f32, v.position.y as f32, v.position.z as f32];
            for k in 0..3 {
                min[k] = min[k].min(pos[k]);
                max[k] = max[k].max(pos[k]);
            }
        }
    }
    println!("AABB min {min:?} max {max:?} (Z-up, mm)");
}
