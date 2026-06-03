//! Reusable skeletal animation clips for VRM humanoid rigs, authored in **VRM
//! humanoid bone names** (so they need no remapping when fed to `VrmAvatar::skin`).
//! The locomotion cycle is tuned for VRM rigs: legs swing on X, knees flex on −X
//! (the natural direction; +X hyperextends, per offline probing).

use vrm_anatomy::animation::{AnimationClip, BoneTrack, Easing, Keyframe, LoopMode};

/// vrm-anatomy's built-in `walk_cycle`/`run_cycle` use Mixamo/RPM bone names; our
/// `skin()` keys on VRM humanoid names. Remap the bones those clips touch; pass
/// anything else through, so a clip already authored in humanoid names is
/// unaffected. Only needed for the vrm-anatomy built-ins — the clips in this
/// module are already humanoid-named.
pub fn remap(name: &str) -> &str {
    match name {
        "Hips" => "hips",
        "Spine" => "spine",
        "Spine1" | "Spine2" => "chest",
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
        other => other,
    }
}

/// A gentle "alive" idle: a slow breathing nod on spine/chest. PingPong so it loops
/// seamlessly. NO hips — hips is the root, so any rotation there tilts the whole
/// figure rigidly (reads as leaning, not breathing).
pub fn idle_clip() -> AnimationClip {
    let kf = |t: f32, r: [f32; 3]| Keyframe {
        time: t,
        rotation: r,
        easing: Easing::EaseInOut,
        stiffness: None,
        damping: None,
    };
    let track = |bone: &str, a: [f32; 3], b: [f32; 3]| BoneTrack {
        bone_name: bone.to_string(),
        keyframes: vec![kf(0.0, a), kf(1.5, b)],
    };
    AnimationClip {
        name: "idle".into(),
        duration: 1.5,
        loop_mode: LoopMode::PingPong,
        tracks: vec![
            track("spine", [0.0, 0.0, 0.0], [0.016, 0.0, 0.0]), // breathe (subtle nod)
            track("chest", [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]),
        ],
    }
}

/// A locomotion cycle (walk or run) in VRM humanoid bone names. Legs swing on X,
/// knees flex one-directionally on −X (`knee` negative), and the pelvis/upper body
/// counter-twist on Y to sway the skirt/hair springs. `dur` sets the cadence;
/// `hip`/`knee`/`foot` are the swing amplitudes (radians). The matching arm swing
/// is applied by the controller (the clip doesn't touch the arms).
pub fn locomotion_clip(name: &str, dur: f32, hip: f32, knee: f32, foot: f32) -> AnimationClip {
    let n = 8;
    let two_pi = 2.0 * std::f32::consts::PI;
    let sin_track = |bone: &str, axis: usize, amp: f32, phase: f32| {
        let kfs = (0..=n)
            .map(|i| {
                let t = dur * i as f32 / n as f32;
                let mut rot = [0.0f32; 3];
                rot[axis] = amp * (two_pi * (t / dur + phase)).sin();
                Keyframe { time: t, rotation: rot, easing: Easing::Linear, stiffness: None, damping: None }
            })
            .collect();
        BoneTrack { bone_name: bone.into(), keyframes: kfs }
    };
    // One-directional flex (bend ≥ 0) during the swing phase, on −X.
    let knee_track = |bone: &str, amp: f32, phase: f32| {
        let kfs = (0..=n)
            .map(|i| {
                let t = dur * i as f32 / n as f32;
                let p = two_pi * (t / dur + phase) + std::f32::consts::FRAC_PI_2;
                let bend = amp * p.sin().max(0.0);
                Keyframe { time: t, rotation: [bend, 0.0, 0.0], easing: Easing::Linear, stiffness: None, damping: None }
            })
            .collect();
        BoneTrack { bone_name: bone.into(), keyframes: kfs }
    };
    let pelvis_yaw = 0.25 * hip;
    let spine_yaw = -0.15 * hip;
    AnimationClip {
        name: name.into(),
        duration: dur,
        loop_mode: LoopMode::Loop,
        tracks: vec![
            sin_track("leftUpperLeg", 0, hip, 0.0),
            sin_track("rightUpperLeg", 0, hip, 0.5),
            knee_track("leftLowerLeg", knee, 0.0), // knee < 0 → −X = natural flex
            knee_track("rightLowerLeg", knee, 0.5),
            sin_track("leftFoot", 0, foot, 0.25),
            sin_track("rightFoot", 0, foot, 0.75),
            sin_track("hips", 1, pelvis_yaw, 0.0), // pelvis twist (sways the skirt)
            sin_track("spine", 1, spine_yaw, 0.0), // counter-twist (sways the hair)
        ],
    }
}

/// Walk preset (cadence + amplitudes tuned by eye). Period = `duration` = 1.0 s.
pub fn walk_clip() -> AnimationClip {
    locomotion_clip("walk", 1.0, -0.45, 0.8, 0.15)
}

/// Run preset. Period = `duration` = 0.6 s.
pub fn run_clip() -> AnimationClip {
    locomotion_clip("run", 0.6, -0.7, 1.2, 0.2)
}
