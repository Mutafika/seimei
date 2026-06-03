//! [`AvatarController`] — drives a [`VrmAvatar`] as a live character: locomotion
//! clips, a turntable body spin, arms-down + walk arm-swing, idle auto-blink, and
//! text-driven lip-sync. [`AvatarController::update`] advances all of it by `dt`
//! and returns the posed meshes for the renderer.
//!
//! UI-agnostic: no window, camera, or egui here — embed it under any renderer
//! (the dev `vrm-viewer` binary is one such host; mearie's persona is another).

use seimei::RenderMesh;
use vrm_anatomy::animation::AnimationPlayer;

use crate::clips::{idle_clip, remap, run_clip, walk_clip};
use crate::lipsync::{Visemes, text_to_visemes};
use crate::{ExpressionPreset, VrmAvatar};

const SPIN_RATE: f32 = 0.6; // rad/s — turntable body spin
/// How far to lower the arms from the bind T-pose toward the sides (0 = T-pose,
/// 1 = straight down).
const ARM_DOWN: f32 = 0.72;
const ARM_SWING: f32 = 0.45; // walk arm-swing amplitude (rad)
/// Idle auto-blink: one quick close/open every `BLINK_PERIOD` s, lasting `BLINK_DUR`.
const BLINK_PERIOD: f32 = 3.4;
const BLINK_DUR: f32 = 0.14;
/// Lip-sync: how wide a vowel opens, and how fast the mouth eases toward its target
/// per frame (a low lerp avoids snapping between visemes → a continuous flap).
const MOUTH_OPEN: f32 = 0.9;
const MOUTH_LERP: f32 = 0.3;

/// The 5 mouth/vowel presets, indexed by the lip-sync viseme index 0..=4.
const VOWELS: [ExpressionPreset; 5] = [
    ExpressionPreset::Aa,
    ExpressionPreset::Ih,
    ExpressionPreset::Ou,
    ExpressionPreset::Ee,
    ExpressionPreset::Oh,
];

/// Blink presets overlay any face, so they're exempt from "exclusive emotion"
/// clearing and from the auto-blink driver's own management.
fn is_blink(p: ExpressionPreset) -> bool {
    matches!(
        p,
        ExpressionPreset::Blink | ExpressionPreset::BlinkLeft | ExpressionPreset::BlinkRight
    )
}

/// A live, drivable VRM character. Owns the avatar and all motion state; call
/// [`update`](Self::update) once per frame.
pub struct AvatarController {
    avatar: VrmAvatar,
    player: AnimationPlayer,
    paused: bool,
    spinning: bool,
    body_yaw: f32,    // accumulated turntable yaw (hips Y)
    gait_period: f32, // current locomotion period (s); 0 = not walking → no arm swing
    auto_blink: bool,
    blink_phase: f32,
    speech: Visemes,
    speech_time: f32,
    mouth: [f32; 5], // smoothed weights for VOWELS
}

impl AvatarController {
    /// Wrap an avatar and start it in the idle clip with auto-blink on.
    pub fn new(avatar: VrmAvatar) -> Self {
        let mut player = AnimationPlayer::new();
        player.play(idle_clip());
        Self {
            avatar,
            player,
            paused: false,
            spinning: false,
            body_yaw: 0.0,
            gait_period: 0.0,
            auto_blink: true,
            blink_phase: 0.0,
            speech: Vec::new(),
            speech_time: 0.0,
            mouth: [0.0; 5],
        }
    }

    /// The wrapped avatar (materials, presets, spring info, etc.).
    pub fn avatar(&self) -> &VrmAvatar {
        &self.avatar
    }
    /// Mutable access to the wrapped avatar (e.g. `set_spring_enabled`).
    pub fn avatar_mut(&mut self) -> &mut VrmAvatar {
        &mut self.avatar
    }

    // --- locomotion ---------------------------------------------------------

    /// Forward sign for the gait. VRM 0.x faces the opposite way from VRM 1.0
    /// (see `lib.rs` `is_vrm0`), so its legs and arms need the inverted sign to
    /// step *forward*. Mirrors the per-version camera flip in vrm-viewer.
    fn gait_dir(&self) -> f32 {
        if self.avatar.is_vrm0() {
            -1.0
        } else {
            1.0
        }
    }

    /// Stop animating → return to the bind pose.
    pub fn bind(&mut self) {
        self.player.stop();
        self.gait_period = 0.0;
    }
    pub fn play_idle(&mut self) {
        self.player.play(idle_clip());
        self.paused = false;
        self.gait_period = 0.0;
    }
    pub fn play_walk(&mut self) {
        let c = walk_clip(self.gait_dir());
        self.gait_period = c.duration;
        self.player.play(c);
        self.paused = false;
    }
    pub fn play_run(&mut self) {
        let c = run_clip(self.gait_dir());
        self.gait_period = c.duration;
        self.player.play(c);
        self.paused = false;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn is_spinning(&self) -> bool {
        self.spinning
    }
    pub fn set_spinning(&mut self, on: bool) {
        self.spinning = on;
    }
    pub fn toggle_spin(&mut self) {
        self.spinning = !self.spinning;
    }

    // --- face ---------------------------------------------------------------

    pub fn auto_blink(&self) -> bool {
        self.auto_blink
    }
    pub fn set_auto_blink(&mut self, on: bool) {
        self.auto_blink = on;
        self.blink_phase = 0.0;
        if !on {
            self.avatar.set_expression(ExpressionPreset::Blink, 0.0);
        }
    }
    pub fn toggle_auto_blink(&mut self) {
        self.set_auto_blink(!self.auto_blink);
    }

    /// Set one emotion/vowel exclusively (clears every other non-blink preset).
    pub fn set_emotion(&mut self, p: ExpressionPreset) {
        for pr in self.avatar.available_presets() {
            if !is_blink(pr) {
                self.avatar.set_expression(pr, 0.0);
            }
        }
        self.avatar.set_expression(p, 1.0);
    }
    /// Clear every non-blink expression (back to the neutral face).
    pub fn clear_face(&mut self) {
        for pr in self.avatar.available_presets() {
            if !is_blink(pr) {
                self.avatar.set_expression(pr, 0.0);
            }
        }
    }

    // --- lip-sync -----------------------------------------------------------

    /// Start lip-syncing `text` (text-rate fake visemes; kana accurate, kanji
    /// neutral, romaji crude). Overlays the current emotion/blink.
    pub fn say(&mut self, text: &str) {
        self.speech = text_to_visemes(text);
        self.speech_time = 0.0;
    }
    pub fn is_speaking(&self) -> bool {
        !self.speech.is_empty()
    }

    // --- per-frame ----------------------------------------------------------

    /// Advance all motion by `dt` seconds and return the posed meshes, one per
    /// primitive (same order as `self.avatar().primitives()`).
    pub fn update(&mut self, dt: f32) -> Vec<RenderMesh> {
        if !self.paused {
            self.player.update(dt);
        }
        // Spin turns the BODY (hips-Y yaw), which moves the hair/skirt spring
        // anchors so the secondary motion has something to react to.
        if self.spinning {
            self.body_yaw += SPIN_RATE * dt;
        }
        let raw = self.player.sample(); // empty when stopped → bind pose
        let mut pose: Vec<(&str, [f32; 3])> = raw.iter().map(|(n, r)| (remap(n), *r)).collect();
        if let Some(h) = pose.iter_mut().find(|(n, _)| *n == "hips") {
            h.1[1] += self.body_yaw;
        } else if self.body_yaw != 0.0 {
            pose.push(("hips", [0.0, self.body_yaw, 0.0]));
        }
        // Arms down out of the T-pose, swinging front/back (opposite each other)
        // when walking — the clips don't touch the arms. `gait_dir` flips the
        // swing for VRM 0.x so the arm leads the correct (forward) step.
        let (ls, rs) = if self.gait_period > 0.0 {
            let phase = self.player.current_time() / self.gait_period * std::f32::consts::TAU;
            let s = ARM_SWING * self.gait_dir() * phase.sin();
            (s, -s)
        } else {
            (0.0, 0.0)
        };
        pose.extend(self.avatar.arms_pose(ARM_DOWN, ls, rs));

        // Lip-sync: walk the viseme schedule, ease the mouth toward the current
        // vowel. Touches only the 5 vowel presets, so an emotion/blink stays on.
        let target_idx: Option<usize> = if self.speech.is_empty() {
            None
        } else {
            self.speech_time += dt;
            let mut t = self.speech_time;
            let mut cur = None;
            let mut ended = true;
            for (vi, dur) in &self.speech {
                if t < *dur {
                    cur = *vi;
                    ended = false;
                    break;
                }
                t -= *dur;
            }
            if ended {
                self.speech.clear();
                self.speech_time = 0.0;
                None
            } else {
                cur
            }
        };
        if target_idx.is_some() || self.mouth.iter().any(|&w| w > 0.001) {
            for k in 0..5 {
                let tgt = if Some(k) == target_idx { MOUTH_OPEN } else { 0.0 };
                self.mouth[k] += (tgt - self.mouth[k]) * MOUTH_LERP;
                if self.mouth[k] < 0.01 {
                    self.mouth[k] = 0.0;
                }
                self.avatar.set_expression(VOWELS[k], self.mouth[k]);
            }
        }

        // Idle auto-blink: a quick close/open overlaid on whatever face is set.
        if self.auto_blink && self.avatar.has_expression(ExpressionPreset::Blink) {
            self.blink_phase += dt;
            let t = self.blink_phase % BLINK_PERIOD;
            let w = if t < BLINK_DUR {
                (t / BLINK_DUR * std::f32::consts::PI).sin()
            } else {
                0.0
            };
            self.avatar.set_expression(ExpressionPreset::Blink, w);
        }

        // skin_dynamic advances the spring-bone (揺れもの) sim by dt each frame.
        self.avatar.skin_dynamic(&pose, dt)
    }
}
