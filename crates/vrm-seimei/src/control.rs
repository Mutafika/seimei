//! [`AvatarController`] — drives a [`VrmAvatar`] as a live character: locomotion
//! clips, a turntable body spin, arms-down + walk arm-swing, idle auto-blink, and
//! text-driven lip-sync. [`AvatarController::update`] advances all of it by `dt`
//! and returns the posed meshes for the renderer.
//!
//! UI-agnostic: no window, camera, or egui here — embed it under any renderer
//! (the dev `vrm-viewer` binary is one such host; mearie's persona is another).

use std::collections::HashMap;

use glam::Mat4;
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
const BLINK_DUR: f32 = 0.24; // 1回の開閉時間。0.14は速すぎ(twitch)＝0.24で柔らかいまばたきに
/// Lip-sync: how wide a vowel opens, and how fast the mouth eases toward its target
/// per frame (a low lerp avoids snapping between visemes → a continuous flap).
const MOUTH_OPEN: f32 = 0.9;
const MOUTH_LERP: f32 = 0.3;
/// 表情(emotion)の weight が目標へ寄る速さ /s。大きいほど速くフェード（スナップ防止）。
const EMOTION_EASE_RATE: f32 = 9.0;

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
    /// 表情の目標 weight（preset→0..1）。set_emotion 系で設定し、毎フレーム emotion_now を
    /// これへ ease させる＝スナップせず滑らかにフェード。blink/母音(lip-sync)は対象外。
    emotion_target: HashMap<ExpressionPreset, f32>,
    /// 現在の実 weight（ease の途中値）。emotion_target へ寄せながら set_expression する。
    emotion_now: HashMap<ExpressionPreset, f32>,
    /// 腕を T-pose から脇へ下ろす量(0=T-pose で真横, 1=真下)。既定は [`ARM_DOWN`]。
    /// 吊り下げ等で腕を開いたままにしたい外部制御から差し替えられる。
    arm_down: f32,
    /// 休めの腕を外へ開く量(rad)。細い胴（MMD体型等）は腕付け根が胴の側面より内側に
    /// あり、真下に下ろすと上腕が胴へ埋まる。これを外転させて胴をかわす。既定0=不変。
    arm_abduct: f32,
    /// 休めの腕を前へ振る量(rad)。+で手が体の前側へ。細い胴で手が尻/腿へ埋まるのを防ぐ。既定0。
    arm_rest_swing: f32,
    /// コントローラ合成の腕（腕下げ＋歩行スイング）を適用するか。モーキャプ等の腕トラックを
    /// 含むカスタムクリップ再生中は false にして二重適用を避ける（play_custom が制御）。
    synth_arms: bool,
    /// 直近フレームの「ポーズ済み world 変換」(native Y-up/m, node index)。外部物理が
    /// 生きた骨位置（歩行で動く手など）にアンカーできるよう毎フレーム保存する。
    last_world: Vec<Mat4>,
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
            emotion_target: HashMap::new(),
            emotion_now: HashMap::new(),
            arm_down: ARM_DOWN,
            arm_abduct: 0.0,
            arm_rest_swing: 0.0,
            synth_arms: true,
            last_world: Vec::new(),
        }
    }

    /// 直近 [`update`](Self::update) 時点のポーズ済み world 変換（native Y-up/m、
    /// glTF node index）。空なら未更新。外部物理のアンカー取得に使う。
    pub fn world(&self) -> &[Mat4] {
        &self.last_world
    }

    /// 腕を脇へ下ろす量(0=真横T-pose, 1=真下)を差し替える。吊り下げで腕を開いた
    /// ままにする等、外部から腕の基本姿勢を制御するための seam。既定は 0.72。
    pub fn set_arm_down(&mut self, fraction: f32) {
        self.arm_down = fraction.clamp(0.0, 1.0);
    }

    /// 休めの腕を外へ開く量(rad)を差し替える。細い胴で上腕が胴へ埋まるのを防ぐ。
    /// 既定0（forticsim 等は不変）。MMD体型は ticsim 側で正値を設定する。
    pub fn set_arm_abduct(&mut self, rad: f32) {
        self.arm_abduct = rad;
    }

    /// 休めの腕を前へ振る量(rad)を差し替える。手を腿の前側へ逃がして尻/腿への埋まりを防ぐ。既定0。
    pub fn set_arm_rest_swing(&mut self, rad: f32) {
        self.arm_rest_swing = rad;
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
        self.synth_arms = true;
    }
    pub fn play_idle(&mut self) {
        self.player.play(idle_clip());
        self.paused = false;
        self.gait_period = 0.0;
        self.synth_arms = true;
    }
    pub fn play_walk(&mut self) {
        let c = walk_clip(self.gait_dir());
        self.gait_period = c.duration;
        self.player.play(c);
        self.paused = false;
        self.synth_arms = true;
    }
    pub fn play_run(&mut self) {
        let c = run_clip(self.gait_dir());
        self.gait_period = c.duration;
        self.player.play(c);
        self.paused = false;
        self.synth_arms = true;
    }

    /// 外部生成クリップ（モーキャプ等）を再生する。`gait_period` は歩行同期系（胸ジグル等）が
    /// 参照する周期で、ループ歩行なら clip.duration、ワンショットなら 0 を渡す。
    /// `synth_arms=false` でコントローラ合成の腕（arms_pose の腕下げ＋振り子スイング）を止める
    /// （モーキャプは腕トラックを含むので合成すると二重になる）。
    pub fn play_custom(&mut self, clip: vrm_anatomy::animation::AnimationClip, gait_period: f32, synth_arms: bool) {
        self.gait_period = gait_period;
        self.synth_arms = synth_arms;
        self.player.play(clip);
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

    /// Set one emotion exclusively at full strength (others fade out). Eased.
    pub fn set_emotion(&mut self, p: ExpressionPreset) {
        self.set_emotion_weighted(p, 1.0);
    }

    /// Set one emotion exclusively at strength `w` (0..1). Every other non-blink
    /// target fades to 0. The actual weight eases toward `w` (no snap).
    pub fn set_emotion_weighted(&mut self, p: ExpressionPreset, w: f32) {
        self.emotion_target.clear();
        let w = w.clamp(0.0, 1.0);
        if w > 0.0 {
            self.emotion_target.insert(p, w);
        }
    }

    /// Blend an emotion in at strength `w` **without** clearing the others — lets
    /// several presets overlay (e.g. 0.7 Happy + 0.3 Surprised). `w<=0` removes it.
    pub fn blend_emotion(&mut self, p: ExpressionPreset, w: f32) {
        let w = w.clamp(0.0, 1.0);
        if w <= 0.0 {
            self.emotion_target.remove(&p);
        } else {
            self.emotion_target.insert(p, w);
        }
    }

    /// Clear every non-blink expression (fades back to the neutral face).
    pub fn clear_face(&mut self) {
        self.emotion_target.clear();
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
        self.update_with_overlay(dt, &[])
    }

    /// Like [`update`](Self::update), but **adds** `overlay` rotations (VRM bone
    /// name → local euler `[X,Y,Z]` rad) onto the composed pose just before
    /// skinning. The joint-control seam for external physics: feed PD wobble,
    /// active-ragdoll targets, etc. Unknown bones are pushed as new entries.
    pub fn update_with_overlay(&mut self, dt: f32, overlay: &[(&str, [f32; 3])]) -> Vec<RenderMesh> {
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
        if self.synth_arms {
            let (ls, rs) = if self.gait_period > 0.0 {
                let phase = self.player.current_time() / self.gait_period * std::f32::consts::TAU;
                let s = ARM_SWING * self.gait_dir() * phase.sin();
                (s, -s)
            } else {
                // 休め: 手を腿の前側へ逃がす前振り（細い胴で手が尻/腿へ埋まるのを防ぐ）。
                (self.arm_rest_swing, self.arm_rest_swing)
            };
            pose.extend(self.avatar.arms_pose(self.arm_down, ls, rs, self.arm_abduct));
        }

        // 表情(emotion)を毎フレーム目標 weight へ ease（スナップ防止）。blink は別管理、
        // 母音は下の lip-sync が後で上書きするのでここで触っても問題ない。
        if !self.emotion_target.is_empty() || !self.emotion_now.is_empty() {
            let step = (dt * EMOTION_EASE_RATE).min(1.0);
            // target にも now にも現れる preset を集める。
            let mut keys: Vec<ExpressionPreset> = self.emotion_now.keys().copied().collect();
            for &k in self.emotion_target.keys() {
                if !keys.contains(&k) {
                    keys.push(k);
                }
            }
            for p in keys {
                if is_blink(p) {
                    continue;
                }
                let tgt = self.emotion_target.get(&p).copied().unwrap_or(0.0);
                let now = self.emotion_now.get(&p).copied().unwrap_or(0.0);
                let mut nw = now + (tgt - now) * step;
                if (nw - tgt).abs() < 0.005 {
                    nw = tgt;
                }
                if nw <= 0.001 && tgt <= 0.001 {
                    self.emotion_now.remove(&p);
                    self.avatar.set_expression(p, 0.0);
                } else {
                    self.emotion_now.insert(p, nw);
                    self.avatar.set_expression(p, nw);
                }
            }
        }

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

        // 外部物理オーバーレイ（PD揺れ・active-ragdoll目標等）を pose に加算。
        for (bone, e) in overlay {
            if let Some(p) = pose.iter_mut().find(|(n, _)| n == bone) {
                p.1[0] += e[0];
                p.1[1] += e[1];
                p.1[2] += e[2];
            } else {
                pose.push((bone, *e));
            }
        }

        // 外部物理アンカー用に「ポーズ済み world 変換」を保存（skin_dynamic と同じ
        // pose から計算）。歩行で動く手などの生きた位置をここから取れる。
        self.last_world = self.avatar.world_for_pose(&pose);

        // skin_dynamic advances the spring-bone (揺れもの) sim by dt each frame.
        self.avatar.skin_dynamic(&pose, dt)
    }
}
