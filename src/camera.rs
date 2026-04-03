//! カメラ制御（glam ベース）

use crate::math::{Point3, Vec3D};
use crate::ray::Ray;
use bytemuck::{Pod, Zeroable};
use glam::{DMat4, DVec3, DVec4};

/// カメラ
pub struct Camera {
    pub position: Point3,
    pub target: Point3,
    pub up: Vec3D,
    /// 視野角（度）
    pub fov: f64,
    pub aspect: f64,
    pub near: f64,
    pub far: f64,
    pub is_orthographic: bool,
    /// 正投影の表示幅（ワールド単位）
    pub ortho_width: f64,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: Point3::new(10000.0, -10000.0, 10000.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vec3D::new(0.0, 0.0, 1.0),
            fov: 45.0,
            aspect: 16.0 / 9.0,
            near: 10.0,
            far: 1000000.0,
            is_orthographic: false,
            ortho_width: 30000.0,
        }
    }

    /// ビュー行列
    pub fn view_matrix(&self) -> DMat4 {
        let eye = self.position.to_dvec3();
        let target = self.target.to_dvec3();
        let up = self.up.to_dvec3();
        DMat4::look_at_rh(eye, target, up)
    }

    /// プロジェクション行列（wgpu NDC: z 0..1 補正済み）
    pub fn projection_matrix(&self) -> DMat4 {
        let proj = if self.is_orthographic {
            let half_w = self.ortho_width / 2.0;
            let half_h = half_w / self.aspect;
            DMat4::orthographic_rh(-half_w, half_w, -half_h, half_h, self.near, self.far)
        } else {
            let fov_rad = self.fov.to_radians();
            DMat4::perspective_rh(fov_rad, self.aspect, self.near, self.far)
        };
        // glam の perspective_rh は OpenGL NDC (z: -1..1)。wgpu は z: 0..1。
        let opengl_to_wgpu = DMat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        ]);
        opengl_to_wgpu * proj
    }

    /// ビュー・プロジェクション行列
    pub fn view_projection_matrix(&self) -> DMat4 {
        self.projection_matrix() * self.view_matrix()
    }

    pub fn set_aspect(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect = width as f64 / height as f64;
        }
    }

    /// オービット回転 — 極点を自由に通過
    pub fn orbit(&mut self, delta_x: f64, delta_y: f64) {
        let sensitivity = 0.01;
        let yaw = -delta_x * sensitivity;
        let pitch = -delta_y * sensitivity;

        let target = self.target.to_dvec3();
        let mut offset = self.position.to_dvec3() - target;
        let radius = offset.length();

        let forward = (-offset).normalize();
        let up = self.up.to_dvec3();

        let right_raw = forward.cross(up);
        let right = if right_raw.length() > 1e-6 {
            right_raw.normalize()
        } else {
            let alt = if forward.z.abs() < 0.9 {
                DVec3::Z
            } else {
                DVec3::Y
            };
            forward.cross(alt).normalize()
        };

        // 回転: upで yaw、rightで pitch
        let rot_yaw = DMat4::from_axis_angle(up.normalize(), yaw);
        let rot_pitch = DMat4::from_axis_angle(right, pitch);

        offset = (rot_pitch * rot_yaw).transform_vector3(offset);

        let new_norm = offset.length();
        if new_norm > 1e-10 {
            offset *= radius / new_norm;
        }

        // up ベクトル導出
        let rotated_up = (rot_pitch * rot_yaw).transform_vector3(up);
        let fwd = (-offset).normalize();
        let up_orth = rotated_up - fwd * fwd.dot(rotated_up);
        let base_up = if up_orth.length() > 1e-6 {
            up_orth.normalize()
        } else {
            let alt = if fwd.z.abs() < 0.9 { DVec3::Z } else { DVec3::Y };
            let r = fwd.cross(alt);
            r.cross(fwd).normalize()
        };

        let new_right = fwd.cross(DVec3::Z);
        let final_up = if new_right.length() > 1e-6 {
            let nr = new_right.normalize();
            let mut turntable_up = nr.cross(fwd).normalize();
            if turntable_up.dot(base_up) < 0.0 {
                turntable_up = -turntable_up;
            }
            turntable_up
        } else {
            base_up
        };

        let new_pos = target + offset;
        self.position = Point3::from_dvec3(new_pos);
        self.up = Vec3D::from_dvec3(final_up);
    }

    /// パン（平行移動）
    pub fn pan(&mut self, delta_x: f64, delta_y: f64, screen_height: u32) {
        let world_per_pixel = if self.is_orthographic {
            let half_h = (self.ortho_width / 2.0) / self.aspect;
            2.0 * half_h / screen_height as f64
        } else {
            let distance = self.position.distance(&self.target);
            let fov_rad = self.fov.to_radians();
            2.0 * distance * (fov_rad / 2.0).tan() / screen_height as f64
        };

        let forward = Vec3D::new(
            self.target.x - self.position.x,
            self.target.y - self.position.y,
            self.target.z - self.position.z,
        ).normalize();
        let right = forward.cross(&self.up).normalize();
        let up = Vec3D::from_dvec3(right.to_dvec3().cross(forward.to_dvec3()));

        let move_x = delta_x * world_per_pixel;
        let move_y = delta_y * world_per_pixel;
        let dx = right.x * move_x + up.x * move_y;
        let dy = right.y * move_x + up.y * move_y;
        let dz = right.z * move_x + up.z * move_y;

        self.position.x -= dx;
        self.position.y -= dy;
        self.position.z -= dz;
        self.target.x -= dx;
        self.target.y -= dy;
        self.target.z -= dz;
    }

    /// ズーム
    pub fn zoom(&mut self, delta: f64) {
        let sensitivity = 0.001;
        let factor = 1.0 + delta * sensitivity;

        if self.is_orthographic {
            self.ortho_width = (self.ortho_width * factor).clamp(100.0, 500000.0);
        } else {
            let offset = self.position.to_dvec3() - self.target.to_dvec3();
            let current_distance = offset.length();
            let new_distance = (current_distance * factor).max(100.0);
            if current_distance > 1e-10 {
                let scale = new_distance / current_distance;
                let new_pos = self.target.to_dvec3() + offset * scale;
                self.position = Point3::from_dvec3(new_pos);
            }
        }
    }

    /// 全体表示
    pub fn fit_to_bounds(&mut self, min: Point3, max: Point3) {
        let center = min.midpoint(&max);
        let size_x = max.x - min.x;
        let size_y = max.y - min.y;
        let size_z = max.z - min.z;
        let max_size = size_x.max(size_y).max(size_z);

        let fov_rad = self.fov.to_radians();
        let distance = (max_size / 2.0) / (fov_rad / 2.0).tan() * 1.5;

        self.target = center;
        self.position = Point3::new(
            center.x + distance * 0.7,
            center.y - distance * 0.7,
            center.z + distance * 0.5,
        );
        self.up = Vec3D::new(0.0, 0.0, 1.0);
    }

    pub fn set_top_view(&mut self) {
        let distance = self.position.distance(&self.target);
        self.position = Point3::new(self.target.x, self.target.y, self.target.z + distance);
        self.up = Vec3D::new(0.0, 1.0, 0.0);
    }

    pub fn set_front_view(&mut self) {
        let distance = self.position.distance(&self.target);
        self.position = Point3::new(self.target.x, self.target.y - distance, self.target.z);
        self.up = Vec3D::new(0.0, 0.0, 1.0);
    }

    pub fn set_right_view(&mut self) {
        let distance = self.position.distance(&self.target);
        self.position = Point3::new(self.target.x + distance, self.target.y, self.target.z);
        self.up = Vec3D::new(0.0, 0.0, 1.0);
    }

    /// ウォークスルー: 前後移動（水平面のみ）
    pub fn move_forward(&mut self, distance: f64) {
        let dx = self.target.x - self.position.x;
        let dy = self.target.y - self.position.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-10 { return; }
        let fx = dx / len * distance;
        let fy = dy / len * distance;
        self.position.x += fx;
        self.position.y += fy;
        self.target.x += fx;
        self.target.y += fy;
    }

    /// ウォークスルー: 左右移動
    pub fn strafe(&mut self, distance: f64) {
        let dx = self.target.x - self.position.x;
        let dy = self.target.y - self.position.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-10 { return; }
        let rx = dy / len * distance;
        let ry = -dx / len * distance;
        self.position.x += rx;
        self.position.y += ry;
        self.target.x += rx;
        self.target.y += ry;
    }

    /// ウォークスルー: マウスルック
    pub fn look(&mut self, delta_x: f64, delta_y: f64) {
        let sensitivity = 0.003;
        let yaw = -delta_x * sensitivity;
        let pitch = -delta_y * sensitivity;

        let mut dx = self.target.x - self.position.x;
        let mut dy = self.target.y - self.position.y;
        let mut dz = self.target.z - self.position.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        let cos_y = yaw.cos();
        let sin_y = yaw.sin();
        let nx = dx * cos_y - dy * sin_y;
        let ny = dx * sin_y + dy * cos_y;
        dx = nx;
        dy = ny;

        let horiz = (dx * dx + dy * dy).sqrt();
        let current_pitch = dz.atan2(horiz);
        let max_pitch = 80.0_f64.to_radians();
        let new_pitch = (current_pitch + pitch).clamp(-max_pitch, max_pitch);
        dz = horiz * new_pitch.tan();

        let new_dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if new_dist > 1e-10 {
            let scale = dist / new_dist;
            dx *= scale;
            dy *= scale;
            dz *= scale;
        }

        self.target.x = self.position.x + dx;
        self.target.y = self.position.y + dy;
        self.target.z = self.position.z + dz;
        self.up = Vec3D::new(0.0, 0.0, 1.0);
    }

    /// スクリーン座標からワールド空間のレイを生成
    pub fn screen_to_ray(&self, screen_x: f64, screen_y: f64, screen_width: u32, screen_height: u32) -> Ray {
        let ndc_x = (2.0 * screen_x / screen_width as f64) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / screen_height as f64);

        let vp = self.view_projection_matrix();
        let inv_vp = vp.inverse();

        // wgpu NDC: z = 0..1
        let near_ndc = DVec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far_ndc = DVec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let near_world = inv_vp * near_ndc;
        let far_world = inv_vp * far_ndc;

        let near_pos = Point3::new(
            near_world.x / near_world.w,
            near_world.y / near_world.w,
            near_world.z / near_world.w,
        );
        let far_pos = Point3::new(
            far_world.x / far_world.w,
            far_world.y / far_world.w,
            far_world.z / far_world.w,
        );

        let dir = Vec3D::new(
            far_pos.x - near_pos.x,
            far_pos.y - near_pos.y,
            far_pos.z - near_pos.z,
        );

        Ray::new(near_pos, dir)
    }

    /// ワールド座標をスクリーン座標に変換
    pub fn world_to_screen(&self, point: &Point3, screen_width: u32, screen_height: u32) -> (f64, f64) {
        let vp = self.view_projection_matrix();
        let p = DVec4::new(point.x, point.y, point.z, 1.0);
        let clip = vp * p;

        if clip.w.abs() < 1e-10 {
            return (0.0, 0.0);
        }

        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;

        let screen_x = (ndc_x + 1.0) * 0.5 * screen_width as f64;
        let screen_y = (1.0 - ndc_y) * 0.5 * screen_height as f64;

        (screen_x, screen_y)
    }

    /// カメラ補間
    pub fn lerp(&self, other: &Camera, t: f64) -> Camera {
        let t = t.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        let up = DVec3::new(
            self.up.x * inv + other.up.x * t,
            self.up.y * inv + other.up.y * t,
            self.up.z * inv + other.up.z * t,
        );
        let up = if up.length() > 1e-10 { up.normalize() } else { DVec3::Z };
        Camera {
            position: Point3::new(
                self.position.x * inv + other.position.x * t,
                self.position.y * inv + other.position.y * t,
                self.position.z * inv + other.position.z * t,
            ),
            target: Point3::new(
                self.target.x * inv + other.target.x * t,
                self.target.y * inv + other.target.y * t,
                self.target.z * inv + other.target.z * t,
            ),
            up: Vec3D::from_dvec3(up),
            fov: self.fov * inv + other.fov * t,
            aspect: self.aspect,
            near: self.near,
            far: self.far,
            is_orthographic: if t < 0.5 { self.is_orthographic } else { other.is_orthographic },
            ortho_width: self.ortho_width * inv + other.ortho_width * t,
        }
    }

    /// ライト視点のビュー・プロジェクション行列（シャドウマップ用）
    pub fn compute_light_view_proj(
        &self,
        light_direction: &Vec3D,
        scene_center: &Point3,
        scene_radius: f64,
    ) -> DMat4 {
        let dir = light_direction.to_dvec3().normalize();
        let light_pos = scene_center.to_dvec3() - dir * scene_radius * 2.0;
        let target = scene_center.to_dvec3();

        let world_z = DVec3::Z;
        let right = dir.cross(world_z);
        let up = if right.length() > 1e-6 {
            right.normalize().cross(dir).normalize()
        } else {
            let alt = DVec3::Y;
            let right = dir.cross(alt).normalize();
            right.cross(dir).normalize()
        };

        let view = DMat4::look_at_rh(light_pos, target, up);
        let ortho = DMat4::orthographic_rh(
            -scene_radius, scene_radius,
            -scene_radius, scene_radius,
            0.1, scene_radius * 4.0,
        );
        let opengl_to_wgpu = DMat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        ]);
        opengl_to_wgpu * ortho * view
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU用のカメラユニフォーム
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub position: [f32; 4],
    pub clip_min: [f32; 4],
    pub clip_max: [f32; 4],
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera) -> Self {
        let view_proj = camera.view_projection_matrix();
        let view = camera.view_matrix();

        Self {
            view_proj: dmat4_to_f32(view_proj),
            view: dmat4_to_f32(view),
            position: [
                camera.position.x as f32,
                camera.position.y as f32,
                camera.position.z as f32,
                1.0,
            ],
            clip_min: [0.0; 4],
            clip_max: [0.0; 4],
        }
    }
}

/// DMat4 → [[f32; 4]; 4]（列優先）
pub fn dmat4_to_f32(m: DMat4) -> [[f32; 4]; 4] {
    let cols = m.to_cols_array_2d();
    [
        [cols[0][0] as f32, cols[0][1] as f32, cols[0][2] as f32, cols[0][3] as f32],
        [cols[1][0] as f32, cols[1][1] as f32, cols[1][2] as f32, cols[1][3] as f32],
        [cols[2][0] as f32, cols[2][1] as f32, cols[2][2] as f32, cols[2][3] as f32],
        [cols[3][0] as f32, cols[3][1] as f32, cols[3][2] as f32, cols[3][3] as f32],
    ]
}
