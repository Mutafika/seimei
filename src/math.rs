//! 3D 数学型（glam ベース）

use glam::{DVec3, DMat4, Vec3};
use serde::{Deserialize, Serialize};

/// 3次元点（f64精度）
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub const fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn to_dvec3(self) -> DVec3 {
        DVec3::new(self.x, self.y, self.z)
    }

    pub fn from_dvec3(v: DVec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    pub fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    pub fn distance(&self, other: &Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn midpoint(&self, other: &Point3) -> Point3 {
        Point3::new(
            (self.x + other.x) / 2.0,
            (self.y + other.y) / 2.0,
            (self.z + other.z) / 2.0,
        )
    }
}

impl Default for Point3 {
    fn default() -> Self {
        Self::origin()
    }
}

impl From<[f64; 3]> for Point3 {
    fn from(arr: [f64; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

impl From<Point3> for [f64; 3] {
    fn from(p: Point3) -> Self {
        [p.x, p.y, p.z]
    }
}

impl From<DVec3> for Point3 {
    fn from(v: DVec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

impl From<Point3> for DVec3 {
    fn from(p: Point3) -> Self {
        DVec3::new(p.x, p.y, p.z)
    }
}

/// 3次元ベクトル（f64精度）
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3D {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub const fn unit_x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    pub const fn unit_y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    pub const fn unit_z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    pub fn to_dvec3(self) -> DVec3 {
        DVec3::new(self.x, self.y, self.z)
    }

    pub fn from_dvec3(v: DVec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len > 1e-10 {
            Self::new(self.x / len, self.y / len, self.z / len)
        } else {
            Self::new(1.0, 0.0, 0.0)
        }
    }

    pub fn dot(&self, other: &Vec3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3D) -> Vec3D {
        Vec3D::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

impl Default for Vec3D {
    fn default() -> Self {
        Self::zero()
    }
}

impl From<DVec3> for Vec3D {
    fn from(v: DVec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

impl From<Vec3D> for DVec3 {
    fn from(v: Vec3D) -> Self {
        DVec3::new(v.x, v.y, v.z)
    }
}

/// 4x4 変換行列（glam ベース）
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    /// 4x4 行列（列優先）
    pub matrix: [[f64; 4]; 4],
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn translation(x: f64, y: f64, z: f64) -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [x, y, z, 1.0],
            ],
        }
    }

    pub fn scale(sx: f64, sy: f64, sz: f64) -> Self {
        Self {
            matrix: [
                [sx, 0.0, 0.0, 0.0],
                [0.0, sy, 0.0, 0.0],
                [0.0, 0.0, sz, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn rotation_z(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            matrix: [
                [c, s, 0.0, 0.0],
                [-s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn to_dmat4(&self) -> DMat4 {
        DMat4::from_cols_array_2d(&self.matrix)
    }

    pub fn from_dmat4(m: DMat4) -> Self {
        Self {
            matrix: m.to_cols_array_2d(),
        }
    }

    pub fn transform_point(&self, p: &Point3) -> Point3 {
        let m = self.to_dmat4();
        let v = m * glam::DVec4::new(p.x, p.y, p.z, 1.0);
        Point3::new(v.x / v.w, v.y / v.w, v.z / v.w)
    }

    /// 行列の合成（self * other）
    pub fn then(&self, other: &Transform) -> Transform {
        Transform::from_dmat4(self.to_dmat4() * other.to_dmat4())
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

/// バウンディングボックス（軸並行）
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: Point3,
    pub max: Point3,
}

impl BoundingBox {
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    pub fn empty() -> Self {
        Self {
            min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    pub fn extend_point(&mut self, p: &Point3) {
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.min.z = self.min.z.min(p.z);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
        self.max.z = self.max.z.max(p.z);
    }

    pub fn extend_box(&mut self, other: &BoundingBox) {
        self.extend_point(&other.min);
        self.extend_point(&other.max);
    }

    pub fn center(&self) -> Point3 {
        self.min.midpoint(&self.max)
    }

    pub fn size(&self) -> Vec3D {
        Vec3D::new(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }

    pub fn contains(&self, p: &Point3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
    }
}

impl Default for BoundingBox {
    fn default() -> Self {
        Self::empty()
    }
}
