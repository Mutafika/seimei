//! レイキャスト

use crate::math::{BoundingBox, Point3, Vec3D};

/// レイ（半直線）
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3D,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3D) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    pub fn point_at(&self, t: f64) -> Point3 {
        Point3::new(
            self.origin.x + self.direction.x * t,
            self.origin.y + self.direction.y * t,
            self.origin.z + self.direction.z * t,
        )
    }

    pub fn intersects_box(&self, bb: &BoundingBox) -> Option<f64> {
        let inv_dir = Vec3D::new(
            1.0 / self.direction.x,
            1.0 / self.direction.y,
            1.0 / self.direction.z,
        );

        let t1 = (bb.min.x - self.origin.x) * inv_dir.x;
        let t2 = (bb.max.x - self.origin.x) * inv_dir.x;
        let t3 = (bb.min.y - self.origin.y) * inv_dir.y;
        let t4 = (bb.max.y - self.origin.y) * inv_dir.y;
        let t5 = (bb.min.z - self.origin.z) * inv_dir.z;
        let t6 = (bb.max.z - self.origin.z) * inv_dir.z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if tmax < 0.0 || tmin > tmax {
            None
        } else {
            Some(if tmin < 0.0 { tmax } else { tmin })
        }
    }

    /// 任意平面との交差点
    pub fn intersect_plane(&self, plane_point: Point3, plane_normal: Vec3D) -> Option<Point3> {
        let denom = plane_normal.dot(&self.direction);
        if denom.abs() < 1e-10 {
            return None;
        }
        let diff = Vec3D::new(
            plane_point.x - self.origin.x,
            plane_point.y - self.origin.y,
            plane_point.z - self.origin.z,
        );
        let t = plane_normal.dot(&diff) / denom;
        if t < 0.0 {
            return None;
        }
        Some(self.point_at(t))
    }

    /// 三角形との交差判定（Moller-Trumbore法）
    pub fn intersects_triangle(&self, v0: &Point3, v1: &Point3, v2: &Point3) -> Option<f64> {
        const EPSILON: f64 = 1e-10;

        let edge1 = Vec3D::new(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        let edge2 = Vec3D::new(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

        let h = self.direction.cross(&edge2);
        let a = edge1.dot(&h);

        if a.abs() < EPSILON {
            return None;
        }

        let f = 1.0 / a;
        let s = Vec3D::new(
            self.origin.x - v0.x,
            self.origin.y - v0.y,
            self.origin.z - v0.z,
        );
        let u = f * s.dot(&h);

        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(&edge1);
        let v = f * self.direction.dot(&q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * edge2.dot(&q);
        if t > EPSILON { Some(t) } else { None }
    }
}

/// 交差結果
#[derive(Debug, Clone)]
pub struct RayHit {
    pub distance: f64,
    pub point: Point3,
    pub normal: Vec3D,
    pub triangle_index: Option<usize>,
}

impl RayHit {
    pub fn new(distance: f64, point: Point3, normal: Vec3D) -> Self {
        Self {
            distance,
            point,
            normal,
            triangle_index: None,
        }
    }
}
