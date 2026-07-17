//! ライティングシステム

use bytemuck::{Pod, Zeroable};

/// 最大ライト数（ストレージバッファ使用）
pub const MAX_LIGHTS: usize = 128;

/// ライト種別
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightKind {
    Directional,
    Point,
}

/// CPU側のライト定義
#[derive(Clone, Debug)]
pub struct Light {
    pub kind: LightKind,
    /// Directional: 光の方向（正規化）、Point: 光源位置
    pub direction_or_position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    /// 有効範囲。0 = 無限
    pub range: f32,
    /// スポットライト半角 (radians)。0 = 全方向
    pub spot_half_angle: f32,
}

impl Light {
    pub fn default_directional() -> Self {
        Self {
            kind: LightKind::Directional,
            direction_or_position: [0.4082, -0.4082, 0.8165],
            color: [1.0, 1.0, 1.0],
            intensity: 0.7,
            range: 0.0,
            spot_half_angle: 0.0,
        }
    }

    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        Self {
            kind: LightKind::Point,
            direction_or_position: position,
            color,
            intensity,
            range: 0.0,
            spot_half_angle: 0.0,
        }
    }

    pub fn directional(direction: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        Self {
            kind: LightKind::Directional,
            direction_or_position: direction,
            color,
            intensity,
            range: 0.0,
            spot_half_angle: 0.0,
        }
    }
}

/// GPU側のライトデータ（48バイト/ライト）
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuLight {
    pub direction_or_position_and_type: [f32; 4],
    pub color_and_intensity: [f32; 4],
    pub extra: [f32; 4],
}

impl GpuLight {
    pub fn from_light(light: &Light) -> Self {
        let light_type = match light.kind {
            LightKind::Directional => 0.0,
            LightKind::Point => 1.0,
        };
        Self {
            direction_or_position_and_type: [
                light.direction_or_position[0],
                light.direction_or_position[1],
                light.direction_or_position[2],
                light_type,
            ],
            color_and_intensity: [
                light.color[0],
                light.color[1],
                light.color[2],
                light.intensity,
            ],
            extra: [
                light.range,
                0.0,
                light.spot_half_angle,
                0.0,
            ],
        }
    }

    fn empty() -> Self {
        Self {
            direction_or_position_and_type: [0.0; 4],
            color_and_intensity: [0.0; 4],
            extra: [0.0; 4],
        }
    }
}

/// GPU側のライトヘッダー（32バイト）
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightHeader {
    pub ambient_and_count: [f32; 4],
    pub mode_flags: [f32; 4],
}

impl LightHeader {
    pub fn new(ambient_color: [f32; 3], light_count: usize, dark_room: bool) -> Self {
        Self {
            ambient_and_count: [
                ambient_color[0],
                ambient_color[1],
                ambient_color[2],
                light_count as f32,
            ],
            mode_flags: [
                if dark_room { 1.0 } else { 0.0 },
                1.0,
                0.0,
                0.0,
            ],
        }
    }

    pub fn default_header() -> Self {
        Self::new([0.3, 0.3, 0.3], 1, false)
    }
}

/// ストレージバッファ用のライトデータ構築ヘルパー
pub struct LightStorageData;

impl LightStorageData {
    pub fn build(
        ambient_color: [f32; 3],
        lights: &[Light],
        dark_room: bool,
    ) -> (LightHeader, Vec<GpuLight>) {
        let count = lights.len().min(MAX_LIGHTS);
        let header = LightHeader::new(ambient_color, count, dark_room);
        let mut gpu_lights: Vec<GpuLight> = lights
            .iter()
            .take(MAX_LIGHTS)
            .map(GpuLight::from_light)
            .collect();
        if gpu_lights.is_empty() {
            gpu_lights.push(GpuLight::empty());
        }
        (header, gpu_lights)
    }
}

/// GPU側のライトユニフォーム（旧形式、8ライト固定配列）
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightUniform {
    pub ambient_and_count: [f32; 4],
    pub lights: [GpuLight; 8],
}

impl LightUniform {
    pub fn default_lighting() -> Self {
        let default_light = GpuLight::from_light(&Light::default_directional());
        let mut lights = [GpuLight::empty(); 8];
        lights[0] = default_light;
        Self {
            ambient_and_count: [0.3, 0.3, 0.3, 1.0],
            lights,
        }
    }

    pub fn from_lights(ambient_color: [f32; 3], light_list: &[Light]) -> Self {
        let count = light_list.len().min(8);
        let mut lights = [GpuLight::empty(); 8];
        for (i, light) in light_list.iter().take(8).enumerate() {
            lights[i] = GpuLight::from_light(light);
        }
        Self {
            ambient_and_count: [
                ambient_color[0],
                ambient_color[1],
                ambient_color[2],
                count as f32,
            ],
            lights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_light_size() {
        assert_eq!(std::mem::size_of::<GpuLight>(), 48);
    }

    #[test]
    fn test_light_header_size() {
        assert_eq!(std::mem::size_of::<LightHeader>(), 32);
    }

    #[test]
    fn test_light_uniform_size() {
        assert_eq!(
            std::mem::size_of::<LightUniform>(),
            16 + 8 * std::mem::size_of::<GpuLight>()
        );
    }

    #[test]
    fn test_storage_data_empty_has_dummy() {
        let (header, gpu_lights) = LightStorageData::build([0.3, 0.3, 0.3], &[], false);
        assert_eq!(header.ambient_and_count[3], 0.0);
        assert_eq!(gpu_lights.len(), 1);
    }

    #[test]
    fn test_storage_data_clamps_to_max() {
        let lights: Vec<Light> = (0..200)
            .map(|_| Light::default_directional())
            .collect();
        let (header, gpu_lights) = LightStorageData::build([0.2, 0.2, 0.2], &lights, false);
        assert_eq!(header.ambient_and_count[3], 128.0);
        assert_eq!(gpu_lights.len(), 128);
    }
}
