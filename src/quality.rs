//! レンダリング品質設定

use serde::{Deserialize, Serialize};

/// 品質プリセット
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QualityPreset {
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl QualityPreset {
    pub fn all() -> &'static [QualityPreset] {
        &[QualityPreset::Low, QualityPreset::Medium, QualityPreset::High, QualityPreset::Ultra]
    }
}

/// シャドウマップ設定
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShadowQuality {
    Off,
    Low,
    Medium,
    High,
    Ultra,
}

impl ShadowQuality {
    pub fn map_size(&self) -> u32 {
        match self {
            ShadowQuality::Off => 0,
            ShadowQuality::Low => 1024,
            ShadowQuality::Medium => 2048,
            ShadowQuality::High | ShadowQuality::Ultra => 4096,
        }
    }

    pub fn cascade_count(&self) -> u32 {
        if matches!(self, ShadowQuality::Ultra) { 4 } else { 1 }
    }

    pub fn is_enabled(&self) -> bool {
        !matches!(self, ShadowQuality::Off)
    }
}

/// MSAA設定
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MsaaSamples {
    Off,
    X2,
    X4,
}

impl MsaaSamples {
    pub fn count(&self) -> u32 {
        match self {
            MsaaSamples::Off => 1,
            MsaaSamples::X2 => 2,
            MsaaSamples::X4 => 4,
        }
    }
}

/// 品質設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySettings {
    pub preset: Option<QualityPreset>,
    pub msaa: MsaaSamples,
    pub shadow: ShadowQuality,
    pub normal_mapping: bool,
    pub emissive: bool,
    pub ibl: bool,
    pub skybox: bool,
    pub ssao: bool,
    pub bloom: bool,
    pub ssr: bool,
    pub dof: bool,
    pub edge_bevel: bool,
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self::from_preset(QualityPreset::Medium)
    }
}

impl QualitySettings {
    pub fn from_preset(preset: QualityPreset) -> Self {
        match preset {
            QualityPreset::Low => Self {
                preset: Some(preset), msaa: MsaaSamples::Off, shadow: ShadowQuality::Off,
                normal_mapping: false, emissive: false, ibl: false, skybox: false,
                ssao: false, bloom: false, ssr: false, dof: false, edge_bevel: false,
            },
            QualityPreset::Medium => Self {
                preset: Some(preset), msaa: MsaaSamples::Off, shadow: ShadowQuality::Low,
                normal_mapping: false, emissive: true, ibl: false, skybox: false,
                ssao: false, bloom: false, ssr: false, dof: false, edge_bevel: false,
            },
            QualityPreset::High => Self {
                preset: Some(preset), msaa: MsaaSamples::X4, shadow: ShadowQuality::Medium,
                normal_mapping: true, emissive: true, ibl: true, skybox: true,
                ssao: true, bloom: true, ssr: false, dof: false, edge_bevel: true,
            },
            QualityPreset::Ultra => Self {
                preset: Some(preset), msaa: MsaaSamples::X4, shadow: ShadowQuality::Ultra,
                normal_mapping: true, emissive: true, ibl: true, skybox: true,
                ssao: true, bloom: true, ssr: true, dof: false, edge_bevel: true,
            },
        }
    }

    pub fn needs_post_process(&self) -> bool {
        self.ssao || self.bloom || self.ssr || self.dof || self.edge_bevel
    }

    pub fn needs_gbuffer(&self) -> bool {
        self.ssao || self.ssr || self.edge_bevel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_roundtrip() {
        for preset in QualityPreset::all() {
            let settings = QualitySettings::from_preset(*preset);
            assert_eq!(settings.preset, Some(*preset));
        }
    }

    #[test]
    fn test_msaa_count() {
        assert_eq!(MsaaSamples::Off.count(), 1);
        assert_eq!(MsaaSamples::X4.count(), 4);
    }

    #[test]
    fn test_shadow_quality() {
        assert!(!ShadowQuality::Off.is_enabled());
        assert!(ShadowQuality::Medium.is_enabled());
        assert_eq!(ShadowQuality::Ultra.cascade_count(), 4);
    }
}
