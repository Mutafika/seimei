//! PLY / Gaussian Splatting ファイルローダー

use crate::math::{BoundingBox, Point3};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use thiserror::Error;
use tracing::info;

/// PLYパーサーエラー
#[derive(Debug, Error)]
pub enum PlyError {
    #[error("IOエラー: {0}")]
    Io(#[from] std::io::Error),

    #[error("ヘッダーパースエラー: {0}")]
    Header(String),

    #[error("データパースエラー: {0}")]
    Data(String),

    #[error("未対応のフォーマット: {0}")]
    UnsupportedFormat(String),
}

/// 点群の1点
#[derive(Debug, Clone)]
pub struct PlyPoint {
    pub position: Point3,
    pub color: [f32; 4],
}

/// 点群データ
#[derive(Debug, Clone)]
pub struct PlyPointCloud {
    pub points: Vec<PlyPoint>,
    pub bounding_box: BoundingBox,
}

/// Gaussian Splatting の1スプラット
#[derive(Debug, Clone)]
pub struct GaussianPoint {
    pub position: Point3,
    pub color: [f32; 3],
    pub opacity: f32,
    pub scale: [f32; 3],
    /// 回転四元数 [w, x, y, z]
    pub rotation: [f32; 4],
}

/// Gaussian Splatting点群データ
#[derive(Debug, Clone)]
pub struct GaussianCloud {
    pub points: Vec<GaussianPoint>,
    pub bounding_box: BoundingBox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PropertyType {
    Float,
    Double,
    Uchar,
    Int,
    Short,
    Ushort,
    Uint,
}

impl PropertyType {
    fn byte_size(&self) -> usize {
        match self {
            PropertyType::Float => 4,
            PropertyType::Double => 8,
            PropertyType::Uchar => 1,
            PropertyType::Int => 4,
            PropertyType::Short => 2,
            PropertyType::Ushort => 2,
            PropertyType::Uint => 4,
        }
    }

    fn from_str(s: &str) -> Result<Self, PlyError> {
        match s {
            "float" | "float32" => Ok(PropertyType::Float),
            "double" | "float64" => Ok(PropertyType::Double),
            "uchar" | "uint8" => Ok(PropertyType::Uchar),
            "int" | "int32" => Ok(PropertyType::Int),
            "short" | "int16" => Ok(PropertyType::Short),
            "ushort" | "uint16" => Ok(PropertyType::Ushort),
            "uint" | "uint32" => Ok(PropertyType::Uint),
            _ => Err(PlyError::Header(format!("未知のプロパティ型: {}", s))),
        }
    }
}

#[derive(Debug, Clone)]
struct PropertyDef {
    name: String,
    prop_type: PropertyType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColorMode {
    StandardRgb,
    GaussianSh,
    None,
}

/// PLYローダー
pub struct PlyLoader;

impl PlyLoader {
    /// PLYファイルを点群として読み込み
    pub fn load(path: &Path) -> Result<PlyPointCloud, PlyError> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);

        let (format, vertex_count, properties) = Self::parse_header(&mut reader)?;
        let color_mode = Self::detect_color_mode(&properties);

        let x_idx = Self::find_property(&properties, "x")?;
        let y_idx = Self::find_property(&properties, "y")?;
        let z_idx = Self::find_property(&properties, "z")?;

        let color_indices = match color_mode {
            ColorMode::StandardRgb => {
                let r = Self::find_property(&properties, "red")?;
                let g = Self::find_property(&properties, "green")?;
                let b = Self::find_property(&properties, "blue")?;
                let a = Self::find_property_opt(&properties, "alpha");
                Some((r, g, b, a))
            }
            ColorMode::GaussianSh => {
                let r = Self::find_property(&properties, "f_dc_0")?;
                let g = Self::find_property(&properties, "f_dc_1")?;
                let b = Self::find_property(&properties, "f_dc_2")?;
                Some((r, g, b, None))
            }
            ColorMode::None => None,
        };

        let mut points = Vec::with_capacity(vertex_count);
        let mut bb = BoundingBox::empty();

        match format {
            PlyFormat::BinaryLittleEndian => {
                let vertex_size: usize = properties.iter().map(|p| p.prop_type.byte_size()).sum();
                let offsets = Self::compute_offsets(&properties);
                let mut buf = vec![0u8; vertex_size * vertex_count];
                reader.read_exact(&mut buf)?;

                for i in 0..vertex_count {
                    let base = i * vertex_size;
                    let x = Self::read_float_value(&buf, base + offsets[x_idx], &properties[x_idx].prop_type)?;
                    let y = Self::read_float_value(&buf, base + offsets[y_idx], &properties[y_idx].prop_type)?;
                    let z = Self::read_float_value(&buf, base + offsets[z_idx], &properties[z_idx].prop_type)?;
                    let position = Self::convert_coords(x, y, z);

                    let color = Self::read_point_color(&buf, base, &offsets, &properties, color_mode, &color_indices)?;
                    bb.extend_point(&position);
                    points.push(PlyPoint { position, color });
                }
            }
            PlyFormat::Ascii => {
                let mut line = String::new();
                for i in 0..vertex_count {
                    line.clear();
                    reader.read_line(&mut line)?;
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() < properties.len() {
                        return Err(PlyError::Data(format!("行 {} のフィールド数が不足", i)));
                    }

                    let x: f64 = parts[x_idx].parse().map_err(|e| PlyError::Data(format!("{}", e)))?;
                    let y: f64 = parts[y_idx].parse().map_err(|e| PlyError::Data(format!("{}", e)))?;
                    let z: f64 = parts[z_idx].parse().map_err(|e| PlyError::Data(format!("{}", e)))?;
                    let position = Self::convert_coords(x, y, z);

                    let color = match (color_mode, &color_indices) {
                        (ColorMode::StandardRgb, Some((ri, gi, bi, ai))) => {
                            let r: f32 = parts[*ri].parse::<f32>().unwrap_or(255.0) / 255.0;
                            let g: f32 = parts[*gi].parse::<f32>().unwrap_or(255.0) / 255.0;
                            let b: f32 = parts[*bi].parse::<f32>().unwrap_or(255.0) / 255.0;
                            let a = ai.map(|i| parts[i].parse::<f32>().unwrap_or(255.0) / 255.0).unwrap_or(1.0);
                            [r, g, b, a]
                        }
                        (ColorMode::GaussianSh, Some((ri, gi, bi, _))) => {
                            let fr: f32 = parts[*ri].parse().unwrap_or(0.0);
                            let fg: f32 = parts[*gi].parse().unwrap_or(0.0);
                            let fb: f32 = parts[*bi].parse().unwrap_or(0.0);
                            [Self::sh_to_rgb(fr), Self::sh_to_rgb(fg), Self::sh_to_rgb(fb), 1.0]
                        }
                        _ => [1.0, 1.0, 1.0, 1.0],
                    };

                    bb.extend_point(&position);
                    points.push(PlyPoint { position, color });
                }
            }
        }

        info!("PLY読み込み完了: {} 点", points.len());
        Ok(PlyPointCloud { points, bounding_box: bb })
    }

    /// Gaussian Splatting PLYファイルを読み込み
    pub fn load_gaussian(path: &Path) -> Result<GaussianCloud, PlyError> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        let (format, vertex_count, properties) = Self::parse_header(&mut reader)?;

        let has_opacity = Self::find_property_opt(&properties, "opacity").is_some();
        let has_scale = Self::find_property_opt(&properties, "scale_0").is_some();
        let has_rot = Self::find_property_opt(&properties, "rot_0").is_some();

        if !has_opacity && !has_scale && !has_rot {
            return Err(PlyError::Header("Gaussian Splatting属性が見つかりません".into()));
        }

        let x_idx = Self::find_property(&properties, "x")?;
        let y_idx = Self::find_property(&properties, "y")?;
        let z_idx = Self::find_property(&properties, "z")?;
        let opacity_idx = Self::find_property_opt(&properties, "opacity");
        let scale_indices = if has_scale {
            Some((
                Self::find_property(&properties, "scale_0")?,
                Self::find_property(&properties, "scale_1")?,
                Self::find_property(&properties, "scale_2")?,
            ))
        } else { None };
        let rot_indices = if has_rot {
            Some((
                Self::find_property(&properties, "rot_0")?,
                Self::find_property(&properties, "rot_1")?,
                Self::find_property(&properties, "rot_2")?,
                Self::find_property(&properties, "rot_3")?,
            ))
        } else { None };

        let color_mode = Self::detect_color_mode(&properties);
        let color_indices = match color_mode {
            ColorMode::GaussianSh => Some((
                Self::find_property(&properties, "f_dc_0")?,
                Self::find_property(&properties, "f_dc_1")?,
                Self::find_property(&properties, "f_dc_2")?,
            )),
            ColorMode::StandardRgb => Some((
                Self::find_property(&properties, "red")?,
                Self::find_property(&properties, "green")?,
                Self::find_property(&properties, "blue")?,
            )),
            ColorMode::None => None,
        };

        let mut points = Vec::with_capacity(vertex_count);
        let mut bb = BoundingBox::empty();

        let vertex_size: usize = properties.iter().map(|p| p.prop_type.byte_size()).sum();
        let offsets = Self::compute_offsets(&properties);

        match format {
            PlyFormat::BinaryLittleEndian => {
                let mut buf = vec![0u8; vertex_size * vertex_count];
                reader.read_exact(&mut buf)?;

                for i in 0..vertex_count {
                    let base = i * vertex_size;
                    let gp = Self::parse_gaussian_point(&buf, base, &offsets, &properties,
                        x_idx, y_idx, z_idx, opacity_idx, scale_indices, rot_indices,
                        color_mode, &color_indices)?;
                    bb.extend_point(&gp.position);
                    points.push(gp);
                }
            }
            PlyFormat::Ascii => {
                let mut line = String::new();
                for i in 0..vertex_count {
                    line.clear();
                    reader.read_line(&mut line)?;
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() < properties.len() {
                        return Err(PlyError::Data(format!("行 {} のフィールド数が不足", i)));
                    }

                    let x: f64 = parts[x_idx].parse().map_err(|e| PlyError::Data(format!("{}", e)))?;
                    let y: f64 = parts[y_idx].parse().map_err(|e| PlyError::Data(format!("{}", e)))?;
                    let z: f64 = parts[z_idx].parse().map_err(|e| PlyError::Data(format!("{}", e)))?;
                    let position = Self::convert_coords(x, y, z);

                    let color = match (color_mode, &color_indices) {
                        (ColorMode::GaussianSh, Some((ri, gi, bi))) => {
                            let fr: f32 = parts[*ri].parse().unwrap_or(0.0);
                            let fg: f32 = parts[*gi].parse().unwrap_or(0.0);
                            let fb: f32 = parts[*bi].parse().unwrap_or(0.0);
                            [Self::sh_to_rgb(fr), Self::sh_to_rgb(fg), Self::sh_to_rgb(fb)]
                        }
                        (ColorMode::StandardRgb, Some((ri, gi, bi))) => {
                            [parts[*ri].parse::<f32>().unwrap_or(255.0) / 255.0,
                             parts[*gi].parse::<f32>().unwrap_or(255.0) / 255.0,
                             parts[*bi].parse::<f32>().unwrap_or(255.0) / 255.0]
                        }
                        _ => [1.0, 1.0, 1.0],
                    };

                    let opacity = opacity_idx.map(|oi| sigmoid(parts[oi].parse().unwrap_or(0.0))).unwrap_or(1.0);
                    let scale = scale_indices.map(|(s0, s1, s2)| {
                        let r0: f32 = parts[s0].parse().unwrap_or(0.0);
                        let r1: f32 = parts[s1].parse().unwrap_or(0.0);
                        let r2: f32 = parts[s2].parse().unwrap_or(0.0);
                        [r0.exp() * 1000.0, r2.exp() * 1000.0, r1.exp() * 1000.0]
                    }).unwrap_or([1.0, 1.0, 1.0]);
                    let rotation = rot_indices.map(|(q0, q1, q2, q3)| {
                        let w: f32 = parts[q0].parse().unwrap_or(1.0);
                        let qx: f32 = parts[q1].parse().unwrap_or(0.0);
                        let qy: f32 = parts[q2].parse().unwrap_or(0.0);
                        let qz: f32 = parts[q3].parse().unwrap_or(0.0);
                        let len = (w * w + qx * qx + qy * qy + qz * qz).sqrt().max(1e-8);
                        [w / len, qx / len, -qz / len, qy / len]
                    }).unwrap_or([1.0, 0.0, 0.0, 0.0]);

                    bb.extend_point(&position);
                    points.push(GaussianPoint { position, color, opacity, scale, rotation });
                }
            }
        }

        info!("Gaussian PLY読み込み完了: {} スプラット", points.len());
        Ok(GaussianCloud { points, bounding_box: bb })
    }

    // ── 内部ヘルパー ──

    fn parse_header(reader: &mut BufReader<std::fs::File>) -> Result<(PlyFormat, usize, Vec<PropertyDef>), PlyError> {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.trim() != "ply" {
            return Err(PlyError::Header("PLYマジックが見つかりません".into()));
        }

        let mut format = None;
        let mut vertex_count = None;
        let mut properties = Vec::new();
        let mut in_vertex_element = false;

        loop {
            line.clear();
            reader.read_line(&mut line)?;
            let trimmed = line.trim();
            if trimmed == "end_header" { break; }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.is_empty() { continue; }

            match parts[0] {
                "format" if parts.len() >= 2 => {
                    format = Some(match parts[1] {
                        "ascii" => PlyFormat::Ascii,
                        "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                        other => return Err(PlyError::UnsupportedFormat(other.to_string())),
                    });
                }
                "element" if parts.len() >= 3 && parts[1] == "vertex" => {
                    vertex_count = Some(parts[2].parse::<usize>().map_err(|e| PlyError::Header(format!("{}", e)))?);
                    in_vertex_element = true;
                }
                "element" => { in_vertex_element = false; }
                "property" if in_vertex_element && parts.len() >= 3 && parts[1] != "list" => {
                    properties.push(PropertyDef {
                        name: parts[2].to_string(),
                        prop_type: PropertyType::from_str(parts[1])?,
                    });
                }
                _ => {}
            }
        }

        Ok((
            format.ok_or_else(|| PlyError::Header("formatが指定されていません".into()))?,
            vertex_count.ok_or_else(|| PlyError::Header("element vertexが見つかりません".into()))?,
            properties,
        ))
    }

    fn compute_offsets(properties: &[PropertyDef]) -> Vec<usize> {
        let mut offs = Vec::with_capacity(properties.len());
        let mut cur = 0usize;
        for p in properties {
            offs.push(cur);
            cur += p.prop_type.byte_size();
        }
        offs
    }

    fn detect_color_mode(properties: &[PropertyDef]) -> ColorMode {
        if properties.iter().any(|p| p.name == "f_dc_0") {
            ColorMode::GaussianSh
        } else if properties.iter().any(|p| p.name == "red") {
            ColorMode::StandardRgb
        } else {
            ColorMode::None
        }
    }

    fn find_property(properties: &[PropertyDef], name: &str) -> Result<usize, PlyError> {
        properties.iter().position(|p| p.name == name)
            .ok_or_else(|| PlyError::Header(format!("プロパティ '{}' が見つかりません", name)))
    }

    fn find_property_opt(properties: &[PropertyDef], name: &str) -> Option<usize> {
        properties.iter().position(|p| p.name == name)
    }

    fn convert_coords(x: f64, y: f64, z: f64) -> Point3 {
        Point3::new(x * 1000.0, -z * 1000.0, y * 1000.0)
    }

    fn sh_to_rgb(f_dc: f32) -> f32 {
        (f_dc * 0.282_094_8 + 0.5).clamp(0.0, 1.0)
    }

    fn read_float_value(buf: &[u8], offset: usize, prop_type: &PropertyType) -> Result<f64, PlyError> {
        match prop_type {
            PropertyType::Float => {
                let bytes: [u8; 4] = buf[offset..offset + 4].try_into().map_err(|_| PlyError::Data("float読み取りエラー".into()))?;
                Ok(f32::from_le_bytes(bytes) as f64)
            }
            PropertyType::Double => {
                let bytes: [u8; 8] = buf[offset..offset + 8].try_into().map_err(|_| PlyError::Data("double読み取りエラー".into()))?;
                Ok(f64::from_le_bytes(bytes))
            }
            _ => Err(PlyError::Data(format!("座標に非浮動小数点型: {:?}", prop_type))),
        }
    }

    fn read_color_value(buf: &[u8], offset: usize, prop_type: &PropertyType) -> Result<f32, PlyError> {
        match prop_type {
            PropertyType::Uchar => Ok(buf[offset] as f32 / 255.0),
            PropertyType::Float => {
                let bytes: [u8; 4] = buf[offset..offset + 4].try_into().map_err(|_| PlyError::Data("float色読み取りエラー".into()))?;
                Ok(f32::from_le_bytes(bytes))
            }
            PropertyType::Ushort => {
                let bytes: [u8; 2] = buf[offset..offset + 2].try_into().map_err(|_| PlyError::Data("ushort色読み取りエラー".into()))?;
                Ok(u16::from_le_bytes(bytes) as f32 / 65535.0)
            }
            _ => Ok(1.0),
        }
    }

    fn read_point_color(
        buf: &[u8], base: usize, offsets: &[usize], properties: &[PropertyDef],
        color_mode: ColorMode, color_indices: &Option<(usize, usize, usize, Option<usize>)>,
    ) -> Result<[f32; 4], PlyError> {
        match (color_mode, color_indices) {
            (ColorMode::StandardRgb, Some((ri, gi, bi, ai))) => {
                let r = Self::read_color_value(buf, base + offsets[*ri], &properties[*ri].prop_type)?;
                let g = Self::read_color_value(buf, base + offsets[*gi], &properties[*gi].prop_type)?;
                let b = Self::read_color_value(buf, base + offsets[*bi], &properties[*bi].prop_type)?;
                let a = if let Some(ai) = ai {
                    Self::read_color_value(buf, base + offsets[*ai], &properties[*ai].prop_type)?
                } else { 1.0 };
                Ok([r, g, b, a])
            }
            (ColorMode::GaussianSh, Some((ri, gi, bi, _))) => {
                let fr = Self::read_float_value(buf, base + offsets[*ri], &properties[*ri].prop_type)? as f32;
                let fg = Self::read_float_value(buf, base + offsets[*gi], &properties[*gi].prop_type)? as f32;
                let fb = Self::read_float_value(buf, base + offsets[*bi], &properties[*bi].prop_type)? as f32;
                Ok([Self::sh_to_rgb(fr), Self::sh_to_rgb(fg), Self::sh_to_rgb(fb), 1.0])
            }
            _ => Ok([1.0, 1.0, 1.0, 1.0]),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_gaussian_point(
        buf: &[u8], base: usize, offsets: &[usize], properties: &[PropertyDef],
        x_idx: usize, y_idx: usize, z_idx: usize,
        opacity_idx: Option<usize>,
        scale_indices: Option<(usize, usize, usize)>,
        rot_indices: Option<(usize, usize, usize, usize)>,
        color_mode: ColorMode,
        color_indices: &Option<(usize, usize, usize)>,
    ) -> Result<GaussianPoint, PlyError> {
        let x = Self::read_float_value(buf, base + offsets[x_idx], &properties[x_idx].prop_type)?;
        let y = Self::read_float_value(buf, base + offsets[y_idx], &properties[y_idx].prop_type)?;
        let z = Self::read_float_value(buf, base + offsets[z_idx], &properties[z_idx].prop_type)?;
        let position = Self::convert_coords(x, y, z);

        let color = match (color_mode, color_indices) {
            (ColorMode::GaussianSh, Some((ri, gi, bi))) => {
                let fr = Self::read_float_value(buf, base + offsets[*ri], &properties[*ri].prop_type)? as f32;
                let fg = Self::read_float_value(buf, base + offsets[*gi], &properties[*gi].prop_type)? as f32;
                let fb = Self::read_float_value(buf, base + offsets[*bi], &properties[*bi].prop_type)? as f32;
                [Self::sh_to_rgb(fr), Self::sh_to_rgb(fg), Self::sh_to_rgb(fb)]
            }
            (ColorMode::StandardRgb, Some((ri, gi, bi))) => {
                let r = Self::read_color_value(buf, base + offsets[*ri], &properties[*ri].prop_type)?;
                let g = Self::read_color_value(buf, base + offsets[*gi], &properties[*gi].prop_type)?;
                let b = Self::read_color_value(buf, base + offsets[*bi], &properties[*bi].prop_type)?;
                [r, g, b]
            }
            _ => [1.0, 1.0, 1.0],
        };

        let opacity = opacity_idx.map(|oi| {
            let raw = Self::read_float_value(buf, base + offsets[oi], &properties[oi].prop_type).unwrap_or(0.0) as f32;
            sigmoid(raw)
        }).unwrap_or(1.0);

        let scale = scale_indices.map(|(s0, s1, s2)| {
            let r0 = Self::read_float_value(buf, base + offsets[s0], &properties[s0].prop_type).unwrap_or(0.0) as f32;
            let r1 = Self::read_float_value(buf, base + offsets[s1], &properties[s1].prop_type).unwrap_or(0.0) as f32;
            let r2 = Self::read_float_value(buf, base + offsets[s2], &properties[s2].prop_type).unwrap_or(0.0) as f32;
            [r0.exp() * 1000.0, r2.exp() * 1000.0, r1.exp() * 1000.0]
        }).unwrap_or([1.0, 1.0, 1.0]);

        let rotation = rot_indices.map(|(q0, q1, q2, q3)| {
            let w = Self::read_float_value(buf, base + offsets[q0], &properties[q0].prop_type).unwrap_or(1.0) as f32;
            let qx = Self::read_float_value(buf, base + offsets[q1], &properties[q1].prop_type).unwrap_or(0.0) as f32;
            let qy = Self::read_float_value(buf, base + offsets[q2], &properties[q2].prop_type).unwrap_or(0.0) as f32;
            let qz = Self::read_float_value(buf, base + offsets[q3], &properties[q3].prop_type).unwrap_or(0.0) as f32;
            let len = (w * w + qx * qx + qy * qy + qz * qz).sqrt().max(1e-8);
            [w / len, qx / len, -qz / len, qy / len]
        }).unwrap_or([1.0, 0.0, 0.0, 0.0]);

        Ok(GaussianPoint { position, color, opacity, scale, rotation })
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
