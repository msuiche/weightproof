//! Rank-1 GLP project vectors and deterministic fixed-point quantization.
//!
//! G0 deliberately supports only `project`, spec version 1, and
//! `residual_stream_post_layer`. Other GLP semantics fail closed.

use alloc::{collections::BTreeSet, string::String, vec::Vec};

use crate::fp::{digest::Digest, heads::configuration_digest};

const MODE_PROJECT: &str = "project";
const HOOK_POST_LAYER: &str = "residual_stream_post_layer";
const GGUF_F32: u32 = 0;
const GGUF_STRING: u32 = 8;
const GGUF_ARRAY: u32 = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlpQuantization {
    pub direction_shift: u32,
    pub alpha_shift: u32,
}

impl GlpQuantization {
    pub fn new(direction_shift: u32, alpha_shift: u32) -> Option<Self> {
        if !(1..=14).contains(&direction_shift)
            || !(1..=31).contains(&alpha_shift)
            || 2 * direction_shift + alpha_shift >= 127
        {
            return None;
        }
        Some(Self {
            direction_shift,
            alpha_shift,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlpLayer {
    pub layer: usize,
    pub direction: Vec<i128>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlpSteering {
    artifact: Digest,
    quantization: GlpQuantization,
    alpha: i128,
    layers: Vec<GlpLayer>,
}

fn push_i128(bytes: &mut Vec<u8>, value: i128) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn fixed_bytes(quantization: GlpQuantization, alpha: i128, layers: &[GlpLayer]) -> Vec<u8> {
    let mut bytes = b"weightproof-glp-fixed-v1\0project\0residual_stream_post_layer".to_vec();
    bytes.extend_from_slice(&quantization.direction_shift.to_le_bytes());
    bytes.extend_from_slice(&quantization.alpha_shift.to_le_bytes());
    push_i128(&mut bytes, alpha);
    bytes.extend_from_slice(&(layers.len() as u64).to_le_bytes());
    for layer in layers {
        bytes.extend_from_slice(&(layer.layer as u64).to_le_bytes());
        bytes.extend_from_slice(&(layer.direction.len() as u64).to_le_bytes());
        for &value in &layer.direction {
            push_i128(&mut bytes, value);
        }
    }
    bytes
}

fn valid_layers(layers: &[GlpLayer]) -> bool {
    let width = layers.first().map_or(0, |layer| layer.direction.len());
    width != 0
        && layers
            .iter()
            .all(|layer| layer.layer != 0 && layer.direction.len() == width)
        && layers.windows(2).all(|pair| pair[0].layer < pair[1].layer)
}

impl GlpSteering {
    pub fn from_fixed_point(
        quantization: GlpQuantization,
        alpha: i128,
        layers: Vec<GlpLayer>,
    ) -> Option<Self> {
        GlpQuantization::new(quantization.direction_shift, quantization.alpha_shift)?;
        if !valid_layers(&layers) {
            return None;
        }
        let artifact = configuration_digest(&[&fixed_bytes(quantization, alpha, &layers)]);
        Some(Self {
            artifact,
            quantization,
            alpha,
            layers,
        })
    }

    pub fn from_gguf(bytes: &[u8], quantization: GlpQuantization) -> Option<Self> {
        GlpQuantization::new(quantization.direction_shift, quantization.alpha_shift)?;
        let parsed = parse_gguf(bytes)?;
        let mut layers = Vec::with_capacity(parsed.tensors.len());
        for tensor in parsed.tensors {
            let start = parsed
                .data_start
                .checked_add(usize::try_from(tensor.offset).ok()?)?;
            let byte_len = tensor.width.checked_mul(4)?;
            let data = bytes.get(start..start.checked_add(byte_len)?)?;
            let mut direction = Vec::with_capacity(tensor.width);
            let mut norm = 0.0f64;
            for value in data.chunks_exact(4) {
                let value = f32::from_le_bytes(value.try_into().ok()?);
                if !value.is_finite() {
                    return None;
                }
                norm += (value as f64) * (value as f64);
                direction.push(quantize(value, quantization.direction_shift)?);
            }
            // F32 export and dot-product accumulation can move the norm slightly.
            // A 0.1% window rejects missing/zero/scaled vectors without rejecting
            // ordinary serialized unit directions.
            if (norm - 1.0).abs() > 0.001 {
                return None;
            }
            layers.push(GlpLayer {
                layer: tensor.layer,
                direction,
            });
        }
        layers.sort_by_key(|layer| layer.layer);
        let alpha = quantize(parsed.alpha, quantization.alpha_shift)?;
        let fixed = Self::from_fixed_point(quantization, alpha, layers)?;
        Some(Self {
            artifact: configuration_digest(&[bytes]),
            ..fixed
        })
    }

    pub fn alpha(&self) -> i128 {
        self.alpha
    }

    pub fn rounding_shift(&self) -> u32 {
        2 * self.quantization.direction_shift + self.quantization.alpha_shift
    }

    pub fn direction(&self, layer: usize) -> Option<&[i128]> {
        self.layers
            .binary_search_by_key(&layer, |entry| entry.layer)
            .ok()
            .map(|index| self.layers[index].direction.as_slice())
    }

    pub fn layers(&self) -> &[GlpLayer] {
        &self.layers
    }

    pub fn commitment(&self) -> Digest {
        let mut bytes = b"weightproof-glp-statement-v1".to_vec();
        bytes.extend_from_slice(&self.artifact.to_le_bytes());
        bytes.extend_from_slice(&fixed_bytes(self.quantization, self.alpha, &self.layers));
        configuration_digest(&[&bytes])
    }
}

fn round_nearest_even(value: f64) -> Option<i128> {
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return None;
    }
    let floor = libm::floor(value);
    let fraction = value - floor;
    let rounded = if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 {
        floor
    } else if (floor as i128) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };
    Some(rounded as i128)
}

fn quantize(value: f32, shift: u32) -> Option<i128> {
    round_nearest_even(value as f64 * (1u64 << shift) as f64)
}

struct Tensor {
    layer: usize,
    width: usize,
    offset: u64,
}

struct ParsedGlp {
    alpha: f32,
    data_start: usize,
    tensors: Vec<Tensor>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Option<&'a [u8]> {
        let end = self.position.checked_add(length)?;
        let value = self.bytes.get(self.position..end)?;
        self.position = end;
        Some(value)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn string(&mut self) -> Option<&'a str> {
        let length = usize::try_from(self.u64()?).ok()?;
        core::str::from_utf8(self.take(length)?).ok()
    }

    fn skip_value(&mut self, value_type: u32) -> Option<()> {
        let width = match value_type {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=7 => 4,
            GGUF_STRING => {
                self.string()?;
                return Some(());
            }
            GGUF_ARRAY => {
                let element_type = self.u32()?;
                if element_type == GGUF_ARRAY {
                    return None;
                }
                let count = self.u64()?;
                for _ in 0..count {
                    self.skip_value(element_type)?;
                }
                return Some(());
            }
            10..=12 => 8,
            _ => return None,
        };
        self.take(width)?;
        Some(())
    }
}

fn parse_layer(name: &str) -> Option<Option<usize>> {
    let Some(suffix) = name.strip_prefix("direction.") else {
        return Some(None);
    };
    if suffix.is_empty()
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
        || (suffix.len() > 1 && suffix.starts_with('0'))
    {
        return None;
    }
    let layer = suffix.parse::<usize>().ok()?;
    if layer == 0 {
        return None;
    }
    Some(Some(layer))
}

fn parse_gguf(bytes: &[u8]) -> Option<ParsedGlp> {
    let mut reader = Reader { bytes, position: 0 };
    if reader.take(4)? != b"GGUF" || reader.u32()? != 3 {
        return None;
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    if tensor_count > bytes.len() as u64 || metadata_count > bytes.len() as u64 {
        return None;
    }

    let mut keys = BTreeSet::new();
    let mut mode: Option<String> = None;
    let mut spec_version = None;
    let mut alpha = None;
    let mut hook: Option<String> = None;
    let mut rank = 1u32;
    let mut alignment = 32u32;
    for _ in 0..metadata_count {
        let key = reader.string()?;
        if !keys.insert(key) {
            return None;
        }
        let value_type = reader.u32()?;
        match key {
            "glp.mode" => {
                if value_type != GGUF_STRING {
                    return None;
                }
                mode = Some(reader.string()?.into());
            }
            "glp.spec_version" => {
                if value_type != 4 {
                    return None;
                }
                spec_version = Some(reader.u32()?);
            }
            "glp.alpha_default" => {
                if value_type != 6 {
                    return None;
                }
                alpha = Some(reader.f32()?);
            }
            "glp.hook_point" => {
                if value_type != GGUF_STRING {
                    return None;
                }
                hook = Some(reader.string()?.into());
            }
            "glp.rank" => {
                if value_type != 4 {
                    return None;
                }
                rank = reader.u32()?;
            }
            "general.alignment" => {
                if value_type != 4 {
                    return None;
                }
                alignment = reader.u32()?;
            }
            "glp.dir_scales" | "glp.layer_scales" => return None,
            _ => reader.skip_value(value_type)?,
        }
    }
    if mode.as_deref()? != MODE_PROJECT
        || spec_version? != 1
        || hook.as_deref()? != HOOK_POST_LAYER
        || rank != 1
        || !alignment.is_power_of_two()
    {
        return None;
    }
    let alpha = alpha?;
    if !alpha.is_finite() {
        return None;
    }

    let mut tensors = Vec::new();
    let mut width = None;
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let dimensions = reader.u32()?;
        if dimensions > 32 {
            return None;
        }
        let layer = parse_layer(name)?;
        if layer.is_some() && dimensions != 1 {
            return None;
        }
        let mut tensor_width = 1usize;
        for _ in 0..dimensions {
            let dimension = usize::try_from(reader.u64()?).ok()?;
            tensor_width = tensor_width.checked_mul(dimension)?;
        }
        let tensor_type = reader.u32()?;
        let offset = reader.u64()?;
        if let Some(layer) = layer {
            if tensor_type != GGUF_F32 || tensor_width == 0 {
                return None;
            }
            if width.is_some_and(|width| width != tensor_width) {
                return None;
            }
            width = Some(tensor_width);
            tensors.push(Tensor {
                layer,
                width: tensor_width,
                offset,
            });
        }
    }
    if tensors.is_empty() {
        return None;
    }
    let data_start =
        reader.position.checked_add(alignment as usize - 1)? & !(alignment as usize - 1);
    if data_start > bytes.len() {
        return None;
    }
    Some(ParsedGlp {
        alpha,
        data_start,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn gguf(
        mode: &str,
        hook: &str,
        version: u32,
        rank: u32,
        tensors: &[(&str, &[f32])],
    ) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&6u64.to_le_bytes());
        for (key, value) in [("glp.mode", mode), ("glp.hook_point", hook)] {
            string(&mut bytes, key);
            bytes.extend_from_slice(&GGUF_STRING.to_le_bytes());
            string(&mut bytes, value);
        }
        for (key, value) in [
            ("glp.spec_version", version),
            ("glp.rank", rank),
            ("general.alignment", 32),
        ] {
            string(&mut bytes, key);
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        string(&mut bytes, "glp.alpha_default");
        bytes.extend_from_slice(&6u32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());

        let mut offset = 0u64;
        for (name, values) in tensors {
            string(&mut bytes, name);
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&GGUF_F32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            offset += (values.len() * 4) as u64;
        }
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        for (_, values) in tensors {
            for value in *values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    fn quantization() -> GlpQuantization {
        GlpQuantization::new(1, 1).unwrap()
    }

    #[test]
    fn parses_rank_one_project_gguf() {
        let bytes = gguf(
            "project",
            "residual_stream_post_layer",
            1,
            1,
            &[("direction.1", &[1.0, 0.0])],
        );
        let steering = GlpSteering::from_gguf(&bytes, quantization()).unwrap();
        assert_eq!(steering.alpha(), 4);
        assert_eq!(steering.rounding_shift(), 3);
        assert_eq!(steering.direction(1), Some(&[2, 0][..]));
        assert_eq!(steering.direction(2), None);
    }

    #[test]
    fn rejects_unsupported_or_malformed_gguf() {
        let valid = gguf(
            "project",
            "residual_stream_post_layer",
            1,
            1,
            &[("direction.1", &[1.0, 0.0])],
        );
        for length in 0..valid.len() {
            assert!(GlpSteering::from_gguf(&valid[..length], quantization()).is_none());
        }
        for (mode, hook, version, rank, name, values) in [
            (
                "add",
                "residual_stream_post_layer",
                1,
                1,
                "direction.1",
                &[1.0, 0.0][..],
            ),
            (
                "project",
                "ffn_out_pre_residual",
                1,
                1,
                "direction.1",
                &[1.0, 0.0][..],
            ),
            (
                "project",
                "residual_stream_post_layer",
                2,
                1,
                "direction.1",
                &[1.0, 0.0][..],
            ),
            (
                "project",
                "residual_stream_post_layer",
                1,
                2,
                "direction.1",
                &[1.0, 0.0][..],
            ),
            (
                "project",
                "residual_stream_post_layer",
                1,
                1,
                "direction.0",
                &[1.0, 0.0][..],
            ),
            (
                "project",
                "residual_stream_post_layer",
                1,
                1,
                "direction.1",
                &[f32::NAN, 0.0][..],
            ),
        ] {
            let bytes = gguf(mode, hook, version, rank, &[(name, values)]);
            assert!(GlpSteering::from_gguf(&bytes, quantization()).is_none());
        }
        let duplicate = gguf(
            "project",
            "residual_stream_post_layer",
            1,
            1,
            &[("direction.1", &[1.0, 0.0]), ("direction.1", &[1.0, 0.0])],
        );
        assert!(GlpSteering::from_gguf(&duplicate, quantization()).is_none());
        let inconsistent = gguf(
            "project",
            "residual_stream_post_layer",
            1,
            1,
            &[("direction.1", &[1.0, 0.0]), ("direction.2", &[1.0])],
        );
        assert!(GlpSteering::from_gguf(&inconsistent, quantization()).is_none());
    }

    #[test]
    fn fixed_point_constructor_validates_and_binds_contents() {
        let layer = GlpLayer {
            layer: 1,
            direction: vec![2, 0],
        };
        let steering =
            GlpSteering::from_fixed_point(quantization(), 4, vec![layer.clone()]).unwrap();
        let changed = GlpSteering::from_fixed_point(quantization(), 2, vec![layer]).unwrap();
        assert_ne!(steering.commitment(), changed.commitment());
        let mut unordered = steering.layers().to_vec();
        unordered.push(GlpLayer {
            layer: 1,
            direction: vec![0, 2],
        });
        assert!(GlpSteering::from_fixed_point(quantization(), 4, unordered).is_none());
    }

    #[test]
    fn quantization_rounds_ties_to_even() {
        assert_eq!(quantize(0.25, 1), Some(0));
        assert_eq!(quantize(0.75, 1), Some(2));
        assert_eq!(quantize(1.25, 1), Some(2));
        let bytes = gguf(
            "project",
            "residual_stream_post_layer",
            1,
            1,
            &[("direction.1", &[1.0, 0.0])],
        );
        let invalid = GlpQuantization {
            direction_shift: 64,
            alpha_shift: 1,
        };
        assert!(GlpSteering::from_gguf(&bytes, invalid).is_none());
    }
}
