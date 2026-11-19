//! Promptable segmentation with SAM-family ONNX models (MobileSAM's encoder + SAM's decoder).
//!
//! Layout of the models (as exported by the MobileSAM ONNX scripts):
//! - encoder: `input_image` `[h, w, 3]` RGB in `0..=255`, long side [`SAM_SIDE`]; it normalises
//!   and pads internally. Output `image_embeddings` `[1, 256, 64, 64]`.
//! - decoder: `image_embeddings`, `point_coords [1, n, 2]` (encoder-frame pixels),
//!   `point_labels [1, n]` (1 include, 0 exclude, 2/3 box corners, -1 padding), `mask_input`
//!   `[1, 1, 256, 256]`, `has_mask_input [1]`, `orig_im_size [2]`. Output `masks` (logits) at
//!   `orig_im_size`, which we set to the encoder frame so masks stay small; [`PartMap`] samples
//!   them back up to document pixels.

use std::collections::HashMap;
use std::io::Cursor;

use tract_onnx::prelude::*;

use crate::{MlError, Result};

/// The encoder's input size: the long side of the image the network sees.
pub const SAM_SIDE: usize = 1024;
/// Most include/exclude points one prompt may have.
pub const MAX_POINTS: usize = 32;
/// Most parts one [`PartMap`] resolves.
pub const MAX_PARTS: usize = 64;
/// Largest source image side accepted (PSB-scale documents are downsampled by the caller).
const MAX_SOURCE_SIDE: usize = 1 << 20;

type Plan = std::sync::Arc<TypedRunnableModel>;

fn model_err(what: &str, e: impl std::fmt::Display) -> MlError {
    MlError::Model(format!("{what}: {e}"))
}

/// An RGB image with samples in `0..=1`, row-major, plus the size of the document area it
/// stands for (it may be a downsampled copy of that area).
pub struct Rgb<'a> {
    pub width: usize,
    pub height: usize,
    pub px: &'a [[f32; 3]],
    /// Width and height of the document area, in document pixels.
    pub source_size: (usize, usize),
}

/// Include or exclude a point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointLabel {
    Include,
    Exclude,
}

/// What to segment: points and/or a box, in document pixels relative to the analysed area.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Prompt {
    pub points: Vec<(f32, f32, PointLabel)>,
    /// `[x0, y0, x1, y1]`.
    pub rect: Option<[f32; 4]>,
}

/// The two networks of a SAM-family model, kept as bytes: the plans depend on the image size,
/// so they are built per analysed image.
pub struct SamModel {
    encoder: Vec<u8>,
    decoder: Vec<u8>,
}

impl SamModel {
    /// Check that both files parse as ONNX and keep them.
    pub fn from_bytes(encoder: Vec<u8>, decoder: Vec<u8>) -> Result<Self> {
        for (name, bytes) in [("encoder", &encoder), ("decoder", &decoder)] {
            if bytes.is_empty() {
                return Err(MlError::Model(format!("the {name} file is empty")));
            }
            let model = tract_onnx::onnx().model_for_read(&mut Cursor::new(bytes.as_slice())).map_err(|e| model_err(name, e))?;
            let expected = if name == "encoder" { 1 } else { 6 };
            if model.inputs.len() != expected {
                return Err(MlError::Model(format!("the {name} has {} inputs, expected {expected}: not a SAM {name}", model.inputs.len())));
            }
        }
        Ok(SamModel { encoder, decoder })
    }

    /// Run the encoder once for an image; prompts are then cheap.
    pub fn embed(&self, img: &Rgb) -> Result<Embedding> {
        let (sw, sh) = frame_size(img)?;
        let input = resize_to_tensor(img, sw, sh)?;
        let enc = tract_onnx::onnx()
            .model_for_read(&mut Cursor::new(self.encoder.as_slice()))
            .and_then(|m| m.with_input_fact(0, f32::fact([sh, sw, 3]).into()))
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| model_err("encoder", e))?;
        let mut out = enc.run(tvec!(input.into())).map_err(|e| model_err("encoder run", e))?;
        if out.is_empty() {
            return Err(MlError::Model("the encoder returned nothing".into()));
        }
        let embeddings = out.remove(0);
        if embeddings.shape() != [1, 256, 64, 64] {
            return Err(MlError::Model(format!("unexpected embedding shape {:?}", embeddings.shape())));
        }
        let (src_w, src_h) = img.source_size;
        Ok(Embedding {
            embeddings,
            decoder: self.decoder.clone(),
            plans: HashMap::new(),
            frame: (sw, sh),
            scale: (sw as f32 / src_w as f32, sh as f32 / src_h as f32),
        })
    }
}

/// The encoder frame for an image: long side [`SAM_SIDE`], aspect kept.
fn frame_size(img: &Rgb) -> Result<(usize, usize)> {
    let (w, h) = (img.width, img.height);
    let (sw, sh) = img.source_size;
    if w == 0 || h == 0 || sw == 0 || sh == 0 {
        return Err(MlError::Input("the image is empty".into()));
    }
    if sw > MAX_SOURCE_SIDE || sh > MAX_SOURCE_SIDE {
        return Err(MlError::Input("the image is too large to analyse".into()));
    }
    if w.checked_mul(h) != Some(img.px.len()) {
        return Err(MlError::Input("the pixel buffer does not match the image size".into()));
    }
    let long = sw.max(sh) as f64;
    let fw = ((sw as f64 * SAM_SIDE as f64 / long).round() as usize).clamp(1, SAM_SIDE);
    let fh = ((sh as f64 * SAM_SIDE as f64 / long).round() as usize).clamp(1, SAM_SIDE);
    Ok((fw, fh))
}

/// Bilinear resize to `fw × fh`, as the encoder's `[h, w, 3]` input in `0..=255`.
fn resize_to_tensor(img: &Rgb, fw: usize, fh: usize) -> Result<Tensor> {
    let (w, h) = (img.width, img.height);
    let (kx, ky) = (w as f32 / fw as f32, h as f32 / fh as f32);
    let mut data = Vec::with_capacity(fw * fh * 3);
    for oy in 0..fh {
        let fy = ((oy as f32 + 0.5) * ky - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for ox in 0..fw {
            let fx = ((ox as f32 + 0.5) * kx - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx.fract());
            let x1 = (x0 + 1).min(w - 1);
            let at = |x: usize, y: usize| img.px.get(y * w + x).copied().unwrap_or([0.0; 3]);
            let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
            for ch in 0..3 {
                let top = a[ch] + (b[ch] - a[ch]) * tx;
                let bot = c[ch] + (d[ch] - c[ch]) * tx;
                let v = top + (bot - top) * ty;
                data.push((if v.is_finite() { v } else { 0.0 }).clamp(0.0, 1.0) * 255.0);
            }
        }
    }
    tract_ndarray::Array3::from_shape_vec((fh, fw, 3), data).map(Tensor::from).map_err(|e| MlError::Input(e.to_string()))
}

/// An analysed image: its embeddings and the decoder plans built so far (one per point count).
pub struct Embedding {
    embeddings: TValue,
    decoder: Vec<u8>,
    plans: HashMap<usize, Plan>,
    /// Encoder frame (width, height).
    frame: (usize, usize),
    /// Document pixels → frame pixels.
    scale: (f32, f32),
}

impl Embedding {
    /// Mask logits for one prompt, in the encoder frame (`> 0` means inside).
    pub fn logits(&mut self, prompt: &Prompt) -> Result<Logits> {
        if prompt.points.len() > MAX_POINTS {
            return Err(MlError::Input(format!("a part can have at most {MAX_POINTS} points")));
        }
        if prompt.points.is_empty() && prompt.rect.is_none() {
            return Err(MlError::Input("a part needs at least one point or a box".into()));
        }
        let (sx, sy) = self.scale;
        let (fw, fh) = self.frame;
        let mut coords: Vec<f32> = Vec::new();
        let mut labels: Vec<f32> = Vec::new();
        for &(x, y, label) in &prompt.points {
            if !x.is_finite() || !y.is_finite() {
                return Err(MlError::Input("a point is not a number".into()));
            }
            coords.extend([(x * sx).clamp(0.0, fw as f32), (y * sy).clamp(0.0, fh as f32)]);
            labels.push(if label == PointLabel::Include { 1.0 } else { 0.0 });
        }
        if let Some([x0, y0, x1, y1]) = prompt.rect {
            if ![x0, y0, x1, y1].iter().all(|v| v.is_finite()) || x1 <= x0 || y1 <= y0 {
                return Err(MlError::Input("the box is empty or not a number".into()));
            }
            coords.extend([(x0 * sx).clamp(0.0, fw as f32), (y0 * sy).clamp(0.0, fh as f32)]);
            coords.extend([(x1 * sx).clamp(0.0, fw as f32), (y1 * sy).clamp(0.0, fh as f32)]);
            labels.extend([2.0, 3.0]);
        } else {
            // SAM expects a padding point when there is no box.
            coords.extend([0.0, 0.0]);
            labels.push(-1.0);
        }
        let n = labels.len();
        let plan = self.plan(n)?.clone();
        let pc = tract_ndarray::Array3::from_shape_vec((1, n, 2), coords).map(Tensor::from).map_err(|e| MlError::Input(e.to_string()))?;
        let pl = tract_ndarray::Array2::from_shape_vec((1, n), labels).map(Tensor::from).map_err(|e| MlError::Input(e.to_string()))?;
        let mask_in = Tensor::zero::<f32>(&[1, 1, 256, 256]).map_err(|e| model_err("decoder", e))?;
        let out = plan
            .run(tvec!(self.embeddings.clone(), pc.into(), pl.into(), mask_in.into(), tensor1(&[0.0f32]).into(), tensor1(&[fh as f32, fw as f32]).into()))
            .map_err(|e| model_err("decoder run", e))?;
        let masks = out.first().ok_or_else(|| MlError::Model("the decoder returned nothing".into()))?;
        let view = masks.to_plain_array_view::<f32>().map_err(|e| model_err("decoder output", e))?;
        let data: Vec<f32> = view.iter().map(|v| if v.is_finite() { *v } else { -1e3 }).collect();
        if data.len() != fw * fh {
            return Err(MlError::Model(format!("unexpected mask shape {:?}", view.shape())));
        }
        let score = out.get(1).and_then(|t| t.to_plain_array_view::<f32>().ok()).and_then(|v| v.iter().next().copied()).unwrap_or(0.0);
        Ok(Logits { width: fw, height: fh, scale: self.scale, data, score })
    }

    /// The decoder plan for `n` prompt points (built once per count: the decoder's last resize
    /// needs fixed shapes in tract).
    fn plan(&mut self, n: usize) -> Result<&Plan> {
        if !self.plans.contains_key(&n) {
            let (fw, fh) = self.frame;
            let plan = tract_onnx::onnx()
                .model_for_read(&mut Cursor::new(self.decoder.as_slice()))
                .and_then(|m| m.with_input_fact(0, f32::fact([1, 256, 64, 64]).into()))
                .and_then(|m| m.with_input_fact(1, f32::fact([1, n, 2]).into()))
                .and_then(|m| m.with_input_fact(2, f32::fact([1, n]).into()))
                .and_then(|m| m.with_input_fact(3, f32::fact([1, 1, 256, 256]).into()))
                .and_then(|m| m.with_input_fact(4, f32::fact([1]).into()))
                .and_then(|m| m.with_input_fact(5, tensor1(&[fh as f32, fw as f32]).into()))
                .and_then(|m| m.into_optimized())
                .and_then(|m| m.into_runnable())
                .map_err(|e| model_err("decoder", e))?;
            self.plans.insert(n, plan);
        }
        self.plans.get(&n).ok_or_else(|| MlError::Model("decoder plan missing".into()))
    }
}

/// One part's mask logits in the encoder frame.
#[derive(Debug, Clone)]
pub struct Logits {
    pub width: usize,
    pub height: usize,
    /// Document pixels → frame pixels.
    pub scale: (f32, f32),
    pub data: Vec<f32>,
    /// The decoder's own quality estimate (predicted IoU).
    pub score: f32,
}

impl Logits {
    /// Bilinear logit at a document pixel centre.
    fn at(&self, x: f32, y: f32) -> f32 {
        let fx = ((x + 0.5) * self.scale.0 - 0.5).clamp(0.0, (self.width.saturating_sub(1)) as f32);
        let fy = ((y + 0.5) * self.scale.1 - 0.5).clamp(0.0, (self.height.saturating_sub(1)) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width.saturating_sub(1)), (y0 + 1).min(self.height.saturating_sub(1)));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let g = |x: usize, y: usize| self.data.get(y * self.width + x).copied().unwrap_or(-1e3);
        let top = g(x0, y0) + (g(x1, y0) - g(x0, y0)) * tx;
        let bot = g(x0, y1) + (g(x1, y1) - g(x0, y1)) * tx;
        top + (bot - top) * ty
    }
}

/// Several parts' logits resolved against each other: each pixel goes to the part with the
/// highest logit, and only when that logit says "inside".
pub struct PartMap {
    parts: Vec<Logits>,
}

impl PartMap {
    pub fn new(parts: Vec<Logits>) -> Result<Self> {
        if parts.is_empty() {
            return Err(MlError::Input("no parts".into()));
        }
        if parts.len() > MAX_PARTS {
            return Err(MlError::Input(format!("at most {MAX_PARTS} parts")));
        }
        Ok(PartMap { parts })
    }

    pub fn len(&self) -> usize {
        self.parts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// The part owning document pixel (`x`, `y`) and its coverage (`0..=1`, soft at edges).
    pub fn at(&self, x: f32, y: f32) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (i, p) in self.parts.iter().enumerate() {
            let v = p.at(x, y);
            if best.is_none_or(|(_, b)| v > b) {
                best = Some((i, v));
            }
        }
        let (i, v) = best?;
        // Logits are about ±10 deep inside or outside; ±2 around 0 becomes a soft edge.
        let coverage = ((v + 2.0) / 4.0).clamp(0.0, 1.0);
        (coverage > 0.0).then_some((i, coverage))
    }

    /// Coverage masks for every part over a `w × h` area at (`x0`, `y0`), row-major, one
    /// `Vec<f32>` per part. Rows are computed in parallel on native targets.
    pub fn masks(&self, x0: i64, y0: i64, w: usize, h: usize) -> Vec<Vec<f32>> {
        let n = self.parts.len();
        let row =
            |y: usize| -> Vec<(usize, f32)> { (0..w).map(|x| self.at((x0 + x as i64) as f32, (y0 + y as i64) as f32).unwrap_or((usize::MAX, 0.0))).collect() };
        #[cfg(not(target_arch = "wasm32"))]
        let rows: Vec<Vec<(usize, f32)>> = {
            use rayon::prelude::*;
            (0..h).into_par_iter().map(row).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let rows: Vec<Vec<(usize, f32)>> = (0..h).map(row).collect();
        let mut out = vec![vec![0.0f32; w * h]; n];
        for (y, r) in rows.into_iter().enumerate() {
            for (x, (i, c)) in r.into_iter().enumerate() {
                if let Some(m) = out.get_mut(i)
                    && let Some(v) = m.get_mut(y * w + x)
                {
                    *v = c;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Logits {
        Logits { width: w, height: h, scale: (1.0, 1.0), data: (0..w * h).map(|i| f(i % w, i / w)).collect(), score: 1.0 }
    }

    #[test]
    fn frame_keeps_aspect_with_long_side_1024() {
        let px = vec![[0.5; 3]; 30 * 40];
        let img = Rgb { width: 30, height: 40, px: &px, source_size: (3584, 4800) };
        assert_eq!(frame_size(&img).unwrap(), (765, 1024));
    }

    #[test]
    fn mismatched_or_empty_images_are_errors_not_panics() {
        let px = vec![[0.5; 3]; 10];
        assert!(frame_size(&Rgb { width: 4, height: 4, px: &px, source_size: (4, 4) }).is_err());
        assert!(frame_size(&Rgb { width: 0, height: 4, px: &[], source_size: (0, 4) }).is_err());
        assert!(frame_size(&Rgb { width: 1, height: 1, px: &px[..1], source_size: (usize::MAX, 1) }).is_err());
    }

    #[test]
    fn overlapping_parts_go_to_the_strongest_logit() {
        // part 0 strong on the left, part 1 strong on the right, both "inside" everywhere.
        let a = logits(10, 1, |x, _| 10.0 - x as f32);
        let b = logits(10, 1, |x, _| x as f32);
        let map = PartMap::new(vec![a, b]).unwrap();
        assert_eq!(map.at(1.0, 0.0).map(|p| p.0), Some(0));
        assert_eq!(map.at(8.0, 0.0).map(|p| p.0), Some(1));
        let masks = map.masks(0, 0, 10, 1);
        assert_eq!(masks.len(), 2);
        assert!(masks[0][1] > 0.99 && masks[1][1] == 0.0);
        assert!(masks[1][8] > 0.99 && masks[0][8] == 0.0);
    }

    #[test]
    fn pixels_outside_every_part_are_unassigned() {
        let map = PartMap::new(vec![logits(4, 4, |_, _| -10.0)]).unwrap();
        assert_eq!(map.at(2.0, 2.0), None);
        assert!(map.masks(0, 0, 4, 4)[0].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn part_limits_are_errors() {
        assert!(PartMap::new(vec![]).is_err());
        assert!(PartMap::new((0..=MAX_PARTS).map(|_| logits(1, 1, |_, _| 1.0)).collect()).is_err());
    }

    #[test]
    fn garbage_model_bytes_are_an_error() {
        assert!(SamModel::from_bytes(vec![], vec![1]).is_err());
        assert!(SamModel::from_bytes(vec![1, 2, 3], vec![4, 5, 6]).is_err());
    }

    /// End to end on the real models when `PHOTOCRAFT_SAM_DIR` points at them (skipped in CI).
    #[test]
    fn real_model_segments_a_bright_square() {
        let Ok(dir) = std::env::var("PHOTOCRAFT_SAM_DIR") else { return };
        let enc = std::fs::read(format!("{dir}/mobile_sam_image_encoder.onnx")).unwrap();
        let dec = std::fs::read(format!("{dir}/sam_mask_decoder_single.onnx")).unwrap();
        let model = SamModel::from_bytes(enc, dec).unwrap();
        let (w, h) = (256usize, 256usize);
        let px: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                if (64..192).contains(&x) && (64..192).contains(&y) { [0.9, 0.1, 0.1] } else { [0.1, 0.3, 0.9] }
            })
            .collect();
        let mut emb = model.embed(&Rgb { width: w, height: h, px: &px, source_size: (w, h) }).unwrap();
        let l = emb.logits(&Prompt { points: vec![(128.0, 128.0, PointLabel::Include)], rect: None }).unwrap();
        let map = PartMap::new(vec![l]).unwrap();
        assert!(map.at(128.0, 128.0).is_some(), "centre of the square is inside");
        assert!(map.at(10.0, 10.0).is_none(), "corner outside the square is outside");
    }
}
