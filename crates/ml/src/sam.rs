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
    /// Where that area starts, in the coordinates prompts use (0, 0 for the whole image; a crop's
    /// corner for a zoomed detail analysis).
    pub origin: (usize, usize),
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
            origin: (img.origin.0 as f32, img.origin.1 as f32),
            extent: (src_w as f32, src_h as f32),
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
    /// The analysed area: its corner and size in prompt coordinates.
    origin: (f32, f32),
    extent: (f32, f32),
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
        let (ox, oy) = self.origin;
        let mut coords: Vec<f32> = Vec::new();
        let mut labels: Vec<f32> = Vec::new();
        for &(x, y, label) in &prompt.points {
            if !x.is_finite() || !y.is_finite() {
                return Err(MlError::Input("a point is not a number".into()));
            }
            coords.extend([((x - ox) * sx).clamp(0.0, fw as f32), ((y - oy) * sy).clamp(0.0, fh as f32)]);
            labels.push(if label == PointLabel::Include { 1.0 } else { 0.0 });
        }
        if let Some([x0, y0, x1, y1]) = prompt.rect {
            if ![x0, y0, x1, y1].iter().all(|v| v.is_finite()) || x1 <= x0 || y1 <= y0 {
                return Err(MlError::Input("the box is empty or not a number".into()));
            }
            coords.extend([((x0 - ox) * sx).clamp(0.0, fw as f32), ((y0 - oy) * sy).clamp(0.0, fh as f32)]);
            coords.extend([((x1 - ox) * sx).clamp(0.0, fw as f32), ((y1 - oy) * sy).clamp(0.0, fh as f32)]);
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
        Ok(Logits { width: fw, height: fh, scale: self.scale, origin: self.origin, extent: self.extent, data, score })
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
    /// The analysed area (corner and size, prompt coordinates); outside it these logits say nothing.
    pub origin: (f32, f32),
    pub extent: (f32, f32),
    pub data: Vec<f32>,
    /// The decoder's own quality estimate (predicted IoU).
    pub score: f32,
}

impl Logits {
    /// Whether pixel (`x`, `y`) lies in the analysed area.
    fn covers(&self, x: f32, y: f32) -> bool {
        let (dx, dy) = (x - self.origin.0, y - self.origin.1);
        dx >= 0.0 && dy >= 0.0 && dx < self.extent.0 && dy < self.extent.1
    }

    /// Bilinear logit at a document pixel centre.
    fn at(&self, x: f32, y: f32) -> f32 {
        let (x, y) = (x - self.origin.0, y - self.origin.1);
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
///
/// A part may also have detail logits from a zoomed analysis of a small area (thin straps, chains
/// and jewellery are below the whole-image analysis's resolution). For the small item the area
/// was made for they replace its whole-image logits there ([`Detail::Replace`]); for the other
/// parts in the area they only add sharpness ([`Detail::Refine`]: the higher of the two), because
/// a zoomed prompt may hold just a few of such a part's points and see only a piece of it.
pub struct PartMap {
    parts: Vec<Vec<(Logits, Detail)>>,
}

/// How zoomed detail logits combine with a part's whole-image logits inside their area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detail {
    /// Use the detail logits alone (the small item the zoomed area was made for).
    Replace,
    /// Use the higher of the detail and whole-image logits (other parts in the area).
    Refine,
}

impl PartMap {
    pub fn new(parts: Vec<Logits>) -> Result<Self> {
        if parts.is_empty() {
            return Err(MlError::Input("no parts".into()));
        }
        if parts.len() > MAX_PARTS {
            return Err(MlError::Input(format!("at most {MAX_PARTS} parts")));
        }
        Ok(PartMap { parts: parts.into_iter().map(|l| vec![(l, Detail::Replace)]).collect() })
    }

    /// Add zoomed detail logits for part `part`.
    pub fn add_detail(&mut self, part: usize, logits: Logits, how: Detail) -> Result<()> {
        let list = self.parts.get_mut(part).ok_or_else(|| MlError::Input(format!("no part {part}")))?;
        if list.len() > MAX_PARTS {
            return Err(MlError::Input("too many detail areas".into()));
        }
        list.push((logits, how));
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.parts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// The part owning document pixel (`x`, `y`) and its coverage (`0..=1`, soft at edges).
    ///
    /// Where a zoomed small item ([`Detail::Replace`]) claims the pixel, only zoomed results
    /// compete, at the same sharpness (a chain against the skin under it). Elsewhere every part
    /// uses the higher of its whole-image and zoomed ([`Detail::Refine`]) logits.
    pub fn at(&self, x: f32, y: f32) -> Option<(usize, f32)> {
        // Per part: whole-image, zoomed-item and zoomed-refine logits at this pixel.
        let mut vals: Vec<(f32, Option<f32>, Option<f32>)> = Vec::with_capacity(self.parts.len());
        for layers in &self.parts {
            let Some((base, _)) = layers.first() else {
                vals.push((f32::NEG_INFINITY, None, None));
                continue;
            };
            let (mut item, mut refine) = (None::<f32>, None::<f32>);
            for (l, how) in layers.iter().skip(1).filter(|(l, _)| l.covers(x, y)) {
                let d = l.at(x, y);
                match how {
                    Detail::Replace => item = Some(item.map_or(d, |v| v.max(d))),
                    Detail::Refine => refine = Some(refine.map_or(d, |v| v.max(d))),
                }
            }
            vals.push((base.at(x, y), item, refine));
        }
        let claimed = vals.iter().any(|(_, item, _)| item.is_some_and(|v| v > 0.0));
        let mut best: Option<(usize, f32)> = None;
        for (i, &(whole, item, refine)) in vals.iter().enumerate() {
            let v = if claimed {
                match item.or(refine) {
                    Some(v) => v,
                    None => continue,
                }
            } else {
                match item {
                    Some(v) => v,
                    None => refine.map_or(whole, |r| whole.max(r)),
                }
            };
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
        Logits {
            width: w,
            height: h,
            scale: (1.0, 1.0),
            origin: (0.0, 0.0),
            extent: (w as f32, h as f32),
            data: (0..w * h).map(|i| f(i % w, i / w)).collect(),
            score: 1.0,
        }
    }

    #[test]
    fn frame_keeps_aspect_with_long_side_1024() {
        let px = vec![[0.5; 3]; 30 * 40];
        let img = Rgb { width: 30, height: 40, px: &px, source_size: (3584, 4800), origin: (0, 0) };
        assert_eq!(frame_size(&img).unwrap(), (765, 1024));
    }

    #[test]
    fn mismatched_or_empty_images_are_errors_not_panics() {
        let px = vec![[0.5; 3]; 10];
        assert!(frame_size(&Rgb { width: 4, height: 4, px: &px, source_size: (4, 4), origin: (0, 0) }).is_err());
        assert!(frame_size(&Rgb { width: 0, height: 4, px: &[], source_size: (0, 4), origin: (0, 0) }).is_err());
        assert!(frame_size(&Rgb { width: 1, height: 1, px: &px[..1], source_size: (usize::MAX, 1), origin: (0, 0) }).is_err());
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
    fn detail_logits_replace_the_whole_image_inside_their_area_only() {
        // Whole image: part 0 wins everywhere. Detail for part 1 over x 4..8: it wins there.
        let mut map = PartMap::new(vec![logits(10, 1, |_, _| 5.0), logits(10, 1, |_, _| -5.0)]).unwrap();
        let mut d = logits(4, 1, |_, _| 9.0);
        d.origin = (4.0, 0.0);
        map.add_detail(1, d.clone(), Detail::Replace).unwrap();
        assert_eq!(map.at(2.0, 0.0).map(|p| p.0), Some(0));
        assert_eq!(map.at(5.0, 0.0).map(|p| p.0), Some(1));
        assert_eq!(map.at(9.0, 0.0).map(|p| p.0), Some(0));
        assert!(map.add_detail(7, logits(1, 1, |_, _| 0.0), Detail::Refine).is_err());
        // Refine never loses what the whole image found: a zoomed prompt that sees only part of
        // part 0 (negative logits) leaves part 0 owning the area.
        let mut map = PartMap::new(vec![logits(10, 1, |_, _| 5.0)]).unwrap();
        let mut partial = logits(4, 1, |_, _| -9.0);
        partial.origin = (4.0, 0.0);
        map.add_detail(0, partial.clone(), Detail::Refine).unwrap();
        assert_eq!(map.at(5.0, 0.0).map(|p| p.0), Some(0));
        let mut map = PartMap::new(vec![logits(10, 1, |_, _| 5.0)]).unwrap();
        map.add_detail(0, partial, Detail::Replace).unwrap();
        assert_eq!(map.at(5.0, 0.0), None);
    }

    #[test]
    fn a_zoomed_item_wins_over_the_part_under_it_only_where_it_claims_pixels() {
        // Skin is strongly inside everywhere at whole-image scale; a zoomed chain claims x 4..6,
        // and the zoomed skin excludes the chain there.
        let mut map = PartMap::new(vec![logits(10, 1, |_, _| 9.0), logits(10, 1, |_, _| -9.0)]).unwrap();
        let mut skin = logits(6, 1, |x, _| if (2..4).contains(&x) { -6.0 } else { 6.0 });
        skin.origin = (2.0, 0.0);
        let mut chain = logits(6, 1, |x, _| if (2..4).contains(&x) { 4.0 } else { -6.0 });
        chain.origin = (2.0, 0.0);
        map.add_detail(0, skin, Detail::Refine).unwrap();
        map.add_detail(1, chain, Detail::Replace).unwrap();
        assert_eq!(map.at(4.0, 0.0).map(|p| p.0), Some(1), "the chain");
        assert_eq!(map.at(6.0, 0.0).map(|p| p.0), Some(0), "skin beside it");
        assert_eq!(map.at(0.0, 0.0).map(|p| p.0), Some(0), "skin outside the zoomed area");
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
        let mut emb = model.embed(&Rgb { width: w, height: h, px: &px, source_size: (w, h), origin: (0, 0) }).unwrap();
        let l = emb.logits(&Prompt { points: vec![(128.0, 128.0, PointLabel::Include)], rect: None }).unwrap();
        let map = PartMap::new(vec![l]).unwrap();
        assert!(map.at(128.0, 128.0).is_some(), "centre of the square is inside");
        assert!(map.at(10.0, 10.0).is_none(), "corner outside the square is outside");
    }
}
