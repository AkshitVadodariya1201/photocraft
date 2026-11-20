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

    /// Which part owns each pixel of a `w × h` area at (`x0`, `y0`), and how much (one owner
    /// map and one 8-bit coverage map, not a float mask per part: a 17 MP photo with 18 prompts
    /// needs ~50 MB this way instead of ~1.2 GB). Rows run in parallel on native targets.
    pub fn assign(&self, x0: i64, y0: i64, w: usize, h: usize) -> Assignment {
        let row = |y: usize| -> Vec<(u16, u8)> {
            (0..w)
                .map(|x| match self.at((x0 + x as i64) as f32, (y0 + y as i64) as f32) {
                    Some((i, c)) => (u16::try_from(i).unwrap_or(NO_PART), (c * 255.0).round().clamp(0.0, 255.0) as u8),
                    None => (NO_PART, 0),
                })
                .collect()
        };
        #[cfg(not(target_arch = "wasm32"))]
        let rows: Vec<Vec<(u16, u8)>> = {
            use rayon::prelude::*;
            (0..h).into_par_iter().map(row).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let rows: Vec<Vec<(u16, u8)>> = (0..h).map(row).collect();
        let mut owner = Vec::with_capacity(w * h);
        let mut coverage = Vec::with_capacity(w * h);
        for r in rows {
            for (o, c) in r {
                owner.push(o);
                coverage.push(c);
            }
        }
        Assignment { width: w, height: h, owner, coverage }
    }
}

/// Owner value for a pixel no part covers.
pub const NO_PART: u16 = u16::MAX;

/// The result of [`PartMap::assign`]: per pixel, the owning part (or [`NO_PART`]) and its
/// coverage `0..=255` (soft at edges).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub width: usize,
    pub height: usize,
    pub owner: Vec<u16>,
    pub coverage: Vec<u8>,
}

impl Assignment {
    /// Coverage of part `part` (or of several parts that make one layer) as an 8-bit mask.
    pub fn mask_of(&self, parts: &[usize]) -> Vec<u8> {
        self.owner.iter().zip(&self.coverage).map(|(o, c)| if parts.iter().any(|p| *p == usize::from(*o)) { *c } else { 0 }).collect()
    }

    /// The union of all parts as a hard 8-bit mask (255 where a part covers at least half), the
    /// input [`Self::apply_outline`]'s refinement expects: refining an already blurred mask
    /// keeps its blur, while a hard one gets its soft edge from the photo itself.
    pub fn outline(&self) -> Vec<u8> {
        self.owner.iter().zip(&self.coverage).map(|(o, c)| if *o != NO_PART && *c >= 128 { 255 } else { 0 }).collect()
    }

    /// Move small stray pieces to the part around them: a connected piece of a part (4-connected,
    /// at least half covered) that is at most `max_px` pixels and under 2 % of that part's
    /// largest piece goes to the part owning at least 80 % of its rim (uncovered pixels count
    /// in the rim too, so a piece at the figure's edge, such as toes between sandal straps, stays).
    /// Returns the pixels moved.
    pub fn reassign_islands(&mut self, max_px: usize) -> usize {
        let (w, h) = (self.width, self.height);
        let n = w.saturating_mul(h);
        if max_px == 0 || n == 0 || self.owner.len() != n || self.coverage.len() != n {
            return 0;
        }
        let solid = |o: u16, c: u8| o != NO_PART && c >= 128;
        // Pass 1: label pieces and measure them.
        let mut label = vec![u32::MAX; n];
        let mut pieces: Vec<(u16, usize, usize)> = Vec::new(); // owner, size, a pixel in it
        let mut stack = Vec::new();
        for start in 0..n {
            if label[start] != u32::MAX || !solid(self.owner[start], self.coverage[start]) {
                continue;
            }
            let Ok(id) = u32::try_from(pieces.len()) else { return 0 };
            let o = self.owner[start];
            label[start] = id;
            stack.push(start);
            let mut size = 0usize;
            while let Some(i) = stack.pop() {
                size += 1;
                let (x, y) = (i % w, i / w);
                for j in [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)].into_iter().flatten() {
                    if label[j] == u32::MAX && self.owner[j] == o && self.coverage[j] >= 128 {
                        label[j] = id;
                        stack.push(j);
                    }
                }
            }
            pieces.push((o, size, start));
        }
        let mut largest = [0usize; 65];
        for &(o, size, _) in &pieces {
            if let Some(l) = largest.get_mut(usize::from(o)) {
                *l = (*l).max(size);
            }
        }
        // Pass 2: small pieces whose rim is mostly one other part move to it.
        let mut moved = 0usize;
        let mut members = Vec::new();
        for (id, &(o, size, seed)) in pieces.iter().enumerate() {
            let big = largest.get(usize::from(o)).copied().unwrap_or(0);
            if size > max_px || size.saturating_mul(50) >= big {
                continue;
            }
            let Ok(id) = u32::try_from(id) else { break };
            let mut rim = [0u32; 66]; // 0..64 parts, 65 = uncovered / image edge
            members.clear();
            stack.push(seed);
            let mut seen = std::collections::HashSet::new();
            seen.insert(seed);
            while let Some(i) = stack.pop() {
                members.push(i);
                let (x, y) = (i % w, i / w);
                let around = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
                if around.iter().any(Option::is_none) {
                    rim[65] += 1;
                }
                for j in around.into_iter().flatten() {
                    if label[j] == id {
                        if seen.insert(j) {
                            stack.push(j);
                        }
                    } else if solid(self.owner[j], self.coverage[j]) {
                        if let Some(r) = rim.get_mut(usize::from(self.owner[j]).min(64)) {
                            *r += 1;
                        }
                    } else {
                        rim[65] += 1;
                    }
                }
            }
            let total: u32 = rim.iter().sum();
            let covered: u32 = rim.iter().take(65).sum();
            let Some((best, count)) = rim.iter().enumerate().take(65).filter(|(p, _)| *p != usize::from(o)).max_by_key(|(_, c)| **c) else { continue };
            // Tiny specks (a 16th of the limit) may sit where several parts meet: they follow the
            // neighbour touching them most unless open background is half their rim. Larger pieces
            // need one part on 80 % of the whole rim (toes between sandal straps stay skin), or to
            // be enclosed by other parts (at most 5 % of the rim open) and then go to the one they
            // touch most (shadow under fingertips where the fabric's edge meets the kurta).
            let tiny = size.saturating_mul(16) <= max_px;
            let enclosed = *count > 0 && (total - covered) * 20 <= total;
            let moves = if tiny { *count > 0 && covered * 2 >= total } else { enclosed || (total > 0 && (*count as f64) >= 0.8 * total as f64) };
            if !moves {
                continue;
            }
            let Ok(best) = u16::try_from(best) else { continue };
            for &i in &members {
                self.owner[i] = best;
            }
            moved += members.len();
        }
        moved
    }

    /// Take the figure's outline from a refined `alpha` (same size): every pixel's coverage
    /// becomes `alpha`; pixels the refined outline drops lose their owner, and pixels it adds go
    /// to the nearest part, at most `reach` steps away (4-connected, breadth first from every
    /// owned pixel), so an edge sharpened against the photo keeps each part's own colours.
    /// Returns the pixels added.
    pub fn apply_outline(&mut self, alpha: &[u8], reach: usize) -> usize {
        let (w, h) = (self.width, self.height);
        let n = w.saturating_mul(h);
        if n == 0 || alpha.len() != n || self.owner.len() != n || self.coverage.len() != n {
            return 0;
        }
        // Breadth-first from all owned pixels the outline keeps, through unowned outline pixels.
        let mut dist = vec![u16::MAX; n];
        let mut queue = std::collections::VecDeque::new();
        for i in 0..n {
            if alpha[i] == 0 {
                self.owner[i] = NO_PART;
                self.coverage[i] = 0;
            } else if self.owner[i] != NO_PART {
                dist[i] = 0;
                queue.push_back(i);
            }
        }
        let reach = u16::try_from(reach.min(u16::MAX as usize - 1)).unwrap_or(0);
        let mut added = 0usize;
        while let Some(i) = queue.pop_front() {
            let d = dist[i];
            if d >= reach {
                continue;
            }
            let (x, y) = (i % w, i / w);
            let neighbours = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
            for j in neighbours.into_iter().flatten() {
                if alpha[j] > 0 && self.owner[j] == NO_PART && dist[j] == u16::MAX {
                    dist[j] = d + 1;
                    self.owner[j] = self.owner[i];
                    added += 1;
                    queue.push_back(j);
                }
            }
        }
        for i in 0..n {
            self.coverage[i] = if self.owner[i] == NO_PART { 0 } else { alpha[i] };
        }
        added
    }

    /// Snap the borders between layers to the photo: every covered pixel within `radius` of a
    /// border between two different layers (`layer_of[part]`; pieces of one layer have no border
    /// between them) is given again to the layer whose nearby pixels look most like it (a
    /// colour-weighted vote of the pixels around it, trusting those far from the border more,
    /// twice over). The whole-image analysis
    /// draws such borders on a coarse grid, so a dupatta's edge on a kurta comes out
    /// stair-stepped and frayed; the photo's own colour edge is where it really is. Thin items
    /// of a distinct colour (a chain on skin) keep their pixels, as those vote for each other.
    /// The outline against the background is left alone, and so are parts marked in `locked`
    /// (small items that already had their own zoomed analysis): their pixels neither move nor
    /// vote. `rgb` holds the photo's pixels for this area. Returns the pixels that changed layer.
    pub fn refine_borders(&mut self, rgb: &[[u8; 3]], layer_of: &[usize], locked: &[bool], radius: usize) -> usize {
        let (w, h) = (self.width, self.height);
        let n = w.saturating_mul(h);
        if radius == 0 || n == 0 || rgb.len() != n || self.owner.len() != n || self.coverage.len() != n {
            return 0;
        }
        let radius = radius.min(64);
        let coverage = &self.coverage;
        let layer = |o: u16| -> Option<usize> {
            if o == NO_PART || locked.get(usize::from(o)).copied().unwrap_or(false) {
                None
            } else {
                Some(layer_of.get(usize::from(o)).copied().unwrap_or(usize::from(o) + 1_000_000))
            }
        };
        let solid_layer = |owner: &[u16], i: usize| -> Option<usize> { if coverage[i] >= 128 { layer(owner[i]) } else { None } };
        // The band: distance (8-connected steps) from a pixel on a border between two layers.
        let mut dist = vec![u8::MAX; n];
        let mut queue = std::collections::VecDeque::new();
        for i in 0..n {
            let Some(l) = solid_layer(&self.owner, i) else { continue };
            let (x, y) = (i % w, i / w);
            let next = [(x + 1 < w).then(|| i + 1), (y + 1 < h).then(|| i + w)];
            for j in next.into_iter().flatten() {
                if solid_layer(&self.owner, j).is_some_and(|m| m != l) {
                    for k in [i, j] {
                        if dist[k] != 0 {
                            dist[k] = 0;
                            queue.push_back(k);
                        }
                    }
                }
            }
        }
        while let Some(i) = queue.pop_front() {
            let d = dist[i];
            if usize::from(d) >= radius {
                continue;
            }
            let (x, y) = (i % w, i / w);
            for (dx, dy) in [(-1i64, -1i64), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if dist[j] == u8::MAX {
                    dist[j] = d + 1;
                    queue.push_back(j);
                }
            }
        }
        let band: Vec<usize> = (0..n).filter(|&i| dist[i] != u8::MAX && solid_layer(&self.owner, i).is_some()).collect();
        // Pixels far from the border are sure of their layer; those near it vote less, so a
        // sliver on the wrong side of the drawn border cannot outvote the photo.
        let trust: Vec<f32> = dist.iter().map(|d| if *d == u8::MAX { 1.0 } else { (f32::from(*d) + 1.0) / (radius as f32 + 1.0) }).collect();
        drop(dist);
        if band.is_empty() {
            return 0;
        }
        // Sample a window twice the band's width on a grid (about 13 x 13 samples), weighted by
        // distance, colour and trust.
        let r = 2 * radius as i64;
        let stride = (r / 6).max(1) as usize;
        let sigma_s = (radius as f32).max(1.0);
        let offsets: Vec<(i64, i64, f32)> = (-r..=r)
            .step_by(stride)
            .flat_map(|dy| (-r..=r).step_by(stride).map(move |dx| (dx, dy)))
            .filter(|&(dx, dy)| dx * dx + dy * dy <= r * r && (dx, dy) != (0, 0))
            .map(|(dx, dy)| (dx, dy, (-((dx * dx + dy * dy) as f32) / (2.0 * sigma_s * sigma_s)).exp()))
            .collect();
        const SIGMA_C: f32 = 16.0;
        let colour_w: Vec<f32> = (0..=3 * 255 * 255).map(|d2| (-(d2 as f32) / (2.0 * SIGMA_C * SIGMA_C)).exp()).collect();
        let mut changed = 0usize;
        // Whether the colours clearly chose a pixel's layer (then smoothing leaves it alone).
        let mut sure: Vec<bool> = Vec::new();
        for _ in 0..2 {
            let owner = &self.owner;
            let vote = |&i: &usize| -> (u16, bool) {
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                let c = rgb[i];
                // Per layer: total weight, and the part of its strongest sample (keeps pieces local).
                let mut tally: Vec<(usize, f32, u16, f32)> = Vec::with_capacity(4);
                for &(dx, dy, ws) in &offsets {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    let Some(l) = solid_layer(owner, j) else { continue };
                    let q = rgb[j];
                    let d2: usize = (0..3)
                        .map(|k| {
                            let d = i32::from(c[k]) - i32::from(q[k]);
                            (d * d) as usize
                        })
                        .sum();
                    let wt = ws * trust[j] * colour_w.get(d2).copied().unwrap_or(0.0);
                    match tally.iter_mut().find(|t| t.0 == l) {
                        Some(t) => {
                            t.1 += wt;
                            if wt > t.3 {
                                t.2 = owner[j];
                                t.3 = wt;
                            }
                        }
                        None => tally.push((l, wt, owner[j], wt)),
                    }
                }
                let mine = solid_layer(owner, i);
                let Some(best) = tally.iter().max_by(|a, b| a.1.total_cmp(&b.1)) else { return (owner[i], true) };
                let total: f32 = tally.iter().map(|t| t.1).sum();
                let sure = best.1 >= 0.75 * total;
                if Some(best.0) == mine || best.1 <= 0.0 { (owner[i], sure) } else { (best.2, sure) }
            };
            #[cfg(not(target_arch = "wasm32"))]
            let next: Vec<(u16, bool)> = {
                use rayon::prelude::*;
                band.par_iter().map(vote).collect()
            };
            #[cfg(target_arch = "wasm32")]
            let next: Vec<(u16, bool)> = band.iter().map(vote).collect();
            sure = next.iter().map(|v| v.1).collect();
            let mut moved = 0usize;
            for (&i, (o, _)) in band.iter().zip(next) {
                if self.owner[i] != o {
                    if layer(self.owner[i]) != layer(o) {
                        moved += 1;
                    }
                    self.owner[i] = o;
                }
            }
            changed += moved;
            if moved == 0 {
                break;
            }
        }
        // Straighten the border: twice, each band pixel whose colours did not clearly choose takes
        // the layer most of its neighbours have (a window of about a third of the band) when they
        // are a clear majority, so teeth left by a printed motif in the border's own colour are
        // cut off while a thin strand of its own colour stays.
        let s = (radius / 3).max(1) as i64;
        for _ in 0..2 {
            let owner = &self.owner;
            let smooth = |(&i, &sure): (&usize, &bool)| -> u16 {
                if sure {
                    return owner[i];
                }
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                let mut tally: Vec<(usize, u32, u16)> = Vec::with_capacity(4);
                let mut total = 0u32;
                for dy in -s..=s {
                    for dx in -s..=s {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let j = ny as usize * w + nx as usize;
                        let Some(l) = solid_layer(owner, j) else { continue };
                        total += 1;
                        match tally.iter_mut().find(|t| t.0 == l) {
                            Some(t) => t.1 += 1,
                            None => tally.push((l, 1, owner[j])),
                        }
                    }
                }
                let mine = solid_layer(owner, i);
                match tally.iter().max_by_key(|t| t.1) {
                    Some(best) if Some(best.0) != mine && best.1 * 3 > total * 2 => best.2,
                    _ => owner[i],
                }
            };
            #[cfg(not(target_arch = "wasm32"))]
            let next: Vec<u16> = {
                use rayon::prelude::*;
                band.par_iter().zip(sure.par_iter()).map(smooth).collect()
            };
            #[cfg(target_arch = "wasm32")]
            let next: Vec<u16> = band.iter().zip(sure.iter()).map(smooth).collect();
            for (&i, o) in band.iter().zip(next) {
                if self.owner[i] != o {
                    if layer(self.owner[i]) != layer(o) {
                        changed += 1;
                    }
                    self.owner[i] = o;
                }
            }
        }
        changed
    }

    /// Fill small holes from the parts around them: a connected area of pixels less than half
    /// covered, not touching the image border and at most `max_px` pixels, is shared out by
    /// nearest part (breadth first from its rim), so a sliver between fingertips and fabric
    /// splits down the middle instead of going all to one side. Background showing through the figure (between an arm and the
    /// body) is usually larger than `max_px` and stays empty. Returns the pixels filled.
    pub fn fill_enclosed_gaps(&mut self, max_px: usize) -> usize {
        let (w, h) = (self.width, self.height);
        let n = w.saturating_mul(h);
        if max_px == 0 || n == 0 || self.owner.len() != n || self.coverage.len() != n {
            return 0;
        }
        let gap = |c: u8| c < 128;
        let mut seen = vec![false; n];
        let mut filled = 0usize;
        let mut stack: Vec<usize> = Vec::new();
        let mut comp: Vec<usize> = Vec::new();
        for start in 0..n {
            if seen[start] || !gap(self.coverage[start]) {
                continue;
            }
            // Flood the gap area (4-connected), with an explicit stack: no recursion.
            comp.clear();
            stack.clear();
            stack.push(start);
            seen[start] = true;
            let mut touches_border = false;
            let mut rim = [0u32; 64 + 1];
            while let Some(i) = stack.pop() {
                comp.push(i);
                let (x, y) = (i % w, i / w);
                if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                    touches_border = true;
                }
                let neighbours = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
                for j in neighbours.into_iter().flatten() {
                    if gap(self.coverage[j]) {
                        if !seen[j] {
                            seen[j] = true;
                            stack.push(j);
                        }
                    } else if let Some(slot) = rim.get_mut(usize::from(self.owner[j])) {
                        *slot += 1;
                    }
                }
            }
            if touches_border || comp.len() > max_px || rim.iter().all(|c| *c == 0) {
                continue;
            }
            // Breadth first from the rim's owned pixels into this hole only.
            let mut queue = std::collections::VecDeque::new();
            let in_hole: std::collections::HashSet<usize> = comp.iter().copied().collect();
            for &i in &comp {
                let (x, y) = (i % w, i / w);
                for j in [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)].into_iter().flatten() {
                    if !in_hole.contains(&j) && self.owner[j] != NO_PART {
                        queue.push_back(j);
                    }
                }
            }
            let mut done: std::collections::HashSet<usize> = std::collections::HashSet::new();
            while let Some(i) = queue.pop_front() {
                let (x, y) = (i % w, i / w);
                for j in [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)].into_iter().flatten() {
                    if in_hole.contains(&j) && done.insert(j) {
                        self.owner[j] = self.owner[i];
                        self.coverage[j] = 255;
                        queue.push_back(j);
                    }
                }
            }
            filled += comp.len();
        }
        filled
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
        let a = map.assign(0, 0, 10, 1);
        assert_eq!((a.owner[1], a.owner[8]), (0, 1));
        assert!(a.coverage[1] > 250 && a.coverage[8] > 250);
        assert_eq!(a.mask_of(&[0])[8], 0);
        assert_eq!(a.mask_of(&[0, 1])[8], a.coverage[8]);
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
        assert!(map.assign(0, 0, 4, 4).owner.iter().all(|o| *o == NO_PART));
    }

    fn grid(rows: &[&str]) -> Assignment {
        // '.' = gap, digits = owning part (full coverage).
        let (w, h) = (rows[0].len(), rows.len());
        let mut owner = Vec::new();
        let mut coverage = Vec::new();
        for r in rows {
            for ch in r.chars() {
                match ch.to_digit(10) {
                    Some(d) => {
                        owner.push(d as u16);
                        coverage.push(255);
                    }
                    None => {
                        owner.push(NO_PART);
                        coverage.push(0);
                    }
                }
            }
        }
        Assignment { width: w, height: h, owner, coverage }
    }

    #[test]
    fn small_enclosed_gaps_go_to_the_part_around_them() {
        let mut a = grid(&["........", ".000111.", ".0..111.", ".000111.", "........"]);
        // The 2-px hole is enclosed; the outside touches the border. Each hole pixel takes a
        // part next to it (the right one is as near to part 1 as to part 0).
        assert_eq!(a.fill_enclosed_gaps(10), 2);
        assert_eq!(a.owner[2 * 8 + 2], 0);
        assert!(matches!(a.owner[2 * 8 + 3], 0 | 1));
        assert_eq!(a.owner[0], NO_PART, "outside stays empty");
    }

    #[test]
    fn a_refined_outline_extends_parts_to_their_nearest_neighbour_and_trims_the_rest() {
        let mut a = grid(&["......", ".0011.", ".0011.", "......"]);
        let w = 6;
        // Refined outline: one pixel wider on the left and right, top row dropped.
        let mut alpha = vec![0u8; 24];
        for y in 1..3 {
            for x in 0..6 {
                alpha[y * w + x] = 200;
            }
        }
        alpha[w + 3] = 0; // trimmed
        assert_eq!(a.apply_outline(&alpha, 2), 4, "four new edge pixels");
        assert_eq!((a.owner[w], a.owner[w + 5]), (0, 1), "each side goes to its nearest part");
        assert_eq!(a.owner[w + 3], NO_PART, "dropped by the outline");
        assert_eq!(a.coverage[w + 1], 200, "coverage follows the outline");
        assert_eq!(a.outline()[w + 3], 0);
        let mut far = grid(&["0...."]);
        assert_eq!(far.apply_outline(&[255; 5], 2), 2, "reach limits how far a part extends");
        assert_eq!(far.owner[4], NO_PART);
        assert_eq!(far.apply_outline(&[1, 2], 2), 0, "mismatched sizes are ignored");
    }

    #[test]
    fn stray_pieces_join_the_part_around_them_but_edge_pieces_stay() {
        // A 2-px piece of part 0 fully inside part 1 moves to part 1.
        let mut a = grid(
            &["000000000000000000000000"; 6]
                .iter()
                .copied()
                .chain(["000000000000111111111111", "000000000000110011111111", "000000000000111111111111"])
                .collect::<Vec<_>>(),
        );
        assert_eq!(a.reassign_islands(10), 2);
        assert_eq!((a.owner[24 * 7 + 14], a.owner[24 * 7 + 15]), (1, 1));
        // A piece with open background on a quarter of its rim (toes between straps) stays.
        let mut b = grid(&["0000000000000000"; 8].iter().copied().chain(["0000000000111...", "0000000000110...", "0000000000111..."]).collect::<Vec<_>>());
        assert_eq!(b.reassign_islands(10), 0);
        assert_eq!(b.owner[16 * 9 + 12], 0);
        assert_eq!(a.reassign_islands(0), 0);
        let mut bad = Assignment { width: 2, height: 2, owner: vec![0], coverage: vec![255] };
        assert_eq!(bad.reassign_islands(10), 0);
    }

    #[test]
    fn a_hole_between_two_parts_is_split_by_nearest_part() {
        // A 1x4 hole between part 0 (left) and part 1 (right) splits 2 / 2.
        let mut a = grid(&["000000011111", "00000....111", "000000011111"]);
        assert_eq!(a.fill_enclosed_gaps(10), 4);
        let row = 12;
        assert_eq!(&a.owner[row + 5..row + 9], &[0, 0, 1, 1]);
    }

    #[test]
    fn a_tiny_speck_where_two_parts_meet_follows_the_larger_neighbour() {
        // A 1-px piece of part 2 touching part 0 on three sides and part 1 on one.
        let mut rows = vec!["2222222222222222"; 6];
        rows.extend(["0000000011111111", "0000000201111111", "0000000011111111"]);
        let mut a = grid(&rows);
        assert_eq!(a.reassign_islands(16), 1);
        assert_eq!(a.owner[16 * 7 + 7], 0);
    }

    #[test]
    fn borders_snap_to_the_colour_edge_and_thin_items_stay() {
        // Two layers whose drawn border (x = 10) is 3 px left of the colour edge (x = 13).
        let (w, h) = (24usize, 12usize);
        let mut a = Assignment { width: w, height: h, owner: vec![0; w * h], coverage: vec![255; w * h] };
        let mut rgb = vec![[200u8, 180, 160]; w * h];
        for y in 0..h {
            for x in 0..w {
                if x >= 10 {
                    a.owner[y * w + x] = 1;
                }
                if x >= 13 {
                    rgb[y * w + x] = [110, 30, 40];
                }
            }
        }
        // A 1-px dark chain (part 2) running down inside part 0, on its own colour.
        for y in 0..h {
            a.owner[y * w + 4] = 2;
            rgb[y * w + 4] = [60, 60, 60];
        }
        let changed = a.refine_borders(&rgb, &[0, 1, 2], &[], 4);
        assert_eq!(changed, 3 * h);
        for y in 0..h {
            assert_eq!(&a.owner[y * w + 9..y * w + 14], &[0, 0, 0, 0, 1], "row {y}");
            assert_eq!(a.owner[y * w + 4], 2, "chain kept");
        }
        // Pieces of one layer have no border between them; bad sizes and radius 0 do nothing.
        let mut same = Assignment { width: w, height: h, owner: (0..w * h).map(|i| u16::from(i % w >= 10)).collect(), coverage: vec![255; w * h] };
        assert_eq!(same.refine_borders(&rgb, &[0, 0], &[], 4), 0);
        assert_eq!(same.refine_borders(&rgb, &[0, 1], &[], 0), 0);
        assert_eq!(same.refine_borders(&rgb[1..], &[0, 1], &[], 4), 0);
        // A tooth of layer 1 sticking 2 px into layer 0 on the same colour as layer 0 is cut off.
        let mut tooth = Assignment { width: w, height: h, owner: (0..w * h).map(|i| u16::from(i % w >= 13)).collect(), coverage: vec![255; w * h] };
        tooth.owner[6 * w + 11] = 1;
        tooth.owner[6 * w + 12] = 1;
        tooth.refine_borders(&rgb, &[0, 1, 2], &[], 4);
        assert_eq!((tooth.owner[6 * w + 11], tooth.owner[6 * w + 12]), (0, 0));
        // A locked part keeps its pixels even where the colours say otherwise.
        let mut kept = Assignment { width: w, height: h, owner: (0..w * h).map(|i| u16::from(i % w >= 10)).collect(), coverage: vec![255; w * h] };
        assert_eq!(kept.refine_borders(&rgb, &[0, 1], &[false, true], 4), 0);
        assert!((0..h).all(|y| kept.owner[y * w + 10] == 1));
    }

    #[test]
    fn an_enclosed_stray_piece_goes_to_the_part_it_touches_most() {
        // A 3x3 piece of part 2 between part 0 and part 1 with no background around it; part 2's
        // main area is the large block on top.
        let mut rows: Vec<String> = vec!["2".repeat(40); 20];
        rows.push(format!("{}{}", "0".repeat(20), "1".repeat(20)));
        for _ in 0..3 {
            rows.push(format!("{}222{}", "0".repeat(19), "1".repeat(18)));
        }
        rows.push(format!("{}{}", "0".repeat(20), "1".repeat(20)));
        let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
        let mut a = grid(&refs);
        // Tiny pieces follow their own rule: use a limit that makes this piece not tiny.
        assert_eq!(a.reassign_islands(100), 9);
        assert!((22 * 40..25 * 40).all(|i| a.owner[i] != 2));
    }

    #[test]
    fn large_gaps_and_border_gaps_stay_empty() {
        let mut a = grid(&["000000", "0....0", "0....0", "000000"]);
        assert_eq!(a.fill_enclosed_gaps(7), 0, "8-px hole is larger than the limit");
        assert_eq!(a.fill_enclosed_gaps(8), 8);
        let mut b = grid(&["0..0", "0000"]);
        assert_eq!(b.fill_enclosed_gaps(100), 0, "touches the border");
        let mut bad = Assignment { width: 3, height: 3, owner: vec![0], coverage: vec![0] };
        assert_eq!(bad.fill_enclosed_gaps(100), 0, "mismatched buffers are ignored, not indexed");
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
