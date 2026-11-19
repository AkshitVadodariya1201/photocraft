//! Split into Parts: cut a pixel layer into named, grouped layers (hair, face, garments,
//! accessories…) with a promptable segmentation model (MobileSAM / SAM, `photocraft-ml`).
//!
//! The caller (an agent, a script, or later a dialog) names each part and points at it:
//! include/exclude points and/or a box in document pixels. The image is analysed once, each
//! part gets a mask, overlapping parts are resolved so every pixel belongs to at most one part,
//! and each part becomes a copy of the source pixels with that mask, inside its group. The
//! source layer is left as it was; the whole split is one undo step.
//!
//! Models are read from `modelDir` or `PHOTOCRAFT_SAM_DIR`: `mobile_sam_image_encoder.onnx`
//! and `sam_mask_decoder_single.onnx`. Desktop only; the web build reports it as unsupported.

use photocraft_doc::{Document, Layer, LayerContent, LayerId, LayerMask};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::{EngineError, Result, Session};

const CMD: &str = "layer.splitIntoParts";
/// File names inside the model folder.
pub const ENCODER_FILE: &str = "mobile_sam_image_encoder.onnx";
pub const DECODER_FILE: &str = "sam_mask_decoder_single.onnx";
/// Longest side of the copy that is analysed (the encoder sees 1024; a little more keeps the
/// downsampling clean).
const ANALYSIS_SIDE: i64 = 2048;
const MAX_NAME: usize = 120;
/// A boxed part this small (long side, as a share of the image's long side) also gets a zoomed
/// analysis: thin straps, chains and jewellery are below the whole-image analysis's resolution.
const DETAIL_SHARE: f32 = 0.35;
/// Most zoomed analyses per split (each costs about as much as the whole-image one).
const MAX_DETAIL_AREAS: usize = 8;
/// Smallest side of a zoomed area, in document pixels.
const MIN_DETAIL_SIDE: f32 = 256.0;
/// Default largest hole filled, as a share of the image's pixels (0.03 %: about 5,200 px on a
/// 17 MP photo). Thin unclaimed slivers between parts are far smaller; background showing through
/// the figure (between an arm and the body) is usually larger and stays empty.
const FILL_GAPS_SHARE: f64 = 0.0003;
/// Most edge-refinement band, in document pixels.
const MAX_REFINE_EDGE: f64 = 64.0;

/// Zoomed areas for the small boxed parts, in document pixels, merged where they overlap: the box
/// with a margin of half its size on every side (at least [`MIN_DETAIL_SIDE`]), inside the canvas.
fn detail_regions(parts: &[PartSpec], area: photocraft_geom::Rect) -> Vec<photocraft_geom::Rect> {
    use photocraft_geom::Rect;
    let long = area.width().max(area.height()) as f32;
    let mut regions: Vec<Rect> = Vec::new();
    for part in parts {
        let Some([x0, y0, x1, y1]) = part.rect else { continue };
        let (bw, bh) = (x1 - x0, y1 - y0);
        if bw.max(bh) > DETAIL_SHARE * long {
            continue;
        }
        let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
        let (hx, hy) = ((bw * 2.0).max(MIN_DETAIL_SIDE) / 2.0, (bh * 2.0).max(MIN_DETAIL_SIDE) / 2.0);
        let r = Rect::new((cx - hx).floor() as i32, (cy - hy).floor() as i32, (cx + hx).ceil() as i32, (cy + hy).ceil() as i32).intersect(&area);
        if r.is_empty() {
            continue;
        }
        // Merge with every overlapping area, repeating so chains of overlaps join.
        let mut merged = r;
        loop {
            let before = regions.len();
            regions.retain(|o| {
                if o.intersect(&merged).is_empty() {
                    true
                } else {
                    merged = merged.union(o);
                    false
                }
            });
            if regions.len() == before {
                break;
            }
        }
        regions.push(merged);
    }
    regions.truncate(MAX_DETAIL_AREAS);
    regions
}

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: CMD.into(), msg: msg.into() }
}

/// One requested part.
#[derive(Debug, Clone, PartialEq)]
struct PartSpec {
    name: String,
    group: Option<String>,
    points: Vec<(f32, f32, bool)>,
    rect: Option<[f32; 4]>,
}

fn num(v: &Value, what: &str) -> Result<f32> {
    let n = v.as_f64().ok_or_else(|| bad(format!("{what} must be a number")))?;
    if !n.is_finite() || n.abs() > 1e9 {
        return Err(bad(format!("{what} is out of range")));
    }
    Ok(n as f32)
}

fn name_of(v: Option<&Value>, what: &str) -> Result<Option<String>> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                return Err(bad(format!("{what} is empty")));
            }
            Ok(Some(t.chars().take(MAX_NAME).collect()))
        }
        Some(_) => Err(bad(format!("{what} must be a string"))),
    }
}

/// Parse and validate `parts` before any work starts.
fn parse_parts(p: &Value) -> Result<Vec<PartSpec>> {
    let arr = p.get("parts").and_then(Value::as_array).ok_or_else(|| bad("parts must be a list of {name, points?, box?, group?}"))?;
    if arr.is_empty() {
        return Err(bad("parts is empty"));
    }
    if arr.len() > photocraft_ml_limits::MAX_PARTS {
        return Err(bad(format!("at most {} parts", photocraft_ml_limits::MAX_PARTS)));
    }
    let mut out = Vec::with_capacity(arr.len());
    for (i, part) in arr.iter().enumerate() {
        let name = name_of(part.get("name"), &format!("parts[{i}].name"))?.ok_or_else(|| bad(format!("parts[{i}] needs a name")))?;
        let group = name_of(part.get("group"), &format!("parts[{i}].group"))?;
        let mut points = Vec::new();
        if let Some(pts) = part.get("points") {
            let pts = pts.as_array().ok_or_else(|| bad(format!("parts[{i}].points must be a list of [x, y, 1|0]")))?;
            if pts.len() > photocraft_ml_limits::MAX_POINTS {
                return Err(bad(format!("parts[{i}] has more than {} points", photocraft_ml_limits::MAX_POINTS)));
            }
            for (j, pt) in pts.iter().enumerate() {
                let a = pt
                    .as_array()
                    .filter(|a| a.len() == 2 || a.len() == 3)
                    .ok_or_else(|| bad(format!("parts[{i}].points[{j}] must be [x, y] or [x, y, 1|0]")))?;
                let (Some(x), Some(y)) = (a.first(), a.get(1)) else { return Err(bad(format!("parts[{i}].points[{j}] is incomplete"))) };
                let include = match a.get(2) {
                    None => true,
                    Some(l) => match l.as_i64() {
                        Some(1) => true,
                        Some(0) => false,
                        _ => return Err(bad(format!("parts[{i}].points[{j}] label must be 1 (include) or 0 (exclude)"))),
                    },
                };
                points.push((num(x, "a point x")?, num(y, "a point y")?, include));
            }
        }
        let rect = match part.get("box") {
            None | Some(Value::Null) => None,
            Some(b) => {
                let a = b.as_array().filter(|a| a.len() == 4).ok_or_else(|| bad(format!("parts[{i}].box must be [x, y, width, height]")))?;
                let v: Vec<f32> = a.iter().map(|n| num(n, "a box value")).collect::<Result<_>>()?;
                let (x, y, w, h) = (v[0], v[1], v[2], v[3]);
                if w <= 0.0 || h <= 0.0 {
                    return Err(bad(format!("parts[{i}].box has no area")));
                }
                Some([x, y, x + w, y + h])
            }
        };
        if !points.iter().any(|p| p.2) && rect.is_none() {
            return Err(bad(format!("parts[{i}] ({name}) needs an include point or a box")));
        }
        out.push(PartSpec { name, group, points, rect });
    }
    Ok(out)
}

/// Limits shared with `photocraft-ml`, kept here so parsing works on every target.
mod photocraft_ml_limits {
    pub const MAX_PARTS: usize = 64;
    pub const MAX_POINTS: usize = 32;
}

fn check(doc: &Document, id: LayerId) -> std::result::Result<(), String> {
    let l = doc.layer(id).ok_or("no such layer")?;
    if !matches!(l.content, LayerContent::Raster(_)) {
        return Err(format!("the layer is {} {} layer, not a pixel layer", l.content.article(), l.content.kind_name()));
    }
    Ok(())
}

fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no layer selected")?;
    check(&d.doc, id)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_model(p: &Value) -> Result<photocraft_ml::SamModel> {
    let dir = match p.get("modelDir").and_then(Value::as_str) {
        Some(d) if !d.trim().is_empty() => std::path::PathBuf::from(d),
        _ => std::env::var_os("PHOTOCRAFT_SAM_DIR").map(std::path::PathBuf::from).ok_or_else(|| {
            EngineError::Other("no segmentation model: pass modelDir or set PHOTOCRAFT_SAM_DIR to a folder with the MobileSAM ONNX files".into())
        })?,
    };
    let read = |f: &str| std::fs::read(dir.join(f)).map_err(|e| EngineError::Other(format!("cannot read {}: {e}", dir.join(f).display())));
    let (enc, dec) = (read(ENCODER_FILE)?, read(DECODER_FILE)?);
    photocraft_ml::SamModel::from_bytes(enc, dec).map_err(|e| EngineError::Other(e.to_string()))
}

#[cfg(target_arch = "wasm32")]
fn run(_s: &mut Session, _p: &Value) -> Result<Value> {
    Err(EngineError::Other("Split into Parts is not available in the web build yet".into()))
}

#[cfg(not(target_arch = "wasm32"))]
fn run(s: &mut Session, p: &Value) -> Result<Value> {
    use photocraft_geom::Rect;
    use photocraft_ml::{Detail, PartMap, PointLabel, Prompt, Rgb};

    use crate::smartselect_cmds::with_doc_sampler;

    let parts = parse_parts(p)?;
    let detail = match p.get("detail") {
        None | Some(Value::Null) => true,
        Some(v) => v.as_bool().ok_or_else(|| bad("detail must be true or false"))?,
    };
    let id = layer_param(s, p)?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    check(doc, id).map_err(EngineError::Other)?;
    let area = doc.bounds();
    if area.is_empty() {
        return Err(EngineError::Other("the canvas is empty".into()));
    }
    let pixels_total = (area.width() as u64).saturating_mul(area.height() as u64);
    // The model's masks are enlarged from a 1024 px analysis, which blurs the outline by a few
    // enlargement steps: refine it against the photo over about 2.5 of them by default.
    let long_side = area.width().max(area.height()) as f64;
    let refine_edge = match p.get("refineEdge") {
        None | Some(Value::Null) => (long_side / photocraft_ml::SAM_SIDE as f64 * 2.5).clamp(2.0, 24.0),
        Some(v) => {
            let n = v.as_f64().filter(|n| n.is_finite() && *n >= 0.0).ok_or_else(|| bad("refineEdge must be a number of pixels, 0 or more"))?;
            n.min(MAX_REFINE_EDGE)
        }
    };
    let fill_gaps = match p.get("fillGaps") {
        None | Some(Value::Null) => (pixels_total as f64 * FILL_GAPS_SHARE) as usize,
        Some(v) => {
            let n = v.as_u64().ok_or_else(|| bad("fillGaps must be a whole number of pixels (0 turns it off)"))?;
            usize::try_from(n.min(pixels_total)).unwrap_or(usize::MAX)
        }
    };
    // Loading reads ~45 MB and parses both networks: do it before the job so a missing model
    // is reported at once.
    let model = load_model(p)?;

    crate::jobs::edit_job(
        s,
        "Split into Parts",
        move |doc, active, ctx| {
            let (w, h) = (area.width() as i64, area.height() as i64);
            let step = ((w.max(h) + ANALYSIS_SIDE - 1) / ANALYSIS_SIDE).max(1) as usize;
            let img = with_doc_sampler(doc, Some(id), false, |smp, _| smp.rgb_scaled(area, step));
            let mut emb = ctx
                .stage(0.0, 0.5, "Analysing image", |_| {
                    model.embed(&Rgb { width: img.w, height: img.h, px: &img.px, source_size: (w as usize, h as usize), origin: (0, 0) })
                })
                .map_err(|e| EngineError::Other(e.to_string()))?;
            if ctx.cancelled() {
                return Err(EngineError::Cancelled);
            }
            let mut logits = Vec::with_capacity(parts.len());
            let n = parts.len() as f32;
            for (i, part) in parts.iter().enumerate() {
                let (a, b) = (0.5 + 0.2 * i as f32 / n, 0.5 + 0.2 * (i + 1) as f32 / n);
                let prompt = Prompt {
                    points: part
                        .points
                        .iter()
                        .map(|&(x, y, inc)| (x - area.x0 as f32, y - area.y0 as f32, if inc { PointLabel::Include } else { PointLabel::Exclude }))
                        .collect(),
                    rect: part.rect.map(|[x0, y0, x1, y1]| [x0 - area.x0 as f32, y0 - area.y0 as f32, x1 - area.x0 as f32, y1 - area.y0 as f32]),
                };
                let l = ctx
                    .stage(a, b, &format!("Finding {}", part.name), |_| emb.logits(&prompt))
                    .map_err(|e| EngineError::Other(format!("{}: {e}", part.name)))?;
                logits.push(l);
                if ctx.cancelled() {
                    return Err(EngineError::Cancelled);
                }
            }
            let mut map = PartMap::new(logits).map_err(|e| EngineError::Other(e.to_string()))?;

            // Zoomed detail for small boxed parts: inside each area, every part with a prompt
            // there is analysed again at the area's own resolution, so a strap competes with the
            // skin under it at the same sharpness.
            let regions = if detail { detail_regions(&parts, area) } else { Vec::new() };
            let nr = regions.len().max(1) as f32;
            for (ri, r) in regions.iter().enumerate() {
                let (a, b) = (0.7 + 0.1 * ri as f32 / nr, 0.7 + 0.1 * (ri + 1) as f32 / nr);
                let (rw, rh) = (r.width() as i64, r.height() as i64);
                let rstep = ((rw.max(rh) + ANALYSIS_SIDE - 1) / ANALYSIS_SIDE).max(1) as usize;
                let crop = with_doc_sampler(doc, Some(id), false, |smp, _| smp.rgb_scaled(*r, rstep));
                let origin = ((r.x0 - area.x0).max(0) as usize, (r.y0 - area.y0).max(0) as usize);
                let mut demb = ctx
                    .stage(a, b, "Analysing details", |_| {
                        model.embed(&Rgb { width: crop.w, height: crop.h, px: &crop.px, source_size: (rw as usize, rh as usize), origin })
                    })
                    .map_err(|e| EngineError::Other(e.to_string()))?;
                let inside = |x: f32, y: f32| x >= r.x0 as f32 && y >= r.y0 as f32 && x < r.x1 as f32 && y < r.y1 as f32;
                for (i, part) in parts.iter().enumerate() {
                    let points: Vec<_> = part
                        .points
                        .iter()
                        .filter(|&&(x, y, _)| inside(x, y))
                        .map(|&(x, y, inc)| (x - area.x0 as f32, y - area.y0 as f32, if inc { PointLabel::Include } else { PointLabel::Exclude }))
                        .collect();
                    let rect = part
                        .rect
                        .filter(|&[x0, y0, x1, y1]| inside(x0, y0) && inside(x1 - 1.0, y1 - 1.0))
                        .map(|[x0, y0, x1, y1]| [x0 - area.x0 as f32, y0 - area.y0 as f32, x1 - area.x0 as f32, y1 - area.y0 as f32]);
                    if rect.is_none() && !points.iter().any(|p| p.2 == PointLabel::Include) {
                        continue;
                    }
                    let l = demb.logits(&Prompt { points, rect }).map_err(|e| EngineError::Other(format!("{}: {e}", part.name)))?;
                    // The small item this area was made for trusts its zoomed result alone; the
                    // other parts here only gain sharpness from it.
                    let how = if rect.is_some() { Detail::Replace } else { Detail::Refine };
                    map.add_detail(i, l, how).map_err(|e| EngineError::Other(e.to_string()))?;
                }
                if ctx.cancelled() {
                    return Err(EngineError::Cancelled);
                }
            }
            let mut assignment = ctx.stage(0.8, 0.9, "Building masks", |_| map.assign(area.x0 as i64, area.y0 as i64, w as usize, h as usize));
            if ctx.cancelled() {
                return Err(EngineError::Cancelled);
            }
            // Sharpen the figure's outline against the photo (the same edge refinement as Select and
            // Mask); pixels it adds go to the nearest part, pixels it drops leave every part.
            let mut edge_added = 0usize;
            if refine_edge > 0.0 {
                let outline = assignment.outline();
                let mut surface =
                    photocraft_raster::Surface::from_interleaved(photocraft_color::PixelFormat::GRAY8, Rect::new(area.x0, area.y0, area.x1, area.y1), &outline);
                surface.prune();
                let params = photocraft_algo::matting::RefineParams {
                    radius: refine_edge as f32,
                    smart_radius: true,
                    smooth: 10.0,
                    feather: 0.5,
                    contrast: 10.0,
                    shift_edge: 0.0,
                };
                let refined = ctx
                    .stage(0.85, 0.9, "Refining edges", |interrupt| {
                        with_doc_sampler(doc, Some(id), false, |smp, _| {
                            photocraft_algo::matting::refine_surface_with_interrupt(smp, &surface, area, &params, None, interrupt)
                        })
                    })
                    .map_err(|e| EngineError::Other(e.to_string()))?;
                if ctx.cancelled() {
                    return Err(EngineError::Cancelled);
                }
                if let Some(refined) = refined {
                    let alpha = refined.to_interleaved(Rect::new(area.x0, area.y0, area.x1, area.y1));
                    if alpha.len() == outline.len() {
                        edge_added = assignment.apply_outline(&alpha, (refine_edge * 2.0).ceil() as usize);
                    }
                }
            }
            // Stray pieces of a part inside another (a speck of kurta in the dupatta) join the
            // part around them; then small holes are filled. Both use the fillGaps size limit.
            let islands_moved = ctx.stage(0.9, 0.93, "Cleaning stray pieces", |_| assignment.reassign_islands(fill_gaps));
            let filled = ctx.stage(0.93, 0.95, "Filling gaps", |_| assignment.fill_enclosed_gaps(fill_gaps));

            let source = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
            let LayerContent::Raster(pixels) = &source.content else { return Err(EngineError::Other("the layer is no longer a pixel layer".into())) };
            let pixels = pixels.clone();

            // Parts that share a name (and group) are pieces of one layer: a drape and its border,
            // a kurta and its hem. The layer takes the first piece's place.
            let mut pieces: Vec<(&PartSpec, Vec<usize>)> = Vec::new();
            for (i, part) in parts.iter().enumerate() {
                match pieces.iter_mut().find(|(p, _)| p.name == part.name && p.group == part.group) {
                    Some((_, list)) => list.push(i),
                    None => pieces.push((part, vec![i])),
                }
            }
            // Groups in order of first mention; inside a group the first part listed is on top.
            let mut groups: Vec<(Option<String>, Vec<Layer>)> = Vec::new();
            let mut made = Vec::new();
            for (part, list) in pieces {
                let bytes = assignment.mask_of(&list);
                if !bytes.iter().any(|b| *b > 0) {
                    made.push(json!({ "name": part.name, "layer": null, "pixels": 0 }));
                    continue;
                }
                let mut surface =
                    photocraft_raster::Surface::from_interleaved(photocraft_color::PixelFormat::GRAY8, Rect::new(area.x0, area.y0, area.x1, area.y1), &bytes);
                surface.prune();
                let covered = bytes.iter().filter(|b| **b >= 128).count();
                let mut layer = Layer::new(part.name.clone(), LayerContent::Raster(pixels.clone()));
                layer.mask = Some(crate::commands::mask_at_depth(LayerMask { surface, ..LayerMask::reveal_all() }, doc.depth)?);
                made.push(json!({ "name": part.name, "group": part.group, "layer": layer.id.0, "pixels": covered }));
                match groups.iter_mut().find(|(g, _)| *g == part.group) {
                    Some((_, list)) => list.push(layer),
                    None => groups.push((part.group.clone(), vec![layer])),
                }
            }
            if made.iter().all(|m| m["layer"].is_null()) {
                return Err(EngineError::Other("no part was found in the image".into()));
            }
            // Insert bottom-up above the source so the first group ends on top.
            let mut top = id;
            for (group, mut list) in groups.into_iter().rev() {
                list.reverse(); // children are bottom-to-top
                match group {
                    Some(name) => {
                        let mut g = Layer::group(name, list);
                        if let LayerContent::Group(gc) = &mut g.content {
                            gc.expanded = false;
                        }
                        top = doc.insert_above(Some(top), g);
                    }
                    None => {
                        for l in list {
                            top = doc.insert_above(Some(top), l);
                        }
                    }
                }
            }
            doc.selection = None;
            *active = Some(top);
            Ok((made, filled, edge_added, islands_moved))
        },
        move |(made, filled, edge_added, islands_moved)| json!({ "source": id.0, "parts": made, "gapsFilled": filled, "edgeAdded": edge_added, "islandsMoved": islands_moved }),
    )
}

/// Split into Parts command spec (no menu item yet: driven by agents and scripts).
pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Split into Parts",
        menu: &[],
        shortcut: None,
        params: r##"{"layer":id?,"parts":[{"name":str,"group":str?,"points":[[x,y,1|0],…]?,"box":[x,y,w,h]?},…] (document pixels; 1 include, 0 exclude; each part needs an include point or a box),"refineEdge":px? (outline refined against the photo over this band; default about 2.5× the analysis's enlargement, 0 = off),"fillGaps":px? (largest enclosed unclaimed hole given to the part around it, and largest stray piece of a part moved to the part around it; default 0.03% of the image, 0 = off),"detail":bool=true (small boxed parts such as jewellery, straps and watches also get a zoomed analysis),"modelDir":str? (default PHOTOCRAFT_SAM_DIR: mobile_sam_image_encoder.onnx + sam_mask_decoder_single.onnx)} → {source, parts:[{name,group,layer,pixels}], gapsFilled, edgeAdded, islandsMoved} (each part becomes a masked copy of the layer inside its group; parts with the same name and group are pieces of one layer; overlapping parts go to the strongest match; one undo step)"##,
        enabled,
        run,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_points_boxes_and_groups() {
        let p = json!({"parts": [
            {"name": "Hair", "group": "Hair", "points": [[10, 20], [30, 40, 0]]},
            {"name": "Watch", "box": [5, 6, 7, 8]}
        ]});
        let parts = parse_parts(&p).unwrap();
        assert_eq!(parts[0].points, vec![(10.0, 20.0, true), (30.0, 40.0, false)]);
        assert_eq!(parts[0].group.as_deref(), Some("Hair"));
        assert_eq!(parts[1].rect, Some([5.0, 6.0, 12.0, 14.0]));
    }

    #[test]
    fn bad_params_are_errors_not_panics() {
        for p in [
            json!({}),
            json!({"parts": []}),
            json!({"parts": "x"}),
            json!({"parts": [{"points": [[1, 2]]}]}),
            json!({"parts": [{"name": "", "points": [[1, 2]]}]}),
            json!({"parts": [{"name": "A"}]}),
            json!({"parts": [{"name": "A", "points": [[1, 2, 0]]}]}),
            json!({"parts": [{"name": "A", "points": [[1]]}]}),
            json!({"parts": [{"name": "A", "points": [[1, 2, 7]]}]}),
            json!({"parts": [{"name": "A", "points": [["x", 2]]}]}),
            json!({"parts": [{"name": "A", "points": [[1e300, 2]]}]}),
            json!({"parts": [{"name": "A", "box": [0, 0, 0, 5]}]}),
            json!({"parts": [{"name": "A", "box": [0, 0, 5]}]}),
            json!({"parts": [{"name": 5, "points": [[1, 2]]}]}),
        ] {
            assert!(parse_parts(&p).is_err(), "{p} should be rejected");
        }
        let many: Vec<Value> = (0..=photocraft_ml_limits::MAX_PARTS).map(|i| json!({"name": format!("P{i}"), "points": [[1, 1]]})).collect();
        assert!(parse_parts(&json!({ "parts": many })).is_err());
    }

    #[test]
    fn small_boxes_get_merged_zoom_areas_and_large_ones_none() {
        let area = photocraft_geom::Rect::new(0, 0, 3584, 4800);
        let p = |name: &str, b: [f32; 4]| PartSpec { name: name.into(), group: None, points: vec![], rect: Some(b) };
        let parts = vec![
            p("Earring L", [1660.0, 790.0, 1710.0, 910.0]),
            p("Earring R", [1990.0, 710.0, 2050.0, 860.0]),
            p("Sandal", [1000.0, 4200.0, 1420.0, 4700.0]),
            p("Dress", [500.0, 1000.0, 3000.0, 4500.0]),
        ];
        let r = detail_regions(&parts, area);
        // Each earring and the sandal get an area; the dress is too large for one.
        assert_eq!(r.len(), 3, "{r:?}");
        // Boxes whose areas overlap share one merged area.
        let close = [p("Ring", [1000.0, 1000.0, 1040.0, 1040.0]), p("Bangle", [1100.0, 1000.0, 1180.0, 1060.0])];
        let m = detail_regions(&close, area);
        assert_eq!(m.len(), 1, "{m:?}");
        assert!(m[0].x0 <= 1000 && m[0].x1 >= 1180);
        assert!(r.iter().all(|x| x.width() >= 256 && x.height() >= 256 && area.intersect(x) == *x));
        assert!(detail_regions(&[p("Big", [0.0, 0.0, 3584.0, 4800.0])], area).is_empty());
        let many: Vec<_> = (0..20).map(|i| p("x", [i as f32 * 170.0, 0.0, i as f32 * 170.0 + 10.0, 10.0])).collect();
        assert!(detail_regions(&many, area).len() <= MAX_DETAIL_AREAS);
        assert!(parse_parts(&json!({"parts": [{"name": "A", "points": [[1, 1]]}], "detail": "x"})).is_ok());
    }

    #[test]
    fn missing_model_is_an_error() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
        let r = s.execute(CMD, json!({"parts": [{"name": "A", "points": [[10, 10]]}], "modelDir": "Z:/no/such/folder"}));
        assert!(r.is_err());
        for bad in [json!("x"), json!(-1), json!(1.5), json!({})] {
            let r = s.execute(CMD, json!({"parts": [{"name": "A", "points": [[10, 10]]}], "fillGaps": bad}));
            assert!(r.is_err(), "fillGaps {bad} should be rejected");
        }
        let r = s.execute(CMD, json!({"parts": [{"name": "A", "points": [[10, 10]]}], "detail": "yes"}));
        assert!(r.is_err());
        for bad in [json!("x"), json!(-1), json!({})] {
            let r = s.execute(CMD, json!({"parts": [{"name": "A", "points": [[10, 10]]}], "refineEdge": bad}));
            assert!(r.is_err(), "refineEdge {bad} should be rejected");
        }
    }

    /// Two pieces with one name become one layer covering both (real model; skipped without it).
    #[test]
    fn pieces_with_one_name_make_one_layer() {
        if std::env::var_os("PHOTOCRAFT_SAM_DIR").is_none() {
            return;
        }
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 256, "height": 256})).unwrap();
        s.edit("paint", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            let n = bg.channels();
            let mut data = Vec::new();
            for y in 0..256 {
                for x in 0..256 {
                    let red = (24..96).contains(&y) && ((24..96).contains(&x) || (160..232).contains(&x));
                    let c = if red { [0.9, 0.1, 0.1] } else { [0.1, 0.3, 0.9] };
                    data.extend_from_slice(&c[..n.min(3)]);
                    if n == 4 {
                        data.push(1.0);
                    }
                }
            }
            bg.write_region(photocraft_geom::Rect::new(0, 0, 256, 256), &data);
            Ok(())
        })
        .unwrap();
        let r = s
            .execute(
                CMD,
                json!({"parts": [
                    {"name": "Squares", "group": "Shapes", "points": [[60, 60], [196, 60, 0]]},
                    {"name": "Squares", "group": "Shapes", "points": [[196, 60], [60, 60, 0]]}
                ]}),
            )
            .unwrap();
        let made = r["parts"].as_array().unwrap();
        assert_eq!(made.len(), 1, "one layer for both pieces: {r}");
        let px = made[0]["pixels"].as_u64().unwrap();
        assert!((7_000..13_000).contains(&px), "both 72×72 squares: {px}");
        let doc = &s.active().unwrap().doc;
        let group = doc.layers.iter().find(|l| l.name == "Shapes").expect("group");
        assert_eq!(group.children().map(|c| c.len()), Some(1));
    }

    /// End to end with the real model when `PHOTOCRAFT_SAM_DIR` is set (skipped otherwise):
    /// a red square on blue splits into a "Square" layer inside a group, the rest unassigned.
    #[test]
    fn splits_a_square_into_a_grouped_masked_layer() {
        if std::env::var_os("PHOTOCRAFT_SAM_DIR").is_none() {
            return;
        }
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 256, "height": 256})).unwrap();
        s.edit("paint", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            let n = bg.channels();
            let mut data = Vec::new();
            for y in 0..256 {
                for x in 0..256 {
                    let c = if (64..192).contains(&x) && (64..192).contains(&y) { [0.9, 0.1, 0.1] } else { [0.1, 0.3, 0.9] };
                    data.extend_from_slice(&c[..n.min(3)]);
                    if n == 4 {
                        data.push(1.0);
                    }
                }
            }
            bg.write_region(photocraft_geom::Rect::new(0, 0, 256, 256), &data);
            Ok(())
        })
        .unwrap();
        let r = s.execute(CMD, json!({"parts": [{"name": "Square", "group": "Shapes", "points": [[128, 128]]}]})).unwrap();
        let part = &r["parts"][0];
        let px = part["pixels"].as_u64().unwrap();
        assert!((12_000..20_000).contains(&px), "about 128×128 px: {px}");
        let doc = &s.active().unwrap().doc;
        let group = doc.layers.iter().find(|l| l.name == "Shapes").expect("group");
        assert_eq!(group.children().map(|c| c.len()), Some(1));
        assert!(group.children().unwrap()[0].mask.is_some());
        // One undo removes the whole split.
        s.execute("edit.undo", json!({})).unwrap();
        assert!(s.active().unwrap().doc.layers.iter().all(|l| l.name != "Shapes"));
    }
}
