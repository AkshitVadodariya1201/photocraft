//! Machine-learning models for PhotoCraft, run in pure Rust through `tract` (no native runtime).
//!
//! The first model family is promptable segmentation: MobileSAM / SAM exported to ONNX as an
//! image encoder plus a mask decoder. Points and boxes in document pixels go in; per-part mask
//! scores come out, and [`PartMap`] resolves overlapping parts so every pixel belongs to at most
//! one part. Model files are passed as bytes, so this crate never touches the file system.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod sam;

pub use sam::{Detail, Embedding, Logits, MAX_PARTS, MAX_POINTS, PartMap, PointLabel, Prompt, Rgb, SAM_SIDE, SamModel};

/// Errors from loading or running a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlError {
    /// The model file is missing, corrupt or not the expected network.
    Model(String),
    /// The image or prompt can't be used (empty, too many points, outside the image…).
    Input(String),
}

impl std::fmt::Display for MlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MlError::Model(m) => write!(f, "model error: {m}"),
            MlError::Input(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for MlError {}

/// Result of an ML operation.
pub type Result<T> = std::result::Result<T, MlError>;
