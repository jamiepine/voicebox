#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "macos")]
pub use macos::*;
#[cfg(target_os = "windows")]
pub use windows::*;
#[cfg(target_os = "linux")]
pub use linux::*;

use std::sync::{Arc, Mutex};

#[cfg(target_os = "macos")]
use screencapturekit::stream::sc_stream::SCStream;

#[derive(Clone)]
pub struct AudioCaptureState {
    pub samples: Arc<Mutex<Vec<f32>>>,
    pub sample_rate: Arc<Mutex<u32>>,
    pub channels: Arc<Mutex<u16>>,
    pub stop_tx: Arc<Mutex<Option<tokio::sync::mpsc::Sender<()>>>>,
    pub error: Arc<Mutex<Option<String>>>,
    #[cfg(target_os = "macos")]
    pub stream: Arc<Mutex<Option<SCStream>>>,
}

impl AudioCaptureState {
    pub fn new() -> Self {
        Self {
            samples: Arc::new(Mutex::new(Vec::new())),
            sample_rate: Arc::new(Mutex::new(44100)),
            channels: Arc::new(Mutex::new(2)),
            stop_tx: Arc::new(Mutex::new(None)),
            error: Arc::new(Mutex::new(None)),
            #[cfg(target_os = "macos")]
            stream: Arc::new(Mutex::new(None)),
        }
    }

    pub fn reset(&self) {
        *self.samples.lock().unwrap() = Vec::new();
        *self.error.lock().unwrap() = None;
    }
}
