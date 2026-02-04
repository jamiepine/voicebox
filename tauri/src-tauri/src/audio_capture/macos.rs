use crate::audio_capture::AudioCaptureState;

pub fn is_supported() -> bool {
    false
}

pub async fn start_capture(
    _state: &AudioCaptureState,
    _max_duration_secs: u32,
) -> Result<(), String> {
    Err("Desktop audio capture is not supported on this platform yet".to_string())
}

pub async fn stop_capture(_state: &AudioCaptureState) -> Result<String, String> {
    Err("Desktop audio capture is not supported on this platform yet".to_string())
}
