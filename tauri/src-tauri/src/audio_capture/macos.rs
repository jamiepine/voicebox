use crate::audio_capture::AudioCaptureState;

pub async fn start_capture(
    _state: &AudioCaptureState,
    _max_duration_secs: u32,
) -> Result<(), String> {
    Err("System audio capture is disabled in this build to prevent a startup crash.".to_string())
}

pub async fn stop_capture(_state: &AudioCaptureState) -> Result<String, String> {
    Err("System audio capture is disabled.".to_string())
}

pub fn is_supported() -> bool {
    false
}
