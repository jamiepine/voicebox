use crate::audio_capture::AudioCaptureState;
use screencapturekit::prelude::*;
use std::sync::{Arc, Mutex};
use base64::Engine;

// Handler for receiving audio samples from ScreenCaptureKit
struct AudioHandler {
    samples: Arc<Mutex<Vec<f32>>>,
    sample_rate: Arc<Mutex<u32>>,
    channels: Arc<Mutex<u16>>,
}

impl SCStreamOutputTrait for AudioHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, output_type: SCStreamOutputType) {
        // Only process audio samples
        if output_type != SCStreamOutputType::Audio {
            return;
        }

        // Get audio format description
        let format_desc = sample.format_description();
        if format_desc.is_none() {
            return;
        }

        // Extract audio samples from the buffer
        if let Some(audio_buffer) = sample.audio_buffer_list() {
            // Update channel count from the buffer
            let num_channels = audio_buffer.num_buffers() as u16;
            *self.channels.lock().unwrap() = num_channels;

            // Extract the sample rate from format description
            // ScreenCaptureKit typically provides 48kHz audio
            *self.sample_rate.lock().unwrap() = 48000;

            // Process each audio buffer
            for i in 0..audio_buffer.num_buffers() {
                if let Some(buffer) = audio_buffer.buffer(i as usize) {
                    let data = buffer.data();
                    let num_frames = buffer.data_byte_size() / std::mem::size_of::<f32>();

                    if !data.is_empty() && num_frames > 0 {
                        // Convert bytes to f32 samples
                        let data_ptr = data.as_ptr() as *const f32;
                        let audio_data = unsafe {
                            std::slice::from_raw_parts(data_ptr, num_frames)
                        };

                        // Append samples to our collection
                        let mut samples = self.samples.lock().unwrap();
                        samples.extend_from_slice(audio_data);
                    }
                }
            }
        }
    }
}

pub async fn start_capture(
    state: &AudioCaptureState,
    max_duration_secs: u32,
) -> Result<(), String> {
    // Reset state
    state.reset();

    // Get shareable content (displays and windows)
    let content = SCShareableContent::get()
        .map_err(|e| format!("Failed to get shareable content. Please grant Screen Recording permission in System Settings → Privacy & Security → Screen & System Audio Recording. Error: {:?}", e))?;

    // Get the main display for audio capture
    let displays = content.displays();
    if displays.is_empty() {
        return Err("No displays found for audio capture".to_string());
    }
    let display = &displays[0];

    // Create a content filter to capture desktop audio
    // We use display-based capture to get system audio
    let filter = SCContentFilter::create()
        .with_display(display)
        .with_excluding_windows(&[])
        .build();

    // Configure audio capture settings
    let config = SCStreamConfiguration::new()
        .with_captures_audio(true) // Enable audio capture
        .with_sample_rate(48000) // 48kHz sample rate
        .with_channel_count(2); // Stereo

    // Create the stream
    let mut stream = SCStream::new(&filter, &config);

    // Create the audio handler
    let handler = AudioHandler {
        samples: state.samples.clone(),
        sample_rate: state.sample_rate.clone(),
        channels: state.channels.clone(),
    };

    // Add the audio output handler
    stream.add_output_handler(handler, SCStreamOutputType::Audio);

    // Start capturing
    stream.start_capture()
        .map_err(|e| format!("Failed to start audio capture: {:?}", e))?;

    // Store the stream in state so we can stop it later
    *state.stream.lock().unwrap() = Some(stream);

    // Set up a timer to auto-stop after max duration
    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::channel::<()>(1);
    *state.stop_tx.lock().unwrap() = Some(stop_tx);

    let state_clone = Arc::new(state.clone());
    tokio::spawn(async move {
        let timeout = tokio::time::Duration::from_secs(max_duration_secs as u64);
        tokio::select! {
            _ = tokio::time::sleep(timeout) => {
                // Auto-stop after max duration
                if let Some(stream) = state_clone.stream.lock().unwrap().take() {
                    let _ = stream.stop_capture();
                }
            }
            _ = stop_rx.recv() => {
                // Manual stop requested
            }
        }
    });

    Ok(())
}

pub async fn stop_capture(state: &AudioCaptureState) -> Result<String, String> {
    // Stop the stream
    if let Some(stream) = state.stream.lock().unwrap().take() {
        stream.stop_capture()
            .map_err(|e| format!("Failed to stop capture: {:?}", e))?;
    }

    // Signal the timer task to stop
    let stop_tx = state.stop_tx.lock().unwrap().take();
    if let Some(tx) = stop_tx {
        let _ = tx.send(()).await;
    }

    // Get the captured samples
    let samples = state.samples.lock().unwrap();
    let sample_rate = *state.sample_rate.lock().unwrap();
    let channels = *state.channels.lock().unwrap();

    if samples.is_empty() {
        return Err("No audio data captured. Make sure audio is playing and permissions are granted.".to_string());
    }

    // Convert f32 samples to i16 PCM for WAV encoding
    let pcm_samples: Vec<i16> = samples
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect();

    // Write to WAV format
    let mut wav_buffer = Vec::new();
    {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut wav_buffer), spec)
            .map_err(|e| format!("Failed to create WAV writer: {}", e))?;

        for sample in pcm_samples {
            writer.write_sample(sample)
                .map_err(|e| format!("Failed to write sample: {}", e))?;
        }

        writer.finalize()
            .map_err(|e| format!("Failed to finalize WAV: {}", e))?;
    }

    // Encode to base64
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&wav_buffer);

    Ok(base64_data)
}

pub fn is_supported() -> bool {
    // ScreenCaptureKit is available on macOS 13.0+
    // Since we're already compiling for macOS, just return true
    true
}
