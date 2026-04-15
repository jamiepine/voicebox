use crate::audio_capture::AudioCaptureState;
use base64::{engine::general_purpose, Engine as _};
use hound::{WavSpec, WavWriter};
use screencapturekit::{
    cm::CMSampleBuffer,
    shareable_content::SCShareableContent,
    stream::{
        configuration::SCStreamConfiguration,
        content_filter::SCContentFilter,
        output_trait::SCStreamOutputTrait,
        output_type::SCStreamOutputType,
        sc_stream::SCStream,
    },
};
use std::io::Cursor;
use std::process::Command;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

pub async fn start_capture(
    state: &AudioCaptureState,
    max_duration_secs: u32,
) -> Result<(), String> {
    if !is_supported() {
        return Err("System audio capture requires macOS 12.3 or newer.".to_string());
    }

    // Reset previous samples
    state.reset();

    // Get shareable content
    let content = SCShareableContent::get()
        .map_err(|e| format!("Failed to get shareable content: {}", e))?;

    // Get first display
    let displays = content.displays();
    if displays.is_empty() {
        return Err("No displays available".to_string());
    }
    let display = &displays[0];

    // Create content filter for desktop audio
    let filter = SCContentFilter::create()
        .with_display(display)
        .with_excluding_windows(&[])
        .build();

    // Create stream configuration - audio only
    let mut config = SCStreamConfiguration::default();
    config.set_captures_audio(true);
    config.set_excludes_current_process_audio(false);
    config.set_sample_rate(48000); // Use i32 directly
    config.set_channel_count(2); // Use i32 directly

    // Create stream using builder
    let (tx, mut rx) = mpsc::channel::<()>(1);
    *state.stop_tx.lock().unwrap() = Some(tx);

    let samples = state.samples.clone();
    let sample_rate = state.sample_rate.clone();
    let channels = state.channels.clone();

    // Set sample rate and channels
    *sample_rate.lock().unwrap() = 48000;
    *channels.lock().unwrap() = 2;

    // Create output handler struct
    struct AudioHandler {
        samples: Arc<Mutex<Vec<f32>>>,
    }

    impl SCStreamOutputTrait for AudioHandler {
        fn did_output_sample_buffer(
            &self,
            sample: CMSampleBuffer,
            _type: SCStreamOutputType,
        ) {
            if _type == SCStreamOutputType::Audio {
                if let Ok(audio_samples) = extract_audio_samples(sample) {
                    let mut samples_guard = self.samples.lock().unwrap();
                    samples_guard.extend_from_slice(&audio_samples);
                }
            }
        }
    }

    let handler = AudioHandler {
        samples: samples.clone(),
    };

    // Create stream
    let mut stream = SCStream::new(&filter, &config);
    
    // Add output handler for audio (order: handler, then output_type)
    stream.add_output_handler(handler, SCStreamOutputType::Audio);

    // Store stream reference
    *state.stream.lock().unwrap() = Some(stream.clone());

    stream.start_capture().map_err(|e| format!("Failed to start capture: {}", e))?;

    // Spawn task to stop after max duration
    let stream_clone = stream.clone();
    tokio::spawn(async move {
        tokio::select! {
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(max_duration_secs as u64)) => {
                // Timeout reached
            }
            _ = rx.recv() => {
                // Manual stop
            }
        }
        let _ = stream_clone.stop_capture();
    });

    Ok(())
}

pub async fn stop_capture(state: &AudioCaptureState) -> Result<String, String> {
    // Signal stop
    if let Some(tx) = state.stop_tx.lock().unwrap().take() {
        let _ = tx.send(());
    }

    // Stop stream if still active
    if let Some(stream) = state.stream.lock().unwrap().take() {
        let _ = stream.stop_capture();
    }

    // Wait a bit for capture to stop
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Get samples
    let samples = state.samples.lock().unwrap().clone();
    let sample_rate = *state.sample_rate.lock().unwrap();
    let channels = *state.channels.lock().unwrap();

    if samples.is_empty() {
        return Err("No audio samples captured".to_string());
    }

    // Convert to WAV
    let wav_data = samples_to_wav(&samples, sample_rate, channels)?;
    
    // Encode to base64
    let base64_data = general_purpose::STANDARD.encode(&wav_data);
    
    Ok(base64_data)
}

pub fn is_supported() -> bool {
    macos_version_at_least(12, 3)
}

fn macos_version_at_least(required_major: u64, required_minor: u64) -> bool {
    let output = match Command::new("sw_vers").arg("-productVersion").output() {
        Ok(output) if output.status.success() => output,
        _ => return false,
    };

    let version = String::from_utf8_lossy(&output.stdout);
    let mut parts = version.trim().split('.');

    let major = parts.next().and_then(|part| part.parse::<u64>().ok()).unwrap_or(0);
    let minor = parts.next().and_then(|part| part.parse::<u64>().ok()).unwrap_or(0);

    major > required_major || (major == required_major && minor >= required_minor)
}

fn extract_audio_samples(sample_buffer: CMSampleBuffer) -> Result<Vec<f32>, String> {
    // Use the crate's built-in method to get audio buffer list
    let audio_buffer_list = sample_buffer
        .audio_buffer_list()
        .ok_or_else(|| "Failed to get audio buffer list".to_string())?;

    let buffers: Vec<_> = audio_buffer_list.iter().collect();
    let num_buffers = buffers.len();
    
    if num_buffers == 0 {
        return Ok(Vec::new());
    }

    // ScreenCaptureKit on macOS provides audio in Float32 format
    // The audio can be either:
    // - Interleaved (1 buffer with L,R,L,R,... samples)
    // - Planar (2 buffers, one for L channel, one for R channel)
    
    if num_buffers == 1 {
        // Interleaved stereo or mono in a single buffer
        let buffer = &buffers[0];
        let data_bytes = buffer.data();
        let num_samples = data_bytes.len() / std::mem::size_of::<f32>();
        
        if num_samples > 0 {
            unsafe {
                let data_ptr = data_bytes.as_ptr() as *const f32;
                let data = std::slice::from_raw_parts(data_ptr, num_samples);
                return Ok(data.to_vec());
            }
        }
    } else {
        // Planar format - separate buffer for each channel
        // We need to interleave them: L0, R0, L1, R1, ...
        let mut channel_data: Vec<Vec<f32>> = Vec::new();
        let mut max_samples = 0;
        
        for buffer in &buffers {
            let data_bytes = buffer.data();
            let num_samples = data_bytes.len() / std::mem::size_of::<f32>();
            
            if num_samples > 0 {
                unsafe {
                    let data_ptr = data_bytes.as_ptr() as *const f32;
                    let data = std::slice::from_raw_parts(data_ptr, num_samples);
                    channel_data.push(data.to_vec());
                    max_samples = max_samples.max(num_samples);
                }
            }
        }
        
        // Interleave the channels
        let mut interleaved = Vec::with_capacity(max_samples * num_buffers);
        for i in 0..max_samples {
            for channel in &channel_data {
                if i < channel.len() {
                    interleaved.push(channel[i]);
                } else {
                    interleaved.push(0.0); // Pad with silence if needed
                }
            }
        }
        
        return Ok(interleaved);
    }

    Ok(Vec::new())
}

fn samples_to_wav(samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<u8>, String> {
    let mut buffer = Vec::new();
    let cursor = Cursor::new(&mut buffer);
    
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::new(cursor, spec)
        .map_err(|e| format!("Failed to create WAV writer: {}", e))?;

    // Convert f32 samples to i16
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        writer.write_sample(i16_sample)
            .map_err(|e| format!("Failed to write sample: {}", e))?;
    }

    writer.finalize()
        .map_err(|e| format!("Failed to finalize WAV: {}", e))?;

    Ok(buffer)
}
