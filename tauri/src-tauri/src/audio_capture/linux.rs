use crate::audio_capture::AudioCaptureState;
use base64::{engine::general_purpose, Engine as _};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use hound::{WavSpec, WavWriter};
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

/// Start capturing system audio on Linux using PulseAudio monitor sources.
///
/// PulseAudio exposes "monitor" devices that mirror the output of each sink,
/// allowing us to capture whatever audio is currently playing on the system.
/// We use `cpal` with the default host (which will be PulseAudio or PipeWire
/// on modern Linux) and look for monitor input devices.
pub async fn start_capture(
    state: &AudioCaptureState,
    max_duration_secs: u32,
) -> Result<(), String> {
    // Reset previous samples
    state.reset();

    let samples = state.samples.clone();
    let sample_rate_arc = state.sample_rate.clone();
    let channels_arc = state.channels.clone();
    let stop_tx = state.stop_tx.clone();
    let error_arc = state.error.clone();

    // Use AtomicBool for stop signal (works across threads)
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_clone = stop_flag.clone();

    // Create tokio channel and spawn a task to bridge it to the AtomicBool
    let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(1);
    *stop_tx.lock().unwrap() = Some(tx);

    tokio::spawn(async move {
        rx.recv().await;
        stop_flag_clone.store(true, Ordering::Relaxed);
    });

    // Spawn capture on a dedicated thread
    thread::spawn(move || {
        let host = cpal::default_host();

        // Try to find a monitor device for system audio capture.
        // On PulseAudio/PipeWire, monitor sources have "monitor" in their name.
        let device = {
            let mut monitor_device = None;

            if let Ok(devices) = host.input_devices() {
                for d in devices {
                    if let Ok(name) = d.name() {
                        let name_lower = name.to_lowercase();
                        if name_lower.contains("monitor") {
                            eprintln!("Linux audio capture: Found monitor device: {}", name);
                            monitor_device = Some(d);
                            break;
                        }
                    }
                }
            }

            match monitor_device {
                Some(d) => d,
                None => {
                    // Fallback to default input device (microphone)
                    eprintln!("Linux audio capture: No monitor device found, falling back to default input");
                    match host.default_input_device() {
                        Some(d) => d,
                        None => {
                            let error_msg = "No audio input device available".to_string();
                            eprintln!("{}", error_msg);
                            *error_arc.lock().unwrap() = Some(error_msg);
                            return;
                        }
                    }
                }
            }
        };

        let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
        eprintln!("Linux audio capture: Using device: {}", device_name);

        // Get supported config
        let config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                let error_msg = format!("Failed to get default input config: {}", e);
                eprintln!("{}", error_msg);
                *error_arc.lock().unwrap() = Some(error_msg);
                return;
            }
        };

        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        let sample_format = config.sample_format();

        eprintln!(
            "Linux audio capture: Config - {}Hz, {} channels, format: {:?}",
            sample_rate, channels, sample_format
        );

        *sample_rate_arc.lock().unwrap() = sample_rate;
        *channels_arc.lock().unwrap() = channels;

        let stream_config = StreamConfig {
            channels,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let samples_clone = samples.clone();
        let error_arc_clone = error_arc.clone();
        let stop_flag_for_stream = stop_flag.clone();

        let err_fn = {
            let error_arc = error_arc.clone();
            move |err: cpal::StreamError| {
                let error_msg = format!("Stream error: {}", err);
                eprintln!("{}", error_msg);
                *error_arc.lock().unwrap() = Some(error_msg);
            }
        };

        let stream = match sample_format {
            SampleFormat::F32 => {
                let samples = samples_clone.clone();
                let stop = stop_flag_for_stream.clone();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        let mut guard = samples.lock().unwrap();
                        guard.extend_from_slice(data);
                    },
                    err_fn,
                    None,
                )
            }
            SampleFormat::I16 => {
                let samples = samples_clone.clone();
                let stop = stop_flag_for_stream.clone();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        let mut guard = samples.lock().unwrap();
                        for &s in data {
                            guard.push(s as f32 / 32768.0);
                        }
                    },
                    err_fn,
                    None,
                )
            }
            SampleFormat::U16 => {
                let samples = samples_clone.clone();
                let stop = stop_flag_for_stream.clone();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        let mut guard = samples.lock().unwrap();
                        for &s in data {
                            guard.push((s as f32 / 32768.0) - 1.0);
                        }
                    },
                    err_fn,
                    None,
                )
            }
            _ => {
                let error_msg = format!("Unsupported sample format: {:?}", sample_format);
                eprintln!("{}", error_msg);
                *error_arc_clone.lock().unwrap() = Some(error_msg);
                return;
            }
        };

        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                let error_msg = format!("Failed to build input stream: {}", e);
                eprintln!("{}", error_msg);
                *error_arc_clone.lock().unwrap() = Some(error_msg);
                return;
            }
        };

        if let Err(e) = stream.play() {
            let error_msg = format!("Failed to start stream: {}", e);
            eprintln!("{}", error_msg);
            *error_arc_clone.lock().unwrap() = Some(error_msg);
            return;
        }

        eprintln!("Linux audio capture: Stream started successfully");

        // Keep thread alive until stop signal
        loop {
            if stop_flag.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // Stream will be dropped here, stopping capture
        eprintln!("Linux audio capture: Stream stopped");
    });

    // Spawn timeout task
    let stop_tx_clone = state.stop_tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(max_duration_secs as u64)).await;
        let tx = stop_tx_clone.lock().unwrap().take();
        if let Some(tx) = tx {
            let _ = tx.send(()).await;
        }
    });

    Ok(())
}

pub async fn stop_capture(state: &AudioCaptureState) -> Result<String, String> {
    // Signal stop
    if let Some(tx) = state.stop_tx.lock().unwrap().take() {
        let _ = tx.send(());
    }

    // Wait a bit for capture to stop
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Check if there was an error during capture
    if let Some(error) = state.error.lock().unwrap().as_ref() {
        return Err(error.clone());
    }

    // Get samples
    let samples = state.samples.lock().unwrap().clone();
    let sample_rate = *state.sample_rate.lock().unwrap();
    let channels = *state.channels.lock().unwrap();

    if samples.is_empty() {
        return Err(
            "No audio samples captured. Make sure audio is playing on your system during recording."
                .to_string(),
        );
    }

    // Convert to WAV
    let wav_data = samples_to_wav(&samples, sample_rate, channels)?;

    // Encode to base64
    let base64_data = general_purpose::STANDARD.encode(&wav_data);

    Ok(base64_data)
}

pub fn is_supported() -> bool {
    // Check if we can find a monitor device for system audio capture
    let host = cpal::default_host();
    if let Ok(devices) = host.input_devices() {
        for d in devices {
            if let Ok(name) = d.name() {
                if name.to_lowercase().contains("monitor") {
                    return true;
                }
            }
        }
    }
    // Even without a monitor, basic input capture is available
    host.default_input_device().is_some()
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

    let mut writer =
        WavWriter::new(cursor, spec).map_err(|e| format!("Failed to create WAV writer: {}", e))?;

    // Convert f32 samples to i16
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        writer
            .write_sample(i16_sample)
            .map_err(|e| format!("Failed to write sample: {}", e))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV: {}", e))?;

    Ok(buffer)
}
