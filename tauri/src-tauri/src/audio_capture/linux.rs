use crate::audio_capture::AudioCaptureState;
use base64::{engine::general_purpose, Engine as _};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, StreamConfig};
use hound::{WavSpec, WavWriter};
use std::io::Cursor;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

fn pactl_stdout(args: &[&str]) -> Option<String> {
    let output = Command::new("pactl").args(args).output().ok()?;

    if !output.status.success() {
        return None;
    }

    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Try to find a PulseAudio/PipeWire monitor source using `pactl`.
/// Returns the source name (e.g. "alsa_output.pci-0000_0d_00.6.analog-stereo.monitor") if found.
fn find_monitor_source_via_pactl() -> Option<String> {
    let stdout = pactl_stdout(&["list", "short", "sources"])?;

    // First, try to find the monitor of the default sink
    let default_sink = pactl_stdout(&["get-default-sink"]);

    // If we know the default sink, look for its .monitor specifically
    if let Some(sink_name) = &default_sink {
        let monitor_name = format!("{}.monitor", sink_name);
        for line in stdout.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 && parts[1] == monitor_name {
                eprintln!(
                    "Linux audio capture: Found default sink monitor via pactl: {}",
                    monitor_name
                );
                return Some(monitor_name);
            }
        }
    }

    // Fallback: find any .monitor source
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 && parts[1].ends_with(".monitor") {
            let name = parts[1].to_string();
            eprintln!(
                "Linux audio capture: Found monitor source via pactl: {}",
                name
            );
            return Some(name);
        }
    }

    None
}

fn find_default_source_via_pactl() -> Option<String> {
    pactl_stdout(&["get-default-source"]).filter(|source| !source.is_empty())
}

fn set_default_source_via_pactl(source: &str) -> Result<(), String> {
    let output = Command::new("pactl")
        .args(["set-default-source", source])
        .output()
        .map_err(|e| format!("Failed to run pactl set-default-source: {}", e))?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if stderr.is_empty() {
        Err(format!(
            "pactl set-default-source {} failed with status {}",
            source, output.status
        ))
    } else {
        Err(format!(
            "pactl set-default-source {} failed: {}",
            source, stderr
        ))
    }
}

struct PulseDefaultSourceGuard {
    previous_source: Option<String>,
    restored: bool,
}

impl PulseDefaultSourceGuard {
    fn switch_to(target_source: &str) -> Result<Self, String> {
        let previous_source = find_default_source_via_pactl().ok_or_else(|| {
            "Failed to read PulseAudio/PipeWire default source before routing monitor capture"
                .to_string()
        })?;

        if previous_source == target_source {
            eprintln!(
                "Linux audio capture: PulseAudio/PipeWire default source already points at monitor: {}",
                target_source
            );
            return Ok(Self {
                previous_source: None,
                restored: true,
            });
        }

        set_default_source_via_pactl(target_source)?;
        eprintln!(
            "Linux audio capture: Temporarily set PulseAudio/PipeWire default source to monitor: {}",
            target_source
        );

        Ok(Self {
            previous_source: Some(previous_source),
            restored: false,
        })
    }

    fn restore(&mut self) {
        if self.restored {
            return;
        }

        self.restored = true;

        if let Some(previous_source) = &self.previous_source {
            match set_default_source_via_pactl(previous_source) {
                Ok(()) => eprintln!(
                    "Linux audio capture: Restored PulseAudio/PipeWire default source: {}",
                    previous_source
                ),
                Err(e) => eprintln!(
                    "Linux audio capture: Failed to restore PulseAudio/PipeWire default source: {}",
                    e
                ),
            }
        }
    }
}

impl Drop for PulseDefaultSourceGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

#[derive(Copy, Clone)]
enum DeviceSelectionTier {
    PactlExact,
    PactlSuffix,
    PactlNormalized,
    MonitorAlias,
    PulseDefaultSource,
}

impl DeviceSelectionTier {
    fn label(self) -> &'static str {
        match self {
            Self::PactlExact => "pactl exact source-name match",
            Self::PactlSuffix => "pactl source-name suffix match",
            Self::PactlNormalized => "pactl normalized source-name match",
            Self::MonitorAlias => "ALSA monitor alias match",
            Self::PulseDefaultSource => "ALSA pulse device routed via pactl default source",
        }
    }
}

struct SelectedDevice {
    device: Device,
    name: String,
    tier: DeviceSelectionTier,
    pulse_source_guard: Option<PulseDefaultSourceGuard>,
}

fn normalized_device_key(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn is_monitor_alias_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".monitor")
        || lower.ends_with("-monitor")
        || lower.ends_with("_monitor")
        || lower.contains(":monitor")
        || lower.contains("monitor:")
        || lower.contains(" monitor")
        || lower.contains("monitor")
}

fn classify_direct_monitor_device(
    name: &str,
    monitor_source: Option<&str>,
) -> Option<DeviceSelectionTier> {
    if let Some(source_name) = monitor_source {
        if name == source_name {
            return Some(DeviceSelectionTier::PactlExact);
        }

        if name.ends_with(source_name) {
            return Some(DeviceSelectionTier::PactlSuffix);
        }

        let normalized_name = normalized_device_key(name);
        let normalized_source = normalized_device_key(source_name);
        if !normalized_source.is_empty() && normalized_name.contains(&normalized_source) {
            return Some(DeviceSelectionTier::PactlNormalized);
        }
    }

    if is_monitor_alias_name(name) {
        return Some(DeviceSelectionTier::MonitorAlias);
    }

    None
}

fn find_direct_monitor_device(host: &Host, monitor_source: Option<&str>) -> Option<SelectedDevice> {
    let mut suffix_monitor_device = None;
    let mut normalized_monitor_device = None;
    let mut alias_monitor_device = None;

    if let Ok(devices) = host.input_devices() {
        for d in devices {
            if let Ok(name) = d.name() {
                match classify_direct_monitor_device(&name, monitor_source) {
                    Some(DeviceSelectionTier::PactlExact) => {
                        return Some(SelectedDevice {
                            device: d,
                            name,
                            tier: DeviceSelectionTier::PactlExact,
                            pulse_source_guard: None,
                        });
                    }
                    Some(DeviceSelectionTier::PactlSuffix) => {
                        if suffix_monitor_device.is_none() {
                            suffix_monitor_device = Some((d, name));
                        }
                    }
                    Some(DeviceSelectionTier::PactlNormalized) => {
                        if normalized_monitor_device.is_none() {
                            normalized_monitor_device = Some((d, name));
                        }
                    }
                    Some(DeviceSelectionTier::MonitorAlias) => {
                        if alias_monitor_device.is_none() {
                            alias_monitor_device = Some((d, name));
                        }
                    }
                    Some(DeviceSelectionTier::PulseDefaultSource) | None => {}
                }
            }
        }
    }

    suffix_monitor_device
        .map(|(device, name)| SelectedDevice {
            device,
            name,
            tier: DeviceSelectionTier::PactlSuffix,
            pulse_source_guard: None,
        })
        .or_else(|| {
            normalized_monitor_device.map(|(device, name)| SelectedDevice {
                device,
                name,
                tier: DeviceSelectionTier::PactlNormalized,
                pulse_source_guard: None,
            })
        })
        .or_else(|| {
            alias_monitor_device.map(|(device, name)| SelectedDevice {
                device,
                name,
                tier: DeviceSelectionTier::MonitorAlias,
                pulse_source_guard: None,
            })
        })
}

fn find_pulse_input_device(host: &Host) -> Option<(Device, String)> {
    let devices = host.input_devices().ok()?;

    for device in devices {
        if let Ok(name) = device.name() {
            if name == "pulse" || name.starts_with("pulse:") {
                return Some((device, name));
            }
        }
    }

    None
}

fn route_pulse_device_to_monitor(
    host: &Host,
    monitor_source: &str,
) -> Result<SelectedDevice, String> {
    let guard = PulseDefaultSourceGuard::switch_to(monitor_source)?;
    let (device, name) = find_pulse_input_device(host).ok_or_else(|| {
        format!(
            "Found PulseAudio/PipeWire monitor source \"{}\" via pactl, but CPAL's ALSA host did not expose a \"pulse\" input PCM after routing",
            monitor_source
        )
    })?;

    Ok(SelectedDevice {
        device,
        name,
        tier: DeviceSelectionTier::PulseDefaultSource,
        pulse_source_guard: Some(guard),
    })
}

fn select_capture_device(
    host: &Host,
    monitor_source: Option<&str>,
) -> Result<SelectedDevice, String> {
    if let Some(device) = find_direct_monitor_device(host, monitor_source) {
        return Ok(device);
    }

    if let Some(source_name) = monitor_source {
        return route_pulse_device_to_monitor(host, source_name).map_err(|e| {
            format!(
                "{}. Refusing to use the default input device because it may capture the microphone instead of system audio.",
                e
            )
        });
    }

    Err(
        "No PulseAudio/PipeWire monitor source was found via pactl, and CPAL's ALSA host did not expose any monitor input PCM. Refusing to use the default input device because it may capture the microphone instead of system audio."
            .to_string(),
    )
}

/// Start capturing system audio on Linux.
///
/// CPAL 0.15 exposes ALSA on Linux, not native PulseAudio/PipeWire hosts. We
/// therefore only use ALSA devices that look like monitor aliases. If the
/// normal ALSA `pulse` PCM is the only viable route, we use `pactl` to point the
/// PulseAudio/PipeWire default source at the discovered monitor while opening
/// that PCM, then restore the previous default source. If neither path is
/// available, capture fails instead of falling back to the user's microphone.
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
        // Try to find a monitor source before selecting a cpal device.
        let monitor_source = find_monitor_source_via_pactl();

        let host = cpal::default_host();

        let mut selected_device = match select_capture_device(&host, monitor_source.as_deref()) {
            Ok(device) => device,
            Err(e) => {
                eprintln!("Linux audio capture: {}", e);
                *error_arc.lock().unwrap() = Some(e);
                return;
            }
        };

        eprintln!(
            "Linux audio capture: Using device via {}: {}",
            selected_device.tier.label(),
            selected_device.name
        );

        // Get supported config
        let config = match selected_device.device.default_input_config() {
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
                selected_device.device.build_input_stream(
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
                selected_device.device.build_input_stream(
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
                selected_device.device.build_input_stream(
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
        if let Some(mut guard) = selected_device.pulse_source_guard.take() {
            guard.restore();
        }

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
    let monitor_source = find_monitor_source_via_pactl();
    let host = cpal::default_host();

    if find_direct_monitor_device(&host, monitor_source.as_deref()).is_some() {
        return true;
    }

    monitor_source.is_some() && find_pulse_input_device(&host).is_some()
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
