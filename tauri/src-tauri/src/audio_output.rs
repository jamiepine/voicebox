use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, StreamConfig};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// ── CoreAudio direct enumeration (macOS only) ────────────────────────────────
// cpal's output_devices() misses HDMI/DisplayPort and some other transports on
// macOS because it only walks kAudioObjectPropertyScopeOutput devices that
// happen to have streams.  Querying AudioObjectGetPropertyData directly with
// kAudioHardwarePropertyDevices gives the full list that Audio MIDI Setup shows.
#[cfg(target_os = "macos")]
mod coreaudio_enum {
    use coreaudio_sys::{
        kAudioHardwarePropertyDefaultInputDevice,
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMaster,
        kAudioObjectSystemObject,
        kAudioDevicePropertyDeviceNameCFString,
        kAudioDevicePropertyStreams,
        kAudioObjectPropertyScopeOutput,
        kAudioObjectPropertyScopeInput,
        AudioDeviceID,
        AudioObjectGetPropertyData,
        AudioObjectGetPropertyDataSize,
        AudioObjectPropertyAddress,
    };
    use core_foundation_sys::string::{
        CFStringGetCStringPtr, CFStringRef, kCFStringEncodingUTF8,
    };
    use std::mem;
    use std::ffi::CStr;

    fn get_all_device_ids() -> Vec<AudioDeviceID> {
        unsafe {
            let addr = AudioObjectPropertyAddress {
                mSelector: kAudioHardwarePropertyDevices,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMaster,
            };
            let mut size: u32 = 0;
            if AudioObjectGetPropertyDataSize(
                kAudioObjectSystemObject,
                &addr,
                0,
                std::ptr::null(),
                &mut size,
            ) != 0 { return vec![]; }

            let count = size as usize / mem::size_of::<AudioDeviceID>();
            let mut ids: Vec<AudioDeviceID> = vec![0u32; count];
            if AudioObjectGetPropertyData(
                kAudioObjectSystemObject,
                &addr,
                0,
                std::ptr::null(),
                &mut size,
                ids.as_mut_ptr() as *mut _,
            ) != 0 { return vec![]; }
            ids
        }
    }

    fn device_name(id: AudioDeviceID) -> Option<String> {
        unsafe {
            let addr = AudioObjectPropertyAddress {
                mSelector: kAudioDevicePropertyDeviceNameCFString,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMaster,
            };
            let mut cf_str: CFStringRef = std::ptr::null();
            let mut size = mem::size_of::<CFStringRef>() as u32;
            if AudioObjectGetPropertyData(
                id,
                &addr,
                0,
                std::ptr::null(),
                &mut size,
                &mut cf_str as *mut _ as *mut _,
            ) != 0 { return None; }
            if cf_str.is_null() { return None; }
            let ptr = CFStringGetCStringPtr(cf_str, kCFStringEncodingUTF8);
            if ptr.is_null() { return None; }
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }

    /// Returns true if the device has at least one stream in `scope`
    /// (kAudioObjectPropertyScopeOutput or kAudioObjectPropertyScopeInput).
    fn has_streams(id: AudioDeviceID, scope: u32) -> bool {
        unsafe {
            let addr = AudioObjectPropertyAddress {
                mSelector: kAudioDevicePropertyStreams,
                mScope: scope,
                mElement: kAudioObjectPropertyElementMaster,
            };
            let mut size: u32 = 0;
            AudioObjectGetPropertyDataSize(
                id,
                &addr,
                0,
                std::ptr::null(),
                &mut size,
            ) == 0 && size > 0
        }
    }

    fn default_device_id(selector: u32) -> Option<AudioDeviceID> {
        unsafe {
            let addr = AudioObjectPropertyAddress {
                mSelector: selector,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMaster,
            };
            let mut id: AudioDeviceID = 0;
            let mut size = mem::size_of::<AudioDeviceID>() as u32;
            if AudioObjectGetPropertyData(
                kAudioObjectSystemObject,
                &addr,
                0,
                std::ptr::null(),
                &mut size,
                &mut id as *mut _ as *mut _,
            ) != 0 { return None; }
            Some(id)
        }
    }

    pub struct CoreAudioDevice {
        pub id: String,
        pub name: String,
        pub is_default: bool,
    }

    pub fn list_output_devices() -> Vec<CoreAudioDevice> {
        let default_id = default_device_id(kAudioHardwarePropertyDefaultOutputDevice);
        get_all_device_ids()
            .into_iter()
            .filter(|&id| has_streams(id, kAudioObjectPropertyScopeOutput))
            .filter_map(|id| {
                let name = device_name(id)?;
                let dev_id = format!("device_{}", name.replace(' ', "_").to_lowercase());
                let is_default = default_id.map_or(false, |d| d == id);
                Some(CoreAudioDevice { id: dev_id, name, is_default })
            })
            .collect()
    }

    pub fn list_input_devices() -> Vec<CoreAudioDevice> {
        let default_id = default_device_id(kAudioHardwarePropertyDefaultInputDevice);
        get_all_device_ids()
            .into_iter()
            .filter(|&id| has_streams(id, kAudioObjectPropertyScopeInput))
            .filter_map(|id| {
                let name = device_name(id)?;
                let dev_id = format!("input_{}", name.replace(' ', "_").to_lowercase());
                let is_default = default_id.map_or(false, |d| d == id);
                Some(CoreAudioDevice { id: dev_id, name, is_default })
            })
            .collect()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AudioOutputDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AudioInputDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

pub struct AudioOutputState {
    stop_flag: Arc<AtomicBool>,
}

impl AudioOutputState {
    pub fn new() -> Self {
        Self {
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    fn host() -> Host {
        // Create a fresh host each time so newly connected devices are visible.
        // cpal's default_host() on macOS re-queries CoreAudio on each call.
        cpal::default_host()
    }

    pub fn stop_all_playback(&self) -> Result<(), String> {
        eprintln!("stop_all_playback: Setting stop flag");
        self.stop_flag.store(true, Ordering::Relaxed);
        eprintln!("stop_all_playback: Stop flag set - active streams will output silence");
        Ok(())
    }

    pub fn list_output_devices(&self) -> Result<Vec<AudioOutputDevice>, String> {
        // On macOS use CoreAudio directly — cpal misses HDMI/DisplayPort devices.
        #[cfg(target_os = "macos")]
        {
            let devices = coreaudio_enum::list_output_devices();
            return Ok(devices
                .into_iter()
                .map(|d| AudioOutputDevice { id: d.id, name: d.name, is_default: d.is_default })
                .collect());
        }

        // Fallback: cpal (Windows / Linux)
        #[cfg(not(target_os = "macos"))]
        {
            let host = Self::host();
            let devices = host
                .output_devices()
                .map_err(|e| format!("Failed to enumerate output devices: {}", e))?;
            let default_device = host.default_output_device();
            let mut result = Vec::new();
            for device in devices {
                let name = device
                    .name()
                    .map_err(|e| format!("Failed to get device name: {}", e))?;
                let id = format!("device_{}", name.replace(' ', "_").to_lowercase());
                let is_default = default_device
                    .as_ref()
                    .map(|d| d.name().unwrap_or_default() == name)
                    .unwrap_or(false);
                result.push(AudioOutputDevice { id, name, is_default });
            }
            return Ok(result);
        }
    }

    pub fn list_input_devices(&self) -> Result<Vec<AudioInputDevice>, String> {
        // On macOS use CoreAudio directly for full device list.
        #[cfg(target_os = "macos")]
        {
            let devices = coreaudio_enum::list_input_devices();
            return Ok(devices
                .into_iter()
                .map(|d| AudioInputDevice { id: d.id, name: d.name, is_default: d.is_default })
                .collect());
        }

        // Fallback: cpal (Windows / Linux)
        #[cfg(not(target_os = "macos"))]
        {
            let host = Self::host();
            let devices = host
                .input_devices()
                .map_err(|e| format!("Failed to enumerate input devices: {}", e))?;
            let default_device = host.default_input_device();
            let mut result = Vec::new();
            for device in devices {
                let name = device
                    .name()
                    .map_err(|e| format!("Failed to get device name: {}", e))?;
                let id = format!("input_{}", name.replace(' ', "_").to_lowercase());
                let is_default = default_device
                    .as_ref()
                    .map(|d| d.name().unwrap_or_default() == name)
                    .unwrap_or(false);
                result.push(AudioInputDevice { id, name, is_default });
            }
            return Ok(result);
        }
    }

    pub async fn play_audio_to_devices(
        &self,
        audio_data: Vec<u8>,
        device_ids: Vec<String>,
    ) -> Result<(), String> {
        eprintln!("play_audio_to_devices called with {} bytes, {} device IDs", audio_data.len(), device_ids.len());
        eprintln!("Requested device IDs: {:?}", device_ids);
        
        // Decode audio file (assuming WAV format)
        eprintln!("Decoding audio data...");
        let (samples, sample_rate, channels) = self.decode_wav(&audio_data)?;
        eprintln!("Audio decoded: {} samples, {}Hz, {} channels", samples.len(), sample_rate, channels);

        // Find devices by ID
        eprintln!("Enumerating output devices...");
        let host = Self::host();
        let devices: Vec<Device> = host
            .output_devices()
            .map_err(|e| format!("Failed to enumerate devices: {}", e))?
            .filter_map(|device| {
                let name = device.name().ok()?;
                let id = format!("device_{}", name.replace(' ', "_").to_lowercase());
                eprintln!("Found device: {} (id: {})", name, id);
                if device_ids.contains(&id) {
                    eprintln!("  -> Matched! Will play to this device");
                    Some(device)
                } else {
                    None
                }
            })
            .collect();

        if devices.is_empty() {
            eprintln!("ERROR: No matching devices found");
            return Err("No matching devices found".to_string());
        }

        eprintln!("Playing to {} device(s)", devices.len());
        
        // Stop any existing playback first
        self.stop_all_playback().ok();
        
        // Reset stop flag for new playback
        self.stop_flag.store(false, Ordering::Relaxed);
        
        // Play to each device
        for (i, device) in devices.iter().enumerate() {
            let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
            eprintln!("Playing to device {}/{}: {}", i + 1, devices.len(), device_name);
            self.play_to_device(device, samples.clone(), sample_rate, channels, self.stop_flag.clone())
                .map_err(|e| format!("Failed to play to device {}: {}", device_name, e))?;
            eprintln!("Successfully started playback on device: {}", device_name);
        }

        eprintln!("play_audio_to_devices completed successfully");
        Ok(())
    }

    fn decode_wav(&self, data: &[u8]) -> Result<(Vec<f32>, u32, u16), String> {
        use symphonia::core::formats::FormatOptions;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;

        eprintln!("decode_wav: Creating MediaSourceStream from {} bytes", data.len());
        let mss = MediaSourceStream::new(
            Box::new(std::io::Cursor::new(data.to_vec())),
            Default::default(),
        );

        eprintln!("decode_wav: Probing audio format...");
        let mut format = symphonia::default::get_probe()
            .format(
                &Default::default(),
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| {
                eprintln!("decode_wav: Failed to probe audio: {}", e);
                format!("Failed to probe audio: {}", e)
            })?
            .format;
        
        eprintln!("decode_wav: Audio format probed successfully");

        eprintln!("decode_wav: Finding audio track...");
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| {
                eprintln!("decode_wav: No audio track found");
                "No audio track found".to_string()
            })?;

        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| {
                eprintln!("decode_wav: No sample rate found in track");
                "No sample rate found".to_string()
            })?;

        let channels = track
            .codec_params
            .channels
            .ok_or_else(|| {
                eprintln!("decode_wav: No channels found in track");
                "No channels found".to_string()
            })?
            .count() as u16;

        eprintln!("decode_wav: Track info - sample_rate: {}, channels: {}", sample_rate, channels);

        eprintln!("decode_wav: Creating decoder...");
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &Default::default())
            .map_err(|e| {
                eprintln!("decode_wav: Failed to create decoder: {}", e);
                format!("Failed to create decoder: {}", e)
            })?;
        
        eprintln!("decode_wav: Decoder created successfully");

        let mut samples = Vec::new();
        let mut packet_count = 0;
        eprintln!("decode_wav: Starting packet decoding loop...");
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(e) => {
                    eprintln!("decode_wav: End of stream or error: {:?}", e);
                    break;
                }
            };

            packet_count += 1;
            let decoded = decoder
                .decode(&packet)
                .map_err(|e| {
                    eprintln!("decode_wav: Decode error on packet {}: {}", packet_count, e);
                    format!("Decode error: {}", e)
                })?;

            // Convert to f32 samples by matching on the buffer type
            use symphonia::core::audio::{AudioBufferRef, Signal};
            use symphonia::core::conv::FromSample;

            let spec = *decoded.spec();
            let num_channels = spec.channels.count();
            let num_frames = decoded.frames();

            eprintln!("decode_wav: Packet {} - {} frames, {} channels", packet_count, num_frames, num_channels);

            // Interleave samples from all channels
            for frame_idx in 0..num_frames {
                for ch in 0..num_channels {
                    let sample_f32 = match &decoded {
                        AudioBufferRef::U8(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::U16(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::U24(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::U32(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::S8(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::S16(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::S24(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::S32(buf) => f32::from_sample(buf.chan(ch)[frame_idx]),
                        AudioBufferRef::F32(buf) => buf.chan(ch)[frame_idx],
                        AudioBufferRef::F64(buf) => buf.chan(ch)[frame_idx] as f32,
                    };
                    samples.push(sample_f32);
                }
            }
        }

        eprintln!("decode_wav: Decoded {} packets, total {} samples", packet_count, samples.len());
        eprintln!("decode_wav: Returning sample_rate={}, channels={}", sample_rate, channels);
        Ok((samples, sample_rate, channels))
    }

    fn play_to_device(
        &self,
        device: &Device,
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<(), String> {
        let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
        eprintln!("play_to_device: Starting playback to device: {}", device_name);
        eprintln!("play_to_device: Input - {} samples, {}Hz, {} channels", samples.len(), sample_rate, channels);
        
        let config = device
            .default_output_config()
            .map_err(|e| format!("Failed to get default config: {}", e))?;

        // Prepare samples for the device's format
        let device_sample_rate = config.sample_rate().0;
        let device_channels = config.channels();
        let device_sample_format = config.sample_format();
        
        eprintln!("play_to_device: Device config - {}Hz, {} channels, format: {:?}", 
                  device_sample_rate, device_channels, device_sample_format);

        // Resample if needed (simple linear interpolation for now)
        let resampled = if device_sample_rate != sample_rate {
            eprintln!("play_to_device: Resampling from {}Hz to {}Hz", sample_rate, device_sample_rate);
            let result = self.resample(&samples, sample_rate, device_sample_rate);
            eprintln!("play_to_device: Resampled {} samples to {} samples", samples.len(), result.len());
            result
        } else {
            eprintln!("play_to_device: No resampling needed");
            samples
        };

        // Interleave/convert channels if needed
        eprintln!("play_to_device: Interleaving channels from {} to {} channels", channels, device_channels);
        let interleaved = self.interleave_channels(&resampled, channels, device_channels);
        eprintln!("play_to_device: Interleaved to {} samples", interleaved.len());

        // Create shared buffer for playback
        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(interleaved));
        let position = Arc::new(AtomicUsize::new(0));
        let buffer_clone = buffer.clone();
        let position_clone = position.clone();

        let err_fn = |err| eprintln!("Playback error: {}", err);

        let stream_config = StreamConfig {
            channels: device_channels,
            sample_rate: cpal::SampleRate(device_sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let stop_flag_clone = stop_flag.clone();
        let stream = match config.sample_format() {
            SampleFormat::F32 => {
                let buffer = buffer_clone.clone();
                let pos = position_clone.clone();
                device
                    .build_output_stream(
                        &stream_config,
                        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                            // Check stop flag - if set, output silence
                            if stop_flag_clone.load(Ordering::Relaxed) {
                                for sample in data.iter_mut() {
                                    *sample = 0.0;
                                }
                                return;
                            }
                            
                            let mut idx = pos.load(Ordering::Relaxed);
                            let buf = buffer.lock().unwrap();
                            for sample in data.iter_mut() {
                                if idx < buf.len() {
                                    *sample = buf[idx];
                                    idx += 1;
                                } else {
                                    *sample = 0.0;
                                }
                            }
                            pos.store(idx, Ordering::Relaxed);
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed to build stream: {}", e))?
            }
            SampleFormat::I16 => {
                let buffer = buffer_clone.clone();
                let pos = position_clone.clone();
                device
                    .build_output_stream(
                        &stream_config,
                        move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                            // Check stop flag - if set, output silence
                            if stop_flag_clone.load(Ordering::Relaxed) {
                                for sample in data.iter_mut() {
                                    *sample = 0;
                                }
                                return;
                            }
                            
                            let mut idx = pos.load(Ordering::Relaxed);
                            let buf = buffer.lock().unwrap();
                            for sample in data.iter_mut() {
                                if idx < buf.len() {
                                    *sample = (buf[idx] * 32767.0) as i16;
                                    idx += 1;
                                } else {
                                    *sample = 0;
                                }
                            }
                            pos.store(idx, Ordering::Relaxed);
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed to build stream: {}", e))?
            }
            SampleFormat::U16 => {
                let buffer = buffer_clone.clone();
                let pos = position_clone.clone();
                device
                    .build_output_stream(
                        &stream_config,
                        move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                            // Check stop flag - if set, output silence
                            if stop_flag_clone.load(Ordering::Relaxed) {
                                for sample in data.iter_mut() {
                                    *sample = 32768;
                                }
                                return;
                            }
                            
                            let mut idx = pos.load(Ordering::Relaxed);
                            let buf = buffer.lock().unwrap();
                            for sample in data.iter_mut() {
                                if idx < buf.len() {
                                    *sample = ((buf[idx] + 1.0) * 32767.5) as u16;
                                    idx += 1;
                                } else {
                                    *sample = 32768;
                                }
                            }
                            pos.store(idx, Ordering::Relaxed);
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| format!("Failed to build stream: {}", e))?
            }
            _ => return Err("Unsupported sample format".to_string()),
        };

        eprintln!("play_to_device: Starting stream playback...");
        stream.play().map_err(|e| {
            eprintln!("play_to_device: Failed to play stream: {}", e);
            format!("Failed to play stream: {}", e)
        })?;
        
        eprintln!("play_to_device: Stream started successfully");

        eprintln!("play_to_device: Function completed successfully");
        Ok(())
    }

    fn resample(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        if from_rate == to_rate {
            return samples.to_vec();
        }

        let ratio = to_rate as f64 / from_rate as f64;
        let new_len = (samples.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = (i as f64 / ratio) as usize;
            if src_idx < samples.len() {
                resampled.push(samples[src_idx]);
            } else {
                resampled.push(0.0);
            }
        }

        resampled
    }

    fn interleave_channels(
        &self,
        samples: &[f32],
        src_channels: u16,
        dst_channels: u16,
    ) -> Vec<f32> {
        if src_channels == dst_channels {
            return samples.to_vec();
        }

        let mut interleaved = Vec::new();
        let samples_per_channel = samples.len() / src_channels as usize;

        for i in 0..samples_per_channel {
            for ch in 0..dst_channels {
                let src_ch = if ch < src_channels { ch } else { src_channels - 1 };
                let idx = (i * src_channels as usize) + src_ch as usize;
                if idx < samples.len() {
                    interleaved.push(samples[idx]);
                } else {
                    interleaved.push(0.0);
                }
            }
        }

        interleaved
    }
}

impl Default for AudioOutputState {
    fn default() -> Self {
        Self::new()
    }
}
