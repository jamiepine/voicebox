use std::path::Path;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::schemars::JsonSchema;
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use serde::Deserialize;
use serde_json::json;

use crate::client::VoiceboxClient;

#[derive(Clone)]
pub struct VoiceboxMcp {
    client: VoiceboxClient,
    tool_router: ToolRouter<Self>,
}

impl VoiceboxMcp {
    pub fn new(client: VoiceboxClient) -> Self {
        Self {
            client,
            tool_router: Self::tool_router(),
        }
    }

    fn pretty_json(value: &serde_json::Value) -> String {
        serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
    }

    async fn save_audio(
        &self,
        bytes: &[u8],
        output_path: Option<&str>,
        default_name: &str,
    ) -> Result<String, String> {
        let path = match output_path {
            Some(p) => p.to_string(),
            None => {
                let dir = std::env::temp_dir().join("voicebox-mcp");
                tokio::fs::create_dir_all(&dir)
                    .await
                    .map_err(|e| e.to_string())?;
                dir.join(default_name).to_string_lossy().to_string()
            }
        };
        tokio::fs::write(&path, bytes)
            .await
            .map_err(|e| e.to_string())?;
        Ok(path)
    }
}

// --- Profile tool params ---

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetProfileParams {
    #[schemars(description = "The voice profile ID")]
    pub profile_id: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateProfileParams {
    #[schemars(description = "Name for the voice profile")]
    pub name: String,
    #[schemars(description = "Language code (e.g. en, zh, ja, ko, de, fr)")]
    pub language: String,
    #[schemars(description = "Optional description of the voice")]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateProfileParams {
    #[schemars(description = "The voice profile ID to update")]
    pub profile_id: i64,
    #[schemars(description = "New name for the profile")]
    pub name: Option<String>,
    #[schemars(description = "New language code")]
    pub language: Option<String>,
    #[schemars(description = "New description")]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteProfileParams {
    #[schemars(description = "The voice profile ID to delete")]
    pub profile_id: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListProfileSamplesParams {
    #[schemars(description = "The voice profile ID")]
    pub profile_id: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddProfileSampleParams {
    #[schemars(description = "The voice profile ID to add a sample to")]
    pub profile_id: i64,
    #[schemars(description = "Path to the audio file on disk")]
    pub file_path: String,
    #[schemars(description = "Text that is spoken in the audio sample")]
    pub reference_text: String,
}

// --- Generation tool params ---

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateSpeechParams {
    #[schemars(description = "Text to synthesize into speech")]
    pub text: String,
    #[schemars(description = "Voice profile ID to use")]
    pub profile_id: i64,
    #[schemars(description = "Language code (e.g. en, zh, ja). Defaults to profile language")]
    pub language: Option<String>,
    #[schemars(description = "Random seed for reproducible generation")]
    pub seed: Option<i64>,
    #[schemars(description = "Model size: 1.7B or 0.6B (qwen only)")]
    pub model_size: Option<String>,
    #[schemars(
        description = "Instruction prompt for voice style (e.g. 'speak slowly', paralinguistic tags like [happy])"
    )]
    pub instruct: Option<String>,
    #[schemars(description = "TTS engine: qwen, luxtts, chatterbox, chatterbox_turbo")]
    pub engine: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetAudioParams {
    #[schemars(description = "The generation ID to download audio for")]
    pub generation_id: i64,
    #[schemars(description = "Path to save the audio file. Uses temp directory if not specified")]
    pub output_path: Option<String>,
}

// --- History tool params ---

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListHistoryParams {
    #[schemars(description = "Filter by voice profile ID")]
    pub profile_id: Option<i64>,
    #[schemars(description = "Search text in generation history")]
    pub search: Option<String>,
    #[schemars(description = "Maximum number of results (1-100, default 50)")]
    pub limit: Option<i64>,
    #[schemars(description = "Offset for pagination")]
    pub offset: Option<i64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetGenerationParams {
    #[schemars(description = "The generation ID")]
    pub generation_id: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteGenerationParams {
    #[schemars(description = "The generation ID to delete")]
    pub generation_id: i64,
}

// --- Transcription tool params ---

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TranscribeParams {
    #[schemars(description = "Path to the audio file to transcribe")]
    pub file_path: String,
    #[schemars(description = "Language hint: en or zh")]
    pub language: Option<String>,
}

// --- Story tool params ---

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateStoryParams {
    #[schemars(description = "Name for the story")]
    pub name: String,
    #[schemars(description = "Optional description")]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetStoryParams {
    #[schemars(description = "The story ID")]
    pub story_id: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateStoryParams {
    #[schemars(description = "The story ID to update")]
    pub story_id: i64,
    #[schemars(description = "New name")]
    pub name: Option<String>,
    #[schemars(description = "New description")]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteStoryParams {
    #[schemars(description = "The story ID to delete")]
    pub story_id: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddStoryItemParams {
    #[schemars(description = "The story ID")]
    pub story_id: i64,
    #[schemars(description = "Generation ID to add to the story")]
    pub generation_id: i64,
    #[schemars(description = "Start time in milliseconds on the timeline")]
    pub start_time_ms: Option<i64>,
    #[schemars(description = "Track number (0-based)")]
    pub track: Option<i64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RemoveStoryItemParams {
    #[schemars(description = "The story ID")]
    pub story_id: i64,
    #[schemars(description = "The story item ID to remove")]
    pub item_id: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MoveStoryItemParams {
    #[schemars(description = "The story ID")]
    pub story_id: i64,
    #[schemars(description = "The story item ID to move")]
    pub item_id: i64,
    #[schemars(description = "New start time in milliseconds")]
    pub start_time_ms: Option<i64>,
    #[schemars(description = "New track number")]
    pub track: Option<i64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportStoryParams {
    #[schemars(description = "The story ID to export")]
    pub story_id: i64,
    #[schemars(
        description = "Path to save the exported audio. Uses temp directory if not specified"
    )]
    pub output_path: Option<String>,
}

// --- Model tool params ---

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LoadModelParams {
    #[schemars(description = "Model size to load: 1.7B or 0.6B (default 1.7B)")]
    pub model_size: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UnloadModelParams {
    #[schemars(
        description = "Specific model name to unload (e.g. qwen-tts-1.7B, chatterbox-tts). Unloads default Qwen if not specified"
    )]
    pub model_name: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DownloadModelParams {
    #[schemars(
        description = "Model name to download (e.g. qwen-tts-1.7B, qwen-tts-0.6B, luxtts, chatterbox-tts, chatterbox-turbo)"
    )]
    pub model_name: String,
}

// --- Tool implementations ---

#[tool_router]
impl VoiceboxMcp {
    // --- Profiles ---

    #[tool(description = "List all voice profiles available for speech generation")]
    async fn list_profiles(&self) -> Result<String, String> {
        let v = self
            .client
            .list_profiles()
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Get details of a specific voice profile")]
    async fn get_profile(
        &self,
        Parameters(p): Parameters<GetProfileParams>,
    ) -> Result<String, String> {
        let v = self
            .client
            .get_profile(p.profile_id)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Create a new voice profile for cloning a voice")]
    async fn create_profile(
        &self,
        Parameters(p): Parameters<CreateProfileParams>,
    ) -> Result<String, String> {
        let body = json!({
            "name": p.name,
            "language": p.language,
            "description": p.description,
        });
        let v = self
            .client
            .create_profile(&body)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Update an existing voice profile")]
    async fn update_profile(
        &self,
        Parameters(p): Parameters<UpdateProfileParams>,
    ) -> Result<String, String> {
        let mut body = json!({});
        if let Some(name) = &p.name {
            body["name"] = json!(name);
        }
        if let Some(lang) = &p.language {
            body["language"] = json!(lang);
        }
        if let Some(desc) = &p.description {
            body["description"] = json!(desc);
        }
        let v = self
            .client
            .update_profile(p.profile_id, &body)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Delete a voice profile and all its samples")]
    async fn delete_profile(
        &self,
        Parameters(p): Parameters<DeleteProfileParams>,
    ) -> Result<String, String> {
        let v = self
            .client
            .delete_profile(p.profile_id)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "List voice samples attached to a profile")]
    async fn list_profile_samples(
        &self,
        Parameters(p): Parameters<ListProfileSamplesParams>,
    ) -> Result<String, String> {
        let v = self
            .client
            .list_profile_samples(p.profile_id)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(
        description = "Add a voice sample to a profile for voice cloning. Requires an audio file and the text spoken in it"
    )]
    async fn add_profile_sample(
        &self,
        Parameters(p): Parameters<AddProfileSampleParams>,
    ) -> Result<String, String> {
        let v = self
            .client
            .add_profile_sample(p.profile_id, &p.file_path, &p.reference_text)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    // --- Generation ---

    #[tool(
        description = "Generate speech from text using a voice profile. Returns generation metadata including the audio file path"
    )]
    async fn generate_speech(
        &self,
        Parameters(p): Parameters<GenerateSpeechParams>,
    ) -> Result<String, String> {
        let mut body = json!({
            "profile_id": p.profile_id,
            "text": p.text,
        });
        if let Some(lang) = &p.language {
            body["language"] = json!(lang);
        }
        if let Some(seed) = p.seed {
            body["seed"] = json!(seed);
        }
        if let Some(size) = &p.model_size {
            body["model_size"] = json!(size);
        }
        if let Some(instruct) = &p.instruct {
            body["instruct"] = json!(instruct);
        }
        if let Some(engine) = &p.engine {
            body["engine"] = json!(engine);
        }
        let v = self
            .client
            .generate(&body)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Download the audio file for a generation and save it to disk")]
    async fn get_audio(&self, Parameters(p): Parameters<GetAudioParams>) -> Result<String, String> {
        let bytes = self
            .client
            .get_audio(p.generation_id)
            .await
            .map_err(|e| e.to_string())?;
        let default_name = format!("generation_{}.wav", p.generation_id);
        let saved_path = self
            .save_audio(&bytes, p.output_path.as_deref(), &default_name)
            .await?;
        Ok(Self::pretty_json(&json!({
            "generation_id": p.generation_id,
            "output_path": saved_path,
            "size_bytes": bytes.len(),
        })))
    }

    // --- History ---

    #[tool(description = "List generation history with optional filters")]
    async fn list_history(
        &self,
        Parameters(p): Parameters<ListHistoryParams>,
    ) -> Result<String, String> {
        let v = self
            .client
            .list_history(p.profile_id, p.search.as_deref(), p.limit, p.offset)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Get details of a specific generation from history")]
    async fn get_generation(
        &self,
        Parameters(p): Parameters<GetGenerationParams>,
    ) -> Result<String, String> {
        let v = self
            .client
            .get_generation(p.generation_id)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Delete a generation and its audio file")]
    async fn delete_generation(
        &self,
        Parameters(p): Parameters<DeleteGenerationParams>,
    ) -> Result<String, String> {
        let v = self
            .client
            .delete_generation(p.generation_id)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(
        description = "Get generation statistics - total count, duration, per-profile breakdown"
    )]
    async fn get_history_stats(&self) -> Result<String, String> {
        let v = self
            .client
            .get_history_stats()
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    // --- Transcription ---

    #[tool(
        description = "Transcribe an audio file to text using Whisper. Returns transcribed text and duration"
    )]
    async fn transcribe(
        &self,
        Parameters(p): Parameters<TranscribeParams>,
    ) -> Result<String, String> {
        if !Path::new(&p.file_path).exists() {
            return Err(format!("File not found: {}", p.file_path));
        }
        let v = self
            .client
            .transcribe(&p.file_path, p.language.as_deref())
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    // --- Stories ---

    #[tool(description = "List all multi-voice story projects")]
    async fn list_stories(&self) -> Result<String, String> {
        let v = self
            .client
            .list_stories()
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Create a new multi-voice story project")]
    async fn create_story(
        &self,
        Parameters(p): Parameters<CreateStoryParams>,
    ) -> Result<String, String> {
        let body = json!({
            "name": p.name,
            "description": p.description,
        });
        let v = self
            .client
            .create_story(&body)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Get a story with all its items and timeline details")]
    async fn get_story(&self, Parameters(p): Parameters<GetStoryParams>) -> Result<String, String> {
        let v = self
            .client
            .get_story(p.story_id)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Update a story's name or description")]
    async fn update_story(
        &self,
        Parameters(p): Parameters<UpdateStoryParams>,
    ) -> Result<String, String> {
        let mut body = json!({});
        if let Some(name) = &p.name {
            body["name"] = json!(name);
        }
        if let Some(desc) = &p.description {
            body["description"] = json!(desc);
        }
        let v = self
            .client
            .update_story(p.story_id, &body)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Delete a story and all its items")]
    async fn delete_story(
        &self,
        Parameters(p): Parameters<DeleteStoryParams>,
    ) -> Result<String, String> {
        let v = self
            .client
            .delete_story(p.story_id)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Add a generation to a story's timeline")]
    async fn add_story_item(
        &self,
        Parameters(p): Parameters<AddStoryItemParams>,
    ) -> Result<String, String> {
        let mut body = json!({ "generation_id": p.generation_id });
        if let Some(t) = p.start_time_ms {
            body["start_time_ms"] = json!(t);
        }
        if let Some(track) = p.track {
            body["track"] = json!(track);
        }
        let v = self
            .client
            .add_story_item(p.story_id, &body)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Remove an item from a story's timeline")]
    async fn remove_story_item(
        &self,
        Parameters(p): Parameters<RemoveStoryItemParams>,
    ) -> Result<String, String> {
        let v = self
            .client
            .remove_story_item(p.story_id, p.item_id)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Move a story item to a new position on the timeline")]
    async fn move_story_item(
        &self,
        Parameters(p): Parameters<MoveStoryItemParams>,
    ) -> Result<String, String> {
        let mut body = json!({});
        if let Some(t) = p.start_time_ms {
            body["start_time_ms"] = json!(t);
        }
        if let Some(track) = p.track {
            body["track"] = json!(track);
        }
        let v = self
            .client
            .move_story_item(p.story_id, p.item_id, &body)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Export a story as a single mixed audio file")]
    async fn export_story(
        &self,
        Parameters(p): Parameters<ExportStoryParams>,
    ) -> Result<String, String> {
        let bytes = self
            .client
            .export_story_audio(p.story_id)
            .await
            .map_err(|e| e.to_string())?;
        let default_name = format!("story_{}.wav", p.story_id);
        let saved_path = self
            .save_audio(&bytes, p.output_path.as_deref(), &default_name)
            .await?;
        Ok(Self::pretty_json(&json!({
            "story_id": p.story_id,
            "output_path": saved_path,
            "size_bytes": bytes.len(),
        })))
    }

    // --- Models ---

    #[tool(
        description = "Get status of all TTS and transcription models - downloaded, loaded, size"
    )]
    async fn get_model_status(&self) -> Result<String, String> {
        let v = self
            .client
            .get_model_status()
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Load a TTS model into memory for generation")]
    async fn load_model(
        &self,
        Parameters(p): Parameters<LoadModelParams>,
    ) -> Result<String, String> {
        let v = self
            .client
            .load_model(p.model_size.as_deref())
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Unload a TTS model from memory to free resources")]
    async fn unload_model(
        &self,
        Parameters(p): Parameters<UnloadModelParams>,
    ) -> Result<String, String> {
        let v = self
            .client
            .unload_model(p.model_name.as_deref())
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Trigger a background download of a TTS model from HuggingFace")]
    async fn download_model(
        &self,
        Parameters(p): Parameters<DownloadModelParams>,
    ) -> Result<String, String> {
        let body = json!({"model_name": p.model_name});
        let v = self
            .client
            .download_model(&body)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    // --- System ---

    #[tool(description = "Check Voicebox server health - GPU status, backend type, model state")]
    async fn health_check(&self) -> Result<String, String> {
        let v = self.client.health().await.map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }

    #[tool(description = "Clear the voice prompt cache to free disk space")]
    async fn clear_cache(&self) -> Result<String, String> {
        let v = self.client.clear_cache().await.map_err(|e| e.to_string())?;
        Ok(Self::pretty_json(&v))
    }
}

#[tool_handler]
impl ServerHandler for VoiceboxMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Voicebox MCP server - local voice synthesis studio. \
                 Generate speech from text using cloned voices, transcribe audio, \
                 manage voice profiles and multi-voice story projects.",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn setup() -> (MockServer, VoiceboxMcp) {
        let mock = MockServer::start().await;
        let client = VoiceboxClient::new(&mock.uri());
        let server = VoiceboxMcp::new(client);
        (mock, server)
    }

    #[tokio::test]
    async fn list_profiles_returns_json() {
        let (mock, server) = setup().await;
        Mock::given(method("GET"))
            .and(path("/profiles"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!([{"id": 1, "name": "Alice"}])),
            )
            .mount(&mock)
            .await;

        let result = server.list_profiles().await.unwrap();
        assert!(result.contains("Alice"));
    }

    #[tokio::test]
    async fn generate_speech_returns_metadata() {
        let (mock, server) = setup().await;
        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": 1,
                "profile_id": 1,
                "text": "Hello world",
                "audio_path": "/data/gen/1.wav",
                "duration": 2.3
            })))
            .mount(&mock)
            .await;

        let params = GenerateSpeechParams {
            text: "Hello world".to_string(),
            profile_id: 1,
            language: Some("en".to_string()),
            seed: None,
            model_size: None,
            instruct: None,
            engine: None,
        };
        let result = server.generate_speech(Parameters(params)).await.unwrap();
        assert!(result.contains("Hello world"));
        assert!(result.contains("/data/gen/1.wav"));
    }

    #[tokio::test]
    async fn health_check_returns_status() {
        let (mock, server) = setup().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "ok",
                "model_loaded": true,
                "gpu_available": true
            })))
            .mount(&mock)
            .await;

        let result = server.health_check().await.unwrap();
        assert!(result.contains("ok"));
    }

    #[tokio::test]
    async fn api_error_propagates() {
        let (mock, server) = setup().await;
        Mock::given(method("GET"))
            .and(path("/profiles/999"))
            .respond_with(ResponseTemplate::new(404).set_body_string("Profile not found"))
            .mount(&mock)
            .await;

        let params = GetProfileParams { profile_id: 999 };
        let result = server.get_profile(Parameters(params)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("404"));
    }

    #[tokio::test]
    async fn create_story_returns_json() {
        let (mock, server) = setup().await;
        Mock::given(method("POST"))
            .and(path("/stories"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"id": 1, "name": "My Podcast"})),
            )
            .mount(&mock)
            .await;

        let params = CreateStoryParams {
            name: "My Podcast".to_string(),
            description: None,
        };
        let result = server.create_story(Parameters(params)).await.unwrap();
        assert!(result.contains("My Podcast"));
    }

    #[tokio::test]
    async fn get_audio_saves_file() {
        let (mock, server) = setup().await;
        let audio_bytes = vec![0u8; 44];
        Mock::given(method("GET"))
            .and(path("/audio/42"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(audio_bytes))
            .mount(&mock)
            .await;

        let tmp = std::env::temp_dir().join("voicebox-mcp-test-audio.wav");
        let params = GetAudioParams {
            generation_id: 42,
            output_path: Some(tmp.to_string_lossy().to_string()),
        };
        let result = server.get_audio(Parameters(params)).await.unwrap();
        assert!(result.contains("voicebox-mcp-test-audio.wav"));
        assert!(result.contains("44")); // size_bytes

        let _ = tokio::fs::remove_file(&tmp).await;
    }

    #[tokio::test]
    async fn model_status_returns_json() {
        let (mock, server) = setup().await;
        Mock::given(method("GET"))
            .and(path("/models/status"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [
                    {"model_name": "qwen-tts-1.7B", "downloaded": true, "loaded": false}
                ]
            })))
            .mount(&mock)
            .await;

        let result = server.get_model_status().await.unwrap();
        assert!(result.contains("qwen-tts-1.7B"));
    }

    #[tokio::test]
    async fn list_history_with_filters() {
        let (mock, server) = setup().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"items": [], "total": 0})),
            )
            .mount(&mock)
            .await;

        let params = ListHistoryParams {
            profile_id: None,
            search: None,
            limit: Some(10),
            offset: None,
        };
        let result = server.list_history(Parameters(params)).await.unwrap();
        assert!(result.contains("total"));
    }

    // --- Profile workflows ---

    #[tokio::test]
    async fn profile_create_then_get() {
        let (mock, server) = setup().await;

        Mock::given(method("POST"))
            .and(path("/profiles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": 5,
                "name": "Test Voice",
                "language": "en",
                "description": "A test voice",
                "created_at": "2025-01-01T00:00:00"
            })))
            .mount(&mock)
            .await;

        let result = server
            .create_profile(Parameters(CreateProfileParams {
                name: "Test Voice".to_string(),
                language: "en".to_string(),
                description: Some("A test voice".to_string()),
            }))
            .await
            .unwrap();
        assert!(result.contains("Test Voice"));
        assert!(result.contains("\"id\": 5"));

        Mock::given(method("GET"))
            .and(path("/profiles/5"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": 5,
                "name": "Test Voice",
                "language": "en",
                "description": "A test voice"
            })))
            .mount(&mock)
            .await;

        let result = server
            .get_profile(Parameters(GetProfileParams { profile_id: 5 }))
            .await
            .unwrap();
        assert!(result.contains("Test Voice"));
    }

    #[tokio::test]
    async fn profile_update_partial_fields() {
        let (mock, server) = setup().await;

        Mock::given(method("PUT"))
            .and(path("/profiles/1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": 1,
                "name": "Updated Name",
                "language": "en"
            })))
            .mount(&mock)
            .await;

        let result = server
            .update_profile(Parameters(UpdateProfileParams {
                profile_id: 1,
                name: Some("Updated Name".to_string()),
                language: None,
                description: None,
            }))
            .await
            .unwrap();
        assert!(result.contains("Updated Name"));
    }

    #[tokio::test]
    async fn profile_delete_returns_message() {
        let (mock, server) = setup().await;

        Mock::given(method("DELETE"))
            .and(path("/profiles/1"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"message": "Profile deleted"})),
            )
            .mount(&mock)
            .await;

        let result = server
            .delete_profile(Parameters(DeleteProfileParams { profile_id: 1 }))
            .await
            .unwrap();
        assert!(result.contains("deleted"));
    }

    // --- Generation workflows ---

    #[tokio::test]
    async fn generate_with_all_options() {
        let (mock, server) = setup().await;

        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": 42,
                "profile_id": 1,
                "text": "Hello world",
                "language": "en",
                "audio_path": "/data/generations/42.wav",
                "duration": 1.8,
                "seed": 12345,
                "instruct": "speak slowly"
            })))
            .mount(&mock)
            .await;

        let result = server
            .generate_speech(Parameters(GenerateSpeechParams {
                text: "Hello world".to_string(),
                profile_id: 1,
                language: Some("en".to_string()),
                seed: Some(12345),
                model_size: Some("1.7B".to_string()),
                instruct: Some("speak slowly".to_string()),
                engine: Some("qwen".to_string()),
            }))
            .await
            .unwrap();

        assert!(result.contains("Hello world"));
        assert!(result.contains("42.wav"));
        assert!(result.contains("12345"));
        assert!(result.contains("speak slowly"));
    }

    #[tokio::test]
    async fn generate_returns_202_when_model_downloading() {
        let (mock, server) = setup().await;

        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(202).set_body_json(json!({
                "message": "Model not cached, download started"
            })))
            .mount(&mock)
            .await;

        let result = server
            .generate_speech(Parameters(GenerateSpeechParams {
                text: "test".to_string(),
                profile_id: 1,
                language: None,
                seed: None,
                model_size: None,
                instruct: None,
                engine: None,
            }))
            .await
            .unwrap();
        assert!(result.contains("download started"));
    }

    #[tokio::test]
    async fn get_audio_to_temp_dir() {
        let (mock, server) = setup().await;

        let fake_wav = vec![0u8; 256];
        Mock::given(method("GET"))
            .and(path("/audio/99"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(fake_wav))
            .mount(&mock)
            .await;

        let result = server
            .get_audio(Parameters(GetAudioParams {
                generation_id: 99,
                output_path: None,
            }))
            .await
            .unwrap();

        assert!(result.contains("generation_99.wav"));
        assert!(result.contains("256"));

        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let path = parsed["output_path"].as_str().unwrap();
        let _ = std::fs::remove_file(path);
    }

    // --- History workflows ---

    #[tokio::test]
    async fn history_with_search_and_pagination() {
        let (mock, server) = setup().await;

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "items": [
                    {"id": 1, "text": "hello there", "profile_name": "Alice"},
                    {"id": 2, "text": "hello world", "profile_name": "Bob"},
                ],
                "total": 2
            })))
            .mount(&mock)
            .await;

        let result = server
            .list_history(Parameters(ListHistoryParams {
                profile_id: None,
                search: Some("hello".to_string()),
                limit: Some(10),
                offset: Some(0),
            }))
            .await
            .unwrap();

        assert!(result.contains("hello there"));
        assert!(result.contains("hello world"));
        assert!(result.contains("\"total\": 2"));
    }

    #[tokio::test]
    async fn history_stats_returns_breakdown() {
        let (mock, server) = setup().await;

        Mock::given(method("GET"))
            .and(path("/history/stats"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "total_generations": 150,
                "total_duration_seconds": 3600.5,
                "by_profile": [
                    {"profile_id": 1, "profile_name": "Alice", "count": 100},
                    {"profile_id": 2, "profile_name": "Bob", "count": 50}
                ]
            })))
            .mount(&mock)
            .await;

        let result = server.get_history_stats().await.unwrap();
        assert!(result.contains("150"));
        assert!(result.contains("Alice"));
        assert!(result.contains("Bob"));
    }

    // --- Transcription ---

    #[tokio::test]
    async fn transcribe_missing_file_returns_error() {
        let (_mock, server) = setup().await;

        let result = server
            .transcribe(Parameters(TranscribeParams {
                file_path: "/nonexistent/audio.wav".to_string(),
                language: None,
            }))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("File not found"));
    }

    // --- Story workflows ---

    #[tokio::test]
    async fn story_full_workflow() {
        let (mock, server) = setup().await;

        Mock::given(method("POST"))
            .and(path("/stories"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": 1,
                "name": "My Podcast",
                "description": "Episode 1",
                "item_count": 0
            })))
            .mount(&mock)
            .await;

        let result = server
            .create_story(Parameters(CreateStoryParams {
                name: "My Podcast".to_string(),
                description: Some("Episode 1".to_string()),
            }))
            .await
            .unwrap();
        assert!(result.contains("My Podcast"));

        Mock::given(method("POST"))
            .and(path("/stories/1/items"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": 10,
                "story_id": 1,
                "generation_id": 42,
                "start_time_ms": 0,
                "track": 0
            })))
            .mount(&mock)
            .await;

        let result = server
            .add_story_item(Parameters(AddStoryItemParams {
                story_id: 1,
                generation_id: 42,
                start_time_ms: Some(0),
                track: Some(0),
            }))
            .await
            .unwrap();
        assert!(result.contains("\"generation_id\": 42"));

        Mock::given(method("PUT"))
            .and(path("/stories/1/items/10/move"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": 10,
                "start_time_ms": 5000,
                "track": 1
            })))
            .mount(&mock)
            .await;

        let result = server
            .move_story_item(Parameters(MoveStoryItemParams {
                story_id: 1,
                item_id: 10,
                start_time_ms: Some(5000),
                track: Some(1),
            }))
            .await
            .unwrap();
        assert!(result.contains("5000"));

        Mock::given(method("GET"))
            .and(path("/stories/1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": 1,
                "name": "My Podcast",
                "items": [{"id": 10, "generation_id": 42, "track": 1}]
            })))
            .mount(&mock)
            .await;

        let result = server
            .get_story(Parameters(GetStoryParams { story_id: 1 }))
            .await
            .unwrap();
        assert!(result.contains("\"items\""));

        Mock::given(method("DELETE"))
            .and(path("/stories/1/items/10"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"message": "Item removed"})),
            )
            .mount(&mock)
            .await;

        let result = server
            .remove_story_item(Parameters(RemoveStoryItemParams {
                story_id: 1,
                item_id: 10,
            }))
            .await
            .unwrap();
        assert!(result.contains("removed"));
    }

    #[tokio::test]
    async fn export_story_saves_audio() {
        let (mock, server) = setup().await;

        let fake_wav = vec![0u8; 1024];
        Mock::given(method("GET"))
            .and(path("/stories/1/export-audio"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(fake_wav))
            .mount(&mock)
            .await;

        let tmp = std::env::temp_dir().join("voicebox-mcp-test-story-export.wav");
        let result = server
            .export_story(Parameters(ExportStoryParams {
                story_id: 1,
                output_path: Some(tmp.to_string_lossy().to_string()),
            }))
            .await
            .unwrap();

        assert!(result.contains("1024"));
        assert!(tmp.exists());

        let _ = std::fs::remove_file(&tmp);
    }

    // --- Model workflows ---

    #[tokio::test]
    async fn model_load_unload_cycle() {
        let (mock, server) = setup().await;

        Mock::given(method("POST"))
            .and(path("/models/load"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"message": "Model loaded"})),
            )
            .mount(&mock)
            .await;

        let result = server
            .load_model(Parameters(LoadModelParams {
                model_size: Some("1.7B".to_string()),
            }))
            .await
            .unwrap();
        assert!(result.contains("loaded"));

        Mock::given(method("POST"))
            .and(path("/models/qwen-tts-1.7B/unload"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"message": "Model unloaded"})),
            )
            .mount(&mock)
            .await;

        let result = server
            .unload_model(Parameters(UnloadModelParams {
                model_name: Some("qwen-tts-1.7B".to_string()),
            }))
            .await
            .unwrap();
        assert!(result.contains("unloaded"));
    }

    #[tokio::test]
    async fn model_download_triggers_background() {
        let (mock, server) = setup().await;

        Mock::given(method("POST"))
            .and(path("/models/download"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"message": "Download started for chatterbox-tts"})),
            )
            .mount(&mock)
            .await;

        let result = server
            .download_model(Parameters(DownloadModelParams {
                model_name: "chatterbox-tts".to_string(),
            }))
            .await
            .unwrap();
        assert!(result.contains("chatterbox-tts"));
    }

    // --- System ---

    #[tokio::test]
    async fn clear_cache_returns_count() {
        let (mock, server) = setup().await;

        Mock::given(method("POST"))
            .and(path("/cache/clear"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"message": "Cache cleared", "files_deleted": 42})),
            )
            .mount(&mock)
            .await;

        let result = server.clear_cache().await.unwrap();
        assert!(result.contains("42"));
    }

    // --- Error handling ---

    #[tokio::test]
    async fn server_500_propagates_as_error() {
        let (mock, server) = setup().await;

        Mock::given(method("GET"))
            .and(path("/profiles"))
            .respond_with(
                ResponseTemplate::new(500).set_body_string("Internal server error: model crashed"),
            )
            .mount(&mock)
            .await;

        let result = server.list_profiles().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("500"));
    }

    #[tokio::test]
    async fn connection_refused_propagates_as_error() {
        let client = VoiceboxClient::new("http://127.0.0.1:1");
        let server = VoiceboxMcp::new(client);

        let result = server.list_profiles().await;
        assert!(result.is_err());
    }
}
