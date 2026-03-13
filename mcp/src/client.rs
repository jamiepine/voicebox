use std::path::Path;

use reqwest::multipart;
use serde_json::Value;

use crate::error::VoiceboxError;

#[derive(Clone)]
pub struct VoiceboxClient {
    http: reqwest::Client,
    base_url: String,
}

impl VoiceboxClient {
    pub fn new(base_url: &str) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    async fn get(&self, path: &str) -> Result<Value, VoiceboxError> {
        let resp = self.http.get(self.url(path)).send().await?;
        self.handle_json(resp).await
    }

    async fn post_json(&self, path: &str, body: &Value) -> Result<Value, VoiceboxError> {
        let resp = self.http.post(self.url(path)).json(body).send().await?;
        self.handle_json(resp).await
    }

    async fn put_json(&self, path: &str, body: &Value) -> Result<Value, VoiceboxError> {
        let resp = self.http.put(self.url(path)).json(body).send().await?;
        self.handle_json(resp).await
    }

    async fn delete(&self, path: &str) -> Result<Value, VoiceboxError> {
        let resp = self.http.delete(self.url(path)).send().await?;
        self.handle_json(resp).await
    }

    async fn post_empty(&self, path: &str) -> Result<Value, VoiceboxError> {
        let resp = self.http.post(self.url(path)).send().await?;
        self.handle_json(resp).await
    }

    async fn handle_json(&self, resp: reqwest::Response) -> Result<Value, VoiceboxError> {
        let status = resp.status().as_u16();
        if status >= 400 {
            let raw = resp.text().await.unwrap_or_default();
            let body = serde_json::from_str::<Value>(&raw)
                .ok()
                .and_then(|v| v["detail"].as_str().map(String::from))
                .unwrap_or(raw);
            return Err(VoiceboxError::Api { status, body });
        }
        Ok(resp.json().await?)
    }

    async fn download_bytes(&self, path: &str) -> Result<Vec<u8>, VoiceboxError> {
        let resp = self.http.get(self.url(path)).send().await?;
        let status = resp.status().as_u16();
        if status >= 400 {
            let body = resp.text().await.unwrap_or_default();
            return Err(VoiceboxError::Api { status, body });
        }
        Ok(resp.bytes().await?.to_vec())
    }

    // --- Profiles ---

    pub async fn list_profiles(&self) -> Result<Value, VoiceboxError> {
        self.get("/profiles").await
    }

    pub async fn get_profile(&self, profile_id: &str) -> Result<Value, VoiceboxError> {
        self.get(&format!("/profiles/{profile_id}")).await
    }

    pub async fn create_profile(&self, body: &Value) -> Result<Value, VoiceboxError> {
        self.post_json("/profiles", body).await
    }

    pub async fn update_profile(
        &self,
        profile_id: &str,
        body: &Value,
    ) -> Result<Value, VoiceboxError> {
        self.put_json(&format!("/profiles/{profile_id}"), body)
            .await
    }

    pub async fn delete_profile(&self, profile_id: &str) -> Result<Value, VoiceboxError> {
        self.delete(&format!("/profiles/{profile_id}")).await
    }

    pub async fn list_profile_samples(&self, profile_id: &str) -> Result<Value, VoiceboxError> {
        self.get(&format!("/profiles/{profile_id}/samples")).await
    }

    pub async fn add_profile_sample(
        &self,
        profile_id: &str,
        file_path: &str,
        reference_text: &str,
    ) -> Result<Value, VoiceboxError> {
        let file_bytes = tokio::fs::read(file_path).await?;
        let file_name = Path::new(file_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "audio.wav".to_string());

        let mime = mime_from_path(Path::new(file_path));
        let file_part = multipart::Part::bytes(file_bytes)
            .file_name(file_name)
            .mime_str(mime)
            .map_err(|e| VoiceboxError::Other(e.to_string()))?;

        let form = multipart::Form::new()
            .part("file", file_part)
            .text("reference_text", reference_text.to_string());

        let resp = self
            .http
            .post(self.url(&format!("/profiles/{profile_id}/samples")))
            .multipart(form)
            .send()
            .await?;
        self.handle_json(resp).await
    }

    // --- Generation ---

    pub async fn generate(&self, body: &Value) -> Result<Value, VoiceboxError> {
        self.post_json("/generate", body).await
    }

    pub async fn get_audio(&self, generation_id: &str) -> Result<Vec<u8>, VoiceboxError> {
        self.download_bytes(&format!("/audio/{generation_id}"))
            .await
    }

    // --- History ---

    pub async fn list_history(
        &self,
        profile_id: Option<&str>,
        search: Option<&str>,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Value, VoiceboxError> {
        let mut query = Vec::new();
        if let Some(pid) = profile_id {
            query.push(format!("profile_id={pid}"));
        }
        if let Some(s) = search {
            query.push(format!("search={}", urlencoding::encode(s)));
        }
        if let Some(l) = limit {
            query.push(format!("limit={l}"));
        }
        if let Some(o) = offset {
            query.push(format!("offset={o}"));
        }
        let qs = if query.is_empty() {
            String::new()
        } else {
            format!("?{}", query.join("&"))
        };
        self.get(&format!("/history{qs}")).await
    }

    pub async fn get_generation(&self, generation_id: &str) -> Result<Value, VoiceboxError> {
        self.get(&format!("/history/{generation_id}")).await
    }

    pub async fn delete_generation(&self, generation_id: &str) -> Result<Value, VoiceboxError> {
        self.delete(&format!("/history/{generation_id}")).await
    }

    pub async fn get_history_stats(&self) -> Result<Value, VoiceboxError> {
        self.get("/history/stats").await
    }

    // --- Transcription ---

    pub async fn transcribe(
        &self,
        file_path: &str,
        language: Option<&str>,
    ) -> Result<Value, VoiceboxError> {
        let file_bytes = tokio::fs::read(file_path).await?;
        let file_name = Path::new(file_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "audio.wav".to_string());

        let mime = mime_from_path(Path::new(file_path));
        let file_part = multipart::Part::bytes(file_bytes)
            .file_name(file_name)
            .mime_str(mime)
            .map_err(|e| VoiceboxError::Other(e.to_string()))?;

        let mut form = multipart::Form::new().part("file", file_part);
        if let Some(lang) = language {
            form = form.text("language", lang.to_string());
        }

        let resp = self
            .http
            .post(self.url("/transcribe"))
            .multipart(form)
            .send()
            .await?;
        self.handle_json(resp).await
    }

    // --- Stories ---

    pub async fn list_stories(&self) -> Result<Value, VoiceboxError> {
        self.get("/stories").await
    }

    pub async fn create_story(&self, body: &Value) -> Result<Value, VoiceboxError> {
        self.post_json("/stories", body).await
    }

    pub async fn get_story(&self, story_id: &str) -> Result<Value, VoiceboxError> {
        self.get(&format!("/stories/{story_id}")).await
    }

    pub async fn update_story(&self, story_id: &str, body: &Value) -> Result<Value, VoiceboxError> {
        self.put_json(&format!("/stories/{story_id}"), body).await
    }

    pub async fn delete_story(&self, story_id: &str) -> Result<Value, VoiceboxError> {
        self.delete(&format!("/stories/{story_id}")).await
    }

    pub async fn add_story_item(
        &self,
        story_id: &str,
        body: &Value,
    ) -> Result<Value, VoiceboxError> {
        self.post_json(&format!("/stories/{story_id}/items"), body)
            .await
    }

    pub async fn remove_story_item(
        &self,
        story_id: &str,
        item_id: &str,
    ) -> Result<Value, VoiceboxError> {
        self.delete(&format!("/stories/{story_id}/items/{item_id}"))
            .await
    }

    pub async fn move_story_item(
        &self,
        story_id: &str,
        item_id: &str,
        body: &Value,
    ) -> Result<Value, VoiceboxError> {
        self.put_json(&format!("/stories/{story_id}/items/{item_id}/move"), body)
            .await
    }

    pub async fn export_story_audio(&self, story_id: &str) -> Result<Vec<u8>, VoiceboxError> {
        self.download_bytes(&format!("/stories/{story_id}/export-audio"))
            .await
    }

    // --- Models ---

    pub async fn get_model_status(&self) -> Result<Value, VoiceboxError> {
        self.get("/models/status").await
    }

    pub async fn load_model(&self, model_size: Option<&str>) -> Result<Value, VoiceboxError> {
        let qs = model_size
            .map(|s| format!("?model_size={s}"))
            .unwrap_or_default();
        self.post_empty(&format!("/models/load{qs}")).await
    }

    pub async fn unload_model(&self, model_name: Option<&str>) -> Result<Value, VoiceboxError> {
        match model_name {
            Some(name) => self.post_empty(&format!("/models/{name}/unload")).await,
            None => self.post_empty("/models/unload").await,
        }
    }

    pub async fn download_model(&self, body: &Value) -> Result<Value, VoiceboxError> {
        self.post_json("/models/download", body).await
    }

    // --- System ---

    pub async fn health(&self) -> Result<Value, VoiceboxError> {
        self.get("/health").await
    }

    pub async fn clear_cache(&self) -> Result<Value, VoiceboxError> {
        self.post_empty("/cache/clear").await
    }
}

fn mime_from_path(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("flac") => "audio/flac",
        Some("ogg") => "audio/ogg",
        Some("m4a") => "audio/mp4",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn list_profiles_returns_array() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/profiles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"id": 1, "name": "Alice", "language": "en"}
            ])))
            .mount(&mock)
            .await;

        let client = VoiceboxClient::new(&mock.uri());
        let result = client.list_profiles().await.unwrap();
        assert!(result.is_array());
        assert_eq!(result[0]["name"], "Alice");
    }

    #[tokio::test]
    async fn get_profile_returns_object() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/profiles/42"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id": 42, "name": "Bob"})),
            )
            .mount(&mock)
            .await;

        let client = VoiceboxClient::new(&mock.uri());
        let result = client.get_profile("42").await.unwrap();
        assert_eq!(result["id"], 42);
    }

    #[tokio::test]
    async fn api_error_returns_error() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/profiles/999"))
            .respond_with(ResponseTemplate::new(404).set_body_string("Not found"))
            .mount(&mock)
            .await;

        let client = VoiceboxClient::new(&mock.uri());
        let err = client.get_profile("999").await.unwrap_err();
        match err {
            VoiceboxError::Api { status, body } => {
                assert_eq!(status, 404);
                assert_eq!(body, "Not found");
            }
            other => panic!("Expected Api error, got: {other}"),
        }
    }

    #[tokio::test]
    async fn create_profile_posts_json() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/profiles"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id": 1, "name": "New"})),
            )
            .mount(&mock)
            .await;

        let client = VoiceboxClient::new(&mock.uri());
        let body = serde_json::json!({"name": "New", "language": "en"});
        let result = client.create_profile(&body).await.unwrap();
        assert_eq!(result["name"], "New");
    }

    #[tokio::test]
    async fn health_check_works() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"status": "ok", "model_loaded": true})),
            )
            .mount(&mock)
            .await;

        let client = VoiceboxClient::new(&mock.uri());
        let result = client.health().await.unwrap();
        assert_eq!(result["status"], "ok");
    }

    #[tokio::test]
    async fn list_history_with_query_params() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"items": [], "total": 0})),
            )
            .mount(&mock)
            .await;

        let client = VoiceboxClient::new(&mock.uri());
        let result = client
            .list_history(Some("1"), Some("hello"), Some(10), Some(0))
            .await
            .unwrap();
        assert_eq!(result["total"], 0);
    }

    #[tokio::test]
    async fn generate_speech_posts_body() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": 1,
                "profile_id": 1,
                "text": "Hello",
                "audio_path": "/data/gen/1.wav",
                "duration": 1.5
            })))
            .mount(&mock)
            .await;

        let client = VoiceboxClient::new(&mock.uri());
        let body = serde_json::json!({
            "profile_id": 1,
            "text": "Hello",
            "language": "en"
        });
        let result = client.generate(&body).await.unwrap();
        assert_eq!(result["audio_path"], "/data/gen/1.wav");
    }

    #[tokio::test]
    async fn download_audio_returns_bytes() {
        let mock = MockServer::start().await;
        let audio_bytes = vec![0u8; 100];
        Mock::given(method("GET"))
            .and(path("/audio/1"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(audio_bytes.clone()))
            .mount(&mock)
            .await;

        let client = VoiceboxClient::new(&mock.uri());
        let result = client.get_audio("1").await.unwrap();
        assert_eq!(result.len(), 100);
    }

    #[tokio::test]
    async fn model_status_works() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models/status"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{"model_name": "qwen-tts-1.7B", "downloaded": true}]
            })))
            .mount(&mock)
            .await;

        let client = VoiceboxClient::new(&mock.uri());
        let result = client.get_model_status().await.unwrap();
        assert!(result["models"].is_array());
    }

    #[tokio::test]
    async fn story_crud_works() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/stories"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id": 1, "name": "My Story"})),
            )
            .mount(&mock)
            .await;

        let client = VoiceboxClient::new(&mock.uri());
        let body = serde_json::json!({"name": "My Story"});
        let result = client.create_story(&body).await.unwrap();
        assert_eq!(result["name"], "My Story");
    }
}
