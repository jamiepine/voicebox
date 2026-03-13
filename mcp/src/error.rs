use thiserror::Error;

#[derive(Debug, Error)]
pub enum VoiceboxError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),

    #[error("API error ({status}): {body}")]
    Api { status: u16, body: String },

    #[error("File I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}
