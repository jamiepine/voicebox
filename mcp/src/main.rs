use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::StreamableHttpService;
use rmcp::ServiceExt;
use tracing_subscriber::EnvFilter;

use voicebox_mcp::client::VoiceboxClient;
use voicebox_mcp::server::VoiceboxMcp;

/// MCP server for Voicebox voice synthesis studio.
///
/// Exposes Voicebox capabilities (TTS, transcription, voice profiles,
/// stories, model management) over the Model Context Protocol.
#[derive(Parser)]
#[command(name = "voicebox-mcp", version)]
struct Cli {
    /// Voicebox API base URL
    #[arg(long, default_value = "http://localhost:17493", env = "VOICEBOX_URL")]
    url: String,

    /// Transport mode
    #[arg(long, default_value = "stdio", env = "VOICEBOX_MCP_TRANSPORT")]
    transport: Transport,

    /// Port for HTTP transport
    #[arg(long, default_value = "3100", env = "VOICEBOX_MCP_PORT")]
    port: u16,

    /// Bind address for HTTP transport
    #[arg(long, default_value = "127.0.0.1", env = "VOICEBOX_MCP_HOST")]
    host: String,
}

#[derive(Clone, Debug)]
enum Transport {
    Stdio,
    Http,
}

impl std::str::FromStr for Transport {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "stdio" => Ok(Self::Stdio),
            "http" => Ok(Self::Http),
            other => Err(format!(
                "unknown transport: {other} (expected stdio or http)"
            )),
        }
    }
}

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stdio => write!(f, "stdio"),
            Self::Http => write!(f, "http"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let client = VoiceboxClient::new(&cli.url);

    match cli.transport {
        Transport::Stdio => {
            tracing::info!("Starting voicebox-mcp (stdio) -> {}", cli.url);
            let server = VoiceboxMcp::new(client);
            let service = server
                .serve(rmcp::transport::stdio())
                .await
                .context("Failed to start stdio transport")?;
            service.waiting().await?;
        }
        Transport::Http => {
            tracing::info!(
                "Starting voicebox-mcp (http) on {}:{} -> {}",
                cli.host,
                cli.port,
                cli.url,
            );

            let mcp_service = StreamableHttpService::new(
                move || Ok(VoiceboxMcp::new(client.clone())),
                Arc::new(LocalSessionManager::default()),
                Default::default(),
            );

            let app = axum::Router::new().route("/mcp", axum::routing::any_service(mcp_service));

            let addr = format!("{}:{}", cli.host, cli.port);
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .context(format!("Failed to bind to {addr}"))?;

            tracing::info!("Listening on {addr}/mcp");
            axum::serve(listener, app).await?;
        }
    }

    Ok(())
}
