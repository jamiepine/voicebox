use rmcp::ServerHandler;
use voicebox_mcp::client::VoiceboxClient;
use voicebox_mcp::server::VoiceboxMcp;

// --- Server construction ---

#[test]
fn server_can_be_constructed() {
    let client = VoiceboxClient::new("http://localhost:17493");
    let _server = VoiceboxMcp::new(client);
}

#[test]
fn server_is_clone() {
    let client = VoiceboxClient::new("http://localhost:17493");
    let server = VoiceboxMcp::new(client);
    let _cloned = server.clone();
}

// --- Tool discovery (public MCP interface) ---

#[test]
fn all_tools_are_registered() {
    let client = VoiceboxClient::new("http://localhost:17493");
    let server = VoiceboxMcp::new(client);

    let expected_tools = [
        // Profiles
        "list_profiles",
        "get_profile",
        "create_profile",
        "update_profile",
        "delete_profile",
        "list_profile_samples",
        "add_profile_sample",
        // Generation
        "generate_speech",
        "get_audio",
        // History
        "list_history",
        "get_generation",
        "delete_generation",
        "get_history_stats",
        // Transcription
        "transcribe",
        // Stories
        "list_stories",
        "get_story",
        "create_story",
        "update_story",
        "delete_story",
        "add_story_item",
        "move_story_item",
        "remove_story_item",
        "export_story",
        // Models
        "get_model_status",
        "load_model",
        "unload_model",
        "download_model",
        // System
        "health_check",
        "clear_cache",
    ];

    for tool_name in &expected_tools {
        assert!(
            server.get_tool(tool_name).is_some(),
            "Tool '{}' not found in server",
            tool_name
        );
    }
}

#[test]
fn tools_have_descriptions() {
    let client = VoiceboxClient::new("http://localhost:17493");
    let server = VoiceboxMcp::new(client);

    let tool = server.get_tool("generate_speech").unwrap();
    assert!(
        tool.description.is_some(),
        "generate_speech should have a description"
    );
    assert!(
        !tool.description.as_ref().unwrap().is_empty(),
        "generate_speech description should not be empty"
    );
}

#[test]
fn server_info_has_instructions() {
    let client = VoiceboxClient::new("http://localhost:17493");
    let server = VoiceboxMcp::new(client);
    let info = server.get_info();
    assert!(info.instructions.is_some());
    assert!(info.instructions.unwrap().contains("Voicebox"));
}
