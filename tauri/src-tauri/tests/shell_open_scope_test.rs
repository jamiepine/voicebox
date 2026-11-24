// The shell plugin anchors the configured open scope as ^{scope}$ before
// matching every JS shell open() call against it.
fn open_scope() -> regex::Regex {
    let config: serde_json::Value =
        serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
    let scope = config["plugins"]["shell"]["open"]
        .as_str()
        .expect("plugins.shell.open must be a regex string");
    regex::Regex::new(&format!("^{scope}$")).unwrap()
}

#[test]
fn allows_web_and_mail_links() {
    let scope = open_scope();
    for url in [
        "https://github.com/jamiepine/voicebox",
        "http://localhost:17493/docs",
        "mailto:hello@voicebox.sh",
    ] {
        assert!(scope.is_match(url), "{url} should be allowed");
    }
}

#[test]
fn rejects_local_and_script_targets() {
    let scope = open_scope();
    for target in [
        "file:///etc/passwd",
        "smb://attacker.example/share",
        "javascript:alert(1)",
        "/Applications/Calculator.app",
        "C:\\Windows\\System32\\calc.exe",
        "\\\\attacker.example\\share",
        "https://",
        "mailto:",
        "https://example.com\nfile:///etc/passwd",
        " https://example.com",
    ] {
        assert!(!scope.is_match(target), "{target} should be rejected");
    }
}
