#[cfg(target_os = "macos")]
use std::process::Command;

fn main() {
    // Link Swift runtime libraries for screencapturekit crate
    #[cfg(target_os = "macos")]
    {
        // Add Swift runtime library paths to RPATH
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
        println!("cargo:rustc-link-arg=-L/usr/lib/swift");

        // Also try Xcode's Swift libraries
        if let Ok(output) = Command::new("xcode-select").arg("-p").output() {
            if output.status.success() {
                let xcode_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let swift_lib_path = format!(
                    "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx",
                    xcode_path
                );
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", swift_lib_path);
                println!("cargo:rustc-link-arg=-L{}", swift_lib_path);
            }
        }
    }

    // Compile macOS Liquid Glass icon
    #[cfg(target_os = "macos")]
    {
        let project_root = env!("CARGO_MANIFEST_DIR");
        // voicebox.icon is in tauri/assets/voicebox.icon (one level up from src-tauri)
        let icon_source = format!("{}/../assets/voicebox.icon", project_root);
        let gen_dir = format!("{}/gen", project_root);

        std::fs::create_dir_all(&gen_dir).expect("Failed to create gen directory");

        if std::path::Path::new(&icon_source).exists() {
            println!("cargo:rerun-if-changed={}", icon_source);
            println!("cargo:rerun-if-changed={}/icon.json", icon_source);
            println!("cargo:rerun-if-changed={}/Assets", icon_source);

            // Save original xcode-select path
            let original_path = Command::new("xcode-select")
                .arg("-p")
                .output()
                .ok()
                .and_then(|out| String::from_utf8(out.stdout).ok())
                .map(|s| s.trim().to_string());

            // Check if we need to switch to Xcode
            let needs_switch = original_path
                .as_ref()
                .map(|path| !path.contains("Xcode.app"))
                .unwrap_or(false);

            if needs_switch {
                // Switch to Xcode temporarily
                let switch_result = Command::new("sudo")
                    .args([
                        "xcode-select",
                        "--switch",
                        "/Applications/Xcode.app/Contents/Developer",
                    ])
                    .status();

                if switch_result.is_err() || !switch_result.unwrap().success() {
                    println!("cargo:warning=Failed to switch to Xcode - skipping icon compilation");
                    println!("cargo:warning=Install full Xcode or run manually: sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer");
                    return;
                }
            }

            let partial_plist = format!("{}/partial.plist", gen_dir);
            let output = Command::new("xcrun")
                .args([
                    "actool",
                    "--compile",
                    &gen_dir,
                    "--output-format",
                    "human-readable-text",
                    "--output-partial-info-plist",
                    &partial_plist,
                    "--app-icon",
                    "voicebox",
                    "--include-all-app-icons",
                    "--target-device",
                    "mac",
                    "--minimum-deployment-target",
                    "11.0",
                    "--platform",
                    "macosx",
                    &icon_source,
                ])
                .output();

            // Switch back to original path only if we changed it
            if needs_switch && original_path.is_some() {
                let orig_path = original_path.unwrap();
                let _ = Command::new("sudo")
                    .args(["xcode-select", "--switch", &orig_path])
                    .status();
            }

            match output {
                Ok(output) => {
                    if !output.status.success() {
                        eprintln!("actool stderr: {}", String::from_utf8_lossy(&output.stderr));
                        eprintln!("actool stdout: {}", String::from_utf8_lossy(&output.stdout));
                        println!("cargo:warning=actool failed to compile icon - continuing without custom icon");
                    } else {
                        println!("Successfully compiled icon to {}", gen_dir);
                    }
                }
                Err(e) => {
                    eprintln!("Failed to execute xcrun actool: {}", e);
                    println!("cargo:warning=Icon compilation skipped - continuing without custom icon");
                }
            }
        } else {
            println!(
                "cargo:warning=Icon source not found at {}, skipping icon compilation",
                icon_source
            );
        }
    }

    tauri_build::build()
}
