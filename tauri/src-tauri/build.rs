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

            match output {
                Ok(output) => {
                    // @modified AJ - Kamyab (Ankit Jain) — Graceful fallback when full Xcode is not installed
                    if !output.status.success() {
                        eprintln!("actool stderr: {}", String::from_utf8_lossy(&output.stderr));
                        eprintln!("actool stdout: {}", String::from_utf8_lossy(&output.stdout));
                        println!("cargo:warning=actool failed to compile icon (full Xcode may be required). Continuing without custom icon.");
                    } else {
                        println!("Successfully compiled icon to {}", gen_dir);
                    }
                }
                Err(e) => {
                    eprintln!("Failed to execute xcrun actool: {}", e);
                    println!("cargo:warning=Could not run actool (full Xcode may be required). Continuing without custom icon.");
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
