use std::ffi::OsString;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Folders the frontend may ask Rust to reveal in the OS file manager.
/// Paths are resolved here, never taken from the webview or the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AppFolder {
    Generations,
    Captures,
    Models,
}

/// Mirrors how the backend picks its model cache: the custom directory the
/// app passes to the sidecar as VOICEBOX_MODELS_DIR wins, then the
/// huggingface_hub env vars, then ~/.cache/huggingface/hub.
pub fn models_dir(
    custom: Option<&str>,
    env: impl Fn(&str) -> Option<OsString>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    let env = |key: &str| env(key).filter(|v| !v.is_empty()).map(PathBuf::from);
    custom
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .or_else(|| env("VOICEBOX_MODELS_DIR"))
        .or_else(|| env("HF_HUB_CACHE"))
        .or_else(|| env("HF_HOME").map(|dir| dir.join("hub")))
        .or_else(|| env("XDG_CACHE_HOME").map(|dir| dir.join("huggingface").join("hub")))
        .or_else(|| home.map(|dir| dir.join(".cache").join("huggingface").join("hub")))
}

/// Resolve `folder` to an existing directory inside its allowed root. Data
/// folders are looked up in each of `data_dirs` in order, so a development
/// backend's data root can be tried before the bundled sidecar's.
pub fn resolve(
    folder: AppFolder,
    data_dirs: &[PathBuf],
    models_dir: Option<&Path>,
) -> Result<PathBuf, String> {
    let subdir = match folder {
        AppFolder::Generations => "generations",
        AppFolder::Captures => "captures",
        AppFolder::Models => {
            let dir = models_dir.ok_or("Models directory is not known")?;
            return resolve_in(dir, dir);
        }
    };
    let mut result = Err("No data directory is known".to_string());
    for data_dir in data_dirs {
        result = resolve_in(data_dir, &data_dir.join(subdir));
        if result.is_ok() {
            break;
        }
    }
    result
}

fn resolve_in(root: &Path, target: &Path) -> Result<PathBuf, String> {
    let root = root
        .canonicalize()
        .map_err(|e| format!("Folder root {} is not accessible: {e}", root.display()))?;
    let target = target
        .canonicalize()
        .map_err(|e| format!("Folder {} is not accessible: {e}", target.display()))?;

    if !target.starts_with(&root) {
        return Err(format!("{} is outside {}", target.display(), root.display()));
    }
    if !target.is_dir() {
        return Err(format!("{} is not a directory", target.display()));
    }
    // Opening a macOS bundle directory launches it instead of browsing it.
    if target
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("app"))
    {
        return Err(format!("{} is an application bundle", target.display()));
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("voicebox-app-folders-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn no_env(_: &str) -> Option<OsString> {
        None
    }

    #[test]
    fn folder_kind_deserializes_from_fixed_names_only() {
        let parse = |s: &str| serde_json::from_str::<AppFolder>(s);
        assert_eq!(parse("\"generations\"").unwrap(), AppFolder::Generations);
        assert_eq!(parse("\"captures\"").unwrap(), AppFolder::Captures);
        assert_eq!(parse("\"models\"").unwrap(), AppFolder::Models);
        assert!(parse("\"/etc\"").is_err());
        assert!(parse("\"../captures\"").is_err());
    }

    #[test]
    fn resolves_data_subfolders() {
        let tmp = TempDir::new("data");
        std::fs::create_dir_all(tmp.0.join("generations")).unwrap();
        std::fs::create_dir_all(tmp.0.join("captures")).unwrap();
        let root = tmp.0.canonicalize().unwrap();

        assert_eq!(
            resolve(AppFolder::Generations, &[tmp.0.clone()], None).unwrap(),
            root.join("generations")
        );
        assert_eq!(
            resolve(AppFolder::Captures, &[tmp.0.clone()], None).unwrap(),
            root.join("captures")
        );
    }

    #[test]
    fn rejects_missing_and_non_directory_targets() {
        let tmp = TempDir::new("missing");
        assert!(resolve(AppFolder::Generations, &[tmp.0.clone()], None).is_err());
        assert!(resolve(AppFolder::Generations, &[], None).is_err());

        std::fs::write(tmp.0.join("captures"), b"").unwrap();
        assert!(resolve(AppFolder::Captures, &[tmp.0.clone()], None).is_err());

        assert!(resolve(AppFolder::Models, &[tmp.0.clone()], None).is_err());
    }

    #[test]
    fn falls_back_to_later_data_dirs() {
        let tmp = TempDir::new("fallback");
        let dev = tmp.0.join("dev");
        let app = tmp.0.join("app");
        std::fs::create_dir_all(dev.join("generations")).unwrap();
        std::fs::create_dir_all(app.join("captures")).unwrap();
        std::fs::create_dir_all(app.join("generations")).unwrap();
        let dirs = [dev.clone(), app.clone()];

        assert_eq!(
            resolve(AppFolder::Generations, &dirs, None).unwrap(),
            dev.canonicalize().unwrap().join("generations")
        );
        assert_eq!(
            resolve(AppFolder::Captures, &dirs, None).unwrap(),
            app.canonicalize().unwrap().join("captures")
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escaping_data_dir() {
        let tmp = TempDir::new("escape");
        let data = tmp.0.join("data");
        let outside = tmp.0.join("outside");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, data.join("generations")).unwrap();

        let err = resolve(AppFolder::Generations, &[data], None).unwrap_err();
        assert!(err.contains("outside"), "{err}");
    }

    #[test]
    fn rejects_app_bundle_models_dir() {
        let tmp = TempDir::new("bundle");
        let bundle = tmp.0.join("Calculator.app");
        std::fs::create_dir_all(&bundle).unwrap();
        assert!(resolve(AppFolder::Models, &[], Some(&bundle)).is_err());
    }

    #[test]
    fn resolves_models_dir() {
        let tmp = TempDir::new("models");
        assert_eq!(
            resolve(AppFolder::Models, &[], Some(&tmp.0)).unwrap(),
            tmp.0.canonicalize().unwrap()
        );
    }

    #[test]
    fn models_dir_precedence() {
        let home = Some(PathBuf::from("/home/u"));
        let env = |vars: &'static [(&'static str, &'static str)]| {
            move |key: &str| {
                vars.iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| OsString::from(*v))
            }
        };

        assert_eq!(
            models_dir(Some("/custom"), env(&[("HF_HUB_CACHE", "/hub")]), home.clone()),
            Some(PathBuf::from("/custom"))
        );
        assert_eq!(
            models_dir(Some(""), env(&[("VOICEBOX_MODELS_DIR", "/vb")]), home.clone()),
            Some(PathBuf::from("/vb"))
        );
        assert_eq!(
            models_dir(None, env(&[("HF_HUB_CACHE", "/hub"), ("HF_HOME", "/hf")]), home.clone()),
            Some(PathBuf::from("/hub"))
        );
        assert_eq!(
            models_dir(None, env(&[("HF_HOME", "/hf")]), home.clone()),
            Some(PathBuf::from("/hf/hub"))
        );
        assert_eq!(
            models_dir(None, env(&[("XDG_CACHE_HOME", "/xdg")]), home.clone()),
            Some(PathBuf::from("/xdg/huggingface/hub"))
        );
        assert_eq!(
            models_dir(None, no_env, home),
            Some(PathBuf::from("/home/u/.cache/huggingface/hub"))
        );
        assert_eq!(models_dir(None, no_env, None), None);
    }
}
