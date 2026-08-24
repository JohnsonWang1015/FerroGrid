//! External transfer tools, declared in config and run on the nodes.
//!
//! FerroGrid does not speak WebDAV, S3 or anything else. Moving data is a
//! solved problem with good tools; what FerroGrid adds is running one of them
//! on every node at once, so each node pulls its own copy instead of relaying
//! a dataset through the controller.
//!
//! A plugin is an **argv template**, never a shell string:
//!
//! ```toml
//! [nextcloud]
//! description = "Nextcloud NAS over WebDAV"
//! fetch = ["ncfetch", "mirror", "{remote}", "--out", "{local}"]
//! push  = ["ncfetch", "upload-folder", "{local}", "{remote}"]
//! workdir = "~/.config/ferrogrid"
//! ```
//!
//! `{remote}` and `{local}` are substituted as whole argv elements and the
//! command is exec'd directly, so a path containing spaces, quotes or `;` is
//! just a path. FerroGrid never sees the plugin's credentials: the tool reads
//! its own configuration from `workdir` on each node.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct Plugin {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub fetch: Vec<String>,
    #[serde(default)]
    pub push: Vec<String>,
    /// Directory the command runs in on each node. This is where the tool
    /// finds its own credentials (a `.env`, a rclone config, ...).
    #[serde(default)]
    pub workdir: String,
}

impl Plugin {
    /// Build the argv for one action, substituting whole elements only.
    pub fn argv(&self, action: &str, remote: &str, local: &str) -> Result<Vec<String>> {
        let template = match action {
            "fetch" => &self.fetch,
            "push" => &self.push,
            other => anyhow::bail!("unknown action `{other}`, expected fetch or push"),
        };
        if template.is_empty() {
            anyhow::bail!("this plugin does not define a `{action}` command");
        }
        Ok(template
            .iter()
            .map(|part| part.replace("{remote}", remote).replace("{local}", local))
            .collect())
    }
}

#[derive(Debug, Default, Clone)]
pub struct Registry {
    pub plugins: BTreeMap<String, Plugin>,
    pub source: Option<PathBuf>,
}

impl Registry {
    /// Load from an explicit path, else the first default location that exists.
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let candidates: Vec<PathBuf> = match explicit {
            Some(p) => vec![p.to_path_buf()],
            None => {
                let mut v = Vec::new();
                if let Ok(home) = std::env::var("HOME") {
                    v.push(PathBuf::from(&home).join(".config/ferrogrid/plugins.toml"));
                }
                v.push(PathBuf::from("plugins.toml"));
                v
            }
        };

        for path in candidates {
            if !path.is_file() {
                continue;
            }
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            let plugins: BTreeMap<String, Plugin> = toml::from_str(&text)
                .with_context(|| format!("parse {}", path.display()))?;
            return Ok(Self { plugins, source: Some(path) });
        }

        // No config is not an error: the cluster simply has no plugins.
        Ok(Self::default())
    }

    pub fn get(&self, name: &str) -> Result<&Plugin> {
        self.plugins.get(name).with_context(|| {
            let known: Vec<&str> = self.plugins.keys().map(String::as_str).collect();
            if known.is_empty() {
                "no plugins configured; see plugins.example.toml".to_string()
            } else {
                format!("unknown plugin `{name}`; configured: {}", known.join(", "))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nextcloud() -> Plugin {
        toml::from_str(
            r#"
            description = "Nextcloud"
            fetch = ["ncfetch", "mirror", "{remote}", "--out", "{local}"]
            push  = ["ncfetch", "upload-folder", "{local}", "{remote}"]
            "#,
        )
        .unwrap()
    }

    #[test]
    fn substitutes_whole_argv_elements() {
        let argv = nextcloud().argv("fetch", "Datasets/adni", "/data/adni").unwrap();
        assert_eq!(
            argv,
            vec!["ncfetch", "mirror", "Datasets/adni", "--out", "/data/adni"]
        );
    }

    #[test]
    fn a_path_with_shell_metacharacters_stays_one_argument() {
        // Exec'd directly, never through a shell, so this is a path and not
        // an injection. push is ["ncfetch", "upload-folder", {local}, {remote}].
        let nasty = "/data/my results; rm -rf /";
        let argv = nextcloud().argv("push", "Backups/x", nasty).unwrap();
        assert_eq!(argv, vec!["ncfetch", "upload-folder", nasty, "Backups/x"]);
    }

    #[test]
    fn rejects_an_action_the_plugin_does_not_define() {
        let p: Plugin = toml::from_str(r#"fetch = ["x", "{remote}", "{local}"]"#).unwrap();
        assert!(p.argv("push", "a", "b").is_err());
        assert!(p.argv("fetch", "a", "b").is_ok());
    }

    #[test]
    fn unknown_action_is_an_error() {
        assert!(nextcloud().argv("sync", "a", "b").is_err());
    }

    #[test]
    fn missing_config_is_not_an_error() {
        let r = Registry::load(Some(Path::new("/nonexistent/plugins.toml"))).unwrap();
        assert!(r.plugins.is_empty());
        assert!(r.get("nextcloud").is_err());
    }
}
