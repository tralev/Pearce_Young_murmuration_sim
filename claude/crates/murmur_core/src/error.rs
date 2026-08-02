//! Shared error type for parameter validation and plugin resolution (registry.rs, params.rs).

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// A parameter builder rejected a field value (params.rs).
    InvalidParam { field: &'static str, reason: String },
    /// A registry has no plugin registered under this name for this trait socket
    /// (design/02_plugins.md §1: "Unknown name → Err(ConfigError{kind: UnknownPlugin, name})").
    UnknownPlugin { socket: &'static str, name: String },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::InvalidParam { field, reason } => {
                write!(f, "invalid parameter `{field}`: {reason}")
            }
            ConfigError::UnknownPlugin { socket, name } => {
                write!(f, "unknown {socket} plugin: `{name}`")
            }
        }
    }
}

impl std::error::Error for ConfigError {}
