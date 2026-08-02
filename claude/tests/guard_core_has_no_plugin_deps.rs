//! Cargo.toml-level import-DAG enforcement of the core/plugin boundary (design/00_overview.md
//! "Core vs. plugin — the governing rule"; roadmap.md Phase 2b). `murmur_core` must depend on
//! zero `murmur_*` plugin crates — regression test for the boundary itself, not just a style
//! preference: if this ever fails, some plugin's "concrete algorithm or strategy" has leaked
//! into infrastructure.

mod support;

#[test]
fn murmur_core_cargo_toml_has_no_murmur_plugin_dependency() {
    let manifest_path = support::workspace_root().join("crates/murmur_core/Cargo.toml");
    let text = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", manifest_path.display()));
    let doc: toml::Value = text
        .parse()
        .expect("murmur_core/Cargo.toml must be valid TOML");

    let mut offending = Vec::new();
    for section in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(table) = doc.get(section).and_then(|v| v.as_table()) {
            for key in table.keys() {
                if key.starts_with("murmur_") && key != "murmur_core" {
                    offending.push(format!("{section}.{key}"));
                }
            }
        }
    }

    assert!(
        offending.is_empty(),
        "murmur_core must depend on zero murmur_* plugin crates (infrastructure-only rule, \
         design/00_overview.md); found: {offending:?}"
    );
}
