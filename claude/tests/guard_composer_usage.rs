//! Every registered plugin name is constructed by at least one test (roadmap.md Phase 2b).
//!
//! Phase 2 has no plugin crates under `crates/plugins/*` yet (Phase 3 adds the first ones), so
//! this guard is a legitimate no-op today — there is nothing to check, not an unimplemented
//! check. Once a plugin crate exports `pub fn register`, this guard requires that function to
//! actually be *called* somewhere in the workspace (its own tests count — calling
//! `register(&mut reg)` and then resolving the plugin from it is exactly "constructed by a
//! test"), not merely defined and never invoked.

mod support;

#[test]
fn every_plugin_register_fn_is_exercised_by_some_test() {
    let plugins_dir = support::workspace_root().join("crates/plugins");
    let Ok(entries) = std::fs::read_dir(&plugins_dir) else {
        return; // no plugins/ subdirectories yet
    };

    let all_files = support::all_rust_source_files();

    for entry in entries.flatten() {
        let crate_dir = entry.path();
        if !crate_dir.is_dir() {
            continue;
        }
        let crate_name = crate_dir.file_name().unwrap().to_string_lossy().to_string();
        let crate_src_dir = crate_dir.join("src");

        let own_files: Vec<_> = all_files
            .iter()
            .filter(|p| p.starts_with(&crate_src_dir))
            .cloned()
            .collect();
        let own_src = support::read_all(&own_files);
        if !own_src.contains("pub fn register") {
            continue; // not (yet) a registrable plugin crate
        }

        // A call site (`register(&mut reg)` from within the crate's own tests, or
        // `crate_name::register(&mut registry)` from elsewhere) is textually distinct from the
        // definition (`pub fn register(r: &mut Registry)`): the parameter name differs (`r`
        // vs. the caller's own variable) and callers always pass `&mut`, so these substrings
        // only match actual invocations — scoped to *this* plugin, not any workspace call.
        let called_internally = own_src.contains("register(&mut");
        let qualified_call = format!("{crate_name}::register(&mut");
        let all_source = support::read_all(&all_files);
        let called_externally = all_source.contains(&qualified_call);

        assert!(
            called_internally || called_externally,
            "plugin crate `{crate_name}` exports register() but it is never called anywhere \
             in the workspace — every registered plugin should be constructed/exercised by at \
             least one test (guard_composer_usage)"
        );
    }
}
