//! Every `CoreParams`/`OcclusionParams` field is read by at least one test — a drift scan
//! catching dead config: a field added (or renamed away from) that nothing actually consumes
//! (roadmap.md Phase 2b). Extends naturally to plugin-private params structs (e.g.
//! `PearceParams`) once Track C plugins land — the field-extraction helper below only assumes
//! a flat `pub struct Name { pub field: Type, ... }` shape, which every params struct in this
//! codebase follows (design/01_core.md §4).

mod support;

#[test]
fn every_core_and_occlusion_param_field_is_read_by_a_test() {
    let params_path = support::workspace_root().join("crates/murmur_core/src/params.rs");
    let params_src = std::fs::read_to_string(&params_path).expect("params.rs must exist");

    let mut fields = extract_struct_fields(&params_src, "CoreParams");
    fields.extend(extract_struct_fields(&params_src, "OcclusionParams"));
    assert!(
        !fields.is_empty(),
        "field extraction found nothing — the scan itself is broken"
    );

    let all_files = support::all_rust_source_files();
    let test_text = support::test_only_text(&all_files);

    let unread: Vec<&String> = fields
        .iter()
        .filter(|field| !test_text.contains(&format!(".{field}")))
        .collect();

    assert!(
        unread.is_empty(),
        "these CoreParams/OcclusionParams fields are never read via `.field` access in any \
         test in the workspace (dead config — guard_config_field_usage): {unread:?}"
    );
}

/// Extracts field names from a `pub struct Name { pub field: Type, ... }` block via simple
/// brace-matched text scanning — sufficient for this codebase's flat params structs, not a
/// general Rust parser.
fn extract_struct_fields(src: &str, struct_name: &str) -> Vec<String> {
    let marker = format!("pub struct {struct_name} {{");
    let start = src
        .find(&marker)
        .unwrap_or_else(|| panic!("could not find `{marker}`"));
    let body_start = start + marker.len();
    let end = body_start
        + src[body_start..]
            .find('}')
            .expect("unterminated struct body");
    let body = &src[body_start..end];

    body.lines()
        .filter_map(|line| {
            let line = line.trim().trim_end_matches(',');
            let line = line.strip_prefix("pub ")?;
            let name = line.split(':').next()?.trim();
            (!name.is_empty()).then(|| name.to_string())
        })
        .collect()
}
