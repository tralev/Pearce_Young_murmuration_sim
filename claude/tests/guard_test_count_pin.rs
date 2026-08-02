//! Silent-test-deletion guard (roadmap.md Phase 2b). A floor, not a ceiling: bump the pin
//! upward as the suite grows; only lower it in the same change that deliberately removes
//! tests (a consolidation), never to make an accidental red build pass.

mod support;

const MINIMUM_TEST_COUNT: usize = 40;

#[test]
fn workspace_test_count_is_at_least_the_pinned_minimum() {
    let files = support::all_rust_source_files();
    let count: usize = files
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .map(|text| text.matches("#[test]").count())
        .sum();

    assert!(
        count >= MINIMUM_TEST_COUNT,
        "workspace #[test] count dropped to {count}, below the pinned minimum of \
         {MINIMUM_TEST_COUNT} — if this is an intentional consolidation, lower the pin \
         deliberately in the same change that removes the tests; don't let it silently regress"
    );
}
