"""The 10 validation results + acceptance harness (design/03_observables_bindings.md §4).

**Scope note.** The full dual-oracle fixture comparison design's Phase 7 describes (a tight-
tolerance f64 numpy-prototype oracle, and a loose-tolerance f32 pymurmur-CSV oracle) needs two
artifacts this environment doesn't have: the standalone numpy prototype is pre-coding-prep work
(roadmap.md §3 item 1) that was not produced as a separate deliverable, and pymurmur (`git_mur`)
is an external reference repository not present in this project. Building and honestly running
the acceptance harness against *this* Rust implementation — the actual "does the model reproduce
the paper's core findings" question — is what this module does; fixture-oracle comparison is a
documented gap, not silently skipped.
"""
