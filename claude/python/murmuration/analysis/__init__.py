"""Heavy analysis on piped positions/velocities (design/03_observables_bindings.md §3).

Nothing here is reimplemented in Rust — H2 robustness, PCA shape, density scaling, and density
correlation time are Python/scipy on top of the core's cheap per-frame numpy views.
"""
