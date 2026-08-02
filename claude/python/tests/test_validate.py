"""Fast smoke tests for murmuration.validate — small N/short runs, checking the harness itself
is wired correctly (types, shapes, no exceptions). The real acceptance run (`python -m
murmuration.validate.acceptance`, ~1-2 minutes at the design's N scale) is not run as part of
the routine pytest suite; see that module's docstring and roadmap.md Phase 7 for the current,
honestly-reported pass/fail status of the 5 hard gates against this implementation.
"""

from murmuration.validate import acceptance, results


def test_p_a_returns_a_value_per_n():
    out = results.p_a_internal_opacity(ns=(60, 120), warmup=30, samples=3, stride=5)
    assert set(out.keys()) == {60, 120}
    for theta in out.values():
        assert 0.0 <= theta <= 1.0


def test_p_c_returns_a_finite_exponent():
    out = results.p_c_density_scaling(ns=(60, 120, 200), warmup=30)
    assert "exponent" in out
    assert out["exponent"] == out["exponent"]  # not NaN


def test_p_d_returns_a_trace_per_phi_p():
    out = results.p_d_no_fragmentation(phi_ps=(0.03, 0.2), n=60, steps=200, check_every=100)
    assert set(out.keys()) == {0.03, 0.2}
    for trace in out.values():
        assert len(trace) == 2
        assert all(v > 0 for v in trace)


def test_y_a_returns_a_plausible_m_star():
    out = results.y_a_m_star(n=100, warmup=30)
    assert 1 <= out["m_star"] <= 12


def test_p_f_phenotypes_have_three_distinct_entries():
    out = results.p_f_phenotypes_distinct(n=60, warmup=30)
    assert set(out.keys()) == {"murmuration", "schooling", "swarming"}
    for entry in out.values():
        assert 0.0 <= entry["alpha"] <= 1.0
        assert 0.0 <= entry["theta"] <= 1.0


def test_no_monotonic_divergence_helper_flags_a_clear_runaway():
    stable = [100.0, 101.0, 99.0, 100.5, 100.2] * 4
    runaway = [100.0 * (1.1**i) for i in range(20)]
    assert acceptance._no_monotonic_divergence(stable)
    assert not acceptance._no_monotonic_divergence(runaway)


def test_monotone_negative_trend_helper():
    # (thickness, m_star) pairs: negative trend means m* falls as thickness rises.
    negative = [(0.3, 8), (0.6, 6), (0.9, 4)]
    positive = [(0.3, 2), (0.6, 5), (0.9, 8)]
    assert acceptance._monotone_negative_trend(negative)
    assert not acceptance._monotone_negative_trend(positive)


# The full `acceptance.run()` (10 results at the design's N scale, including P-d's 10,000-step
# traces) takes ~1-2 minutes — deliberately not exercised here to keep the routine pytest
# suite fast; run `python -m murmuration.validate.acceptance` directly for the real report.
