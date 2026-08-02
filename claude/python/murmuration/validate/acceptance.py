"""Applies tolerances to the 10 results and prints a pass/fail table (design/03 §4). Five
starred results are hard gates; run as `python -m murmuration.validate.acceptance` — exits
non-zero if any hard gate fails.

Tolerances are the design doc's own starting values (design/03_observables_bindings.md §4),
explicitly not re-derived here: "Phase 7 is where they get tightened or loosened against what
the actual implementations produce."
"""

import sys

import numpy as np

from murmuration.validate import results as R


def _no_monotonic_divergence(trace, slope_tolerance=0.05) -> bool:
    """P-d's 'no fragmentation' check: fit a line to R_max over the sampled window and reject
    only a clearly positive trend relative to the series' own scale — noise/equilibration
    wobble is expected, unbounded growth is not."""
    trace = np.asarray(trace, dtype=np.float64)
    if trace.mean() <= 0:
        return True
    x = np.arange(len(trace))
    slope, _intercept = np.polyfit(x, trace, 1)
    relative_slope = slope * len(trace) / trace.mean()
    return relative_slope < slope_tolerance


def _monotone_negative_trend(pairs) -> bool:
    """Y-c's sign check: Pearson correlation between thickness and m* should be negative."""
    thickness = np.array([t for t, _m in pairs])
    m_star = np.array([m for _t, m in pairs])
    if np.std(thickness) == 0 or np.std(m_star) == 0:
        return False
    corr = np.corrcoef(thickness, m_star)[0, 1]
    return corr < 0


def run() -> tuple[list[dict], bool]:
    """Runs all 10 results, applies tolerances, returns (rows, all_hard_gates_passed)."""
    rows = []
    all_hard_passed = True

    def record(id_, description, hard, value, passed, detail=""):
        nonlocal all_hard_passed
        rows.append(
            {
                "id": id_,
                "description": description,
                "hard": hard,
                "value": value,
                "passed": passed,
                "detail": detail,
            }
        )
        if hard and not passed:
            all_hard_passed = False

    # P-a (hard)
    theta_by_n = R.p_a_internal_opacity()
    p_a_passed = all(0.25 <= v <= 0.35 for v in theta_by_n.values())
    record("P-a", "internal opacity Theta-bar in [0.25, 0.35]", True, theta_by_n, p_a_passed)

    # P-b (report) — reuses P-a's data
    p_b = R.p_b_theta_vs_inverse_n(theta_by_n)
    record("P-b", "Theta vs 1/N linear (R^2 ~ 0.99)", False, p_b, None)

    # P-c (hard)
    p_c = R.p_c_density_scaling()
    b = p_c["exponent"]
    p_c_passed = -0.7 <= b <= -0.3
    record("P-c", "density scaling exponent in [-0.7, -0.3]", True, b, p_c_passed, detail=str(p_c["rho_by_n"]))

    # P-d (hard)
    p_d = R.p_d_no_fragmentation()
    p_d_passed = all(_no_monotonic_divergence(trace) for trace in p_d.values())
    record("P-d", "R_max bounded (no monotonic divergence)", True, p_d, p_d_passed)

    # P-e (report)
    p_e = R.p_e_tau_rho_vs_phi_p()
    record("P-e", "tau_rho decreases as phi_p increases", False, p_e, None)

    # P-f (report)
    p_f = R.p_f_phenotypes_distinct()
    record("P-f", "three phenotypes give distinct (alpha, Theta) regimes", False, p_f, None)

    # Y-a (hard)
    y_a = R.y_a_m_star()
    y_a_passed = y_a["m_star"] in (5, 6, 7)
    record("Y-a", "argmax_m robustness-per-neighbour in {5,6,7}", True, y_a["m_star"], y_a_passed)

    # Y-b (report)
    y_b = R.y_b_m_star_vs_n()
    record("Y-b", "m* independent of flock size N", False, y_b, None)

    # Y-c (hard)
    y_c = R.y_c_m_star_vs_thickness()
    pairs = [(v["thickness"], v["m_star"]) for v in y_c.values()]
    y_c_passed = _monotone_negative_trend(pairs)
    record("Y-c", "m* vs thickness: negative trend", True, y_c, y_c_passed)

    # Y-d (report)
    y_d = R.y_d_connectivity()
    record("Y-d", "sensing graph connected for all m >= 5", False, y_d, None)

    return rows, all_hard_passed


def print_table(rows) -> None:
    print(f"{'ID':<5} {'hard?':<6} {'result':<10} {'description'}")
    print("-" * 70)
    for row in rows:
        if row["hard"]:
            status = "PASS" if row["passed"] else "FAIL"
        else:
            status = "report"
        print(f"{row['id']:<5} {'yes' if row['hard'] else 'no':<6} {status:<10} {row['description']}")
        print(f"      value: {row['value']}")


def main() -> int:
    rows, all_hard_passed = run()
    print_table(rows)
    print()
    if all_hard_passed:
        print("All hard gates PASSED.")
        return 0
    else:
        failed = [r["id"] for r in rows if r["hard"] and not r["passed"]]
        print(f"Hard gate FAILURES: {failed}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
