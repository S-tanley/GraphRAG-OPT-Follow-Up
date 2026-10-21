"""
Statistics for the corrected Table 2 (MICRO 2026 rebuttal), stdlib-only.

Two quantities, two different uncertainties (kept distinct on purpose):

  * Per-cell ACCURACY uncertainty -> Wilson score 95% CI (Wilson 1927). This is
    the marginal binomial CI for one (model, dataset, method) accuracy.

  * REUSE-FIDELITY claim (GraphRAG-Opt vs GraphRAG-R) -> a PAIRED comparison on
    the SAME questions, so its uncertainty depends only on the discordant pairs,
    not the marginal accuracies. We report:
      - paired difference (Opt-R) with a Wald paired 95% CI,
      - exact mid-p McNemar test (Fagerland et al. 2013; Dietterich 1998 shows
        McNemar is the right paired test for single-run classifiers),
      - a non-inferiority verdict at margin delta (default 2.5 pts): Opt is
        non-inferior to R iff the one-sided 95% lower bound on (p_Opt - p_R) is
        above -delta (Tango 1998; Liu et al. 2002). A non-significant McNemar is
        NOT used as evidence of equivalence -- non-inferiority is tested directly.

Pairing is "a free reduction in estimator variance" (Miller 2024, arXiv:2411.00640)
and is exact here because the sampler fixes seed 42 and selects the SAME question
IDs for every model and method before sharding.

Formulas + reference values independently verified.
"""

import math

Z95 = 1.959964      # two-sided 95%
Z90_1SIDED = 1.6449  # one-sided 95% (= lower bound of a 90% two-sided CI)


def wilson_ci(k, n, z=Z95):
    """95% Wilson score interval for a binomial proportion k/n -> (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return (center - half, center + half)


def wilson_halfwidth(k, n, z=Z95):
    lo, hi = wilson_ci(k, n, z)
    return (hi - lo) / 2.0


def mcnemar_midp(b, c):
    """Two-sided mid-p McNemar p-value for discordant counts b, c.
    b = #(Opt correct, R wrong); c = #(R correct, Opt wrong)."""
    nd = b + c
    if nd == 0:
        return 1.0
    m = min(b, c)
    total = 2 ** nd
    cdf_lt = sum(math.comb(nd, i) for i in range(m))   # P(X < m)
    p_eq = math.comb(nd, m) / total                    # P(X = m)
    return 2.0 * (cdf_lt / total) + p_eq


def mcnemar_exact(b, c):
    """Two-sided exact-conditional McNemar p-value (conservative)."""
    nd = b + c
    if nd == 0:
        return 1.0
    m = min(b, c)
    s = sum(math.comb(nd, i) for i in range(m + 1))     # P(X <= m)
    return min(1.0, 2.0 * s / (2 ** nd))


def paired_diff_ci(b, c, N, z=Z95):
    """Wald 95% CI for the paired difference (p_Opt - p_R) -> (D_hat, lo, hi)."""
    if N == 0:
        return (0.0, -1.0, 1.0)
    D = (b - c) / N
    var = b + c - (b - c) ** 2 / N
    half = z * math.sqrt(var) / N if var > 0 else 0.0
    return (D, D - half, D + half)


def noninferiority(b, c, N, delta=0.025, z=Z90_1SIDED):
    """Paired non-inferiority of Opt vs R at margin delta -> (passes, lower_bound).
    Passes iff the one-sided 95% lower bound on (p_Opt - p_R) exceeds -delta."""
    if N == 0:
        return (False, 0.0)
    D = (b - c) / N
    var = b + c - (b - c) ** 2 / N
    lower = D - z * math.sqrt(var) / N if var > 0 else D
    return (lower > -delta, lower)


def paired_counts(opt_hits, r_hits):
    """From two aligned 0/1 lists (same questions, same order) -> (b, c, n_d, N).
    b = Opt correct & R wrong; c = R correct & Opt wrong."""
    assert len(opt_hits) == len(r_hits), (len(opt_hits), len(r_hits))
    b = sum(1 for o, r in zip(opt_hits, r_hits) if o and not r)
    c = sum(1 for o, r in zip(opt_hits, r_hits) if r and not o)
    return b, c, b + c, len(opt_hits)


def fmt_pct_ci(k, n):
    """'85.94 (±2.1)' style: percentage point estimate + Wilson half-width in pts."""
    if n == 0:
        return "--"
    p = 100.0 * k / n
    hw = 100.0 * wilson_halfwidth(k, n)
    return f"{p:.2f} ($\\pm${hw:.1f})"


def _selftest():
    def approx(a, b, t=1e-5):
        assert abs(a - b) <= t, (a, b)
    approx(wilson_ci(250, 500)[0], 0.45634); approx(wilson_ci(250, 500)[1], 0.54366)
    approx(wilson_ci(400, 500)[0], 0.76271); approx(wilson_ci(5, 10)[0], 0.23659)
    assert wilson_ci(0, 0) == (0.0, 1.0)
    approx(wilson_halfwidth(250, 500), 0.04366, 1e-4)
    approx(mcnemar_midp(20, 5), 0.00249, 1e-4); approx(mcnemar_exact(20, 5), 0.00408, 1e-4)
    assert mcnemar_midp(0, 0) == 1.0; approx(mcnemar_midp(3, 0), 0.125)
    approx(mcnemar_midp(25, 15), 0.11728, 1e-4)
    approx(paired_diff_ci(20, 5, 500)[0], 0.03); approx(paired_diff_ci(20, 5, 500)[2], 0.04942, 1e-4)
    approx(paired_diff_ci(12, 13, 500)[1], -0.02160, 1e-4)
    ok, lo = noninferiority(12, 13, 500); assert ok; approx(lo, -0.01845, 1e-4)
    ok, lo = noninferiority(20, 40, 500); assert not ok; approx(lo, -0.06531, 1e-4)
    print("table2_stats self-test PASSED")


if __name__ == "__main__":
    _selftest()
