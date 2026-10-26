"""Quality cuts on an IFT reconstruction.

If a reconstruction fails these cuts it is likely bad, and the event should
be rejected from analysis. The cuts are the LOFAR1 IFT analysis cuts, and 
are likely different for a different analysis.

``reduced chi2``
    must be below 2

``core uncertainty``
    must be below 50 m

``Xmax railing``
    If more than 1% of the Xmax posterior sits in the outer 2% of the prior, the fit is
    likely railed against the prior edge because it could not find a good fit.
    These events often still have a decent chi square, but their result is not 
    reliable. The parametric fits used reference simulations of which not many were in the
    prior edge region, so often the parametrizations are already extrapolating in this range.

The thresholds live in :data:`QUALITY_CUTS` and can be overridden per call.
"""

import numpy as np


#: Default thresholds. ``xmax_rail_edge`` is the width of the sliver at each end
#: of the prior, as a fraction of the prior range; ``xmax_rail_max`` is how much
#: posterior mass may sit in one of them.
QUALITY_CUTS = {
    "red_chisq_max": 2.0,
    "core_unc_max_m": 50.0,
    "xmax_rail_edge": 0.02,
    "xmax_rail_max": 0.01,
}


def xmax_rail_fractions(xmax_samples, prior_min, prior_max, edge=None):
    """Fraction of the Xmax posterior sitting against each end of its prior.

    Parameters
    ----------
    xmax_samples : array_like
        Reported Xmax, one per posterior sample [g/cm2].
    prior_min, prior_max : float
        The Xmax prior this event was fitted under [g/cm2].
    edge : float, optional
        Sliver width as a fraction of the prior range. Defaults to
        ``QUALITY_CUTS["xmax_rail_edge"]``.

    Returns
    -------
    tuple of float
        ``(low, high)``, or ``(nan, nan)`` when there are no usable samples or
        the prior range is degenerate.
    """
    if edge is None:
        edge = QUALITY_CUTS["xmax_rail_edge"]
    samples = np.asarray(xmax_samples, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    span = float(prior_max) - float(prior_min)
    if samples.size < 1 or not np.isfinite(span) or span <= 0.0:
        return float("nan"), float("nan")
    position = (samples - float(prior_min)) / span
    return (float(np.mean(position < edge)),
            float(np.mean(position > 1.0 - edge)))


def evaluate(red_chisq, core_x_std, core_y_std, xmax_samples,
             xmax_prior_min, xmax_prior_max, cuts=None):
    """Apply the quality cuts and return the verdict.

    Every cut is evaluated.

    Returns
    -------
    dict
        ``passed`` (bool), ``reason`` (str, empty when it passed), ``reasons``
        (list of str), the measured value of each cut, and the thresholds used.
    """
    cuts = {**QUALITY_CUTS, **(cuts or {})}
    reasons = []
    checks = []

    chi2 = float(red_chisq) if red_chisq is not None else float("nan")
    if not np.isfinite(chi2):
        reasons.append("reduced chi2 is not available")
        checks.append(("reduced chi2", "n/a", "< %.2f" % cuts["red_chisq_max"], False))
    elif chi2 >= cuts["red_chisq_max"]:
        reasons.append("reduced chi2 %.2f is not < %.2f" % (chi2, cuts["red_chisq_max"]))
        checks.append(("reduced chi2", "%.3f" % chi2,
                       "< %.2f" % cuts["red_chisq_max"], False))
    else:
        checks.append(("reduced chi2", "%.3f" % chi2,
                       "< %.2f" % cuts["red_chisq_max"], True))

    core_unc = float(np.hypot(float(core_x_std), float(core_y_std)))
    if not np.isfinite(core_unc):
        reasons.append("core uncertainty is not available")
        checks.append(("core uncertainty", "n/a",
                       "< %.1f m" % cuts["core_unc_max_m"], False))
    elif core_unc >= cuts["core_unc_max_m"]:
        reasons.append("core uncertainty %.1f m is not < %.1f m"
                       % (core_unc, cuts["core_unc_max_m"]))
        checks.append(("core uncertainty", "%.1f m" % core_unc,
                       "< %.1f m" % cuts["core_unc_max_m"], False))
    else:
        checks.append(("core uncertainty", "%.1f m" % core_unc,
                       "< %.1f m" % cuts["core_unc_max_m"], True))

    rail_low, rail_high = xmax_rail_fractions(
        xmax_samples, xmax_prior_min, xmax_prior_max, edge=cuts["xmax_rail_edge"])
    rail_limit = "<= %.3f in outer %.0f%%" % (
        cuts["xmax_rail_max"], 100.0 * cuts["xmax_rail_edge"])
    if not np.isfinite(rail_low) or not np.isfinite(rail_high):
        reasons.append("Xmax railing could not be tested (no usable posterior "
                       "samples or no prior bounds)")
        checks.append(("Xmax rail low/high", "n/a", rail_limit, False))
    elif max(rail_low, rail_high) > cuts["xmax_rail_max"]:
        end = "lower" if rail_low >= rail_high else "upper"
        reasons.append(
            "Xmax railed against the %s prior bound: %.3f of the posterior "
            "sits in the outer %.0f%% of the prior, limit %.3f"
            % (end, max(rail_low, rail_high), 100.0 * cuts["xmax_rail_edge"],
               cuts["xmax_rail_max"]))
        checks.append(("Xmax rail low/high", "%.3f / %.3f" % (rail_low, rail_high),
                       rail_limit, False))
    else:
        checks.append(("Xmax rail low/high", "%.3f / %.3f" % (rail_low, rail_high),
                       rail_limit, True))

    return {
        "passed": not reasons,
        "reason": "; ".join(reasons),
        "reasons": reasons,
        "checks": checks,
        "red_chisq": chi2,
        "core_unc_m": core_unc,
        "xmax_rail_low": rail_low,
        "xmax_rail_high": rail_high,
        "cuts": cuts,
    }


def format_report(verdict, event_id=None):
    """Multi-line, grep-able summary of a verdict from :func:`evaluate`."""
    head = "QUALITY: %s" % ("ACCEPTED" if verdict["passed"] else "REJECTED")
    if event_id is not None:
        head += "  (event %s)" % event_id
    lines = [head]
    for name, value, limit, ok in verdict["checks"]:
        lines.append("  %-18s %-14s  limit %-24s %s"
                     % (name, value, limit, "pass" if ok else "FAIL"))
    if not verdict["passed"]:
        lines.append("  reason: %s" % verdict["reason"])
    return "\n".join(lines)
