#
# Estimate d(n/V)/dt from sealed-body vacuum and temperature observations.
#
# Inverts the van der Waals equation of state sample-by-sample to get molar
# density rho(t), then least-squares fits rho vs. time.  The slope is the
# estimated leak rate; the regression standard error is its 1-sigma
# uncertainty.  With no real leak, the slope should be consistent with zero;
# a |z| = slope / sigma large compared to 3 indicates a significant trend.
#
# Defaults assume Slocum native column names (m_present_time, m_vacuum,
# m_veh_temp); override with --time-col / --vacuum-col / --temp-col for
# other CSV schemas.
#
# Pat Welch, pat@mousebrains.com

import argparse
import csv
import logging

import numpy as np
from scipy import stats

from slocum_tpw.simulate_leak import INHG_TO_PA, P_ATM_PA, vdw_density_vec


def load_csv(
    path: str,
    time_col: str = "m_present_time",
    vacuum_col: str = "m_vacuum",
    temp_col: str = "m_veh_temp",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, vacuum, and temperature columns from a CSV file.

    Non-numeric rows and rows with non-finite values are silently skipped.
    Returns three numpy arrays (time in seconds, vacuum in inHg, temperature
    in degC), sorted by time.

    Raises ``ValueError`` if the file is empty / unreadable, or ``KeyError``
    if the requested columns are missing.
    """
    t_list: list[float] = []
    v_list: list[float] = []
    T_list: list[float] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty or unreadable CSV")
        missing = [c for c in (time_col, vacuum_col, temp_col) if c not in reader.fieldnames]
        if missing:
            raise KeyError(f"{path}: missing column(s) {missing}; available {reader.fieldnames}")
        for row in reader:
            try:
                ti = float(row[time_col])
                vi = float(row[vacuum_col])
                Ti = float(row[temp_col])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(ti) and np.isfinite(vi) and np.isfinite(Ti)):
                continue
            t_list.append(ti)
            v_list.append(vi)
            T_list.append(Ti)

    t = np.asarray(t_list, dtype=float)
    v = np.asarray(v_list, dtype=float)
    T = np.asarray(T_list, dtype=float)
    order = np.argsort(t)
    return t[order], v[order], T[order]


def fit_leak_rate(time_s, vacuum_inHg, temperature_c) -> dict:
    """Fit d(n/V)/dt from observations.

    Inverts van der Waals per sample to get the inferred molar density, then
    does a least-squares linear fit of rho vs. time.

    Parameters are array-like:

    - ``time_s``: time in seconds (must be increasing; sorting is *not*
      enforced here — pass sorted values, e.g. from :func:`load_csv`)
    - ``vacuum_inHg``: measured vacuum in inHg (so absolute P = P_atm - vacuum)
    - ``temperature_c``: measured air temperature in degC

    Returns a dict containing:

    ===================== =====================================================
    ``slope``             d(n/V)/dt estimate, mol/(m^3 * s)
    ``slope_stderr``      1-sigma regression standard error on slope
    ``slope_95ci``        1.96 * slope_stderr (half-width of 95% CI)
    ``slope_per_day``     slope expressed as mol/(m^3 * day)
    ``slope_stderr_per_day`` 1-sigma in mol/(m^3 * day)
    ``intercept``         mol/m^3 at t = time_s[0]
    ``intercept_stderr``  1-sigma on intercept
    ``sigma_rho``         residual scatter of rho about the fit
    ``z_score``           slope / slope_stderr (|z| > ~3 => real trend)
    ``n_points``          number of valid rows used
    ``time_span_s``       time[-1] - time[0]
    ``time``              time (s) of valid samples
    ``rho``               inferred molar density (mol/m^3) at each valid sample
    ===================== =====================================================

    Raises ``ValueError`` if fewer than 3 valid rows survive.
    """
    time_s = np.asarray(time_s, dtype=float)
    vacuum_inHg = np.asarray(vacuum_inHg, dtype=float)
    temperature_c = np.asarray(temperature_c, dtype=float)

    if time_s.size < 3:
        raise ValueError(f"need at least 3 samples to fit a slope, got {time_s.size}")

    P_abs_Pa = P_ATM_PA - vacuum_inHg * INHG_TO_PA
    T_K = temperature_c + 273.15

    rho = vdw_density_vec(P_abs_Pa, T_K)
    good = np.isfinite(rho)
    bad = int((~good).sum())
    if bad:
        logging.warning("%d row(s) failed vdW inversion; dropped from fit", bad)

    t_g = time_s[good]
    rho_g = rho[good]
    if t_g.size < 3:
        raise ValueError("too few valid rows after vdW inversion")

    reg = stats.linregress(t_g, rho_g)
    rho_fit = reg.intercept + reg.slope * t_g
    sigma_rho = float((rho_g - rho_fit).std(ddof=2))
    z = reg.slope / reg.stderr if reg.stderr > 0 else float("nan")

    return {
        "slope": float(reg.slope),
        "slope_stderr": float(reg.stderr),
        "slope_95ci": float(1.96 * reg.stderr),
        "slope_per_day": float(reg.slope * 86400.0),
        "slope_stderr_per_day": float(reg.stderr * 86400.0),
        "intercept": float(reg.intercept),
        "intercept_stderr": float(reg.intercept_stderr),
        "sigma_rho": sigma_rho,
        "z_score": float(z),
        "n_points": int(t_g.size),
        "time_span_s": float(t_g[-1] - t_g[0]),
        "time": t_g,
        "rho": rho_g,
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add analyze-leak arguments to the parser."""
    parser.add_argument("csv_file", type=str, help="Path to input CSV")
    parser.add_argument(
        "--time-col",
        type=str,
        default="m_present_time",
        help="Time column name, seconds (default: m_present_time)",
    )
    parser.add_argument(
        "--vacuum-col",
        type=str,
        default="m_vacuum",
        help="Vacuum column name, inHg (default: m_vacuum)",
    )
    parser.add_argument(
        "--temp-col",
        type=str,
        default="m_veh_temp",
        help="Temperature column name, degC (default: m_veh_temp)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="PATH",
        help="Save a fit diagnostic plot to PATH (default: no plot)",
    )


def run(args: argparse.Namespace) -> int:
    """Execute the analyze-leak command."""
    try:
        t, vacuum, temp = load_csv(
            args.csv_file,
            time_col=args.time_col,
            vacuum_col=args.vacuum_col,
            temp_col=args.temp_col,
        )
    except (ValueError, KeyError, OSError) as e:
        logging.error("failed to read %s: %s", args.csv_file, e)
        return 1

    if t.size < 3:
        logging.error("not enough usable rows in %s (got %d)", args.csv_file, t.size)
        return 1

    try:
        result = fit_leak_rate(t, vacuum, temp)
    except ValueError as e:
        logging.error("fit failed: %s", e)
        return 1

    print(f"file                : {args.csv_file}")
    print(f"rows used           : {result['n_points']}")
    print(
        f"time span           : {result['time_span_s']:.1f} s "
        f"({result['time_span_s'] / 86400.0:.4f} days)"
    )
    print(f"rho range           : {result['rho'].min():.4f} .. {result['rho'].max():.4f} mol/m^3")
    print(f"residual sigma(rho) : {result['sigma_rho']:.4e} mol/m^3")
    print()
    print("Linear fit: rho(t) = intercept + slope * t")
    print(f"  slope              = {result['slope']:+.4e} mol/(m^3 * s)")
    print(f"  slope 1-sigma      = {result['slope_stderr']:.4e} mol/(m^3 * s)")
    print(f"  slope 95% CI       = +/- {result['slope_95ci']:.4e} mol/(m^3 * s)")
    print()
    print(f"  slope (per day)    = {result['slope_per_day']:+.4e} mol/(m^3 * day)")
    print(f"  slope 1-sigma (/d) = {result['slope_stderr_per_day']:.4e} mol/(m^3 * day)")
    print()
    print(f"  intercept          = {result['intercept']:.6f} mol/m^3")
    print(f"  intercept 1-sigma  = {result['intercept_stderr']:.4e} mol/m^3")
    print(f"  slope / sigma      = {result['z_score']:+.2f}  (|z| > ~3 suggests a real trend)")

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t_g = result["time"]
        rho_g = result["rho"]
        rho0 = rho_g[0]
        rho_fit = result["intercept"] + result["slope"] * t_g
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            (t_g - t_g[0]) / 3600.0,
            rho_g - rho0,
            ".",
            ms=0.5,
            alpha=0.3,
            label="inferred n/V - rho[0]",
        )
        ax.plot(
            (t_g - t_g[0]) / 3600.0,
            rho_fit - rho0,
            "-",
            color="C1",
            lw=1.4,
            label=(
                f"fit: slope = {result['slope']:+.3e} +/- {result['slope_stderr']:.1e} mol/m^3/s"
            ),
        )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Time from first sample (hours)")
        ax.set_ylabel("n/V - rho[0]  (mol/m^3)")
        ax.set_title(f"Leak fit: {args.csv_file}")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"  plot written to    : {args.plot}")

    return 0
