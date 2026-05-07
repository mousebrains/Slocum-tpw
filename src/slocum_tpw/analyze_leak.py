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
from pathlib import Path

import numpy as np
from scipy import stats

from slocum_tpw.simulate_leak import INHG_TO_PA, P_ATM_PA, vdw_density_vec

# Molar mass of dry air, g/mol; multiplying mol/m^3 by this yields g/m^3 == mg/L.
M_AIR = 28.9647

_NETCDF_SUFFIXES = {".nc", ".nc4", ".netcdf", ".cdf"}


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


def load_netcdf(
    path: str,
    time_col: str = "m_present_time",
    vacuum_col: str = "m_vacuum",
    temp_col: str = "m_veh_temp",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, vacuum, and temperature variables from a NetCDF file.

    Non-finite values are silently dropped.  Returns three numpy arrays
    (time in seconds, vacuum in inHg, temperature in degC), sorted by time.
    Datetime64 time variables are converted to POSIX seconds.

    Raises ``KeyError`` if any requested variable is missing.
    """
    import xarray as xr

    with xr.open_dataset(path) as ds:
        missing = [c for c in (time_col, vacuum_col, temp_col) if c not in ds.variables]
        if missing:
            raise KeyError(f"{path}: missing variable(s) {missing}; available {list(ds.variables)}")
        t_raw = ds[time_col].values
        v = np.asarray(ds[vacuum_col].values, dtype=float).ravel()
        T = np.asarray(ds[temp_col].values, dtype=float).ravel()

    if np.issubdtype(t_raw.dtype, np.datetime64):
        t = (t_raw - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        t = np.asarray(t, dtype=float).ravel()
    else:
        t = np.asarray(t_raw, dtype=float).ravel()

    if not (t.size == v.size == T.size):
        raise ValueError(
            f"{path}: variables have inconsistent lengths "
            f"({time_col}={t.size}, {vacuum_col}={v.size}, {temp_col}={T.size})"
        )

    good = np.isfinite(t) & np.isfinite(v) & np.isfinite(T)
    t = t[good]
    v = v[good]
    T = T[good]
    order = np.argsort(t)
    return t[order], v[order], T[order]


def _load(
    path: str,
    time_col: str,
    vacuum_col: str,
    temp_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to ``load_netcdf`` for ``.nc``-style files, else ``load_csv``."""
    if Path(path).suffix.lower() in _NETCDF_SUFFIXES:
        return load_netcdf(path, time_col=time_col, vacuum_col=vacuum_col, temp_col=temp_col)
    return load_csv(path, time_col=time_col, vacuum_col=vacuum_col, temp_col=temp_col)


def _ar1_correction(resid: np.ndarray) -> dict:
    """Lag-1 autocorrelation of residuals and the implied stderr inflation.

    Assumes near-uniform sample spacing.  Returns a dict with ``rho1``,
    ``factor`` (multiplier for OLS stderr), and ``n_eff`` (effective sample
    size).  When |rho1| >= 1 (degenerate), factor is ``inf`` and n_eff is 0.
    """
    if resid.size < 2:
        return {"rho1": float("nan"), "factor": float("nan"), "n_eff": 0.0}
    a = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    if not np.isfinite(a) or abs(a) >= 1.0:
        return {"rho1": a, "factor": float("inf"), "n_eff": 0.0}
    return {
        "rho1": a,
        "factor": float(np.sqrt((1 + a) / (1 - a))),
        "n_eff": float(resid.size * (1 - a) / (1 + a)),
    }


def _fit_linear_sinusoid(t_s: np.ndarray, y: np.ndarray, period_s: float) -> dict:
    """Joint OLS fit of ``y = a + b*t + c*cos(omega*t) + d*sin(omega*t)``.

    The origin is shifted to ``t_s[0]`` internally for numerical conditioning,
    so ``intercept`` is the value at ``t = t_s[0]`` and ``phase`` is referenced
    to that origin.  Slope, amplitude, and residuals are unchanged by the shift.
    """
    t_local = np.asarray(t_s, dtype=float) - float(t_s[0])
    n = t_local.size
    omega = 2 * np.pi / period_s
    X = np.column_stack([np.ones(n), t_local, np.cos(omega * t_local), np.sin(omega * t_local)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid / (n - X.shape[1]))
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "intercept_stderr": float(se[0]),
        "slope_stderr": float(se[1]),
        "amplitude": float(np.hypot(beta[2], beta[3])),
        "phase": float(np.arctan2(beta[3], beta[2])),
        "residuals": resid,
    }


def fit_leak_rate(
    time_s,
    vacuum_inHg,
    temperature_c,
    *,
    ar1: bool = False,
    sinusoid_period_s: float | None = None,
) -> dict:
    """Fit d(n/V)/dt from observations.

    Inverts van der Waals per sample to get the inferred molar density, then
    does a least-squares linear fit of rho vs. time.

    Parameters are array-like:

    - ``time_s``: time in seconds (must be increasing; sorting is *not*
      enforced here — pass sorted values, e.g. from :func:`load_csv`)
    - ``vacuum_inHg``: measured vacuum in inHg (so absolute P = P_atm - vacuum)
    - ``temperature_c``: measured air temperature in degC

    Optional keyword-only diagnostics:

    - ``ar1``: also report an AR(1)-corrected slope stderr based on the lag-1
      autocorrelation of the residuals.  Adds ``ar1_*`` keys.
    - ``sinusoid_period_s``: when set, also fit ``rho(t) = a + b*t +
      c*cos(omega*t) + d*sin(omega*t)`` with this period (seconds) and report
      its linear-trend slope.  Adds ``sin_*`` keys.  Combining with ``ar1``
      adds ``sin_ar1_*`` keys for the AR(1) correction on the joint residual.

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
    residuals = rho_g - rho_fit
    sigma_rho = float(residuals.std(ddof=2))
    z = reg.slope / reg.stderr if reg.stderr > 0 else float("nan")

    result: dict = {
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

    if ar1:
        c = _ar1_correction(residuals)
        ar1_se = result["slope_stderr"] * c["factor"]
        result["ar1_rho1"] = c["rho1"]
        result["ar1_factor"] = c["factor"]
        result["ar1_n_eff"] = c["n_eff"]
        result["ar1_slope_stderr"] = ar1_se
        result["ar1_slope_stderr_per_day"] = ar1_se * 86400.0
        result["ar1_t_value"] = (
            float(reg.slope / ar1_se) if np.isfinite(ar1_se) and ar1_se > 0 else float("nan")
        )

    if sinusoid_period_s is not None:
        sf = _fit_linear_sinusoid(t_g, rho_g, sinusoid_period_s)
        sin_t = sf["slope"] / sf["slope_stderr"] if sf["slope_stderr"] > 0 else float("nan")
        result["sin_period_s"] = float(sinusoid_period_s)
        result["sin_slope"] = sf["slope"]
        result["sin_slope_stderr"] = sf["slope_stderr"]
        result["sin_slope_per_day"] = sf["slope"] * 86400.0
        result["sin_slope_stderr_per_day"] = sf["slope_stderr"] * 86400.0
        result["sin_intercept"] = sf["intercept"]
        result["sin_intercept_stderr"] = sf["intercept_stderr"]
        result["sin_amplitude"] = sf["amplitude"]
        result["sin_phase"] = sf["phase"]
        result["sin_t_value"] = float(sin_t)
        if ar1:
            cs = _ar1_correction(sf["residuals"])
            sin_ar1_se = sf["slope_stderr"] * cs["factor"]
            result["sin_ar1_rho1"] = cs["rho1"]
            result["sin_ar1_factor"] = cs["factor"]
            result["sin_ar1_n_eff"] = cs["n_eff"]
            result["sin_ar1_slope_stderr"] = sin_ar1_se
            result["sin_ar1_slope_stderr_per_day"] = sin_ar1_se * 86400.0
            result["sin_ar1_t_value"] = (
                float(sf["slope"] / sin_ar1_se)
                if np.isfinite(sin_ar1_se) and sin_ar1_se > 0
                else float("nan")
            )

    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add analyze-leak arguments to the parser."""
    parser.add_argument(
        "input_file",
        type=str,
        metavar="FILE",
        help=(
            "Path to input CSV or NetCDF file (NetCDF is detected by .nc/.nc4/.netcdf/.cdf suffix)"
        ),
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="m_present_time",
        help="Time column/variable name, seconds (default: m_present_time)",
    )
    parser.add_argument(
        "--vacuum-col",
        type=str,
        default="m_vacuum",
        help="Vacuum column/variable name, inHg (default: m_vacuum)",
    )
    parser.add_argument(
        "--temp-col",
        type=str,
        default="m_veh_temp",
        help="Temperature column/variable name, degC (default: m_veh_temp)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="PATH",
        help="Save a fit diagnostic plot to PATH (default: no plot)",
    )
    parser.add_argument(
        "--ar1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Report an AR(1)-corrected slope stderr from the lag-1 residual "
            "autocorrelation (default: enabled; pass --no-ar1 to disable)"
        ),
    )
    parser.add_argument(
        "--sinusoid",
        action="store_true",
        help=(
            "Also fit rho(t) = a + b*t + c*cos(omega*t) + d*sin(omega*t) and "
            "report the linear-trend slope from that joint model"
        ),
    )
    parser.add_argument(
        "--sinusoid-period",
        type=float,
        default=24.0,
        metavar="HOURS",
        help="Period (hours) for --sinusoid (default: 24.0)",
    )


def run(args: argparse.Namespace) -> int:
    """Execute the analyze-leak command."""
    try:
        t, vacuum, temp = _load(
            args.input_file,
            time_col=args.time_col,
            vacuum_col=args.vacuum_col,
            temp_col=args.temp_col,
        )
    except (ValueError, KeyError, OSError) as e:
        logging.error("failed to read %s: %s", args.input_file, e)
        return 1

    if t.size < 3:
        logging.error("not enough usable rows in %s (got %d)", args.input_file, t.size)
        return 1

    sinusoid_period_s = args.sinusoid_period * 3600.0 if args.sinusoid else None
    try:
        result = fit_leak_rate(
            t,
            vacuum,
            temp,
            ar1=args.ar1,
            sinusoid_period_s=sinusoid_period_s,
        )
    except ValueError as e:
        logging.error("fit failed: %s", e)
        return 1

    print(f"file                : {args.input_file}")
    print(f"rows used           : {result['n_points']}")
    print(
        f"time span           : {result['time_span_s']:.1f} s "
        f"({result['time_span_s'] / 86400.0:.4f} days)"
    )
    rho_min = result["rho"].min() * M_AIR
    rho_max = result["rho"].max() * M_AIR
    sigma_rho = result["sigma_rho"] * M_AIR
    slope_s = result["slope"] * M_AIR
    slope_se_s = result["slope_stderr"] * M_AIR
    slope_95ci = result["slope_95ci"] * M_AIR
    slope_day = result["slope_per_day"] * M_AIR
    slope_se_day = result["slope_stderr_per_day"] * M_AIR
    intercept = result["intercept"] * M_AIR
    intercept_se = result["intercept_stderr"] * M_AIR
    print(f"rho range           : {rho_min:.4f} .. {rho_max:.4f} mg/L")
    print(f"residual sigma(rho) : {sigma_rho:.4e} mg/L")
    print()
    if args.ar1:
        ar1_se_s = result["ar1_slope_stderr"] * M_AIR
        ar1_se_day = result["ar1_slope_stderr_per_day"] * M_AIR
        ar1_95ci = 1.96 * ar1_se_s
        print("Linear fit (AR(1)-corrected stderr): rho(t) = intercept + slope * t")
        print(
            f"  AR(1) details      : rho_1 = {result['ar1_rho1']:+.4f}, "
            f"n_eff = {result['ar1_n_eff']:.0f}, factor = {result['ar1_factor']:.3f}"
        )
        print(
            f"  slope              = {slope_s:+.4e} +/- {ar1_se_s:.4e} "
            f"mg/L/s  (T-value = {result['ar1_t_value']:+.2f})"
        )
        print(f"  slope 95% CI       = +/- {ar1_95ci:.4e} mg/L/s")
        print()
        print(f"  slope (per day)    = {slope_day:+.4e} +/- {ar1_se_day:.4e} mg/L/day")
        print(
            f"  uncorrected (OLS)  = +/- {slope_se_day:.4e} mg/L/day "
            f"(T-value = {result['z_score']:+.2f})"
        )
    else:
        print("Linear fit: rho(t) = intercept + slope * t")
        print(
            f"  slope              = {slope_s:+.4e} +/- {slope_se_s:.4e} "
            f"mg/L/s  (T-value = {result['z_score']:+.2f})"
        )
        print(f"  slope 95% CI       = +/- {slope_95ci:.4e} mg/L/s")
        print()
        print(f"  slope (per day)    = {slope_day:+.4e} +/- {slope_se_day:.4e} mg/L/day")
    print()
    print(f"  intercept          = {intercept:.6f} mg/L")
    print(f"  intercept 1-sigma  = {intercept_se:.4e} mg/L")
    print("  (|T-value| > ~3 suggests a real trend)")

    if args.sinusoid:
        sin_slope_day = result["sin_slope_per_day"] * M_AIR
        sin_se_day = result["sin_slope_stderr_per_day"] * M_AIR
        sin_amp = result["sin_amplitude"] * M_AIR
        print()
        if args.ar1:
            sin_ar1_se_day = result["sin_ar1_slope_stderr_per_day"] * M_AIR
            print(f"Linear + {args.sinusoid_period:g}-hour sinusoid fit (AR(1)-corrected stderr):")
            print(
                f"  AR(1) details      : rho_1 = {result['sin_ar1_rho1']:+.4f}, "
                f"n_eff = {result['sin_ar1_n_eff']:.0f}, "
                f"factor = {result['sin_ar1_factor']:.3f}"
            )
            print(
                f"  slope (per day)    = {sin_slope_day:+.4e} +/- {sin_ar1_se_day:.4e} mg/L/day "
                f"(T-value = {result['sin_ar1_t_value']:+.2f})"
            )
            print(
                f"  uncorrected (OLS)  = +/- {sin_se_day:.4e} mg/L/day "
                f"(T-value = {result['sin_t_value']:+.2f})"
            )
        else:
            print(f"Linear + {args.sinusoid_period:g}-hour sinusoid fit:")
            print(
                f"  slope (per day)    = {sin_slope_day:+.4e} +/- {sin_se_day:.4e} mg/L/day "
                f"(T-value = {result['sin_t_value']:+.2f})"
            )
        print(f"  sinusoid amplitude = {sin_amp:.4f} mg/L")

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        from datetime import UTC, datetime

        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        from slocum_tpw.simulate_leak import R

        P_abs = P_ATM_PA - vacuum * INHG_TO_PA
        T_K = temp + 273.15
        rho_ideal = (P_abs * M_AIR) / (R * T_K)
        rho_vdw = vdw_density_vec(P_abs, T_K) * M_AIR

        ok = np.isfinite(rho_vdw)
        t_days = (t - t[0]) / 86400.0

        def _ols(x, y):
            n = x.size
            xm = x.mean()
            ym = y.mean()
            Sxx = float(np.sum((x - xm) ** 2))
            slope = float(np.sum((x - xm) * (y - ym)) / Sxx)
            intercept = float(ym - slope * xm)
            resid = y - (intercept + slope * x)
            s2 = float(np.sum(resid**2) / (n - 2))
            stderr = float(np.sqrt(s2 / Sxx))
            return intercept, slope, stderr, slope / stderr, resid

        a_i, b_i, se_i, t_i, resid_i = _ols(t_days, rho_ideal)
        a_w, b_w, se_w, t_w, resid_w = _ols(t_days[ok], rho_vdw[ok])

        sin_period_d = (args.sinusoid_period / 24.0) if args.sinusoid else None

        def _sin_fit(x, y):
            return _fit_linear_sinusoid(x, y, sin_period_d)

        # When --ar1, swap displayed stderr/T to the AR(1)-corrected ones so
        # the legend reflects the same "primary" numbers as the printed output.
        if args.ar1:
            f_i = _ar1_correction(resid_i)["factor"]
            f_w = _ar1_correction(resid_w)["factor"]
            disp_se_i, disp_t_i = se_i * f_i, b_i / (se_i * f_i)
            disp_se_w, disp_t_w = se_w * f_w, b_w / (se_w * f_w)
        else:
            disp_se_i, disp_t_i = se_i, t_i
            disp_se_w, disp_t_w = se_w, t_w
        if args.sinusoid:
            sin_i = _sin_fit(t_days, rho_ideal)
            sin_w = _sin_fit(t_days[ok], rho_vdw[ok])
            if args.ar1:
                sf_i = _ar1_correction(sin_i["residuals"])["factor"]
                sf_w = _ar1_correction(sin_w["residuals"])["factor"]
                disp_sin_se_i = sin_i["slope_stderr"] * sf_i
                disp_sin_se_w = sin_w["slope_stderr"] * sf_w
            else:
                disp_sin_se_i = sin_i["slope_stderr"]
                disp_sin_se_w = sin_w["slope_stderr"]
            disp_sin_t_i = sin_i["slope"] / disp_sin_se_i
            disp_sin_t_w = sin_w["slope"] / disp_sin_se_w

        kind = "AR(1)" if args.ar1 else "OLS"

        def _label(name, b, se, t_, sin_b, sin_se, sin_t):
            head = f"{name} fit: {b:+.3f} +/- {se:.3f} mg/L/day  (T_{kind}={t_:+.1f})"
            if sin_b is not None:
                head += f"; +sin: {sin_b:+.3f} +/- {sin_se:.3f}  (T_{kind}={sin_t:+.1f})"
            return head

        lab_i = _label(
            "ideal",
            b_i,
            disp_se_i,
            disp_t_i,
            sin_i["slope"] if args.sinusoid else None,
            disp_sin_se_i if args.sinusoid else None,
            disp_sin_t_i if args.sinusoid else None,
        )
        lab_w = _label(
            "vdW  ",
            b_w,
            disp_se_w,
            disp_t_w,
            sin_w["slope"] if args.sinusoid else None,
            disp_sin_se_w if args.sinusoid else None,
            disp_sin_t_w if args.sinusoid else None,
        )

        t_dt = np.array([datetime.fromtimestamp(s, tz=UTC) for s in t])
        t_num = mdates.date2num(t_dt)

        fig = plt.figure(figsize=(13, 7))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.1, 1.4],
            height_ratios=[1, 1],
            wspace=0.28,
            hspace=0.18,
        )
        ax_scatter = fig.add_subplot(gs[:, 0])
        ax_T = fig.add_subplot(gs[0, 1])
        ax_V = ax_T.twinx()
        ax_rho = fig.add_subplot(gs[1, 1], sharex=ax_T)

        sc = ax_scatter.scatter(vacuum, temp, c=t_num, cmap="viridis", s=6, alpha=0.7)
        ax_scatter.set_xlabel("m_vacuum (inHg)")
        ax_scatter.set_ylabel("m_veh_temp (deg C)")
        ax_scatter.set_title("m_veh_temp vs m_vacuum")
        ax_scatter.grid(True, alpha=0.3)
        cb = fig.colorbar(sc, ax=ax_scatter, location="bottom", pad=0.10, fraction=0.05)
        cb.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        cb.ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        cb.set_label("time (UTC)")
        for lbl in cb.ax.get_xticklabels():
            lbl.set_rotation(20)
            lbl.set_ha("right")

        ax_T.plot(t_dt, temp, ".", ms=2, alpha=0.6, color="C3")
        ax_T.set_ylabel("m_veh_temp (deg C)", color="C3")
        ax_T.tick_params(axis="y", colors="C3")
        ax_T.invert_yaxis()
        ax_T.grid(True, alpha=0.3)
        ax_T.set_title("time series")
        plt.setp(ax_T.get_xticklabels(), visible=False)
        ax_V.plot(t_dt, vacuum, ".", ms=2, alpha=0.6, color="C0")
        ax_V.set_ylabel("m_vacuum (inHg)", color="C0")
        ax_V.tick_params(axis="y", colors="C0")

        ax_rho.plot(t_dt, rho_ideal, ".", ms=2, alpha=0.5, color="C2")
        ax_rho.plot(t_dt[ok], rho_vdw[ok], ".", ms=2, alpha=0.5, color="C1")
        ax_rho.plot(t_dt, a_i + b_i * t_days, "-", lw=1.8, color="magenta", label=lab_i)
        ax_rho.plot(t_dt[ok], a_w + b_w * t_days[ok], "-", lw=1.8, color="cyan", label=lab_w)
        ax_rho.set_ylabel("density (mg/L)")
        ax_rho.set_xlabel("time (UTC)")
        ax_rho.grid(True, alpha=0.3)
        ax_rho.legend(loc="best", framealpha=0.9, fontsize=9)
        ax_rho.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_rho.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        for lbl in ax_rho.get_xticklabels():
            lbl.set_rotation(20)
            lbl.set_ha("right")

        fig.suptitle(f"Leak fit: {args.input_file}", y=0.995)
        fig.savefig(args.plot, dpi=140)
        print(f"  plot written to    : {args.plot}")

    return 0
