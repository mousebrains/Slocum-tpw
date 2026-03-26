#
# Estimate when a Slocum glider will need to be recovered
# based on battery percentage decay.
#
# There are a lot of assumptions here, such as constant in time usage.
# If your operation is changing modes, please don't use this, or
# set the start/end times appropriately.
#
# Jan-2025, Pat Welch

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import t

S_PER_DAY = 86400  # seconds per day
ONE_DAY = np.timedelta64(1, "D")
FIT_COLORS = ["tab:green", "tab:orange", "tab:purple", "tab:red", "tab:brown", "tab:pink"]


def _safe_sqrt(x):
    """sqrt that propagates NaN and clamps negative values to 0."""
    if np.isnan(x):
        return float("nan")
    return float(np.sqrt(max(0, x)))


def prepare_dataset(source, time_var="time", sensor="m_lithium_battery_relative_charge"):
    """Load and clean a dataset for recovery fitting.

    Parameters
    ----------
    source : str, Path, or xr.Dataset
        NetCDF filename or an already-opened Dataset.
    time_var : str
        Name of the time variable in the dataset.
    sensor : str
        Name of the battery sensor variable.

    Returns
    -------
    xr.Dataset
        Cleaned dataset with 'time' dimension coordinate and sensor variable,
        sorted by time with duplicates and NaN sensor values removed.

    Raises
    ------
    KeyError
        If required variables are not found in the dataset.
    """
    if isinstance(source, (str, Path)):
        ds = xr.open_dataset(source)
    else:
        ds = source.copy()

    for name in (time_var, sensor):
        if name not in ds:
            raise KeyError(f"Variable '{name}' not found in dataset")

    ds = ds.drop_vars(set(ds) - {time_var, sensor})

    # Get time values, converting float epoch seconds to datetime64
    time_vals = ds[time_var].values
    needs_reassign = False

    if time_vals.dtype.kind == "f":
        time_vals = time_vals.astype("datetime64[s]")
        needs_reassign = True
    elif time_var != "time" or "time" not in ds.dims:
        needs_reassign = True

    if needs_reassign:
        sensor_dims = ds[sensor].dims
        dim_name = sensor_dims[0] if sensor_dims else next(iter(ds.dims))
        logging.debug("Setting time coordinate from %s on dimension %s", time_var, dim_name)
        ds = ds.assign_coords(time=(dim_name, time_vals))
        if dim_name != "time":
            ds = ds.swap_dims({dim_name: "time"})

    if time_var != "time" and time_var in ds:
        ds = ds.drop_vars(time_var)

    ds = ds.drop_duplicates("time", keep="first")
    ds = ds.sel(time=ds.time[np.logical_not(ds[sensor].isnull())])
    ds = ds.sortby("time")

    return ds.load()


def fit_recovery(
    ds,
    sensor="m_lithium_battery_relative_charge",
    threshold=15,
    confidence=0.95,
    ndays=None,
    tau=None,
    start=None,
    stop=None,
):
    """Fit a linear model to battery data and estimate recovery date.

    Parameters
    ----------
    ds : xr.Dataset
        Cleaned dataset from prepare_dataset().
    sensor : str
        Name of the battery sensor variable.
    threshold : float
        Battery percentage at which recovery should happen.
    confidence : float
        Confidence level for intervals (0 < confidence < 1).
    ndays : float or None
        If set, only use the last ndays days of data.
    tau : float or None
        If set, apply exponential downweighting to older data.
        Each point is weighted by exp(-age/tau) where age is in days
        from the most recent observation.
    start : str or None
        If set, only use data after this UTC time.
    stop : str or None
        If set, only use data before this UTC time.

    Returns
    -------
    dict or None
        Fit results dict with keys: time, sensor_values, dDays, slope,
        intercept, slope_ci, intercept_ci, recovery_date, recovery_ci_days,
        r_squared, pvalue, n_points, threshold, confidence, ndays, tau.
        Returns None if the fit fails (insufficient data or near-zero slope).
    """
    if ndays is not None:
        etime = ds.time[-1]
        stime = etime - np.timedelta64(int(ndays * S_PER_DAY), "s")
        ds = ds.sel(time=slice(stime, etime))
    elif start is not None or stop is not None:
        stime = np.datetime64(start) if start is not None else ds.time[0]
        etime = np.datetime64(stop) if stop is not None else ds.time[-1]
        ds = ds.sel(time=slice(stime, etime))

    if ds.time.size < 3:
        logging.warning("Not enough data to fit (%d points, need >= 3)", ds.time.size)
        return None

    dDays = (ds.time.data - ds.time.data[0]) / ONE_DAY

    # Exponential downweighting: polyfit squares its w argument,
    # so pass sqrt of the desired effective weight.
    if tau is not None:
        age = dDays[-1] - dDays  # days from newest (0 for latest)
        fit_weights = np.exp(-age / (2 * tau))
    else:
        fit_weights = None

    try:
        coeffs, cov = np.polyfit(dDays, ds[sensor], 1, cov=True, w=fit_weights)
    except (np.linalg.LinAlgError, ValueError) as e:
        logging.warning("Fit failed: %s", e)
        return None

    slope, intercept = coeffs

    if abs(slope) < 1e-10:
        logging.warning("Near-zero slope — cannot estimate recovery date")
        return None

    d_recovery = (threshold - intercept) / slope
    t_recover_by = ds.time[0].data + np.timedelta64(round(d_recovery * S_PER_DAY), "s")
    # Round to nearest hour
    t_recover_by = (t_recover_by + np.timedelta64(30, "m")).astype("datetime64[h]")

    if d_recovery < 0:
        logging.warning("Recovery date is in the past (positive slope — battery increasing?)")

    if not np.all(np.isfinite(cov)):
        logging.warning("Unstable fit — confidence intervals may be unreliable")

    alpha = 1 - confidence

    # Propagate uncertainty including covariance between slope and intercept
    # d_recovery = (threshold - intercept) / slope
    # ∂d/∂intercept = -1/slope, ∂d/∂slope = -d_recovery/slope
    var_recovery = (cov[1, 1] + d_recovery**2 * cov[0, 0] + 2 * d_recovery * cov[0, 1]) / slope**2
    sigma_recovery = _safe_sqrt(var_recovery)
    sigma_intercept = _safe_sqrt(cov[1, 1])
    sigma_slope = _safe_sqrt(cov[0, 0])

    # R-squared (weighted if tau is set)
    y_pred = intercept + slope * dDays
    if tau is not None:
        eff_w = np.exp(-age / tau)  # effective weights
        w_mean = float(np.average(ds[sensor], weights=eff_w))
        ss_res = np.sum(eff_w * (ds[sensor] - y_pred) ** 2).item()
        ss_tot = np.sum(eff_w * (ds[sensor] - w_mean) ** 2).item()
    else:
        ss_res = np.sum((ds[sensor] - y_pred) ** 2).item()
        ss_tot = np.sum((ds[sensor] - ds[sensor].mean()) ** 2).item()
    if ss_tot == 0:
        r_squared = float("nan")
    else:
        r_squared = 1 - ss_res / ss_tot

    # p-value for slope
    n = dDays.size
    df = n - 2
    if sigma_slope > 0:
        t_stat = slope / sigma_slope
        pvalue = 2 * (1 - t.cdf(abs(t_stat), df))
    else:
        pvalue = float("nan")

    # Confidence intervals
    ts = abs(t.ppf(alpha / 2, df))
    ci_intercept = sigma_intercept * ts
    ci_slope = sigma_slope * ts
    ci_recovery = sigma_recovery * ts

    return {
        "time": ds.time,
        "sensor_values": ds[sensor],
        "dDays": dDays,
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_ci": float(ci_slope) if np.isfinite(ci_slope) else None,
        "intercept_ci": float(ci_intercept) if np.isfinite(ci_intercept) else None,
        "recovery_date": t_recover_by,
        "recovery_ci_days": float(ci_recovery) if np.isfinite(ci_recovery) else None,
        "r_squared": float(r_squared) if np.isfinite(r_squared) else None,
        "pvalue": float(pvalue) if np.isfinite(pvalue) else None,
        "n_points": int(n),
        "threshold": threshold,
        "confidence": confidence,
        "ndays": ndays,
        "tau": tau,
    }


def _parse_float_list(args):
    """Parse repeated and/or comma-separated float arguments.

    Returns None if args is None or empty.
    """
    if not args:
        return None
    result = []
    for arg in args:
        for part in arg.split(","):
            part = part.strip()
            if part:
                result.append(float(part))
    return result if result else None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add recover-by arguments to the parser."""
    parser.add_argument(
        "filename",
        type=str,
        nargs="+",
        help="Input NetCDF file(s) containing time and sensor variables",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="m_lithium_battery_relative_charge",
        help="Sensor name to fit to",
    )
    parser.add_argument(
        "--ndays",
        type=str,
        action="append",
        help="Days from last date to include (repeatable, comma-separated)",
    )
    parser.add_argument(
        "--tau",
        type=str,
        action="append",
        help="Exponential decay time constant in days — full dataset weighted by "
        "exp(-age/tau) (repeatable, comma-separated)",
    )
    parser.add_argument("--start", type=str, help="Only use data after this UTC time")
    parser.add_argument(
        "--stop",
        type=str,
        help="Only use data before this UTC time",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15,
        help="Battery percentage at which recovery should happen",
    )
    parser.add_argument("--time", type=str, default="time", help="Name of time variable")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95)",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Output results as JSON"
    )
    parser.add_argument("--plot", action="store_true", help="Generate a plot")
    parser.add_argument("--output", type=str, help="Save plot to file instead of displaying")


def run(args: argparse.Namespace) -> int:
    """Execute the recover-by command."""
    ndays_list = _parse_float_list(args.ndays)
    tau_list = _parse_float_list(args.tau)

    has_windows = ndays_list is not None or tau_list is not None
    if has_windows and args.start is not None:
        logging.error("--ndays/--tau and --start cannot be used together")
        return 2
    if has_windows and args.stop is not None:
        logging.error("--ndays/--tau and --stop cannot be used together")
        return 2
    if not 0 < args.confidence < 1:
        logging.error("--confidence must be between 0 and 1")
        return 2

    ci_pct = f"{args.confidence * 100:g}"

    # Build list of (ndays, tau, start, stop) windows to fit
    windows = []
    if ndays_list:
        windows.extend((nd, None, None, None) for nd in ndays_list)
    if tau_list:
        windows.extend((None, tau, None, None) for tau in tau_list)
    if not windows:
        windows.append((None, None, args.start, args.stop))

    multi_window = len(windows) > 1

    if args.plot or args.output:
        import matplotlib

        if args.output and not args.plot:
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(len(args.filename), 1, sharex=True, squeeze=False)
        fig.subplots_adjust(hspace=0)

    success = False
    results = []
    plotted_indices = set()

    for index, fn in enumerate(args.filename):
        try:
            ds = prepare_dataset(fn, time_var=args.time, sensor=args.sensor)
        except KeyError as e:
            logging.error("%s: %s", fn, e)
            continue
        except (OSError, ValueError) as e:
            logging.error("Failed to process %s: %s", fn, e)
            continue

        if ds.time.size < 3:
            logging.error("Not enough data in %s (%d points, need >= 3)", fn, ds.time.size)
            continue

        for win_idx, (ndays, tau, start, stop) in enumerate(windows):
            result = fit_recovery(
                ds,
                sensor=args.sensor,
                threshold=args.threshold,
                confidence=args.confidence,
                ndays=ndays,
                tau=tau,
                start=start,
                stop=stop,
            )
            if result is None:
                if ndays is not None:
                    win_str = f" (ndays={ndays})"
                elif tau is not None:
                    win_str = f" (\u03c4={tau})"
                else:
                    win_str = ""
                logging.warning("Fit failed for %s%s", fn, win_str)
                continue

            success = True
            r = result

            if not args.json_output:
                if multi_window:
                    if ndays is not None:
                        label = f"{ndays:g}d"
                    elif tau is not None:
                        label = f"\u03c4={tau:g}d"
                    else:
                        label = "full"
                    print(f"\n{fn} [{label}]")
                else:
                    print(f"\n{fn}")
                print(f"Sensor:            {args.sensor}")
                print(f"Sensor threshold:  {args.threshold}")
                ci_i = r["intercept_ci"] if r["intercept_ci"] is not None else float("nan")
                ci_s = r["slope_ci"] if r["slope_ci"] is not None else float("nan")
                ci_r = r["recovery_ci_days"] if r["recovery_ci_days"] is not None else float("nan")
                print(f"Intercept ({ci_pct}%):   {r['intercept']:.4f}+-{ci_i:.4f}")
                print(f"Slope ({ci_pct}%, /day):  {r['slope']:.4f}+-{ci_s:.4f}")
                r_sq = r["r_squared"] if r["r_squared"] is not None else float("nan")
                pv = r["pvalue"] if r["pvalue"] is not None else float("nan")
                print(f"R-squared:         {r_sq:.4f}")
                print(f"Pvalue:            {pv:.4f}")
                recover_str = str(r["recovery_date"]) + ":00"
                print(f"Recovery By ({ci_pct}%): {recover_str}+-{ci_r:.2f} (days)")

            results.append(
                {
                    "file": fn,
                    "sensor": args.sensor,
                    "threshold": args.threshold,
                    "confidence": args.confidence,
                    "ndays": ndays,
                    "tau": tau,
                    "n_points": r["n_points"],
                    "intercept": r["intercept"],
                    "intercept_ci": r["intercept_ci"],
                    "slope": r["slope"],
                    "slope_ci": r["slope_ci"],
                    "r_squared": r["r_squared"],
                    "pvalue": r["pvalue"],
                    "recovery_date": str(r["recovery_date"]),
                    "recovery_ci_days": r["recovery_ci_days"],
                }
            )

            if args.plot or args.output:
                ax = axs[index, 0]
                plotted_indices.add(index)

                if win_idx == 0:
                    # Plot raw data once per file
                    if multi_window:
                        ax.plot(
                            ds.time,
                            ds[args.sensor],
                            ".",
                            color="tab:blue",
                            markersize=3,
                            alpha=0.5,
                            label="data",
                            zorder=1,
                        )
                    else:
                        ax.plot(
                            ds.time,
                            ds[args.sensor],
                            "o",
                            label=Path(fn).name,
                        )
                    ax.axhline(y=args.threshold, color="gray", linestyle="--", alpha=0.5)

                slope = r["slope"]
                intercept = r["intercept"]
                abs_slope = abs(slope)
                sign = "-" if slope < 0 else "+"

                if multi_window:
                    color = FIT_COLORS[win_idx % len(FIT_COLORS)]
                    if ndays is not None:
                        ndays_label = f"{ndays:g}d"
                    elif tau is not None:
                        ndays_label = f"\u03c4={tau:g}d"
                    else:
                        ndays_label = "full"
                    r_sq = r["r_squared"]
                    r_sq_str = f", R\u00b2={r_sq:.3f}" if r_sq is not None else ""
                    ci_r = r["recovery_ci_days"]
                    ci_str = f"\u00b1{ci_r:.1f}d" if ci_r is not None else ""
                    fit_label = (
                        f"{ndays_label}: {intercept:.1f}{sign}{abs_slope:.2f}/day"
                        f"{r_sq_str}\n"
                        f"  recover {r['recovery_date']}{ci_str}"
                        f" (n={r['n_points']})"
                    )
                else:
                    color = "r"
                    fit_label = f"{intercept:.1f}{sign}{abs_slope:.2f} * days"
                    fit_label += f"\nRecovery by {r['recovery_date']}"

                time_vals = r["time"]
                dDays = r["dDays"]
                ax.plot(
                    time_vals,
                    intercept + slope * dDays,
                    color=color,
                    linewidth=1.5,
                    label=fit_label,
                    zorder=2,
                )

                # Extend fit line to recovery date
                if r["recovery_date"] > time_vals[-1].values:
                    last_val = float(intercept + slope * dDays[-1].item())
                    ax.plot(
                        [time_vals[-1].values, r["recovery_date"]],
                        [last_val, args.threshold],
                        color=color,
                        linestyle="--",
                        alpha=0.5,
                        linewidth=1.5,
                        zorder=2,
                    )

                ax.set_ylabel(args.sensor)
                if multi_window:
                    ax.legend(fontsize="x-small", loc="best")
                else:
                    ax.legend()
                ax.grid(True, alpha=0.3)

    if args.json_output:
        print(json.dumps(results, indent=2))

    if args.plot or args.output:
        # Remove blank subplots for files that failed
        for i in range(len(args.filename)):
            if i not in plotted_indices:
                fig.delaxes(axs[i, 0])
        if plotted_indices:
            ax = fig.axes[-1]
            ax.set_xlabel("Time (UTC)")
            plt.title(f"{args.sensor} threshold {args.threshold}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            if args.output:
                plt.savefig(args.output)
                logging.info("Plot saved to %s", args.output)
            else:
                plt.show()
        plt.close(fig)

    return 0 if success else 1
