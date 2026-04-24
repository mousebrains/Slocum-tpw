#
# Simulate sealed-body vacuum and temperature observations for a Slocum Glider
# under a sinusoidal thermal cycle, optionally with a constant-rate leak.
#
# Uses the van der Waals equation of state for air.  Writes a CSV with columns
# (m_present_time, m_vacuum, m_veh_temp) in Slocum native units
# (seconds, inHg, degC).
#
# Pat Welch, pat@mousebrains.com

import argparse
import csv
import logging

import numpy as np
from scipy.optimize import brentq

# Universal gas constant, J/(mol * K)
R = 8.31446

# Van der Waals constants for air
A_VDW = 0.1358  # Pa * m^6 / mol^2
B_VDW = 3.64e-5  # m^3 / mol

# Unit conversions
INHG_TO_PA = 3386.389
P_ATM_INHG = 29.9213
P_ATM_PA = P_ATM_INHG * INHG_TO_PA


def vdw_pressure(rho, T):
    """Van der Waals absolute pressure (Pa) from molar density rho (mol/m^3) and T (K)."""
    return rho * R * T / (1.0 - rho * B_VDW) - A_VDW * rho * rho


def vdw_density(P, T):
    """Invert van der Waals: return rho (mol/m^3) given absolute P (Pa) and T (K)."""
    rho_ideal = P / (R * T)
    return brentq(
        lambda r: vdw_pressure(r, T) - P,
        0.5 * rho_ideal,
        2.0 * rho_ideal,
        xtol=1e-14,
        rtol=1e-12,
    )


def simulate(
    days: float,
    timestep: float,
    vacuum_drop_per_day: float = 0.0,
    initial_vacuum: float = 10.0,
    volume_l: float = 50.0,
    temp_mean_c: float = 17.5,
    temp_amplitude_c: float = 2.5,
    temp_period_hours: float = 24.0,
    sigma_pressure: float = 0.001,
    sigma_temperature: float = 0.1,
    seed: int | None = None,
    t0_epoch: float = 0.0,
) -> dict:
    """Simulate sealed-body observations.

    Returns a dict with keys:

    - ``time``: seconds since *t0_epoch* (offset by *t0_epoch* so absolute
      timestamps are also supported)
    - ``vacuum_inHg``: noisy vacuum observations (inHg)
    - ``temperature_c``: noisy vehicle temperature observations (degC)
    - ``vacuum_true_inHg``, ``temperature_true_c``: underlying noise-free signals
    - ``drho_dt_true``: constant leak rate implied by *vacuum_drop_per_day*
      evaluated at the reference (starting) temperature, in mol/(m^3 * s)
    - ``rho0``: initial molar density (mol/m^3)

    Positive *vacuum_drop_per_day* means vacuum is decreasing (gas leaking in).
    """
    V = volume_l * 1e-3
    duration_s = days * 86400.0
    period_s = temp_period_hours * 3600.0

    # Initial state: t=0 is at the thermal maximum (T_mean + T_amp).
    T0_K = (temp_mean_c + temp_amplitude_c) + 273.15
    P0_Pa = P_ATM_PA - initial_vacuum * INHG_TO_PA
    rho0 = vdw_density(P0_Pa, T0_K)

    # Implied constant d(rho)/dt that produces the requested vacuum drift
    # per day, measured at T = T0.
    total_drop_inHg = vacuum_drop_per_day * days
    P_end_Pa = P0_Pa + total_drop_inHg * INHG_TO_PA
    rho_end = vdw_density(P_end_Pa, T0_K)
    drho_dt_true = (rho_end - rho0) / duration_s if duration_s > 0 else 0.0

    # Uniform time grid, inclusive of endpoint.
    t = np.arange(0.0, duration_s + timestep, timestep)

    omega = 2.0 * np.pi / period_s
    T_true_c = temp_mean_c + temp_amplitude_c * np.cos(omega * t)
    T_true_k = T_true_c + 273.15

    rho_true = rho0 + drho_dt_true * t
    P_true_pa = vdw_pressure(rho_true, T_true_k)
    vacuum_true_inHg = (P_ATM_PA - P_true_pa) / INHG_TO_PA

    rng = np.random.default_rng(seed)
    vacuum_obs = vacuum_true_inHg + rng.normal(0.0, sigma_pressure, size=t.size)
    temp_obs = T_true_c + rng.normal(0.0, sigma_temperature, size=t.size)

    return {
        "time": t + t0_epoch,
        "vacuum_inHg": vacuum_obs,
        "temperature_c": temp_obs,
        "vacuum_true_inHg": vacuum_true_inHg,
        "temperature_true_c": T_true_c,
        "drho_dt_true": drho_dt_true,
        "rho0": rho0,
        "volume_m3": V,
    }


def write_csv(path: str, time_s, vacuum_inHg, temperature_c) -> None:
    """Write simulated observations to CSV with Slocum native column names."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m_present_time", "m_vacuum", "m_veh_temp"])
        for ti, vi, Ti in zip(time_s, vacuum_inHg, temperature_c):
            w.writerow([f"{ti:.3f}", f"{vi:.6f}", f"{Ti:.4f}"])


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add simulate-leak arguments to the parser."""
    parser.add_argument(
        "-o", "--output", type=str, default="simulated.csv",
        help="Output CSV path (default: simulated.csv)",
    )
    parser.add_argument(
        "--days", type=float, default=4.0,
        help="Simulation duration in days (default: 4)",
    )
    parser.add_argument(
        "--timestep", type=float, default=3.0,
        help="Sampling interval in seconds (default: 3)",
    )
    parser.add_argument(
        "--vacuum-drop-per-day", type=float, default=0.0,
        help=(
            "Signed leak rate in inHg/day of vacuum DROP.  Positive = vacuum "
            "decreasing (gas leaking IN).  Negative = vacuum increasing (gas "
            "leaking OUT).  Default: 0 (no leak)."
        ),
    )
    parser.add_argument(
        "--sigma-pressure", type=float, default=0.001,
        help="1-sigma Gaussian noise on vacuum, inHg (default: 0.001)",
    )
    parser.add_argument(
        "--sigma-temp", type=float, default=0.1,
        help="1-sigma Gaussian noise on temperature, degC (default: 0.1)",
    )
    parser.add_argument(
        "--initial-vacuum", type=float, default=10.0,
        help="Initial vacuum at t=0, inHg (default: 10)",
    )
    parser.add_argument(
        "--volume", type=float, default=50.0,
        help="Sealed gas volume, liters (default: 50)",
    )
    parser.add_argument(
        "--temp-mean", type=float, default=17.5,
        help="Mean air temperature, degC (default: 17.5)",
    )
    parser.add_argument(
        "--temp-amplitude", type=float, default=2.5,
        help="Thermal amplitude, degC (default: 2.5)",
    )
    parser.add_argument(
        "--temp-period-hours", type=float, default=24.0,
        help="Thermal cycle period, hours (default: 24)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--t0-epoch", type=float, default=0.0,
        help=(
            "Seconds at t=0 (default: 0); set to a Unix epoch time to get "
            "absolute timestamps in the output CSV."
        ),
    )


def run(args: argparse.Namespace) -> int:
    """Execute the simulate-leak command."""
    result = simulate(
        days=args.days,
        timestep=args.timestep,
        vacuum_drop_per_day=args.vacuum_drop_per_day,
        initial_vacuum=args.initial_vacuum,
        volume_l=args.volume,
        temp_mean_c=args.temp_mean,
        temp_amplitude_c=args.temp_amplitude,
        temp_period_hours=args.temp_period_hours,
        sigma_pressure=args.sigma_pressure,
        sigma_temperature=args.sigma_temp,
        seed=args.seed,
        t0_epoch=args.t0_epoch,
    )
    write_csv(args.output, result["time"], result["vacuum_inHg"], result["temperature_c"])

    n = result["time"].size
    total_drop = args.vacuum_drop_per_day * args.days
    logging.info(
        "wrote %d rows to %s (days=%.4f dt=%.3fs drop=%+.4f inHg "
        "truth d(n/V)/dt=%+.4e mol/m^3/s)",
        n, args.output, args.days, args.timestep, total_drop, result["drho_dt_true"],
    )
    print(f"Wrote {n} rows to {args.output}")
    print(f"  duration            : {args.days:.4f} days")
    print(f"  timestep            : {args.timestep:.4f} s")
    print(f"  initial vacuum      : {args.initial_vacuum:.4f} inHg")
    print(f"  sigma(pressure)     : {args.sigma_pressure:.4e} inHg")
    print(f"  sigma(temperature)  : {args.sigma_temp:.4e} degC")
    print(
        f"  vacuum drop / day   : {args.vacuum_drop_per_day:+.6f} inHg/day "
        f"(total {total_drop:+.4f} inHg over run)"
    )
    print(f"  initial n/V         : {result['rho0']:.6f} mol/m^3")
    print(
        f"  truth d(n/V)/dt     : {result['drho_dt_true']:+.4e} mol/(m^3 * s) "
        f"= {result['drho_dt_true'] * 86400.0:+.4e} mol/(m^3 * day)"
    )
    return 0
