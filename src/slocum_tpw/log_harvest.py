#
# Harvest records from Slocum glider log files and save to NetCDF
#
# June-2023, Pat Welch, pat@mousebrains.com

import argparse
import datetime
import logging
import math
import os.path
import re

import numpy as np
import pandas as pd
import xarray as xr

from slocum_tpw.slocum_utils import mk_degrees_scalar

_RE_VEHICLE = re.compile(r"^Vehicle Name:\s+(\w+)$")
_RE_CURRTIME = re.compile(r"^Curr Time:\s+\w+\s+(\w+\s+\d+\s+\d+:\d+:\d+\s+\d+)\s+MT:\s+(\d+)")
_RE_GPS = re.compile(
    r"^GPS Location:\s+(-?\d+[.]\d+)\s+N\s+(-?\d+[.]\d+)\s+E\s+"
    r"measured\s+([+-]?\d+[.]?\d*[e]?[+-]?\d*) secs ago$"
)
_RE_SENSOR = re.compile(
    r"^sensor:(\w+)[(](.+)[)]=([+-]?\d+[.]?\d*[e]?[+-]?\d*)\s+"
    r"(\d+[.]?\d*[e]?[+-]?\d*) secs ago$"
)


def parse_log_file(fn: str, glider: str) -> pd.DataFrame:
    """Parse a single Slocum log file and return a DataFrame of binned observations."""
    logging.debug("Loading %s %s", fn, glider)
    info: dict[str, list] = {}
    times: list[float] = []

    with open(fn, "rb") as fp:
        currTime = None
        for line in fp:
            line = line.strip()
            try:
                line = str(line, "utf-8")
            except UnicodeDecodeError:
                continue

            matches = _RE_VEHICLE.match(line)
            if matches:
                glider = matches[1]

            matches = _RE_CURRTIME.match(line)
            if matches:
                try:
                    currTime = (
                        datetime.datetime.strptime(matches[1], "%b %d %H:%M:%S %Y")
                        .replace(tzinfo=datetime.timezone.utc)
                        .timestamp()
                    )
                except ValueError:
                    logging.warning("Invalid timestamp in %s: %s", fn, matches[1])
                    continue
                continue

            matches = _RE_GPS.match(line)
            if matches:
                if currTime is None:
                    continue
                try:
                    lat = mk_degrees_scalar(float(matches[1]))
                    lon = mk_degrees_scalar(float(matches[2]))
                    dt = float(matches[3])
                except ValueError:
                    continue
                if (
                    dt > 1e300
                    or math.isnan(lat)
                    or math.isnan(lon)
                    or abs(lat) > 90
                    or abs(lon) > 180
                ):
                    continue
                t = currTime - dt
                times.append(t)
                if "GPS" not in info:
                    info["GPS"] = []
                info["GPS"].append((t, lat, lon))
                continue

            matches = _RE_SENSOR.match(line)
            if matches:
                if currTime is None:
                    continue
                name = matches[1]
                try:
                    val = float(matches[3])
                    dt = float(matches[4])
                except ValueError:
                    continue
                if name.endswith("_lat") or name.endswith("_lon"):
                    val = mk_degrees_scalar(val)
                if dt > 1e300:
                    continue
                if name not in info:
                    info[name] = []
                t = currTime - dt
                times.append(t)
                info[name].append((t, val))

    if not times:
        return pd.DataFrame()

    # Build time grid with 100-second bins
    time_bins = np.unique(np.round(np.array(times), -2))
    n = len(time_bins)

    # Pre-build lookup: rounded_time -> index
    time_to_idx = {t: i for i, t in enumerate(time_bins)}

    # Allocate numpy arrays for all columns
    columns: dict[str, np.ndarray] = {"t": time_bins, "glider": np.array([glider] * n)}
    for key in sorted(info):
        if key == "GPS":
            columns["lat"] = np.full(n, np.nan)
            columns["lon"] = np.full(n, np.nan)
        else:
            columns[key] = np.full(n, np.nan)

    # Fill arrays using dict lookup instead of argmin
    for key in sorted(info):
        for row in info[key]:
            rounded = float(np.round(row[0], -2))
            idx = time_to_idx.get(rounded)
            if idx is None:
                # Fallback: find nearest bin
                idx = np.argmin(np.abs(time_bins - rounded))
            if key == "GPS":
                columns["lat"][idx] = row[1]
                columns["lon"][idx] = row[2]
            else:
                columns[key][idx] = row[1]

    df = pd.DataFrame(columns)
    df["t"] = df["t"].astype("datetime64[s]")
    return df


def process_files(filenames: list[str], t0: str | None, nc: str) -> None:
    """Process multiple log files, filter by t0, and write to NetCDF."""
    items = []
    for fn in sorted(filenames):
        fields = os.path.basename(fn).split("_")
        if len(fields) < 2:
            logging.warning("Skipping %s: filename does not match {glider}_{timestamp}_*.log", fn)
            continue
        if t0 and fields[1] < t0:
            continue
        glider = fields[0]
        try:
            a = parse_log_file(fn, glider)
        except OSError as e:
            logging.error("Failed to read %s: %s", fn, e)
            continue
        if a is not None and a.t.size:
            items.append(a)

    if not items:
        logging.warning("No valid log files found, writing empty %s", nc)
        xr.Dataset().to_netcdf(nc)
        return

    ds = xr.Dataset.from_dataframe(pd.concat(items, ignore_index=True))
    logging.info("Writing %s to %s", ds.sizes, nc)
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    ds.to_netcdf(nc, encoding=encoding)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add log-harvest arguments to the parser."""
    parser.add_argument("--t0", type=str, default=None, help="Earliest timestamp prefix to include")
    parser.add_argument("--nc", type=str, default="log.nc", help="Output NetCDF filename")
    parser.add_argument("filename", type=str, nargs="+", help="Log files to parse")


def run(args: argparse.Namespace) -> int:
    """Execute the log-harvest command."""
    process_files(args.filename, args.t0, args.nc)
    return 0
