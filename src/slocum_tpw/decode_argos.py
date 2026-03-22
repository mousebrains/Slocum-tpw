#
# Decode ARGOS satellite position messages into NetCDF
#
# Pat Welch, pat@mousebrains.com

import argparse
import datetime
import logging
import re

import pandas as pd
import xarray as xr

# ARGOS message line format:
# Fields: unknown, ident, nLines, nBytes, satellite, locationClass,
#         YYYY-MM-DD HH:MM:SS, lat, lon, altitude, frequency
_ARGOS_RE = re.compile(
    r"""
    \d+                       \s+  # field 1 (unused)
    (?P<ident>\d+)            \s+  # ident
    (?P<nLines>\d+)           \s+  # nLines
    (?P<nBytes>\d+)           \s+  # nBytes
    (?P<satellite>.)          \s+  # satellite ID
    (?P<locationClass>.)      \s+  # location class
    (?P<year>\d{4}) - (?P<month>\d{2}) - (?P<day>\d{2})   \s+  # date
    (?P<hour>\d{2}) : (?P<minute>\d{2}) : (?P<second>\d{2}) \s+  # time
    (?P<lat>\d+[.]\d+)       \s+  # latitude
    (?P<lon>\d+[.]\d+)       \s+  # longitude
    (?P<altitude>\d+[.]\d+)  \s+  # altitude
    (?P<frequency>\d+)             # frequency
""",
    re.VERBOSE,
)


def proc_file(fn: str) -> pd.DataFrame | None:
    """Parse a single ARGOS message file and return a DataFrame, or None if empty."""
    records: dict[str, list] = {
        "ident": [],
        "nLines": [],
        "nBytes": [],
        "satellite": [],
        "locationClass": [],
        "time": [],
        "lat": [],
        "lon": [],
        "altitude": [],
        "frequency": [],
    }

    with open(fn) as fp:
        for line in fp:
            line = line.strip()
            matches = _ARGOS_RE.fullmatch(line)
            if not matches:
                continue
            try:
                t = datetime.datetime(
                    int(matches["year"]),
                    int(matches["month"]),
                    int(matches["day"]),
                    int(matches["hour"]),
                    int(matches["minute"]),
                    int(matches["second"]),
                    tzinfo=datetime.timezone.utc,
                )
            except ValueError:
                continue
            records["ident"].append(int(matches["ident"]))
            records["nLines"].append(int(matches["nLines"]))
            records["nBytes"].append(int(matches["nBytes"]))
            records["satellite"].append(matches["satellite"])
            records["locationClass"].append(matches["locationClass"])
            records["time"].append(t)
            records["lat"].append(float(matches["lat"]))
            records["lon"].append(float(matches["lon"]))
            records["altitude"].append(float(matches["altitude"]))
            records["frequency"].append(int(matches["frequency"]))

    if len(records["ident"]) == 0:
        return None
    return pd.DataFrame(records)


def process_files(fn_nc: str, filenames: list[str]) -> None:
    """Process multiple ARGOS files and write results to NetCDF."""
    records = []
    for fn in filenames:
        try:
            df = proc_file(fn)
        except OSError as e:
            logging.error("Failed to read %s: %s", fn, e)
            continue
        if df is not None:
            records.append(df)

    if not records:
        logging.warning(
            "No ARGOS records found in %d files, writing empty %s", len(filenames), fn_nc
        )
        xr.Dataset().to_netcdf(fn_nc)
        return

    df = pd.concat(records, ignore_index=True)
    # Strip timezone for NetCDF compatibility (times are always UTC)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df.set_index("time")
    ds = df.to_xarray()
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    ds.to_netcdf(fn_nc, encoding=encoding)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add decode-argos arguments to the parser."""
    parser.add_argument("filename", type=str, nargs="+", help="ARGOS message files to decode")
    parser.add_argument("--nc", type=str, default="tpw.nc", help="Output NetCDF filename")


def run(args: argparse.Namespace) -> int:
    """Execute the decode-argos command."""
    process_files(args.nc, args.filename)
    return 0
