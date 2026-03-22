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
    (\d+)       \s+   # field 1 (unused)
    (\d+)       \s+   # ident
    (\d+)       \s+   # nLines
    (\d+)       \s+   # nBytes
    (.)         \s+   # satellite ID
    (.)         \s+   # location class
    (\d{4}) - (\d{2}) - (\d{2}) \s+  # date YYYY-MM-DD
    (\d{2}) : (\d{2}) : (\d{2}) \s+  # time HH:MM:SS
    (\d+[.]\d+) \s+   # latitude
    (\d+[.]\d+) \s+   # longitude
    (\d+[.]\d+) \s+   # altitude
    (\d+)              # frequency
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
                    int(matches[7]),
                    int(matches[8]),
                    int(matches[9]),
                    int(matches[10]),
                    int(matches[11]),
                    int(matches[12]),
                    tzinfo=datetime.timezone.utc,
                )
            except ValueError:
                continue
            records["ident"].append(int(matches[2]))
            records["nLines"].append(int(matches[3]))
            records["nBytes"].append(int(matches[4]))
            records["satellite"].append(matches[5])
            records["locationClass"].append(matches[6])
            records["time"].append(t)
            records["lat"].append(float(matches[13]))
            records["lon"].append(float(matches[14]))
            records["altitude"].append(float(matches[15]))
            records["frequency"].append(float(matches[16]))

    if len(records["ident"]) == 0:
        return None
    return pd.DataFrame(records)


def process_files(fn_nc: str, filenames: list[str]) -> None:
    """Process multiple ARGOS files and write results to NetCDF."""
    records = []
    for fn in filenames:
        df = proc_file(fn)
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
    ds.to_netcdf(fn_nc)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add decode-argos arguments to the parser."""
    parser.add_argument("filename", type=str, nargs="+", help="ARGOS message files to decode")
    parser.add_argument("--nc", type=str, default="tpw.nc", help="Output NetCDF filename")


def run(args: argparse.Namespace) -> int:
    """Execute the decode-argos command."""
    process_files(args.nc, args.filename)
    return 0
