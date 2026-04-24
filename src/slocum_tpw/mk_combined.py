#
# Combine log, flight, and science data for a glider into CF-compliant NetCDF
#
# June-2023, Pat Welch, pat@mousebrains.com

import argparse
import datetime
import logging
import os.path

import gsw
import numpy as np
import pandas as pd
import xarray as xr

from slocum_tpw.slocum_utils import mk_degrees


def mk_combo(
    gld: str | None,
    fn_output: str,
    fn_log: str,
    fn_flt: str,
    fn_sci: str,
) -> bool:
    """Merge log, flight, and science data into a single CF-compliant NetCDF.

    Returns True on success, False on failure.
    """
    for fn in (fn_log, fn_flt, fn_sci):
        if not os.path.isfile(fn):
            logging.error("Input file not found: %s", fn)
            return False

    # --- Load log data ---
    with xr.open_dataset(fn_log, engine="netcdf4") as ds:
        required = {"t", "m_water_vx", "m_water_vy"}
        missing = required - set(ds.data_vars)
        if missing:
            logging.error("Log file %s missing variables: %s", fn_log, missing)
            return False
        if gld and "glider" in ds.data_vars:
            ds = ds.sel(index=ds.index[ds.glider == gld])
            if len(ds.index) == 0:
                logging.error("No data for glider %s in %s", gld, fn_log)
                return False
        dfLog = pd.DataFrame()
        dfLog["timeu"] = ds.t.data.astype("datetime64[s]").astype("int64").astype(float)
        dfLog["latu"] = ds.lat.data if "lat" in ds else np.full(len(ds.index), np.nan)
        dfLog["lonu"] = ds.lon.data if "lon" in ds else np.full(len(ds.index), np.nan)
        dfLog["u"] = ds.m_water_vx
        dfLog["v"] = ds.m_water_vy
        dfLog = dfLog.dropna(axis=0, subset=["latu", "lonu", "u", "v"], how="all")
        dfLog = dfLog[dfLog.timeu > 0]
        (_t, ix) = np.unique(dfLog.timeu, return_index=True)
        dfLog = dfLog.iloc[ix]

        # Interpolate missing GPS in log data
        qLat = np.logical_not(np.isnan(dfLog.latu))
        qLon = np.logical_not(np.isnan(dfLog.lonu))
        if qLat.sum() >= 2:
            missing = np.logical_not(qLat)
            dfLog.loc[missing, "latu"] = np.interp(
                dfLog.timeu[missing],
                dfLog.timeu[qLat],
                dfLog.latu[qLat],
                left=np.nan,
                right=np.nan,
            )
        if qLon.sum() >= 2:
            missing = np.logical_not(qLon)
            dfLog.loc[missing, "lonu"] = np.interp(
                dfLog.timeu[missing],
                dfLog.timeu[qLon],
                dfLog.lonu[qLon],
                left=np.nan,
                right=np.nan,
            )

        dfLog = dfLog.dropna(axis=0, subset=("u", "v"), how="any")
        if dfLog.empty:
            logging.error("No valid log data after filtering for %s", gld)
            return False
        dfLog = dfLog.set_index("timeu")

    # --- Load flight data (GPS fixes) ---
    with xr.open_dataset(fn_flt, engine="netcdf4") as ds:
        flt = pd.DataFrame()
        flt["time"] = ds.m_present_time
        flt["latGPS"] = mk_degrees(ds.m_gps_lat.data)
        flt["lonGPS"] = mk_degrees(ds.m_gps_lon.data)
        flt = flt.dropna(axis=0, subset=flt.columns, how="any")
        flt = flt[flt.time > 0]
        (_t, ix) = np.unique(flt.time, return_index=True)
        flt = flt.iloc[ix]
        if len(flt) < 2:
            logging.error("Fewer than 2 GPS fixes in %s for %s, cannot interpolate", fn_flt, gld)
            return False
        flt_time = flt.time.to_numpy()
        flt_lat = flt.latGPS.to_numpy()
        flt_lon = flt.lonGPS.to_numpy()

    # --- Load science data (CTD) ---
    with xr.open_dataset(fn_sci, engine="netcdf4") as ds:
        sci = pd.DataFrame()
        sci["time"] = ds.sci_m_present_time
        sci["t"] = ds.sci_water_temp
        sci["C"] = ds.sci_water_cond * 10  # S/m -> mS/cm for GSW
        sci["P"] = ds.sci_water_pressure * 10  # bar -> dbar for GSW
        sci = sci.dropna(axis=0, subset=sci.columns, how="any")
        sci = sci[sci.time > 0]
        sci = sci[sci.t > 0]
        sci = sci[sci.P > 0]
        # Sanity check converted units (catch data already in mS/cm or dbar)
        if sci.C.median() > 100:
            logging.warning(
                "Median conductivity %.1f mS/cm seems high — "
                "is sci_water_cond already in mS/cm instead of S/m?",
                sci.C.median(),
            )
        if sci.P.median() > 12000:
            logging.warning(
                "Median pressure %.1f dbar seems high — "
                "is sci_water_pressure already in dbar instead of bar?",
                sci.P.median(),
            )
        sci["lat"] = np.interp(sci.time, flt_time, flt_lat, left=np.nan, right=np.nan)
        sci["lon"] = np.interp(sci.time, flt_time, flt_lon, left=np.nan, right=np.nan)
        n_before = len(sci)
        sci = sci.dropna(axis=0, subset=["lat", "lon"], how="any")
        if sci.empty:
            logging.error("No science data with valid GPS positions for %s", gld)
            return False
        if len(sci) < n_before:
            logging.info(
                "%s: dropped %d/%d science rows outside GPS time range",
                gld,
                n_before - len(sci),
                n_before,
            )

        # Oceanographic calculations via TEOS-10 / GSW
        sci["depth"] = -gsw.conversions.z_from_p(sci.P.to_numpy(), sci.lat.to_numpy())
        sci["s"] = gsw.SP_from_C(sci.C.to_numpy(), sci.t.to_numpy(), sci.P.to_numpy())
        sa = gsw.SA_from_SP(
            sci.s.to_numpy(), sci.P.to_numpy(), sci.lon.to_numpy(), sci.lat.to_numpy()
        )
        sci["theta"] = gsw.conversions.pt0_from_t(sa, sci.t.to_numpy(), sci.P.to_numpy())
        ct = gsw.conversions.CT_from_pt(sa, sci.theta.to_numpy())
        sci["sigma"] = gsw.density.sigma0(sa, ct)
        sci["rho"] = gsw.density.rho_t_exact(sa, sci.t.to_numpy(), sci.P.to_numpy()) - 1000
        sci = sci.drop(columns=["C", "P"])
        n_total = len(sci)
        sci = sci.dropna(axis=0, subset=[c for c in sci.columns if c != "time"], how="any")
        (_t, ix) = np.unique(sci.time, return_index=True)
        sci = sci.iloc[ix]
        logging.info(
            "%s: %d science records (%d dropped during QC)", gld, len(sci), n_total - len(sci)
        )
        if sci.empty:
            logging.error("No valid science data remaining for %s", gld)
            return False
        sci = sci.set_index("time")

    # --- Merge and add metadata ---
    ds = xr.merge([dfLog.to_xarray(), sci.to_xarray()])

    platform = f"TWR Slocum {gld}" if gld else "TWR Slocum"

    attrs = dict(
        time=dict(
            units="seconds since 1970-01-01",
            calendar="proleptic_gregorian",
            standard_name="time",
            long_name="Time",
        ),
        lat=dict(
            units="degrees_north",
            long_name="latitude",
            standard_name="latitude",
            ancillary_variables="lon",
            coordinate_reference_frame="WGS84",
            reference="WGS84",
            comment="WGS84",
            valid_min=-90.0,
            valid_max=90.0,
            observation_type="GPS",
            platform=platform,
        ),
        lon=dict(
            units="degrees_east",
            long_name="longitude",
            standard_name="longitude",
            ancillary_variables="lat",
            coordinate_reference_frame="WGS84",
            reference="WGS84",
            comment="WGS84",
            valid_min=-180.0,
            valid_max=180.0,
            observation_type="GPS",
            platform=platform,
        ),
        u=dict(
            units="m s-1",
            long_name="m_water_vx",
            standard_name="eastward_sea_water_velocity",
            valid_min=-10.0,
            valid_max=10.0,
            comment="Depth averaged eastward current",
            observation_type="calculation",
            platform=platform,
        ),
        v=dict(
            units="m s-1",
            long_name="m_water_vy",
            standard_name="northward_sea_water_velocity",
            valid_min=-10.0,
            valid_max=10.0,
            comment="Depth averaged northward current",
            observation_type="calculation",
            platform=platform,
        ),
        t=dict(
            units="degree_Celsius",
            long_name="temperature",
            standard_name="sea_water_temperature",
            units_metadata="temperature: on_scale",
        ),
        depth=dict(
            units="m",
            positive="down",
            long_name="depth",
            standard_name="depth",
            accuracy=0.01,
            precision=0.001,
            resolution=0.001,
            valid_max=1000.0,
            valid_min=0.0,
            reference_datum="sea surface",
            ancillary_variables="s",
            comment="GPCTD",
            instrument="GPCTD",
            observation_type="in-situ",
            platform=platform,
        ),
        s=dict(
            units="1",
            long_name="salinity",
            standard_name="sea_water_practical_salinity",
        ),
        theta=dict(
            units="degree_Celsius",
            long_name="potentialTemperature",
            standard_name="sea_water_potential_temperature",
            units_metadata="temperature: on_scale",
        ),
        sigma=dict(
            units="kg m-3",
            long_name="potential density anomaly (sigma-0)",
            standard_name="sea_water_sigma_theta",
            comment="Potential density minus 1000 kg m-3, referenced to 0 dbar",
        ),
        rho=dict(
            units="kg m-3",
            long_name="density anomaly (rho - 1000)",
            standard_name="sea_water_sigma_t",
            comment="In-situ density minus 1000 kg m-3",
        ),
    )
    attrs["lonu"] = dict(attrs["lon"])
    attrs["latu"] = dict(attrs["lat"])

    ds.time.attrs.update(attrs["time"])
    ds.timeu.attrs.update(attrs["time"])
    for key in ds:
        if key in attrs:
            ds[key].attrs.update(attrs[key])

    now = datetime.datetime.now(datetime.UTC)
    ds.attrs.update(
        dict(
            title=f"{gld}" if gld else "TWR Slocum",
            comment="Salinity is not thermal mass corrected",
            history=f"Generated {now}",
            Conventions="CF-1.13",
            date_created=f"{now}",
            date_issued=f"{now}",
            date_modified=f"{now}",
            institution="Oregon State University Glider Laboratory",
        )
    )

    encoding = dict(zlib=True, complevel=4)
    for key in ds:
        ds[key].encoding.update(encoding)

    ds.to_netcdf(fn_output)
    return True


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add mk-combined arguments to the parser."""
    parser.add_argument("--prefix", type=str, default="osu", help="Institution prefix")
    parser.add_argument("--glider", type=int, help="Glider number (filters log file by glider ID)")
    parser.add_argument("--output", type=str, required=True, help="Output NetCDF filename")
    parser.add_argument("--nc-log", type=str, default="log.nc", help="Input log NetCDF")
    parser.add_argument("--nc-flight", type=str, help="Input flight NetCDF")
    parser.add_argument("--nc-science", type=str, help="Input science NetCDF")


def run(args: argparse.Namespace) -> int:
    """Execute the mk-combined command."""
    gld = f"{args.prefix}{args.glider}" if args.glider is not None else None

    if args.nc_flight is None:
        if gld is None:
            logging.error("--nc-flight is required when --glider is not specified")
            return 1
        args.nc_flight = os.path.join(os.path.dirname(args.nc_log), f"flt.{gld}.nc")

    if args.nc_science is None:
        if gld is None:
            logging.error("--nc-science is required when --glider is not specified")
            return 1
        args.nc_science = os.path.join(os.path.dirname(args.nc_log), f"sci.{gld}.nc")

    ok = mk_combo(gld, args.output, args.nc_log, args.nc_flight, args.nc_science)
    return 0 if ok else 1
