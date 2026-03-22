"""Shared test fixtures for slocum_tpw."""

import numpy as np
import pytest
import xarray as xr

# Base epoch time used for flight and science fixtures.
# Exact calendar date doesn't matter — only internal consistency does.
BASE_EPOCH = 1686580200.0


@pytest.fixture
def log_nc(tmp_path):
    """Create a minimal log NetCDF file with two time steps for one glider."""
    t0 = np.datetime64("2023-06-12T15:30:00")
    t1 = np.datetime64("2023-06-12T15:46:40")  # +1000s
    ds = xr.Dataset(
        {
            "t": ("index", np.array([t0, t1], dtype="datetime64[s]")),
            "glider": ("index", ["osu684", "osu684"]),
            "lat": ("index", [44.5, 44.6]),
            "lon": ("index", [-124.0, -124.1]),
            "m_water_vx": ("index", [0.1, 0.2]),
            "m_water_vy": ("index", [0.05, 0.15]),
        }
    )
    fn = tmp_path / "log.nc"
    ds.to_netcdf(fn)
    return fn


@pytest.fixture
def flt_nc(tmp_path):
    """Create a minimal flight NetCDF with 3 GPS fixes in DDMM.MM format."""
    ds = xr.Dataset(
        {
            "m_present_time": (
                "row",
                [BASE_EPOCH - 100, BASE_EPOCH + 500, BASE_EPOCH + 1100],
            ),
            "m_gps_lat": ("row", [4430.0, 4436.0, 4442.0]),  # 44.5, 44.6, 44.7 deg
            "m_gps_lon": ("row", [-12400.0, -12406.0, -12412.0]),  # -124.0, -124.1, -124.2
        }
    )
    fn = tmp_path / "flt.osu684.nc"
    ds.to_netcdf(fn)
    return fn


@pytest.fixture
def sci_nc(tmp_path):
    """Create a minimal science NetCDF with CTD data."""
    ds = xr.Dataset(
        {
            "sci_m_present_time": ("row", [BASE_EPOCH + 100, BASE_EPOCH + 600]),
            "sci_water_temp": ("row", [15.0, 15.5]),
            "sci_water_cond": ("row", [4.0, 4.1]),  # S/m
            "sci_water_pressure": ("row", [1.0, 1.5]),  # bar
        }
    )
    fn = tmp_path / "sci.osu684.nc"
    ds.to_netcdf(fn)
    return fn


# --- Sample ARGOS file content ---

ARGOS_LINE_VALID = "12345 67890 5 100 A 3 2023-06-15 12:30:45 44.123 124.567 0.000 401650000"
ARGOS_LINE_VALID_2 = "12345 67891 3 80 B 2 2023-06-15 13:00:00 44.200 124.600 0.000 401650001"
ARGOS_LINE_BAD_DATE = "12345 67890 5 100 A 3 2023-13-45 12:30:45 44.123 124.567 0.000 401650000"
ARGOS_LINE_MALFORMED = "this is not a valid argos line"


# --- Sample log file content ---

SAMPLE_LOG = (
    b"Vehicle Name: osu684\n"
    b"Curr Time: Mon Jun 12 15:30:00 2023 MT: 12345\n"
    b"GPS Location: 4430.5000 N -12406.0000 E measured 10.0 secs ago\n"
    b"sensor:sci_water_temp(celsius)=15.5 10.0 secs ago\n"
    b"sensor:m_water_vx(m/s)=0.1 10.0 secs ago\n"
    b"sensor:m_water_vy(m/s)=0.05 10.0 secs ago\n"
    b"Curr Time: Mon Jun 12 15:35:00 2023 MT: 12645\n"
    b"GPS Location: 4431.0000 N -12407.0000 E measured 5.0 secs ago\n"
    b"sensor:sci_water_temp(celsius)=15.6 5.0 secs ago\n"
)
