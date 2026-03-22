"""Tests for slocum_tpw.mk_combined."""

import numpy as np
import xarray as xr

from slocum_tpw.mk_combined import mk_combo


class TestMkCombo:
    def test_full_pipeline(self, tmp_path, log_nc, flt_nc, sci_nc):
        output = tmp_path / "combined.nc"
        result = mk_combo("osu684", str(output), str(log_nc), str(flt_nc), str(sci_nc))
        assert result is True
        assert output.exists()
        with xr.open_dataset(str(output)) as ds:
            # Science variables
            assert "t" in ds
            assert "s" in ds
            assert "depth" in ds
            assert "theta" in ds
            assert "sigma" in ds
            assert "rho" in ds
            # Log variables
            assert "u" in ds
            assert "v" in ds
            # GPS
            assert "lat" in ds
            assert "lon" in ds

    def test_output_has_cf_attributes(self, tmp_path, log_nc, flt_nc, sci_nc):
        output = tmp_path / "combined.nc"
        mk_combo("osu684", str(output), str(log_nc), str(flt_nc), str(sci_nc))
        with xr.open_dataset(str(output)) as ds:
            assert ds.attrs["Conventions"] == "CF-1.13"
            assert "date_created" in ds.attrs
            assert ds["s"].attrs["standard_name"] == "sea_water_practical_salinity"
            assert ds["depth"].attrs["positive"] == "down"

    def test_oceanographic_values_reasonable(self, tmp_path, log_nc, flt_nc, sci_nc):
        output = tmp_path / "combined.nc"
        mk_combo("osu684", str(output), str(log_nc), str(flt_nc), str(sci_nc))
        with xr.open_dataset(str(output)) as ds:
            # Depth should be positive (positive=down)
            assert (ds.depth.values > 0).all()
            # Salinity should be in reasonable range (30-40 PSU for ocean)
            assert (ds.s.values > 20).all()
            assert (ds.s.values < 45).all()
            # Temperature should be close to input (15-15.5 C)
            assert (ds.t.values > 10).all()
            assert (ds.t.values < 20).all()
            # Sigma-0 should be in reasonable range (20-30 kg/m3)
            assert (ds.sigma.values > 15).all()
            assert (ds.sigma.values < 35).all()

    def test_missing_log_file(self, tmp_path, flt_nc, sci_nc):
        output = tmp_path / "combined.nc"
        result = mk_combo(
            "osu684", str(output), str(tmp_path / "missing.nc"), str(flt_nc), str(sci_nc)
        )
        assert result is False

    def test_missing_flt_file(self, tmp_path, log_nc, sci_nc):
        output = tmp_path / "combined.nc"
        result = mk_combo(
            "osu684", str(output), str(log_nc), str(tmp_path / "missing.nc"), str(sci_nc)
        )
        assert result is False

    def test_missing_sci_file(self, tmp_path, log_nc, flt_nc):
        output = tmp_path / "combined.nc"
        result = mk_combo(
            "osu684", str(output), str(log_nc), str(flt_nc), str(tmp_path / "missing.nc")
        )
        assert result is False

    def test_missing_log_variables(self, tmp_path, flt_nc, sci_nc):
        """Log file missing required m_water_vx/vy variables."""
        ds = xr.Dataset({"t": ("index", [np.datetime64("2023-06-12T15:30:00")])})
        bad_log = tmp_path / "bad_log.nc"
        ds.to_netcdf(bad_log)
        output = tmp_path / "combined.nc"
        result = mk_combo("osu684", str(output), str(bad_log), str(flt_nc), str(sci_nc))
        assert result is False

    def test_insufficient_gps_fixes(self, tmp_path, log_nc, sci_nc):
        """Flight file with fewer than 2 GPS fixes cannot interpolate."""
        from .conftest import BASE_EPOCH

        ds = xr.Dataset(
            {
                "m_present_time": ("row", [BASE_EPOCH]),
                "m_gps_lat": ("row", [4430.0]),
                "m_gps_lon": ("row", [-12400.0]),
            }
        )
        bad_flt = tmp_path / "flt.osu684.nc"
        ds.to_netcdf(bad_flt)
        output = tmp_path / "combined.nc"
        result = mk_combo("osu684", str(output), str(log_nc), str(bad_flt), str(sci_nc))
        assert result is False

    def test_science_outside_gps_range(self, tmp_path, log_nc, flt_nc):
        """Science data outside flight GPS time range should be dropped."""
        from .conftest import BASE_EPOCH

        # Science times well outside the flight GPS range
        ds = xr.Dataset(
            {
                "sci_m_present_time": ("row", [BASE_EPOCH + 99999, BASE_EPOCH + 100000]),
                "sci_water_temp": ("row", [15.0, 15.5]),
                "sci_water_cond": ("row", [4.0, 4.1]),
                "sci_water_pressure": ("row", [1.0, 1.5]),
            }
        )
        bad_sci = tmp_path / "sci.osu684.nc"
        ds.to_netcdf(bad_sci)
        output = tmp_path / "combined.nc"
        result = mk_combo("osu684", str(output), str(log_nc), str(flt_nc), str(bad_sci))
        assert result is False

    def test_no_glider_filter(self, tmp_path, log_nc, flt_nc, sci_nc):
        """Passing None for glider should process all data."""
        output = tmp_path / "combined.nc"
        result = mk_combo(None, str(output), str(log_nc), str(flt_nc), str(sci_nc))
        assert result is True
        assert output.exists()

    def test_empty_log_after_velocity_filter(self, tmp_path, flt_nc, sci_nc):
        """Log data with all-NaN velocities should fail."""
        t0 = np.datetime64("2023-06-12T15:30:00")
        t1 = np.datetime64("2023-06-12T15:46:40")
        ds = xr.Dataset(
            {
                "t": ("index", np.array([t0, t1], dtype="datetime64[s]")),
                "glider": ("index", ["osu684", "osu684"]),
                "lat": ("index", [44.5, 44.6]),
                "lon": ("index", [-124.0, -124.1]),
                "m_water_vx": ("index", [np.nan, np.nan]),
                "m_water_vy": ("index", [np.nan, np.nan]),
            }
        )
        bad_log = tmp_path / "log.nc"
        ds.to_netcdf(bad_log)
        output = tmp_path / "combined.nc"
        result = mk_combo("osu684", str(output), str(bad_log), str(flt_nc), str(sci_nc))
        assert result is False

    def test_science_partial_gps_overlap(self, tmp_path, log_nc, flt_nc):
        """Science data partially within GPS range logs the drop count."""
        from .conftest import BASE_EPOCH

        ds = xr.Dataset(
            {
                "sci_m_present_time": (
                    "row",
                    [BASE_EPOCH + 100, BASE_EPOCH + 600, BASE_EPOCH + 99999],
                ),
                "sci_water_temp": ("row", [15.0, 15.5, 15.0]),
                "sci_water_cond": ("row", [4.0, 4.1, 4.0]),
                "sci_water_pressure": ("row", [1.0, 1.5, 1.0]),
            }
        )
        sci = tmp_path / "sci.osu684.nc"
        ds.to_netcdf(sci)
        output = tmp_path / "combined.nc"
        result = mk_combo("osu684", str(output), str(log_nc), str(flt_nc), str(sci))
        assert result is True
