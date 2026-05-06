"""Tests for slocum_tpw.analyze_leak."""

import numpy as np
import pytest
import xarray as xr

from slocum_tpw.analyze_leak import _load, fit_leak_rate, load_csv, load_netcdf
from slocum_tpw.simulate_leak import simulate, write_csv


class TestFit:
    def test_no_leak_consistent_with_zero(self):
        """1 day with default noise and no leak: |z| should be small."""
        r = simulate(days=1.0, timestep=3.0, vacuum_drop_per_day=0.0, seed=11)
        fit = fit_leak_rate(r["time"], r["vacuum_inHg"], r["temperature_c"])
        assert abs(fit["z_score"]) < 5.0

    def test_known_leak_recovered_within_sigma(self):
        """0.3 inHg drop over 4 days recovers truth within a few sigma."""
        r = simulate(days=4.0, timestep=3.0, vacuum_drop_per_day=0.075, seed=13)
        fit = fit_leak_rate(r["time"], r["vacuum_inHg"], r["temperature_c"])
        assert abs(fit["slope"] - r["drho_dt_true"]) < 5.0 * fit["slope_stderr"]

    def test_strong_detection(self):
        """0.3 inHg / 4 days over a few-day window yields a huge z-score."""
        r = simulate(days=4.0, timestep=3.0, vacuum_drop_per_day=0.075, seed=17)
        fit = fit_leak_rate(r["time"], r["vacuum_inHg"], r["temperature_c"])
        assert fit["z_score"] > 100.0

    def test_outflow_leak_has_negative_slope(self):
        """Negative vacuum_drop_per_day (vacuum rising = gas out) -> negative slope."""
        r = simulate(days=2.0, timestep=3.0, vacuum_drop_per_day=-0.05, seed=19)
        fit = fit_leak_rate(r["time"], r["vacuum_inHg"], r["temperature_c"])
        assert fit["slope"] < 0.0
        assert fit["z_score"] < -10.0

    def test_result_dict_keys(self):
        r = simulate(days=0.2, timestep=10.0, seed=21)
        fit = fit_leak_rate(r["time"], r["vacuum_inHg"], r["temperature_c"])
        required = {
            "slope",
            "slope_stderr",
            "slope_95ci",
            "slope_per_day",
            "slope_stderr_per_day",
            "intercept",
            "intercept_stderr",
            "sigma_rho",
            "z_score",
            "n_points",
            "time_span_s",
            "time",
            "rho",
        }
        assert required.issubset(fit.keys())

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            fit_leak_rate(
                np.array([0.0, 1.0]),
                np.array([10.0, 10.0]),
                np.array([20.0, 20.0]),
            )


class TestLoadCsv:
    def test_roundtrip_via_write_csv(self, tmp_path):
        r = simulate(days=0.05, timestep=3.0, seed=23)
        fn = tmp_path / "obs.csv"
        write_csv(str(fn), r["time"], r["vacuum_inHg"], r["temperature_c"])

        t, v, T = load_csv(str(fn))
        # write_csv rounds to 3 dp (time), 6 dp (vacuum), 4 dp (temp)
        np.testing.assert_allclose(t, r["time"], atol=1e-3)
        np.testing.assert_allclose(v, r["vacuum_inHg"], atol=1e-6)
        np.testing.assert_allclose(T, r["temperature_c"], atol=1e-4)

    def test_column_override(self, tmp_path):
        fn = tmp_path / "obs.csv"
        fn.write_text("timestamp,vac_inHg,temp_C\n0.0,10.0,20.0\n3.0,10.001,20.0\n6.0,9.999,20.0\n")
        t, _v, _T = load_csv(
            str(fn), time_col="timestamp", vacuum_col="vac_inHg", temp_col="temp_C"
        )
        assert t.size == 3
        np.testing.assert_allclose(t, [0.0, 3.0, 6.0])

    def test_missing_column_raises_key_error(self, tmp_path):
        fn = tmp_path / "obs.csv"
        fn.write_text("time,vacuum\n0,10\n")
        with pytest.raises(KeyError):
            load_csv(str(fn))

    def test_non_numeric_and_nan_rows_dropped(self, tmp_path):
        fn = tmp_path / "obs.csv"
        fn.write_text(
            "m_present_time,m_vacuum,m_veh_temp\n"
            "0.0,10.0,20.0\n"
            "NaN,10.0,20.0\n"
            "3.0,garbage,20.0\n"
            "6.0,10.0,20.0\n"
        )
        t, _v, _T = load_csv(str(fn))
        assert t.size == 2
        np.testing.assert_allclose(t, [0.0, 6.0])

    def test_unsorted_input_is_sorted(self, tmp_path):
        fn = tmp_path / "obs.csv"
        fn.write_text(
            "m_present_time,m_vacuum,m_veh_temp\n6.0,10.0,20.0\n0.0,10.0,20.0\n3.0,10.0,20.0\n"
        )
        t, _, _ = load_csv(str(fn))
        np.testing.assert_allclose(t, [0.0, 3.0, 6.0])


class TestEndToEnd:
    def test_csv_pipeline_recovers_leak(self, tmp_path):
        """simulate -> write_csv -> load_csv -> fit_leak_rate: recover truth."""
        r = simulate(
            days=2.0,
            timestep=6.0,
            vacuum_drop_per_day=0.05,
            seed=29,
        )
        fn = tmp_path / "pipeline.csv"
        write_csv(str(fn), r["time"], r["vacuum_inHg"], r["temperature_c"])

        t, v, T = load_csv(str(fn))
        fit = fit_leak_rate(t, v, T)
        assert abs(fit["slope"] - r["drho_dt_true"]) < 5.0 * fit["slope_stderr"]
        assert fit["z_score"] > 50.0

    def test_netcdf_pipeline_recovers_leak(self, tmp_path):
        """simulate -> NetCDF -> load_netcdf -> fit_leak_rate: recover truth."""
        r = simulate(
            days=2.0,
            timestep=6.0,
            vacuum_drop_per_day=0.05,
            seed=31,
        )
        fn = tmp_path / "pipeline.nc"
        ds = xr.Dataset(
            {
                "m_present_time": ("i", r["time"]),
                "m_vacuum": ("i", r["vacuum_inHg"]),
                "m_veh_temp": ("i", r["temperature_c"]),
            }
        )
        ds.to_netcdf(fn)

        t, v, T = load_netcdf(str(fn))
        fit = fit_leak_rate(t, v, T)
        assert abs(fit["slope"] - r["drho_dt_true"]) < 5.0 * fit["slope_stderr"]
        assert fit["z_score"] > 50.0


def _write_obs_nc(path, time=None, vacuum=None, temp=None, time_units=None):
    """Write a small NetCDF with the standard column names; helper for tests."""
    if time is None:
        time = np.array([0.0, 3.0, 6.0])
    if vacuum is None:
        vacuum = np.full(len(time), 10.0)
    if temp is None:
        temp = np.full(len(time), 20.0)
    time_attrs = {"units": time_units} if time_units else {}
    ds = xr.Dataset(
        {
            "m_present_time": ("i", np.asarray(time), time_attrs),
            "m_vacuum": ("i", np.asarray(vacuum)),
            "m_veh_temp": ("i", np.asarray(temp)),
        }
    )
    ds.to_netcdf(path)


class TestLoadNetcdf:
    def test_basic_roundtrip(self, tmp_path):
        fn = tmp_path / "obs.nc"
        _write_obs_nc(
            fn,
            time=[0.0, 3.0, 6.0, 9.0],
            vacuum=[10.0, 10.001, 9.999, 10.0],
            temp=[20.0, 20.1, 19.9, 20.0],
        )
        t, v, T = load_netcdf(str(fn))
        np.testing.assert_allclose(t, [0.0, 3.0, 6.0, 9.0])
        np.testing.assert_allclose(v, [10.0, 10.001, 9.999, 10.0])
        np.testing.assert_allclose(T, [20.0, 20.1, 19.9, 20.0])

    def test_column_override(self, tmp_path):
        fn = tmp_path / "obs.nc"
        ds = xr.Dataset(
            {
                "timestamp": ("i", np.array([0.0, 3.0, 6.0])),
                "vac_inHg": ("i", np.array([10.0, 10.001, 9.999])),
                "temp_C": ("i", np.array([20.0, 20.0, 20.0])),
            }
        )
        ds.to_netcdf(fn)
        t, _v, _T = load_netcdf(
            str(fn), time_col="timestamp", vacuum_col="vac_inHg", temp_col="temp_C"
        )
        np.testing.assert_allclose(t, [0.0, 3.0, 6.0])

    def test_missing_variable_raises_key_error(self, tmp_path):
        fn = tmp_path / "obs.nc"
        ds = xr.Dataset(
            {
                "m_present_time": ("i", np.array([0.0, 1.0])),
                "m_vacuum": ("i", np.array([10.0, 10.0])),
            }
        )
        ds.to_netcdf(fn)
        with pytest.raises(KeyError):
            load_netcdf(str(fn))

    def test_non_finite_dropped(self, tmp_path):
        fn = tmp_path / "obs.nc"
        _write_obs_nc(
            fn,
            time=[0.0, 3.0, 6.0, 9.0],
            vacuum=[10.0, np.nan, 9.999, 10.0],
            temp=[20.0, 20.0, 20.0, np.inf],
        )
        t, _v, _T = load_netcdf(str(fn))
        np.testing.assert_allclose(t, [0.0, 6.0])

    def test_unsorted_input_is_sorted(self, tmp_path):
        fn = tmp_path / "obs.nc"
        _write_obs_nc(fn, time=[6.0, 0.0, 3.0])
        t, _v, _T = load_netcdf(str(fn))
        np.testing.assert_allclose(t, [0.0, 3.0, 6.0])

    def test_datetime64_time_converted_to_posix(self, tmp_path):
        fn = tmp_path / "obs.nc"
        # Write CF-compliant time so xarray decodes it as datetime64 on read.
        times = np.array([0.0, 60.0, 120.0])  # seconds since epoch
        ds = xr.Dataset(
            {
                "m_present_time": (
                    "i",
                    times,
                    {"units": "seconds since 1970-01-01T00:00:00"},
                ),
                "m_vacuum": ("i", np.array([10.0, 10.0, 10.0])),
                "m_veh_temp": ("i", np.array([20.0, 20.0, 20.0])),
            }
        )
        ds.to_netcdf(fn)
        t, _v, _T = load_netcdf(str(fn))
        # Should round-trip back to POSIX seconds, regardless of dtype on disk.
        np.testing.assert_allclose(t, [0.0, 60.0, 120.0])


class TestDispatch:
    def test_dispatch_csv(self, tmp_path):
        fn = tmp_path / "obs.csv"
        fn.write_text("m_present_time,m_vacuum,m_veh_temp\n0.0,10.0,20.0\n3.0,10.0,20.0\n")
        t, _v, _T = _load(str(fn), "m_present_time", "m_vacuum", "m_veh_temp")
        assert t.size == 2

    def test_dispatch_netcdf(self, tmp_path):
        fn = tmp_path / "obs.nc"
        _write_obs_nc(fn)
        t, _v, _T = _load(str(fn), "m_present_time", "m_vacuum", "m_veh_temp")
        assert t.size == 3

    def test_dispatch_uppercase_extension(self, tmp_path):
        fn = tmp_path / "obs.NC"
        _write_obs_nc(fn)
        t, _v, _T = _load(str(fn), "m_present_time", "m_vacuum", "m_veh_temp")
        assert t.size == 3
