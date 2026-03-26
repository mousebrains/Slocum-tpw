"""Tests for slocum_tpw.recover_by."""

import json

import numpy as np
import pytest
import xarray as xr

from slocum_tpw.cli import main
from slocum_tpw.recover_by import _safe_sqrt, fit_recovery, prepare_dataset


def make_linear_nc(path, start="2025-01-01", n_days=51, batt_start=100.0, batt_end=50.0):
    """Create a NetCDF file with perfectly linear battery decay."""
    times = np.datetime64(start) + np.arange(n_days).astype("timedelta64[D]")
    battery = np.linspace(batt_start, batt_end, n_days)
    ds = xr.Dataset(
        {"m_lithium_battery_relative_charge": ("time", battery)},
        coords={"time": times},
    )
    ds.to_netcdf(path)


def _run(argv):
    """Run CLI and return exit code."""
    with pytest.raises(SystemExit) as exc_info:
        main(["recover-by"] + argv)
    return exc_info.value.code


class TestRecoverBy:
    def test_linear_decay(self, tmp_path, capsys):
        """Perfectly linear decay should produce exact recovery date and R²=1."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        # slope = (50 - 100) / 50 = -1 %/day
        # d_recovery = (15 - 100) / -1 = 85 days
        # 2025-01-01 + 85 days = 2025-03-27
        rc = _run(["--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "2025-03-27" in out
        assert "R-squared:   1.0000" in out

    def test_ndays_filter(self, tmp_path, capsys):
        """Using --ndays should restrict the fit window."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--ndays", "10", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Slope:" in out
        assert "Intercept:" in out
        assert "R-squared:" in out
        assert "DOF:" in out
        assert "Recovery By:" in out

    def test_start_stop_filter(self, tmp_path, capsys):
        """Using --start/--stop should restrict the fit window."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--start", "2025-01-10", "--stop", "2025-02-10", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Slope:" in out
        assert "Recovery By:" in out

    def test_missing_variable(self, tmp_path):
        """Missing sensor variable should skip the file and return failure."""
        nc = tmp_path / "test.nc"
        times = np.datetime64("2025-01-01") + np.arange(10).astype("timedelta64[D]")
        ds = xr.Dataset(
            {"wrong_sensor": ("time", np.linspace(100, 90, 10))},
            coords={"time": times},
        )
        ds.to_netcdf(nc)

        rc = _run([str(nc)])
        assert rc == 1

    def test_multiple_files(self, tmp_path, capsys):
        """Processing multiple files should report results for each."""
        nc1 = tmp_path / "a.nc"
        nc2 = tmp_path / "b.nc"
        make_linear_nc(nc1)
        make_linear_nc(nc2, batt_start=90, batt_end=45)

        rc = _run(["--threshold", "15", str(nc1), str(nc2)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "a.nc" in out
        assert "b.nc" in out
        assert out.count("Recovery By") == 2

    def test_too_few_points(self, tmp_path):
        """Fewer than 3 data points should skip the file and return failure."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=2)

        rc = _run([str(nc)])
        assert rc == 1

    def test_exactly_three_points(self, tmp_path, capsys):
        """Exactly 3 data points should succeed."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=3)

        rc = _run(["--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Recovery By" in out

    def test_ndays_stop_conflict(self, tmp_path):
        """--ndays and --stop together should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--ndays", "7", "--stop", "2025-02-01", str(nc)])
        assert rc == 2

    def test_plot_all_files_fail(self, tmp_path):
        """--output with all files failing should not crash."""
        plot_path = tmp_path / "plot.png"
        rc = _run(["--output", str(plot_path), str(tmp_path / "nonexistent.nc")])
        assert rc == 1

    def test_output_saves_plot(self, tmp_path):
        """--output should save a plot file."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)
        plot_path = tmp_path / "plot.png"

        rc = _run(["--output", str(plot_path), str(nc)])
        assert rc == 0
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_posixtime_float(self, tmp_path, capsys):
        """Float epoch seconds should be auto-converted to datetime64."""
        nc = tmp_path / "test.nc"
        epoch_start = int(
            (np.datetime64("2025-01-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
        )
        times = epoch_start + np.arange(51, dtype=np.float64) * 86400
        battery = np.linspace(100, 50, 51)
        ds = xr.Dataset(
            {"m_lithium_battery_relative_charge": ("time", battery)},
            coords={"time": times},
        )
        ds.to_netcdf(nc)

        rc = _run(["--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "2025-03-27" in out

    def test_positive_slope(self, tmp_path, capsys):
        """Increasing battery should warn about past recovery date but succeed."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, batt_start=50, batt_end=100)

        rc = _run(["--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Recovery By" in out

    def test_missing_file(self, tmp_path):
        """Nonexistent file should be logged and return failure."""
        rc = _run([str(tmp_path / "nonexistent.nc")])
        assert rc == 1

    def test_constant_data(self, tmp_path):
        """Constant battery values produce near-zero slope and should fail."""
        nc = tmp_path / "test.nc"
        times = np.datetime64("2025-01-01") + np.arange(20).astype("timedelta64[D]")
        battery = np.full(20, 80.0)
        ds = xr.Dataset(
            {"m_lithium_battery_relative_charge": ("time", battery)},
            coords={"time": times},
        )
        ds.to_netcdf(nc)

        rc = _run([str(nc)])
        assert rc == 1

    def test_nan_sensor_data(self, tmp_path, capsys):
        """NaN values in sensor data should be filtered out before fitting."""
        nc = tmp_path / "test.nc"
        times = np.datetime64("2025-01-01") + np.arange(51).astype("timedelta64[D]")
        battery = np.linspace(100, 50, 51)
        battery[10] = np.nan
        battery[20] = np.nan
        battery[30] = np.nan
        ds = xr.Dataset(
            {"m_lithium_battery_relative_charge": ("time", battery)},
            coords={"time": times},
        )
        ds.to_netcdf(nc)

        rc = _run(["--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Recovery By" in out

    def test_json_output(self, tmp_path, capsys):
        """--json should produce valid JSON with expected fields."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--json", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 1
        r = data[0]
        assert r["sensor"] == "m_lithium_battery_relative_charge"
        assert r["threshold"] == 15.0
        assert r["confidence"] == 0.95
        assert "2025-03-27" in r["recovery_date"]
        assert r["r_squared"] == pytest.approx(1.0)
        assert r["slope"] == pytest.approx(-1.0)
        assert r["intercept"] == pytest.approx(100.0)
        assert r["n_points"] == 51

    def test_custom_confidence(self, tmp_path, capsys):
        """Custom confidence level should change output labels and CI width."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--confidence", "0.99", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "(99%)" in out
        assert "Intercept:" in out
        assert "Slope:" in out
        assert "Recovery By:" in out

    def test_start_only(self, tmp_path, capsys):
        """Using --start without --stop should restrict start but use all data to end."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--start", "2025-02-01", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Recovery By" in out

    def test_ndays_start_conflict(self, tmp_path):
        """--ndays and --start together should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--ndays", "7", "--start", "2025-01-10", str(nc)])
        assert rc == 2

    def test_confidence_zero(self, tmp_path):
        """--confidence 0 should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--confidence", "0", str(nc)])
        assert rc == 2

    def test_confidence_one(self, tmp_path):
        """--confidence 1 should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--confidence", "1", str(nc)])
        assert rc == 2

    def test_multiple_files_partial_failure(self, tmp_path, capsys):
        """One bad file should not prevent good files from processing."""
        nc_good = tmp_path / "good.nc"
        nc_bad = tmp_path / "bad.nc"
        make_linear_nc(nc_good)
        # bad file has wrong sensor name
        times = np.datetime64("2025-01-01") + np.arange(10).astype("timedelta64[D]")
        ds = xr.Dataset(
            {"wrong_sensor": ("time", np.linspace(100, 90, 10))},
            coords={"time": times},
        )
        ds.to_netcdf(nc_bad)

        rc = _run(["--threshold", "15", str(nc_good), str(nc_bad)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Recovery By" in out


class TestPrepareDataset:
    def test_from_file(self, tmp_path):
        """prepare_dataset should load and clean a NetCDF file."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        ds = prepare_dataset(nc)
        assert "time" in ds.dims
        assert "m_lithium_battery_relative_charge" in ds
        assert ds.time.size == 51

    def test_from_dataset(self, tmp_path):
        """prepare_dataset should accept an xr.Dataset directly."""
        times = np.datetime64("2025-01-01") + np.arange(20).astype("timedelta64[D]")
        battery = np.linspace(100, 80, 20)
        ds_in = xr.Dataset(
            {"m_lithium_battery_relative_charge": ("time", battery)},
            coords={"time": times},
        )
        ds = prepare_dataset(ds_in)
        assert "time" in ds.dims
        assert ds.time.size == 20

    def test_custom_time_var(self, tmp_path):
        """prepare_dataset should handle a non-standard time variable name."""
        nc = tmp_path / "test.nc"
        times = np.datetime64("2025-01-01") + np.arange(20).astype("timedelta64[D]")
        battery = np.linspace(100, 80, 20)
        ds = xr.Dataset(
            {
                "t": ("index", times),
                "m_lithium_battery_relative_charge": ("index", battery),
            }
        )
        ds.to_netcdf(nc)

        result = prepare_dataset(nc, time_var="t")
        assert "time" in result.dims
        assert "t" not in result
        assert result.time.size == 20

    def test_missing_variable(self, tmp_path):
        """prepare_dataset should raise KeyError for missing variables."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        with pytest.raises(KeyError):
            prepare_dataset(nc, sensor="nonexistent")

    def test_nan_filtering(self, tmp_path):
        """prepare_dataset should filter out NaN sensor values."""
        nc = tmp_path / "test.nc"
        times = np.datetime64("2025-01-01") + np.arange(10).astype("timedelta64[D]")
        battery = np.linspace(100, 90, 10)
        battery[3] = np.nan
        battery[7] = np.nan
        ds = xr.Dataset(
            {"m_lithium_battery_relative_charge": ("time", battery)},
            coords={"time": times},
        )
        ds.to_netcdf(nc)

        result = prepare_dataset(nc)
        assert result.time.size == 8

    def test_auto_detect_time_coord(self, tmp_path):
        """Auto-detect should find 'time' as a coordinate."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        ds = prepare_dataset(nc, time_var=None)
        assert "time" in ds.dims
        assert ds.time.size == 51

    def test_auto_detect_t_variable(self, tmp_path):
        """Auto-detect should find 't' by name."""
        nc = tmp_path / "test.nc"
        times = np.datetime64("2025-01-01") + np.arange(20).astype("timedelta64[D]")
        battery = np.linspace(100, 80, 20)
        ds = xr.Dataset(
            {
                "t": ("index", times),
                "m_lithium_battery_relative_charge": ("index", battery),
            }
        )
        ds.to_netcdf(nc)

        result = prepare_dataset(nc, time_var=None)
        assert "time" in result.dims
        assert result.time.size == 20

    def test_auto_detect_posix_timestamp_units(self, tmp_path):
        """Auto-detect should find a float variable with units='timestamp'."""
        nc = tmp_path / "test.nc"
        epoch_start = int(
            (np.datetime64("2025-01-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
        )
        posix_times = epoch_start + np.arange(20, dtype=np.float64) * 86400
        battery = np.linspace(100, 80, 20)
        ds = xr.Dataset(
            {
                "m_present_time": ("i", posix_times),
                "m_lithium_battery_relative_charge": ("i", battery),
            }
        )
        ds["m_present_time"].attrs["units"] = "timestamp"
        ds.to_netcdf(nc)

        result = prepare_dataset(nc, time_var=None)
        assert "time" in result.dims
        assert result.time.size == 20

    def test_auto_detect_cf_units(self, tmp_path):
        """Auto-detect should find a variable with CF time units."""
        nc = tmp_path / "test.nc"
        seconds = np.arange(20, dtype=np.float64) * 86400
        battery = np.linspace(100, 80, 20)
        ds = xr.Dataset(
            {
                "obs_time": ("row", seconds),
                "m_lithium_battery_relative_charge": ("row", battery),
            }
        )
        ds["obs_time"].attrs["units"] = "seconds since 2025-01-01"
        ds.to_netcdf(nc)

        result = prepare_dataset(nc, time_var=None)
        assert "time" in result.dims
        assert result.time.size == 20

    def test_auto_detect_name_ends_with_time(self, tmp_path):
        """Auto-detect should find a variable whose name ends with _time."""
        nc = tmp_path / "test.nc"
        epoch_start = int(
            (np.datetime64("2025-01-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
        )
        posix_times = epoch_start + np.arange(20, dtype=np.float64) * 86400
        battery = np.linspace(100, 80, 20)
        ds = xr.Dataset(
            {
                "sci_m_present_time": ("row", posix_times),
                "m_lithium_battery_relative_charge": ("row", battery),
            }
        )
        ds.to_netcdf(nc)

        result = prepare_dataset(nc, time_var=None)
        assert "time" in result.dims
        assert result.time.size == 20

    def test_auto_detect_no_time_raises(self, tmp_path):
        """Auto-detect should raise KeyError when no time variable is found."""
        nc = tmp_path / "test.nc"
        ds = xr.Dataset(
            {
                "voltage": ("row", np.linspace(100, 80, 20)),
                "m_lithium_battery_relative_charge": ("row", np.linspace(100, 80, 20)),
            }
        )
        ds.to_netcdf(nc)

        with pytest.raises(KeyError, match="Cannot auto-detect"):
            prepare_dataset(nc, time_var=None)

    def test_explicit_time_var_not_found(self, tmp_path):
        """Explicit time_var that doesn't exist should raise KeyError."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        with pytest.raises(KeyError, match="bogus"):
            prepare_dataset(nc, time_var="bogus")


class TestFitRecovery:
    def _make_ds(self, n_days=51, batt_start=100.0, batt_end=50.0):
        """Create a cleaned dataset for testing fit_recovery."""
        times = np.datetime64("2025-01-01") + np.arange(n_days).astype("timedelta64[D]")
        battery = np.linspace(batt_start, batt_end, n_days)
        ds = xr.Dataset(
            {"m_lithium_battery_relative_charge": ("time", battery)},
            coords={"time": times},
        )
        return ds

    def test_basic_fit(self):
        """fit_recovery should return correct results for linear data."""
        ds = self._make_ds()
        result = fit_recovery(ds, threshold=15)
        assert result is not None
        assert result["slope"] == pytest.approx(-1.0)
        assert result["intercept"] == pytest.approx(100.0)
        assert result["r_squared"] == pytest.approx(1.0)
        assert "2025-03-27" in str(result["recovery_date"])
        assert result["ndays"] is None

    def test_with_ndays(self):
        """fit_recovery with ndays should restrict the window."""
        ds = self._make_ds(n_days=100)
        result = fit_recovery(ds, threshold=15, ndays=10)
        assert result is not None
        assert result["n_points"] == 11
        assert result["ndays"] == 10

    def test_with_start_stop(self):
        """fit_recovery with start/stop should restrict the window."""
        ds = self._make_ds(n_days=100)
        result = fit_recovery(ds, threshold=15, start="2025-02-01", stop="2025-03-01")
        assert result is not None
        assert result["n_points"] == 29

    def test_too_few_points(self):
        """fit_recovery should return None with fewer than 3 points."""
        ds = self._make_ds(n_days=2)
        result = fit_recovery(ds, threshold=15)
        assert result is None

    def test_constant_data(self):
        """fit_recovery should return None for constant data (zero slope)."""
        times = np.datetime64("2025-01-01") + np.arange(20).astype("timedelta64[D]")
        battery = np.full(20, 80.0)
        ds = xr.Dataset(
            {"m_lithium_battery_relative_charge": ("time", battery)},
            coords={"time": times},
        )
        result = fit_recovery(ds, threshold=15)
        assert result is None

    def test_result_keys(self):
        """fit_recovery result should contain all expected keys."""
        ds = self._make_ds()
        result = fit_recovery(ds, threshold=15)
        expected_keys = {
            "time",
            "sensor_values",
            "dDays",
            "slope",
            "intercept",
            "slope_ci",
            "intercept_ci",
            "recovery_date",
            "recovery_ci_days",
            "r_squared",
            "pvalue",
            "n_points",
            "dof",
            "threshold",
            "confidence",
            "ndays",
            "tau",
        }
        assert set(result.keys()) == expected_keys

    def test_with_tau(self):
        """fit_recovery with tau should use full data with exponential weights."""
        ds = self._make_ds(n_days=100)
        result = fit_recovery(ds, threshold=15, tau=10)
        assert result is not None
        assert result["tau"] == 10
        assert result["ndays"] is None
        assert result["n_points"] == 100

    def test_tau_with_limited_data(self):
        """tau with only a few days of data should still work."""
        ds = self._make_ds(n_days=5)
        result = fit_recovery(ds, threshold=15, tau=10)
        assert result is not None
        assert result["n_points"] == 5
        assert result["tau"] == 10

    def test_unweighted_dof(self):
        """Without tau, dof should be n - 2."""
        ds = self._make_ds(n_days=51)
        result = fit_recovery(ds, threshold=15)
        assert result["dof"] == 49.0

    def test_tau_reduces_effective_dof(self):
        """With tau, effective dof should be less than n - 2."""
        ds = self._make_ds(n_days=100)
        unweighted = fit_recovery(ds, threshold=15)
        weighted = fit_recovery(ds, threshold=15, tau=5)
        assert weighted["dof"] < unweighted["dof"]
        # tau=5 with 100 days: most old data is heavily downweighted,
        # effective n should be much less than 100
        assert weighted["dof"] < 20

    def test_tau_widens_confidence_intervals(self):
        """With tau on noisy data, CIs should be wider than unweighted."""
        # Non-linear data so CIs are non-zero
        times = np.datetime64("2025-01-01") + np.arange(100).astype("timedelta64[D]")
        rng = np.random.RandomState(42)
        battery = np.linspace(100, 50, 100) + rng.normal(0, 2, 100)
        ds = xr.Dataset(
            {"m_lithium_battery_relative_charge": ("time", battery)},
            coords={"time": times},
        )
        unweighted = fit_recovery(ds, threshold=15)
        weighted = fit_recovery(ds, threshold=15, tau=5)
        assert unweighted is not None and weighted is not None
        assert weighted["recovery_ci_days"] > unweighted["recovery_ci_days"]

    def test_tau_shifts_recovery_date(self):
        """tau weighting on non-linear data should differ from unweighted fit."""
        # Create data with a slope change: steeper recent decay
        times = np.datetime64("2025-01-01") + np.arange(100).astype("timedelta64[D]")
        battery = np.empty(100)
        battery[:50] = np.linspace(100, 80, 50)  # -0.408/day
        battery[50:] = np.linspace(80, 40, 50)  # -0.816/day
        ds = xr.Dataset(
            {"m_lithium_battery_relative_charge": ("time", battery)},
            coords={"time": times},
        )
        unweighted = fit_recovery(ds, threshold=15)
        weighted = fit_recovery(ds, threshold=15, tau=10)
        assert unweighted is not None
        assert weighted is not None
        # tau weighting emphasizes the steeper recent decay
        assert weighted["slope"] < unweighted["slope"]


class TestMultiNdays:
    def test_repeated_flags(self, tmp_path, capsys):
        """--ndays specified multiple times should produce multiple results."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--ndays", "7", "--ndays", "30", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "[7d]" in out
        assert "[30d]" in out
        assert out.count("Recovery By") == 2

    def test_comma_separated(self, tmp_path, capsys):
        """--ndays with comma-separated values should produce multiple results."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--ndays", "7,30", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "[7d]" in out
        assert "[30d]" in out
        assert out.count("Recovery By") == 2

    def test_mixed_repeated_and_comma(self, tmp_path, capsys):
        """Mixing repeated flags and comma-separated values."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--ndays", "3,7", "--ndays", "30", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "[3d]" in out
        assert "[7d]" in out
        assert "[30d]" in out
        assert out.count("Recovery By") == 3

    def test_json_output(self, tmp_path, capsys):
        """--json with multiple ndays should include ndays in each result."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--json", "--ndays", "7,30", "--threshold", "15", str(nc)])
        assert rc == 0

        data = json.loads(capsys.readouterr().out)
        assert len(data) == 2
        assert data[0]["ndays"] == 7.0
        assert data[1]["ndays"] == 30.0

    def test_plot_output(self, tmp_path):
        """--output with multiple ndays should save a plot."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)
        plot_path = tmp_path / "plot.png"

        rc = _run(["--ndays", "7,30", "--output", str(plot_path), str(nc)])
        assert rc == 0
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_single_ndays_no_bracket(self, tmp_path, capsys):
        """Single --ndays should not show bracket label in output."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--ndays", "10", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "[10d]" not in out
        assert "Recovery By" in out

    def test_full_keyword(self, tmp_path, capsys):
        """--ndays full should use the entire dataset."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--ndays", "7,full", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "[7d]" in out
        assert "[full]" in out
        assert out.count("Recovery By") == 2

    def test_full_keyword_json(self, tmp_path, capsys):
        """--ndays full should appear as null ndays in JSON output."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--json", "--ndays", "7,full", "--threshold", "15", str(nc)])
        assert rc == 0

        data = json.loads(capsys.readouterr().out)
        assert len(data) == 2
        assert data[0]["ndays"] == 7.0
        assert data[1]["ndays"] is None

    def test_full_keyword_case_insensitive(self, tmp_path, capsys):
        """--ndays FULL and Full should both work."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--ndays", "FULL", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Recovery By" in out


class TestTau:
    def test_single_tau(self, tmp_path, capsys):
        """Single --tau should produce a result."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--tau", "10", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Recovery By" in out

    def test_multi_tau_comma(self, tmp_path, capsys):
        """Comma-separated --tau should produce multiple results."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--tau", "3,10", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "[\u03c4=3d]" in out
        assert "[\u03c4=10d]" in out
        assert out.count("Recovery By") == 2

    def test_multi_tau_repeated(self, tmp_path, capsys):
        """Repeated --tau flags should produce multiple results."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--tau", "3", "--tau", "10", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "[\u03c4=3d]" in out
        assert "[\u03c4=10d]" in out
        assert out.count("Recovery By") == 2

    def test_ndays_and_tau_combined(self, tmp_path, capsys):
        """--ndays and --tau can be combined for different window types."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--ndays", "7", "--tau", "10", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "[7d]" in out
        assert "[\u03c4=10d]" in out
        assert out.count("Recovery By") == 2

    def test_tau_json(self, tmp_path, capsys):
        """--json with --tau should include tau in results."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--json", "--tau", "5,15", "--threshold", "15", str(nc)])
        assert rc == 0

        data = json.loads(capsys.readouterr().out)
        assert len(data) == 2
        assert data[0]["tau"] == 5.0
        assert data[0]["ndays"] is None
        assert data[1]["tau"] == 15.0

    def test_tau_plot(self, tmp_path):
        """--output with --tau should save a plot."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)
        plot_path = tmp_path / "plot.png"

        rc = _run(["--tau", "3,10", "--output", str(plot_path), str(nc)])
        assert rc == 0
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_tau_start_conflict(self, tmp_path):
        """--tau and --start together should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--tau", "5", "--start", "2025-01-10", str(nc)])
        assert rc == 2

    def test_tau_stop_conflict(self, tmp_path):
        """--tau and --stop together should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--tau", "5", "--stop", "2025-02-01", str(nc)])
        assert rc == 2


class TestInputValidation:
    def test_ndays_non_numeric(self, tmp_path):
        """--ndays with non-numeric value should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--ndays", "abc", str(nc)])
        assert rc == 2

    def test_tau_non_numeric(self, tmp_path):
        """--tau with non-numeric value should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--tau", "xyz", str(nc)])
        assert rc == 2

    def test_ndays_zero(self, tmp_path):
        """--ndays 0 should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--ndays", "0", str(nc)])
        assert rc == 2

    def test_ndays_negative(self, tmp_path):
        """--ndays with negative value should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--ndays", "-5", str(nc)])
        assert rc == 2

    def test_tau_zero(self, tmp_path):
        """--tau 0 should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--tau", "0", str(nc)])
        assert rc == 2

    def test_tau_negative(self, tmp_path):
        """--tau with negative value should be rejected."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc)

        rc = _run(["--tau", "-3", str(nc)])
        assert rc == 2


class TestPlotEdgeCases:
    def test_first_window_fails_still_plots_data(self, tmp_path):
        """If the first ndays window has too few points, raw data should
        still be plotted when a later window succeeds."""
        nc = tmp_path / "test.nc"
        # 10 days of data: ndays=1 gives 2 points (fails), ndays=7 succeeds
        make_linear_nc(nc, n_days=10)
        plot_path = tmp_path / "plot.png"

        rc = _run(["--ndays", "1,7", "--output", str(plot_path), str(nc)])
        assert rc == 0
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_full_keyword_plot_label(self, tmp_path):
        """--ndays full,7 plot should show 'full' label in legend."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)
        plot_path = tmp_path / "plot.png"

        rc = _run(["--ndays", "full,7", "--output", str(plot_path), str(nc)])
        assert rc == 0
        assert plot_path.exists()

    def test_tau_window_fail_log(self, tmp_path, capsys):
        """A tau window that fails should log with tau label and not crash."""
        nc = tmp_path / "test.nc"
        # Only 4 days of data; ndays=1 gives 2 points (fails), tau=1 succeeds
        make_linear_nc(nc, n_days=4)

        rc = _run(["--ndays", "1", "--tau", "1", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Recovery By" in out


class TestSafeSqrt:
    def test_nan_input(self):
        """_safe_sqrt should propagate NaN."""
        result = _safe_sqrt(float("nan"))
        assert np.isnan(result)

    def test_negative_input(self):
        """_safe_sqrt should clamp negative to 0."""
        assert _safe_sqrt(-1.0) == 0.0

    def test_positive_input(self):
        """_safe_sqrt should return sqrt of positive values."""
        assert _safe_sqrt(4.0) == pytest.approx(2.0)


class TestParseEdgeCases:
    def test_trailing_comma(self, tmp_path, capsys):
        """--ndays with trailing comma should be handled gracefully."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--ndays", "7,,30", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "[7d]" in out
        assert "[30d]" in out
        assert out.count("Recovery By") == 2


class TestThinning:
    def _make_bursty_nc(self, path, n_hours=24, points_per_burst=4):
        """Create a NetCDF file with bursty data (multiple points per hour)."""
        sensor = "m_lithium_battery_relative_charge"
        times = []
        values = []
        base = np.datetime64("2025-01-01")
        for h in range(n_hours):
            base_val = 100.0 - h * 0.5  # -0.5 %/hour = -12 %/day
            for j in range(points_per_burst):
                times.append(base + np.timedelta64(h * 3600 + j * 100, "s"))
                values.append(base_val - j * 0.001)  # tiny within-burst variation
        ds = xr.Dataset(
            {sensor: ("time", np.array(values, dtype=np.float64))},
            coords={"time": np.array(times)},
        )
        ds.to_netcdf(path)

    def test_thin_reduces_points(self, tmp_path):
        """Thinning should reduce bursty data to ~1 point per hour."""
        nc = tmp_path / "test.nc"
        self._make_bursty_nc(nc, n_hours=24, points_per_burst=4)

        ds_raw = prepare_dataset(nc, thin=0)
        ds_thin = prepare_dataset(nc, thin=1)
        assert ds_raw.time.size == 96  # 24 * 4
        assert ds_thin.time.size == 24

    def test_thin_creates_bin_stderr(self, tmp_path):
        """Thinning with multi-sample bins should create _bin_stderr."""
        nc = tmp_path / "test.nc"
        self._make_bursty_nc(nc, n_hours=24, points_per_burst=4)

        ds = prepare_dataset(nc, thin=1)
        assert "_bin_stderr" in ds

    def test_thin_disabled(self, tmp_path):
        """thin=0 should disable thinning."""
        nc = tmp_path / "test.nc"
        self._make_bursty_nc(nc, n_hours=24, points_per_burst=4)

        ds = prepare_dataset(nc, thin=0)
        assert ds.time.size == 96
        assert "_bin_stderr" not in ds

    def test_thin_no_stderr_for_daily_data(self, tmp_path):
        """Daily data (1 point per bin) should not create _bin_stderr."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=30)

        ds = prepare_dataset(nc, thin=1)
        assert "_bin_stderr" not in ds

    def test_thin_affects_dof(self, tmp_path):
        """Thinned data should have lower dof than raw (fewer independent points)."""
        nc = tmp_path / "test.nc"
        self._make_bursty_nc(nc, n_hours=48, points_per_burst=5)

        ds_raw = prepare_dataset(nc, thin=0)
        ds_thin = prepare_dataset(nc, thin=1)
        r_raw = fit_recovery(ds_raw, threshold=15)
        r_thin = fit_recovery(ds_thin, threshold=15)
        assert r_raw is not None and r_thin is not None
        assert r_thin["dof"] < r_raw["dof"]

    def test_thin_with_tau(self, tmp_path):
        """Thinning combined with tau should use both weight sources."""
        nc = tmp_path / "test.nc"
        self._make_bursty_nc(nc, n_hours=48, points_per_burst=4)

        ds = prepare_dataset(nc, thin=1)
        r = fit_recovery(ds, threshold=15, tau=1)
        assert r is not None
        # tau reduces dof via Kish; n=48 hourly bins, tau=1d on 2-day span
        assert r["dof"] < 46  # less than n-2

    def test_thin_bin_stderr_weighting(self, tmp_path):
        """Bin stderr should produce valid weights for fitting."""
        nc = tmp_path / "test.nc"
        self._make_bursty_nc(nc, n_hours=48, points_per_burst=5)

        ds_thin = prepare_dataset(nc, thin=1)
        assert "_bin_stderr" in ds_thin
        stderr = ds_thin["_bin_stderr"].values
        # All stderr values should be finite and positive
        assert np.all(np.isfinite(stderr))
        assert np.all(stderr > 0)
        # Fit with bin weights should succeed
        r = fit_recovery(ds_thin, threshold=15)
        assert r is not None

    def test_cli_thin_zero_disables(self, tmp_path, capsys):
        """--thin 0 via CLI should disable thinning."""
        nc = tmp_path / "test.nc"
        self._make_bursty_nc(nc, n_hours=24, points_per_burst=4)

        rc = _run(["--thin", "0", "--threshold", "15", str(nc)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Recovery By" in out


class TestMainModule:
    def test_python_m_version(self):
        """python -m slocum_tpw --version should work."""
        import subprocess

        result = subprocess.run(
            ["python3", "-m", "slocum_tpw", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "slocum-tpw" in result.stdout
