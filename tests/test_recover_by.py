"""Tests for slocum_tpw.recover_by."""

import json

import numpy as np
import pytest
import xarray as xr

from slocum_tpw.cli import main
from slocum_tpw.recover_by import fit_recovery, prepare_dataset


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
        assert "R-squared:         1.0000" in out

    def test_ndays_filter(self, tmp_path, capsys):
        """Using --ndays should restrict the fit window."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--ndays", "10", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Slope (95%, /day):" in out
        assert "Intercept (95%):" in out
        assert "R-squared:" in out
        assert "Recovery By (95%):" in out

    def test_start_stop_filter(self, tmp_path, capsys):
        """Using --start/--stop should restrict the fit window."""
        nc = tmp_path / "test.nc"
        make_linear_nc(nc, n_days=100)

        rc = _run(["--start", "2025-01-10", "--stop", "2025-02-10", "--threshold", "15", str(nc)])
        assert rc == 0

        out = capsys.readouterr().out
        assert "Slope (95%, /day):" in out
        assert "Recovery By (95%):" in out

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
        assert "Intercept (99%):" in out
        assert "Slope (99%, /day):" in out
        assert "Recovery By (99%):" in out

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
