"""Tests for slocum_tpw.analyze_leak."""

import numpy as np
import pytest

from slocum_tpw.analyze_leak import fit_leak_rate, load_csv
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
