"""Tests for slocum_tpw.simulate_leak."""

import csv

import numpy as np
import pytest

from slocum_tpw.simulate_leak import (
    INHG_TO_PA,
    P_ATM_PA,
    R,
    simulate,
    vdw_density,
    vdw_pressure,
    write_csv,
)


class TestVdw:
    def test_roundtrip(self):
        T = 293.15
        P = P_ATM_PA - 10.0 * INHG_TO_PA
        rho = vdw_density(P, T)
        assert vdw_pressure(rho, T) == pytest.approx(P, rel=1e-10)

    def test_ideal_gas_limit_at_low_density(self):
        """Van der Waals should be near ideal for low-pressure air."""
        T = 293.15
        P = P_ATM_PA - 10.0 * INHG_TO_PA
        rho = vdw_density(P, T)
        rho_ideal = P / (R * T)
        assert rho == pytest.approx(rho_ideal, rel=1e-3)


class TestSimulate:
    def test_noise_free_periodicity(self):
        # Noise-free, no leak, 1 day: vacuum at t=0 and t=24h match to <1e-6 inHg
        r = simulate(
            days=1.0, timestep=60.0,
            sigma_pressure=0.0, sigma_temperature=0.0,
            vacuum_drop_per_day=0.0,
        )
        assert r["vacuum_true_inHg"][0] == pytest.approx(10.0, abs=1e-6)
        idx24 = int(round(86400.0 / 60.0))
        assert r["vacuum_true_inHg"][idx24] == pytest.approx(
            r["vacuum_true_inHg"][0], abs=1e-6
        )

    def test_leak_produces_expected_total_drop(self):
        """0.075 inHg/day over 4 days should leave vacuum 0.3 inHg lower at T0."""
        r = simulate(
            days=4.0, timestep=3600.0, vacuum_drop_per_day=0.075,
            sigma_pressure=0.0, sigma_temperature=0.0,
        )
        # Last sample is at 4*86400 s, temperature back at maximum (T0)
        assert r["vacuum_true_inHg"][-1] == pytest.approx(9.7, abs=1e-4)

    def test_drho_dt_matches_ideal_gas_estimate(self):
        """For 0.3 inHg drop over 4 days, truth rate should match ideal-gas."""
        r = simulate(
            days=4.0, timestep=3600.0, vacuum_drop_per_day=0.075,
            sigma_pressure=0.0, sigma_temperature=0.0,
        )
        T0 = 20.0 + 273.15
        dP_Pa = 0.3 * INHG_TO_PA
        duration_s = 4.0 * 86400.0
        expected = dP_Pa / (R * T0) / duration_s  # mol/m^3/s
        assert r["drho_dt_true"] == pytest.approx(expected, rel=5e-3)

    def test_seed_reproducibility(self):
        r1 = simulate(days=0.1, timestep=60.0, seed=42)
        r2 = simulate(days=0.1, timestep=60.0, seed=42)
        np.testing.assert_allclose(r1["vacuum_inHg"], r2["vacuum_inHg"])
        np.testing.assert_allclose(r1["temperature_c"], r2["temperature_c"])

    def test_zero_sigma_gives_truth(self):
        r = simulate(
            days=0.1, timestep=60.0,
            sigma_pressure=0.0, sigma_temperature=0.0, seed=7,
        )
        np.testing.assert_allclose(r["vacuum_inHg"], r["vacuum_true_inHg"])
        np.testing.assert_allclose(r["temperature_c"], r["temperature_true_c"])

    def test_t0_epoch_offset(self):
        r = simulate(days=0.01, timestep=3.0, t0_epoch=1_700_000_000.0, seed=3)
        assert r["time"][0] == pytest.approx(1_700_000_000.0)
        assert r["time"][-1] == pytest.approx(1_700_000_000.0 + 0.01 * 86400.0)

    def test_no_leak_has_zero_truth_rate(self):
        r = simulate(days=1.0, timestep=60.0, vacuum_drop_per_day=0.0, seed=1)
        assert r["drho_dt_true"] == 0.0


class TestWriteCsv:
    def test_columns_and_row_count(self, tmp_path):
        r = simulate(days=0.01, timestep=3.0, seed=1)
        fn = tmp_path / "out.csv"
        write_csv(str(fn), r["time"], r["vacuum_inHg"], r["temperature_c"])

        with open(fn) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["m_present_time", "m_vacuum", "m_veh_temp"]
            rows = list(reader)
        assert len(rows) == r["time"].size

    def test_values_roundtrip_through_csv(self, tmp_path):
        r = simulate(days=0.01, timestep=3.0, seed=5)
        fn = tmp_path / "out.csv"
        write_csv(str(fn), r["time"], r["vacuum_inHg"], r["temperature_c"])

        with open(fn) as f:
            reader = csv.DictReader(f)
            first = next(reader)
        assert float(first["m_present_time"]) == pytest.approx(r["time"][0], abs=1e-3)
        assert float(first["m_vacuum"]) == pytest.approx(r["vacuum_inHg"][0], abs=1e-6)
        assert float(first["m_veh_temp"]) == pytest.approx(r["temperature_c"][0], abs=1e-4)
