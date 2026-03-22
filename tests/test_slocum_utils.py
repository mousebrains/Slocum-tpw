"""Tests for slocum_tpw.slocum_utils."""

import numpy as np
import pytest

from slocum_tpw.slocum_utils import mk_degrees, mk_degrees_scalar


class TestMkDegreesScalar:
    def test_positive_with_minutes(self):
        # 44 degrees 30 minutes = 44.5 degrees
        assert mk_degrees_scalar(4430.0) == pytest.approx(44.5)

    def test_negative_with_minutes(self):
        assert mk_degrees_scalar(-4430.0) == pytest.approx(-44.5)

    def test_zero(self):
        assert mk_degrees_scalar(0.0) == 0.0

    def test_only_minutes(self):
        # 0 degrees 30 minutes = 0.5 degrees
        assert mk_degrees_scalar(30.0) == pytest.approx(0.5)

    def test_no_minutes(self):
        # 44 degrees 0 minutes = 44.0 degrees
        assert mk_degrees_scalar(4400.0) == pytest.approx(44.0)

    def test_typical_longitude(self):
        # -124 degrees 6 minutes = -124.1 degrees
        assert mk_degrees_scalar(-12406.0) == pytest.approx(-124.1)

    def test_small_value(self):
        # 1 degree 30 minutes = 1.5 degrees
        assert mk_degrees_scalar(130.0) == pytest.approx(1.5)

    def test_fractional_minutes(self):
        # 44 degrees 30.5 minutes
        expected = 44.0 + 30.5 / 60
        assert mk_degrees_scalar(4430.5) == pytest.approx(expected)

    def test_over_180_returns_nan(self):
        import math

        assert math.isnan(mk_degrees_scalar(99900.0))


class TestMkDegrees:
    def test_basic_array(self):
        arr = np.array([4430.0, -4430.0, 0.0])
        result = mk_degrees(arr)
        expected = np.array([44.5, -44.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_values_become_nan(self):
        # 999 degrees is > 180, should become NaN
        arr = np.array([4430.0, 99900.0])
        result = mk_degrees(arr)
        assert result[0] == pytest.approx(44.5)
        assert np.isnan(result[1])

    def test_boundary_180(self):
        # Exactly 180 degrees should be kept
        # 180 degrees 0 minutes = 18000.0 DDMM.MM
        arr = np.array([18000.0])
        result = mk_degrees(arr)
        assert result[0] == pytest.approx(180.0)

    def test_just_over_180_is_nan(self):
        # 180 degrees 0.1 minutes → slightly > 180 → NaN
        arr = np.array([18000.1])
        result = mk_degrees(arr)
        assert np.isnan(result[0])

    def test_empty_array(self):
        result = mk_degrees(np.array([]))
        assert len(result) == 0

    def test_all_negative(self):
        arr = np.array([-12400.0, -12406.0, -12412.0])
        result = mk_degrees(arr)
        expected = np.array([-124.0, -124.1, -124.2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_preserves_array_length(self):
        arr = np.array([4430.0, 4436.0, 4442.0, 4448.0])
        result = mk_degrees(arr)
        assert len(result) == 4
