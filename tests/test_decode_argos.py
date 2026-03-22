"""Tests for slocum_tpw.decode_argos."""

import datetime

import numpy as np
import pytest
import xarray as xr

from slocum_tpw.decode_argos import proc_file, process_files

from .conftest import (
    ARGOS_LINE_BAD_DATE,
    ARGOS_LINE_MALFORMED,
    ARGOS_LINE_VALID,
    ARGOS_LINE_VALID_2,
)


class TestProcFile:
    def test_valid_single_line(self, tmp_path):
        fn = tmp_path / "argos.txt"
        fn.write_text(ARGOS_LINE_VALID + "\n")
        df = proc_file(str(fn))
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["ident"] == 67890
        assert df.iloc[0]["lat"] == pytest.approx(44.123)
        assert df.iloc[0]["lon"] == pytest.approx(124.567)
        assert df.iloc[0]["satellite"] == "A"
        assert df.iloc[0]["locationClass"] == "3"

    def test_valid_multiple_lines(self, tmp_path):
        fn = tmp_path / "argos.txt"
        fn.write_text(ARGOS_LINE_VALID + "\n" + ARGOS_LINE_VALID_2 + "\n")
        df = proc_file(str(fn))
        assert df is not None
        assert len(df) == 2
        assert df.iloc[0]["ident"] == 67890
        assert df.iloc[1]["ident"] == 67891

    def test_empty_file(self, tmp_path):
        fn = tmp_path / "empty.txt"
        fn.write_text("")
        assert proc_file(str(fn)) is None

    def test_malformed_lines_skipped(self, tmp_path):
        fn = tmp_path / "mixed.txt"
        fn.write_text(ARGOS_LINE_MALFORMED + "\n" + ARGOS_LINE_VALID + "\n")
        df = proc_file(str(fn))
        assert df is not None
        assert len(df) == 1

    def test_bad_date_skipped(self, tmp_path):
        fn = tmp_path / "bad_date.txt"
        fn.write_text(ARGOS_LINE_BAD_DATE + "\n")
        assert proc_file(str(fn)) is None

    def test_timestamp_has_utc(self, tmp_path):
        """proc_file returns timezone-aware UTC timestamps."""
        fn = tmp_path / "argos.txt"
        fn.write_text(ARGOS_LINE_VALID + "\n")
        df = proc_file(str(fn))
        assert df is not None
        t = df.iloc[0]["time"]
        assert t.tzinfo == datetime.timezone.utc
        assert t.year == 2023
        assert t.month == 6
        assert t.day == 15

    def test_only_malformed_returns_none(self, tmp_path):
        fn = tmp_path / "junk.txt"
        fn.write_text(ARGOS_LINE_MALFORMED + "\n" + "more junk\n")
        assert proc_file(str(fn)) is None


class TestProcessFiles:
    def test_writes_netcdf(self, tmp_path):
        fn = tmp_path / "argos.txt"
        fn.write_text(ARGOS_LINE_VALID + "\n")
        nc = tmp_path / "output.nc"
        process_files(str(nc), [str(fn)])
        assert nc.exists()
        with xr.open_dataset(str(nc)) as ds:
            assert "ident" in ds

    def test_multiple_files(self, tmp_path):
        fn1 = tmp_path / "argos1.txt"
        fn1.write_text(ARGOS_LINE_VALID + "\n")
        fn2 = tmp_path / "argos2.txt"
        fn2.write_text(ARGOS_LINE_VALID_2 + "\n")
        nc = tmp_path / "output.nc"
        process_files(str(nc), [str(fn1), str(fn2)])
        with xr.open_dataset(str(nc)) as ds:
            assert len(ds.time) == 2

    def test_empty_files_write_empty_netcdf(self, tmp_path):
        fn = tmp_path / "empty.txt"
        fn.write_text("")
        nc = tmp_path / "output.nc"
        process_files(str(nc), [str(fn)])
        assert nc.exists()

    def test_no_files_writes_empty_netcdf(self, tmp_path):
        nc = tmp_path / "output.nc"
        process_files(str(nc), [])
        assert nc.exists()

    def test_missing_file_skipped(self, tmp_path):
        """A nonexistent file should be logged and skipped, not crash."""
        nc = tmp_path / "output.nc"
        process_files(str(nc), [str(tmp_path / "nonexistent.txt")])
        assert nc.exists()

    def test_frequency_is_integer(self, tmp_path):
        """Frequency field should be stored as int, not float."""
        fn = tmp_path / "argos.txt"
        fn.write_text(ARGOS_LINE_VALID + "\n")
        df = proc_file(str(fn))
        assert df is not None
        assert df.iloc[0]["frequency"] == 401650000
        assert isinstance(df.iloc[0]["frequency"], (int, np.integer))
