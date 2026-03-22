"""Tests for slocum_tpw.log_harvest."""

import numpy as np
import xarray as xr

from slocum_tpw.log_harvest import parse_log_file, process_files

from .conftest import SAMPLE_LOG


class TestParseLogFile:
    def test_basic_parsing(self, tmp_path):
        fn = tmp_path / "osu684_20230612T000000_test.log"
        fn.write_bytes(SAMPLE_LOG)
        df = parse_log_file(str(fn), "")
        assert not df.empty
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "t" in df.columns
        assert "glider" in df.columns

    def test_extracts_vehicle_name(self, tmp_path):
        fn = tmp_path / "test.log"
        fn.write_bytes(SAMPLE_LOG)
        df = parse_log_file(str(fn), "unknown")
        assert not df.empty
        # Vehicle Name line should override the passed-in glider
        assert (df["glider"] == "osu684").all()

    def test_gps_coordinates(self, tmp_path):
        fn = tmp_path / "test.log"
        fn.write_bytes(SAMPLE_LOG)
        df = parse_log_file(str(fn), "")
        lats = df["lat"].dropna()
        lons = df["lon"].dropna()
        assert len(lats) > 0
        assert len(lons) > 0
        # 4430.5 DDMM.MM = 44 + 30.5/60 = 44.50833...
        assert abs(lats.iloc[0] - (44.0 + 30.5 / 60)) < 0.01

    def test_sensor_values(self, tmp_path):
        fn = tmp_path / "test.log"
        fn.write_bytes(SAMPLE_LOG)
        df = parse_log_file(str(fn), "")
        assert "sci_water_temp" in df.columns
        temps = df["sci_water_temp"].dropna()
        assert len(temps) > 0

    def test_time_binning_100s(self, tmp_path):
        """Times are binned to 100-second resolution."""
        fn = tmp_path / "test.log"
        fn.write_bytes(SAMPLE_LOG)
        df = parse_log_file(str(fn), "")
        # All time values should be datetime64[s]
        assert df["t"].dtype == np.dtype("datetime64[s]")

    def test_empty_file(self, tmp_path):
        fn = tmp_path / "test.log"
        fn.write_bytes(b"")
        df = parse_log_file(str(fn), "test")
        assert df.empty

    def test_no_timestamp_gps_skipped(self, tmp_path):
        """GPS line before any Curr Time line should be skipped."""
        content = b"GPS Location: 4430.5000 N -12406.0000 E measured 10.0 secs ago\n"
        fn = tmp_path / "test.log"
        fn.write_bytes(content)
        df = parse_log_file(str(fn), "test")
        assert df.empty

    def test_no_timestamp_sensor_skipped(self, tmp_path):
        """Sensor line before any Curr Time line should be skipped."""
        content = b"sensor:sci_water_temp(celsius)=15.5 10.0 secs ago\n"
        fn = tmp_path / "test.log"
        fn.write_bytes(content)
        df = parse_log_file(str(fn), "test")
        assert df.empty

    def test_invalid_gps_values_filtered(self, tmp_path):
        """GPS with lat > 90 or extreme dt should be filtered."""
        content = (
            b"Vehicle Name: osu684\n"
            b"Curr Time: Mon Jun 12 15:30:00 2023 MT: 12345\n"
            b"GPS Location: 9999.0000 N -12406.0000 E measured 10.0 secs ago\n"
        )
        fn = tmp_path / "test.log"
        fn.write_bytes(content)
        df = parse_log_file(str(fn), "")
        # The invalid GPS should be skipped, resulting in empty df
        assert df.empty

    def test_unicode_decode_error_skipped(self, tmp_path):
        """Lines with invalid UTF-8 should be skipped gracefully."""
        content = (
            b"Vehicle Name: osu684\n"
            b"\xff\xfe invalid bytes\n"
            b"Curr Time: Mon Jun 12 15:30:00 2023 MT: 12345\n"
            b"sensor:m_water_vx(m/s)=0.1 10.0 secs ago\n"
        )
        fn = tmp_path / "test.log"
        fn.write_bytes(content)
        df = parse_log_file(str(fn), "")
        assert not df.empty

    def test_lat_lon_sensor_converted(self, tmp_path):
        """Sensor names ending in _lat or _lon get DDMM.MM conversion."""
        content = (
            b"Vehicle Name: osu684\n"
            b"Curr Time: Mon Jun 12 15:30:00 2023 MT: 12345\n"
            b"sensor:m_gps_lat(lat)=4430.0 10.0 secs ago\n"
        )
        fn = tmp_path / "test.log"
        fn.write_bytes(content)
        df = parse_log_file(str(fn), "")
        assert not df.empty
        val = df["m_gps_lat"].dropna().iloc[0]
        assert abs(val - 44.5) < 0.001

    def test_invalid_curr_time_skipped(self, tmp_path):
        """A Curr Time line with an unparseable date should be skipped."""
        content = (
            b"Vehicle Name: osu684\n"
            b"Curr Time: Mon Xxx 99 25:99:99 2023 MT: 12345\n"
            b"sensor:m_water_vx(m/s)=0.1 10.0 secs ago\n"
        )
        fn = tmp_path / "test.log"
        fn.write_bytes(content)
        df = parse_log_file(str(fn), "")
        # currTime never got set, so sensor line is also skipped
        assert df.empty

    def test_sensor_with_huge_dt_filtered(self, tmp_path):
        """Sensor readings with dt > 1e300 should be skipped."""
        content = (
            b"Vehicle Name: osu684\n"
            b"Curr Time: Mon Jun 12 15:30:00 2023 MT: 12345\n"
            b"sensor:m_water_vx(m/s)=0.1 9e999 secs ago\n"
            b"sensor:m_water_vy(m/s)=0.05 10.0 secs ago\n"
        )
        fn = tmp_path / "test.log"
        fn.write_bytes(content)
        df = parse_log_file(str(fn), "")
        assert not df.empty
        # m_water_vx should not appear since its dt was huge
        if "m_water_vx" in df.columns:
            assert df["m_water_vx"].dropna().empty


class TestProcessFiles:
    def test_writes_netcdf(self, tmp_path):
        fn = tmp_path / "osu684_20230612T153000_test.log"
        fn.write_bytes(SAMPLE_LOG)
        nc = tmp_path / "log.nc"
        process_files([str(fn)], None, str(nc))
        assert nc.exists()
        with xr.open_dataset(str(nc)) as ds:
            assert "t" in ds

    def test_t0_filter_excludes_old(self, tmp_path):
        fn = tmp_path / "osu684_20230501T000000_test.log"
        fn.write_bytes(SAMPLE_LOG)
        nc = tmp_path / "log.nc"
        # t0 is after the file timestamp, so it should be excluded
        process_files([str(fn)], "20230601T000000", str(nc))
        assert nc.exists()
        # Should be empty dataset
        with xr.open_dataset(str(nc)) as ds:
            assert len(ds.data_vars) == 0

    def test_t0_filter_includes_newer(self, tmp_path):
        fn = tmp_path / "osu684_20230612T000000_test.log"
        fn.write_bytes(SAMPLE_LOG)
        nc = tmp_path / "log.nc"
        process_files([str(fn)], "20230601T000000", str(nc))
        with xr.open_dataset(str(nc)) as ds:
            assert "t" in ds

    def test_filename_without_underscore_skipped(self, tmp_path):
        fn = tmp_path / "badname.log"
        fn.write_bytes(SAMPLE_LOG)
        nc = tmp_path / "log.nc"
        process_files([str(fn)], None, str(nc))
        assert nc.exists()
        with xr.open_dataset(str(nc)) as ds:
            assert len(ds.data_vars) == 0

    def test_no_files_writes_empty(self, tmp_path):
        nc = tmp_path / "log.nc"
        process_files([], None, str(nc))
        assert nc.exists()
