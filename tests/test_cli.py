"""Tests for slocum_tpw.cli."""

import numpy as np
import pytest
import xarray as xr

from slocum_tpw.cli import main

from .conftest import ARGOS_LINE_VALID, SAMPLE_LOG


class TestCLI:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "slocum-tpw" in captured.out

    def test_no_subcommand_exits(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2

    def test_decode_argos_subcommand(self, tmp_path):
        fn = tmp_path / "argos.txt"
        fn.write_text(ARGOS_LINE_VALID + "\n")
        nc = tmp_path / "output.nc"
        with pytest.raises(SystemExit) as exc_info:
            main(["decode-argos", "--nc", str(nc), str(fn)])
        assert exc_info.value.code == 0
        assert nc.exists()

    def test_log_harvest_subcommand(self, tmp_path):
        fn = tmp_path / "osu684_20230612T000000_test.log"
        fn.write_bytes(SAMPLE_LOG)
        nc = tmp_path / "log.nc"
        with pytest.raises(SystemExit) as exc_info:
            main(["log-harvest", "--nc", str(nc), str(fn)])
        assert exc_info.value.code == 0
        assert nc.exists()

    def test_mk_combined_subcommand(self, tmp_path, log_nc, flt_nc, sci_nc):
        output = tmp_path / "combined.nc"
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "mk-combined",
                    "--glider",
                    "684",
                    "--output",
                    str(output),
                    "--nc-log",
                    str(log_nc),
                    "--nc-flight",
                    str(flt_nc),
                    "--nc-science",
                    str(sci_nc),
                ]
            )
        assert exc_info.value.code == 0
        assert output.exists()

    def test_recover_by_subcommand(self, tmp_path):
        nc = tmp_path / "test.nc"
        times = np.datetime64("2025-01-01") + np.arange(51).astype("timedelta64[D]")
        battery = np.linspace(100, 50, 51)
        ds = xr.Dataset(
            {"m_lithium_battery_relative_charge": ("time", battery)},
            coords={"time": times},
        )
        ds.to_netcdf(nc)
        with pytest.raises(SystemExit) as exc_info:
            main(["recover-by", "--threshold", "15", str(nc)])
        assert exc_info.value.code == 0

    def test_verbose_flag(self, tmp_path):
        fn = tmp_path / "argos.txt"
        fn.write_text(ARGOS_LINE_VALID + "\n")
        nc = tmp_path / "output.nc"
        with pytest.raises(SystemExit) as exc_info:
            main(["--verbose", "decode-argos", "--nc", str(nc), str(fn)])
        assert exc_info.value.code == 0

    def test_debug_flag(self, tmp_path):
        fn = tmp_path / "argos.txt"
        fn.write_text(ARGOS_LINE_VALID + "\n")
        nc = tmp_path / "output.nc"
        with pytest.raises(SystemExit) as exc_info:
            main(["--debug", "decode-argos", "--nc", str(nc), str(fn)])
        assert exc_info.value.code == 0

    def test_mk_combined_missing_glider_and_flight(self, tmp_path, log_nc):
        """Without --glider and --nc-flight, should fail."""
        output = tmp_path / "combined.nc"
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "mk-combined",
                    "--output",
                    str(output),
                    "--nc-log",
                    str(log_nc),
                ]
            )
        assert exc_info.value.code == 1

    def test_mk_combined_auto_generates_paths(self, tmp_path, log_nc, flt_nc, sci_nc):
        """With --glider, ncFlight and ncScience are auto-derived from ncLog path."""
        output = tmp_path / "combined.nc"
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "mk-combined",
                    "--glider",
                    "684",
                    "--output",
                    str(output),
                    "--nc-log",
                    str(log_nc),
                ]
            )
        # Should succeed since flt.osu684.nc and sci.osu684.nc exist in tmp_path
        assert exc_info.value.code == 0

    def test_mk_combined_missing_glider_and_science(self, tmp_path, log_nc):
        """Without --glider and --nc-science, should fail."""
        output = tmp_path / "combined.nc"
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "mk-combined",
                    "--output",
                    str(output),
                    "--nc-log",
                    str(log_nc),
                    "--nc-flight",
                    str(tmp_path / "flt.nc"),
                ]
            )
        assert exc_info.value.code == 1
