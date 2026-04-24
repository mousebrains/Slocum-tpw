# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

slocum-tpw — TWR Slocum glider data processing utilities, published on PyPI as `slocum-tpw`. GPL-3.0-or-later. Python >= 3.12.

## Build & Install

```bash
pip install -e ".[dev]"    # editable install with dev dependencies
```

## Test Commands

```bash
pytest                           # run all tests
pytest tests/test_slocum_utils.py  # run one test file
pytest -k "test_full_pipeline"   # run tests matching a pattern
pytest --cov=slocum_tpw --cov-report=term-missing  # with coverage
```

## Lint

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Architecture

- **src/slocum_tpw/cli.py** — Single entry point (`slocum-tpw`) with subcommands: `decode-argos`, `log-harvest`, `mk-combined`, `recover-by`, `simulate-leak`, `analyze-leak`. Uses argparse with subparsers. Global `--verbose`/`--debug` flags control logging.
- **src/slocum_tpw/decode_argos.py** — Parses ARGOS satellite position messages into NetCDF. Regex-based line parser. Public API: `proc_file()`, `process_files()`.
- **src/slocum_tpw/log_harvest.py** — Parses Slocum glider log files (binary read, UTF-8 decode). Extracts GPS, sensors, timestamps. Bins to 100-second resolution using hash-table lookup. Public API: `parse_log_file()`, `process_files()`.
- **src/slocum_tpw/mk_combined.py** — Merges log, flight, and science NetCDF data for a single glider. Computes oceanographic variables (depth, salinity, potential temperature, density) via GSW/TEOS-10. Outputs CF-1.13 compliant NetCDF. Public API: `mk_combo()`.
- **src/slocum_tpw/recover_by.py** — Estimates glider recovery date from battery decay. Linear regression on battery charge over time, with confidence intervals via t-distribution. Supports multiple `--ndays` windows (repeated/comma-separated, `full` keyword for entire dataset), `--tau` exponential downweighting (also repeatable/comma-separated), auto-detection of time variables (POSIX floats, CF units, Slocum conventions), `--thin` for thinning bursty data to bin means with stderr weighting, JSON output, and matplotlib plots. Public API: `prepare_dataset()`, `fit_recovery()`, `FIT_COLORS`.
- **src/slocum_tpw/simulate_leak.py** — Simulates sealed-body vacuum and vehicle-temperature observations under a sinusoidal thermal cycle with an optional constant-rate leak, using the van der Waals EOS for air. Writes CSV with Slocum native column names (`m_present_time`, `m_vacuum`, `m_veh_temp`). Public API: `simulate()`, `write_csv()`, `vdw_pressure()`, `vdw_density()` (scalar, `brentq`), `vdw_density_vec()` (vectorized Newton).
- **src/slocum_tpw/analyze_leak.py** — Estimates d(n/V)/dt and its 1-sigma uncertainty from vacuum and temperature observations. Inverts van der Waals per sample to get molar density, then least-squares fits density vs. time. Column names are overridable so it can be run on real glider CSV exports. Public API: `load_csv()`, `fit_leak_rate()`.
- **src/slocum_tpw/slocum_utils.py** — DDMM.MM to decimal degrees conversion. Public API: `mk_degrees_scalar()`, `mk_degrees()`.

Each subcommand module exposes: `add_arguments(parser)`, `run(args) -> int`, and its core processing functions.

## Key Dependencies

numpy, pandas, xarray, gsw (TEOS-10 oceanographic calculations), netcdf4, scipy (t-distribution for recover-by), matplotlib (plotting for recover-by).

## CI/CD

- **ci.yml** — Lint (ruff), test (Python 3.12 + 3.13 matrix with coverage), and build verification. Runs on push/PR to main.
- **release.yml** — Triggered by `v*` tags. Publishes to test.pypi.org, then creates a GitHub release.
- **dependabot.yml** — Monthly updates for GitHub Actions and pip dependencies.

To release: bump `__version__` in `src/slocum_tpw/__init__.py`, commit, then `git tag v0.1.0 && git push --tags`.

## Conventions

- src layout (`src/slocum_tpw/`)
- snake_case for public function names
- Each subcommand module has `add_arguments(parser)` + `run(args) -> int` + core functions
- Tests mirror source structure in `tests/`
- Log-harvest filenames must follow `{glider}_{timestamp}_*.log` pattern for t0 filtering
- README.md serves as the PyPI long description — keep it comprehensive with full CLI and API docs
