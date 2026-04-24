# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `simulate-leak` subcommand: simulate sealed-body vacuum and vehicle
  temperature observations for a Slocum Glider with a sinusoidal thermal
  cycle and optional constant-rate leak, using the van der Waals equation of
  state for air.  Writes CSV with Slocum native column names
  (`m_present_time`, `m_vacuum`, `m_veh_temp`).
- `analyze-leak` subcommand: estimate d(n/V)/dt and its 1-sigma uncertainty
  from a CSV of sealed-body observations by inverting van der Waals per
  sample and least-squares fitting the inferred molar density vs. time.
  Column names are overridable so it can be run on real glider CSV exports.
- Public APIs `slocum_tpw.simulate_leak` (`simulate()`, `write_csv()`,
  `vdw_pressure()`, `vdw_density()`) and `slocum_tpw.analyze_leak`
  (`load_csv()`, `fit_leak_rate()`).

## [0.1.5] - 2026-03-26

### Added
- `--thin` for thinning bursty data to bin means with within-bin stderr as
  inverse-variance fit weights (default: 1 hour, `--thin 0` to disable)
- Precision vs importance weight distinction: Kish's DOF correction applies
  only to `--tau` importance weights, not bin-stderr precision weights
- Python 3.14 added to CI test matrix and PyPI classifiers

### Changed
- `codecov-action` updated from v5 to v6 (Node.js 24)

## [0.1.4] - 2026-03-26

### Added
- Public API for `recover-by`: `prepare_dataset()`, `fit_recovery()`, `FIT_COLORS`
- `--ndays` accepts multiple values (repeatable and/or comma-separated, e.g. `--ndays 3,7,full`)
- `--ndays full` keyword to include the entire dataset alongside windowed fits
- `--tau` for exponential downweighting of historical data (weight = exp(-age/tau)), also repeatable/comma-separated
- Auto-detection of time variables (searches `time`, `t`, datetime64 dtypes, CF time units, `units='timestamp'`, names ending in `_time`); `--time` now optional
- Kish's effective degrees of freedom for tau-weighted fits, with covariance matrix rescaling for honest confidence intervals
- DOF shown in text output, JSON, and plot legends
- Per-subplot titles showing filename and data point count
- Input validation for `--ndays`/`--tau` (positive, numeric)

### Changed
- `--time` default changed from `"time"` to auto-detect
- `prepare_dataset()` `time_var` parameter default changed from `"time"` to `None`
- Compact text output (4 lines per result instead of 7)
- Compact plot legend (single line per fit, recovery date first)
- Plot uses `fig.suptitle` for sensor/threshold, `ax.set_title` for per-file info

### Fixed
- Plot raw-data gate used `win_idx == 0`; if the first window failed, raw data was never plotted
- Non-numeric `--ndays`/`--tau` values caused uncaught traceback instead of clean error

## [0.1.3] - 2026-03-25

### Changed
- Speed up `log-harvest` with incremental processing and parse optimizations

## [0.1.2] - 2026-03-22

### Changed
- Add pypi.org publishing to release workflow

## [0.1.1] - 2026-03-22

### Changed
- Bump GitHub Actions to Node.js 24 compatible versions

## [0.1.0] - 2026-03-22

### Added
- `decode-argos` subcommand: decode ARGOS satellite position messages into NetCDF
- `log-harvest` subcommand: parse Slocum glider log files into NetCDF
- `mk-combined` subcommand: merge log, flight, and science data into CF-1.13 NetCDF with TEOS-10 oceanographic calculations (depth, salinity, potential temperature, density)
- `recover-by` subcommand: estimate glider recovery date from battery decay via linear regression with confidence intervals
- Unified CLI entry point (`slocum-tpw`) with `--verbose`/`--debug` logging
- Full test suite (84+ tests, 96% coverage)
- CI/CD with GitHub Actions (lint, test matrix, build verification, PyPI publishing)
- Pre-commit hooks (ruff lint/format, YAML/TOML validation)

[Unreleased]: https://github.com/mousebrains/Slocum-tpw/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/mousebrains/Slocum-tpw/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/mousebrains/Slocum-tpw/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/mousebrains/Slocum-tpw/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/mousebrains/Slocum-tpw/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/mousebrains/Slocum-tpw/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mousebrains/Slocum-tpw/releases/tag/v0.1.0
