# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-03-22

### Added
- `decode-argos` subcommand: decode ARGOS satellite position messages into NetCDF
- `log-harvest` subcommand: parse Slocum glider log files into NetCDF
- `mk-combined` subcommand: merge log, flight, and science data into CF-1.13 NetCDF with TEOS-10 oceanographic calculations (depth, salinity, potential temperature, density)
- `recover-by` subcommand: estimate glider recovery date from battery decay via linear regression with confidence intervals
- Unified CLI entry point (`slocum-tpw`) with `--verbose`/`--debug` logging
- Full test suite (84+ tests, 96% coverage)
- CI/CD with GitHub Actions (lint, test matrix, build verification, test.pypi.org publishing)
- Pre-commit hooks (ruff lint/format, YAML/TOML validation)

[Unreleased]: https://github.com/mousebrains/Slocum-tpw/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mousebrains/Slocum-tpw/releases/tag/v0.1.0
