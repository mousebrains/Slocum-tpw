# slocum-tpw

[![PyPI version](https://img.shields.io/pypi/v/slocum-tpw)](https://pypi.org/project/slocum-tpw/)
[![Python](https://img.shields.io/pypi/pyversions/slocum-tpw)](https://pypi.org/project/slocum-tpw/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Tests](https://img.shields.io/github/actions/workflow/status/mousebrains/Slocum-tpw/ci.yml?label=tests)](https://github.com/mousebrains/Slocum-tpw/actions)
[![codecov](https://img.shields.io/codecov/c/github/mousebrains/Slocum-tpw)](https://codecov.io/gh/mousebrains/Slocum-tpw)

TWR Slocum glider data processing utilities.

## Installation

```bash
pip install slocum-tpw
```

## Usage

```bash
slocum-tpw decode-argos --nc output.nc argos_messages.txt
slocum-tpw log-harvest --nc log.nc osu684_*.log
slocum-tpw mk-combined --glider 684 --output osu684.nc --nc-log log.nc
```

### Subcommands

| Command | Description |
|---|---|
| `decode-argos` | Decode ARGOS satellite position messages into NetCDF |
| `log-harvest` | Parse Slocum glider log files and extract GPS, sensors, timestamps into NetCDF |
| `mk-combined` | Merge log, flight, and science data into CF-1.13 compliant NetCDF with derived oceanographic variables |

### Global Options

| Flag | Effect |
|---|---|
| `--verbose` | Enable INFO-level logging |
| `--debug` | Enable DEBUG-level logging |
| `--version` | Show version and exit |

## Development

```bash
git clone https://github.com/mousebrains/Slocum-tpw.git
cd Slocum-tpw
pip install -e ".[dev]"
pytest
```

## License

[GPL-3.0-or-later](LICENSE)
