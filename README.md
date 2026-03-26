# slocum-tpw

[![PyPI version](https://img.shields.io/pypi/v/slocum-tpw)](https://pypi.org/project/slocum-tpw/)
[![Python](https://img.shields.io/pypi/pyversions/slocum-tpw)](https://pypi.org/project/slocum-tpw/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Tests](https://img.shields.io/github/actions/workflow/status/mousebrains/Slocum-tpw/ci.yml?label=tests)](https://github.com/mousebrains/Slocum-tpw/actions)
[![codecov](https://img.shields.io/codecov/c/github/mousebrains/Slocum-tpw)](https://codecov.io/gh/mousebrains/Slocum-tpw)

Command-line tools and Python library for processing
[TWR Slocum](https://www.teledynemarine.com/brands/webb-research) glider data.
Includes ARGOS message decoding, glider log harvesting, multi-source data
merging with TEOS-10 oceanographic calculations, and battery-based recovery
date estimation.

## Installation

```bash
pip install slocum-tpw
```

Requires Python 3.12 or later.

## Quick Start

```bash
# Decode ARGOS satellite positions
slocum-tpw decode-argos --nc argos.nc messages/*.txt

# Harvest glider log files
slocum-tpw log-harvest --nc log.nc osu684_*.log

# Combine log, flight, and science data into CF-1.13 NetCDF
slocum-tpw mk-combined --glider 684 --output osu684.nc

# Estimate recovery date from battery decay
slocum-tpw recover-by --threshold 15 flight.nc
```

All subcommands support `--verbose` (INFO) and `--debug` (DEBUG) logging flags.

---

## CLI Reference

### `slocum-tpw decode-argos`

Decode ARGOS satellite position messages into NetCDF.

```
slocum-tpw decode-argos [--nc OUTPUT] FILE [FILE ...]
```

| Argument | Description |
|---|---|
| `FILE` | One or more ARGOS message files to decode (required) |
| `--nc OUTPUT` | Output NetCDF filename (default: `tpw.nc`) |

Each input file is parsed line-by-line for ARGOS position records containing
ident, satellite ID, location class, timestamp, latitude, longitude, altitude,
and frequency. Results are concatenated and written as an xarray Dataset indexed
by time.

**Example:**

```bash
slocum-tpw decode-argos --nc argos.nc incoming/argos_2025*.txt
```

---

### `slocum-tpw log-harvest`

Parse Slocum glider log files and extract GPS positions, sensor readings, and
timestamps into NetCDF.

```
slocum-tpw log-harvest [--t0 T0] [--nc OUTPUT] FILE [FILE ...]
```

| Argument | Description |
|---|---|
| `FILE` | One or more log files to parse (required) |
| `--t0 T0` | Earliest timestamp prefix to include (filters by filename) |
| `--nc OUTPUT` | Output NetCDF filename (default: `log.nc`) |

Log files must follow the naming convention `{glider}_{timestamp}_*.log`
(e.g., `osu684_20230612T153000_network.log`). The `--t0` filter compares
against the timestamp portion of the filename.

The parser extracts vehicle name, GPS coordinates (converted from DDMM.MM to
decimal degrees), and sensor readings. All observations are binned to
100-second time resolution.

**Example:**

```bash
slocum-tpw log-harvest --t0 20230601T000000 --nc log.nc osu684/logs/*.log
```

---

### `slocum-tpw mk-combined`

Merge log, flight, and science NetCDF data for a single glider into a
CF-1.13 compliant NetCDF file with derived oceanographic variables.

```
slocum-tpw mk-combined --output FILE [--glider N] [--prefix PFX]
                        [--nc-log FILE] [--nc-flight FILE] [--nc-science FILE]
```

| Argument | Description |
|---|---|
| `--output FILE` | Output NetCDF filename (required) |
| `--glider N` | Glider number; used to filter log data and auto-derive input paths |
| `--prefix PFX` | Institution prefix (default: `osu`); combined with glider number as `{prefix}{glider}` |
| `--nc-log FILE` | Input log NetCDF (default: `log.nc`) |
| `--nc-flight FILE` | Input flight NetCDF; auto-derived as `flt.{prefix}{glider}.nc` if `--glider` is set |
| `--nc-science FILE` | Input science NetCDF; auto-derived as `sci.{prefix}{glider}.nc` if `--glider` is set |

**Input requirements:**

- **Log file** must contain variables `t`, `m_water_vx`, `m_water_vy`, and
  optionally `lat`, `lon`, `glider`.
- **Flight file** must contain `m_present_time`, `m_gps_lat`, `m_gps_lon`
  (in DDMM.MM format).
- **Science file** must contain `sci_m_present_time`, `sci_water_temp`,
  `sci_water_cond` (S/m), `sci_water_pressure` (bar).

**Processing pipeline:**

1. Load log data, interpolate missing GPS, filter by glider ID
2. Load flight GPS fixes, build lat/lon interpolators (minimum 2 fixes required)
3. Load science CTD data, assign GPS via flight interpolation
4. Compute oceanographic variables using [GSW](https://github.com/TEOS-10/GSW-Python) (TEOS-10):
   - **depth** from pressure via `gsw.z_from_p`
   - **practical salinity** from conductivity via `gsw.SP_from_C`
   - **absolute salinity** via `gsw.SA_from_SP`
   - **potential temperature** via `gsw.pt0_from_t`
   - **conservative temperature** via `gsw.CT_from_pt`
   - **sigma-0** (potential density anomaly) via `gsw.sigma0`
   - **rho** (in-situ density anomaly) via `gsw.rho_t_exact`
5. Merge log and science datasets, add CF-1.13 metadata, write with zlib compression

**Example:**

```bash
# With --glider, flight and science paths are auto-derived from --nc-log location:
slocum-tpw mk-combined --glider 684 --output osu684.nc --nc-log data/log.nc

# Or specify all paths explicitly:
slocum-tpw mk-combined --output combined.nc \
    --nc-log log.nc --nc-flight flt.osu684.nc --nc-science sci.osu684.nc
```

---

### `slocum-tpw recover-by`

Estimate when a Slocum glider will need to be recovered based on battery
percentage decay. Fits a linear regression to battery charge over time and
extrapolates to a threshold.

```
slocum-tpw recover-by [options] FILE [FILE ...]
```

| Argument | Description |
|---|---|
| `FILE` | One or more NetCDF files with time and battery sensor data (required) |
| `--sensor NAME` | Sensor variable name (default: `m_lithium_battery_relative_charge`) |
| `--threshold PCT` | Battery percentage at which recovery should happen (default: `15`) |
| `--time NAME` | Name of time variable (default: `time`) |
| `--confidence LEVEL` | Confidence level for intervals, 0 < x < 1 (default: `0.95`) |
| `--ndays N` | Use only the last N days of data; use `full` for the entire dataset. Repeatable and/or comma-separated (e.g. `--ndays 3,7,full`). Cannot combine with `--start`/`--stop` |
| `--tau T` | Exponential decay time constant in days — full dataset weighted by exp(-age/T); repeatable and/or comma-separated. Cannot combine with `--start`/`--stop` |
| `--start TIME` | Use data after this UTC time (cannot combine with `--ndays`/`--tau`) |
| `--stop TIME` | Use data before this UTC time (cannot combine with `--ndays`/`--tau`) |
| `--json` | Output results as JSON instead of text |
| `--plot` | Display an interactive matplotlib plot |
| `--output FILE` | Save plot to file instead of displaying |

**Algorithm:**

The tool fits a linear model `sensor = intercept + slope * days` to the battery
data and solves for the day when the sensor reaches the threshold. Uncertainty
is propagated through partial derivatives of the recovery date with respect to
slope and intercept, using the covariance matrix from `numpy.polyfit`.
Confidence intervals use the t-distribution with n-2 degrees of freedom.

The tool handles multiple time formats (CF datetime coordinates, float epoch
seconds), removes duplicates and NaN values, and validates that at least 3
data points are available for fitting.

**Text output:**

```
Sensor:            m_lithium_battery_relative_charge
Sensor threshold:  15
Intercept (95%):   100.0000+-0.0000
Slope (95%, /day): -1.0000+-0.0000
R-squared:         1.0000
Pvalue:            0.0000
Recovery By (95%): 2025-03-27T00:00+-0.00 (days)
```

**JSON output** (`--json`):

```json
[{
  "file": "flight.nc",
  "sensor": "m_lithium_battery_relative_charge",
  "threshold": 15.0,
  "confidence": 0.95,
  "ndays": null,
  "tau": null,
  "n_points": 51,
  "intercept": 100.0,
  "intercept_ci": 0.0,
  "slope": -1.0,
  "slope_ci": 0.0,
  "r_squared": 1.0,
  "pvalue": 0.0,
  "recovery_date": "2025-03-27T00",
  "recovery_ci_days": 0.0
}]
```

**Examples:**

```bash
# Basic usage
slocum-tpw recover-by --threshold 15 flight.nc

# Use only the last 14 days and save a plot
slocum-tpw recover-by --ndays 14 --output battery.png flight.nc

# Compare multiple time windows on one plot (use 'full' for entire dataset)
slocum-tpw recover-by --ndays 3,7,full --plot flight.nc

# Exponential downweighting (recent data weighted more)
slocum-tpw recover-by --tau 5,15 --output battery.png flight.nc

# Mix ndays windows and tau weighting
slocum-tpw recover-by --ndays 7 --tau 10 --plot flight.nc

# Process multiple gliders, output JSON
slocum-tpw recover-by --json flt.osu684.nc flt.osu685.nc

# Restrict time window
slocum-tpw recover-by --start 2025-01-10 --stop 2025-02-10 flight.nc

# Custom confidence level and sensor
slocum-tpw recover-by --confidence 0.99 --sensor my_battery_pct data.nc
```

---

## Python API

All subcommand functionality is available as importable functions.

### `slocum_tpw.decode_argos`

```python
from slocum_tpw.decode_argos import proc_file, process_files

# Parse a single ARGOS file
df = proc_file("messages.txt")          # Returns DataFrame or None
print(df.columns)                       # ident, nLines, nBytes, satellite,
                                        # locationClass, time, lat, lon,
                                        # altitude, frequency

# Process multiple files and write NetCDF
process_files("output.nc", ["file1.txt", "file2.txt"])
```

#### `proc_file(fn: str) -> pd.DataFrame | None`

Parse a single ARGOS message file. Returns a DataFrame with columns `ident`,
`nLines`, `nBytes`, `satellite`, `locationClass`, `time`, `lat`, `lon`,
`altitude`, `frequency`. Returns `None` if no valid records are found.

#### `process_files(fn_nc: str, filenames: list[str]) -> None`

Process multiple ARGOS files, concatenate results, and write to NetCDF. Writes
an empty dataset if no records are found.

---

### `slocum_tpw.log_harvest`

```python
from slocum_tpw.log_harvest import parse_log_file, process_files

# Parse a single log file
df = parse_log_file("osu684_20230612T153000.log", "osu684")
print(df.columns)  # t, glider, lat, lon, sci_water_temp, m_water_vx, ...

# Process multiple files with timestamp filter
process_files(["file1.log", "file2.log"], t0="20230601T000000", nc="log.nc")
```

#### `parse_log_file(fn: str, glider: str) -> pd.DataFrame`

Parse a single Slocum log file. Reads in binary mode with UTF-8 decoding
(skips invalid bytes). Extracts vehicle name, GPS coordinates (converted from
DDMM.MM), and sensor readings. All observations are binned to 100-second time
resolution using a hash-table lookup. Returns an empty DataFrame if no data is
found.

#### `process_files(filenames: list[str], t0: str | None, nc: str) -> None`

Process multiple log files. Filenames are expected to have the format
`{glider}_{timestamp}_*.log`. Files with timestamps before `t0` are skipped.
Results are concatenated and written to NetCDF.

---

### `slocum_tpw.mk_combined`

```python
from slocum_tpw.mk_combined import mk_combo

success = mk_combo(
    gld="osu684",
    fn_output="osu684.nc",
    fn_log="log.nc",
    fn_flt="flt.osu684.nc",
    fn_sci="sci.osu684.nc",
)
```

#### `mk_combo(gld: str | None, fn_output: str, fn_log: str, fn_flt: str, fn_sci: str) -> bool`

Merge log, flight, and science data into a single CF-1.13 compliant NetCDF.
Interpolates GPS positions, computes TEOS-10 oceanographic variables (depth,
salinity, potential temperature, density), adds comprehensive metadata, and
writes with zlib compression. Pass `gld=None` to skip glider filtering.
Returns `True` on success, `False` on failure.

**Output variables:**

| Variable | Units | Description |
|---|---|---|
| `u` | m/s | Depth-averaged eastward current |
| `v` | m/s | Depth-averaged northward current |
| `t` | &deg;C | In-situ temperature |
| `s` | 1 | Practical salinity (PSS-78) |
| `depth` | m | Depth (positive down) |
| `theta` | &deg;C | Potential temperature (ref. 0 dbar) |
| `sigma` | kg/m&sup3; | Potential density anomaly (sigma-0) |
| `rho` | kg/m&sup3; | In-situ density anomaly (rho - 1000) |
| `lat`, `lon` | degrees | GPS position (science times) |
| `latu`, `lonu` | degrees | GPS position (log times) |

---

### `slocum_tpw.recover_by`

```python
from slocum_tpw.recover_by import prepare_dataset, fit_recovery, FIT_COLORS

# Load and clean a NetCDF file (handles float epoch times, custom time vars)
ds = prepare_dataset("flight.nc")
ds = prepare_dataset("glider.nc", time_var="t", sensor="m_lithium_battery_relative_charge")

# Fit battery decay and estimate recovery date
result = fit_recovery(ds, threshold=15)
print(result["recovery_date"], result["slope"])

# Restrict to last 14 days
result = fit_recovery(ds, threshold=15, ndays=14)

# Exponential downweighting (recent data weighted more heavily)
result = fit_recovery(ds, threshold=15, tau=7)

# Use result arrays for custom plotting
import matplotlib.pyplot as plt
plt.plot(result["time"], result["sensor_values"], ".")
plt.plot(result["time"], result["intercept"] + result["slope"] * result["dDays"])
```

#### `prepare_dataset(source, time_var="time", sensor="m_lithium_battery_relative_charge") -> xr.Dataset`

Load and clean a dataset for recovery fitting. *source* can be a filename,
`pathlib.Path`, or an existing `xr.Dataset`. Handles float epoch seconds
(auto-converted to datetime64), non-standard time variable names (renamed and
swapped to a `time` dimension), duplicates, NaN sensor values, and sorting.

Raises `KeyError` if required variables are missing, `OSError` if a file path
cannot be opened.

#### `fit_recovery(ds, sensor=..., threshold=15, confidence=0.95, ndays=None, tau=None, start=None, stop=None) -> dict | None`

Fit a linear model to battery data and extrapolate to the threshold. Returns
`None` if the fit fails (fewer than 3 points, near-zero slope), otherwise a
dict with:

| Key | Type | Description |
|---|---|---|
| `time` | `xr.DataArray` | Time coordinates used in the fit |
| `sensor_values` | `xr.DataArray` | Sensor values used |
| `dDays` | `np.ndarray` | Days since first data point (float64) |
| `slope`, `intercept` | `float` | Linear fit coefficients |
| `slope_ci`, `intercept_ci` | `float \| None` | Confidence interval half-widths |
| `recovery_date` | `np.datetime64` | Estimated recovery date (hourly resolution) |
| `recovery_ci_days` | `float \| None` | CI half-width on recovery date (days) |
| `r_squared`, `pvalue` | `float \| None` | Goodness-of-fit statistics |
| `n_points` | `int` | Number of data points used |
| `threshold`, `confidence` | `float` | Input parameters echoed back |
| `ndays`, `tau` | `float \| None` | Window parameters echoed back |

When `tau` is set, the fit uses weighted least squares with effective weight
`exp(-age/tau)` where *age* is days from the most recent observation. R-squared
is computed as weighted R-squared.

#### `FIT_COLORS`

List of matplotlib color names used for multi-window plots:
`["tab:green", "tab:orange", "tab:purple", "tab:red", "tab:brown", "tab:pink"]`.

---

### `slocum_tpw.slocum_utils`

```python
from slocum_tpw.slocum_utils import mk_degrees_scalar, mk_degrees
import numpy as np

# Scalar: 44 degrees 30 minutes -> 44.5 degrees
mk_degrees_scalar(4430.0)   # 44.5
mk_degrees_scalar(-12406.0) # -124.1

# Vectorized (values > 180 degrees become NaN)
arr = np.array([4430.0, -12406.0, 99900.0])
mk_degrees(arr)  # [44.5, -124.1, NaN]
```

#### `mk_degrees_scalar(degmin: float) -> float`

Convert a single DDMM.MM value to decimal degrees. Returns `NaN` if the
absolute result exceeds 180.

#### `mk_degrees(degmin: np.ndarray) -> np.ndarray`

Vectorized conversion of DDMM.MM values to decimal degrees. Values whose
absolute result exceeds 180 are set to `NaN`.

---

## Development

```bash
git clone https://github.com/mousebrains/Slocum-tpw.git
cd Slocum-tpw
pip install -e ".[dev]"
pytest                                              # run tests
pytest --cov=slocum_tpw --cov-report=term-missing   # with coverage
ruff check src/ tests/                              # lint
ruff format src/ tests/                             # format
```

Pre-commit hooks are configured for trailing whitespace, YAML/TOML validation,
and ruff linting/formatting:

```bash
pip install pre-commit
pre-commit install
```

## License

[GPL-3.0-or-later](LICENSE)
