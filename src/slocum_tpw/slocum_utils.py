#
# Shared utility functions for Slocum glider data processing
#
# June-2023, Pat Welch, pat@mousebrains.com

import math

import numpy as np


def mk_degrees_scalar(degmin: float) -> float:
    """Convert a single DDMM.MM value to decimal degrees.

    Returns NaN if the absolute result exceeds 180.
    """
    sgn = -1 if degmin < 0 else +1
    degmin = abs(degmin)
    deg = math.floor(degmin / 100)
    minutes = degmin % 100
    result = sgn * (deg + minutes / 60)
    if abs(result) > 180:
        return math.nan
    return result


def mk_degrees(degmin: np.ndarray) -> np.ndarray:
    """Convert an array of DDMM.MM values to decimal degrees.

    Values whose absolute result exceeds 180 are set to NaN.
    """
    sgn = np.where(degmin < 0, -1.0, 1.0)
    degmin = np.abs(degmin)
    deg = np.floor(degmin / 100)
    minutes = np.mod(degmin, 100)
    deg = sgn * (deg + minutes / 60)
    deg[np.abs(deg) > 180] = np.nan
    return deg
