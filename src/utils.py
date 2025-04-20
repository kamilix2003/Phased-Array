import numpy as np
from scipy.constants import c

def wavelength(frequency: float) -> float:
    return c / frequency

def linear_to_db(linear: np.ndarray[float]) -> np.ndarray[float]:
    return 20 * np.log10(linear)