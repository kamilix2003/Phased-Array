import numpy as np
from scipy.constants import c

def wavelength(frequency: float) -> float:
    return c / frequency

def linear_to_db(linear: np.ndarray[float]) -> np.ndarray[float]:
    return 10 * np.log10(linear)

def config_plot(ax) -> None:
    ax.set_ylim(-50, 0)
    ax.grid(True)