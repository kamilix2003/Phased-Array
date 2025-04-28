import numpy as np
from scipy.constants import c

def wavelength(frequency: float) -> float:
    return c / frequency

def linear_to_db(linear: np.ndarray[float]) -> np.ndarray[float]:
    return 10 * np.log10(linear)

def config_plot(ax, polar: bool) -> None:
    if polar:
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(-1)
        ax.set_thetalim(-np.pi, np.pi)
    else:
        pass
    ax.set_ylim(-30, 0)
    ax.grid(True)