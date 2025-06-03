
import numpy as np
from scipy.constants import c
from utils import wavelength

class AntennaArray:
    def __init__(self, name: str,
                 num_elements: int,
                 spacings: np.ndarray[float],
                 weights: np.ndarray) -> None:
        self.name: str = name
        self.num_elements: int = num_elements
        self.spacings: np.ndarray[float] = spacings
        self.weights: np.ndarray[float] = weights
        
def phase_shift(array: AntennaArray,
                frequency: float,
                theta: np.ndarray[float],
                beta: np.ndarray[float]
                ) -> np.ndarray[float, float]:
    
    return 2 * np.pi * frequency / c * array.spacings[:, np.newaxis] * np.sin(theta) + beta[:, np.newaxis]

def array_factor(array: AntennaArray,
                 frequency: float,
                 theta: np.ndarray[float],
                 beta: np.ndarray[float]
                 ) -> np.ndarray[complex]:
    
    array_response = array.weights[:, np.newaxis] * np.exp(1j * phase_shift(array, frequency, theta, beta))
    return np.sum(array_response, axis=0) / array.num_elements

def symetric_spacing(num_elements: int, spacing) -> np.ndarray[float]:
    if type(spacing) is not np.ndarray:
        spacing = np.array([spacing])
    if num_elements % 2 == 0:
        if len(spacing) == 1:
            return (np.arange(-num_elements//2, num_elements//2) + .5) * spacing
        elif len(spacing) == (num_elements // 2):
            return np.concatenate((-spacing[::-1], spacing))
        else:
            raise ValueError("Spacing must be a single value or an array of length num_elements // 2.")
    else:
        if len(spacing) == 1:
            return (np.arange(-(num_elements)//2 + 1, (num_elements+1)//2)) * spacing
        elif len(spacing) == (num_elements // 2):
            return np.concatenate((-spacing[::-1], [0], spacing))
        else:
            raise ValueError("Spacing must be a single value or an array of length num_elements // 2.")
        

def beam_direction(array: AntennaArray, angle: float) -> np.ndarray[float]:
    beta = np.linspace(0, array.num_elements - 1, array.num_elements) * angle
    return beta
        
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from utils import linear_to_db

    theta = np.linspace(-np.pi, np.pi, 360)
    frequency = 2.4e9  # 1 GHz
    # pattern_antenna = np.cos(theta) ** 2 * np.cos(theta / 2) ** 4

    aa = AntennaArray('array', 4, symetric_spacing(4, 0.75) * wavelength(frequency), np.ones(4))
    af = array_factor(aa, frequency, theta, beam_direction(aa, 0))
    
    fig = plt.figure(figsize=(10, 5))
    ax = plt.plot(np.degrees(theta), linear_to_db(np.abs(af)), label='Array Factor')
    plt.ylim(-30, 0)
    plt.xlim(-90, 90)
    plt.show()

    pass