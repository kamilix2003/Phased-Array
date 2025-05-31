
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

def uniform_spacing(num_elements: int, spacing: float) -> np.ndarray[float]:
    if num_elements % 2 == 0:
        return np.linspace(-num_elements/2, num_elements/2, num_elements) * spacing
    else:
        return np.linspace(-(num_elements + spacing)/2, (num_elements + spacing)/2, num_elements) * spacing

def nonuniform_spacing(num_elements: int, spacings: np.ndarray[float]) -> np.ndarray[float]:
    out = np.zeros(num_elements)
    if num_elements % 2 == 0:
        out[num_elements//2:] = np.cumsum(spacings[:num_elements//2])
        out[:num_elements//2] = -np.cumsum(spacings[:num_elements//2])[::-1]
    else:
        out[num_elements//2] = 0
        out[num_elements//2+1:] = np.cumsum(spacings[:num_elements//2])
        out[:num_elements//2] = -np.cumsum(spacings[:num_elements//2])[::-1]
        
    return out

def beam_direction(array: AntennaArray, angle: float) -> np.ndarray[float]:
    beta = np.linspace(0, array.num_elements - 1, array.num_elements) * angle
    return beta
        
if __name__ == "__main__":

    pass