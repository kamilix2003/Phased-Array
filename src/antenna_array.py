
import numpy as np
from scipy.constants import c

from antenna_element import AntennaElement
      
class AntennaArray:

    def __init__(self, name: str, antenna: AntennaElement, num_elements: int, spacing: float) -> None:
        self.name: str = name
        self.antenna: AntennaElement = antenna
        self.num_elements: int = num_elements
        self.spacing: float = spacing
        
    def psi(self, frequency: float, theta: np.ndarray[float], beta: float) -> np.ndarray[float]:
        k = 2 * np.pi * frequency / c
        return k * self.spacing * np.cos(theta) + beta
    
    def array_factor(self, frequency: float, theta: np.ndarray[float], beta: float) -> np.ndarray[complex]:
        # element_indices = np.arange(self.num_elements)[:, np.newaxis]
        # phase_shifts = self.psi(frequency, theta, beta) * element_indices
        # array_response = np.exp(1j * phase_shifts)
        # return np.sum(array_response, axis=0) / self.num_elements
        return np.sin(self.num_elements * self.psi(frequency, theta, beta) / 2) / np.sin(self.psi(frequency, theta, beta) / 2) / self.num_elements
        
    def radiation_pattern(self, frequency: float, theta: np.ndarray[float], beta: float) -> np.ndarray[complex]:
        return self.array_factor(frequency, theta, beta) * self.antenna.patterns[0].pattern
    
    
        
        