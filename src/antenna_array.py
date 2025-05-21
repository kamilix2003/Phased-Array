
import numpy as np
from scipy.constants import c

class AntennaArray:

    def __init__(self, name: str,
                 num_elements: int,
                 spacings: np.ndarray[float],
                 weights: np.ndarray) -> None:
        self.name: str = name
        self.num_elements: int = num_elements
        self.spacings: np.ndarray[float] = spacings
        self.weights: np.ndarray[float] = weights
        
def phase_shift(array: AntennaArray, frequency: float, theta: np.ndarray[float], beta: np.ndarray[float]) -> np.ndarray[float, float]:
    if array.num_elements != len(beta):
        raise ValueError(f"Number of elements {array.num_elements} does not match the length of beta {len(beta)}")
    if array.num_elements != len(array.spacings):
        raise ValueError(f"Number of elements {array.num_elements} does not match the length of spacings {len(array.spacings)}")
    
    return 2 * np.pi * frequency / c * array.spacings[:, np.newaxis] * np.sin(theta) + beta[:, np.newaxis]

def array_factor(array: AntennaArray, frequency: float, theta: np.ndarray[float], beta: np.ndarray[float]) -> np.ndarray[complex]:
    if array.num_elements != len(array.weights):
        raise ValueError(f"Number of elements {array.num_elements} does not match the length of weights {len(array.weights)}")
    
    array_response = array.weights[:, np.newaxis] * np.exp(1j * phase_shift(array, frequency, theta, beta))
    return np.sum(array_response, axis=0) / array.num_elements

def uniform_spacing(num_elements: int, spacing: float) -> np.ndarray[float]:
    if num_elements % 2 == 0:
        return np.linspace(-num_elements/2, num_elements/2, num_elements) * spacing
    else:
        return np.linspace(-(num_elements + spacing)/2, (num_elements + spacing)/2, num_elements) * spacing

def beam_direction(array: AntennaArray, frequency: float, angle: float) -> np.ndarray[float]:
    beta = np.linspace(0, array.num_elements - 1, array.num_elements) * angle
    return beta
       
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import wavelength
    
    # Example usage
    num_elements = 4
    frequency = 2.4e9  # 1 GHz
    spacing = 0.3 * wavelength(frequency)
    theta = np.linspace(-np.pi/2, np.pi/2, 360)
    
    spacings = uniform_spacing(num_elements, spacing)
    weights = np.ones(num_elements)  # Uniform weights
    
    antenna_array = AntennaArray("Uniform Linear Array", num_elements, spacings, weights)
    angles = np.arange(0, 2*np.pi, np.radians(22.5))
    
    plt.figure(figsize=(10, 6))
    
    for angle in angles:
        beta = beam_direction(antenna_array, frequency, angle)
        array_response = array_factor(antenna_array, frequency, theta, beta)
        
        plt.plot(theta, np.abs(array_response), label=f'Array Factor for angle {np.degrees(angle):.2f}')
    
    plt.title('Antenna Array Pattern')
    plt.legend()
    plt.show()