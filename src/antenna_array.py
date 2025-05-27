
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
 
def get_test_data():
    frequency = 2.4e9  # Frequency in Hz
    theta = np.linspace(-np.pi, np.pi, 3600)
    array = AntennaArray("Test Array", 4, uniform_spacing(4, wavelength(frequency)), np.array([1, 1, 1, 1]))
    
    test_stearing_angles = np.linspace(-np.pi/2, np.pi/2, 360)
    test_spacings = np.linspace(0.1, 0.5, 4) * wavelength(frequency)
    test_pattern = np.abs(np.sinc(theta * 2))  # Example pattern
    
    results = np.zeros((len(test_spacings), len(test_stearing_angles), len(theta)), dtype=complex)
    
    for i, spacing in enumerate(test_spacings):
        array.spacings = uniform_spacing(array.num_elements, spacing)
        for j, angle in enumerate(test_stearing_angles):
            beta = beam_direction(array, frequency, angle)
            pattern = test_pattern * array_factor(array, frequency, theta, beta)
            # print(f"Spacing: {spacing}, Angle: {angle}, Pattern: {pattern}")
            results[i, j, :] = pattern
    return results, theta, test_stearing_angles, test_spacings, test_pattern
       
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import numpy as np
    import antenna_array as aa
    from utils import wavelength, linear_to_db
    
    results, theta, test_stearing_angles, test_spacings, test_pattern = get_test_data()
            
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(projection='polar')
    ax.set_ylim(-30, 0)
    ax.plot(theta, linear_to_db(np.abs(results[0, 140, :])))
    plt.show()
    
    pass