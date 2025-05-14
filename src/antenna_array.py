
import numpy as np
from scipy.constants import c

from antenna_element import AntennaElement
      
class AntennaArray:

    def __init__(self, name: str,
                 antenna: AntennaElement,
                 num_elements: int,
                 spacings: np.ndarray[float],
                 weights: np.ndarray) -> None:
        self.name: str = name
        self.antenna: AntennaElement = antenna
        self.num_elements: int = num_elements
        self.spacings: np.ndarray[float] = spacings
        self.weights: np.ndarray[float] = weights
        
    def psi(self, frequency: float, theta: np.ndarray[float], beta: np.ndarray[float]) -> np.ndarray[float, float]:
        if self.num_elements != len(beta):
            raise ValueError(f"Number of elements {self.num_elements} does not match the length of beta {len(beta)}")
        if self.num_elements != len(self.spacings):
            raise ValueError(f"Number of elements {self.num_elements} does not match the length of spacings {len(self.spacings)}")
        
        k = 2 * np.pi * frequency / c
        
        return k * self.spacings[:, np.newaxis] * np.sin(theta) + beta[:, np.newaxis]
    
    def array_factor(self, frequency: float, theta: np.ndarray[float], beta: np.ndarray[float]) -> np.ndarray[complex]:
        if self.num_elements != len(self.weights):
            raise ValueError(f"Number of elements {self.num_elements} does not match the length of weights {len(self.weights)}")
        
        phase_shifts = self.psi(frequency, theta, beta)
        array_response = self.weights[:, np.newaxis] * np.exp(1j * phase_shifts)
        return np.sum(array_response, axis=0) / self.num_elements
        
    def radiation_pattern(self, frequency: float, theta: np.ndarray[float], beta: float) -> np.ndarray[complex]:
        return self.array_factor(frequency, theta, beta) * self.antenna.patterns[0].pattern

def equal_spacing(num_elements: int, spacing: float) -> np.ndarray[float]:
    if num_elements % 2 == 0:
        return np.linspace(-num_elements/2, num_elements/2, num_elements) * spacing
    else:
        return np.linspace(-(num_elements + spacing)/2, (num_elements + spacing)/2, num_elements) * spacing

def plot_array(array: AntennaArray,
               array_factor: np.ndarray[float],
               pattern: np.ndarray[float],
               polar: bool = True):
    
    theta = array.antenna.patterns[0].theta
    
    fig = plt.figure(figsize=(10, 6))
    
    ax1 = plt.subplot(131, polar=polar)
    config_plot(ax1, polar)
    plt.plot(theta, linear_to_db(np.abs(array_factor)), label='Array Factor')
    plt.title('Array Factor')

    ax2 = plt.subplot(132, polar=polar)
    config_plot(ax2, polar)
    plt.plot(theta, linear_to_db(pattern1.pattern), label='Element Pattern')
    plt.title('Element Pattern')

    ax3 = plt.subplot(133, polar=polar)
    config_plot(ax3, polar)
    plt.plot(theta, linear_to_db(np.abs(pattern)), label='Radiation Pattern')
    plt.plot(theta, np.abs(E), label='Radiation Pattern')
    plt.title('Antenna Array Pattern')
    
    plt.show()
    
    return fig, ax1, ax2, ax3

        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.constants import c

    from antenna_element import AntennaElement
    from radiation_pattern import RadiationPattern
    from antenna_array import AntennaArray
    from utils import wavelength, linear_to_db, config_plot

    N = 360
    theta = np.linspace(-np.pi, np.pi, N)
    frequency = 2.4e9  # Frequency in Hz
    num_elements = 4

    pattern1 = RadiationPattern(
        name='Pattern1',
        frequency=frequency,
        theta=theta,
        pattern=np.cos(theta/2)**2  # Example pattern
        # pattern=np.sinc(theta - np.pi/2)**2  # Example pattern
    )

    element1 = AntennaElement(
        name='Element1',
        patterns=np.array([pattern1])
    )

    spacing = wavelength(frequency) / 4
    array = AntennaArray(
        name='Array1',
        antenna=element1,
        num_elements=num_elements,
        spacings=equal_spacing(num_elements, spacing),
        weights=np.ones(num_elements)  # Uniform weights
    )
    
    print(array.spacings)
    
    beta = np.zeros(num_elements)  # Phase shifts for each element

    af = array.array_factor(frequency, theta, beta)

    E = array.radiation_pattern(frequency, theta, beta)

    fig, ax1, ax2, ax3 = plot_array(array, af, E, polar=False)
    
