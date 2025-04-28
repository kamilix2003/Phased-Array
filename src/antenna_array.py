
import numpy as np
from scipy.constants import c

from antenna_element import AntennaElement
      
class AntennaArray:

    def __init__(self, name: str, antenna: AntennaElement, num_elements: int, spacings: np.ndarray[float]) -> None:
        self.name: str = name
        self.antenna: AntennaElement = antenna
        self.num_elements: int = num_elements
        self.spacings: np.ndarray[float] = spacings
        
    def psi(self, frequency: float, theta: np.ndarray[float], beta: float) -> np.ndarray[float, float]:
        k = 2 * np.pi * frequency / c
        return k * self.spacings[:, np.newaxis] * np.cos(theta) + beta * np.arange(self.num_elements)[:, np.newaxis]
    
    def array_factor(self, frequency: float, theta: np.ndarray[float], beta: float) -> np.ndarray[complex]:
        phase_shifts = self.psi(frequency, theta, beta)
        array_response = np.exp(1j * phase_shifts)
        return np.sum(array_response, axis=0) / self.num_elements
        
    def radiation_pattern(self, frequency: float, theta: np.ndarray[float], beta: float) -> np.ndarray[complex]:
        return self.array_factor(frequency, theta, beta) * self.antenna.patterns[0].pattern
    
    
        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.constants import c

    from antenna_element import AntennaElement
    from radiation_pattern import RadiationPattern
    from antenna_array import AntennaArray
    from utils import wavelength, linear_to_db, config_plot

    N = 1000
    theta = np.linspace(-np.pi, np.pi, N)
    frequency = 2.4e9  # Frequency in Hz

    pattern1 = RadiationPattern(
        name='Pattern1',
        frequency=frequency,
        theta=theta,
        # pattern=np.cos(theta/2 + np.pi/4)**2  # Example pattern
        pattern=np.sinc(theta - np.pi/2)**2  # Example pattern
    )

    element1 = AntennaElement(
        name='Element1',
        patterns=np.array([pattern1])
    )

    array = AntennaArray(
        name='Array1',
        antenna=element1,
        num_elements=4,
        spacings=wavelength(frequency) * np.array([-.6, -.2, .2, .6])
    )

    beta = 0

    af = array.array_factor(frequency, theta, beta)

    E = array.radiation_pattern(frequency, theta, beta)

    fig = plt.figure(figsize=(10, 6))
    polar = True
    ax = plt.subplot(131, polar=polar)
    config_plot(ax, polar)
    plt.plot(theta, linear_to_db(np.abs(af)), label='Array Factor')
    plt.title('Array Factor')

    ax = plt.subplot(132, polar=polar)
    config_plot(ax, polar)
    plt.plot(pattern1.theta, linear_to_db(pattern1.pattern), label='Element Pattern')
    plt.title('Element Pattern')

    ax = plt.subplot(133, polar=polar)
    config_plot(ax, polar)
    plt.plot(theta, linear_to_db(np.abs(E)), label='Radiation Pattern')
    plt.plot(theta, np.abs(E), label='Radiation Pattern')
    plt.title('Antenna Array Pattern')
    
    plt.show()