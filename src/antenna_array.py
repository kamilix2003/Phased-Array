
import numpy as np
from scipy.constants import c
from numpy.fft import fftshift, ifft

class AntennaArray:
    def __init__(self, name: str,
                 num_elements: int,
                 spacings: np.ndarray[float],
                 weights: np.ndarray) -> None:
        self.name: str = name
        self.num_elements: int = num_elements
        self.spacings: np.ndarray[float] = spacings
        self.weights: np.ndarray[float] = weights
        
def phase_shift(spacings,
                frequency: float,
                theta: np.ndarray[float],
                beta: np.ndarray[float]
                ) -> np.ndarray[float, float]:
    if beta.dtype == np.complexfloating:
        beta = np.angle(beta)
    return 2 * np.pi * frequency / c * spacings[:, np.newaxis] * np.cos(theta - np.pi/2) + beta[:, :, np.newaxis]

def array_factor(weights,
                 num_elements,
                 psi
                 ):
    if weights.dtype == np.complexfloating:
        weights = np.abs(weights)
    array_response = weights[:, np.newaxis] * np.exp(1j * psi)
    return np.sum(array_response, axis=1) / num_elements

def array_pattern(array_space: np.ndarray[complex], 
                  element_pattern: np.ndarray[float] = None):
    
    array_factor = ifft(array_space, axis=0)
    array_factor = fftshift(array_factor, axes=0)
    array_factor /= np.max(np.abs(array_factor))
    # array_factor = ifftshift(array_space, axes=0)
    
    if element_pattern is not None:
        array_factor = np.abs(array_factor * element_pattern[np.newaxis, :])
    else:
        array_factor = np.abs(array_factor)[:, np.newaxis]
    
    return array_factor 

def main():
    import matplotlib.pyplot as plt
    from utils import linear_to_db
    from spacing import gen_spacing
    from beam_steering import steer_to_phase    
    theta = np.linspace(-np.pi/2, np.pi/2, 360)
    frequency = 2.4e9  # 1 GHz
    # pattern_antenna = np.cos(theta) ** 2 * np.cos(theta / 2) ** 4
    num_element = 7
    
    # sps = gen_spacing(num_element, np.array([0.4, 0.4, 0.4]) * (c / frequency))
    # print(sps)
    # sa = np.linspace(-np.pi/4, np.pi/4, 9)
    # b = steer_to_phase(num_element, sps, sa, frequency)
    # ps = phase_shift(sps, frequency, theta, b)
    # af = array_factor(np.ones(num_element), num_element, ps)
    # fig = plt.figure()
    # plt.plot(theta, linear_to_db(np.abs(af[4, :])))
    # plt.ylim(-40, 1)
    # plt.show()
    
    N = 361
    d = .125
    u = np.linspace(-1, 1, N)
    sin_theta = u / d
    theta = np.degrees(np.arcsin(np.clip(sin_theta, -1, 1)))
    array_space_quant = .1
    array_space = np.zeros(N, dtype=complex)
    print(array_space_quant)
    array_elements = 8 * np.array([0, 1, 2, 3])
    shift = np.ones_like(array_elements) * np.pi/4
    array_space[array_elements] = np.exp(1j * 2 * np.pi * d * array_space_quant * shift)
    ap = array_pattern(array_space)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(theta, np.abs(ap))
    ax.set_ylim(-.1, 1.1)
    ax.legend()
    ax.grid(True)
    ax2 = fig.add_subplot(2, 1, 2)
    ax3 = ax2.twinx()
    ax2.stem(sin_theta, np.abs(array_space))
    ax2.set_title('Array Space')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlabel('Element Index')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
    
    pass

if __name__ == "__main__":
    main()
    