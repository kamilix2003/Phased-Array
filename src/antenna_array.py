
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
        
def phase_shift(spacings,
                frequency: float,
                theta: np.ndarray[float],
                beta: np.ndarray[float]
                ) -> np.ndarray[float, float]:
    
    return 2 * np.pi * frequency / c * spacings[:, np.newaxis] * np.sin(theta) + beta[:, :, np.newaxis]

def array_factor(weights,
                 num_elements,
                 psi
                 ) -> np.ndarray[complex]:
    
    array_response = weights[:, np.newaxis] * np.exp(1j * psi)
    return np.sum(array_response, axis=1) / num_elements

# def symmetric_spacing(num_elements: int, spacing) -> np.ndarray[float]:
#     if type(spacing) is not np.ndarray:
#         spacing = np.array([spacing])
#     if num_elements % 2 == 0:
#         if len(spacing) == 1:
#             return (np.arange(-num_elements//2, num_elements//2) + .5) * spacing
#         elif len(spacing) == (num_elements // 2):
#             return np.concatenate((-spacing[::-1], spacing))
#         else:
#             raise ValueError("Spacing must be a single value or an array of length num_elements // 2.")
#     else:
#         if len(spacing) == 1:
#             return (np.arange(-(num_elements)//2 + 1, (num_elements+1)//2)) * spacing
#         elif len(spacing) == (num_elements // 2):
#             return np.concatenate((-spacing[::-1], [0], spacing))
#         else:
#             raise ValueError("Spacing must be a single value or an array of length num_elements // 2.")
        

def main():
    import matplotlib.pyplot as plt
    from utils import linear_to_db
    from spacing import gen_spacing
    from beam_steering import steer_to_phase    
    theta = np.linspace(-np.pi/2, np.pi/2, 360)
    frequency = 2.4e9  # 1 GHz
    # pattern_antenna = np.cos(theta) ** 2 * np.cos(theta / 2) ** 4
    num_element = 6
    
    sps = gen_spacing(num_element, np.array([0.25, 0.25, 0.4]) * (c / frequency))
    print(sps)
    sa = np.linspace(-np.pi/4, np.pi/4, 9)
    b = steer_to_phase(num_element, sps, sa, frequency)
    ps = phase_shift(sps, frequency, theta, b)
    af = array_factor(np.ones(num_element), num_element, ps)
    fig = plt.figure()
    plt.plot(theta, linear_to_db(np.abs(af[4, :])))
    plt.ylim(-40, 1)
    plt.show()
    
    pass

if __name__ == "__main__":
    main()
    