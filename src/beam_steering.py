import numpy as np
import matplotlib.pyplot as plt

from spacing import gen_spacing
from scipy.constants import c

def steer_to_phase(N_elements, spacings, steer_angles, frequency):

    """ 
        Rows: beta for each element
        Cols: steer angles
    """

    D = gen_spacing(N_elements, spacings)
    
    return np.sin(steer_angles[:, np.newaxis]) * 2 * np.pi * frequency / c * D

def main():
    frequency = 2.4e9

    spacings = np.array([0.1, 0.15, 0.2])
    steer_angle = np.radians(np.linspace(-45, 45, 15))
    
    phase_shift = steer_to_phase(7, spacings, steer_angle, 2.4e9)

if __name__ == "__main__":
    main()
    