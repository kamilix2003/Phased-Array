import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c


import antenna_array as aa
from pattern_measurements import FNBW, HPBW, FSLBW
from utils import wavelength, linear_to_db, config_plot

N = 360
theta = np.linspace(-np.pi, np.pi, N)
frequency = 2.4e9  # Frequency in Hz

pattern = np.abs(np.sinc(theta * 2))  # Example pattern

lambda_spacing =  np.linspace(0.1, 1.1, 10)
# angles = np.arange(-np.pi/4, np.pi/4, np.radians(22.5))  # Angles in radians
# angles = np.array([ -np.pi/4, 0, np.pi/4])
angles = np.linspace(-np.pi, np.pi, 36)  # Angles in radians

def test_array(num_elements: int, spacing: float, angles: np.ndarray[float]) -> None:
    spacings = aa.uniform_spacing(num_elements, spacing)
    weights = np.ones(num_elements)  # Uniform weights
    
    antenna_array = aa.AntennaArray("Uniform Linear Array", num_elements, spacings, weights)
    
    plt.figure(figsize=(10, 6))
    
    FNBWs = np.zeros_like(angles)
    HPBWs = np.zeros_like(angles)
    FSLBWs = np.zeros_like(angles)
    
    fig = plt.figure(1, figsize=(10, 6))
    ax = plt.axes(projection='polar')
    for i, angle in enumerate(angles):
        beta = aa.beam_direction(antenna_array, frequency, angle)
        array_response = pattern * aa.array_factor(antenna_array, frequency, theta, beta)
        ax.plot(theta, linear_to_db(np.abs(array_response)), label=f'Angle: {np.degrees(angle):.1f}°')
        try:
            FNBWs[i] = FNBW((np.abs(array_response)), theta)[0]
        except ValueError as e:
            print(f"Error calculating FNBW for angle {angle}: {e}")
            FNBWs[i] = np.nan
        try:
            HPBWs[i] = HPBW((np.abs(array_response)), theta)[0]
        except ValueError as e:
            print(f"Error calculating HPBW for angle {angle}: {e}")
            HPBWs[i] = np.nan
        try:
            FSLBWs[i] = FSLBW((np.abs(array_response)), theta)[0]
        except ValueError as e:
            print(f"Error calculating FSLBW for angle {angle}: {e}")
            FSLBWs[i] = np.nan
    config_plot(ax, polar=True)
    
    plt.figure(2, figsize=(10, 6))
    plt.plot(angles, np.degrees(FNBWs), label='FNBW')
    plt.plot(angles, np.degrees(HPBWs), label='HPBW')
    plt.plot(angles, np.degrees(FSLBWs), label='FSLBW')
    plt.xlabel('Angle (radians)')
    plt.title(f'Antenna Array Beamwidths for {num_elements} Elements')  
    plt.legend()    
    plt.show()
    
test_array(4, 0.3 * wavelength(frequency), angles)