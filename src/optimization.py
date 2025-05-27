from antenna_array import AntennaArray, uniform_spacing, beam_direction, array_factor
from utils import wavelength

import pattern_measurements as pm
import numpy as np

def get_test_data(test_spacings):
    frequency = 2.4e9  # Frequency in Hz
    theta = np.linspace(-np.pi, np.pi, 3600)
    array = AntennaArray("Test Array", 4, uniform_spacing(4, wavelength(frequency)), np.array([1, 1, 1, 1]))
    
    test_stearing_angles = np.linspace(-np.pi/2, np.pi/2, 360)
    # test_spacings = np.linspace(0.1, 0.5, 4) * wavelength(frequency)
    test_pattern = np.abs(np.sinc(theta * 2))  # Example pattern
    
    results = np.zeros((len(test_spacings), len(test_stearing_angles), len(theta)), dtype=complex)
    
    for i, spacing in enumerate(test_spacings):
        array.spacings = uniform_spacing(array.num_elements, spacing)
        for j, angle in enumerate(test_stearing_angles):
            beta = beam_direction(array, frequency, angle)
            pattern = test_pattern * np.abs(array_factor(array, frequency, theta, beta))
            # print(f"Spacing: {spacing}, Angle: {angle}, Pattern: {pattern}")
            results[i, j, :] = pattern
    return results, theta, test_stearing_angles, test_spacings, test_pattern

def print_results(pattern, theta):
  print(" HPBW:", pm.HPBW(pattern, theta))
  print(" FNBW:", pm.FNBW(pattern, theta))
  print(" FSLBW:", pm.FSLBW(pattern, theta))
  print(" FSL Peak:", pm.FSL_peak(pattern, theta))

def optimize_pattern(spacing):
  patterns, theta, _, _, _ = get_test_data(spacing)
  pattern = patterns[0, patterns.shape[1]//2, :]  # Use the first pattern for optimization
  print("Optimizing for spacing:", spacing)
  # print_results(pattern, theta)  
  out1 = np.abs(pm.FSL_peak(pattern, theta) - 0.2)
  out2 = np.abs(pm.FNBW(pattern, theta) - np.pi/8)
  return out1 + out2 
  


if __name__ == "__main__":
  import numpy as np
  from scipy.constants import c
  from scipy import optimize
  import matplotlib.pyplot as plt
  
  import antenna_array as aa
  import pattern_measurements as pm
  
  from utils import wavelength, linear_to_db
  
  test_spacings = np.array([0.25]) * wavelength(2.4e9)
  results, theta, test_stearing_angles, test_spacings, test_pattern = get_test_data(test_spacings)

  opt_result = optimize.minimize(
      optimize_pattern,
      x0=np.array([0.1]),
      bounds=[(0.1, 0.5)],
      method='Nelder-Mead'
  )
  
  print(opt_result)
  
  array = aa.AntennaArray(
      "Optimized Array",
      4,
      aa.uniform_spacing(4, opt_result.x[0] * wavelength(2.4e9)),
      np.array([1, 1, 1, 1])
  )
  beta = aa.beam_direction(array, 2.4e9, 0)
  pattern = np.abs(aa.array_factor(array, 2.4e9, theta, beta)) * test_pattern
  
  ax = pm.plot_pattern(pattern, theta, label='Optimized Pattern')
  plt.show()
  