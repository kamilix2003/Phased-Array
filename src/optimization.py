from antenna_array import AntennaArray, uniform_spacing, beam_direction, array_factor
from utils import wavelength

from scipy.signal import correlate

import pattern_measurements as pm
import numpy as np

def get_test_data(test_spacings):
    frequency = 2.4e9  # Frequency in Hz
    theta = np.linspace(-np.pi, np.pi, 3600)
    array = AntennaArray("Test Array", 4, uniform_spacing(4, wavelength(frequency)), np.array([1, 1, 1, 1]))
    
    test_stearing_angles = np.linspace(-np.pi/2, np.pi/2, 360)
    # test_spacings = np.linspace(0.1, 0.5, 4) * wavelength(frequency)
    test_pattern = np.abs(np.sinc(theta ))  # Example pattern
    
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

def optimize_pattern(spacing, *args):
  
  patterns, theta, stearing_angle, _, _ = get_test_data(spacing)
  angle = patterns.shape[1]//2
  theta_angle = np.where(np.isclose(theta, stearing_angle[angle], atol=1e-3))[0][0]
  pattern = patterns[0, angle, :]  # Use the first pattern for optimization
  pattern_goal = args[0]['pattern_goal']
  out = np.abs(correlate(pattern, pattern_goal, mode='full')).max()  # Max correlation with itself
  print(f'Optimizing spacing: {spacing}, Correlation: {out}')
  return out
  


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

  pattern_goal = np.ones_like(test_pattern) * 1e-5  # Example desired pattern, can be modified
  pattern_goal[(theta < np.pi/4) & (theta > -np.pi/4)] = 1  # Set a desired pattern for the first quarter of the angle range

  args = {
    "text": "test",
    "pattern_goal": pattern_goal,
  }

  opt_result = optimize.minimize(
      optimize_pattern,
      x0=np.array([0.25]),
      bounds=[(0.01, 0.99)],
      method='powell',
      args=args
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
  ax.plot(theta, linear_to_db(pattern_goal), 'r--', label='Desired Pattern')
  plt.show()
  