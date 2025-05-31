from antenna_array import AntennaArray, uniform_spacing, nonuniform_spacing, beam_direction, array_factor
from utils import wavelength, linear_to_db

from scipy.signal import correlate

import pattern_measurements as pm
import numpy as np

def outside_bounds(theta, pattern, bounds_upper, bounds_lower):
  pattern_out = (pattern > bounds_upper) | (pattern < bounds_lower)
  print(f"Number of points outside bounds: {np.sum(pattern_out)}")
  return theta[pattern_out], pattern[pattern_out]

def optimize_pattern(spacing, *args):
  print(f"Optimizing with spacing: {spacing}")
  text = args[0]["text"]
  frequency = args[0]["frequency"]
  pattern_antenna = args[0]["pattern_antenna"]
  pattern_goal_upper = args[0]["pattern_goal_upper"]
  pattern_goal_lower = args[0]["pattern_goal_lower"]
  beam_sweap = args[0]["beam_sweap"]
  theta = args[0]["theta"]
  num_elements = args[0]["num_elements"]

  weights = np.ones(num_elements)  
  aa = AntennaArray(f'array', num_elements, nonuniform_spacing(num_elements, spacing), weights)

  beta = beam_direction(aa, beam_sweap[len(beam_sweap) // 2])
  af = array_factor(aa, 
                    frequency,
                    theta,
                    beta)

  pattern_array = np.abs(af) * pattern_antenna
  
  a = np.sum(np.abs(pattern_array[~(pattern_array > pattern_goal_lower)] \
    - pattern_goal_lower[~(pattern_array > pattern_goal_lower)]))
  b = np.sum(np.abs(pattern_array[~(pattern_array < pattern_goal_upper)] \
    - pattern_goal_upper[~(pattern_array < pattern_goal_upper)]))
  out = a + b

  return out


if __name__ == "__main__":
  import numpy as np
  from scipy.constants import c
  from scipy import optimize
  import matplotlib.pyplot as plt
  
  import antenna_array as aa
  import pattern_measurements as pm
  
  from utils import wavelength, linear_to_db, db_to_linear
  
  N = 360
  theta = np.linspace(-np.pi, np.pi, N)
  pattern_antenna = np.sinc(theta * 2) ** 2  # Example pattern, can be modified

  N_beam_width_lower = 10
  N_beam_width_upper = 15
  pattern_goal_upper = db_to_linear(-20 * np.ones_like(theta))
  pattern_goal_upper[N//2 - N_beam_width_upper:N//2 + N_beam_width_upper] = db_to_linear(1e-9 * np.ones(2 * N_beam_width_upper))
  pattern_goal_lower = 0 * np.ones_like(theta)
  pattern_goal_lower[N//2 - N_beam_width_lower:N//2 + N_beam_width_lower] = db_to_linear(-3 * np.ones(2 * N_beam_width_lower))
  
  sweap_angles = np.arange(-7, 8) * np.radians(22.5)

  num_elements = 5

  args = {
    "text": "test",
    "frequency": 2.4e9,  # Example frequency
    "pattern_antenna": pattern_antenna,
    "pattern_goal_upper": pattern_goal_upper,
    "pattern_goal_lower": pattern_goal_lower,
    "theta": theta,
    "beam_sweap": sweap_angles,
    "num_elements": num_elements
  }

  opt_result = optimize.minimize(
      optimize_pattern,
      x0=np.array([0.1, 0.1]),
      bounds=[(0.1, 0.5), (0.1, 0.5)],
      method='Powell',
      args=args
  )
  
  print(opt_result)
  print(f'spacing: {nonuniform_spacing(num_elements, opt_result.x)}')
  aa = AntennaArray(f'Optimized array', num_elements, nonuniform_spacing(num_elements, opt_result.x), np.ones(num_elements))

  beta = beam_direction(aa, 0)
  af = array_factor(aa, 
                    2.4e9,
                    theta,
                    beta)
  pattern_array = np.abs(af) * pattern_antenna
  
  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(2, 2, (1, 2))
  ax.plot(theta, linear_to_db(pattern_goal_upper), label='Goal Upper', linestyle='--', color='black')
  ax.plot(theta, linear_to_db(pattern_goal_lower), label='Goal Lower', linestyle='--', color='black')
  ax.plot(theta, linear_to_db(pattern_array), label='Array Pattern')
  theta_out, pattern_out = outside_bounds(theta, pattern_array, pattern_goal_upper, pattern_goal_lower)
  print(theta_out.shape, pattern_out.shape)
  ax.plot(theta_out, linear_to_db(pattern_out), 'r.', label='Outside Bounds')
  ax.set_ylim(-40, 3)
  ax.legend()
  
  ax = fig.add_subplot(2, 2, 3)
  ax.plot(theta, linear_to_db(pattern_antenna), label='Antenna Pattern')
  ax.set_ylim(-40, 3)
  ax.legend()
  
  ax = fig.add_subplot(2, 2, 4)
  ax.plot(theta, linear_to_db(np.abs(af)), label='Array Factor')
  ax.set_ylim(-40, 3)
  ax.legend()
  
  plt.show()
