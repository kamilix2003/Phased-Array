from antenna_array import AntennaArray, beam_direction, array_factor, symmetric_spacing
from utils import wavelength, linear_to_db


import pattern_measurements as pm
import numpy as np

def outside_bounds(theta, pattern, bounds_upper, bounds_lower):
  pattern_out = (pattern > bounds_upper) | (pattern < bounds_lower)
  print(f"Number of points outside bounds: {np.sum(pattern_out)}")
  return theta[pattern_out], pattern[pattern_out]

def optimize_pattern(x, *args):
  # print(f"Optimizing with spacing: {spacing}")
  spacing = np.array([x[0]])
  weights = np.array(x[1:])
  text = args[0]["text"]
  frequency = args[0]["frequency"]
  pattern_antenna = args[0]["pattern_antenna"]
  pattern_goal_upper = args[0]["pattern_goal_upper"]
  pattern_goal_lower = args[0]["pattern_goal_lower"]
  beam_sweep = args[0]["beam_sweep"]
  theta = args[0]["theta"]
  num_elements = args[0]["num_elements"]

  # weights = np.ones(num_elements)  
  aa = AntennaArray(f'array', 
                    num_elements,
                    symmetric_spacing(num_elements, spacing) * wavelength(frequency),
                    weights)

  beta = beam_direction(aa, beam_sweep[len(beam_sweep) // 2])
  af = array_factor(aa, 
                    frequency,
                    theta,
                    beta)

  pattern_array = np.abs(af) * pattern_antenna
  
  a = 5*np.sum(np.abs(pattern_array[~(pattern_array > pattern_goal_lower)] \
    - pattern_goal_lower[~(pattern_array > pattern_goal_lower)]))
  b = np.sum(np.abs(pattern_array[~(pattern_array < pattern_goal_upper)] \
    - pattern_goal_upper[~(pattern_array < pattern_goal_upper)]))
    
  out = a + b

  if args[0]["logger"] is not None:
    args[0]["logger"][0].append(spacing[0])
    args[0]["logger"][1].append(out)

  return out

if __name__ == "__main__":
  import numpy as np
  from scipy.constants import c
  from scipy import optimize
  import matplotlib.pyplot as plt
  
  import antenna_array as aa
  import pattern_measurements as pm
  
  from utils import wavelength, linear_to_db, db_to_linear
  
  from import_pattern import extract_pattern_data, prepare_data

  power_matrix, angles, frequencies = extract_pattern_data()
  power_pattern = prepare_data(power_matrix, angles, frequencies, 360)[:, 0]

  N = 360
  theta = np.linspace(-np.pi/2, np.pi/2, N)
  # pattern_antenna = np.sinc(theta * 2) ** 2
  # pattern_antenna = np.cos(theta) ** 2  * np.cos(theta / 2) ** 4
  pattern_antenna = db_to_linear(power_pattern - np.max(power_pattern))

  N_beam_width_lower = (np.abs(theta - np.radians(7.5)).argmin() - N//2) // 2
  N_beam_width_upper = (np.abs(theta - np.radians(25)).argmin() - N//2) // 2

  pattern_goal_upper = db_to_linear(-20 * np.ones_like(theta))
  pattern_goal_upper[N//2 - N_beam_width_upper:N//2 + N_beam_width_upper] = db_to_linear(1e-9 * np.ones(2 * N_beam_width_upper))
  pattern_goal_lower = 0 * np.ones_like(theta)
  pattern_goal_lower[N//2 - N_beam_width_lower:N//2 + N_beam_width_lower] = db_to_linear(-1 * np.ones(2 * N_beam_width_lower))
  
  sweep_angles = np.arange(-7, 8) * np.radians(22.5)

  num_elements = 7

  logger = [[], []]

  args = {
    "text": "test",
    "frequency": 2.4e9,  # Example frequency
    "pattern_antenna": pattern_antenna,
    "pattern_goal_upper": pattern_goal_upper,
    "pattern_goal_lower": pattern_goal_lower,
    "theta": theta,
    "beam_sweep": sweep_angles,
    "num_elements": num_elements,
    "logger" : logger
  }

  spacing_constraints = np.array([(0.1, 0.8)])
  initial_spacing = np.array([0.2])

  weight_constraints = np.array([(0.5, 1)] * num_elements)
  initial_weights = np.ones(num_elements)
  
  constraints = np.concatenate((spacing_constraints, weight_constraints), axis=0)
  initial = np.concatenate((initial_spacing, initial_weights), axis=0)


  opt_result = optimize.minimize(
      optimize_pattern,
      x0=initial,
      bounds=constraints,
      method='Powell',
      args=args
  )
  
  print(opt_result)
  
  opt_spacing = opt_result.x[0]
  opt_weights = opt_result.x[1:]
  print(f'opt_spacing: {opt_spacing}, opt_weights: {opt_weights}')
  print(f'spacing: {symmetric_spacing(num_elements, opt_spacing)}')
  aa = AntennaArray(f'Optimized array',
                    num_elements,
                    symmetric_spacing(num_elements, opt_spacing) * wavelength(2.4e9),
                    opt_weights)

  beta = beam_direction(aa, 0)
  af = array_factor(aa, 
                    2.4e9,
                    theta,
                    beta)
  pattern_array = np.abs(af) * pattern_antenna
  
  theta = np.degrees(theta)
  
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
  ax.grid()
  
  ax = fig.add_subplot(2, 2, 3)
  ax.plot(theta, linear_to_db(pattern_antenna), label='Antenna Pattern')
  ax.plot(theta, linear_to_db(np.abs(af)), label='Array Factor')
  ax.set_ylim(-40, 3)
  ax.legend()
  ax.grid()
  
  ax = fig.add_subplot(2, 2, 4)
  
  
  n = np.arange(len(logger[0]))
  ax.plot(n, logger[0], "g", label='Spacing')
  ax.set_xlabel('Iteration')
  ax.set_ylabel('Spacing (m)')
  ax.legend()
  ax.grid()
  
  ax = ax.twinx()

  ax.plot(n, logger[1], "r", label='Optimization Progress')
  ax.set_ylabel('Optimization Progress')
  ax.legend()
  ax.grid()
  
  plt.show()
