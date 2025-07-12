import numpy as np
import matplotlib.pyplot as plt

from antenna_array import array_factor, phase_shift
from beam_steering import gen_spacing, steer_to_phase
from utils import linear_to_db, plot_array
import pattern_measurements as pm
from scipy.constants import c

def optimize_pattern(x, theta, element_pattern, N_elements, steer_angles, frequency):

  d = gen_spacing(N_elements, x) * c / frequency
  sa = steer_to_phase(N_elements, d, steer_angles, frequency)
  psi = phase_shift(d, frequency, theta, sa);
  af = array_factor(np.ones(N_elements), N_elements, psi)
  ap = np.abs(af * element_pattern)
  
  steer_accuracy = 0
  for i, steer_angle in enumerate(steer_angles):
    # print(f'{i}: {np.degrees(steer_angle)}')
    main_idx = pm.get_lobe(ap[i, :], theta)
    steer_error = np.abs(steer_angle - np.average(theta[main_idx]))
    steer_accuracy += steer_error
    
  out = steer_accuracy
  print(out)
  return out

from scipy.optimize import minimize
import random
from pattern import gen_rect_patter

def main():
  plt.figure()
  N = 180
  theta = np.linspace(-np.pi/2, np.pi/2, N)
  f = 2.4e9
  
  ep = gen_rect_patter(theta, f)
  sa = np.linspace(-np.pi/4, np.pi/4, 5)
  
  print(f'scan angles: {sa}')
  
  n = 7
  spacing_bounds = np.array([(0.1, 1)] * (n//2))
  initial_spacing = np.random.uniform(low=spacing_bounds[0][0],
                                      high=spacing_bounds[0][1],
                                      size=(n//2))
  
  print(f'Number of elements: {n}')
  print(f'spacing bounds: {spacing_bounds}')
  print(f'initial spacing: {initial_spacing}')
  
  opt_results = minimize(
    fun=lambda x: optimize_pattern(x, 
                                   theta=theta,
                                   element_pattern=ep,
                                   N_elements=n,
                                   frequency=f,
                                   steer_angles=sa),
    x0=initial_spacing,
    bounds=spacing_bounds,
    method="Powell"
  )
  
  print(opt_results)
  
  plot_array(gen_spacing(n, opt_results.x))
  
  pass

if __name__ == "__main__":
  main()