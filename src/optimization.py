import numpy as np
import matplotlib.pyplot as plt

from antenna_array import array_factor, phase_shift
from beam_steering import gen_spacing, steer_to_phase
from utils import wavelength, linear_to_db
import pattern_measurements as pm

def optimize_pattern(x, theta, element_pattern, N_elements, steer_angles, frequency):

  d = gen_spacing(N_elements, x)
  sa = steer_to_phase(N_elements, d, steer_angles, frequency)
  psi = phase_shift(d, frequency, theta, sa);
  af = array_factor(np.ones(N_elements), N_elements, psi)
  ap = np.abs(af * element_pattern)
  
  steer_accuracy = 0
  for i, steer_angle in enumerate(steer_angles):
    print(f'{i}: {np.degrees(steer_angle)}')
    main_idx = pm.get_lobe(ap[i, :], theta)
    steer_accuracy += np.abs(steer_angle - theta[np.argmax(ap[i, main_idx])])
    
    plt.plot(theta[main_idx], ap[i, main_idx])
    plt.show()
    
  out = steer_accuracy
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
  sa = np.linspace(-np.pi/4, np.pi, 5)
  
  n = 7
  spacing_bounds = np.array([(0.1, 1)] * (n//2))
  initial_spacing = np.random.uniform(low=spacing_bounds[0][0],
                                      high=spacing_bounds[0][1],
                                      size=(n//2))
  
  print(n, spacing_bounds, initial_spacing)
  
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
  
  pass

if __name__ == "__main__":
  main()