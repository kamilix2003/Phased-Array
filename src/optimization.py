import numpy as np
import matplotlib.pyplot as plt

from antenna_array import array_factor, phase_shift
from beam_steering import gen_spacing, steer_to_phase
from utils import linear_to_db, plot_array
import pattern_measurements as pm
from scipy.constants import c

class optimizer_manager:
  
  def __init__(self, N_elements):
    self.iteration = 0
    self.N_elements = N_elements
    self.out: list = []
  
  def increment(self, out):
    self.iteration += 1
    self.out.append(out)

def optimize_pattern(x,
                     pattern_optimizer: optimizer_manager,
                     theta,
                     element_pattern,
                     N_elements,
                     steer_angles,
                     frequency):

  print(pattern_optimizer.iteration)

  d = gen_spacing(N_elements, x) * c / frequency
  sa = steer_to_phase(N_elements, d, steer_angles, frequency)
  psi = phase_shift(d, frequency, theta, sa);
  af = array_factor(np.ones(N_elements), N_elements, psi)
  ap = np.abs(af * element_pattern)
  
  peak_power = 0
  # for i, steer_angle in enumerate(steer_angles):
  #   main_idx = pm.get_lobe(ap[i, :], theta)
  #   center_main_idx = main_idx[np.argmax(theta[main_idx])]
  #   # peak_power += 1 - ap[i, center_main_idx]  #????????????
  
  steer_accuracy = np.zeros(steer_angles.size)
  for i, steer_angle in enumerate(steer_angles):
    main_idx = pm.get_lobe(ap[i, :], theta)
    avg_main_angle = np.average(theta[main_idx])
    center_main_angle = theta[main_idx[np.argmax(theta[main_idx])]]
    steer_error = np.abs(steer_angle - center_main_angle)
    steer_accuracy[i] = steer_error
    
  beam_goal = np.radians(15)
  beam_error = np.zeros(steer_angles.size)
  for i, steer_angle in enumerate(steer_angles):
    main_idx = pm.get_lobe(ap[i, :], theta)
    print(np.abs((theta[main_idx[-1]] - theta[main_idx[0]])))
    beam_error[i] = np.abs((theta[main_idx[-1]] - theta[main_idx[0]]) - beam_goal)  
        
  out = 0
  out += np.max(steer_accuracy)
  out += np.sum(beam_error)
  out += peak_power
  
  pattern_optimizer.increment(out)
  
  print(d, out)
  return out

from scipy.optimize import minimize
import random
from pattern import gen_rect_patter

def main():
  N = 180
  theta = np.linspace(-np.pi/2, np.pi/2, N)
  f = 2.4e9
  
  ep = gen_rect_patter(theta, f)
  sa = np.linspace(-np.pi/6, np.pi/6, 3)
  
  print(f'scan angles: {sa}')
  
  n = 7
  spacing_bounds = np.array([(0.1, .95)] * (n//2))
  initial_spacing = np.random.uniform(low=spacing_bounds[0][0],
                                      high=spacing_bounds[0][1],
                                      size=(n//2))
  
  print(f'Number of elements: {n}')
  print(f'spacing bounds: {spacing_bounds}')
  print(f'initial spacing: {initial_spacing}')
  
  opt_manager = optimizer_manager(N_elements=n)
  
  opt_results = minimize(
    fun=lambda x: optimize_pattern(x, 
                                   pattern_optimizer=opt_manager,
                                   theta=theta,
                                   element_pattern=ep,
                                   N_elements=n,
                                   frequency=f,
                                   steer_angles=sa),
    x0=initial_spacing,
    bounds=spacing_bounds,
    method="Powell"
    # method="Nelder-Mead"
  )
  
  print(opt_results)
  
  fig = plot_array(gen_spacing(n, opt_results.x), steer_angles=sa)
  ax = fig.add_subplot(2, 1, 2)
  ax.plot(opt_manager.out)
      
  plt.show()
  pass

if __name__ == "__main__":
  main()