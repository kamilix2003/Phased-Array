import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import pandas as pd

from antenna_array import array_factor, phase_shift
from beam_steering import steer_to_phase, quantize_phase, gen_steer_directions
from spacing import gen_spacing
from weights import gen_weights

from cost import side_lobe_cost, coverage_cost

def optimize_pattern(x,
                     theta,
                     element_pattern,
                     N_elements,
                     scan_step,
                     frequency,
                     log:dict=None):

  x_spacing = x[0]
  x_weights = x[1]
  
  spacing = np.array([x_spacing])
  weights = np.ones(N_elements)
  
  d = gen_spacing(N_elements, spacing) * c / frequency
  sa = gen_steer_directions(N_elements, step=scan_step)
  psi = phase_shift(d, frequency, theta, sa);
  af = array_factor(weights, N_elements, psi)
  ap = np.abs(af) * element_pattern
  
  sll_cost = 0
  cov_cost = 0
  
  for i in range(ap.shape[0]):
    # sll_cost += side_lobe_cost(theta, ap[i, :], threshold=-5)
    pass
  cov_cost += coverage_cost(theta, ap)
  
  spacing_cost = 0
  # spacing_cost = np.clip((spacing - 0.5), 0, None)
  
  out = cov_cost + sll_cost + spacing_cost
  
  if log is not None:
    log['iteration'].append(len(log['iteration']) + 1)
    log['spacing'].append(spacing)
    log['weights'].append(weights[0])
    log['sll_cost'].append(sll_cost)
    log['cov_cost'].append(cov_cost)
    log['total_cost'].append(out)
  
  return out

from scipy.optimize import minimize
from pattern import gen_rect_pattern

def main():
  N = 360
  theta = np.linspace(-np.pi/2, np.pi/2, N)
  f = 2.4e9
  
  ep = gen_rect_pattern(theta, f)
  scan_step = 2
  
  n = 5
  
  d_initial = np.random.uniform(low=0.1, high=0.95)
  d_bounds = np.array([(0.1, 1.0)])
  w_initial = np.random.uniform(low=0.1, high=1.0)
  w_bounds = np.array([(0.1, 1.0)])
  scan_step_initial = 1
  scan_step_bounds = np.array([(1, 3)])
  
  x_initial = np.concatenate([d_initial, w_initial], axis=None)
  x_bounds = np.concatenate([d_bounds, w_bounds], axis=0)
  
  print(x_bounds, x_initial)
  
  opt_log = {
    "iteration": [],
    "bandwidth": [],
    "spacing": [],
    "weights": [],
    "sll_cost": [],
    "cov_cost": [],
    "total_cost": []
  }
  
  opt_results = minimize(
    fun=lambda x: optimize_pattern(x, 
                                   theta=theta,
                                   element_pattern=ep,
                                   N_elements=n,
                                   frequency=f,
                                   scan_step=scan_step,
                                   log = opt_log),
    x0=x_initial,
    bounds=x_bounds,
    # method="SLSQP"
    # method="COBYLA"
    method="Powell"
    # method="Nelder-Mead"
  )
  
  print(opt_results)
    
  from utils import get_pattern
  from plotting import plot_pattern, fill_HPBW
  from beam_steering import gen_steer_directions
  
  d = gen_spacing(n, [opt_results.x[0]]) * c / f
  
  pattern = get_pattern(theta, f, n, d, gen_steer_directions(n, step=scan_step))
  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(3, 2, (1, 2))
  plot_pattern(ax, pattern, theta)
  fill_HPBW(ax, theta, pattern)
  ax = fig.add_subplot(3, 2, 3)
  ax.plot(opt_log['iteration'], opt_log['total_cost'], label='Total Cost')
  ax.legend()
  ax = fig.add_subplot(3, 2, 4)
  ax.plot(opt_log['iteration'], opt_log['spacing'], label='spacing (λ)')
  ax.legend()
  ax = fig.add_subplot(3, 2, (5, 6))
  ax.plot(opt_log['iteration'], opt_log['cov_cost'], label='Coverage Cost')
  ax.plot(opt_log['iteration'], opt_log['sll_cost'], label='SLL Cost')
  ax.legend()
  
  plt.show()
    
  pass

if __name__ == "__main__":
  main()
  