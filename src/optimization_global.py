import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from optimization import optimize_pattern, get_log_dict

from scipy.optimize import basinhopping
from pattern import gen_rect_pattern
from spacing import gen_spacing

def main():
  N = 360
  theta = np.linspace(-np.pi/2, np.pi/2, N)
  f = 2.4e9
  
  ep = gen_rect_pattern(theta, f)
  scan_step = 1
  
  n = 8
  
  d_initial = np.random.uniform(low=0.1, high=0.95)
  d_bounds = np.array([(0.1, 1.0)])
  w_initial = np.random.uniform(low=0.1, high=1.0, size=n)
  w_bounds = np.array([(0.1, 1.0)] * n)
  
  x_initial = np.concatenate([d_initial, w_initial], axis=None)
  x_bounds = np.concatenate([d_bounds, w_bounds], axis=0)
  
  print(x_bounds, x_initial)
  
  opt_log = get_log_dict()
  
  opt_results = basinhopping(
    func=lambda x: optimize_pattern(x, 
                                   theta=theta,
                                   element_pattern=ep,
                                   N_elements=n,
                                   frequency=f,
                                   scan_step=scan_step,
                                   log = opt_log),
    x0=x_initial,
    # bounds=x_bounds,
    minimizer_kwargs={"method": "Nelder-Mead", "bounds": x_bounds},
    niter=100,
    # method="SLSQP"
    # method="COBYLA"
    # method="Powell"
    # method="Nelder-Mead"
  )
  
  print(opt_results)
    
  from utils import get_pattern
  from plotting import plot_pattern, fill_HPBW, summerize_pattern, plot_array_layout, plot_element_pattern
  from beam_steering import gen_steer_directions
  
  d = gen_spacing(n, [opt_results.x[0]]) * c / f
  
  pattern = get_pattern(theta, f, n, d, gen_steer_directions(n, step=scan_step))
  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(3, 3, (1, 5))
  plot_pattern(ax, pattern, theta)
  fill_HPBW(ax, theta, pattern)
  plot_element_pattern(ax, ep, theta)
  ax = fig.add_subplot(3, 3, 3)
  ax.plot(opt_log['iteration'], opt_log['spacing'], label='spacing (λ)')
  ax.set_ylim(0, 1)
  ax.legend()
  ax = fig.add_subplot(3, 3, 6)
  ax.plot(opt_log['iteration'], opt_log['cost']['sll'], label='SLL Cost')
  ax.plot(opt_log['iteration'], opt_log['cost']['cov'], label='Coverage Cost')
  ax.plot(opt_log['iteration'], opt_log['total_cost'], label='Total Cost')
  ax.legend()
  ax = fig.add_subplot(3, 3, 7)
  plot_array_layout(ax, n, d, center=True)
  ax = fig.add_subplot(3, 3, 8)
  from cost import coverage_cost
  ax.plot(theta, coverage_cost(theta, pattern, cost_pattern=True),
          label=f'Coverage Cost: {coverage_cost(theta, pattern):.4f}')
  ax.legend()
    
  plt.show()
  
  print("Optimized Spacing (λ):", opt_results.x[0])
  summerize_pattern(pattern, theta, print=True)
    
  pass

if __name__ == "__main__":
  main()
  