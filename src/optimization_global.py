import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from optimization import optimize_pattern

from scipy.optimize import basinhopping
from pattern import gen_rect_pattern
from spacing import gen_spacing

def main():
  N = 360
  theta = np.linspace(-np.pi/2, np.pi/2, N)
  f = 2.4e9
  
  ep = gen_rect_pattern(theta, f)
  scan_step = 3
  
  n = 5
  
  d_initial = np.random.uniform(low=0.1, high=0.95)
  d_bounds = np.array([(0.1, 1.0)])
  w_initial = np.random.uniform(low=0.1, high=1.0, size=n)
  w_bounds = np.array([(0.1, 1.0)] * n)
  
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
    minimizer_kwargs={"method": "Powell", "bounds": x_bounds},
    niter=100,
    # method="SLSQP"
    # method="COBYLA"
    # method="Powell"
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
  ax.plot(opt_log['iteration'], opt_log['weights'], label='weights')
  ax.legend()
  ax = fig.add_subplot(3, 2, (5, 6))
  ax.plot(opt_log['iteration'], opt_log['cov_cost'], label='Coverage Cost')
  ax.plot(opt_log['iteration'], opt_log['sll_cost'], label='SLL Cost')
  ax.legend()
  
  plt.show()
    
  pass

if __name__ == "__main__":
  main()
  