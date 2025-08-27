from scipy.constants import c
import numpy as np

def plot_pattern(ax, pattern, theta):
  
  import matplotlib.pyplot as plt
  import pattern_measurements as pm
    
  from utils import linear_to_db
  
  pattern_db = linear_to_db(pattern)
  theta_deg = np.degrees(theta)
  
  ylim = (-40, 3)
  for i in range(pattern_db.shape[0]):
    ax.plot(theta_deg, pattern_db[i, :], label=f'angle: {np.degrees(pm.main_lobe_direction(pattern_db[i, :], theta)):.2f} deg')
  
  ax.set_ylim(ylim)
  ax.legend()
  ax.grid(True)
  ax.set_xlabel('Theta (degrees)')
  ax.set_ylabel('Array Pattern (dB)')
  
def fill_HPBW(ax, theta, pattern):
  
  import matplotlib.pyplot as plt
  import pattern_measurements as pm
  
  from utils import linear_to_db
  
  pattern_db = linear_to_db(pattern)
  theta_deg = np.degrees(theta)
  
  ylim = (-40, 3)
  for i in range(pattern_db.shape[0]):
    bw = pm.HPBW_bounds(pattern_db[i, :], theta)
    print(bw)
    ax.fill_between(theta_deg, ylim[0], pattern_db[i, :], where=(theta_deg >= np.degrees(bw[0])) & (theta_deg <= np.degrees(bw[1])), alpha=0.3)
    ax.vlines(np.degrees(bw[0]), -10, ylim[0], linestyle='--', alpha=0.5)
    ax.vlines(np.degrees(bw[1]), -10, ylim[0], linestyle='--', alpha=0.5)