from scipy.constants import c
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

def summerize_pattern(pattern, theta, print=False):
  import pattern_measurements as pm
  
  main_lobe_dirs = []
  HPBWs = []
  SLLs = []
  
  for i in range(pattern.shape[0]):
    main_lobe_dirs.append(np.degrees(pm.main_lobe_direction(pattern[i, :], theta)))
    HPBWs.append(np.degrees(pm.HPBW(pattern[i, :], theta)))
    SLLs.append(pm.SLL(pattern[i, :], theta))
  
  steer_limits = (np.min(main_lobe_dirs), np.max(main_lobe_dirs))
  
  summery = {
    "main_lobe_dirs": main_lobe_dirs,
    "HPBWs": HPBWs,
    "SLLs": SLLs,
    "steer_limits": steer_limits
  }
  
  if print:
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(summery)
  
  return summery 

def plot_pattern(ax: plt.axis, pattern, theta):
  
  import pattern_measurements as pm
    
  from utils import linear_to_db
  
  pattern_db = linear_to_db(pattern)
  theta_deg = np.degrees(theta)
  
  ylim = (-40, 3)
  
  if ax.name == 'polar':
      ax.set_theta_zero_location("N")
      ax.set_theta_direction(-1)
      ax.set_thetalim(np.radians([-90, 90]))
      ax.set_rlim(ylim)
  
  for i in range(pattern_db.shape[0]):
    
    if ax.name == 'polar':
      ax.plot(theta, pattern_db[i, :], label=f'angle: {np.degrees(pm.main_lobe_direction(pattern_db[i, :], theta)):.2f} deg')
    else:
      ax.plot(theta_deg, pattern_db[i, :], label=f'angle: {np.degrees(pm.main_lobe_direction(pattern_db[i, :], theta)):.2f} deg')
  
  ax.set_ylim(ylim)
  ax.legend()
  ax.grid(True)
  ax.set_xlabel('Theta (degrees)')
  ax.set_ylabel('Array Pattern (dB)')
  
def plot_element_pattern(ax: Axes, element_pattern, theta):
  
  from utils import linear_to_db
  
  pattern_db = linear_to_db(element_pattern)
  theta_deg = np.degrees(theta)
  
  ylim = (-40, 3)
  
  if ax.name == 'polar':
      ax.set_theta_zero_location("N")
      ax.set_theta_direction(-1)
      ax.set_thetalim(np.radians([-90, 90]))
      ax.set_rlim(ylim)
  
  if ax.name == 'polar':
    ax.plot(theta, pattern_db, label='Element Pattern', alpha=0.5, linestyle='--', color='black')
  else:
    ax.plot(theta_deg, pattern_db, label='Element Pattern', alpha=0.5, linestyle='--', color='black')
  
  ax.set_ylim(ylim)
  ax.legend()
  ax.grid(True)
  ax.set_xlabel('Theta (degrees)')
  ax.set_ylabel('Element Pattern (dB)')
  
def fill_HPBW(ax: Axes, theta, pattern):
  
  import pattern_measurements as pm
  
  from utils import linear_to_db
  
  pattern_db = linear_to_db(pattern)
  theta_deg = np.degrees(theta)
  
  ylim = (-40, 3)
  for i in range(pattern_db.shape[0]):
    bw_idx = pm.HPBW_bounds(pattern[i, :], theta, idx=True)
    bw = (theta[bw_idx[0]], theta[bw_idx[1]])
    if ax.name == 'polar':
      ax.fill_between(theta, ylim[0], pattern_db[i, :], where=(theta >= bw[0]) & (theta <= bw[1]), alpha=0.3)
      ax.vlines(bw[0], pattern_db[i, bw_idx[0]], ylim[0], linestyle='--', alpha=0.5)
      ax.vlines(bw[1], pattern_db[i, bw_idx[1]], ylim[0], linestyle='--', alpha=0.5)
    else:
      ax.fill_between(theta_deg, ylim[0], pattern_db[i, :], where=(theta_deg >= np.degrees(bw[0])) & (theta_deg <= np.degrees(bw[1])), alpha=0.3)
      ax.vlines(np.degrees(bw[0]), pattern_db[i, bw_idx[0]], ylim[0], linestyle='--', alpha=0.5)
      ax.vlines(np.degrees(bw[1]), pattern_db[i, bw_idx[1]], ylim[0], linestyle='--', alpha=0.5)
      
def plot_array_layout(ax: Axes, n_elements, d, weights=None, center = True):
  
  import matplotlib.pyplot as plt
  
  d -= d[-1] / 2 if center else 0
  
  if weights is None: weights = np.ones(n_elements)
  ax.stem(d, weights)
  ax.set_title('Antenna Array Layout')
  ax.grid(True)