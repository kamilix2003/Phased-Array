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

def plot_pattern(ax: Axes, pattern, theta, decibel=True, element_pattern=None):
  
  import pattern_measurements as pm
    
  from utils import linear_to_db
  
  if pattern.ndim == 2:
    pattern = pattern[0, :]
  
  pattern_db = linear_to_db(pattern)
  theta_deg = np.degrees(theta)
  
  if decibel:
    ylim = (-20, 1)
  else:
    ylim = (0, 1.05)
  
  if ax.name == 'polar':
      ax.set_theta_zero_location("N")
      ax.set_theta_direction(-1)
      ax.set_thetalim(np.radians([-90, 90]))
      ax.set_rlim(ylim)

  threshold = 0.5 if not decibel else -3
  fnbw_idx = pm.get_lobe(pattern, theta)
  fnbw = np.abs(theta[fnbw_idx[-1]] - theta[fnbw_idx[0]])
  if decibel:
    hpbw_idx = fnbw_idx[pattern_db[fnbw_idx] > (np.max(pattern_db) + threshold)]
  else:
    hpbw_idx = fnbw_idx[pattern[fnbw_idx] > threshold * np.max(pattern)]
  hpbw = np.abs(theta[hpbw_idx[-1]] - theta[hpbw_idx[0]])
  
  pattern_wo_main = np.copy(pattern)
  pattern_wo_main[fnbw_idx] = 0
  side_lobe_peak_idx = np.argmax(pattern_wo_main)
  main_lobe_peak_idx = np.argmax(pattern)
  
  if decibel:
    ax.plot(theta_deg, pattern_db, label=f'direction: {np.degrees(pm.main_lobe_direction(pattern_db, theta)):.2f} deg')
    ax.plot(theta_deg[fnbw_idx], pattern_db[fnbw_idx], color='red', alpha=0.5, label=f'FNBW: {np.degrees(fnbw):.2f} deg')
    ax.plot(theta_deg[hpbw_idx], pattern_db[hpbw_idx], color='green', alpha=1, label=f'HPBW: {np.degrees(hpbw):.2f} deg')
    ax.hlines(pattern_db[side_lobe_peak_idx], xmin=theta_deg[side_lobe_peak_idx], xmax=theta_deg[main_lobe_peak_idx], color='black', linestyle='--', alpha=0.5, label=f'SLL/GLL: {pattern_db[side_lobe_peak_idx]:.2f} dB')
    ax.vlines(theta_deg[main_lobe_peak_idx], pattern_db[side_lobe_peak_idx], pattern_db[main_lobe_peak_idx],
              color='black', linestyle='--', alpha=0.5, label=f'ML-SL/GL distance: {np.abs(theta_deg[main_lobe_peak_idx]-theta_deg[side_lobe_peak_idx]):.2f} deg')
  else:
    ax.plot(theta_deg, pattern, label=f'direction: {np.degrees(pm.main_lobe_direction(pattern, theta)):.2f} deg')
    ax.plot(theta_deg[fnbw_idx], pattern[fnbw_idx], color='red', alpha=0.5, label=f'FNBW: {np.degrees(fnbw):.2f} deg')
    ax.plot(theta_deg[hpbw_idx], pattern[hpbw_idx], color='green', alpha=1, label=f'HPBW: {np.degrees(hpbw):.2f} deg')
    ax.hlines(pattern[side_lobe_peak_idx], xmin=theta_deg[side_lobe_peak_idx], xmax=theta_deg[main_lobe_peak_idx], color='black', linestyle='--', alpha=0.5, label=f'SLL/GLL: {pattern[side_lobe_peak_idx]:.2f}')
    ax.vlines(theta_deg[main_lobe_peak_idx], pattern[side_lobe_peak_idx], pattern[main_lobe_peak_idx],
              color='black', linestyle='--', alpha=0.5, label=f'ML-SL/GL distance: {np.abs(theta_deg[main_lobe_peak_idx]-theta_deg[side_lobe_peak_idx]):.2f} deg')
    
  if element_pattern is not None:
    ep_db = linear_to_db(element_pattern)
    if decibel:
      ax.plot(theta_deg, ep_db, label='Element Pattern', alpha=0.5, linestyle='--', color='black')
    else:
      ax.plot(theta_deg, element_pattern, label='Element Pattern', alpha=0.5, linestyle='--', color='black')
      
  
  

  ax.set_ylim(ylim)
  ax.legend()
  ax.grid(True)
  ax.set_xlabel('Theta (degrees)')
  ax.set_ylabel('Array Pattern (dB)' if decibel else 'Array Pattern (linear)')
  
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