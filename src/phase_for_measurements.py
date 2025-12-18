import numpy as np

import matplotlib.pyplot as plt
from scipy.constants import c

from antenna_array import phase_shift, array_factor
from pattern import gen_patch_pattern
from pattern_measurements import (main_lobe_direction, get_lobe)

def main():    
  
  N = 4
  n_bits = 4
  lsb_shift = 2 * np.pi / (2 ** n_bits)
  theta = np.linspace(-np.pi/2, np.pi/2, 360)
  # theta = np.linspace(-np.pi, np.pi, 360)
  fs = np.array([2.4e9, 3e9])
  line_styles = ['-', '--', ':', '-.']
  line_colors = [f"C{i}" for i in range(20)]
  ep = gen_patch_pattern(theta)
  ep_db = 20 * np.log10(ep / np.max(ep))
  
  weights = np.ones(N)
  d = 50e-3 * np.arange(N)
  print(f"element positions (m): {d}")
  beta = (np.arange(N) * lsb_shift) * np.arange(0, (2**n_bits) // (N - 1) + 1)[:, np.newaxis]
  print(f"Beta (degrees): {np.degrees(beta)}")
  
  fig = plt.figure(figsize=(10, 6))
  cols = 2
  # axs = fig.subplots(beta.shape[0] // cols, cols, sharex=True, subplot_kw={"projection": "polar"})
  axs = fig.subplots(beta.shape[0] // cols, cols, sharex=True)
  axs = axs.flatten()
    
  for i, f in enumerate(fs):
    af = array_factor(weights, N, 
                      phase_shift(d, f, theta, beta))
    ap = np.abs(af)# * ep
    ap_db = 20 * np.log10(ap / np.max(ap))
    for b in range(beta.shape[0]):
      ax = axs[b]
      
      ax.plot((theta), ep_db, color='black', linewidth=.5, linestyle='--')
      ax.plot((theta), ap_db[b, :], label=f'f={f/1e9} GHz, beam direction={main_lobe_direction(ap_db[b, :], theta, format='degrees'):.1f} degrees', linestyle=line_styles[i], color=line_colors[b])
      ax.legend(loc='upper right', fontsize='small')
      ax.set_title(f'Phase shift step: {np.degrees(lsb_shift * (b)):.1f} degrees, control bytes: {[f"0x{int(s):01x}" for s in (beta[b, :] // lsb_shift)%16]}')
      ax.set_ylim(-30, 3)
      ax.grid()
    
  
  plt.show()
      
if __name__ == "__main__":
  main()