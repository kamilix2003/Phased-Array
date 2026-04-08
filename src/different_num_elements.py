import numpy as np

import matplotlib.pyplot as plt
from scipy.constants import c

from antenna_array import phase_shift, array_factor
from pattern import gen_patch_pattern
from pattern_measurements import (main_lobe_direction, get_lobe, FNBW, SLL)

def main():    
  
  fs_label = 14
  fs_title = 16
  
  Ns = np.array([2, 4, 6])
  n_bits = 4
  lsb_shift = 2 * np.pi / (2 ** n_bits)
  theta = np.linspace(-np.pi/2, np.pi/2, 360)
  # theta = np.linspace(-np.pi, np.pi, 360)
  theta_deg = np.degrees(theta)
  fs = np.array([2.4e9, 3e9])
  f = fs[0]
  line_styles = ['-', '--', ':', '-.']
  line_colors = [f"C{i}" for i in range(20)]
  ep = gen_patch_pattern(theta)
  ep_db = 20 * np.log10(ep / np.max(ep))
  
  fig1 = plt.figure(figsize=(6, 5))
  fig2 = plt.figure(figsize=(6, 5))
  cols = 2
  # # axs = fig.subplots(beta.shape[0] // cols, cols, sharex=True, subplot_kw={"projection": "polar"})
  # axs = fig.subplots(Ns.shape[0] // cols, cols)
  # axs = axs.flatten()
  
  axs = [None] * 2
  axs[0] = fig1.add_subplot(1, 1, 1)
  axs[1] = fig2.add_subplot(1, 1, 1)
  
  for i, N in enumerate(Ns):
    weights = np.ones(N)
    ds = [
          50e-3 * np.arange(N),
          62.5e-3 * np.arange(N),
          75e-3 * np.arange(N),
          100e-3 * np.arange(N),
          125e-3 * np.arange(N)
          ]
    d = ds[0]
    print(f"element positions (m): {ds}")
    # beta = (np.arange(N) * lsb_shift) * np.arange(0, (2**n_bits) // (N - 1) + 1)[:, np.newaxis]
    progression = 3
    beta = (np.arange(N) * lsb_shift) * np.array([0, progression])[:, np.newaxis]
    print(f"Beta (degrees): {np.degrees(beta)}")
          
    af = array_factor(weights, N, 
                      phase_shift(d, f, theta, beta))
    ap = np.abs(af)# * ep
    ap_db = 20 * np.log10(ap / np.max(ap))
    for b in range(beta.shape[0]):
      ax = axs[b]
      
      ax.plot((theta_deg), ap_db[b, :], label=f"N = {N}, spacing = {d[1] / (c/f):.2f} λ", linestyle=line_styles[i], color=line_colors[i])
      ax.legend(loc='lower right')
      ax.set_ylim(-30, 3)
      ax.set_xlim(-90, 90)
      ax.set_ylabel("Magnitude (dB)", fontsize=fs_label)
      ax.set_xlabel("Angle (degrees)", fontsize=fs_label)
      ax.grid()
      
  axs[0].set_title(f'Array factor, progression = {0} degrees', fontsize=fs_title) #, control bytes: {[f"0x{int(s):01x}" for s in (beta[b, :] // lsb_shift)%16]}')
  axs[1].set_title(f'Array factor, progression = {np.degrees(progression * lsb_shift)} degrees', fontsize=fs_title) #, control bytes: {[f"0x{int(s):01x}" for s in (beta[b, :] // lsb_shift)%16]}')
  
  plt.show()
      
if __name__ == "__main__":
  main()