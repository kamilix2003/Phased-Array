from antenna_array import array_factor, phase_shift
from pattern import gen_rect_pattern
from beam_steering import steer_to_phase, quantize_phase, gen_steer_directions
from spacing import gen_spacing
from cost import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

def main():
  N = 360
  theta = np.linspace(-np.pi/2, np.pi/2, N)
  f = 2.4e9
  
  element_pattern = gen_rect_pattern(theta, f)

  fig = plt.figure(figsize=(10, 6))
  spacings = np.linspace(0.1, 1, 30)
  
  beams = [2, 4]
  for i, beam_count in enumerate(beams, start=1):
    ax = fig.add_subplot(len(beams), 1, i)
  
    for N_elements in [2, 4, 5]:
      cov_cost = []
      beta =  np.arange(beam_count)[:, np.newaxis] * np.arange(N_elements) * (np.pi / 2**4)
      weights = np.ones(N_elements)
      for spacing in spacings:
        d = gen_spacing(N_elements, [spacing]) * c / f
        psi = phase_shift(d, f, theta, beta);
        af = array_factor(weights, N_elements, psi)
        ap = np.abs(af) * element_pattern
        # cov_cost.append(coverage_cost_v2(theta, ap, digital=False))
        cov_cost.append(coverage_cost(theta, ap))
      
      ax.plot(spacings, cov_cost, marker='o', label=f'Coverage Cost N={N_elements}')
      ax.legend()
      ax.set_ylabel('Coverage Cost')
      ax.set_title(f'Coverage Cost vs Element Spacing for {beam_count} Beams')
      ax.grid(True)
      
  ax.set_xlabel('Element Spacing (λ)')
  plt.show()
  
if __name__ == "__main__":
  main()