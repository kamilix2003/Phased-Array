import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

# k = 2 * np.pi / wavelength

def beamwidth_steering(beam_angle, d, N):
  """ Half-power beamwidth with beam steering taken into account """
  x1 = np.arccos( np.clip(np.cos(beam_angle), -.75, .75) - (2.782 / (N * 2 * np.pi * d)))
  x2 = np.arccos( np.clip(np.cos(beam_angle), -.75, .75) + (2.782 / (N * 2 * np.pi * d)))
  return np.degrees(x1 - x2)

def main():
  
  N = 4
  bits = 4
  lsb_shift = 2 * np.pi / (2 ** bits)
  phase_progression = np.arange(0, 2**bits) * lsb_shift
  phase_progression = phase_progression[:6]
  phase_progression = np.concatenate((-phase_progression[::-1], phase_progression[1:]))
  ds = np.array([0.4, 0.5, 0.6])
  # beam_angles = np.arange(0, 180, 1)
  
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)
  ax.set_xticks(np.degrees(phase_progression))
  ax.set_yticks(np.arange(0, 91, 5))
  
  for i, d in enumerate(ds):
    beam_angles = np.degrees(np.arcsin(phase_progression / (2 * np.pi * d))) + 90
    bw = np.array([beamwidth_steering(np.radians(ba), d, N) for ba in beam_angles])
    print(bw)
    # ax.plot(beam_angles, bw, label=f'd={d} lambda', marker='o')
    ax.plot(np.degrees(phase_progression), bw, label=f'd={d} λ', marker='o')
    
  ax.set_xlabel("Phase progression (degrees)")
  ax.set_ylabel("Half-Power Beamwidth (degrees)")
  ax.set_title(f"Half-Power Beamwidth vs Phase Progression")
  ax.grid()
  ax.set_ylim(20, 50)
  ax.legend()
  plt.show()

if __name__ == "__main__":
    main()