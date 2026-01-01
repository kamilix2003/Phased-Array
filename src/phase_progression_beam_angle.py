import numpy as np
import matplotlib.pyplot as plt

def main():
  
  bits = 4
  lsb_shift = 2 * np.pi / (2 ** bits)
  line_styles = ['-', '--', ':', '-.']
  line_colors = [f"C{i}" for i in range(20)]
  # phase_progression = np.linspace(-np.pi, np.pi, 500)
  phase_progression = np.arange(0, 2**bits) * lsb_shift
  phase_progression = phase_progression[:6]
  phase_progression = np.concatenate((-phase_progression[::-1], phase_progression[1:]))
  print(f"Phase progression (radians): {phase_progression}", phase_progression.shape)
  phase_progression_deg = np.degrees(phase_progression)
  
  ds = np.array([0.4, 0.5, 0.6])
  
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)
  ax.set_xticks(phase_progression_deg)
  ax.set_yticks(np.arange(-90, 91, 10))
  ax.set_yticks(np.arange(-90, 91, 1), minor=True)
  
  for i, d in enumerate(ds):
    beam_angles = np.degrees(np.arcsin(phase_progression / (2 * np.pi * d)))
    ax.plot(phase_progression_deg, beam_angles, label=f'd={d} λ max steer angle={beam_angles[-1]:.1f}°', marker='o', color=line_colors[i], linestyle=line_styles[i])
    ax.hlines(beam_angles[0], phase_progression_deg[0] * 1.1, phase_progression_deg[-1] * 1.1, color=line_colors[i], linestyles=line_styles[-1], linewidth=0.75)
    ax.hlines(beam_angles[-1], phase_progression_deg[0] * 1.1, phase_progression_deg[-1] * 1.1, color=line_colors[i], linestyles=line_styles[-1], linewidth=0.75)
  
  ax.set_xlim(phase_progression_deg[0] * 1.1, phase_progression_deg[-1] * 1.1)
  ax.set_xlabel("Phase progression (degrees)")
  ax.set_ylabel("Beam angle (degrees)")
  ax.legend(loc="lower right", fontsize='small')
  ax.set_title(f"Beam Angle vs Phase Progression ({bits}-bit phase shifter)")
  ax.grid()
  plt.show()
  
if __name__ == "__main__":
  main()