import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from pattern_import import RadiationPattern
from pattern_measurements import HPBW

def main():
  
  fs_label = 14
  fs_title = 16
  
  base_path = "src/full_measurements/ant_final/"
  extension = ".txt"
  L_pattern_paths = [
    "prom_2",
    "00000000",
    "00010203(01020304)",
    "00020406",
    "00030609",
    "00040812(dobre hardcode)",
    "00051015(hardcode)"
  ]
  R_pattern_paths = [
    "prom_2",
    "00000000",
    "03020100",
    "06040200",
    "09060300",
    "12080400",
    "15100500"
  ]
  # pattern_paths = L_pattern_paths
  # pattern_paths = R_pattern_paths
  pattern_paths = L_pattern_paths[2::][::-1] + R_pattern_paths[1::]
  print(pattern_paths)
  pattern_paths = [base_path + p + extension for p in pattern_paths]
  # frequencies = [2.4, 2.5, 3, 3.5]
  # frequencies = [2.4, 3]
  frequencies = [2.4]

  line_styles = ['-', '--', ':', '-.']
  line_colors = [f"C{i}" for i in range(20)]

  fig = plt.figure(figsize=(10, 6))
  # ax = fig.add_subplot(1, 1, 1)
  axs = fig.subplots(4, 3).flatten()

  beam_directions = []

  for i, pattern_path in enumerate(pattern_paths[:]):
    pattern = RadiationPattern(pattern_path)
    theta = np.radians(pattern.azimuth)
    data, indexes = pattern.get_pattern(frequency=frequencies)
    print(f"Pattern from {pattern_path}")
    for j, freq_idx in enumerate(indexes['frequency']):
      freq = pattern.frequency[freq_idx]
      current_pattern = data[:, 0, 0, 0, j]
      current_pattern -= np.max(current_pattern)
      beam_dir = theta[np.argmax(current_pattern  + 2*np.cos(theta / 5))]  # adding a small progression to shift the main lobe direction
      beam_directions.append(np.degrees(beam_dir))
      
      # axs[i].plot(np.degrees(theta), current_pattern, linestyle=line_styles[0], color=line_colors[i], label=f"Progression {i * 22.5}, dir={np.degrees(beam_dir):.2f}°")
      axs[i].plot(np.degrees(theta), current_pattern, linestyle=line_styles[0], color=line_colors[i], label=f"Progression {i * 22.5}, dir={np.degrees(beam_dir):.2f}°")
      # axs[i].hlines(-3, -90, 90, colors='gray', linestyles='dashed', linewidth=0.5)
      
    axs[i].set_xlim(-90, 90)
    # axs[i].set_ylim(-30, 3)
    axs[i].grid()
    axs[i].set_xticks(np.arange(-90, 91, 15))
    axs[i].legend() 
  
  hpbw_values = [48-22, 42-15, 38-10, 30-4, 22+9, 16+12, 7+22, 0+32, 42-12, 63-24, 19+12]
  print(hpbw_values)
  # REVERSED sll_values = [-5, -3.7, -8, -7.5, 7.3, -8, -7, -3.5, -5, -4.5, -3]
  sll_values = np.array([-3, -4.5, -5, -3.5, -7, -8, 7.3, -7.5, -6.3, -3.7, -5])
    
  expected_beam_directions = [ -0,  -8, -18, -27, -38, -51] 
  expected_beam_directions = np.array(expected_beam_directions[::-1] + expected_beam_directions[1::])
  expected_beam_directions[5:] = expected_beam_directions[5:] * -1
  expected_hpbw = [32, 33, 34, 37, 44, 60,]
  expected_hpbw = np.array(expected_hpbw[::-1] + expected_hpbw[1::])
  
  
  print(expected_hpbw, expected_beam_directions)
  
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)
  ax_err = plt.twinx(ax)
  progression = np.arange(-5, 6) * 22.5
  ax.plot(progression, beam_directions, marker='o', linestyle='', label='Measured Beam Directions')
  ax.plot(progression[-1], beam_directions[-1], marker='o', linestyle='', color="red", label='Invalid Measurement')
  ax_err.bar(progression[:-1], np.abs(expected_beam_directions[:-1] - beam_directions[:-1]), width=2, alpha=0.3, label='Error (degrees)', color="red")
  ax_err.set_ylim(0, 10)
  ax_err.set_ylabel("Error (degrees)")
  ax_err.legend()
  ax.plot(progression, expected_beam_directions, marker='x', linestyle='', label='Expected Beam Directions')
  ax.set_xlabel("Progression (degrees)", fontsize=fs_label)
  ax.set_ylabel("Beam Direction (degrees)", fontsize=fs_label)
  ax.set_title("Beam Direction vs. Phase Progression", fontsize=fs_title)
  ax.legend()
  ax.grid(True)
  ax.set_ylim(-60, 60)
  ax.set_yticks(np.arange(-60, 61, 10))
  ax.set_yticks(np.arange(-60, 61, 1), minor=True)
  ax.set_xticks(progression)
  
  fig = plt.figure(figsize=(8, 5))
  ax = fig.add_subplot(1, 1, 1)
  progression = np.arange(-5, 6) * 22.5
  ax.plot(progression, hpbw_values, marker='o', linestyle='', label='Measured HPBW')
  ax.plot(progression, expected_hpbw, marker='x', linestyle='', label='Expected HPBW')
  ax.set_xlabel("Progression (degrees)", fontsize=fs_label)
  ax.set_ylabel("Beam width (degrees)", fontsize=fs_label)
  ax.set_title("Beam width vs. Phase Progression", fontsize=fs_title)
  ax.legend()
  ax.grid(True)
  ax.set_xticks(progression)
  
  fig = plt.figure(figsize=(8, 5))
  ax = fig.add_subplot(1, 1, 1)
  
  ax.plot(progression, sll_values, marker='o', linestyle='', label='Measured SLL')
  ax.set_xlabel("Progression (degrees)", fontsize=fs_label)
  ax.set_ylabel("Side Lobe Level (dB)", fontsize=fs_label)
  
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)
  
  ep_pattern = RadiationPattern(base_path + "prom_2" + extension)
  theta = np.radians(ep_pattern.azimuth)
  data, indexes = ep_pattern.get_pattern(frequency=[2.4, 3])
  
  for j, freq_idx in enumerate(indexes['frequency']):
      freq = ep_pattern.frequency[freq_idx]
      current_pattern = data[:, 0, 0, 0, j]
      ep = current_pattern.copy()
      current_pattern -= np.max(current_pattern)
      ax.plot(np.degrees(theta), current_pattern, linestyle=line_styles[j], color=line_colors[j], label=f"frequency {freq} GHz")
  
  ax.set_xlim(-90, 90)
  ax.set_ylim(-30, 3)
  ax.grid()
  ax.set_xticks(np.arange(-90, 91, 15))
  ax.set_title(f"Normalized radiation pattern of single antenna element", fontsize=fs_title)
  ax.set_xlabel("Angle (degrees)", fontsize=fs_label)
  ax.set_ylabel("Magnitude (dB)", fontsize=fs_label)
  ax.legend()
  
  path_idx = [5, 8]
  # path_idx = np.arange(len(pattern_paths))
  for i in path_idx:
    pattern_path = pattern_paths[i]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    pattern = RadiationPattern(pattern_path)
    theta = np.radians(pattern.azimuth)
    data, indexes = pattern.get_pattern(frequency=frequencies)
    print(f"Pattern from {pattern_path}")
    for j, freq_idx in enumerate(indexes['frequency']):
      freq = pattern.frequency[freq_idx]
      current_pattern = data[:, 0, 0, 0, j]
      current_pattern -= np.max(current_pattern)
      beam_dir = theta[np.argmax(current_pattern  + 2*np.cos(theta / 5))]  # adding a small progression to shift the main lobe direction
      beam_directions.append(np.degrees(beam_dir))

      # ax.plot(ep_pattern.azimuth, ep, linestyle='--', color='gray', label="Element pattern")
      ax.plot(np.degrees(theta), current_pattern, linestyle=line_styles[0], color=line_colors[0], label=f"beam direction={np.degrees(beam_dir):.2f}°")
      ax.hlines(-3, -90, 90, colors='gray', linestyles='dashed', linewidth=0.5)
      
    ax.set_xlim(-90, 90)
    ax.set_ylim(-30, 3)
    ax.grid()
    ax.set_xticks(np.arange(-90, 91, 15))
    ax.set_title(f"Normalized radiation pattern, Progression {(i - 5) * 22.5} degrees", fontsize=fs_title)
    ax.set_xlabel("Angle (degrees)", fontsize=fs_label)
    ax.set_ylabel("Magnitude (dB)", fontsize=fs_label)
    ax.legend() 
      
  appendix_paths = [
    "00051015(hardcode)",
    "00040812(dobre hardcode)",
    "00020406",
    "00030609(hard code)",
    "00010203(01020304)",
    "03020100",
    "06040200",
    "09060300",
    "12080400",
    "15100500"
  ]
  progs = np.arange(-5, 0).tolist() + np.arange(1, 6).tolist()
  fig1 = plt.figure(figsize=(12, 12))
  fig2 = plt.figure(figsize=(12, 12))
  axs = fig1.subplots(5, 1, sharex=True, sharey=True).flatten()
  axs = np.append(axs, fig2.subplots(5, 1, sharex=True, sharey=True).flatten())
  beam_directions = []
  for i, pattern_path in enumerate(appendix_paths):
    full_path = base_path + pattern_path + extension
    pattern = RadiationPattern(full_path)
    theta = np.radians(pattern.azimuth)
    data, indexes = pattern.get_pattern(frequency=frequencies)
    print(f"Pattern from {full_path}")
    for j, freq_idx in enumerate(indexes['frequency']):
      freq = pattern.frequency[freq_idx]
      current_pattern = data[:, 0, 0, 0, j]
      current_pattern -= np.max(current_pattern)
      beam_dir = theta[np.argmax(current_pattern  + 2*np.cos(theta / 5))]  # adding a small progression to shift the main lobe direction
      beam_directions.append(np.degrees(beam_dir))
      axs[i].plot(np.degrees(theta), current_pattern, linestyle=line_styles[0], color=line_colors[i])
      axs[i].hlines(-3, -180, 180, colors='gray', linestyles='dashed', linewidth=0.5)
      
    axs[i].set_xlim(-180, 180)
    axs[i].set_ylim(-30, 3)
    axs[i].grid()
    axs[i].set_xticks(np.arange(-180, 181, 15))
    axs[i].set_title(f"Progression {progs[i] * 22.5} degrees, beam angle {beam_directions[i]:.2f}degrees")
    # axs[i].set_xlabel("Angle (degrees)")
    axs[i].set_ylabel("Magnitude (dB)")
  
  axs[4].set_xlabel("Angle (degrees)")
  axs[-1].set_xlabel("Angle (degrees)")
  fig1.savefig("prog1.svg")
  fig2.savefig("prog2.svg")
    
  plt.show()
    
if __name__ == "__main__":
    main()