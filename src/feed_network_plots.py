import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

def main():
  
  fs_label = 14
  fs_title = 16
  
  line_styles = ['-', '--', ':', '-.']
  line_colors = [f"C{i}" for i in range(20)]
  
  divider_base_path = "src/divider_measurements/"
  divider_specific_path = "combined/"
  sim_path = "C:\\Users\\kamaj\\code\\Phased-Array\\src\\awr_output\\embedded"  
  divider_extension = ".s5p"
  ps_path = "C:\\Users\\kamaj\\code\\Phased-Array\\src\\phase_shifter"
  
  p1 = np.arange(3, -1, -1) * np.arange(5, 0, -1)[:, np.newaxis]
  p2 = np.arange(0, 4) * np.arange(0, 6)[:, np.newaxis]
  progression = np.concatenate((p1, p2))
  progression_names = [f"{prog[0]:02d}{prog[1]:02d}{prog[2]:02d}{prog[3]:02d}" for prog in progression]
  ps_names = [f"{ps_path}\\{ps_name:02d}.s2p" for ps_name in np.arange(0, 15)]
  phase_shifter_progression_names = [[f"{prog[0]:02d}", f"{prog[1]:02d}", f"{prog[2]:02d}", f"{prog[3]:02d}"] for prog in progression]
  
  ps_nets = [rf.Network(ps_name) for ps_name in ps_names]
  nets = [rf.Network(divider_base_path + divider_specific_path + f"{name}{divider_extension}") for name in progression_names]
  sim_nets = [rf.Network(sim_path + f"\\{name}{divider_extension}") for name in progression_names]
  
  s_db_np = np.array([net.s_db for net in nets])
  s_deg_np = np.array([net.s_deg for net in nets])
  # s_deg_np = np.array([net.s_deg_unwrap for net in nets])
  # s_deg_np -= np.max(s_deg_np)
  f = nets[0].f / 1e9
  
  
  ps_s_db_np = np.array([net['2-4ghz'].s_db for net in ps_nets])
  
  s11_mean = np.mean(s_db_np[:, :, 0 , 0], axis=0)
  s11_std = np.std(s_db_np[:, :, 0 , 0], axis=0)
  s11_max = np.max(s_db_np[:, :, 0 , 0], axis=0)
  
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)
  
  ax.plot(f, s_db_np[:, :, 0 , 0].T, linestyle=line_styles[0], color='black', alpha=0.1)
  ax.plot(f, s11_mean, linestyle=line_styles[0], color=line_colors[0], label='S11 Mean')
  
  ax.plot(f, s_db_np[:, :, 1 , 2].T, linestyle=line_styles[1], color='black', alpha=0.1)
  ax.plot(f, np.mean(s_db_np[:, :, 1, 2], axis=0), linestyle=line_styles[1], color=line_colors[1], label='Intra Branch Isolation Mean')
  # ax.plot(f, np.mean(s_db_np[:, :, 3, 4], axis=0), linestyle=line_styles[1], color=line_colors[1], label='Branch Isolation Mean')
  
  ax.plot(f, s_db_np[:, :, 3 , 2].T, linestyle=line_styles[2], color='black', alpha=0.1)
  ax.plot(f, np.mean(s_db_np[:, :, 3, 2], axis=0), linestyle=line_styles[2], color=line_colors[2], label='Inter Branch Isolation Mean')
  # ax.plot(f, np.mean(s_db_np[:, :, 4, 2], axis=0), linestyle=line_styles[2], color=line_colors[2], label='Inter Branch Isolation Mean')
  # ax.plot(f, np.mean(s_db_np[:, :, 4, 1], axis=0), linestyle=line_styles[2], color=line_colors[2], label='Inter Branch Isolation Mean')
  # ax.plot(f, np.mean(s_db_np[:, :, 3, 1], axis=0), linestyle=line_styles[2], color=line_colors[2], label='Inter Branch Isolation Mean')
  
  ax.set_xlim(2, 4)
  ax.set_xticks(np.arange(2, 4.01, 0.25))
  ax.set_ylim(-40, 0)
  ax.legend()
  ax.grid()
  ax.set_xlabel("Frequency (GHz)")
  ax.set_ylabel("Magnitude (dB)")
  ax.set_title("S-parameters Measurement Results")
  
  fig = plt.figure(figsize=(6, 5))
  # ax = fig.add_subplot(1, 2, 1)
  ax = fig.add_subplot(1, 1, 1)
  
  ax.plot(f, s_db_np[:, :, 0 , 1].T, linestyle=line_styles[0], color=line_colors[0], alpha=0.1)
  ax.plot(f, np.mean(s_db_np[:, :, 0, 1], axis=0), linestyle=line_styles[0], color=line_colors[0], label='S21 Mean')
 

  ax.plot(f, s_db_np[:, :, 0 , 2].T, linestyle=line_styles[1], color=line_colors[1], alpha=0.1)
  ax.plot(f, np.mean(s_db_np[:, :, 0, 2], axis=0), linestyle=line_styles[1], color=line_colors[1], label='S31 Mean')
 

  ax.plot(f, s_db_np[:, :, 0 , 3].T, linestyle=line_styles[2], color=line_colors[2], alpha=0.1)
  ax.plot(f, np.mean(s_db_np[:, :, 0, 3], axis=0), linestyle=line_styles[2], color=line_colors[2], label='S41 Mean')
 

  ax.plot(f, s_db_np[:, :, 0 , 4].T, linestyle=line_styles[3], color=line_colors[3], alpha=0.1)
  ax.plot(f, np.mean(s_db_np[:, :, 0, 4], axis=0), linestyle=line_styles[3], color=line_colors[3], label='S51 Mean')
 
  
  ax.legend()
  ax.grid()
  ax.set_xlabel("Frequency (GHz)", fontsize=fs_label)
  ax.set_ylabel("Magnitude (dB)", fontsize=fs_label)
  ax.set_title("S-parameters Measurement Results", fontsize=fs_title)
  ax.set_xlim(2, 4)
  ax.set_xticks(np.arange(2, 4.01, 0.25))
  ax.set_ylim(-12, -8)

  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)
  
  ax.plot(f, np.mean(s_db_np[:, :, 0, 1], axis=0) - np.mean(ps_s_db_np[:, :, 0, 1], axis=0), linestyle=line_styles[0], color=line_colors[0], label='S21 Mean')
  ax.plot(f, np.mean(s_db_np[:, :, 0, 2], axis=0) - np.mean(ps_s_db_np[:, :, 0, 1], axis=0), linestyle=line_styles[1], color=line_colors[1], label='S31 Mean')
  ax.plot(f, np.mean(s_db_np[:, :, 0, 3], axis=0) - np.mean(ps_s_db_np[:, :, 0, 1], axis=0), linestyle=line_styles[2], color=line_colors[2], label='S41 Mean')
  ax.plot(f, np.mean(s_db_np[:, :, 0, 4], axis=0) - np.mean(ps_s_db_np[:, :, 0, 1], axis=0), linestyle=line_styles[3], color=line_colors[3], label='S51 Mean')
  
  ax.legend()
  ax.grid()
  ax.set_xlabel("Frequency (GHz)", fontsize=fs_label)
  ax.set_ylabel("Magnitude (dB)", fontsize=fs_label)
  ax.set_title("Mean S-parameters compensated for Phase Shifter insertion loss", fontsize=fs_title)
  ax.set_xlim(2, 4)
  ax.set_xticks(np.arange(2, 4.01, 0.25))
  ax.set_ylim(-8, -6)
  
  # fig = plt.figure(figsize=(10, 6))
  # ax = fig.add_subplot(1, 1, 1)
  
  # s_db_np_sim = np.array([net.s_db for net in sim_nets])
  # f_sim = sim_nets[0].f / 1e9
  
  # ax.plot(f_sim, s_db_np_sim[:, :, 0 , 0].T, linestyle=line_styles[0], color='black', alpha=0.1)
  # ax.plot(f_sim, np.mean(s_db_np_sim[:, :, 0, 0], axis=0), linestyle=line_styles[0], color=line_colors[0], label='S11 Mean')
  # ax.plot(f, s_db_np[:, :, 0 , 0].T, linestyle=line_styles[1], color='red', alpha=0.1)
  # ax.plot(f, np.mean(s_db_np[:, :, 0, 0], axis=0), linestyle=line_styles[1], color='orange', label='S11 Measured Mean')
  
  # ax.set_xlim(2, 4)
  # ax.set_xticks(np.arange(2, 4, 0.5))
  # ax.set_xticks(np.arange(2, 4, 0.1), minor=True)
  # ax.set_ylim(-40, 0)
  
  fig1 = plt.figure(figsize=(6, 5))
  fig2 = plt.figure(figsize=(6, 5))
  # ax = fig.add_subplot(1, 1, 1)
  
  axs = [fig1.add_subplot(1, 1, 1), fig2.add_subplot(1, 1, 1)]
  ax = axs[0]
  
  ref_idx = 5
  show_idx = np.arange(0, 11)
  # show_idx = np.arange(0, 5)
  # show_idx = np.arange(10, 11)
  lsb_shift = 22.5
  ideal_phase = lsb_shift * np.arange(0, 16)
  ideal_phase_np = ideal_phase[:, np.newaxis] * np.ones_like(f)[np.newaxis, :]
    
  # s_deg_np = np.unwrap(s_deg_np + 180, axis=1, period=360)    
  s_phase_np = s_deg_np[:, :, 0, 1:5] - s_deg_np[ref_idx, :, 0, 1:5]
  s_phase_np[s_phase_np > 0] -= 360
  s_phase_np = -s_phase_np
  
  phase_progression = s_phase_np[:, :, 0:3] - s_phase_np[:, :, 1:4]
  
  phase_progression[phase_progression > 180] -= 360
  phase_progression[phase_progression < -180] += 360
  # phase_progression = np.abs(phase_progression)
  print(phase_progression.shape)
  
  progression_error = np.zeros_like(phase_progression)
  for i in range(phase_progression.shape[0]):
    for j in range(phase_progression.shape[2]):
      progression_error[i, :, j] = phase_progression[i, :, j] + ((i - 5) * lsb_shift)
  
  progression_error = np.sqrt(progression_error ** 2)
  
  ax.plot(f, progression_error[show_idx, :, 0].T, linestyle=line_styles[0], color=line_colors[0], alpha=0.1)
  ax.plot(f, progression_error[show_idx, :, 1].T, linestyle=line_styles[1], color=line_colors[1], alpha=0.1)
  ax.plot(f, progression_error[show_idx, :, 2].T, linestyle=line_styles[2], color=line_colors[2], alpha=0.1)

  mean_progression_error = np.mean(progression_error, axis=0)
  ax.plot(f, mean_progression_error[:, 0], linestyle=line_styles[0], color=line_colors[0], label='Mean phase difference error 1-2')
  ax.plot(f, mean_progression_error[:, 1], linestyle=line_styles[1], color=line_colors[1], label='Mean phase difference error 2-3')
  ax.plot(f, mean_progression_error[:, 2], linestyle=line_styles[2], color=line_colors[2], label='Mean phase difference error 3-4')  
  
  ax.legend()
  ax.set_xlim(2, 4)
  ax.set_xticks(np.arange(2, 4.01, 0.25))
  ax.grid()
  ax.set_ylim(0, 10)
  ax.set_xlabel("Frequency (GHz)", fontsize=fs_label)
  ax.set_ylabel("Phase (degrees)", fontsize=fs_label)
  ax.set_title("RMS Phase Progression Error", fontsize=fs_title)
  
  ax = axs[1]
  
  s_phase_error_np = np.zeros_like(s_phase_np)
  for i in range(s_phase_np.shape[0]):
    for j in range(s_phase_np.shape[2]):
      s_phase_error_np[i, :, j] = s_phase_np[i, :, j] - ideal_phase_np[progression[i, j], :]
  
  s_phase_error_np[s_phase_error_np > 180] -= 360
    
  s_phase_error_np = np.sqrt(s_phase_error_np ** 2)
    
  ax.plot(f, s_phase_error_np[:, :, 0].T, linestyle=line_styles[0], color=line_colors[0], alpha=0.1)
  ax.plot(f, np.mean(s_phase_error_np[:, :, 0], axis=0), linestyle=line_styles[0], color=line_colors[0], label="S21 Mean")
  ax.plot(f, s_phase_error_np[:, :, 1].T, linestyle=line_styles[1], color=line_colors[1], alpha=0.1)
  ax.plot(f, np.mean(s_phase_error_np[:, :, 1], axis=0), linestyle=line_styles[1], color=line_colors[1], label="S31 Mean")
  ax.plot(f, s_phase_error_np[:, :, 2].T, linestyle=line_styles[2], color=line_colors[2], alpha=0.1)
  ax.plot(f, np.mean(s_phase_error_np[:, :, 2], axis=0), linestyle=line_styles[2], color=line_colors[2], label="S41 Mean")
  ax.plot(f, s_phase_error_np[:, :, 3].T, linestyle=line_styles[3], color=line_colors[3], alpha=0.1)
  ax.plot(f, np.mean(s_phase_error_np[:, :, 3], axis=0), linestyle=line_styles[3], color=line_colors[3], label="S51 Mean")
  
  ax.legend()
  ax.set_xlim(2, 4)
  ax.set_xticks(np.arange(2, 4.01, 0.25))
  ax.grid()
  ax.set_ylim(0, 10)
  ax.set_xlabel("Frequency (GHz)", fontsize=fs_label)
  ax.set_ylabel("Phase (degrees)", fontsize=fs_label)
  ax.set_title("Phase Shifter RMS Phase Error", fontsize=fs_title)
  plt.show()
    
  
if __name__ == "__main__":
  main()