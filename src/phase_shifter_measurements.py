import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

def main():
  
  line_styles = ['-', '--', ':', '-.']
  line_colors = [f"C{i}" for i in range(20)]
  
  base_path = "src/phase_shifter/"
  extension = ".s2p"
  phase_shifter_paths = [f"{i:02d}" for i in range(16)]
  phase_shifter_paths.append("ref")
  phase_shifter_paths = [base_path + p + extension for p in phase_shifter_paths]
  print(phase_shifter_paths)
  
  nets = [rf.Network(net_path) for net_path in phase_shifter_paths]
  
  s_mag_np = np.array([net.s_mag for net in nets[:-1]])
  s_deg_np = np.array([net.s_deg_unwrap for net in nets[:-1]])
  
  ref_mag_np = nets[-1].s_mag
  ref_deg_np = 2*(nets[-1].s_deg + 180 - 12.5) % 360 - 180  # Wrap to [-180, 180]
  # ref_deg_np = nets[-1].s_deg_unwrap
  
  print(s_mag_np.shape)
  
  s11_mag_max = np.max(s_mag_np[:, :, 0 , 0], axis=0)
  s21_mag_max = np.max(s_mag_np[:, :, 1 , 0], axis=0)
  s12_mag_max = np.max(s_mag_np[:, :, 0 , 1], axis=0)
  s22_mag_max = np.max(s_mag_np[:, :, 1 , 1], axis=0)
  
  s11_mag_min = np.min(s_mag_np[:, :, 0 , 0], axis=0)
  s21_mag_min = np.min(s_mag_np[:, :, 1 , 0], axis=0)
  s12_mag_min = np.min(s_mag_np[:, :, 0 , 1], axis=0)
  s22_mag_min = np.min(s_mag_np[:, :, 1 , 1], axis=0)
  
  s11_mag_mean = np.mean(s_mag_np[:, :, 0 , 0], axis=0)
  s21_mag_mean = np.mean(s_mag_np[:, :, 1 , 0], axis=0)
  s12_mag_mean = np.mean(s_mag_np[:, :, 0 , 1], axis=0)
  s22_mag_mean = np.mean(s_mag_np[:, :, 1 , 1], axis=0)
  
  s11_mag_std = np.std(s_mag_np[:, :, 0 , 0], axis=0)
  s21_mag_std = np.std(s_mag_np[:, :, 1 , 0], axis=0)
  s12_mag_std = np.std(s_mag_np[:, :, 0 , 1], axis=0)
  s22_mag_std = np.std(s_mag_np[:, :, 1 , 1], axis=0)

  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)
  
  ax.plot(nets[0].f/1e9, 20 * np.log10(s_mag_np[:-1, :, 1 ,0]).T, linestyle=line_styles[3], color="black", alpha=0.1)
  ax.plot(nets[0].f/1e9, 20 * np.log10(s_mag_np[-1, :, 1 ,0]).T, linestyle=line_styles[3], color="black", alpha=0.1, label='Individual S21 Magnitudes')
  ax.plot(nets[0].f/1e9, 20 * np.log10(s21_mag_min), linestyle=line_styles[1], color=line_colors[1], alpha=0.75, label='S21 Min Magnitude')
  ax.plot(nets[0].f/1e9, 20 * np.log10(s21_mag_mean+s21_mag_std), linestyle=line_styles[2], color=line_colors[1], label='Mean ± Std Dev')
  ax.plot(nets[0].f/1e9, 20 * np.log10(s21_mag_mean-s21_mag_std), linestyle=line_styles[2], color=line_colors[1])
  ax.plot(nets[0].f/1e9, 20 * np.log10(s21_mag_mean), linestyle=line_styles[0], color=line_colors[1], label='S21 Mean Magnitude')
  
  ax.set_xlabel("Frequency (GHz)")
  ax.set_ylabel("Magnitude (dB)")
  ax.set_title("Phase Shifter S21 Magnitudes")
  ax.grid(True, which='both', linestyle='--', linewidth=0.5)
  ax.set_yticks(np.arange(-4, 0, 0.1), minor=True)
  ax.set_yticks(np.arange(-4, 1, 0.5), minor=False)
  ax.set_xticks(np.arange(2, 5, 0.1), minor=True)
  ax.set_xticks(np.arange(2, 5, 0.5), minor=False)
  ax.legend()
  ax.set_xlim(2, 4)
  ax.set_ylim(-4, -2)

  fig = plt.figure(figsize=(10, 5))
  # ax = fig.add_subplot(1, 2, 1)
  axs = fig.subplots(1, 2)
  ax = axs[0]

  ax.plot(nets[0].f/1e9, 20 * np.log10(s_mag_np[:-1, :, 0 ,0]).T, linestyle=line_styles[3], color="black", alpha=0.1)
  ax.plot(nets[0].f/1e9, 20 * np.log10(s_mag_np[-1, :, 0 ,0]).T, linestyle=line_styles[3], color="black", alpha=0.1, label='Individual S11 Magnitudes')
  ax.plot(nets[0].f/1e9, 20 * np.log10(s11_mag_max), linestyle=line_styles[1], color=line_colors[0], label='S11 Max Magnitude')
  ax.plot(nets[0].f/1e9, 20 * np.log10(s11_mag_mean+s11_mag_std), linestyle=line_styles[2], color=line_colors[0], label='Mean ± Std Dev')
  ax.plot(nets[0].f/1e9, 20 * np.log10(s11_mag_mean-s11_mag_std), linestyle=line_styles[2], color=line_colors[0])
  ax.plot(nets[0].f/1e9, 20 * np.log10(s11_mag_mean), linestyle=line_styles[0], color=line_colors[0], label='S11 Mean Magnitude')
    
  ax.set_xlabel("Frequency (GHz)")
  ax.set_ylabel("Magnitude (dB)")
  ax.set_title("Phase Shifter S11 Magnitudes")
  ax.grid(True, which='both', linestyle='--', linewidth=0.5)
  ax.set_yticks(np.arange(-40, 0, 1), minor=True)
  ax.set_yticks(np.arange(-40, 1, 5), minor=False)
  ax.set_xticks(np.arange(2, 5, 0.1), minor=True)
  ax.set_xticks(np.arange(2, 5, 0.5), minor=False)
  ax.legend()
  ax.set_xlim(2, 4)
  ax.set_ylim(-40, 0)
    
  # ax = fig.add_subplot(1, 2, 2)
  ax = axs[1]
  
  ax.plot(nets[0].f/1e9, 20 * np.log10(s_mag_np[:-1, :, 1 ,1]).T, linestyle=line_styles[3], color="black", alpha=0.1)
  ax.plot(nets[0].f/1e9, 20 * np.log10(s_mag_np[-1, :, 1 ,1]).T, linestyle=line_styles[3], color="black", alpha=0.1, label='Individual S22 Magnitudes')
  ax.plot(nets[0].f/1e9, 20 * np.log10(s22_mag_max), linestyle=line_styles[1], color=line_colors[3], label='S22 Max Magnitude')
  ax.plot(nets[0].f/1e9, 20 * np.log10(s22_mag_mean+s22_mag_std), linestyle=line_styles[2], color=line_colors[3], label='Mean ± Std Dev')
  ax.plot(nets[0].f/1e9, 20 * np.log10(s22_mag_mean-s22_mag_std), linestyle=line_styles[2], color=line_colors[3])
  ax.plot(nets[0].f/1e9, 20 * np.log10(s22_mag_mean), linestyle=line_styles[0], color=line_colors[3], label='S22 Mean Magnitude')
  
  ax.set_xlabel("Frequency (GHz)")
  ax.set_ylabel("Magnitude (dB)")
  ax.set_title("Phase Shifter S22 Magnitudes")
  ax.grid(True, which='both', linestyle='--', linewidth=0.5)
  ax.set_yticks(np.arange(-40, 0, 1), minor=True)
  ax.set_yticks(np.arange(-40, 1, 5), minor=False)
  ax.set_xticks(np.arange(2, 5, 0.1), minor=True)
  ax.set_xticks(np.arange(2, 5, 0.5), minor=False)
  ax.legend()
  ax.set_xlim(2, 4)
  ax.set_ylim(-40, 0)
  
  fig = plt.figure(figsize=(15, 5))
  # ax = fig.add_subplot(1, 1, 1)
  axs = fig.subplots(1, 3)
  ax = axs[0]
  
  f = nets[0].f/1e9
  ps_00_ref = s_deg_np[:, :, 1 ,0] - s_deg_np[0, :, 1, 0]
  ps_expected = - np.ones_like(ps_00_ref) * (np.arange(16).reshape(-1, 1) * 22.5)
  ps_error = np.sqrt((ps_00_ref - ps_expected)**2)
  ps_error_mean = np.mean(ps_error, axis=0)
  ps_error_std = np.std(ps_error, axis=0)
  ps_error_min = np.min(ps_error, axis=0)
  ps_error_max = np.max(ps_error, axis=0)
  
  # ax.plot(f, ps_00_ref.T, linestyle=line_styles[0], color=line_colors[5], label='Phase Shifter S21 Phase Std Dev from Reference')
  # ax.plot(f, ps_expected.T, linestyle=line_styles[1], color=line_colors[6], label='Expected Phase Shift')
  ax.plot(f, ps_error[0].T, linestyle=line_styles[3], color="black", alpha=0.1, label='Individual Phase Shift Errors')
  ax.plot(f, ps_error[1:].T, linestyle=line_styles[3], color="black", alpha=0.1)
  ax.plot(f, ps_error_mean, linestyle=line_styles[0], color=line_colors[4], label='Mean Phase Shift Error')
  ax.plot(f, ps_error_mean + ps_error_std, linestyle=line_styles[2], color=line_colors[4], label='Mean ± Std Dev')
  ax.plot(f, ps_error_mean - ps_error_std, linestyle=line_styles[2], color=line_colors[4])
  ax.plot(f, ps_error_max, linestyle=line_styles[1], color=line_colors[4], label='Max Phase Shift Error')
    
  ax.set_xlim(2, 4)
  ax.set_ylim(0, 15)
  ax.grid()
  ax.set_xlabel("Frequency (GHz)")
  ax.set_ylabel("Phase Shift Error (Degrees)")
  ax.legend()
  ax.set_title("Absolute RMS Phase Shift Error")
  
  
  bits = [1,2,4,8]
  lsb_shift = 22.5
  ps_error_bits = ps_error[bits, :]
  for i, bit in enumerate(bits):
    ax = axs[1]
    ax.plot(f, ps_error_bits[i], linestyle=line_styles[i], color=line_colors[i], label=f'Bit D{3 + i}: {bits[i] * lsb_shift}° ')
    ax = axs[2]
    ax.plot(f, 100 * ps_error_bits[i] / (bit * lsb_shift), linestyle=line_styles[i], color=line_colors[i], label=f'Bit D{3 + i}: {bits[i] * lsb_shift}° ')
  
  ax = axs[1]
  ax.set_xlim(2, 4)
  ax.set_ylim(0, 15)
  ax.grid()
  ax.set_xlabel("Frequency (GHz)")
  ax.set_ylabel("Phase Shift Error (Degrees)")
  ax.legend()
  ax.set_title("Absolute RMS Phase Shift Error")
  
  ax = axs[2]
  ax.set_xlim(2, 4)
  ax.set_ylim(0, 10)
  ax.grid()
  ax.set_xlabel("Frequency (GHz)")
  ax.set_ylabel("Relative Phase Shift Error (%)")
  ax.legend()
  ax.set_title("Relative RMS Phase Shift Error")
  
  plt.show()
  
if __name__ == "__main__":
  main()