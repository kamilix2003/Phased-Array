import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

def main():
  
  fs_label = 14
  fs_title = 16
    
  line_styles = ['-', '--', ':', '-.']
  line_colors = [f"C{i}" for i in range(20)]
  
  base_path = "src/awr_output/correct/"
  extension = ".s3p"
  names = [
    "EM_Extract_2_way",
    "Physical_2_way",
    "Symbolic_2_way",
    "symbolic_2_way_1_section",
    "symbolic_2_way_bi",
  ]
  
  nets = [rf.Network(base_path + name + extension) for name in names]
  
  s_db_np = np.array([net.s_db for net in nets])
  s_deg_np = np.array([net.s_deg for net in nets])
  f = nets[0].f / 1e9
  
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)

  ax.plot(f, s_db_np[4, :, 0 , 0], linestyle=line_styles[0], color=line_colors[0], label='S11 (2 section)')
  ax.plot(f, s_db_np[3, :, 0 , 0], linestyle=line_styles[1], color=line_colors[1], label='S11 (1 Section)')
  ax.hlines(-20, xmin=1, xmax=5, colors='black', linestyles=line_styles[2], alpha=0.5)
  
  ax.set_xlim(1, 5)
  ax.set_ylim(-40, 0)
  ax.legend()
  ax.grid()
  ax.set_xlabel("Frequency (GHz)", fontsize=fs_label)
  ax.set_ylabel("Magnitude (dB)", fontsize=fs_label)
  ax.set_title("2-way Divider Simulation Results Comparison", fontsize=fs_title)
  
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)
  
  ax.plot(f, s_db_np[2, :, 0, 0], linestyle=line_styles[0], color=line_colors[0], label='S11')
  ax.plot(f, s_db_np[2, :, 0, 1], linestyle=line_styles[1], color=line_colors[1], label='S21')
  ax.plot(f, s_db_np[2, :, 0, 2], linestyle=line_styles[2], color=line_colors[2], label='S31')
  
  ax.set_xlim(1, 5)
  ax.set_xticks(np.arange(1, 5.01, 0.5))
  ax.set_xticks(np.arange(1, 5.01, 0.1), minor=True)
  ax.set_ylim(-40, 0)
  ax.legend()
  ax.grid()
  ax.set_xlabel("Frequency (GHz)", fontsize=fs_label)
  ax.set_ylabel("Magnitude (dB)", fontsize=fs_label)
  ax.set_title("2-way Divider Symbolic Simulation Results", fontsize=fs_title)
    
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)

  ax.plot(f, s_db_np[0, :, 0, 0], linestyle=line_styles[0], color=line_colors[0], label='EM simulation')
  ax.plot(f, s_db_np[1, :, 0, 0], linestyle=line_styles[1], color=line_colors[1], label='Physical simulation')
  ax.plot(f, s_db_np[2, :, 0, 0], linestyle=line_styles[2], color=line_colors[2], label='Symbolic simulation')
    
  ax.set_xlim(1, 5)
  ax.set_xticks(np.arange(1, 5.01, 0.5))
  ax.set_xticks(np.arange(1, 5.01, 0.1), minor=True)
  ax.set_ylim(-40, 0)
  ax.legend()
  ax.grid()
  ax.set_xlabel("Frequency (GHz)", fontsize=fs_label)
  ax.set_ylabel("Magnitude (dB)", fontsize=fs_label)
  ax.set_title("2-way Divider S11", fontsize=fs_title)
    
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)

  ax.plot(f, s_db_np[0, :, 0, 1], linestyle=line_styles[0], color=line_colors[0], label='EM simulation')
  ax.plot(f, s_db_np[1, :, 0, 1], linestyle=line_styles[1], color=line_colors[1], label='Physical simulation')
  ax.plot(f, s_db_np[2, :, 0, 1], linestyle=line_styles[2], color=line_colors[2], label='Symbolic simulation')
    
  ax.set_xlim(1, 5)
  ax.set_xticks(np.arange(1, 5.01, 0.5))
  ax.set_xticks(np.arange(1, 5.01, 0.1), minor=True)
  ax.set_ylim(-3.5, -2.9)
  ax.legend()
  ax.grid()
  ax.set_xlabel("Frequency (GHz)", fontsize=fs_label)
  ax.set_ylabel("Magnitude (dB)", fontsize=fs_label)
  ax.set_title("2-way Divider S21", fontsize=fs_title)
      
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)

  ax.plot(f, s_db_np[0, :, 1, 2], linestyle=line_styles[0], color=line_colors[0], label='EM simulation')
  ax.plot(f, s_db_np[1, :, 1, 2], linestyle=line_styles[1], color=line_colors[1], label='Physical simulation')
  ax.plot(f, s_db_np[2, :, 1, 2], linestyle=line_styles[2], color=line_colors[2], label='Symbolic simulation')
    
  ax.set_xlim(1, 5)
  ax.set_xticks(np.arange(1, 5.01, 0.5))
  ax.set_xticks(np.arange(1, 5.01, 0.1), minor=True)
  ax.set_ylim(-40, 0)
  ax.legend()
  ax.grid()
  ax.set_xlabel("Frequency (GHz)", fontsize=fs_label)
  ax.set_ylabel("Magnitude (dB)", fontsize=fs_label)
  ax.set_title("2-way Divider S32", fontsize=fs_title)
    
  plt.show()
  
  
  
if __name__ == "__main__":
  main()