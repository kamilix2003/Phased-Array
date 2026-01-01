import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

def main():
    
  line_styles = ['-', '--', ':', '-.']
  line_colors = [f"C{i}" for i in range(20)]
  
  paths = [
    "C:\\Users\\kamaj\\code\\Phased-Array\\src\\awr_output\\correct\\EM_Extract_4_way.s5p",
    "C:\\Users\\kamaj\\code\\Phased-Array\\src\\divider_measurements\\combined\\00000000.s5p", 
    "C:\\Users\\kamaj\\code\\Phased-Array\\src\\awr_output\\correct\\full_00000000.s5p"
  ]
  
  nets = [rf.Network(path) for path in paths]

  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)
  
  ax.plot(nets[0].f / 1e9, nets[0].s_db[:, 0 , 0], linestyle=line_styles[0], color=line_colors[0], label='S11 (4-way EM Extract)')
  ax.plot(nets[1].f / 1e9, nets[1].s_db[:, 0 , 0], linestyle=line_styles[1], color=line_colors[1], label='S11 (4-way Measured)')
  ax.plot(nets[2].f / 1e9, nets[2].s_db[:, 0 , 0], linestyle=line_styles[2], color=line_colors[2], label='S11 (4-way Full Simulated)')
  ax.hlines(-20, xmin=1, xmax=5, colors='black', linestyles=line_styles[2], alpha=0.5)
  
  plt.show()
  
if __name__ == "__main__":
  main()