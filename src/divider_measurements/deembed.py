import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

def main():
  
  ps_base_path = "src/phase_shifter/"
  divider_base_path = "src/divider_measurements/combined/"
  output_base_path = "src/divider_measurements/deembedded/"
  
  p1 = np.arange(3, -1, -1) * np.arange(5, 0, -1)[:, np.newaxis]
  p2 = np.arange(0, 4) * np.arange(0, 6)[:, np.newaxis]
  progression = np.concatenate((p1, p2))
  progression_names = [f"{prog[0]:02d}{prog[1]:02d}{prog[2]:02d}{prog[3]:02d}" for prog in progression]
  phase_shifter_progression_names = [[f"{prog[0]:02d}", f"{prog[1]:02d}", f"{prog[2]:02d}", f"{prog[3]:02d}"] for prog in progression]

  deembedded_networks = []
  og_networks = []

  for idx, prog_name in enumerate(progression_names):
    phase_shifter_paths = [f"{ps_base_path}{phase_shifter_progression_names[idx][i]}.s2p" for i in range(4)]
    divider_path = f"{divider_base_path}{prog_name}.s5p"
    output_path = f"{output_base_path}{prog_name}.s5p"
    # print(phase_shifter_paths, divider_path, output_path)
    
    div_net_og = rf.Network(divider_path)
    div_net = rf.Network(divider_path)
    ps_nets = [rf.Network(ps_path) for ps_path in phase_shifter_paths]
    
    for port in range(1, 5):
      div_net = rf.connect(div_net, port, ps_nets[port-1].inv, 0)
    
    deembedded_networks.append(div_net)
    og_networks.append(div_net_og)
    div_net.write_touchstone(output_path)
    
  de_s_db_np = np.array([net.s_db for net in deembedded_networks])
  og_s_db_np = np.array([net.s_db for net in og_networks])
  f = deembedded_networks[0].f / 1e9
  
  fig = plt.figure(figsize=(6, 5))
  
  ax = fig.add_subplot(1, 1, 1)
  
  ax.plot(f, de_s_db_np[:, :, 0 , 0].T, color='black', linestyle='-', alpha=0.3)
  ax.plot(f, np.mean(de_s_db_np[:, :, 0 , 0], axis=0), color='red', linestyle='-', label='Mean S11')
  
  ax.plot(f, og_s_db_np[:, :, 0 , 0].T, color='blue', linestyle='--', alpha=0.3)
  ax.plot(f, np.mean(og_s_db_np[:, :, 0 , 0], axis=0), color='cyan', linestyle='--', label='Mean S11 (Original)')
  ax.legend()
  ax.set_xlabel("Frequency (GHz)")
  ax.set_ylabel("S11 (dB)")
  ax.set_title("Deembedded Divider S11")
  plt.show()
  
  fig = plt.figure(figsize=(6, 5))
  
  ax = fig.add_subplot(1, 1, 1)
  
  ax.plot(f, de_s_db_np[:, :, 1 , 0].T, color='black', linestyle='-', alpha=0.3)
  ax.plot(f, np.mean(de_s_db_np[:, :, 1 , 0], axis=0), color='red', linestyle='-', label='Mean S21')
  
  ax.plot(f, og_s_db_np[:, :, 1 , 0].T, color='blue', linestyle='--', alpha=0.3)
  ax.plot(f, np.mean(og_s_db_np[:, :, 1 , 0], axis=0), color='cyan', linestyle='--', label='Mean S21 (Original)')
  ax.legend()
  ax.set_xlabel("Frequency (GHz)")
  ax.set_ylabel("S21 (dB)")
  ax.set_title("Deembedded Divider S21")
  plt.show()
  
  
if __name__ == "__main__":
  main()