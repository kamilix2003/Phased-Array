import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

def main():
  
  line_styles = ['-', '--', ':', '-.']
  line_colors = [f"C{i}" for i in range(20)]
  
  src_path = "C:\\Users\\kamaj\\code\\Phased-Array\\src\\awr_output\\correct\\EM_Extract_4_way.s5p"
  dest_path = "C:\\Users\\kamaj\\code\\Phased-Array\\src\\awr_output\\embedded"
  ps_path = "C:\\Users\\kamaj\\code\\Phased-Array\\src\\phase_shifter"
  
  div_ext = ".s5p"
  ps_ext = ".s2p"
  
  
  p1 = np.arange(3, -1, -1) * np.arange(5, 0, -1)[:, np.newaxis]
  p2 = np.arange(0, 4) * np.arange(0, 6)[:, np.newaxis]
  progression = np.concatenate((p1, p2))
  progression_names = [f"{prog[0]:02d}{prog[1]:02d}{prog[2]:02d}{prog[3]:02d}" for prog in progression]
  phase_shifter_progression_names = [[f"{prog[0]:02d}", f"{prog[1]:02d}", f"{prog[2]:02d}", f"{prog[3]:02d}"] for prog in progression]
  
  for i, name in enumerate(progression_names):
    div_net = rf.Network(src_path)
    ps_nets = [rf.Network(f"{ps_path}\\{ps_name}{ps_ext}") for ps_name in phase_shifter_progression_names[i]]
    
    # Embed phase shifters into divider network
    embedded_net = div_net.copy()
    for j, ps_net in enumerate(ps_nets):
      embedded_net = rf.connect(embedded_net, j+1, ps_net, 0)
    
    # Save the embedded network
    # embedded_net.write_touchstone(f"{dest_path}\\full_{name}{div_ext}")
    # embedded_net.plot_s_db(m=0, n=0, label=f'Embedded {name}')
    # div_net.plot_s_db(m=0, n=0, label='Original Divider', linestyle='--')
    # embedded_net.plot_s_deg(m=0, n=slice(1,5), label=f'Embedded {name}')
    # plt.show()
    
    embedded_net.write_touchstone(f"{dest_path}\\{name}{div_ext}")
  
if __name__ == "__main__":
  main()