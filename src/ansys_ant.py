import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

import pandas as pd

def main():
  
  ant_s_path = "C:\\Users\\kamaj\\code\\Phased-Array\\src\\ansys\\antennas_original_design_rogers.s1p"
  ant_s_net = rf.Network(ant_s_path)
  
  f = ant_s_net.f / 1e9
  s_db = ant_s_net.s_db[:, 0, 0]
  
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(f, s_db, linestyle='-', color='C0')
  ax.set_xlim(2, 4)
  ax.set_ylim(-30, 0)
  ax.grid()
  ax.set_xlabel("Frequency (GHz)")
  ax.set_ylabel("S11 (dB)")
  ax.set_title("HFSS Simulation Results")
  
  # plt.show()
  
  gain_data_path = "C:\\Users\\kamaj\\code\\Phased-Array\\src\\ansys\\2d_gain.csv"
  gain_df = pd.read_csv(gain_data_path)
  labels = gain_df.columns.tolist()
  print(labels)
  
  gain_df_24_0 = gain_df[(gain_df[labels[0]] == 2.4) & (gain_df[labels[1]] == 0)].to_numpy()
  gain_df_24_90 = gain_df[(gain_df[labels[0]] == 2.4) & (gain_df[labels[1]] == 90)].to_numpy()
  gain_df_3_0 = gain_df[(gain_df[labels[0]] == 3.0) & (gain_df[labels[1]] == 0)].to_numpy()
  gain_df_3_90 = gain_df[(gain_df[labels[0]] == 3.0) & (gain_df[labels[1]] == 90)].to_numpy()
  
  fig = plt.figure(figsize=(10, 5))
  ax1 = fig.add_subplot(1, 2, 1)
  # ax1.set_theta_zero_location("N")
  ax2 = fig.add_subplot(1, 2, 2)
  # ax2.set_theta_zero_location("N")
  
  ax1.plot(gain_df_24_0[:, 2], gain_df_24_0[:, 3], linestyle='-', color='C0', label='XZ Plane')
  # ax1.plot(gain_df_24_90[:, 2], gain_df_24_90[:, 3], linestyle='--', color='C1', label='YZ Plane')
  ax1.set_title("Antenna Gain at 2.4 GHz")
  ax1.set_xlabel("Angle (degrees)")
  ax1.set_ylabel("Gain (dBi)")
  ax1.legend()
  ax1.grid()
  
  ax2.plot(gain_df_3_0[:, 2], gain_df_3_0[:, 3], linestyle='-', color='C0', label='XZ Plane')
  # ax2.plot(gain_df_3_90[:, 2], gain_df_3_90[:, 3], linestyle='--', color='C1', label='YZ Plane')
  ax2.set_title("Antenna Gain at 3.0 GHz")
  ax2.set_xlabel("Angle (degrees)")
  ax2.set_ylabel("Gain (dBi)")
  ax2.legend()
  ax2.grid()
  
  plt.show()
  
if __name__ == "__main__":
  main()