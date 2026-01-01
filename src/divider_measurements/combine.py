import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

path = "src/divider_measurements/"
src_dir = "full_unprocessed/"
src_ext = ".s4p"
dst_dir = "combined/"
dst_ext = ".s5p"
p1 = np.arange(3, -1, -1) * np.arange(5, 0, -1)[:, np.newaxis]
p2 = np.arange(0, 4) * np.arange(0, 6)[:, np.newaxis]
progression = np.concatenate((p1, p2))
progression_names = [f"{prog[0]:02d}{prog[1]:02d}{prog[2]:02d}{prog[3]:02d}" for prog in progression]

for i, progression_name in enumerate(progression_names):

  f1 = path + src_dir + f"L_{progression_name}{src_ext}"
  f2 = path + src_dir + f"R_{progression_name}{src_ext}"

  # 1. Load your 4-port touchstone files
  nw_a = rf.Network(f1) # Ports: In, Out1, Out2, Out3
  nw_b = rf.Network(f2) # Ports: In, Out2, Out3, Out4
    
  # 2. Create a blank 5-port Network with the same frequency range
  freq = nw_a.frequency
  full_5port = rf.Network()
  full_5port.frequency = freq
  full_5port.s = np.zeros((len(freq), 5, 5), dtype=complex)
  full_5port.z0 = 50

  # --- DATA FROM MEASUREMENT A ---
  # Maps Global 0,1,2,3 directly from VNA 0,1,2,3
  full_5port.s[:, 0:4, 0:4] = nw_a.s

  # --- DATA FROM MEASUREMENT B ---
  # We specifically need the data involving Global Port 4 (VNA Port 2 in Meas B)
  # VNA Indices in Meas B: P1=0, P2=1 (Out4), P3=2, P4=3

  # Reflection and Input Transmission
  full_5port.s[:, 4, 4] = nw_b.s[:, 1, 1] # S44
  full_5port.s[:, 4, 0] = nw_b.s[:, 1, 0] # S4-In
  full_5port.s[:, 0, 4] = nw_b.s[:, 0, 1] # SIn-4

  # Isolation with Output 2 and Output 3
  full_5port.s[:, 4, 2] = nw_b.s[:, 1, 2] # S4-2
  full_5port.s[:, 2, 4] = nw_b.s[:, 2, 1] # S2-4
  full_5port.s[:, 4, 3] = nw_b.s[:, 1, 3] # S4-3
  full_5port.s[:, 3, 4] = nw_b.s[:, 3, 1] # S3-4

  # --- THE MISSING DATA: S41 / S14 ---
  # Since Out 1 and Out 4 were never on the VNA at the same time, 
  # you can approximate this by copying S32 or S21 if the device is symmetric.
  full_5port.s[:, 4, 1] = full_5port.s[:, 3, 2] 
  full_5port.s[:, 1, 4] = full_5port.s[:, 2, 3]

  # 4. Save the result
  full_5port.write_touchstone(path + dst_dir + f"{progression_name}{dst_ext}")
