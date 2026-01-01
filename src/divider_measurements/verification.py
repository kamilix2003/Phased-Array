import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

def verify_combined_network(combined_file, meas_a_file, meas_b_file):
    # Load the combined 5-port network
    combined_nw = rf.Network(combined_file)
    
    # Load the original 4-port measurements
    meas_a_nw = rf.Network(meas_a_file)
    meas_b_nw = rf.Network(meas_b_file)
    
    freq = combined_nw.frequency

    # Verify Measurement A data
    s_meas_a = combined_nw.s[:, 0:4, 0:4]
    s_diff_a = np.abs(s_meas_a - meas_a_nw.s)
    
    # Verify Measurement B data
    s_meas_b = np.zeros_like(meas_b_nw.s)
    s_meas_b[:, 0, 0] = combined_nw.s[:, 0, 0]  # In-In
    s_meas_b[:, 1, 1] = combined_nw.s[:, 4, 4]  # Out4-Out4
    s_meas_b[:, 0, 1] = combined_nw.s[:, 0, 4]  # In-Out4
    s_meas_b[:, 1, 0] = combined_nw.s[:, 4, 0]  # Out4-In
    s_meas_b[:, 1, 2] = combined_nw.s[:, 4, 2]  # Out4-Out2
    s_meas_b[:, 2, 1] = combined_nw.s[:, 2, 4]  # Out2-Out4
    s_meas_b[:, 1, 3] = combined_nw.s[:, 4, 3]  # Out4-Out3
    s_meas_b[:, 3, 1] = combined_nw.s[:, 3, 4]  # Out3-Out4
    
    s_diff_b = np.abs(s_meas_b - meas_b_nw.s)
    
    # Plot differences for Measurement A
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Difference in Measurement A")
    plt.imshow(np.max(s_diff_a, axis=0), cmap='hot', interpolation='nearest')
    plt.colorbar(label='Max |S_combined - S_meas_a|')
    plt.xlabel('Port Index')
    plt.ylabel('Port Index')
    
    # Plot differences for Measurement B
    plt.subplot(1,2,2)
    plt.title("Difference in Measurement B")
    plt.imshow(np.max(s_diff_b, axis=0), cmap='hot', interpolation='nearest')
    plt.colorbar(label='Max |S_combined - S_meas_b|')
    plt.xlabel('Port Index')
    plt.ylabel('Port Index')
    plt.tight_layout()
    # plt.show()
    
if __name__ == "__main__":
    path = "src/divider_measurements/"
    prog_name = "00020406"
    combined_file = path + f"combined/{prog_name}.s5p"
    meas_a_file = path + f"full_unprocessed/L_{prog_name}.s4p"
    meas_b_file = path + f"full_unprocessed/R_{prog_name}.s4p"
    
    # verify_combined_network(combined_file, meas_a_file, meas_b_file)
    
    path = "src/divider_measurements/"
    dst_dir = "combined/"
    # dst_dir = "full_unprocessed/"
    dst_ext = ".s5p"
    # dst_ext = ".s4p"
    p1 = np.arange(3, -1, -1) * np.arange(5, 0, -1)[:, np.newaxis]
    p2 = np.arange(0, 4) * np.arange(0, 6)[:, np.newaxis]
    progression = np.concatenate((p1, p2))
    progression_names = [f"{prog[0]:02d}{prog[1]:02d}{prog[2]:02d}{prog[3]:02d}" for prog in progression]
    # progression_names = [f"L_{prog[0]:02d}{prog[1]:02d}{prog[2]:02d}{prog[3]:02d}" for prog in progression]
    
    for i, progression_name in enumerate(progression_names):

        f1 = path + dst_dir + f"{progression_name}{dst_ext}"
        nw = rf.Network(f1)
        
        s_deg = nw.s_deg
        
        # for port in range(1, 4):
        #     if np.any(s_deg[:, 0, port] > 0):
        #         print("aaa")
        #         s_deg[:, 0, port] -= 360
        
        phase_progression = []
        for port in range(1, 5):
            phase_progression.append(np.abs(s_deg[:, 0, port] - s_deg[:, 0, 1]))
            plt.plot(nw.frequency.f/1e9, s_deg[:, 0, port], label=f'Port {port}')
        
        phase_progression = np.array(phase_progression)
        plt.legend()
        plt.show()
        # for port in range(1, 4):
        #     ps_diff = np.abs(np.mean(phase_progression[port-1, :]) - np.mean(phase_progression[port, :]))
        #     expected_diff = np.abs(22.5 * (progression[i][port-1] - progression[i][port]))
        #     if np.abs(ps_diff - expected_diff) > 10:
        #         phase_progression[port, :] = 360 - phase_progression[port, :]
        print(f'Progression: {progression_name}, Phase Shifter Angles: {np.mean(phase_progression, axis=1)}, Phase progression: {np.diff(np.mean(phase_progression, axis=1))}')