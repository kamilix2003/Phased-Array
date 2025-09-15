import numpy as np
from scipy.signal import find_peaks
from scipy.constants import c

def find_maximas(pattern : np.ndarray[float], theta: np.ndarray[float]):
    peaks, _ = find_peaks(pattern, width=1, rel_height=1)
    return peaks

def find_nulls(pattern : np.ndarray[float], theta: np.ndarray[float]):
    nulls, _ = find_peaks(-pattern)
    return nulls

def get_main_lobe(pattern, theta, nth):
    peaks_idx = find_maximas(pattern, theta)
    nulls_idx = find_nulls(pattern, theta)
    
    main_peak_idx = peaks_idx[np.argmax(pattern[peaks_idx])]
    print(peaks_idx, nulls_idx, main_peak_idx)
    left_null_idx = nulls_idx[main_peak_idx > nulls_idx][-1]
    right_null_idx = nulls_idx[main_peak_idx < nulls_idx][0]
    
    return np.arange(left_null_idx, right_null_idx+1)

def get_lobe(pattern, theta, nth = 0):
    peaks_idx = find_maximas(pattern, theta)
    nulls_idx = find_nulls(pattern, theta)
    main_peak_idx = np.argmax(pattern[peaks_idx])
    # print(main_peak_idx, nth, peaks_idx.size)
    if main_peak_idx + nth < peaks_idx.size:
        get_peak_idx = peaks_idx[main_peak_idx + nth]
    else:
        return 0
    # print(peaks_idx, nulls_idx, get_peak_idx)
    if nulls_idx[get_peak_idx > nulls_idx].size != 0:
        left_null_idx = nulls_idx[get_peak_idx > nulls_idx][-1]
    else:
        left_null_idx = 0
    if nulls_idx[get_peak_idx < nulls_idx].size != 0:
        right_null_idx = nulls_idx[get_peak_idx < nulls_idx][0]
    else:
        right_null_idx = pattern.size
    
    # print(left_null_idx, right_null_idx)
    if right_null_idx != pattern.size: right_null_idx+=1
    return np.arange(left_null_idx, right_null_idx)
     
def FNBW(pattern, theta, nth=0):
    main_lobe = get_lobe(pattern, theta, nth)
    if main_lobe.size == 0:
        return 0
    main_lobe_width = np.abs(theta[main_lobe[-1]] - theta[main_lobe[0]])
    return main_lobe_width

def HPBW(pattern, theta, nth=0):
    main_lobe = get_lobe(pattern, theta, nth)
    if main_lobe.size == 0:
        return 0
    if np.any(pattern[main_lobe] < 0):
        threshold = np.max(pattern[main_lobe]) - 3
    else:
        threshold = np.max(pattern[main_lobe]) / 2
    hp_lobe = np.where(pattern[main_lobe] >= threshold)[0]
    if hp_lobe.size == 0:
        return 0
    hp_lobe_width = np.abs(theta[hp_lobe[-1]] - theta[hp_lobe[0]])
    return hp_lobe_width

def HPBW_bounds(pattern, theta, idx=False):
    main_lobe = get_lobe(pattern, theta)
    if main_lobe.size == 0:
        return 0, 0
    if np.any(pattern[main_lobe] < 0):
        threshold = np.max(pattern[main_lobe]) - 3
    else:
        threshold = np.max(pattern[main_lobe]) * .5
    hp_lobe = np.where(pattern[main_lobe] >= threshold)[0]
    if hp_lobe.size == 0:
        return 0, 0
    if idx:
        return main_lobe[hp_lobe[0]], main_lobe[hp_lobe[-1]]
    return theta[main_lobe[hp_lobe[0]]], theta[main_lobe[hp_lobe[-1]]]

def main_lobe_direction(pattern, theta):
    main_lobe_idx = np.argmax(pattern)
    return theta[main_lobe_idx]

def SLL(pattern, theta):
    
    main_lobe = get_lobe(pattern, theta)
    if main_lobe.size == 0:
        return 0
    side_lobes = np.setdiff1d(np.arange(pattern.size), main_lobe)
    if side_lobes.size == 0:
        return 0
    sll = np.max(pattern[side_lobes])
    if np.any(pattern < 0):
        sll_db = sll - np.max(pattern[main_lobe])
    else:
        sll_db = 10 * np.log10(sll / np.max(pattern[main_lobe]))
    return sll_db

def main():
    
    from utils import get_pattern, linear_to_db
    import matplotlib.pyplot as plt
    
    theta = np.linspace(-np.pi/2, np.pi/2, 360)
    frequency = 2.4e9  # 1 GHz
    num_element = 4
    n = 2
    ap = get_pattern(theta, frequency, num_element, 0.75, np.radians(np.linspace(0, 30, n)))[0, :]
    ap_db = linear_to_db(ap)
    
    print(f'HPBW: {np.degrees(HPBW(ap_db, theta)):.2f}')
    print(f'HPBW bounds: {np.degrees(HPBW_bounds(ap_db, theta))}')
    print(f'FNBW: {np.degrees(FNBW(ap_db, theta)):.2f}')
    print(f'Main lobe direction: {np.degrees(main_lobe_direction(ap_db, theta)):.2f} degrees')
        
    plt.plot(np.degrees(theta), ap_db, "--k")
    peak_idx = find_maximas(ap_db, theta)
    plt.plot(np.degrees(theta[peak_idx]), ap_db[peak_idx], "x")
    null_idx = find_nulls(ap_db, theta)
    plt.plot(np.degrees(theta[null_idx]), ap_db[null_idx], "x")
    
    main_idx = get_main_lobe(ap_db, theta, 0)
    plt.plot(np.degrees(theta[main_idx]), ap_db[main_idx], "r")
    
    bw = HPBW_bounds(ap_db, theta, idx=True)
    plt.fill_between(np.degrees(theta[bw[0]:bw[1]]), ap_db[bw[0]:bw[1]], -60, color='red', alpha=0.2, label='HPBW')
    
    plt.grid()
    plt.ylim((-60, 3))
    plt.show()

if __name__ == "__main__":
    main()
    pass