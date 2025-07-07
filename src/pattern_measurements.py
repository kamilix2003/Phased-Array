import numpy as np
from scipy import signal
from scipy.constants import c
from utils import linear_to_db

def find_maximas(pattern : np.ndarray[float], theta: np.ndarray[float]):
    peaks, _ = signal.find_peaks(pattern, width=5, rel_height=5)
    return peaks

def find_nulls(pattern : np.ndarray[float], theta: np.ndarray[float]):
    nulls, _ = signal.find_peaks(-pattern)
    return nulls

def get_main_lobe(pattern, theta, nth):
    peaks_idx = find_maximas(pattern, theta)
    nulls_idx = find_nulls(pattern, theta)
    
    main_peak_idx = peaks_idx[np.argmax(pattern[peaks_idx])]
    print(peaks_idx, nulls_idx, main_peak_idx)
    left_null_idx = nulls_idx[main_peak_idx > nulls_idx][-1]
    right_null_idx = nulls_idx[main_peak_idx < nulls_idx][0]
    
    return np.arange(left_null_idx, right_null_idx)

def get_main_lobe(pattern, theta, nth = 0):
    peaks_idx = find_maximas(pattern, theta)
    nulls_idx = find_nulls(pattern, theta)
    main_peak_idx = np.argmax(pattern[peaks_idx])
    print(main_peak_idx, nth, peaks_idx.size)
    if main_peak_idx + nth < peaks_idx.size:
        get_peak_idx = peaks_idx[main_peak_idx + nth]
    else:
        return 0
    print(peaks_idx, nulls_idx, get_peak_idx)
    if nulls_idx[get_peak_idx > nulls_idx].size != 0:
        left_null_idx = nulls_idx[get_peak_idx > nulls_idx][-1]
    else:
        left_null_idx = 0
    if nulls_idx[get_peak_idx < nulls_idx].size != 0:
        right_null_idx = nulls_idx[get_peak_idx < nulls_idx][0]
    else:
        right_null_idx = pattern.size
    
    print(left_null_idx, right_null_idx)
    return np.arange(left_null_idx, right_null_idx)
     

def main():
    import matplotlib.pyplot as plt
    from utils import linear_to_db
    from spacing import gen_spacing
    from beam_steering import steer_to_phase    
    from antenna_array import phase_shift, array_factor
    from pattern import gen_rect_patter
    theta = np.linspace(-np.pi/2, np.pi/2, 360)
    frequency = 2.4e9  # 1 GHz
    # pattern_antenna = np.cos(theta) ** 2 * np.cos(theta / 2) ** 4
    num_element = 7

    ant_pat = gen_rect_patter(theta, frequency)
    sps = gen_spacing(num_element, np.array([0.4, 0.4, 0.4]) * (c / frequency))
    sa = np.linspace(-np.pi/4, np.pi/4, 9)
    b = steer_to_phase(num_element, sps, sa, frequency)
    ps = phase_shift(sps, frequency, theta, b)
    af = array_factor(np.ones(num_element), num_element, ps)
    ap = ant_pat * np.abs(af[4, :])
    ap_db = linear_to_db(ap)
    
    fig = plt.figure()
    # plt.plot(theta, ap_db, "--k")
    peak_idx = find_maximas(ap_db, theta)
    plt.plot(theta[peak_idx], ap_db[peak_idx], "x")
    null_idx = find_nulls(ap_db, theta)
    plt.plot(theta[null_idx], ap_db[null_idx], "x")
    
    main_idx = get_main_lobe(ap_db, theta, 0)
    plt.plot(theta[main_idx], ap_db[main_idx], "r")
    
    for i in np.arange(-2, 3):
        print(i)
        main_idx = get_main_lobe(ap_db, theta, i)
        plt.plot(theta[main_idx], ap_db[main_idx], ls=":")
    
    plt.ylim((-60, 3))
    plt.show()

if __name__ == "__main__":
    main()
    pass