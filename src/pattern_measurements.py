import numpy as np
from scipy import signal

from utils import linear_to_db

def find_maximas(pattern : np.ndarray[float], theta: np.ndarray[float]):
    peaks, _ = signal.find_peaks(pattern)
    return pattern[peaks], theta[peaks]

def find_nulls(pattern : np.ndarray[float], theta: np.ndarray[float]):
    nulls, _ = signal.find_peaks(-pattern)
    return pattern[nulls], theta[nulls]

def find_main_peak(pattern : np.ndarray[float], theta: np.ndarray[float]):
    peaks, _ = signal.find_peaks(pattern)
    main_lobe = peaks[np.argmax(pattern[peaks])]
    return pattern[main_lobe], theta[main_lobe]

def find_nth_peak(pattern: np.ndarray, theta: np.ndarray, n: int):
    peaks, _ = signal.find_peaks(pattern)
    if len(peaks) == 0:
        return None, None
    main_lobe_idx = peaks[np.argmax(pattern[peaks])]
    sorted_peaks = np.sort(theta[peaks])
    main_lobe_theta = theta[main_lobe_idx]
    # Find index of main lobe in sorted peaks
    main_lobe_sorted_idx = np.where(sorted_peaks == main_lobe_theta)[0][0]
    # n > 0: n-th peak after main lobe, n < 0: n-th peak before main lobe
    target_idx = main_lobe_sorted_idx + n
    if target_idx < 0 or target_idx >= len(sorted_peaks):
        return None, None
    target_theta = sorted_peaks[target_idx]
    # Find the corresponding pattern value
    target_pattern = pattern[np.where(np.isclose(theta, target_theta))[0][0]]
    return target_pattern, target_theta

def get_lobe(pattern: np.ndarray, theta: np.ndarray, n: int):
    # Find main lobe peak and its angle
    # main_lobe_peak, main_lobe_theta = find_main_peak(pattern, theta)
    lobe_peaks, lobe_theta = find_nth_peak(pattern, theta, n)
    # Find nulls and their angles
    _, nulls_theta = find_nulls(pattern, theta)
    # Sort nulls by angle
    sorted_nulls = np.sort(nulls_theta)
    # Find the nulls just before and after the main lobe
    before = sorted_nulls[sorted_nulls < lobe_theta]
    after = sorted_nulls[sorted_nulls > lobe_theta]
    if len(before) == 0 or len(after) == 0:
        return None, None
    lower_null = before[-1]
    upper_null = after[0]
    # Get indices within the main lobe region
    mask = (theta >= lower_null) & (theta <= upper_null)
    return pattern[mask], theta[mask]

def FNBW(pattern : np.ndarray[float], theta: np.ndarray[float]) -> float:
    _, main_lobe_theta = get_lobe(pattern, theta, 0)
    return np.abs(main_lobe_theta[0] - main_lobe_theta[-1])

def HPBW(pattern : np.ndarray[float], theta: np.ndarray[float]) -> float:
    main_lobe, main_lobe_theta = get_lobe(pattern, theta, 0)

    thershold = main_lobe.max() / np.sqrt(2)
    
    left_idx = np.where(main_lobe >= thershold)[0][0]
    right_idx = np.where(main_lobe >= thershold)[0][-1]
    
    left_theta = main_lobe_theta[left_idx]
    right_theta = main_lobe_theta[right_idx]
    
    return np.abs(left_theta - right_theta)

def FSLBW(pattern : np.ndarray[float], theta: np.ndarray[float]) -> float:
    left_peak, left_theta = find_nth_peak(pattern, theta, -1)
    right_peak, right_theta = find_nth_peak(pattern, theta, 1)
    if left_peak is None or right_peak is None:
        return None
    return np.abs(left_theta - right_theta)

def FSL_peak(pattern: np.ndarray[float], theta: np.ndarray[float]) -> float:
    left_peak, left_theta = find_nth_peak(pattern, theta, -1)
    right_peak, right_theta = find_nth_peak(pattern, theta, 1)
    if left_peak is None or right_peak is None:
        return None
    return np.abs(left_peak + right_peak) / 2

def get_envelope(pattern: np.ndarray[float], theta: np.ndarray[float]) -> np.ndarray[float]:
    peaks, theta_peaks = find_maximas(pattern, theta)
    from scipy import interpolate
    
    # envelope = interpolate.CubicSpline(theta_peaks, peaks, bc_type='natural')(theta)
    # envelope = np.interp(theta, theta_peaks, peaks)
    envelope = interpolate.make_interp_spline(theta_peaks, peaks, k=1)(theta)
    return envelope

def pattern_measurement(method, patterns, theta):
    measurement = np.zeros_like(patterns[:, 0], dtype=float)
    for i in range(patterns.shape[0]):
        measurement[i] = method(patterns[i], theta)
    return measurement

def plot_pattern(pattern, theta, ax=None, label=None):
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
    ax.plot(theta, linear_to_db(pattern), label=label)
    ax.grid(True)
    ax.axvline(get_lobe(pattern, theta, 0)[1][0], color='red', linestyle='--', label='Main Lobe')
    ax.axvline(get_lobe(pattern, theta, 0)[1][-1], color='red', linestyle='--', label='Main Lobe')
    ax.title.set_text(f'FNBW: {FNBW(pattern, theta):.2f} rad, HPBW: {HPBW(pattern, theta):.2f} rad')    
    return ax

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import antenna_array as aa
    from utils import linear_to_db, config_plot
    
    results, theta, test_stearing_angles, test_spacings, test_pattern = aa.get_test_data()
    
    spcaing_idx = 1
    angle_idx = len(test_stearing_angles+10) // 2
    
    pattern = np.abs(results[spcaing_idx, angle_idx, :])
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1, projection='rectilinear')
    # ax.plot(theta, linear_to_db(pattern), label='Pattern')
    config_plot(ax)
       
    nlobe, ntheta = get_lobe(pattern, theta, 0)
    ax.plot(ntheta, linear_to_db(nlobe), color='red', label='Main Lobe')
    nlobe, ntheta = get_lobe(pattern, theta, 1)
    ax.plot(ntheta, linear_to_db(nlobe), color='green', label='Right Lobe')
    nlobe, ntheta = get_lobe(pattern, theta, -1)
    ax.plot(ntheta, linear_to_db(nlobe), color='blue', label='Left Lobe')
    env = get_envelope(pattern, theta)
    ax.plot(theta, linear_to_db(pattern), color='black', label='pattern')
    ax.plot(theta, linear_to_db(env), color='orange', label='Envelope', linestyle='--')
    
    plt.legend()
    plt.show()

        
    patterns = results[2, :, :]
    pattern_HPBW = pattern_measurement(HPBW, patterns, theta)
    pattern_FNBW = pattern_measurement(FNBW, patterns, theta)
    pattern_FSLBW = pattern_measurement(FSLBW, patterns, theta)
    pattern_FSL_peak = pattern_measurement(FSL_peak, patterns, theta)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(2, 1, 1)
    ax.plot(np.degrees(test_stearing_angles), np.degrees(pattern_HPBW), label='HPBW')
    ax.plot(np.degrees(test_stearing_angles), np.degrees(pattern_FNBW), label='FNBW')
    ax.plot(np.degrees(test_stearing_angles), np.degrees(pattern_FSLBW), label='FSLBW')
    ax.legend()
    ax = plt.subplot(2, 1, 2)
    ax.plot(np.degrees(test_stearing_angles), pattern_FSL_peak, label='FSL Peak')
    ax.legend()
    plt.show()
    
    