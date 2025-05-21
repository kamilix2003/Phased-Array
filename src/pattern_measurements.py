import numpy as np
from scipy.signal import argrelmax, argrelmin

def find_nulls(pattern : np.ndarray[float]) -> np.ndarray[float]:
        
    null_indices = argrelmin(pattern, order=1)[0]
    return null_indices

def find_maximas(pattern : np.ndarray[float]) -> np.ndarray[float]:
    
    maxima_indices = argrelmax(pattern, order=1)[0]
    return maxima_indices

def _find_left_right_nulls(nulls : np.ndarray[float], main_lobe: int) -> tuple[int, int]:
    left_null = (nulls[nulls < main_lobe])[-1] if np.any(nulls < main_lobe) else None
    right_null = (nulls[nulls > main_lobe])[0] if np.any(nulls > main_lobe) else None
    
    return left_null, right_null

def FNBW(pattern : np.ndarray[float], theta: np.ndarray[float]) -> tuple[float, int, int]:
    arg_main_lobe = np.argmax(pattern)
    arg_nulls = find_nulls(pattern)
    arg_first_null_left ,arg_first_null_right = _find_left_right_nulls(arg_nulls, arg_main_lobe)
    
    return np.abs(theta[arg_first_null_left] - theta[arg_first_null_right]), arg_first_null_left, arg_first_null_right

def HPBW(pattern : np.ndarray[float], theta: np.ndarray[float], beta: float) -> tuple[float, int, int]:
    HP_threshold = 0.71 # -3dB in linear scale
    arg_HP_left = np.argmin(np.abs(pattern - HP_threshold))
    arg_HP_right = np.argmin(np.abs(pattern[::-1] - HP_threshold))
    print(arg_HP_left, arg_HP_right)
    return np.abs(theta[arg_HP_left] - theta[arg_HP_right]), arg_HP_left, arg_HP_right

def FSLBW(pattern : np.ndarray[float], theta: np.ndarray[float], beta: float) -> tuple[float, int, int]:
    arg_maximas = find_maximas(pattern)
    arg_main_lobe = np.argmax(pattern)
    arg_first_max_left ,arg_first_max_right = _find_left_right_nulls(arg_maximas, arg_main_lobe)
    
    return np.abs(theta[arg_first_max_left] - theta[arg_first_max_right]), arg_first_max_left, arg_first_max_right
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    theta = np.linspace(-np.pi, np.pi, 360)
    pattern = np.abs(np.sinc(theta*2))  # Example pattern
    
    # TODO: exceptions for patterns with no nulls
    # pattern = np.abs(np.cos(theta/2)**2)  # Example pattern
    
    fig = plt.figure(figsize=(10, 6))
    ax = plt.polar()
    plt.plot(theta, pattern, label='Radiation Pattern')
    
    fnbw, x1, x2 = FNBW(pattern, theta)
    fnbw_plot = plt.axvline(theta[x1], color='r', linestyle='--')
    plt.axvline(theta[x2], color='r', linestyle='--')
    
    hpbw, x1, x2 = HPBW(pattern, theta, 0)
    hpbw_plot = plt.axvline(theta[x1], color='g', linestyle='--')
    plt.axvline(theta[x2], color='g', linestyle='--')

    fslbw, x1, x2 = FSLBW(pattern, theta, 0)
    fsbw_plot = plt.axvline(theta[x1], color='b', linestyle='--')
    plt.axvline(theta[x2], color='b', linestyle='--')
    
    plt.title('Antenna Radiation Pattern with FNBW, HPBW, and FSLBW')
    plt.legend([fnbw_plot, hpbw_plot, fsbw_plot], [f'FNBW = {np.rad2deg(fnbw):.2f}', f'HPBW = {np.rad2deg(hpbw):.2f}', f'FSLBW = {np.rad2deg(fslbw):.2f}'])
        
    plt.show()
    
    
    pass