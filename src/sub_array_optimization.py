
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from antenna_array import array_factor, phase_shift
from pattern import gen_rect_pattern,gen_patch_pattern

from plotting import *
from pattern_measurements import *

import pandas as pd

def get_all_phase_shifts(num_elements: int,
                         num_of_bits: int = 4) -> np.ndarray:
  all_combinations = np.arange((2**num_of_bits)**num_elements, dtype=int)
  phase_shifts = np.zeros((len(all_combinations), num_elements), dtype=int)
  for combination_index, combination in enumerate(all_combinations):
    for element_index in range(num_elements):
      phase_shifts[combination_index, element_index] = (combination >> (element_index * num_of_bits)) & (2**num_of_bits - 1)
  return phase_shifts

def sub_array_all_phase_shifts(sub_array_size: int,
                                  sub_array_count: int,
                                  num_of_bits: int = 4) -> np.ndarray:
  sub_array_phase_shifts = get_all_phase_shifts(sub_array_size, num_of_bits)
  return np.concatenate([sub_array_phase_shifts for _ in range(sub_array_count)], axis=1)

def recover_phase_shifts(phase_shift_index: int,
                         num_elements: int,
                         num_of_bits: int = 4) -> np.ndarray:
  phase_shifts = np.zeros((num_elements,), dtype=int)
  for element_index in range(num_elements):
    phase_shifts[element_index] = (phase_shift_index >> (element_index * num_of_bits)) & (2**num_of_bits - 1)
  return phase_shifts

def calculate_element_positions(num_of_sub_arrays, sub_array_size, major_spacing, minor_spacing):
  element_positions = np.zeros(num_of_sub_arrays * sub_array_size)
  for sub_array_index in range(num_of_sub_arrays):
    for element_index in range(sub_array_size):
      overall_index = sub_array_index * sub_array_size + element_index
      element_positions[overall_index] = (sub_array_index * major_spacing) + (element_index * minor_spacing)
  return element_positions

def side_grating_level(pattern: np.ndarray,
                        theta: np.ndarray,
                        scale = "linear"):
  out = np.zeros((pattern.shape[0],))
  for i in range(pattern.shape[0]):
    main_lobe = get_lobe(pattern[i, :], theta)
    if main_lobe.size == 0 or main_lobe.size == pattern.shape[1]:
      out[i] = 1
      continue
    pattern_wo_main_lobe = np.copy(pattern[i, :])
    pattern_wo_main_lobe[main_lobe] = 0
    out[i] = np.max(pattern[i, :]) - np.max(pattern_wo_main_lobe)
  
  return out

def main_lobe_quality(pattern: np.ndarray,
                     theta: np.ndarray,
                     element_pattern: np.ndarray):
  out = np.zeros((pattern.shape[0],))
  for i in range(pattern.shape[0]):
    beam_peak = np.argmax(pattern[i, :])
    out[i] = pattern[i, beam_peak] / element_pattern[beam_peak]
  return out


def generate_patterns(phase_shifts,
                      major_spacing = .5,
                      minor_spacing = .5,
                      number_of_bits = 4,
                      operation_frequency = 2.4e9,
                      spacing_unit = 'wavelength'):
  theta = np.linspace(-np.pi/2, np.pi/2, 360)
    
  num_of_sub_arrays = 2
  sub_array_size = 2
  
  phase_shifts = (phase_shifts / (2**number_of_bits) * 2 * np.pi).astype(np.float64)
  
  if spacing_unit == 'wavelength':
    element_positions = calculate_element_positions(num_of_sub_arrays, sub_array_size, major_spacing, minor_spacing) * (c / operation_frequency)  # convert to meters
  elif spacing_unit == 'meters':
    element_positions = calculate_element_positions(num_of_sub_arrays, sub_array_size, major_spacing, minor_spacing)
  
  af = array_factor(weights=np.ones(num_of_sub_arrays*sub_array_size),
                    num_elements=num_of_sub_arrays*sub_array_size,
                    psi=phase_shift(spacings=element_positions,
                                    frequency=operation_frequency,
                                    theta=theta,
                                    beta=phase_shifts))
  
  # ep = gen_rect_pattern(theta=theta, freq=operation_frequency)
  ep = gen_patch_pattern(theta)
  ap = ep * np.abs(af)
  
  return ap, ep, theta

def pattern_direction_bins(df, bin_count=91, angle_range=(-90, 90)):
  bin_edges = np.linspace(angle_range[0], angle_range[1], bin_count + 1)
  bins = np.zeros((bin_count,), dtype=object)
  for i in range(bin_count):
      bin_lower = bin_edges[i]
      bin_upper = bin_edges[i + 1]
      bin_df = df[(df['Main Lobe Direction (deg)'] >= bin_lower) & (df['Main Lobe Direction (deg)'] < bin_upper)]
      bins[i] = bin_df

  return bins

def generate_dataframe_results(array_pattern, element_pattern, theta, number_of_bits=4):
    main_lobe_dir = main_lobe_direction(array_pattern, theta)
    main_lobe_dir_deg = np.degrees(main_lobe_dir)
    main_lobe_mag = np.max(array_pattern, axis=1)
    main_lobe_mag_norm = main_lobe_mag / np.max(main_lobe_mag)

    sgl_ratio = side_grating_level(array_pattern, theta)
    ml_quality = main_lobe_quality(array_pattern, theta, element_pattern)

    df = pd.DataFrame({
        'Phase Shift Index': np.arange(array_pattern.shape[0]),
        'Phase Shifts': [recover_phase_shifts(idx, number_of_bits) for idx in range(array_pattern.shape[0])],
        'HPBW (deg)': np.degrees(HPBW(array_pattern, theta)),
        'FNBW (deg)': np.degrees(FNBW(array_pattern, theta)),
        'Main Lobe Direction (deg)': main_lobe_dir_deg,
        'Main Lobe Magnitude (norm)': main_lobe_mag_norm,
        'Side Grating Level': sgl_ratio,
        'Main Lobe Quality': ml_quality
    })
    return df

def main():
  
  nb = 2
  
  # ps = sub_array_all_phase_shifts(sub_array_size=2, sub_array_count=2, num_of_bits=4)
  ps = get_all_phase_shifts(num_elements=4, num_of_bits=nb)

  test = calculate_element_positions(num_of_sub_arrays=2, sub_array_size=2, major_spacing=100e-3, minor_spacing=50e-3)
  print(test)

  ap, ep, theta = generate_patterns(phase_shifts=ps, major_spacing=.5, minor_spacing=1, number_of_bits=nb)
  df = generate_dataframe_results(ap, ep, theta)

  max_sweep_angle = 90
  bin_count = 64
  
  bin_width = (2 * max_sweep_angle) / bin_count
  print(f'Bin width: {bin_width} degrees')
  
  plt.hist(df["Main Lobe Direction (deg)"], bins=bin_count)
  plt.show()

  bins = pattern_direction_bins(df, bin_count=bin_count, angle_range=(-max_sweep_angle, max_sweep_angle))

  print(bins.shape)

  for i, bin_df in enumerate(bins):
    pass    



if __name__ == "__main__":
    main()
    pass