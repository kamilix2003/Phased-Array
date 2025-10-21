
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from antenna_array import array_factor, phase_shift
from pattern import gen_rect_pattern

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

def calculate_element_positions(num_of_sub_arrays, sub_array_size, major_spacing, minor_spacing):
  element_positions = np.zeros(num_of_sub_arrays * sub_array_size)
  for sub_array_index in range(num_of_sub_arrays):
    for element_index in range(sub_array_size):
      overall_index = sub_array_index * sub_array_size + element_index
      element_positions[overall_index] = (sub_array_index * major_spacing) + (element_index * minor_spacing)
  return element_positions

def main():
  
  theta = np.linspace(-np.pi/2, np.pi/2, 360)
    
  num_of_sub_arrays = 2
  sub_array_size = 2
  
  major_spacing = 0.5  # in wavelengths
  minor_spacing = 0.3  # in wavelengths
  
  element_positions = calculate_element_positions(num_of_sub_arrays, sub_array_size, major_spacing, minor_spacing)
  
  phase_shiftes = get_all_phase_shifts(num_elements=num_of_sub_arrays*sub_array_size, num_of_bits=4)
  
  af = array_factor(weights=np.ones(num_of_sub_arrays*sub_array_size),
                    num_elements=num_of_sub_arrays*sub_array_size,
                    psi=phase_shift(spacings=element_positions,
                                    frequency=2.4e9,
                                    theta=theta,
                                    beta=phase_shiftes))
  
  ep = gen_rect_pattern(theta=theta, freq=2.4e9)
  ap = ep * af
  
  print(main_lobe_direction(ap, theta))
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plot_pattern(ax, ap, theta)
  
  plt.show()
  pass


if __name__ == "__main__":
    main()