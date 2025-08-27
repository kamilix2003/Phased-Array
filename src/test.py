import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from antenna_array import phase_shift, array_factor
from pattern import gen_rect_pattern
from utils import linear_to_db

from pandas import DataFrame

from pattern_measurements import HPBW

# Parameters
N = 2  # number of antenna elements
theta = np.linspace(-np.pi, np.pi, 361)  # angle space
frequency = 2.4e9  # frequency in Hz
shift_bits = 4  # number of bits for phase shift
lsb_shift = np.pi * 2 / (2 ** shift_bits)  # least significant bit shift in radians
lsb_shift_m = c / (frequency * (2 ** shift_bits))  # LSB shift in meters
print(f"LSB shift: {np.degrees(lsb_shift)} degrees, {lsb_shift} radians, {lsb_shift_m} m")

beta = (np.arange(N) * lsb_shift) * np.arange(2**shift_bits)[:, np.newaxis]  # phase shifts for each element
print(f"Phase shifts (beta): {np.degrees(beta)} degrees")

d = np.array([0, lsb_shift_m])
print(f"Element spacing (d): {d}")

fig = plt.figure(figsize=(10, 6))

for i in range(2, 2**shift_bits, 2):
  weights = np.ones(N)
  psi = phase_shift(d * i, frequency, theta, beta)
  af = array_factor(weights, N, psi)
  ap = np.abs(af) * gen_rect_pattern(theta, frequency)
  
  dir_list = []
  amp_list = []
  for j in range(ap.shape[0]):
    dir_list.append(np.degrees(theta[np.argmax(ap[j, :])]))
    amp_list.append(np.max(ap[j, :]))

  dir_list_sort, amp_list_sort = zip(*sorted(zip(dir_list, amp_list)))

  plt.plot(dir_list_sort, linear_to_db(amp_list_sort), label=f'Phase Shift {i}')

  print(f'd = {d[1] * i:.4f}m, max steer = {np.max(np.abs(dir_list)):.2f} degrees')
  print(f"direction = {list(f'{angle:.2f}' for angle in dir_list)} degrees")
  print(f"amplitude = {list(f'{amp:.2f}' for amp in amp_list)}")

plt.show()


# plt.figure(figsize=(10, 6)) 
# for i in range(ap.shape[0]):
#   plt.plot(np.degrees(theta), ap[i, :], label=f"Beam {i+1} (Phase Shift {i})")
# plt.grid()
# plt.show()
