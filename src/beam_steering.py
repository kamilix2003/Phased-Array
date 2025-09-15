import numpy as np
from scipy.constants import c

def steer_to_phase(N_elements, spacings, steer_angles, frequency):
    return -spacings * np.sin(steer_angles[:, np.newaxis]) / (c / frequency) * 2 * np.pi 

def quantize_phase(phase, shift_bits, max_shift=2 * np.pi):
  lsb = 2 * np.pi / (2 ** shift_bits)
  return np.min([np.round(phase / lsb) * lsb, np.ones_like(phase) * max_shift], axis=0)

def gen_steer_directions(N, shift_bits=4, step = 1, symmetric=True, extend=0):
  out = (np.arange(N) * (2 * np.pi / (2 ** shift_bits)) * (np.arange(-1, (2**shift_bits) // (N - 1)) + 1 + extend)[:, np.newaxis])[::step, :]
  if symmetric:
    out = np.concatenate([out, -out], axis=0)
    out = np.unique(out, axis=0)
    out = out[np.argsort(np.abs(out[:, 0])), :]
  return out
    
def __steer_directions_count(N, shift_bits):
  return (2 ** shift_bits) // (N - 1)

def main():    
  
  import matplotlib.pyplot as plt
  from scipy.constants import c

  from antenna_array import phase_shift, array_factor
  from pattern import gen_rect_pattern
  from utils import linear_to_db

  from pattern_measurements import HPBW
  
  N = 5  # number of antenna elements
  theta = np.linspace(-np.pi/2, np.pi/2, 361)  # angle space
  frequency = 2.4e9  # frequency in Hz
  shift_bits = 4 # number of bits for phase shift

  lsb_shift = np.pi * 2 / (2 ** shift_bits)  # least significant bit shift in radians
  lsb_shift_m = c / (frequency * (2 ** shift_bits))  # LSB shift in meters
  print(f"LSB shift: {np.degrees(lsb_shift)} degrees, {lsb_shift} radians, {lsb_shift_m} m")

  beta = gen_steer_directions(N, shift_bits)  # phase shifts for each element
  print(f"Phase shifts (beta): {np.degrees(beta)} degrees")
  weights = np.ones(N)  # uniform weights
  print(f"Weights: {weights}")

  d_list = np.linspace(1, 2**shift_bits, 64)[:, np.newaxis] * np.arange(N) * lsb_shift_m  # element spacing

  steer_angle = np.zeros((d_list.shape[0], beta.shape[0]))
  peak_amplitude = np.zeros((d_list.shape[0], beta.shape[0]))
  HPBWs = np.zeros((d_list.shape[0], beta.shape[0]))

  for d_idx, d in enumerate(d_list):
    psi = phase_shift(d, frequency, theta, beta)
    af = array_factor(weights, N, psi)
    ap = np.abs(af) * gen_rect_pattern(theta, frequency)
    
    steer_angle[d_idx, :] = np.degrees(theta[np.argmax(ap, axis=1)])
    peak_amplitude[d_idx, :] = np.max(ap, axis=1)
    for j in range(steer_angle.shape[1]):
      HPBWs[d_idx, j] = HPBW(ap[j, :], theta)

  fig = plt.figure(figsize=(10, 6))
  ax_angle = fig.add_subplot(3, 1, 1)
  ax_angle.grid()
  ax_angle.set_ylabel('Max steer Angle (degrees)')
  ax_angle.set_xlabel('Element Spacing (lsb)')
  ax_angle.set_ylim(-90, 90)
  ax_amp = fig.add_subplot(3, 1, 2)
  ax_amp.grid()
  ax_amp.set_ylabel('Peak Amplitude')
  ax_amp.set_xlabel('Element Spacing (lsb)')
  ax_amp.set_ylim(0, 1)
  ax_bw = fig.add_subplot(3, 1, 3)
  ax_bw.set_ylabel('HPBW (degrees)')
  ax_bw.set_xlabel('Element Spacing (lsb)')
  ax_bw.set_ylim(0, 180)
  ax_bw.grid()
  for i in range(steer_angle.shape[1]):
    ax_angle.plot(d_list[:, 1] / lsb_shift_m, steer_angle[:, i], label=f'Beam: {i}')
    ax_amp.plot(d_list[:, 1] / lsb_shift_m, peak_amplitude[:, i], label=f'Beam: {i}')
    ax_bw.plot(d_list[:, 1] / lsb_shift_m, np.degrees(HPBWs[:, i]), label=f'Beam: {i}')
  ax_bw.legend()
  ax_amp.legend()
  ax_angle.legend()
  plt.show()
    
if __name__ == "__main__":
  main()