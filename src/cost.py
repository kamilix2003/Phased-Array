import numpy as np
import pattern_measurements as pm

def bandwidth_cost(theta, pattern, goal_bandwidth, type = 'HPBW'):
  if type == 'HPBW':
    bw = pm.HPBW(pattern, theta)
  elif type == 'FNBW':
    bw = pm.FNBW(pattern, theta)
  else:
    raise ValueError("Invalid bandwidth type. Use 'HPBW' or 'FNBW'.")
  return np.abs(bw - goal_bandwidth)
  pass

def beam_steering_cost(theta, pattern, steer_angle):
  beam_direction = pm.main_lobe_direction(pattern, theta)
  return np.abs(beam_direction - steer_angle)
  pass

def side_lobe_cost(theta, pattern, threshold=0.01):
  l_lobe = pm.get_lobe(pattern, theta, 1)
  r_lobe = pm.get_lobe(pattern, theta, -1)
  l_max = np.max(pattern[l_lobe])
  r_max = np.max(pattern[r_lobe])
  if 0 < threshold < 1:
    if l_max < threshold and r_max < threshold:
      return 0  # Side lobes are below the threshold
    return np.abs(l_max - threshold) + np.abs(r_max - threshold)
  else:
    from utils import linear_to_db
    if linear_to_db(l_max) < threshold and linear_to_db(r_max) < threshold:
      return 0  # Side lobes are below the threshold
    return np.abs(linear_to_db(l_max) - threshold) + np.abs(linear_to_db(r_max) - threshold)

def beam_boundry_cost(theta, pattern):
  if pattern.shape[0] < 2:
    raise ValueError("Pattern must have at least two steer angles for coverage cost calculation.")
  cost = np.zeros(pattern.shape[0] - 1)
  for i in range(1, pattern.shape[0]):
    cost[i - 1] = np.abs(pm.HPBW_bounds(pattern[i, :], theta)[0] - pm.HPBW_bounds(pattern[i - 1, :], theta)[1])
  # print(f'Coverage cost: {cost}')
  return np.abs(np.sum(cost))
  pass

def coverage_cost(theta, pattern, thershold = .707, digital=False):
  if pattern.shape[0] < 2:
    raise ValueError("Pattern must have at least two steer angles for coverage cost calculation.")
  overlap = np.zeros_like(pattern, dtype=float)
  for i in range(pattern.shape[0]):
    main_lobe_i = pm.get_lobe(pattern[i, :], theta)
    bw_mask = pattern[i, main_lobe_i] >= thershold * np.max(pattern[i, :])
    if digital:
      overlap[i, main_lobe_i[bw_mask]] += 1
    else:
      overlap[i, main_lobe_i[bw_mask]] = pattern[i, main_lobe_i[bw_mask]]
  
  # import matplotlib.pyplot as plt
  # fig = plt.figure()
  # ax = fig.add_subplot(4, 1, 1)
  # for i in range(pattern.shape[0]):
  #   ax.plot(np.degrees(theta), overlap[i, :], label=f'Steer Angle {i+1}')
  # plt.legend()
  # temp = np.abs(np.prod(overlap + 1, axis=0))-1
  # fig.add_subplot(4, 1, 2).plot(np.degrees(theta), temp, label='overlap prod')
  # plt.legend()
  # fig.add_subplot(4, 1, 3).plot(np.degrees(theta), temp - np.sum(overlap, axis=0), label='prod - sum')
  # plt.legend()
  # fig.add_subplot(4, 1, 4).plot(np.degrees(theta), np.clip(np.sum(overlap, axis=0) - 1, 0, None), label='abs(sum - 1)')
  # plt.legend()
  # plt.show()
    
  if digital:
    return np.average(np.abs(np.prod(overlap + 1, axis=0))-1)
  else:
    return np.average(np.abs(np.prod(overlap + 1, axis=0))-1)
  
def main():

  import matplotlib.pyplot as plt
  
  from antenna_array import AntennaArray, phase_shift, array_factor
  from beam_steering import steer_to_phase
  from scipy.constants import c
  from pattern import gen_rect_pattern
  from utils import linear_to_db, get_pattern
  from plotting import plot_pattern
  from spacing import gen_spacing
  from beam_steering import quantize_phase, gen_steer_directions
  
  theta = np.linspace(-np.pi/2, np.pi/2, 360)
  frequency = 2.4e9  # 2.4 GHz
  num_elements = 6
  uni_spacing = gen_spacing(num_elements, [0.75]) * c / frequency
  beta = gen_steer_directions(num_elements, 4, step=1)
    
  ap = get_pattern(theta, frequency, num_elements, uni_spacing, beta)
  ap_db = linear_to_db(ap)
  
  print(f'HPBW: {pm.HPBW(ap[0, :], theta)}')
  print(f'Side lobe cost: {side_lobe_cost(theta, ap[0, :])}')
  
  print(f'HPBW: {pm.HPBW(ap[1, :], theta)}')
  print(f'Side lobe cost: {side_lobe_cost(theta, ap[1, :])}')
  
  print(f'coverage cost: {coverage_cost(theta, ap, digital=True)}')
  
  fig, ax = plt.subplots()
  for i in range(ap.shape[0]):
    ap_db_i = linear_to_db(ap[i, :])
    ax.plot(np.degrees(theta), ap_db_i, label='Steer Angle 1')
    bw = pm.HPBW_bounds(ap_db_i, theta, idx=True)
    plt.fill_between(np.degrees(theta[bw[0]:bw[1]]), ap_db_i[bw[0]:bw[1]], -60, color='red', alpha=0.2)

  ax.set_xlabel('Theta (radians)')
  ax.set_ylabel('Array Pattern (dB)')
  ax.set_title('Array Pattern for Different Steer Angles')
  ax.legend()
  ax.set_ylim(-60, 3)
  plt.grid()
  plt.show()
  
  pass

if __name__ == "__main__":
  main()