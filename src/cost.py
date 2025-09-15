import numpy as np
import pattern_measurements as pm

debug = False

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

def get_side_lobe(theta, pattern):
  l_max = None
  r_max = None
  
  lobe_count = pm.find_maximas(pattern, theta).size
  print(lobe_count)
  
  current_lobe = 1
  while l_max is None or r_max is None:
    if current_lobe > 1:
      print(f'Checking lobe {current_lobe}')
      
    l_lobe = pm.get_lobe(pattern, theta, -current_lobe)
    ll_neighbour = pm.get_lobe(pattern, theta, -current_lobe - 1)
    lr_neighbour = pm.get_lobe(pattern, theta, -current_lobe + 1)
    
    r_lobe = pm.get_lobe(pattern, theta, current_lobe)
    rl_neighbour = pm.get_lobe(pattern, theta, current_lobe - 1)
    rr_neighbour = pm.get_lobe(pattern, theta, current_lobe + 1)
    if np.max(pattern[lr_neighbour]) > np.max(pattern[l_lobe]) > np.max(pattern[ll_neighbour]):
      l_max = np.max(pattern[l_lobe])
    if np.max(pattern[rl_neighbour]) > np.max(pattern[r_lobe]) > np.max(pattern[rr_neighbour]):
      r_max = np.max(pattern[r_lobe])
      
    if current_lobe >= lobe_count // 2 - 1 :  # Prevent infinite loop
      l_max = np.max(pattern[pm.get_lobe(pattern, theta, -1)])
      r_max = np.max(pattern[pm.get_lobe(pattern, theta, 1)])
    else:
      current_lobe += 1
  return l_max, r_max

def side_lobe_cost_v1(theta, pattern, threshold=0.01):
  l_max, r_max = get_side_lobe(theta, pattern)
  if 0 < threshold < 1:
    if l_max < threshold and r_max < threshold:
      return 0  # Side lobes are below the threshold
    return np.abs(l_max - threshold) + np.abs(r_max - threshold)
  else:
    from utils import linear_to_db
    if linear_to_db(l_max) < threshold and linear_to_db(r_max) < threshold:
      return 0  # Side lobes are below the threshold
    return np.abs(linear_to_db(l_max) - threshold) + np.abs(linear_to_db(r_max) - threshold)

def side_lobe_cost_v2(theta, pattern, threshold=0.1):
  
  pattern_wo_main = np.copy(pattern)
  sll_level = np.zeros(pattern.shape[0])
  for i, ap in enumerate(pattern):
    main_lobe = pm.get_lobe(ap, theta)
    pattern_wo_main[i, main_lobe] = 0
    sll_level[i] = np.max(pattern_wo_main[i, :])
    
  if debug:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(311)
    for i in range(pattern.shape[0]):
      ax.plot(np.degrees(theta), pattern_wo_main[i, :], label=f'Steer Angle {i+1}')
      ax.set_title('Pattern without Main Lobe')
      ax.legend()
    ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    ax = fig.add_subplot(312)
    ax.plot(np.arange(sll_level.size), sll_level, marker='o', label='SLL Level')
    ax.set_title('Side Lobe Level for Each Steer Angle')
    ax.grid()
    plt.show()
  
  return np.sum(np.clip(sll_level - threshold, 0, None)) / pattern.shape[0]

def side_lobe_cost(theta, pattern, **kwargs):
  return side_lobe_cost_v2(theta, pattern, **kwargs)

def coverage_cost_v1(theta, pattern):
  if pattern.shape[0] < 2:
    raise ValueError("Pattern must have at least two steer angles for coverage cost calculation.")
  cost = np.zeros(pattern.shape[0] - 1)
  for i in range(1, pattern.shape[0]):
    cost[i - 1] = np.abs(pm.HPBW_bounds(pattern[i, :], theta)[0] - pm.HPBW_bounds(pattern[i - 1, :], theta)[1])
  # print(f'Coverage cost: {cost}')
  return np.abs(np.sum(cost))
  pass

def coverage_cost_v2(theta, pattern, thershold = .5, digital=False):
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
  
  if debug:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(4, 1, 1)
    for i in range(pattern.shape[0]):
      ax.plot(np.degrees(theta), overlap[i, :], label=f'Steer Angle {i+1}')
    plt.legend()
    temp = np.abs(np.prod(overlap + 1, axis=0))-1
    fig.add_subplot(4, 1, 2).plot(np.degrees(theta), temp, label='overlap prod')
    plt.legend()
    fig.add_subplot(4, 1, 3).plot(np.degrees(theta), temp - np.sum(overlap, axis=0), label='prod - sum')
    plt.legend()
    fig.add_subplot(4, 1, 4).plot(np.degrees(theta), np.clip(np.sum(overlap, axis=0) - 1, 0, None), label='abs(sum - 1)')
    plt.legend()
    plt.show()
    
  if digital:
    return np.average(np.abs(np.prod(overlap + 1, axis=0))-1)
  else:
    return np.average(np.abs(np.prod(overlap + 1, axis=0))-1)
  
def coverage_cost_v3(theta, pattern, thershold = .5):
  if pattern.shape[0] < 2:
    raise ValueError("Pattern must have at least two steer angles for coverage cost calculation.")
    
  pattern_main = np.zeros_like(pattern)
  for i in range(pattern.shape[0]):
    main_lobe = pm.get_lobe(pattern[i, :], theta)
    thershold_mask = pattern[i, main_lobe] >= thershold * np.max(pattern[i, :])
    main_lobe = main_lobe[thershold_mask]
    pattern_main[i, main_lobe] = pattern[i, main_lobe]
    
  sum_pattern = np.sum(pattern_main, axis=0)
  prod_pattern = np.prod(pattern_main + 1, axis=0) - 1
  cost = np.abs(prod_pattern - sum_pattern)
  
  if debug:
    import matplotlib.pyplot as plt
      
    plt.figure(figsize=(8, 6))
      
    plt.subplot(311).plot(np.degrees(theta), sum_pattern, label='Pattern Product')
    plt.subplot(312).plot(np.degrees(theta), prod_pattern, label='Pattern Product')
    plt.subplot(313).plot(np.degrees(theta), cost, label='Pattern Product')
    plt.show()
  
  return np.average(cost)
  pass

def coverage_cost_v4(theta, pattern, thershold = .5, cost_pattern = False):
  if pattern.shape[0] < 2:
    raise ValueError("Pattern must have at least two steer angles for coverage cost calculation.")
    
  pattern_main = np.zeros_like(pattern)
  l_bound, r_bound = theta.size - 1, 0
  for i in range(pattern.shape[0]):
    main_lobe = pm.get_lobe(pattern[i, :], theta)
    thershold_mask = pattern[i, main_lobe] >= thershold * np.max(pattern[i, :])
    main_lobe = main_lobe[thershold_mask]
    l_bound, r_bound = min(l_bound, main_lobe[0]), max(r_bound, main_lobe[-1])
    pattern_main[i, main_lobe] = pattern[i, main_lobe]
  
  beam_count = pattern.shape[0]
    
  sum_pattern = np.sum(pattern_main, axis=0) / beam_count
  prod_pattern = (np.prod(pattern_main + 1, axis=0) - 1) / beam_count
  max_pattern = np.max(pattern_main, axis=0) / beam_count
  cost = np.abs(sum_pattern - max_pattern)
  
  # cost[l_bound:r_bound+1][sum_pattern[l_bound:r_bound+1] == 0] = 1
  # cost *= np.sum(cost) / (r_bound - l_bound + 1)
    
    
  if debug:
    import matplotlib.pyplot as plt
      
    plt.figure(figsize=(8, 6))
      
    plt.subplot(411).plot(np.degrees(theta), sum_pattern, label='Pattern Sum')
    plt.subplot(412).plot(np.degrees(theta), max_pattern, label='Pattern Max')
    plt.subplot(414).plot(np.degrees(theta), cost, label='Pattern Cost')
    plt.legend()
    plt.show()
  
  if cost_pattern:
    return cost
  return np.average(cost)
  pass
  
def coverage_cost(thata, pattern, **kwargs):
  return coverage_cost_v4(thata, pattern, **kwargs)

def total_cost(theta, pattern, log=None):
  sll_cost = side_lobe_cost(theta, pattern, threshold=0.1)
  cov_cost = coverage_cost(theta, pattern, thershold=0.5)
  total_cost = sll_cost + cov_cost
  
  if log is not None:
    log['cost']['sll'].append(sll_cost)
    log['cost']['cov'].append(cov_cost)
    log['total_cost'].append(total_cost)
  
  return total_cost
  
def main():

  import matplotlib.pyplot as plt
  
  from antenna_array import AntennaArray, phase_shift, array_factor
  from beam_steering import steer_to_phase
  from scipy.constants import c
  from pattern import gen_rect_pattern
  from utils import linear_to_db, get_pattern
  from plotting import plot_pattern, fill_HPBW
  from spacing import gen_spacing
  from beam_steering import quantize_phase, gen_steer_directions
  
  theta = np.linspace(-np.pi/2, np.pi/2, 360)
  frequency = 2.4e9  # 2.4 GHz
  num_elements = 6
  uni_spacing = gen_spacing(num_elements, [0.75]) * c / frequency
  beta = gen_steer_directions(num_elements, 4, step=3)
    
  ap = get_pattern(theta, frequency, num_elements, uni_spacing, beta)
  ap_db = linear_to_db(ap)
    
  print(f'HPBW: {pm.HPBW(ap[0, :], theta)}')
  # print(f'Side lobe cost: {side_lobe_cost(theta, ap[0, :])}')
  
  print(f'HPBW: {pm.HPBW(ap[1, :], theta)}')
  # print(f'Side lobe cost: {side_lobe_cost(theta, ap[1, :])}')
  
  print(f"SLL: {side_lobe_cost(theta, ap, threshold=0.05)}")
  
  print(f'coverage cost v1: {coverage_cost_v1(theta, ap)}')
  print(f'coverage cost v2: {coverage_cost_v2(theta, ap, digital=False)}')
  print(f'coverage cost v3: {coverage_cost_v3(theta, ap)}')
  print(f'coverage cost v4: {coverage_cost_v4(theta, ap)}')
  
  fig, ax = plt.subplots()
  for i in range(ap.shape[0]):
    ap_db_i = linear_to_db(ap[i, :])
    ax.plot(np.degrees(theta), ap_db_i, label='Steer Angle 1')
  fill_HPBW(ax, theta, ap)

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