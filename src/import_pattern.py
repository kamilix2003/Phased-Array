import os
import numpy as np

def extract_pattern_data(file_path=None):
    """
    Extract data from patch_pattern.txt file.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the patch_pattern.txt file. If None, defaults to data/patch_pattern.txt.
    
    Returns:
    --------
    tuple
        A tuple containing:
        - 2D numpy array with measured power values (rows: angles, columns: frequencies)
        - 1D numpy array with angle values in degrees
        - 1D numpy array with frequency values in GHz
    """
    if file_path is None:
        # Get the absolute path of the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        file_path = os.path.join(project_root, 'data', 'patch_pattern.txt')
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse the metadata
    metadata = {}
    line_idx = 0
    
    # Skip comment line if present
    if lines[0].strip().startswith('//'):
        line_idx += 1
    
    # Parse metadata until empty line or data starts
    while line_idx < len(lines) and not lines[line_idx].strip().startswith('0\t'):
        line = lines[line_idx].strip()
        if line and ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
        line_idx += 1
        
    # Skip empty line if present
    if not lines[line_idx].strip():
        line_idx += 1
    
    # Extract frequency values from the first data line
    freq_line = lines[line_idx].strip().split('\t')
    frequencies = np.array([float(freq.replace(',', '.')) for freq in freq_line[1:]])
    
    # Extract angles and power values
    angles = []
    power_values = []
    
    for line in lines[line_idx+1:]:
        if line.strip():  # Skip empty lines
            values = line.strip().split('\t')
            try:
                angle = float(values[0])
                powers = [float(p.replace(',', '.')) for p in values[1:]]
                angles.append(angle)
                power_values.append(powers)
            except ValueError:
                continue  # Skip lines that can't be parsed
    
    angles = np.array(angles)
    power_matrix = np.array(power_values)
    
    return power_matrix, angles, frequencies

from scipy.interpolate import pchip_interpolate
def prepare_data(power_matrix, angles, frequencies, N):
  power = power_matrix[:, frequencies == 2.5]  # Get power values at 2.5 GHz
  theta = np.linspace(0, 180, N)
  power_interp = pchip_interpolate(angles, power, theta)
  return power_interp

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from numpy import interp
    
    power_matrix, angles, frequencies = extract_pattern_data()
    
    # Print some information about the data
    print(f"Data shape: {power_matrix.shape}")
    print(f"Angle range: {angles.min()} to {angles.max()} degrees, {len(angles)} values")
    print(f"Frequency range: {frequencies.min()} to {frequencies.max()} GHz, {len(frequencies)} values")
    
    # Print a small sample of the power matrix
    theta = np.linspace(-np.pi/2, np.pi/2, 360)
    power = prepare_data(power_matrix, angles, frequencies, 360)
    power = power - np.max(power)  # Normalize power to max value

    plt.figure(figsize=(10, 6))
    plt.plot(theta, power)
    plt.title('Measured Power Pattern at 2.4 GHz')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Power (dB)')
    plt.grid()
    plt.show()
    