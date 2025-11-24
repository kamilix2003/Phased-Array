
import numpy as np

class RadiationPattern:
  
  dims = {
    'frequency': 0,
    'azimuth': 1,
    'elevation': 2,
    'polarization': 3,
    'offset': 4
  }
  
  def __init__(self, file_path):
    self.file_path = file_path
    self.config = {}
    self.azimuth_angles = None
    self.elevation_angles = None
    self.polarization = None
    self.offset_distance = None
    self.pattern_data = None
    self.frequency = None
    self.raw_data_matrix = None
    self.data = None
    self._load_pattern()

  def _load_pattern(self):
    
    config_size = 17
    
    with open(self.file_path, 'r') as f:
      for _ in range(config_size):
        line = f.readline()
        key, value = line.strip().split(':', 1)
        if key == "IF bandwidth":
          self.config[key.strip()] = value.strip()
        else:
          print(value.strip())
          self.config[key.strip()] = int(value.strip())
        
      if self.config['Azimuth step [deg]'] != 0:
        self.azimuth_angles = np.arange(self.config['Azimuth start [deg]'],
                                      self.config['Azimuth stop [deg]'] + 1,
                                      self.config['Azimuth step [deg]'])
      
      if self.config['Elevation step [deg]'] != 0:
        self.elevation_angles = np.arange(self.config['Elevation start [deg]'],
                                        self.config['Elevation stop [deg]'] + 1,
                                        self.config['Elevation step [deg]'])
      
      if self.config['Polarization step [deg]'] != 0:
        self.polarization = np.arange(self.config['Polarization start [deg]'],
                                    self.config['Polarization stop [deg]'] + 1,
                                    self.config['Polarization step [deg]'])
      if self.config['Offset step [mm]'] != 0:
        self.offset_distance = np.arange(self.config['Offset start [mm]'],
                                       self.config['Offset stop [mm]'] + 1,
                                       self.config['Offset step [mm]'])
      
      
      f.readline()  # Skip empty line
        
      self.raw_data_matrix = np.array([line.split() for line in f.readlines()])
      
      self.frequency = self.raw_data_matrix[0, 4::2]
      
      self.data = np.zeros(((
        len(self.frequency) if self.frequency is not None else 1,
        len(self.azimuth_angles) if self.azimuth_angles is not None else 1,
        len(self.elevation_angles) if self.elevation_angles is not None else 1,
        len(self.polarization) if self.polarization is not None else 1,
        len(self.offset_distance) if self.offset_distance is not None else 1
      )), dtype=complex)
      
      for line in self.raw_data_matrix[1:]:
        
        mag = line[4::2].astype(float)
        phase = line[5::2].astype(float)
        complex_data = mag * np.exp(1j * np.deg2rad(phase))
        
        azimuth_idx = elevation_idx = polarization_idx = offset_idx = 0
        
        if self.azimuth_angles is not None:
          azimuth_idx = np.where(self.azimuth_angles == int(line[0]))[0][0]
        if self.elevation_angles is not None:
          elevation_idx = np.where(self.elevation_angles == int(line[1]))[0][0]
        if self.polarization is not None:
          polarization_idx = np.where(self.polarization == int(line[2]))[0][0]
        if self.offset_distance is not None:
          offset_idx = np.where(self.offset_distance == int(line[3]))[0][0]
        
        # print(azimuth_idx, elevation_idx, polarization_idx, offset_idx)
        
        self.data[:, azimuth_idx, elevation_idx, polarization_idx, offset_idx] = complex_data
      
    pass


if __name__ == "__main__":
  import numpy as np
  from matplotlib.axes import Axes
  import matplotlib.pyplot as plt
  
  path = "src/patch_measurements/antena_v01_azymut.txt"
  pattern = RadiationPattern(path)
  # print(pattern.config)
  # print(pattern.azimuth_angles)
  # print(pattern.elevation_angles)
  # print(pattern.polarization)
  # print(pattern.offset_distance)  
  # print(pattern.frequency)
  # print(pattern.raw_data_matrix.shape)
  f_idx = 300
  plt.plot(pattern.azimuth_angles, np.abs(pattern.data[f_idx, :, 0, 0, 0]))
  plt.show()
