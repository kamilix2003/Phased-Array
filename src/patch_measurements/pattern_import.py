
class RadiationPattern:
  
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
      
        
      self.raw_data_matrix = f.readlines()
      
    pass


if __name__ == "__main__":
  import numpy as np
  from matplotlib.axes import Axes
  import matplotlib.pyplot as plt
  
  path = "src/patch_measurements/antena_v01_azymut.txt"
  pattern = RadiationPattern(path)
  print(pattern.config)
  print(pattern.azimuth_angles)
  print(pattern.elevation_angles)
  