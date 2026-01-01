
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

class RadiationPattern:
  
  dims = {
    'azimuth': 0,
    'elevation': 1,
    'polarization': 2,
    'offset': 3,
    'frequency': 4
  }
  
  def __init__(self, file_path):
    
    self.file_path = file_path
    
    self.config = {}
    
    self.azimuth = None
    self.elevation = None
    self.polarization = None
    self.offset = None
    self.frequency = None
    
    self.raw_data_matrix = None
    self.data = None
    self.magnitude = None
    self.magnitude_db = None
    self.phase = None
    
    self.dataframe = None
    
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
          self.config[key.strip()] = float(value.strip())
        
      if self.config['Azimuth step [deg]'] != 0:
        self.azimuth = np.arange(self.config['Azimuth start [deg]'],
                                      self.config['Azimuth stop [deg]'] + 1,
                                      self.config['Azimuth step [deg]'])
      
      if self.config['Elevation step [deg]'] != 0:
        self.elevation = np.arange(self.config['Elevation start [deg]'],
                                        self.config['Elevation stop [deg]'] + 1,
                                        self.config['Elevation step [deg]'])
      
      if self.config['Polarization step [deg]'] != 0:
        self.polarization = np.arange(self.config['Polarization start [deg]'],
                                    self.config['Polarization stop [deg]'] + 1,
                                    self.config['Polarization step [deg]'])
        
      if self.config['Offset step [mm]'] != 0:
        self.offset = np.arange(self.config['Offset start [mm]'],
                                       self.config['Offset stop [mm]'] + 1,
                                       self.config['Offset step [mm]'])
      
      
      f.readline()  # Skip empty line
        
      self.raw_data_matrix = np.array([line.split() for line in f.readlines()])
      
      self.frequency = self.raw_data_matrix[0, 4::2].astype(float)
      
      self.data = np.zeros(((
        len(self.azimuth) if self.azimuth is not None else 1,
        len(self.elevation) if self.elevation is not None else 1,
        len(self.polarization) if self.polarization is not None else 1,
        len(self.offset) if self.offset is not None else 1,
        len(self.frequency) if self.frequency is not None else 1
      )), dtype=complex)
      
      self.magnitude = np.zeros_like(self.data, dtype=float)
      self.magnitude_db = np.zeros_like(self.data, dtype=float)
      self.phase = np.zeros_like(self.data, dtype=float)
      
      for line in self.raw_data_matrix[1:]:
        
        mag_db = line[4::2].astype(float)
        mag = np.power(10, mag_db / 20)
        phase = line[5::2].astype(float)
        complex_data = mag * np.exp(1j * np.deg2rad(phase))
        
        azimuth_idx = elevation_idx = polarization_idx = offset_idx = 0
        
        if self.azimuth is not None:
          azimuth_idx = np.where(self.azimuth == float(line[self.dims.get("azimuth")]))[0][0]
        if self.elevation is not None:
          elevation_idx = np.where(self.elevation == float(line[self.dims.get("elevation")]))[0][0]
        if self.polarization is not None:
          polarization_idx = np.where(self.polarization == float(line[self.dims.get("polarization")]))[0][0]
        if self.offset is not None:
          offset_idx = np.where(self.offset == float(line[self.dims.get("offset")]))[0][0]
        
        # print(azimuth_idx, elevation_idx, polarization_idx, offset_idx)
        
        self.data[azimuth_idx, elevation_idx, polarization_idx, offset_idx, :] = complex_data
        self.magnitude[azimuth_idx, elevation_idx, polarization_idx, offset_idx, :] = mag
        self.magnitude_db[azimuth_idx, elevation_idx, polarization_idx, offset_idx, :] = mag_db
        self.phase[azimuth_idx, elevation_idx, polarization_idx, offset_idx, :] = phase
      
  def get_pattern(self, azimuth = None, elevation = None, polarization = None,
                  offset = None, frequency = None):
    azimuth_idx = elevation_idx = polarization_idx = offset_idx = frequency_idx = slice(None)

    if azimuth is not None:
      azimuth_idx = self.get_value_idx('azimuth', azimuth)
    if elevation is not None:
      elevation_idx = self.get_value_idx('elevation', elevation)
    if polarization is not None:
      polarization_idx = self.get_value_idx('polarization', polarization)
    if offset is not None:
      offset_idx = self.get_value_idx('offset', offset)
    if frequency is not None:
      frequency_idx = self.get_value_idx('frequency', frequency)    
    
    indexes = {
      'azimuth': azimuth_idx,
      'elevation': elevation_idx,
      'polarization': polarization_idx,
      'offset': offset_idx,
      'frequency': frequency_idx
    }
    
    return self.magnitude_db[azimuth_idx, elevation_idx, polarization_idx, offset_idx, frequency_idx], indexes
    
  def get_value_idx(self, dim_name, values):
    if dim_name not in self.dims:
      raise ValueError(f"Dimension {dim_name} not recognized.")
    idx = []
    for val in values:
      index = np.argmin(np.abs(self.__getattribute__(dim_name) - val))
      idx.append(index)
    return np.array(idx)
  
  def get_value_range_idx(self, dim_name, value_range):
    if dim_name not in self.dims:
      raise ValueError(f"Dimension {dim_name} not recognized.")
    dim_values = self.__getattribute__(dim_name)
    idx = np.where((dim_values >= value_range[0]) & (dim_values <= value_range[1]))[0]
    return idx

if __name__ == "__main__":
  import numpy as np
  from matplotlib.axes import Axes
  import matplotlib.pyplot as plt
  
  # path = "src/patch_measurements/antena_v01_azymut.txt"
  path = "src/patch_measurements/antena_v01_azymut90deg_5xoffset.txt"
  pattern = RadiationPattern(path)
  global_max = np.max(pattern.magnitude_db)
  print("Global max dB:", global_max)
  
  data, ranges = pattern.get_pattern()
  data -= np.max(data, axis = 0)
  print("Data shape:", data.shape)
  
  freq_idxs = pattern.get_value_range_idx('frequency', (2, 4))
  
  std_db = np.std(data[:, 0, 0, 0, freq_idxs], axis=0)
  var_db = np.var(data[:, 0, 0, 0, freq_idxs], axis=0)
  meadian_db = np.median(data[:, 0, 0, 0, freq_idxs], axis=0)
  mean_db = np.mean(data[:, 0, 0, 0, freq_idxs], axis=0)
  
    
  plt.plot(pattern.frequency[freq_idxs], std_db, label='Std dB')
  # plt.plot(pattern.frequency[freq_idxs], var_db, label='Var dB')
  # plt.plot(pattern.frequency[freq_idxs], meadian_db, label='Median dB')
  # plt.plot(pattern.frequency[freq_idxs], mean_db, label='Mean dB')
  # plt.plot(pattern.frequency[freq_idxs], spread_db, label='Spread dB')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Magnitude (dB)')
  plt.title('Radiation Pattern Statistics vs Frequency')
  plt.legend()
  plt.grid()
  plt.show()
  
  
