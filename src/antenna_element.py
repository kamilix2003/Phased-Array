import numpy as np

from radiation_pattern import RadiationPattern

class AntennaElement:
    
    def __init__(self, name: str, patterns: np.ndarray[RadiationPattern]) -> None:
        self.name: str = name
        self.patterns: np.ndarray[RadiationPattern] = patterns