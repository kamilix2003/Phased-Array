import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from antenna_element import AntennaElement
from radiation_pattern import RadiationPattern
from antenna_array import AntennaArray
from utils import wavelength

theta = np.linspace(-np.pi, np.pi, 3600)
frequency = 2.4e9  # Frequency in Hz

pattern1 = RadiationPattern(
    name='Pattern1',
    frequency=frequency,
    theta=theta,
    pattern=np.cos(theta/2)**2  # Example pattern
    # pattern=np.sinc(theta)**2  # Example pattern
)

element1 = AntennaElement(
    name='Element1',
    patterns=np.array([pattern1])
)

array = AntennaArray(
    name='Array1',
    antenna=element1,
    num_elements=4,
    spacing=wavelength(frequency) * 0.4  # Spacing in meters
)

beta = np.pi / 3

af = array.array_factor(frequency, theta, beta)

E = array.radiation_pattern(frequency, theta, beta)

fig = plt.figure(figsize=(10, 6))

ax = plt.subplot(131, projection='polar')
ax.set_theta_zero_location('N')
plt.polar(theta, np.abs(af), label='Array Factor')
plt.title('Array Factor')

ax = plt.subplot(132, projection='polar')
ax.set_theta_zero_location('N')
plt.polar(pattern1.theta, pattern1.pattern, label='Element Pattern')
plt.title('Element Pattern')

ax = plt.subplot(133, projection='polar')
ax.set_theta_zero_location('N')
plt.polar(theta, np.abs(E), label='Array Factor')
plt.title('Antenna Array Pattern')

plt.show()