# From https://pythontic.com/visualization/charts/sinewave

import numpy as np
import matplotlib.pyplot as plt

# Generate time values for the sine wave
time = np.arange(0, 25, 0.1)

# Calculate the amplitude of the sine wave
amplitude = np.sin(time)

# Plot the sine wave
plt.plot(time, amplitude)

# Add title and axis labels
plt.title("Sine Wave")
plt.xlabel("Time")
plt.ylabel("Amplitude = sin(time)")

# Add grid and horizontal line at y=0
plt.grid(True, which="both", linestyle="--", color="gray", linewidth=0.5)
plt.axhline(y=0, color="black", linewidth=1)

# Display the sine wave plot
plt.show()