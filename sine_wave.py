import numpy as np
import matplotlib.pyplot as plt

def plot_sine_wave(frequency, duration, sampling_rate):
    time = np.arange(0, duration, 1/sampling_rate)
    amplitude = np.sin(2 * np.pi * frequency * time)

    plt.plot(time, amplitude)
    plt.title('Sine Wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

plot_sine_wave(870, 1, 44100)
print("6")