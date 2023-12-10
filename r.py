import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa.display
from pydub import AudioSegment
import scipy

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer")

        self.file_path = None
        self.audio_data = None
        self.freq_plot = None

        self.plot_count = 6
        self.current_freq_plot = 0
        self.frequency_values = [250, 1000, 8000]

        self.create_widgets()

    def create_widgets(self):
        # Load File Button
        self.current_plot_label = tk.Label(self.root, text="Current Plot: None")
        self.current_plot_label.pack(pady=10)

        self.file_label = tk.Label(self.root, text="Current File: None")
        self.file_label.pack()

        load_button = tk.Button(self.root, text="Load File", command=self.load_file)
        load_button.pack()

        self.toggle_freq_button = tk.Button(root, text="Change Plot", command=self.show_next_plot)
        self.toggle_freq_button.pack(pady=10)
        # Display Length
        self.length_label = tk.Label(self.root, text="Length: None seconds")
        self.length_label.pack()

        # Display Resonance Frequency
        self.resonance_label = tk.Label(self.root, text="Resonance: None Hz")
        self.resonance_label.pack()

        # Display Low, Mid, High Frequencies
        self.frequency_labels = tk.Label(self.root, text="Low: None | Mid: None | High: None")
        self.frequency_labels.pack()

        # Waveform Plot
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack()

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])

        if file_path:
            self.file_path = file_path
            self.file_label.config(text=f"File: {os.path.basename(self.file_path)}")

            self.process_audio()

    def process_audio(self):
        # Check if the file is a .wav file
        if not self.file_path.lower().endswith('.wav'):
            wav_path = self.convert_to_wav()
        else:
            wav_path = self.file_path

        # Read .wav file
        self.sample_rate, audio_data = wavfile.read(wav_path)
        self.audio_data = audio_data / np.max(np.abs(audio_data), axis=0)
        duration = len(audio_data) / self.sample_rate

        # Display length
        self.length_label.config(text=f"Length: {duration:.2f} seconds")

        # Display plot
        self.display_plot()

    def display_plot(self):
        self.current_plot_label.config(text=f"Current Plot: {self.get_plot_name()}")
        if self.current_freq_plot == 0:
            self.plot_spectrogram()
            return
        if self.current_freq_plot == 1 or self.current_freq_plot == 2 or self.current_freq_plot == 3:
            target = self.frequency_values[self.current_freq_plot - 1]
            self.plot_frequency(self.frequency_check(target))
            return
        if self.current_freq_plot == 4:
            return
        if self.current_freq_plot == 5:
            return

    def get_plot_name(self):
        if self.current_freq_plot == 0: return "Spectrogram"
        if self.current_freq_plot == 1: return "Low"
        if self.current_freq_plot == 2: return "Mid"
        if self.current_freq_plot == 3: return "High"
        if self.current_freq_plot == 4: return "Combined"
        if self.current_freq_plot == 5: return "Waveform"

    def show_next_plot(self):
        if not self.file_path:
            return
        self.current_freq_plot = (self.current_freq_plot + 1) % self.plot_count
        self.display_plot()

    def convert_to_wav(self):
        # Convert to .wav using pydub
        audio = AudioSegment.from_file(self.file_path)
        wav_path = os.path.splitext(self.file_path)[0] + ".wav"
        audio.export(wav_path, format="wav")
        return wav_path

    def find_target_frequency(self, freqs, target):
        for x in freqs:
            if x > target:
                break
        return x

    def frequency_check(self, target):
        # you can choose a frequency which you want to check
        target_frequency = self.find_target_frequency(self.frequencies, target)
        index_of_frequency = np.where(self.frequencies == target_frequency)[0][0]
        # find a sound data for a particular frequency
        data_for_frequency = self.spectrum[index_of_frequency]
        # change a digital signal for a values in decibels
        data_in_db_fun = 10 * np.log10(data_for_frequency)
        return data_in_db_fun

    def find_nearest_value(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def plot_spectrogram(self):
        if self.freq_plot:
            self.freq_plot.clear()
        else:
            plt.figure()
            self.freq_plot = plt.subplot(111)

        self.spectrum, self.frequencies, self.time_values, _ = plt.specgram(self.audio_data, Fs=self.sample_rate,
                                              NFFT=1024, cmap=plt.get_cmap('autumn_r'))

        plt.show()

    def plot_frequency(self, db_data):
        if self.freq_plot:
            self.freq_plot.clear()
        else:
            plt.figure()
            self.freq_plot = plt.subplot(111)

        plt.plot(self.time_values, db_data, linewidth=1, alpha=0.7, color='#004bc6')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (dB)')

        index_of_max = np.argmax(db_data)
        self.resonance_label.config(text=f"Resonance: {np.round(self.frequencies[index_of_max], 3)} Hz")

        value_of_max = db_data[index_of_max]
        plt.plot(self.time_values[index_of_max], db_data[index_of_max], 'go')

        sliced_array = db_data[index_of_max:]

        value_of_max_less_5 = self.find_nearest_value(sliced_array, value_of_max - 5)
        index_of_max_less_5 = np.where(db_data == value_of_max_less_5)
        plt.plot(self.time_values[index_of_max_less_5], db_data[index_of_max_less_5], 'yo')

        value_of_max_less_25 = self.find_nearest_value(sliced_array, value_of_max - 25)
        index_of_max_less_25 = np.where(db_data == value_of_max_less_25)

        plt.plot(self.time_values[index_of_max_less_25], db_data[index_of_max_less_25], 'ro')

        time_average = (self.time_values[index_of_max] + self.time_values[index_of_max_less_5] + self.time_values[index_of_max_less_25]) / 3
        rt20 = (self.time_values[index_of_max_less_5] - self.time_values[index_of_max_less_25])[0]
        rt60 = 3 * rt20

        self.frequency_labels.config(text=f"Average: {np.round(time_average[0], 3)}, Diff: {np.round(time_average[0] - 0.5, 3)}")

        plt.grid()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()