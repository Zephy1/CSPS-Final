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

        self.create_widgets()

    def create_widgets(self):
        # Load File Button
        load_button = tk.Button(self.root, text="Load File", command=self.load_file)
        load_button.pack(pady=10)

        # Display File Name
        self.file_label = tk.Label(self.root, text="File: None")
        self.file_label.pack()

        # Display Length
        self.length_label = tk.Label(self.root, text="Length: None seconds")
        self.length_label.pack()

        # Display Resonance Frequency
        self.resonance_label = tk.Label(self.root, text="Resonance: None Hz")
        self.resonance_label.pack()

        # Display Low, Mid, High Frequencies
        self.frequency_labels = tk.Label(self.root, text="Low: None dB | Mid: None dB | High: None dB")
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

        # Display waveform
        # self.plot_waveform()
        # self.plot_decibel_vs_frequency()
        self.plot_line_chart()
        self.plot_frequency_ranges()

        # Compute and display resonance frequency
        resonance_freq = self.compute_resonance_frequency()
        self.resonance_label.config(text=f"Resonance: {resonance_freq:.2f} Hz")

        # Compute and display low, mid, high frequencies
        low, mid, high = self.compute_frequency_ranges()
        self.frequency_labels.config(text=f"Low: {low:.2f} dB | Mid: {mid:.2f} dB | High: {high:.2f} dB")

    def convert_to_wav(self):
        # Convert to .wav using pydub
        audio = AudioSegment.from_file(self.file_path)
        wav_path = os.path.splitext(self.file_path)[0] + ".wav"
        audio.export(wav_path, format="wav")
        return wav_path

    def plot_waveform(self):
        plt.figure(figsize=(10, 4))
        time = np.arange(0, len(self.audio_data)) / self.sample_rate
        plt.plot(time, self.audio_data)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig("waveform.png")

        # Display the waveform plot
        self.plot_image("waveform.png")

    def compute_resonance_frequency(self):
        # Your resonance frequency computation logic here
        # This is just a placeholder, replace with your actual implementation
        return np.random.uniform(50, 1000)

    def plot_image(self, image_path):
        img = tk.PhotoImage(file=image_path)
        panel = tk.Label(self.plot_frame, image=img)
        panel.image = img
        panel.pack()

    def plot_decibel_vs_frequency(self):
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
        librosa.display.specshow(D, sr=44100, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig("decibel_vs_frequency.png")

        # Display the decibel vs frequency plot
        self.plot_image("decibel_vs_frequency.png")

    def plot_line_chart(self):
        # Compute the frequency content and mean decibels
        frequencies, mean_decibels = self.compute_frequency_content()

        # Plot Line Chart
        plt.figure(figsize=(10, 4))
        plt.plot(frequencies, mean_decibels)
        plt.title('Decibels vs Frequency')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Mean Decibels')
        plt.tight_layout()
        plt.savefig("line_chart.png")

        # Display the line chart plot
        self.plot_image("line_chart.png")

    def compute_frequency_content(self):
        # Compute the frequency content and mean decibels
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
        frequencies = librosa.core.fft_frequencies(sr=44100)
        mean_decibels = np.mean(D, axis=1)
        return frequencies, mean_decibels

    def plot_frequency_ranges(self):
        # Compute the frequency content and mean decibels
        frequencies, mean_decibels = self.compute_frequency_content()

        # Define frequency ranges
        low_freq_range = (0, 1000)
        mid_freq_range = (1000, 5000)
        high_freq_range = (5000, 20000)

        # Filter frequencies and decibels for each range
        low_freq_indices = np.where((frequencies >= low_freq_range[0]) & (frequencies < low_freq_range[1]))[0]
        mid_freq_indices = np.where((frequencies >= mid_freq_range[0]) & (frequencies < mid_freq_range[1]))[0]
        high_freq_indices = np.where((frequencies >= high_freq_range[0]) & (frequencies < high_freq_range[1]))[0]

        low_freq_values = mean_decibels[low_freq_indices]
        mid_freq_values = mean_decibels[mid_freq_indices]
        high_freq_values = mean_decibels[high_freq_indices]

        # Plot Line Chart for Frequency Ranges
        plt.figure(figsize=(10, 4))

        plt.plot(frequencies[low_freq_indices], low_freq_values, label='Low Frequencies')
        plt.plot(frequencies[mid_freq_indices], mid_freq_values, label='Mid Frequencies')
        plt.plot(frequencies[high_freq_indices], high_freq_values, label='High Frequencies')

        plt.title('Decibels vs Frequency Ranges')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Mean Decibels')
        plt.legend()
        plt.tight_layout()
        plt.savefig("frequency_ranges_chart.png")

        # Display the line chart for frequency ranges
        self.plot_image("frequency_ranges_chart.png")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()