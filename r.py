import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import librosa
from pydub import AudioSegment

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer")

        self.file_label = tk.Label(root, text="No file selected")
        self.file_label.pack()

        self.load_button = tk.Button(root, text="Load File", command=self.load_file)
        self.load_button.pack()

        # self.process_button = tk.Button(root, text="Process File", command=self.process_file)
        # self.process_button.pack()

        self.freq_button = tk.Button(root, text="Alternate Frequency Plot", command=self.toggle_freq_plot)
        self.freq_button.pack()

        self.combine_button = tk.Button(root, text="Combine Plots", command=self.combine_plots)
        self.combine_button.pack()

        self.status_label = tk.Label(root, text="")
        self.status_label.pack()

        self.filename = None
        self.waveform_plot = None
        self.freq_plot = None
        self.current_freq_plot = 0
        # self.file_processed = False

    def load_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav")])
        self.file_label.config(text=f"File: {os.path.basename(self.filename)}")
        # self.file_processed = False

        self.process_file()
        self.toggle_freq_plot()

    def process_file(self):
        if self.filename:
            # Check if the file is a .wav file, if not, convert it
            if not self.filename.lower().endswith(".wav"):
                converted_filename = self.convert_to_wav()
                self.filename = converted_filename

            # Load the .wav file
            sample_rate, audio_data = wavfile.read(self.filename)

            # Check for meta tags and remove them
            # (Note: This is a placeholder, you may need a more specific implementation based on your requirements)
            audio_data = self.remove_meta_tags(audio_data)

            # Display the title value in seconds
            duration = librosa.get_duration(y=audio_data, sr=sample_rate)
            self.status_label.config(text=f"Duration: {duration:.2f} seconds")

            # Display the waveform
            self.display_waveform(audio_data, sample_rate)

            # Compute and display the resonance frequencies
            resonance_freq = self.compute_resonance_frequency(audio_data, sample_rate)
            self.status_label.config(text=f"Resonance Frequency: {resonance_freq:.2f} Hz")

            # Compute and display RT60 for low, mid, and high frequencies
            rt60_low, rt60_mid, rt60_high = self.compute_rt60(audio_data, sample_rate)
            self.status_label.config(text=f"RT60 Low: {rt60_low:.2f} s, RT60 Mid: {rt60_mid:.2f} s, RT60 High: {rt60_high:.2f} s")

            self.freq_plot = None  # Reset freq_plot
            # self.file_processed = True

    def convert_to_wav(self):
        # Convert to .wav format using pydub
        audio = AudioSegment.from_file(self.filename)
        converted_filename = self.filename.replace(os.path.splitext(self.filename)[1], ".wav")
        audio.export(converted_filename, format="wav")
        return converted_filename

    def remove_meta_tags(self, audio_data):
        # Placeholder for removing meta tags, modify as needed
        return audio_data

    def display_waveform(self, audio_data, sample_rate):
        time = np.arange(0, len(audio_data)) / sample_rate
        if self.waveform_plot:
            self.waveform_plot.clear()
        else:
            plt.figure()
            self.waveform_plot = plt.subplot(111)
        self.waveform_plot.plot(time, audio_data)
        self.waveform_plot.set_xlabel('Time (s)')
        self.waveform_plot.set_ylabel('Amplitude')
        self.waveform_plot.set_title('Waveform')
        plt.show()

    def compute_resonance_frequency(self, audio_data, sample_rate):
        # Placeholder for computing resonance frequency, modify as needed
        return np.random.uniform(50, 200)

    def compute_rt60(self, audio_data, sample_rate):
        # Placeholder for computing RT60 for low, mid, and high frequencies, modify as needed
        return np.random.uniform(0.1, 0.5), np.random.uniform(0.5, 1.0), np.random.uniform(1.0, 2.0)

    def toggle_freq_plot(self):
        self.current_freq_plot = (self.current_freq_plot + 1) % 3
        self.display_freq_plot()
        # if self.file_processed:
        #     self.current_freq_plot = (self.current_freq_plot + 1) % 3
        #     self.display_freq_plot()
        # else:
        #     self.status_label.config(text="Please process the file first.")

    def display_freq_plot(self):
        if self.waveform_plot:
            freq_range = ["Low", "Mid", "High"]
            freq_label = freq_range[self.current_freq_plot]

            # Placeholder for displaying frequency plot
            # You should replace this with your actual frequency analysis code
            freq_data = np.random.rand(100)  # Replace with your frequency data
            freq_axis = np.linspace(0, 1, len(freq_data))

            if self.freq_plot:
                self.freq_plot.clear()
            else:
                plt.figure()
                self.freq_plot = plt.subplot(111)

            self.freq_plot.plot(freq_axis, freq_data)
            self.freq_plot.set_xlabel('Frequency')
            self.freq_plot.set_ylabel('Amplitude')
            self.freq_plot.set_title(f'{freq_label} Frequency Plot')
            plt.show()

    def combine_plots(self):
        if self.waveform_plot and self.freq_plot:
            # Placeholder for combining plots
            # You should replace this with your actual combination code
            plt.figure()
            combined_plot = plt.subplot(111)

            # Plot waveform
            combined_plot.plot(self.waveform_plot.get_lines()[0].get_xdata(), self.waveform_plot.get_lines()[0].get_ydata(), label='Waveform')

            # Plot frequency data
            freq_data = np.random.rand(100)  # Replace with your frequency data
            freq_axis = np.linspace(0, 1, len(freq_data))
            combined_plot.plot(freq_axis, freq_data, label='Frequency Plot')

            combined_plot.set_xlabel('Time/Frequency')
            combined_plot.set_ylabel('Amplitude')
            combined_plot.set_title('Combined Plot')
            combined_plot.legend()
            plt.show()
        else:
            self.status_label.config(text="Please process the file first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()