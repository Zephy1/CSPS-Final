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
        self.audio_data = None
        self.sample_rate = 0

        self.waveform_plot = None
        self.freq_plot = None
        self.spec_plot = None
        self.combined_plot = None

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
            self.sample_rate, self.audio_data = wavfile.read(self.filename)

            # Check for meta tags and remove them
            # (Note: This is a placeholder, you may need a more specific implementation based on your requirements)
            self.audio_data = self.remove_meta_tags(self.audio_data)

            # Normalize audio data to the range [-1, 1]
            self.audio_data = self.audio_data / np.max(np.abs(self.audio_data), axis=0)

            # Display the title value in seconds
            duration = librosa.get_duration(y=self.audio_data, sr=self.sample_rate)
            self.status_label.config(text=f"Duration: {duration:.2f} seconds")

            # Display the waveform
            self.display_waveform(self.audio_data, self.sample_rate)

            # Compute and display the resonance frequencies
            resonance_freq = self.compute_resonance_frequency(self.audio_data, self.sample_rate)
            self.status_label.config(text=f"Resonance Frequency: {resonance_freq:.2f} Hz")

            # Compute and display RT60 for low, mid, and high frequencies
            _, rt60_low, rt60_mid, rt60_high = self.compute_rt60(self.audio_data, self.sample_rate)
            self.status_label.config(
                text=f"RT60 Low: {rt60_low:.2f} s, RT60 Mid: {rt60_mid:.2f} s, RT60 High: {rt60_high:.2f} s")

            self.freq_plot = None  # Reset freq_plot

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
        # Compute FFT
        fft_result = np.fft.fft(audio_data)

        # Get the corresponding frequencies
        frequencies = np.fft.fftfreq(len(fft_result), 1 / sample_rate)

        # Find the index of the maximum amplitude
        peak_index = np.argmax(np.abs(fft_result))

        # Convert index to frequency
        resonance_freq = np.abs(frequencies[peak_index])

        return resonance_freq

    def compute_rt60(self, audio_data, sample_rate):
        # Compute the short-time Fourier transform (STFT)
        Zxx = np.abs(librosa.stft(audio_data, hop_length=512, n_fft=1024))

        # Define frequency bands (you may adjust these as needed)
        low_band = (20, 200)  # Low frequencies
        mid_band = (200, 2000)  # Mid frequencies
        high_band = (2000, 8000)  # High frequencies

        # Compute the energy in each frequency band
        energy_low = np.sum(
            np.abs(Zxx[(librosa.time_to_frames(2), librosa.time_to_frames(4))]))  # Adjust time frames as needed
        energy_mid = np.sum(
            np.abs(Zxx[(librosa.time_to_frames(2), librosa.time_to_frames(4))]))  # Adjust time frames as needed
        energy_high = np.sum(
            np.abs(Zxx[(librosa.time_to_frames(2), librosa.time_to_frames(4))]))  # Adjust time frames as needed

        # Compute RT60 using the energy in each band
        rt60_low = -60 / np.mean(20 * np.log10(energy_low / np.max(energy_low)))
        rt60_mid = -60 / np.mean(20 * np.log10(energy_mid / np.max(energy_mid)))
        rt60_high = -60 / np.mean(20 * np.log10(energy_high / np.max(energy_high)))

        return Zxx, rt60_low, rt60_mid, rt60_high

    def toggle_freq_plot(self):
        self.current_freq_plot = (self.current_freq_plot + 1) % 3
        self.display_freq_plot()
        # if self.file_processed:
        #     self.current_freq_plot = (self.current_freq_plot + 1) % 3
        #     self.display_freq_plot()
        # else:
        #     self.status_label.config(text="Please process the file first.")

    def display_freq_plot(self):
        # if self.file_processed:
        freq_range = ["Low", "Mid", "High"]
        freq_label = freq_range[self.current_freq_plot]

        # Compute STFT using librosa
        Zxx, rt60_low, rt60_mid, rt60_high = self.compute_rt60(self.audio_data, len(self.audio_data))

        make_new_plot = False
        if self.freq_plot:
            self.freq_plot.clear()
        else:
            make_new_plot = True
            plt.figure()
            self.freq_plot = plt.subplot(211)

            # Display spectrogram
            librosa.display.specshow(librosa.amplitude_to_db(Zxx, ref=np.max), y_axis='log', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram of Waveform')

        # Display frequency plot
        if make_new_plot:
            self.freq_plot = plt.subplot(212)
            freq_data = np.mean(Zxx, axis=1)  # Extract mean amplitude along the time axis
            freq_axis = librosa.fft_frequencies(sr=len(self.audio_data), n_fft=1024)
            plt.plot(freq_axis, freq_data, label=f'{freq_label} Frequency Plot')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title(f'{freq_label} Frequency Plot')

        # Display RT60 values
        # plt.axvline(rt60_low, color='r', linestyle='--', label=f'RT60 Low: {rt60_low:.2f} s')
        # plt.axvline(rt60_mid, color='g', linestyle='--', label=f'RT60 Mid: {rt60_mid:.2f} s')
        # plt.axvline(rt60_high, color='b', linestyle='--', label=f'RT60 High: {rt60_high:.2f} s')

        plt.legend()
        plt.tight_layout()
        plt.show()
        # else:
        #     self.status_label.config(text="Please load and process the file first.")

    def combine_plots(self):
        if self.waveform_plot and self.freq_plot:
            # Placeholder for combining plots
            # You should replace this with your actual combination code
            if self.combined_plot:
                self.combined_plot.clear()
            else:
                plt.figure()
                self.combined_plot = plt.subplot(111)

            # Plot waveform
            self.combined_plot.plot(self.waveform_plot.get_lines()[0].get_xdata(), self.waveform_plot.get_lines()[0].get_ydata(), label='Waveform')

            # Plot frequency data
            freq_data = np.random.rand(100)  # Replace with your frequency data
            freq_axis = np.linspace(0, 1, len(freq_data))
            self.combined_plot.plot(freq_axis, freq_data, label='Frequency Plot')

            self.combined_plot.set_xlabel('Time/Frequency')
            self.combined_plot.set_ylabel('Amplitude')
            self.combined_plot.set_title('Combined Plot')
            self.combined_plot.legend()
            plt.show()
        else:
            self.status_label.config(text="Please process the file first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()