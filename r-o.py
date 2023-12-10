import os
import tkinter as tk
from tkinter import filedialog
import librosa
import matplotlib.pyplot as plt
import numpy as np
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

        # self.freq_button = tk.Button(root, text="Alternate Frequency Plot", command=self.toggle_freq_plot)
        # self.freq_button.pack()
        #
        # self.combine_button = tk.Button(root, text="Combine Plots", command=self.combine_plots)
        # self.combine_button.pack()

        self.status_label = tk.Label(root, text="")
        self.status_label.pack()

        self.filename = None
        self.audio_data = None
        self.sample_rate = 10000

        self.waveform_plot = None
        self.freq_plot = None
        self.spec_plot = None
        self.combined_plot = None

        self.current_freq_plot = 0
        # self.file_processed = False

    def load_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3;*.aac")])
        self.file_label.config(text=f"File: {os.path.basename(self.filename)}")

        self.process_file()
        # self.toggle_freq_plot()

    def process_file(self):
        if self.filename:
            # Check if the file is a .wav file, if not, convert it
            if not self.filename.lower().endswith(".wav"):
                converted_filename = self.convert_to_wav()
                self.filename = converted_filename

            # Load the audio file using librosa
            self.audio_data, self.sample_rate = librosa.load(self.filename, mono=True, sr=None)

            # Check for meta tags and remove them
            # (Note: This is a placeholder, you may need a more specific implementation based on your requirements)
            self.audio_data = self.remove_meta_tags(self.audio_data)

            # Normalize audio data to the range [-1, 1]
            self.audio_data /= np.max(np.abs(self.audio_data))

            # Display the title value in seconds
            duration = librosa.get_duration(y=self.audio_data, sr=self.sample_rate)
            self.status_label.config(text=f"Duration: {duration:.2f} seconds")

            # Display the waveform
            # self.display_waveform(self.audio_data, self.sample_rate)

            # Compute and display the resonance frequencies
            resonance_freq = self.compute_resonance_frequency(self.audio_data, self.sample_rate)
            self.status_label.config(text=f"Resonance Frequency: {resonance_freq:.2f} Hz")

            # Compute and display RT60 for low, mid, and high frequencies
            _, rt60_low, rt60_mid, rt60_high = self.compute_rt60(self.audio_data, self.sample_rate)
            self.status_label.config(
                text=f"RT60 Low: {rt60_low:.2f} s, RT60 Mid: {rt60_mid:.2f} s, RT60 High: {rt60_high:.2f} s")

            # Display six plots
            self.display_six_plots()

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

    # def display_six_plots(self):
    #     # Plot waveform
    #     self.display_waveform(self.audio_data, self.sample_rate)
    #
    #     # Plot RT60 for low, mid, and high frequencies
    #     _, rt60_low, rt60_mid, rt60_high = self.compute_rt60(self.audio_data, self.sample_rate)
    #     self.plot_rt60(rt60_low, "Low Frequency RT60", 3)
    #     self.plot_rt60(rt60_mid, "Mid Frequency RT60", 4)
    #     self.plot_rt60(rt60_high, "High Frequency RT60", 5)
    #
    #     # Plot an additional waveform, you can choose another plot type if desired
    #     self.display_waveform(self.audio_data, self.sample_rate)
    def display_six_plots(self):
        # Create a new figure
        plt.figure()

        # Plot waveform
        self.display_waveform(self.audio_data, self.sample_rate, subplot_index=1)

        # Plot RT60 for low, mid, and high frequencies
        # _, rt60_low, rt60_mid, rt60_high = self.compute_rt60(self.audio_data, self.sample_rate)
        # self.plot_rt60(rt60_low, "Low Frequency RT60", 3)
        # self.plot_rt60(rt60_mid, "Mid Frequency RT60", 4)
        # self.plot_rt60(rt60_high, "High Frequency RT60", 5)
        self.display_frequency_content(self.audio_data, self.sample_rate, low_band=(20, 200), subplot_index=2,
                                       label='Low Frequency Content')
        self.display_frequency_content(self.audio_data, self.sample_rate, mid_band=(200, 2000), subplot_index=3,
                                       label='Mid Frequency Content')
        self.display_frequency_content(self.audio_data, self.sample_rate, high_band=(2000, 8000), subplot_index=4,
                                       label='High Frequency Content')

        # Plot an additional waveform, you can choose another plot type if desired
        # self.display_waveform(self.audio_data, self.sample_rate, subplot_index=6)

        # Show the plots
        # plt.tight_layout()
        plt.show()

    def display_frequency_content(self, audio_data, sample_rate, low_band=None, mid_band=None, high_band=None,
                                  subplot_index=None, label=None):
        Zxx = np.abs(librosa.stft(audio_data, hop_length=512, n_fft=self.sample_rate))

        if low_band:
            low_energy = np.sum(np.abs(Zxx[low_band[0]:low_band[1], :]), axis=0)
            plt.subplot(3, 3, subplot_index)
            self.plot_frequency_content(low_energy, sample_rate, label)
            plt.title(f'Frequency Content - {label}')

        if mid_band:
            mid_energy = np.sum(np.abs(Zxx[mid_band[0]:mid_band[1], :]), axis=0)
            plt.subplot(3, 3, subplot_index + 1)
            self.plot_frequency_content(mid_energy, sample_rate, label)
            plt.title(f'Frequency Content - {label}')

        if high_band:
            high_energy = np.sum(np.abs(Zxx[high_band[0]:high_band[1], :]), axis=0)
            plt.subplot(3, 3, subplot_index + 2)
            self.plot_frequency_content(high_energy, sample_rate, label)
            plt.title(f'Frequency Content - {label}')

    def plot_frequency_content(self, energy, sample_rate, label):
        frequency_axis = librosa.fft_frequencies(sr=sample_rate, n_fft=self.sample_rate)
        energy = np.pad(energy, (0, len(frequency_axis) - len(energy)), mode='constant', constant_values=0)
        plt.plot(frequency_axis, energy, label=label)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Energy')
        plt.legend()

    def plot_rt60(self, rt60_values, title, subplot_index):
        plt.subplot(3, 2, subplot_index)
        plt.plot(rt60_values)
        plt.title(title)
        plt.xlabel("Frequency Band")
        plt.ylabel("RT60 (s)")
        plt.tight_layout()

    # def display_waveform(self, audio_data, sample_rate):
    #     time = np.arange(0, len(audio_data)) / sample_rate
    #     if self.waveform_plot:
    #         self.waveform_plot.clear()
    #     else:
    #         plt.figure()
    #         self.waveform_plot = plt.subplot(111)
    #     self.waveform_plot.plot(time, audio_data)
    #     self.waveform_plot.set_xlabel('Time (s)')
    #     self.waveform_plot.set_ylabel('Amplitude')
    #     self.waveform_plot.set_title('Waveform')
    #     plt.show()
    def display_waveform(self, audio_data, sample_rate, subplot_index=1):
        time = np.arange(0, len(audio_data)) / sample_rate
        plt.subplot(3, 2, subplot_index)  # Create a subplot within the existing figure
        plt.plot(time, audio_data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveform')
        plt.tight_layout()

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
        Zxx = np.abs(librosa.stft(audio_data, hop_length=512, n_fft=self.sample_rate))

        # Define frequency bands (you may adjust these as needed)
        low_band = (20, 200)  # Low frequencies
        mid_band = (200, 2000)  # Mid frequencies
        high_band = (2000, 8000)  # High frequencies

        # Compute the energy in each frequency band
        frames_2s = librosa.time_to_frames(2, sr=sample_rate, hop_length=512)
        frames_4s = librosa.time_to_frames(4, sr=sample_rate, hop_length=512)

        energy_low = np.sum(np.abs(Zxx[low_band[0]:low_band[1], frames_2s:frames_4s]))
        energy_mid = np.sum(np.abs(Zxx[mid_band[0]:mid_band[1], frames_2s:frames_4s]))
        energy_high = np.sum(np.abs(Zxx[high_band[0]:high_band[1], frames_2s:frames_4s]))

        # Compute RT60 using the energy in each band
        rt60_low = -60 / np.mean(20 * np.log10(energy_low / np.max(energy_low)))
        rt60_mid = -60 / np.mean(20 * np.log10(energy_mid / np.max(energy_mid)))
        rt60_high = -60 / np.mean(20 * np.log10(energy_high / np.max(energy_high)))

        return Zxx, rt60_low, rt60_mid, rt60_high

    # def toggle_freq_plot(self):
    #     self.current_freq_plot = (self.current_freq_plot + 1) % 3
    #     self.display_freq_plot()
    #     # if self.file_processed:
    #     #     self.current_freq_plot = (self.current_freq_plot + 1) % 3
    #     #     self.display_freq_plot()
    #     # else:
    #     #     self.status_label.config(text="Please process the file first.")

    # def display_freq_plot(self):
    #     # if self.file_processed:
    #     freq_range = ["Low", "Mid", "High"]
    #     freq_label = freq_range[self.current_freq_plot]
    #
    #     # Compute STFT using librosa
    #     Zxx, rt60_low, rt60_mid, rt60_high = self.compute_rt60(self.audio_data, len(self.audio_data))
    #
    #     make_new_plot = False
    #     if self.freq_plot:
    #         self.freq_plot.clear()
    #     else:
    #         make_new_plot = True
    #         plt.figure()
    #         self.freq_plot = plt.subplot(211)
    #
    #         # Display spectrogram
    #         librosa.display.specshow(librosa.amplitude_to_db(Zxx, ref=np.max), y_axis='log', x_axis='time')
    #         plt.colorbar(format='%+2.0f dB')
    #         plt.title('Spectrogram of Waveform')
    #
    #     # Display frequency plot
    #     if make_new_plot:
    #         self.freq_plot = plt.subplot(212)
    #         freq_data = np.mean(Zxx, axis=1)  # Extract mean amplitude along the time axis
    #         freq_axis = librosa.fft_frequencies(sr=len(self.audio_data), n_fft=1024)
    #         plt.plot(freq_axis, freq_data, label=f'{freq_label} Frequency Plot')
    #         plt.xlabel('Frequency (Hz)')
    #         plt.ylabel('Amplitude')
    #         plt.title(f'{freq_label} Frequency Plot')
    #
    #     # Display RT60 values
    #     # plt.axvline(rt60_low, color='r', linestyle='--', label=f'RT60 Low: {rt60_low:.2f} s')
    #     # plt.axvline(rt60_mid, color='g', linestyle='--', label=f'RT60 Mid: {rt60_mid:.2f} s')
    #     # plt.axvline(rt60_high, color='b', linestyle='--', label=f'RT60 High: {rt60_high:.2f} s')
    #
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    #     # else:
    #     #     self.status_label.config(text="Please load and process the file first.")
    #
    # def combine_plots(self):
    #     if self.waveform_plot and self.freq_plot:
    #         # Placeholder for combining plots
    #         # You should replace this with your actual combination code
    #         if self.combined_plot:
    #             self.combined_plot.clear()
    #         else:
    #             plt.figure()
    #             self.combined_plot = plt.subplot(111)
    #
    #         # Plot waveform
    #         self.combined_plot.plot(self.waveform_plot.get_lines()[0].get_xdata(),
    #                                 self.waveform_plot.get_lines()[0].get_ydata(), label='Waveform')
    #
    #         # Plot frequency data
    #         freq_data = np.random.rand(100)  # Replace with your frequency data
    #         freq_axis = np.linspace(0, 1, len(freq_data))
    #         self.combined_plot.plot(freq_axis, freq_data, label='Frequency Plot')
    #
    #         self.combined_plot.set_xlabel('Time/Frequency')
    #         self.combined_plot.set_ylabel('Amplitude')
    #         self.combined_plot.set_title('Combined Plot')
    #         self.combined_plot.legend()
    #         plt.show()
    #     else:
    #         self.status_label.config(text="Please process the file first.")

    # def plot_waveform(self):
    #     plt.figure(figsize=(10, 4))
    #     time = np.arange(0, len(self.audio_data)) / self.sample_rate
    #     plt.plot(time, self.audio_data)
    #     plt.title('Waveform')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Amplitude')
    #     plt.tight_layout()
    #     plt.savefig("waveform.png")
    #
    #     # Display the waveform plot
    #     self.plot_image("waveform.png")
    #
    # def compute_resonance_frequency(self):
    #     # Your resonance frequency computation logic here
    #     # This is just a placeholder, replace with your actual implementation
    #     return np.random.uniform(50, 1000)
    #
    # def plot_image(self, image_path):
    #     img = tk.PhotoImage(file=image_path)
    #     panel = tk.Label(self.plot_frame, image=img)
    #     panel.image = img
    #     panel.pack()
    #
    # def plot_decibel_vs_frequency(self):
    #     plt.figure(figsize=(10, 4))
    #     D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
    #     librosa.display.specshow(D, sr=44100, x_axis='time', y_axis='log')
    #     plt.colorbar(format='%+2.0f dB')
    #     plt.title('Spectrogram')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Frequency (Hz)')
    #     plt.tight_layout()
    #     plt.savefig("decibel_vs_frequency.png")
    #
    #     # Display the decibel vs frequency plot
    #     self.plot_image("decibel_vs_frequency.png")
    #
    # def plot_line_chart(self):
    #     # Compute the frequency content and mean decibels
    #     frequencies, mean_decibels = self.compute_frequency_content()
    #
    #     # Plot Line Chart
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(frequencies, mean_decibels)
    #     plt.title('Decibels vs Frequency')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Mean Decibels')
    #     plt.tight_layout()
    #     plt.savefig("line_chart.png")
    #
    #     # Display the line chart plot
    #     self.plot_image("line_chart.png")
    #
    # def compute_frequency_content(self):
    #     # Compute the frequency content and mean decibels
    #     D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
    #     frequencies = librosa.core.fft_frequencies(sr=44100)
    #     mean_decibels = np.mean(D, axis=1)
    #     return frequencies, mean_decibels
    #
    # def plot_frequency_ranges(self):
    #     # Compute the frequency content and mean decibels
    #     frequencies, mean_decibels = self.compute_frequency_content()
    #
    #     # Define frequency ranges
    #     low_freq_range = (0, 1000)
    #     mid_freq_range = (1000, 5000)
    #     high_freq_range = (5000, 20000)
    #
    #     # Filter frequencies and decibels for each range
    #     low_freq_indices = np.where((frequencies >= low_freq_range[0]) & (frequencies < low_freq_range[1]))[0]
    #     mid_freq_indices = np.where((frequencies >= mid_freq_range[0]) & (frequencies < mid_freq_range[1]))[0]
    #     high_freq_indices = np.where((frequencies >= high_freq_range[0]) & (frequencies < high_freq_range[1]))[0]
    #
    #     low_freq_values = mean_decibels[low_freq_indices]
    #     mid_freq_values = mean_decibels[mid_freq_indices]
    #     high_freq_values = mean_decibels[high_freq_indices]
    #
    #     # Plot Line Chart for Frequency Ranges
    #     plt.figure(figsize=(10, 4))
    #
    #     plt.plot(frequencies[low_freq_indices], low_freq_values, label='Low Frequencies')
    #     plt.plot(frequencies[mid_freq_indices], mid_freq_values, label='Mid Frequencies')
    #     plt.plot(frequencies[high_freq_indices], high_freq_values, label='High Frequencies')
    #
    #     plt.title('Decibels vs Frequency Ranges')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Mean Decibels')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("frequency_ranges_chart.png")
    #
    #     # Display the line chart for frequency ranges
    #     self.plot_image("frequency_ranges_chart.png")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()