import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer")

        self.gif_label = Label(root)
        self.frame_count = 9
        self.gif_frames = [PhotoImage(file='music.gif', format='gif -index %i' % (i)) for i in range(self.frame_count)]

        self.file_path = None
        self.audio_data = None
        self.freq_plot = None

        self.plot_count = 6
        self.current_freq_plot = 0
        self.frequency_values = [250, 1000, 8000]

        self.create_widgets()

    def create_widgets(self):
        self.gif_label.pack()
        # Load File Button
        self.current_plot_label = tk.Label(self.root, text="Current Plot: None")
        self.current_plot_label.pack(pady=10)

        self.file_label = tk.Label(self.root, text="Current File: None")
        self.file_label.pack()

        load_button = tk.Button(self.root, text="Load File", command=self.load_file)
        load_button.pack()

        self.toggle_freq_button = tk.Button(root, text="Change Plot", command=self.show_next_plot)
        self.toggle_freq_button.pack(pady=10)

        self.show_spectrogram_button = tk.Button(root, text="Show spectrogram", command=self.show_spectrogram)
        self.show_spectrogram_button.pack()

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
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.aac")])

        # if the file is a valid process the file
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

        # check for more than one channel and remove them
        if audio_data[1].size > 1:
            audio_data = audio_data[:, 0]

        # normalize audio data so it can be better shown on a plot
        self.audio_data = audio_data / np.max(np.abs(audio_data), axis=0)
        duration = len(audio_data) / self.sample_rate

        # Display length
        self.length_label.config(text=f"Length: {duration:.2f} seconds")

        # Display plot
        self.display_plot()

    def display_plot(self):
        plot_name = self.get_plot_name(self.current_freq_plot)
        self.current_plot_label.config(text=f"Current Plot: {plot_name}")
        if self.current_freq_plot == 0:
            # selected plot is spectrogram
            self.plot_spectrogram()
            return
        if self.current_freq_plot == 1 or self.current_freq_plot == 2 or self.current_freq_plot == 3:
            # selected plot is low, mid, or high
            target = self.frequency_values[self.current_freq_plot - 1]
            self.update_figure_plot()
            self.plot_frequency(self.frequency_check(target), self.get_plot_color(self.current_freq_plot), plot_name)
            plt.show()
            return
        if self.current_freq_plot == 4:
            # selected plot is combined
            self.plot_combined()
            return
        if self.current_freq_plot == 5:
            # selected plot is waveform
            self.plot_waveform()
            return

    def get_plot_name(self, value):
        if value == 0: return "Spectrogram"
        if value == 1: return "Low"
        if value == 2: return "Mid"
        if value == 3: return "High"
        if value == 4: return "Combined"
        if value == 5: return "Waveform"

    def get_plot_color(self, value):
        if value == 0: return ""
        if value == 1: return "r"
        if value == 2: return "g"
        if value == 3: return "b"
        if value == 4: return ""
        if value == 5: return "b"

    def show_next_plot(self):
        if not self.file_path:
            return
        self.current_freq_plot = (self.current_freq_plot + 1) % self.plot_count
        self.display_plot()

    def show_spectrogram(self):
        if not self.file_path:
            return
        self.current_freq_plot = 0
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

    def update_figure_plot(self):
        # reuse the already shown plot or make a new one
        if self.freq_plot:
            self.freq_plot.clear()
        else:
            plt.figure()
            self.freq_plot = plt.subplot(111)

    def plot_spectrogram(self):
        self.update_figure_plot()

        self.spectrum, self.frequencies, self.time_values, _ = plt.specgram(self.audio_data, Fs=self.sample_rate,
                                              NFFT=1024, cmap=plt.get_cmap('autumn_r'))

        plt.show()

    def plot_frequency(self, db_data, color, label):
        plt.plot(self.time_values, db_data, linewidth=1, alpha=0.7, color=f'{color}', label=f'{label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (dB)')

        index_of_max = np.argmax(db_data)
        self.resonance_label.config(text=f"Resonance: {np.round(self.frequencies[index_of_max], 3)} Hz")

        value_of_max = db_data[index_of_max]
        plt.plot(self.time_values[index_of_max], value_of_max, 'go')

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

    def plot_combined(self):
        self.update_figure_plot()

        data = [self.frequency_check(self.frequency_values[0]), self.frequency_check(self.frequency_values[1]), self.frequency_check(self.frequency_values[2])]

        self.plot_frequency(data[0], self.get_plot_color(1), self.get_plot_name(1))
        self.plot_frequency(data[1], self.get_plot_color(2), self.get_plot_name(2))
        self.plot_frequency(data[2], self.get_plot_color(3), self.get_plot_name(3))
        plt.show()

    def plot_waveform(self):
        self.update_figure_plot()

        time = np.arange(0, len(self.audio_data)) / self.sample_rate
        plt.plot(time, self.audio_data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveform')
        plt.tight_layout()
        plt.show()

    def update_gif(self, ind):
        frame = self.gif_frames[ind]
        ind += 1
        if ind == self.frame_count:
            ind = 0
        self.gif_label.configure(image=frame)
        root.after(100, self.update_gif, ind)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.after(0, app.update_gif, 0)
    root.mainloop()