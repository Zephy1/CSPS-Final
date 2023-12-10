This Python application is designed to analyze and visualize audio files. It provides a user-friendly interface using Tkinter and incorporates various libraries for audio processing, plotting, and analysis. The application supports loading WAV files, processing them, and displaying different types of plots to aid in understanding the characteristics of the audio.

Features
Load and Process Audio Files

Click the "Load Audio" button to open a file dialog and select an audio file (supported formats: WAV, MP3, AAC).
The selected file will be displayed, indicating a successful load.

Audio Processing:
The application processes the audio file, ensuring it is in WAV format. If not, it converts the file to WAV using the Pydub library.
Information about the resonance frequency, RT60 times, and the duration of the file are then displayed in the GUI.

Plot Audio:

Click the "Plot" button to generate informative plots and display relevant information about the audio file.
The application calculates and prints the total time of the audio, the frequency of the greatest amplitude, and the RT60 (reverberation time) for low, mid, and high frequencies.
Users can cycle through different plots, including Spectrogram, Low Frequency, Mid Frequency, High Frequency, Combined Frequency, and Waveform.

Dependencies
The following Python libraries are required to run the application:

numpy
matplotlib
scipy
pydub
You can install these dependencies using the following command:
python -m pip install numpy matplotlib scipy pydub
or
python -m pip install -r requirements.txt

How to Run
To execute the Audio Analyzer, run the provided Python script. Ensure that you have the necessary dependencies installed. The application will open a graphical user interface, allowing you to interact with the audio analysis tools.

python -m audio_analyzer

Notes
The application uses a GIF image ("music.gif") for visual elements. Ensure the image file is present in the same directory as the script and named properly.

Credits:
 - The music GIF:
   https://wifflegif.com/gifs/706374-music-notes-pixel-art-gif

 - Stackoverflow posts:
  https://stackoverflow.com/questions/23154400/read-the-data-of-a-single-channel-from-a-stereo-wave-file-in-python
  https://stackoverflow.com/questions/28518072/play-animations-in-gif-with-tkinter
  