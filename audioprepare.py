import librosa, librosa.display, librosa.feature
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import stft
from scipy.io import wavfile

file = "Dm_RockGB_JO_3.wav"
DATA_PATH = "Training"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 4 #seconds

# load audio file with Librosa
signal, sr = librosa.load(file, sr=22050) # signal is a 1D  numpy array of each value of the magnitude per sample
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title('Waveform of Dm')
plt.show()

# Move From time domain to frequency domain
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
freq = np.linspace(0, sr, len(magnitude))
left_frequency = freq[:int(len(freq)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]
plt.plot(left_frequency, left_magnitude, 'r')
plt.xlabel("Freq (Hz)")
plt.ylabel("Magnitude")
plt.title('Note Dm in Frequency Domain')
plt.show()

# Short Time Fourier Transform -> Spectogram
n_fft = 2048
hop_length = 512 # amount each transform is shifted to the right
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectogram = np.abs(stft)
log_spec = librosa.amplitude_to_db(spectogram)
librosa.display.specshow(log_spec, sr=sr, hop_length=hop_length)
plt.xlabel("Time (s)")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

# MFCC
MFCCS = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCS, sr=sr, hop_length=hop_length)
plt.xlabel("Time (s)")
plt.ylabel("Mel Scale")
plt.title("Mel Spectogram For note Dm")
plt.colorbar()
plt.show()


file = "Dm_RockGB_JO_3.wav"
y, sr = librosa.load(file)

# Extract chroma features
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Plot chroma features
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chroma Feature')
plt.tight_layout()
plt.show()

signal, sr = librosa.load(file)

# Calculate zero-crossing rate
zcr = librosa.feature.zero_crossing_rate(signal)

# Print the mean ZCR
print('Mean ZCR:', zcr.mean())

# Load audio file
fs, audio = wavfile.read(file)

# Set STFT parameters
window = 'hann' # Window type
nperseg = 256  # Length of each segment
noverlap = 128  # Overlap between segments


