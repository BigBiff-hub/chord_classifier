import librosa, librosa.display, librosa.feature
import matplotlib.pyplot as plt
import numpy as np
import os

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
plt.show()

# Move From time domain to frequency domain
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
freq = np.linspace(0, sr, len(magnitude))
left_frequency = freq[:int(len(freq)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]
plt.plot(left_frequency, left_magnitude)
plt.xlabel("Freq (Hz)")
plt.ylabel("Magnitude")
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
plt.ylabel("MFCC")
plt.title("Mel Spectogram")
plt.colorbar()
plt.show()




