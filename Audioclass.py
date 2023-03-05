import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

# Define paths to files


POS = os.path.join('Data', 'Piano_C_clips')
NEG = os.path.join('Data', 'Not_Piano_C_Clips')


# Data Loading function
# Convert to 16KHz and single Channel
# Load encoded wav file
def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


pos = tf.data.Dataset.list_files(POS + '\*.wav')
neg = tf.data.Dataset.list_files(NEG + '\*.wav')

# Adding labels + combine positive and negative -rebalanced or have less negative sample to improve performance going
# forward 1 - flag for piano c note 0 - flag for a non piano c note

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

lengths = []
for file in os.listdir(os.path.join('Data', 'Piano_C_clips')):
    tensor_wave = load_wav_16k_mono(os.path.join('Data', 'Piano_C_clips', file))
    lengths.append(len(tensor_wave))

print(f' Max {tf.math.reduce_max(lengths)}')
print(f' Average {tf.math.reduce_mean(lengths)}')
print(f' Min {tf.math.reduce_min(lengths)}')


def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:24000]
    zero_padding = tf.zeros([24000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
plt.figure(figsize=(30, 20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()
