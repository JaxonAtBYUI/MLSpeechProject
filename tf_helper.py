import numpy as np
import tensorflow as tf


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def get_spectrogram(waveform):

  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)

  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


def preprocess_audiobuffer(waveform):
    
    waveform =  waveform / 32768

    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    spectogram = get_spectrogram(waveform)
    
    # add one dimension
    spectogram = tf.expand_dims(spectogram, 0)
    
    return spectogram