pip install librosa

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def compute_mel_spectrogram(audio_file, normalize=True):
    """
    Computes the mel spectrogram of an audio file.

    :param audio_file: Path to the audio file.
    :param normalize: Whether to normalize the mel spectrogram.
    :return: Mel spectrogram (as a NumPy array).
    """
    # Load the audio file
    y, sr = librosa.load(audio_file)
    
    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, fmax=8000)
    
    # Convert to dB
    S_dB = librosa.power_to_db(S, ref=np.max)

    return S_dB

def plot_mel_spectrogram(S_dB):
    """
    Plots the mel spectrogram.

    :param S_dB: Mel spectrogram (as a NumPy array).
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()