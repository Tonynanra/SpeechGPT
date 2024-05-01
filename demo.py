import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from denoise import denoise
from zeropad import zeropad
from normalize import normalize

sample_rate, audio_signal = wavfile.read("0a0b46ae_nohash_0.wav")

fig, axes = plt.subplots(4, 1, figsize=(12, 12))  # Create 4 subplots
times = np.arange(len(audio_signal)) / sample_rate

axes[0].plot(times, audio_signal)
axes[0].set_title('Original Signal')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')

denoised_signal = denoise(audio_signal, noise_threshold=0.01)
times_denoised = np.arange(len(denoised_signal)) / sample_rate  # Time axis for the denoised signal

# Plot the denoised signal
axes[1].plot(times_denoised, denoised_signal)
axes[1].set_title('Denoised Signal')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude')

padded_signal = zeropad(denoised_signal, sample_rate)
times_padded = np.arange(len(padded_signal)) / sample_rate  # Time axis for the padded signal

# Plot the zero-padded signal
axes[2].plot(times_padded, padded_signal)
axes[2].set_title('Zero-Padded Signal')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Amplitude')
normalized_signal = normalize(padded_signal, sample_rate)
times_normalized = np.arange(len(normalized_signal)) / sample_rate  # Time axis for the normalized signal

# Plot the normalized signal
axes[3].plot(times_normalized, normalized_signal)
axes[3].set_title('Normalized Signal')
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('Amplitude')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()