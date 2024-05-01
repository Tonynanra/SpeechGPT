import numpy as np
import scipy.io.wavfile as wavfile
from denoise import denoise
from zeropad import zeropad
from normalize import normalize

sample_rate, audio_signal = wavfile.read(".wav")

denoised_signal = denoise(audio_signal, noise_threshold=0.01)
padded_signal = zeropad(denoised_signal, sample_rate)
normalized_signal = normalize(padded_signal, sample_rate)

# Save the processed audio
wavfile.write("processed_audio.wav", sample_rate, normalized_signal.astype(np.int16))
