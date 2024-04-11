import numpy as np
import scipy.signal as signal

def magnitude_filter(audio_signal, noise_threshold):
    # Perform FFT to convert audio signal to frequency domain
    fft_signal = np.fft.fft(audio_signal)
    
    # Calculate magnitude spectrum
    magnitude_spectrum = np.abs(fft_signal)
    
    # Find the noise threshold
    max_magnitude = np.max(magnitude_spectrum)
    threshold = noise_threshold * max_magnitude
    
    # Filter out frequencies below the noise threshold
    filtered_spectrum = np.where(magnitude_spectrum < threshold, 0, magnitude_spectrum)
    
    # Reconstruct the filtered signal using inverse FFT
    filtered_signal = np.fft.ifft(filtered_spectrum).real
    
    return filtered_signal
