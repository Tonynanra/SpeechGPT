# test
from mel import compute_mel_spectrogram
from zeropad import zeropad
from denoise import magnitude_filter
from normalize import normalize_audio
import os
print("test")

# PREPROCESSING FUNCTIONS

# Padding function
# Input: table of {audio file, label}, length to pad to
# Output: table of {padded audio file, label}
# Iterates through all the audio files and right-pads them with 0 to a length of [seconds]

# Amplitude normalization
# Input: audio file
# Output: audio file
# Normalizes amplitude of sound

# Noise Filter
# Input: audio file, threshold
# Output: audio file
# Eliminiates frequencies with a magnitude below [some threshold]
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft

def filter_noise(wav_file_path, threshold_magnitude):
    # Step 1: Read the WAV file
    sample_rate, data = wavfile.read(wav_file_path)
    
    # Assuming the audio is stereo, we need to handle both channels
    if data.ndim == 2:
        left_channel = data[:, 0]
        right_channel = data[:, 1]
    else:
        # For mono, both channels are the same
        left_channel = right_channel = data
    
    # Step 2: Fourier Transform for both channels
    left_fft = fft(left_channel)
    right_fft = fft(right_channel)
    
    # Step 3: Filter frequencies in both channels
    left_fft_filtered = np.where(np.abs(left_fft) < threshold_magnitude, 0, left_fft)
    right_fft_filtered = np.where(np.abs(right_fft) < threshold_magnitude, 0, right_fft)
    
    # Step 4: Inverse Fourier Transform
    left_channel_filtered = ifft(left_fft_filtered).astype(data.dtype)
    right_channel_filtered = ifft(right_fft_filtered).astype(data.dtype)
    
    # Merge channels back for stereo audio
    if data.ndim == 2:
        modified_data = np.column_stack((left_channel_filtered, right_channel_filtered))
    else:
        modified_data = left_channel_filtered
    
    # Step 5: Write the modified audio data back to a WAV file
    output_file_path = "filtered_" + wav_file_path
    wavfile.write(output_file_path, sample_rate, modified_data)
    
    return output_file_path

# Example usage
# filter_noise("path_to_your_audio_file.wav", threshold_magnitude=1000)


# Band Pass?
# Eliminate frequencies outside human hearing
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

def band_pass_filter(wav_file_path):
    # Human hearing constants
    LOW_FREQ = 20      # 20 Hz
    HIGH_FREQ = 20000  # 20 kHz

    # Step 1: Read the WAV file
    sample_rate, data = wavfile.read(wav_file_path)

    # Step 2: Normalize the frequency range for the butterworth filter
    nyquist = 0.5 * sample_rate
    low = LOW_FREQ / nyquist
    high = HIGH_FREQ / nyquist

    # Step 3: Design a band-pass Butterworth filter
    sos = butter(N=10, Wn=[low, high], btype='bandpass', analog=False, output='sos')

    # Step 4: Apply the filter (can handle stereo by applying filter to each channel if necessary)
    if data.ndim == 2:
        filtered_left = sosfilt(sos, data[:, 0])
        filtered_right = sosfilt(sos, data[:, 1])
        filtered_data = np.column_stack((filtered_left, filtered_right)).astype(data.dtype)
    else:
        filtered_data = sosfilt(sos, data).astype(data.dtype)

    # Step 5: Write the modified audio data back to a new WAV file
    output_file_path = "bandpassed_" + wav_file_path
    wavfile.write(output_file_path, sample_rate, filtered_data)
    
    return output_file_path

# Example usage:
# band_pass_filter("path_to_your_audio_file.wav")


# Mel Spectrogram
# Input: audio file, window of spectrogram
# Output: spectrogram
# Turns audio file into mel spectrogram

# ONCE THAT IS DONE
# Write one function that iterates through a table of {audio file, label} and runs each function on the audio files.
# Replaces

training_samples = dict()

def preprocessing_function(waveform : np.ndarray, sampleRate : int) -> np.ndarray:
    """
    Preprocesses the audio file to be used in the training of the model.
    Args:
        waveform (:obj:`np.array`): 1D waveform of raw audio.
        sampleRate (:obj:`int`): Sample rate of the audio.
    Returns:
        mel_spectrogram(:obj:`np.ndarray`): Mel spectrogram of the audio.
    """
    # TODO: change implementation to the above specification
    
    for file in os.listdir(directory):
        zeropad(file, sampleRate)
        magnitude_filter(file, 30) # TODO: change the 30 to appropiate threshold
        normalize_audio(file)
        spectrogram = compute_mel_spectrogram(file)
        training_samples[os.path.basename(file)].append()





# MODEL

# TRAINING
# Input: table of {audio file, label}
# Trains the CNN on the audio files and labels

# INFERENCE
# Input: untrained audio file
# Output: word/sentence predicted

# ACCURACY EVALUATION
# do an inference on many untrained audio files, count how many were correct
