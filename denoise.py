#%%
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
#%%

def spectral_subtraction_from_wav(file_path, noise_level=0.02, attenuation=2):
    """
    Perform noise reduction on speech audio from a WAV file using spectral subtraction.
    
    Parameters:s
        file_path (str): Path to the input WAV file.
        noise_level (float): Noise level estimate (default is 0.02).
        attenuation (float): Amount of attenuation to apply (default is 2).
    
    Returns:
        ndarray: Noise-reduced speech audio data.
    """
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(file_path)
    
    # Ensure mono audio (if stereo)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Convert audio data to floating point representation
    audio_data = audio_data.astype(np.float64)
    
    # Normalize audio data to range [-1, 1]
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data /= max_val
    
    # Perform spectral subtraction
    processed_audio = spectral_subtraction(audio_data, noise_level, attenuation)
    
    # Scale back to original range and convert to 16-bit integer format
    processed_audio *= max_val * 32767
    processed_audio = processed_audio.astype(np.int16)
    
    return sample_rate, processed_audio