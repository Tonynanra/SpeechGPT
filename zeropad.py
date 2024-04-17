import numpy as np

def zeropad(audio_signal, sample_rate):
    # Calculate the target number of samples for a 30-second audio
    tgt_length = 30 * sample_rate
    
    # Determine the current size of the audio signal
    current_length = audio_signal.size
    
    # If the audio signal is shorter than the target, pad it with zeros
    if current_length < tgt_length:
        # Calculate how many zeros are needed
        padding_length = tgt_length - current_length
        # Create an array of zeros
        padding = np.zeros(padding_length, dtype=audio_signal.dtype)
        # Concatenate the original audio signal with the padding array
        audio_signal = np.concatenate((audio_signal, padding))
    elif current_length > tgt_length:
        # If the audio signal is longer, truncate it to the target length
        audio_signal = audio_signal[:tgt_length]
    
    return audio_signal
