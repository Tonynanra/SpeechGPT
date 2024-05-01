import numpy as np

def zeropad(audio_signal, sample_rate):
    tgt_length = 30 * sample_rate
    
    current_length = audio_signal.size
    
    if current_length < tgt_length:

        padding_length = tgt_length - current_length

        padding = np.zeros(padding_length, dtype=audio_signal.dtype)

        audio_signal = np.concatenate((audio_signal, padding))

    elif current_length > tgt_length:
        
        audio_signal = audio_signal[:tgt_length]
    
    return audio_signal
