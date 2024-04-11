import numpy as np

def zeropad(audio_signal, sample_rate)

    #find target number of samples to get to 30 s
    tgt_length = 30 * sample_rate
    
    #add zeroes if its too short
    while len(audio_signal) < tgt_length
        audio_signal.append(0)
            
    #truncate if its too long
    while len(audio_signal) > tgt_length
        audio_signal.pop()
    
    return audio_signal
