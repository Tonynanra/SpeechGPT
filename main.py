# main

# PREPROCESSING FUNCTIONS

# Padding function
# Input: table of {audio file, label}, length to pad to
# Output: table of {padded audio file, label}
# Iterates through all the audio files and right-pads them with 0 to a length of [seconds]
from zeropad import zeropad

# Amplitude normalization
# Input: audio file
# Output: audio file
# Normalizes amplitude of sound
from normalize import normalize

# Noise Filter
# Input: audio file, threshold
# Output: audio file
# Eliminiates frequencies with a magnitude below [some threshold]
from denoise import denoise

# ONCE THAT IS DONE
# Write one function that iterates through a table of {audio file, label} and runs each function on the audio files.
# Replaces

def preprocessing_function(waveform, sampleRate, noise_threshold):
    stage1 = zeropad(waveform, sampleRate)
    stage2 = denoise(stage1, noise_threshold)
    stage3 = normalize(stage2, sampleRate)

    return stage3

