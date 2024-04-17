# main

import os
print("test")

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
from normalize import normalize_audio

# Noise Filter
# Input: audio file, threshold
# Output: audio file
# Eliminiates frequencies with a magnitude below [some threshold]
from denoise import denoise

# Mel Spectrogram
# Input: audio file, window of spectrogram
# Output: spectrogram
# Turns audio file into mel spectrogram
from mel import compute_mel_spectrogram

# ONCE THAT IS DONE
# Write one function that iterates through a table of {audio file, label} and runs each function on the audio files.
# Replaces

training_samples = dict()

def preprocessing_function(waveform, sampleRate):
    stage1 = zeropad(waveform, sampleRate)
    stage2 = denoise(stage1, 0.1)
    stage3 = normalize_audio(stage2)
    return stage3





# MODEL

# TRAINING
# Input: table of {audio file, label}
# Trains the CNN on the audio files and labels

# INFERENCE
# Input: untrained audio file
# Output: word/sentence predicted

# ACCURACY EVALUATION
# do an inference on many untrained audio files, count how many were correct
