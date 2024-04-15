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

# Band Pass?
# Eliminate frequencies outside human hearing

# Mel Spectrogram
# Input: audio file, window of spectrogram
# Output: spectrogram
# Turns audio file into mel spectrogram

# ONCE THAT IS DONE
# Write one function that iterates through a table of {audio file, label} and runs each function on the audio files.
# Replaces

def preprocessing_function(directory, sampleRate):
    for file in os.listdir(directory):
        zeropad(file, sampleRate)
        magnitude_filter(file, 30) # TODO: change the 30 to appropiate threshold
        normalize_audio(file)
        spectrogram = compute_mel_spectrogram(file)





# MODEL

# TRAINING
# Input: table of {audio file, label}
# Trains the CNN on the audio files and labels

# INFERENCE
# Input: untrained audio file
# Output: word/sentence predicted

# ACCURACY EVALUATION
# do an inference on many untrained audio files, count how many were correct