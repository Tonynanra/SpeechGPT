# test

print("test")

# PREPROCESSING FUNCTIONS


# Amplitude normalization
# Input: audio file
# Output: audio file
# Normalizes amplitude of sound


# Replace 'file.wav' with your actual audio file's name and extension
normalized_audio = normalize_audio('file.wav')

# Save the normalized audio to a new file
normalized_audio.export("normalized_file.wav", format="wav")


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

# Compute the mel spectrogram
S_dB = compute_mel_spectrogram(audio_file)

# Plot the mel spectrogram
plot_mel_spectrogram(S_dB)


# ONCE THAT IS DONE
# Write one function that iterates through a table of {audio file, label} and runs each function on the audio files.
# Replaces


# MODEL

# TRAINING
# Input: table of {audio file, label}
# Trains the CNN on the audio files and labels

# INFERENCE
# Input: untrained audio file
# Output: word/sentence predicted

# ACCURACY EVALUATION
# do an inference on many untrained audio files, count how many were correct