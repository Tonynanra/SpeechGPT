# test

print("test")

# PREPROCESSING FUNCTIONS
pip install pydub

from pydub import AudioSegment
from pydub.playback import play

def normalize_audio(file_path, target_dBFS=-20.0):
    """
    Normalizes the audio file's amplitude to a target dBFS.

    :param file_path: Path to the audio file.
    :param target_dBFS: Target amplitude in dBFS. Default is -20.0 dBFS.
    :return: Normalized AudioSegment object.
    """
    audio = AudioSegment.from_file(file_path)

    # Calculate the difference between the target dBFS and the current average dBFS
    change_in_dBFS = target_dBFS - audio.dBFS

    # Apply the necessary gain to normalize the amplitude
    normalized_audio = audio.apply_gain(change_in_dBFS)

    return normalized_audio

# Replace 'file.wav' with your actual audio file's name and extension
normalized_audio = normalize_audio('file.wav')

# Optionally, you can play back the normalized audio to test
play(normalized_audio)

# Save the normalized audio to a new file
normalized_audio.export("normalized_file.wav", format="wav")


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





# MODEL

# TRAINING
# Input: table of {audio file, label}
# Trains the CNN on the audio files and labels

# INFERENCE
# Input: untrained audio file
# Output: word/sentence predicted

# ACCURACY EVALUATION
# do an inference on many untrained audio files, count how many were correct