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
