#%%
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
from pydub.playback import play
#%%

from pydub import AudioSegment
import numpy as np

from pydub import AudioSegment
import numpy as np

def audio_segment_to_numpy_array(audio_segment):
    """
    Convert a pydub.AudioSegment object to a NumPy array.

    :param audio_segment: AudioSegment object.
    :return: NumPy array containing the audio signal.
    """
    # Get the raw data from the AudioSegment object
    samples = audio_segment.get_array_of_samples()
    
    # Create a NumPy array from the samples
    audio_array = np.array(samples, dtype=np.int16)

    # Reshape the array into stereo if the audio segment has 2 channels
    if audio_segment.channels == 2:
        audio_array = audio_array.reshape((-1, 2))
    
    return audio_array

# Example usage:
# Assuming you have an AudioSegment object `audio_segment`
# numpy_array = audio_segment_to_numpy_array(audio_segment)


def numpy_array_to_audio_segment(audio_array, sample_rate, channels=1):
    """
    Convert a NumPy array to a pydub.AudioSegment object.
    
    :param audio_array: NumPy array containing the audio signal.
    :param sample_rate: The sample rate of the audio.
    :param channels: Number of audio channels (1 for mono, 2 for stereo).
    :return: AudioSegment object.
    """
    # Ensure array is an integer type suitable for audio (int16 is typical for PCM audio)
    if audio_array.dtype != np.int16:
        audio_array = (audio_array * 32767).astype(np.int16)
    
    # Convert array to bytes
    audio_bytes = audio_array.tobytes()
    
    # Create an AudioSegment from this byte data
    audio_segment = AudioSegment(
        data=audio_bytes,
        sample_width=audio_array.dtype.itemsize,
        frame_rate=sample_rate,
        channels=channels
    )
    return audio_segment

# Example usage:
# Assuming you have a numpy array `audio_array` and a `sample_rate` e.g., 44100 Hz
# audio_segment = numpy_array_to_audio_segment(audio_array, sample_rate)


def normalize(audio_signal, sample_rate, target_dBFS=-20.0):
    """
    Normalizes the audio file's amplitude to a target dBFS.

    :param file_path: Path to the audio file.
    :param target_dBFS: Target amplitude in dBFS. Default is -20.0 dBFS.
    :return: Normalized AudioSegment object.
    """
    audSeg = numpy_array_to_audio_segment(audio_signal, sample_rate, channels=1)
    # Calculate the difference between the target dBFS and the current average dBFS
    change_in_dBFS = target_dBFS - audSeg.dBFS

    # Apply the necessary gain to normalize the amplitude
    normalized_audio = audSeg.apply_gain(change_in_dBFS)

    outputSignal = audio_segment_to_numpy_array(normalized_audio)

    return outputSignal
