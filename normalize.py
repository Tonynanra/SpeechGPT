#%%
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
from pydub.playback import play
#%%

from pydub import AudioSegment
import numpy as np



def audio_segment_to_numpy_array(audio_segment):
    # Get the raw data from the AudioSegment object
    samples = audio_segment.get_array_of_samples()
    
    # Create a NumPy array from the samples
    audio_array = np.array(samples, dtype=np.int16)

    if audio_segment.channels == 2:
        audio_array = audio_array.reshape((-1, 2))
    
    return audio_array


def numpy_array_to_audio_segment(audio_array, sample_rate, channels=1):
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

    audSeg = numpy_array_to_audio_segment(audio_signal, sample_rate, channels=1)
    # Calculate the difference between the target dBFS and the current average dBFS
    change_in_dBFS = target_dBFS - audSeg.dBFS

    # Apply the necessary gain to normalize the amplitude
    normalized_audio = audSeg.apply_gain(change_in_dBFS)

    outputSignal = audio_segment_to_numpy_array(normalized_audio)

    return outputSignal
