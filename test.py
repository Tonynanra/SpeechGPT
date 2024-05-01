#%%
from main import preprocessing_function
from datasets import load_dataset, DatasetDict, DownloadConfig
import soundfile
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
#%%

audio = AudioSegment.from_file("seven.wav", "m4a")

# Convert to mono and get data as numpy array
audio = audio.set_channels(1)
data = np.array(audio.get_array_of_samples())
sound = preprocessing_function(data, audio.frame_rate, 0)
fig, ax = plt.subplots(2, figsize=(8,10))
ax[0].plot(np.abs((np.fft.fft(data))))
ax[1].plot(np.abs((np.fft.fft(sound))))
ax[0].set_title('Original Audio (Recording)')
ax[1].set_title('Preprocessed Audio (Recording)')
ax[0].set_xlabel('Frequency')
ax[0].set_ylabel('Magnitude')
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Magnitude')
soundfile.write('test.wav', sound, audio.frame_rate)
soundfile.write('test_original.wav', data, audio.frame_rate)
# %%
