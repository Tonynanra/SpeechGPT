from scipy.io import wavfile
import numpy as np 
import matplotlib.pyplot as plt 
from zeropad import zeropad


samplerate, data = wavfile.read('./0a0b46ae_nohash_0.wav')

plt.plot(data)
plt.show()

# plt.figure(2)
# zeropadded = zeropad(data, samplerate)
# plt.plot(zeropadded)
# plt.show()

plt.figure(3)
fftdata = np.fft.fft(data)
plt.plot(fftdata)
plt.show()

# data to be plotted
# x = np.arange(1, 11) 
# y = x * x
 
# plotting
# plt.title("Line graph") 
# plt.xlabel("X axis") 
# plt.ylabel("Y axis") 
# plt.plot(x, y, color ="red") 
# plt.show()