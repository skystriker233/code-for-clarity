import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


samplerate, data = wavfile.read('L5076_Actions - One Minute Smile_remix.wav')
if len(data.shape) > 1:
    data = data[:, 0]

duration = len(data)/samplerate

fft_out = np.fft.rfft(data)

freqs = np.fft.rfftfreq(len(data), 1/samplerate)

plt.plot(freqs, np.abs(fft_out))
plt.xlim([6000, 8000])
plt.title('after using low pass filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()