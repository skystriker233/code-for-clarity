import numpy as np
from scipy.io import wavfile


samplerate, data = wavfile.read('L5076_Actions - One Minute Smile_remix.wav')
if len(data.shape) > 1:
    data = data[:, 0]

# fft_out = np.fft.rfft(data)
# freqs = np.fft.rfftfreq(len(data), 1/samplerate)
# amplitudes = np.abs(fft_out)
# max_amplitude = np.max(amplitudes)
# threshold = 1/4 * max_amplitude
# selected_freq_index = np.where(amplitudes > threshold)
# selected_freq = freqs[selected_freq_index]
# print(selected_freq)
# #print(np.median(selected_freq))

# freq_set = [500,1000,2000,3000,4000,6000,8000]

def selectbands(audio_array):
    freq_set = [500, 1000, 2000, 3000, 4000, 6000, 8000]
    fft_out = np.fft.rfft(audio_array)
    samplerate = 44100
    freqs = np.fft.rfftfreq(len(audio_array), 1 / samplerate)
    amplitudes = np.abs(fft_out)
    max_amplitude = np.max(amplitudes)
    threshold = 1 / 4 * max_amplitude
    selected_freq_index = np.where(amplitudes > threshold)
    selected_freq = freqs[selected_freq_index]
    count = np.array([0, 0, 0, 0, 0, 0])
    bands = []
    for i in range(len(freq_set)-1):
        for j in range(selected_freq.shape[0]):
            if selected_freq[j] > freq_set[i] and selected_freq[j]<=freq_set[i+1]:
                count[i]+=1
    indices = np.argpartition(count, -2)[-2:]
    indices = np.sort(indices)

    for index in indices:
        bands.append((freq_set[index],freq_set[index+1]))

    return bands

print(selectbands(data))


