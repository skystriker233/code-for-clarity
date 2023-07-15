import json
import numpy as np
import soundfile as sf
import os
# from pydub import AudioSegment
import librosa

# l_drum = AudioSegment.from_wav("L5000_Actions - One Minute Smile_left_drums.wav")
# l_vocal = AudioSegment.from_wav("L5000_Actions - One Minute Smile_left_vocals.wav")
# played_togther = l_vocal.overlay(l_drum)
# played_togther.export("output.wav", format="wav")
# os.system("start output.wav")

# audio_path = "L5000_Actions - One Minute Smile_left_drums.wav"
# audio, sr = librosa.load(audio_path, sr=None)
# print(len(audio)
# )

with open("listeners.valid.json", encoding="utf-8") as file:

    listener_valid_audiograms = json.load(file)

print(listener_valid_audiograms)

critical_frequencies = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])
audiogram_left = np.array([15, 20, 30, 30, 45, 50, 60, 75])
audiogram_right = np.array([15, 25, 30, 35, 40, 40, 60, 70])


def classify_listeners(critical_freq, audiogram_l, audiogram_r):
    result = {'left': {'low': 0, 'high': 0},
              'right': {'low': 0, 'high': 0}}  # 0 means not serous, 1 means serious

    num_frequency = critical_freq.shape[0]
    count_low_freq = 0
    count_high_freq = 0
    vote_low_s_l = 0
    vote_low_ns_l = 0
    vote_high_s_l = 0
    vote_high_ns_l = 0
    vote_low_s_r = 0
    vote_low_ns_r = 0
    vote_high_s_r = 0
    vote_high_ns_r = 0
    freq_threshold = 4000  # decide high frequency or low frequency
    level_threshold = 50  # decide whether it is a serious hearing loss on a certain frequency

    for frequency in critical_freq:

        if frequency < freq_threshold:
            count_low_freq += 1
        else:
            count_high_freq = num_frequency - count_low_freq
            break

    for index in range(count_low_freq):

        if audiogram_l[index] > level_threshold:
            vote_low_s_l += 1
        else:
            vote_low_ns_l += 1

        if audiogram_r[index] > level_threshold:
            vote_low_s_r += 1
        else:
            vote_low_ns_r += 1

    if vote_low_s_l >= vote_low_ns_l:
        result['left']['low'] = 1
    else:
        result['left']['low'] = 0

    if vote_low_s_r >= vote_low_ns_r:
        result['right']['low'] = 1
    else:
        result['right']['low'] = 0

    for index in range(count_low_freq, num_frequency):
        if audiogram_l[index] > level_threshold:
            vote_high_s_l += 1
        else:
            vote_high_ns_l += 1

        if audiogram_r[index] > level_threshold:
            vote_high_s_r += 1
        else:
            vote_high_ns_r += 1

    if vote_high_s_l >= vote_high_ns_l:
        result['left']['high'] = 1
    else:
        result['left']['high'] = 0

    if vote_high_s_r >= vote_high_ns_r:
        result['right']['high'] = 1
    else:
        result['right']['high'] = 0

    return result


def classify_frequency_listeners(audiogram_left,audiogram_right):
    # 0(None) below 25 dbs,  1(slight): between 26 and 40 dbs,  2(severe): above 40 dbs

    result_left = list([0]*audiogram_left)
    result_right =list([0]*audiogram_right)

    for index in range(audiogram_left.shape[0]):

        if audiogram_left[index] < 20:
            result_left[index] = 0
        elif audiogram_left[index] >= 20 and audiogram_left[index] < 35:
            result_left[index] = 1
        elif audiogram_left[index] >= 35 and audiogram_left[index] < 50:
            result_left[index] = 2
        elif audiogram_left[index] >= 50 and audiogram_left[index] < 65:
            result_left[index] = 3
        else:
            result_left[index] = 4


        if audiogram_right[index] < 20:
            result_right[index] = 0
        elif audiogram_right[index] >= 20 and audiogram_right[index] < 35:
            result_right[index] = 1
        elif audiogram_right[index] >= 35 and audiogram_right[index] < 50:
            result_right[index] = 2
        elif audiogram_right[index] >= 50 and audiogram_right[index] < 65:
            result_right[index] = 3
        else:
            result_right[index] = 4

    return result_left,result_right


#print(classify_listeners(critical_frequencies, audiogram_left, audiogram_right))
print(classify_frequency_listeners(audiogram_left,audiogram_right))
#print(np.array(listener_valid_audiograms["L5076"]['audiogram_cfs']))



def change_volume(audio, db):
    factor = np.power(10.0, db / 20.0)
    print(f"The audio should be multiplied by a factor of {factor}")
    return audio * factor

def judge_volume(str,audio,db1,db2,db3,db4):
    if str == "left_vocals" or str == "right_vocals":
        return change_volume(audio,db1)
    elif str == "left_bass" or str == "right_bass":
        return change_volume(audio,db2)
    elif str == "left_drums" or str == "right_drums":
        return change_volume(audio,db3)
    elif str == "left_other" or str == "right_other":
        return change_volume(audio,db4)

