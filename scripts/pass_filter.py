import numpy as np

def a1_coefficient(break_frequency, sampling_rate):
    tan = np.tan(np.pi * break_frequency / sampling_rate)
    return (tan - 1) / (tan + 1)


def allpass_filter(input_signal, break_frequency, sampling_rate):
    # Initialize the output array
    allpass_output = np.zeros_like(input_signal)

    # Initialize the inner 1-sample buffer
    dn_1 = 0

    for n in range(input_signal.shape[0]):
        # The allpass coefficient is computed for each sample
        # to show its adaptability
        a1 = a1_coefficient(break_frequency[n], sampling_rate)

        # The allpass difference equation
        # Check the article on the allpass filter for an
        # in-depth explanation
        allpass_output[n] = a1 * input_signal[n] + dn_1

        # Store a value in the inner buffer for the
        # next iteration
        dn_1 = input_signal[n] - a1 * allpass_output[n]
    return allpass_output


def allpass_based_filter(input_signal, cutoff_frequency, \
    sampling_rate, highpass=False, amplitude=1.0):
    # Perform allpass filtering
    allpass_output = allpass_filter(input_signal, \
        cutoff_frequency, sampling_rate)

    # If we want a highpass, we need to invert
    # the allpass output in phase
    if highpass:
        allpass_output *= -1

    # Sum the allpass output with the direct path
    filter_output = input_signal + allpass_output

    # Scale the amplitude to prevent clipping
    filter_output *= 0.5

    # Apply the given amplitude
    filter_output *= amplitude

    return filter_output

#sampling_rate = 44100


#How to use it: input_signal is a numpy array, it is just for one channel. It is a lowpass filter if highpass=False, amplitude can be set as 1.0
#output = allpass_based_filter(input_signal,cutoff_frequency,44100,highpass=False,amplitude=1.0)
#cutoff_frequency example: cutoff_frequency = np.geomspace(8000,6000,input_signal.shape[0])







