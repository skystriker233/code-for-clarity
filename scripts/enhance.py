""" Run the dummy enhancement. """
from __future__ import annotations

# pylint: disable=too-many-locals
# pylint: disable=import-error
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from omegaconf import DictConfig
from scipy.io import wavfile
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
from torchaudio.transforms import Fade, Resample
from torchaudio.models import HDemucs

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Audiogram, Listener
from clarity.utils.signal_processing import denormalize_signals, normalize_signal
from recipes.cad1.task1.baseline.evaluate import make_song_listener_list
from multibandCompressor import MultibandCompressor

logger = logging.getLogger(__name__)


def separate_sources(
    model: torch.nn.Module,
    mix: torch.Tensor,
    sample_rate: int,
    segment: float = 10.0,
    overlap: float = 0.1,
    device: torch.device | str | None = None,
):
    """
    Apply model to a given mixture.
    Use fade, and add segments together in order to add model segment by segment.

    Args:
        model (torch.nn.Module): model to use for separation
        mix (torch.Tensor): mixture to separate, shape (batch, channels, time)
        sample_rate (int): sampling rate of the mixture
        segment (float): segment length in seconds
        overlap (float): overlap between segments, between 0 and 1
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.

    Returns:
        torch.Tensor: estimated sources

    Based on https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html
    """
    device = mix.device if device is None else torch.device(device)
    mix = torch.as_tensor(mix, device=device)

    if mix.ndim == 1:
        # one track and mono audio
        mix = mix.unsqueeze(0)
    elif mix.ndim == 2:
        # one track and stereo audio
        mix = mix.unsqueeze(0)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, 4, channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0

    return final.cpu().detach().numpy()


def get_device(device: str) -> tuple:
    """Get the Torch device.

    Args:
        device (str): device type, e.g. "cpu", "gpu0", "gpu1", etc.

    Returns:
        torch.device: torch.device() appropiate to the hardware available.
        str: device type selected, e.g. "cpu", "cuda".
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"

    if device.startswith("gpu"):
        device_index = int(device.replace("gpu", ""))
        if device_index > torch.cuda.device_count():
            raise ValueError(f"GPU device index {device_index} is not available.")
        return torch.device(f"cuda:{device_index}"), "cuda"

    if device == "cpu":
        return torch.device("cpu"), "cpu"

    raise ValueError(f"Unsupported device type: {device}")


def map_to_dict(sources: np.ndarray, sources_list: list[str]) -> dict:
    """Map sources to a dictionary separating audio into left and right channels.

    Args:
       sources (np.ndarray): Signal to be mapped to dictionary.
       sources_list (list): List of strings used to index dictionary.

    Returns:
        Dictionary: A dictionary of separated source audio split into channels.
    """
    audios = dict(zip(sources_list, sources))

    signal_stems = {}
    for source in sources_list:
        audio = audios[source]
        signal_stems[f"left_{source}"] = audio[0]
        signal_stems[f"right_{source}"] = audio[1]

    return signal_stems


# pylint: disable=unused-argument
def decompose_signal(
    config: DictConfig,
    model: torch.nn.Module,
    signal: np.ndarray,
    sample_rate: int,
    device: torch.device,
    listener: Listener,
) -> dict[str, np.ndarray]:
    """
    Decompose signal into 8 stems.

    The left and right audiograms are ignored by the baseline system as it
    is performing personalised decomposition.
    Instead, it performs a standard music decomposition using the
    HDEMUCS model trained on the MUSDB18 dataset.

    Args:
        config (DictConfig): Configuration object.
        model (torch.nn.Module): Torch model.
        signal (np.ndarray): Signal to be decomposed.
        sample_rate (int): Sample frequency.
        device (torch.device): Torch device to use for processing.
        listener (Listener).

     Returns:
         Dictionary: Indexed by sources with the associated model as values.
    """

    if config.separator.model == "demucs":
        signal, ref = normalize_signal(signal)

    model_sample_rate = (
        model.sample_rate if config.separator.model == "openunmix" else 44100
    )

    if sample_rate != model_sample_rate:
        resampler = Resample(sample_rate, model_sample_rate)
        signal = resampler(signal)

    sources = separate_sources(
        model, torch.from_numpy(signal), sample_rate, device=device
    )
    # only one element in the batch
    sources = sources[0]
    if config.separator.model == "demucs":
        sources = denormalize_signals(sources, ref)

    signal_stems = map_to_dict(sources, config.separator.sources)
    return signal_stems


def apply_baseline_ha(
    enhancer: NALR,
    compressor: Compressor,
    signal: np.ndarray,
    audiogram: Audiogram,
    apply_compressor: bool = False,
) -> np.ndarray:
    """
    Apply NAL-R prescription hearing aid to a signal.

    Args:
        enhancer: A NALR object that enhances the signal.
        compressor: A Compressor object that compresses the signal.
        signal: An ndarray representing the audio signal.
        listener_audiogram: An ndarray representing the listener's audiogram.
        cfs: An ndarray of center frequencies.
        apply_compressor: A boolean indicating whether to include the compressor.

    Returns:
        An ndarray representing the processed signal.
    """
    print("XXX", audiogram)
    nalr_fir, _ = enhancer.build(audiogram)
    proc_signal = enhancer.apply(nalr_fir, signal)
    if apply_compressor:
        proc_signal, _, _ = compressor.process(proc_signal)
    return proc_signal


def process_stems_for_listener(
    stems: dict,
    enhancer: NALR,
    compressor: Compressor,
    listener: Listener,
    apply_compressor: bool = False,
) -> dict:
    """Process the stems from sources.

    Args:
        stems (dict) : Dictionary of stems
        enhancer (NALR) : NAL-R prescription hearing aid
        compressor (Compressor) : Compressor
        listener: Listener object
        cfs (np.ndarray) : Center frequencies
        apply_compressor (bool) : Whether to apply the compressor
    Returns:
        processed_sources (dict) : Dictionary of processed stems
    """

    processed_stems = {}

    for stem_str in stems:
        stem_signal = stems[stem_str]

        # Determine the audiogram to use
        audiogram = (
            listener.audiogram_left
            if stem_str.startswith("l")
            else listener.audiogram_right
        )

        # Apply NALR prescription to stem_signal
        proc_signal = apply_baseline_ha(
            enhancer, compressor, stem_signal, audiogram, apply_compressor
        )
        processed_stems[stem_str] = proc_signal
    return processed_stems



def clip_signal(signal: np.ndarray, soft_clip: bool = False) -> tuple[np.ndarray, int]:
    """Clip and save the processed stems.

    Args:
        signal (np.ndarray): Signal to be clipped and saved.
        soft_clip (bool): Whether to use soft clipping.

    Returns:
        signal (np.ndarray): Clipped signal.
        n_clipped (int): Number of samples clipped.
    """

    if soft_clip:
        signal = np.tanh(signal)
    n_clipped = np.sum(np.abs(signal) > 1.0)
    np.clip(signal, -1.0, 1.0, out=signal)
    return signal, int(n_clipped)


def to_16bit(signal: np.ndarray) -> np.ndarray:
    return (32768.0 * signal).astype(np.int16)
    
def change_volume(audio, db):
    factor = np.power(10.0, db / 20.0)
    print(f"The audio should be multiplied by a factor of {factor}")
    return audio * factor

def judge_volume(str1,audio,db1,db2,db3,db4):
    if str1 == "left_vocals" or str1 == "right_vocals":
        return change_volume(audio,db1)
    elif str1 == "left_bass" or str1 == "right_bass":
        return change_volume(audio,db2)
    elif str1 == "left_drums" or str1 == "right_drums":
        return change_volume(audio,db3)
    elif str1 == "left_other" or str1 == "right_other":
        return change_volume(audio,db4)

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
    

@hydra.main(config_path="", config_name="config")
def enhance(config: DictConfig) -> None:
    """
    Run the music enhancement.
    The system decomposes the music into vocal, drums, bass, and other stems.
    Then, the NAL-R prescription procedure is applied to each stem.
    Args:
        config (dict): Dictionary of configuration options for enhancing music.

    Returns 8 stems for each song:
        - left channel vocal, drums, bass, and other stems
        - right channel vocal, drums, bass, and other stems
    """

    enhanced_folder = Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    # Training stage
    #
    # The baseline is using an off-the-shelf model trained on the MUSDB18 dataset
    # Training listeners and song are not necessary in this case.
    #
    # Training songs and audiograms can be read like this:
    #
    #  with open(config.path.listeners_train_file, "r", encoding="utf-8") as file:
    #        listener_train_audiograms = json.load(file)
    #
    #  with open(config.path.music_train_file, "r", encoding="utf-8") as file:
    #        song_data = json.load(file)
    #  songs_train = pd.DataFrame.from_dict(song_data)
    #
    # train_song_listener_pairs = make_song_listener_list(
    #     songs_train['Track Name'], listener_train_audiograms
    # )

    if config.separator.model == "demucs":
    
        separation_model = HDEMUCS_HIGH_MUSDB.get_model()
        
       
        
    else:
        separation_model = torch.hub.load("sigsep/open-unmix-pytorch", "umxhq", niter=0)
       
        
    
    device, _ = get_device(config.separator.device)
    separation_model.to(device)

    # Processing Validation Set
    # Load listener audiograms and songs
    listener_dict = Listener.load_listener_dict(config.path.listeners_valid_file)

    with open(config.path.music_valid_file, encoding="utf-8") as file:
        song_data = json.load(file)
    songs_valid = pd.DataFrame.from_dict(song_data)

    valid_song_listener_pairs = make_song_listener_list(
        songs_valid["Track Name"], listener_dict
    )
    # Select a batch to process
    valid_song_listener_pairs = valid_song_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    enhancer = NALR(**config.nalr)
    compressor = Compressor(**config.compressor)

    # Decompose each song into left and right vocal, drums, bass, and other stems
    # and process each stem for the listener
    prev_song_name = None
    num_song_list_pair = len(valid_song_listener_pairs)
    for idx, song_listener in enumerate(valid_song_listener_pairs, 1):
        song_name, listener_name = song_listener
        logger.info(
            f"[{idx:03d}/{num_song_list_pair:03d}] "
            f"Processing {song_name} for {listener_name}..."
        )
        # Get the listener's audiogram
        listener = listener_dict[listener_name]
        
        #print(classify_listeners(np.array(listener.audiogram_left.frequencies), np.array(listener.audiogram_left.levels), np.array(listener.audiogram_right.levels)))
        
        # Find the music split directory
        split_directory = (
            "test"
            if songs_valid.loc[songs_valid["Track Name"] == song_name, "Split"].iloc[0]
            == "test"
            else "train"
        )

        # Read the mixture signal
        # Convert to 32-bit floating point and transpose
        # from [samples, channels] to [channels, samples]
        if prev_song_name != song_name:
            # Decompose song only once
            prev_song_name = song_name

            sample_rate, mixture_signal = wavfile.read(
                Path(config.path.music_dir)
                / split_directory
                / song_name
                / "mixture.wav"
            )
            mixture_signal = (mixture_signal / 32768.0).astype(np.float32).T
            assert sample_rate == config.sample_rate

            stems: dict[str, ndarray] = decompose_signal(
                config,
                separation_model,
                mixture_signal,
                sample_rate,
                device,
                listener,
            )

        # Baseline applies NALR prescription to each stem instead of using the
        # listener's audiograms in the decomposition. This stem can be skipped
        # if the listener's audiograms are used in the decomposition
        processed_stems = process_stems_for_listener(
            stems,
            enhancer,
            compressor,
            listener,
            config.apply_compressor,
        )

        # save processed stems
        n_samples = processed_stems[list(processed_stems.keys())[0]].shape[0]
        output_left, output_right = np.zeros(n_samples), np.zeros(n_samples)
        #bands = [(400,500)] 
        #output_left1,output_right1 = np.zeros(n_samples), np.zeros(n_samples)
        
        bands = []
        
        for stem_str, stem_signal in processed_stems.items():
            if stem_str == "left_vocals" or stem_str == "right_vocals":
                bands = selectbands(stem_signal)
                print(bands)
        
        
        for stem_str, stem_signal in processed_stems.items():
            #if stem_str == "left_vocals" or stem_str == "right_":
                
            multicompressor = MultibandCompressor(bands)
            compressed_signal = multicompressor.process(stem_signal, sample_rate, compressor)
            stem_signal = compressed_signal
            if stem_str.startswith("l"):
                print(stem_str)
                output_left += stem_signal
                #output_left += compressed_signal
                #output_left1 += judge_volume(stem_str,stem_signal,2,1,1,2)
            else:
                print(stem_str)
                output_right += stem_signal
                #output_right += compressed_signal
                #output_right1 += judge_volume(stem_str,stem_signal,2,1,1,2)

            filename = (
                enhanced_folder
                / f"{listener.id}"
                / f"{song_name}"
                / f"{listener.id}_{song_name}_{stem_str}.wav"
            )
            filename.parent.mkdir(parents=True, exist_ok=True)

            # Clip and save stem signals
            clipped_signal, n_clipped = clip_signal(stem_signal, config.soft_clip)
            if n_clipped > 0:
                logger.warning(f"Writing {filename}: {n_clipped} samples clipped")
            wavfile.write(filename, config.sample_rate, to_16bit(clipped_signal))
        
        
        #cutoff_left_freq = np.geomspace(8000, 6000, output_left.shape[0])
        #cutoff_right_freq = np.geomspace(8000, 6000, output_right.shape[0])
        #output_left2 = allpass_based_filter(output_left, cutoff_left_freq, 44100, highpass=False, amplitude=1.0)
        #output_right2 = allpass_based_filter(output_right, cutoff_right_freq, 44100, highpass=False, amplitude=1.0)
        enhanced = np.stack([output_left, output_right], axis=1)
        
        enhanced_left = output_left  #edit
        enhanced_right = output_right #edit
        filename = (
            enhanced_folder
            / f"{listener.id}"
            / f"{song_name}"
            / f"{listener.id}_{song_name}_remix.wav"
        )
        
        filename_left = (
            enhanced_folder
            / f"{listener.id}"
            / f"{song_name}"
            / f"{listener.id}_{song_name}_left_mixture.wav"
        )
        
        filename_right = (
            enhanced_folder
            / f"{listener.id}"
            / f"{song_name}"
            / f"{listener.id}_{song_name}_right_mixture.wav"
        )
        
        # clip and save enhanced signal
        clipped_signal, n_clipped = clip_signal(enhanced, config.soft_clip)
        if n_clipped > 0:
            logger.warning(f"Writing {filename}: {n_clipped} samples clipped")
        wavfile.write(filename, config.sample_rate, to_16bit(clipped_signal))
        #wavfile.write(filename_left, config.sample_rate, output_left2)
        #wavfile.write(filename_right, config.sample_rate, output_right2)
        
        


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()
