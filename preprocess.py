import os
import numpy as np
from pysoundfile import SoundFile, read_mode, write_mode, snd_types, snd_subtypes, snd_endians
from features import rms, peak
import pdb

def preprocess_sample(path_raw, path_processed):
    """ process the sample of the sample base to be
    cutted to contain no silence and to be normalized
    to a defined rms value

    """
    fs = 44100
    sound_file = SoundFile(path_raw, mode=read_mode)
    if sound_file.sample_rate != fs:
        print("Wrong sample rate of ", path_raw, "  ", sound_file.sample_rate, " should be ", fs)
        return
    data = preprocess_sample_data(sound_file[:])
    wave_file_float = snd_types['WAV']|snd_subtypes['FLOAT']|snd_endians['FILE']
    sound_file = SoundFile(path_processed, sample_rate=fs, channels=1, format=wave_file_float, mode=write_mode)
    sound_file.write(data)


def preprocess_sample_data(data):
    """ all desired preprocessing steps of the audio
    data of one sample
    """
    data = pan_to_mono(data)
    data = remove_silence(data)
    return normalize_rms(data)


def pan_to_mono(data):
    """ merge several audio channels into one channel
    """
    return np.mean(data, axis=1)


def remove_silence(data, silence_threshold_db=-75):
    """ remove silence at the beginning and end
    of an audio file saved in data
    """
    silence_threshold = db_to_lin(silence_threshold_db)
    start_idx = np.argmax(data > silence_threshold)
    end_idx = len(data)-np.argmax(data[::-1] > silence_threshold)-1
    return data[start_idx:end_idx]


def normalize_rms(data, rms_desired=0.5):
    """ set the rms value of an audio file to the defined
    value
    """
    rms_actual = rms(data)
    return rms_desired*data/rms_actual


def db_to_lin(value_db):
    """ calculate linear value corresponding to
    given value in decibel
    """
    return 10**(value_db*0.05)


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    preprocess_sample(input_file, output_file)
