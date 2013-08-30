import os
import types
import pandas as pd
import numpy as np
from scipy.signal import hann
from pysoundfile import SoundFile, read_mode
import features


def all_features():
    """Return all functions defined in the module "features"."""
    for name in dir(features):
        func = features.__dict__[name]
        if isinstance(func, types.FunctionType):
            yield(name, func)

def walk_files(sample_path, progress=False):
        if root == sample_path:
            continue
    """Iterate over all files in subdirectories of sample_path.

    Dotfiles are ignored. If progress is True, a status message is
    printed for every file.

    """
    for root, dirs, files in os.walk(sample_path):
        for idx, file in enumerate(files):
            if file.startswith('.'): continue
            if progress:
                print('\rDirectory: %s (%i%%) %s%s' %
                      (root, idx*100/len(files), file, ' '*20), end='')
            yield(root+'/'+file)
        return #for debugging and stuff

def blocks(sound_file, block_len, overlap=0.5, window=hann):
    """Returns blocks of audio data from a sound file.

    sound_file must be an instance of pysoundfile.SoundFile.

    Each block will be of length block_len. The last block in the file
    is likely to be shorter. Blocks will overlap according to overlap
    and be windowed by the window function. window is a function that
    takes a numeric argument and returns a window of that length.

    """
    sound_file.seek_absolute(0)
    while sound_file.seek(0) < len(sound_file)-1:
        sound_file.seek(int(-block_len*2))
        data = sound_file.read(block_len)[:,0]
        data *= window(len(data))
        fft_data = np.fft.rfft(data)
        yield(data, fft_data)


def extract_features(path, block_len_sec):
    """Calculates features for each block of a sound file at a given path.

    This reads the sound file at path, cuts it up in windowed,
    overlapping blocks of length block_len_sec (in seconds), and
    executes all functions defined in the module "features" on all
    blocks.

    Returns a pandas DataFrame with feature names as columns and block
    indices as rows.

    """
    file = SoundFile(path, mode=read_mode)
    block_len = int(file.sample_rate*block_len_sec)
    feature_data = {}
    for name, func in all_features():
        feature_data[name] = \
            np.array([func(block, fft_block)
                      for block, fft_block in blocks(file, block_len)])
    return pd.DataFrame(feature_data)


if __name__ == '__main__':
    # features = {}
    # for file in walk_files('SampleBase'):
    #    features[file] = feature_extraction(file)
    block_len_sec = 0.02
    feature_data = {file: extract_features(file, block_len_sec) for file in walk_files('SampleBase', progress=True)}
