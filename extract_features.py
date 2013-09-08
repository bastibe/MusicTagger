import os
import sys
import inspect
import pandas as pd
from multiprocessing import Pool
from scipy.signal import hann
from numpy.fft import rfft
from pysoundfile import SoundFile, read_mode
import features


def all_features():
    """Return all functions defined in the module "features"."""
    for name, function in inspect.getmembers(features):
        if inspect.isfunction(function):
            yield(name, function)


def walk_files(sample_path):
    """Iterate over all files in subdirectories of sample_path.

    Dotfiles are ignored.

    """
    for root, dirs, files in os.walk(sample_path):
        for idx, file in enumerate(files):
            if file.startswith('.'): continue
            yield(root+'/'+file)


def blocks(sound_file, block_len, overlap=0.5, window=hann):
    """Returns blocks of audio data from a sound file.

    sound_file must be an instance of pysoundfile.SoundFile.

    Each block will be of length block_len. The last block in the file
    is likely to be shorter. Blocks will overlap according to overlap
    and be windowed by the window function. window is a function that
    takes a numeric argument and returns a window of that length.

    This will only read the first channel if there are more than one.

    """
    sound_file.seek_absolute(0)
    while sound_file.seek(0) < len(sound_file)-1:
        sound_file.seek(int(-block_len*overlap))
        data = sound_file.read(block_len)[:,0] # first channel only
        data *= window(len(data))
        yield(data, rfft(data))


def extract_features(path, block_len_sec=0.02):
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
    feature_data = { 'file': os.path.relpath(path),
                     'tag': os.path.basename(os.path.dirname(path)) }
    for name, func in all_features():
        feature_data[name] = [func(*data) for data in blocks(file, block_len)]
    return pd.DataFrame(feature_data)

if __name__ == '__main__':
    sample_path = 'SampleBase'
    if sys.platform == 'win32':
        feature_data = pd.concat([extract_features(f) for f in walk_files(sample_path)])
    else:
        pool = Pool(processes=4)
        feature_data = pd.concat(pool.map(extract_features, walk_files(sample_path), chunksize=100))
    feature_data.to_hdf('feature_data.hdf', 'features')
