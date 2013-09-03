import os
import sys
import pandas as pd
import numpy as np
import pdb
from scipy.signal import welch, hann
from extract_features import walk_files
from pysoundfile import SoundFile, read_mode


def blocks(sound_file, block_len, psd_len, overlap=0.5):
    """Returns power spectral density (psd) of blocks of audio data 
    from a sound file.

    sound_file must be an instance of pysoundfile.SoundFile.

    Each block will be of length block_len. The psd length is defined
    by psd_len.The last block in the file is likely to be shorter and 
    will be dismissed. Blocks will overlap according to overlap and 
    be windowed by the window function. window is a function that 
    takes a numeric argument and returns a window of that length.

    This will only read the first channel if there are more than one.

    """
    sound_file.seek_absolute(0)
    while sound_file.seek(0) < len(sound_file)-1:
        sound_file.seek(int(-block_len*overlap))
        data = sound_file.read(block_len)[:,0] # first channel only
        yield(welch(data, sound_file.sample_rate, nperseg=psd_len, nfft=psd_len)[1])


def extract_psd(path, block_len_sec=0.02):
    """extracts the psd for each block of a sound file at a given path.

    This reads the sound file at path, cuts it up in windowed,
    overlapping blocks of length block_len_sec (in seconds), and
    calculates the psd.

    Returns a pandas DataFrame with block indices as rows.
    
    """
    file = SoundFile(path, mode=read_mode)
    block_len = int(file.sample_rate*block_len_sec)
    psd_len = 128
    psd_data = { 'file': os.path.relpath(path),
                 'tag': os.path.basename(os.path.dirname(path)) }
    psd_data['psd'] = [pd.Series(psd) for psd in blocks(file, block_len, psd_len)]
    return pd.DataFrame(psd_data)

    
if __name__ == '__main__':
    if sys.platform == 'win32':
        psd_data = pd.concat([extract_psd(f) for f in walk_files('SampleBase/DrumsPercussive')])
    else:
        pool = Pool(processes=4)
        psd_data = pd.concat(pool.map(extract_psd, walk_files('SampleBase'), chunksize=100))
    psd_data.to_hdf('feature_data.hdf', 'psd')
    