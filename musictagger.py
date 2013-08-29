import os
import pandas as pd
import numpy as np
from pysoundfile import SoundFile, read_mode

def walk_files(sample_path, progress=False):
    for root, dirs, files in os.walk(sample_path):
        for idx, file in enumerate(files):
            if file.startswith('.'): continue
            if progress: print('\rDirectory: %s (%i%%) %s' % (root, idx*100/len(files), file), end='')
            yield(root+'/'+file)

def blocks(sound_file, block_len):
    sound_file.seek_absolute(0)
    while sound_file.seek(0) < len(sound_file)-1:
        yield(sound_file.read(block_len))

features = { 'rms': lambda x: np.sqrt(np.mean(x*x)),
             'peak': lambda x: np.max(np.abs(x)) }

def extract_features(path, block_len_sec):
    file = SoundFile(path, file_mode=read_mode)
    block_len = int(file.samplerate*block_len_sec)
    feature_data = {}
    for name, func in features.items():
        feature_data[name] = np.array([func(block) for block in blocks(file, block_len)])
    return pd.DataFrame(feature_data)

if __name__ == '__main__':
    # features = {}
    # for file in walk_files('SampleBase'):
    #    features[file] = feature_extraction(file)
    block_len_sec = 0.02
    features = {file: extract_features(file, block_len_sec) for file in walk_files('SampleBase', progress=True)}
