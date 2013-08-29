import os
import pandas as pd
import numpy as np
from scipy.signal import hann
from pysoundfile import SoundFile, read_mode

def walk_files(sample_path, progress=False):
    for root, dirs, files in os.walk(sample_path):
        if root == sample_path:
            continue
        for idx, file in enumerate(files):
            if file.startswith('.'): continue
            if progress: print('\rDirectory: %s (%i%%) %s%s' % (root, idx*100/len(files), file, ' '*20), end='')
            yield(root+'/'+file)
        return #for debugging and stuff

            
def blocks(sound_file, block_len):
    sound_file.seek_absolute(0)
    while sound_file.seek(0) < len(sound_file)-1:
        data = sound_file.read(block_len)[:,0] 
        #data *= hann(len(data))
        fft_data = np.fft.rfft(data)
        sound_file.seek(-block_len//2)
        yield(data, fft_data)

             
def extract_features(path, block_len_sec):
    print(path)
    file = SoundFile(path, file_mode=read_mode)
    block_len = int(file.samplerate*block_len_sec)
    feature_data = {}
    for name, func in features.items():
        feature_data[name] = np.array([func(block, fft_block) for block, fft_block in blocks(file, block_len)])
    return pd.DataFrame(feature_data)
             
        
def rms(data, fft_data):
    return np.sqrt(np.mean(data*data))
    
    
def peak(data, fft_data):
    return np.max(np.abs(data)) 
    
    
def spectral_centroid(data, fft_data):
    freq = np.fft.fftfreq(len(data))[:len(data)/2+1]
    spec = np.abs(fft_data)
    centroid = np.sum(spec*freq)/np.sum(spec)
    return centroid


def log_spectral_centroid(data, fft_data):
    freq = np.fft.fftfreq(len(data))[:len(data)/2+1]
    freq = freq+1
    spec = np.abs(fft_data)
    log_centroid = np.sum(spec*np.log(freq))/np.sum(spec)
    return log_centroid
    
    
features = { 'rms': rms,
             'peak':  peak,
             'centroid': spectral_centroid,
             'log_centroid': log_spectral_centroid}
             

if __name__ == '__main__':
    # features = {}
    # for file in walk_files('SampleBase'):
    #    features[file] = feature_extraction(file)
    block_len_sec = 0.02
    print('lala')
    features = {file: extract_features(file, block_len_sec) for file in walk_files('SampleBase', progress=True)}
