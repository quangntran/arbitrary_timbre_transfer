from glob import glob
import numpy as np
import argparse
import os 
import librosa
from data_utils import * 
#
#n_fft = 2048
#hop_length = 512
#n_mels = 128
#sr = 22050
#fixed_len = 224 # length of each chunk of spectrogram to split
#instrument_list = ['harpsichord', 'piano']

n_fft = 1024
sr = 22050
IMG_DIM = (256,256,1)
raw_audio_path = './data/raw_audios/'
spectrogram_path =  './data/spectrogram'
instrument_list = ['harpsichord', 'piano']


def make_dir(path):
    """Make dir if path has not existed"""
    if not os.path.exists(path):
        os.makedirs(path)

def audio_to_melspec(path):
    """
    Given path to an audio file, return normalized mel-spectrogram
    """
    y = load_audio(path, sr=sr)
    # trim silent edges
    audio_arr, _ = librosa.effects.trim(y)
    mag, phase = forward_transform(audio_arr, nfft=n_fft)
    mag_db = amplitude_to_db(mag)
    mag_sliced = slice_magnitude(mag_db, IMG_DIM[1])
    mag_sliced = (mag_sliced * 2) - 1
    return mag_sliced, phase


parser = argparse.ArgumentParser()
parser.add_argument('--raw_audio_path', type=str, default='./data/raw_audios/', help='Path to .wav files')
parser.add_argument('--spectrogram_path', type=str, default='./data/spectrogram', help='Path to output spectrograms')

opts = parser.parse_args()
    
make_dir(opts.spectrogram_path)

# convert the audio .wav files to spectrograms and split them into fixed lengths

for inst in instrument_list:
    path = os.path.join(raw_audio_path, inst)
    raw_audio_list = glob(f'{path}/*.wav')
    output_mag_path = os.path.join(spectrogram_path, inst, 'mag')
    output_phase_path = os.path.join(spectrogram_path, inst, 'phase')
    make_dir(output_mag_path)
    make_dir(output_phase_path)
    for audio in raw_audio_list:
        arr_count = 0 # number of saved .npy
        audio_file_name = audio.split('/')[-1]
        mag_sliced, phase = audio_to_melspec(audio)
        for i in range(mag_sliced.shape[0]-1):
            piece_mag = mag_sliced[i,:,:,0]
            piece_phase = phase[:,IMG_DIM[0]*i:IMG_DIM[0]*(i+1)]
            # the chunk is with name [instrument name]_[raw audio file name]_[order of the chunk].npy. 
            # save magnitude
            np.save(output_mag_path+f'/{inst}_{audio_file_name}_{i}.npy', piece_mag)
            # save phase
            np.save(output_phase_path+f'/{inst}_{audio_file_name}_{i}.npy', piece_phase)
        

                 