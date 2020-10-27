from glob import glob
import numpy as np
import argparse
import os 
import librosa


n_fft = 2048
hop_length = 512
n_mels = 128
sr = 22050
fixed_len = 224 # length of each chunk of spectrogram to split
instrument_list = ['harpsichord', 'piano']

def make_dir(path):
    """Make dir if path has not existed"""
    if not os.path.exists(path):
        os.makedirs(path)
        
def audio_to_melspec(path):
    """
    Given path to an audio file, return mel-spectrogram
    """
    y, sr = librosa.load(path)
    # trim silent edges
    audio_arr, _ = librosa.effects.trim(y)
    spec = librosa.feature.melspectrogram(audio_arr, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return spec


parser = argparse.ArgumentParser()
parser.add_argument('--raw_audio_path', type=str, default='./data/raw_audios/', help='Path to .wav files')
parser.add_argument('--spectrogram_path', type=str, default='./data/spectrogram', help='Path to output spectrograms')

opts = parser.parse_args()

    
make_dir(opts.spectrogram_path)

# convert the audio .wav files to spectrograms and split them into fixed lengths
for inst in instrument_list:
    path = os.path.join(opts.raw_audio_path, inst)
    raw_audio_list = glob(f'{path}/*.wav')
    output_path = os.path.join(opts.spectrogram_path, inst)
    make_dir(output_path)
    for audio in raw_audio_list:
        arr_count = 0 # number of saved .npy
        audio_file_name = audio.split('/')[-1]
        spec= audio_to_melspec(audio)
        spec_len = spec.shape[1]
        for i in range(spec_len//fixed_len):
            # the chunk is with name [instrument name]_[raw audio file name]_[order of the chunk].npy. 
            np.save(output_path+f'/{inst}_{audio_file_name}_{arr_count}.npy',spec[:,fixed_len*i:fixed_len*(i+1)])
            arr_count += 1
                    