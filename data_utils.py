# References
# https://github.com/hmartelb/Pix2Pix-Timbre-Transfer/blob/master/code/data.py
import librosa 
import numpy as np

def load_audio(filename, sr=44100):    
    return librosa.core.load(filename, sr=sr)[0]

def forward_transform(audio, nfft=1024, normalize=True, crop_hf=True):
    window = np.hanning(nfft)
    S = librosa.stft(audio, n_fft=nfft, hop_length=int(nfft/2), window=window)
    mag, phase = np.abs(S), np.angle(S)
    if(crop_hf):
        mag = remove_hf(mag)
    if(normalize):
        mag = 2 * mag / np.sum(window)
    return mag, phase

def amplitude_to_db(mag, amin=1/(2**16), normalize=True):
    mag_db = 20*np.log1p(mag/amin)
    if(normalize):
        mag_db /= 20*np.log1p(1/amin)
    return mag_db

def slice_magnitude(mag, slice_size):
    magnitudes = np.stack([mag], axis=2)
    return slice_first_dim(magnitudes, slice_size)

def remove_hf(mag):
    return mag[0:int(mag.shape[0]/2), :]

def slice_first_dim(array, slice_size):
    n_sections = int(np.floor(array.shape[1]/slice_size))
    has_last_mag = n_sections*slice_size < array.shape[1]

    last_mag = np.zeros(shape=(1, array.shape[0], slice_size, array.shape[2]))
    last_mag[:,:,:array.shape[1]-(n_sections*slice_size),:] = array[:,n_sections*int(slice_size):,:]
    
    if(n_sections > 0):
        array = np.expand_dims(array, axis=0)
        sliced = np.split(array[:,:,0:n_sections*slice_size,:], n_sections, axis=2)
        sliced = np.concatenate(sliced, axis=0)
        if(has_last_mag): # Check for reminder
            sliced = np.concatenate([sliced, last_mag], axis=0)
    else:
        sliced = last_mag
    return sliced