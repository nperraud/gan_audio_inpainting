import numpy as np
from matplotlib import pyplot as plt
import os
from wavio import read
from scipy.signal import firwin


        
        
def downsample3(sig,Nwin=32):
    win = firwin(numtaps=Nwin, cutoff=0.55)
    new_sig = sig.copy()
    new_sig = np.convolve(new_sig,win, 'same')
    new_sig = new_sig[2::3]
    return new_sig

def normalize(x):
    m = np.max(np.abs(x))
    return x/m

def toint16(x):
    return np.int16(x*(2**15))

def tomono(x):
    return (x[:,0]+x[:,1])/2

if __name__ == '__main__':

    pathaudio = '../data/piano/train'

    files = os.listdir(pathaudio)
    paths = []
    for file in files:
        if file[-3:]=='wav':
            paths.append(os.path.join(pathaudio, file))

    waveforms = np.array([], dtype=np.int16)
    for i, filepath in enumerate(paths):
        wavobj = read(paths[10])
        fs = wavobj.rate
        assert(fs==48000)
        waveform = wavobj.data.copy()
        waveform = normalize(waveform)
        waveform = tomono(waveform)
        waveform = downsample3(waveform)
        waveform = toint16(waveform)
        waveforms = np.append(waveforms, waveform, axis=0)

    np.savez('../data/piano/piano-train',waveforms)