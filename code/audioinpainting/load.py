import numpy as np
import os
from .path import data_root_path
from gantools.data import Dataset
from gantools.data import transformation
from functools import partial




def do_nothing(x):
    return x

def load_solo_rawdata():
    pathdata = os.path.join(data_root_path(), 'guitar/guitar-train.npz')
    return np.load(pathdata)['arr_0']

def load_piano_rawdata():
    pathdata = os.path.join(data_root_path(), 'piano/piano-train.npz')
    return np.load(pathdata)['arr_0']


def load_audio_dataset(
        shuffle=True, scaling=1, patch=False, augmentation=False, spix=None, smooth=None, type='solo'):
    ''' Load a Nsynth dataset object.

     Arguments
    ---------
    * shuffle: shuffle the data (default True)
    * scaling : downscale the image by a factor (default 1)
    * path : downscale the image by a factor (default 1)
    * scaling : downscale the image by a factor (default 1)
    '''

    if type == 'solo':
        sig = load_solo_rawdata()
        #sig = sig[:, :2**15]
    elif type == 'piano':
        sig = load_piano_rawdata()
    else:
        raise ValueError('Incorrect value for type')

    if len(sig.shape)==1:
        sig = np.reshape(sig, [1,len(sig)])

    # if augmentation and (not patch):
    #     raise ValueError('Augementation works only with patches.')
    
    # 1) Transform the data
    def transform(x):
        x = x/(2**15)
        x = (0.99*x.T/np.max(np.abs(x), axis=1)).T
        return x
    sig = transform(sig)


    # 2) Downsample
    Nwin = 32
    if scaling>1:
        # sig = blocks.downsample(sig, scaling)
        sig = transformation.downsample_1d(sig, scaling, Nwin=Nwin)

    if smooth is not None:
        sig = sig[:, :(sig.shape[1]//smooth)*smooth]
        sig_down = transformation.downsample_1d(sig, smooth, Nwin=Nwin)
        sig_smooth = transformation.upsamler_1d(sig_down, smooth, Nwin=Nwin)

        sig = np.concatenate((np.expand_dims(sig, axis=2), np.expand_dims(sig_smooth, axis=2)), axis=2)
    if patch:

        slice_fn = partial(transformation.slice_1d_patch, spix=spix)
    else:
        if spix is not None:
            slice_fn = partial(transformation.slice_1d, spix=spix)
        else:
            slice_fn = do_nothing

    if augmentation:
        transform = partial(transformation.random_shift_1d, roll=False, spix=spix)
    else:
        transform = do_nothing
    # 3) Make a dataset
    dataset = Dataset(sig, shuffle=shuffle, transform=transform, slice_fn=slice_fn)

    return dataset

