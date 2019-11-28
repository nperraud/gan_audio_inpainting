"""Threaded generator.
    This is an example to transform a generator into a separate process.
    This is useful if your generator take some time. As an example, it can be
    ued to loads data from the disk and does some preprocessing.
"""
#!/usr/bin/env python
# coding: utf-8

import multiprocessing
import os
import time
import numpy as np
from wavio import read
from scipy.signal import firwin
from gantools.data import transformation
from gantools.utils import compose2
from functools import partial
import pandas as pd
import itertools

data_root_path = '/scratch/snx3000/pebner/gan_audio_inpainting/data'

def do_nothing(x):
    return x


def get_data(scaling=1, smooth=None, phase='train'):

    def load_maestro_paths(phase='train'):
        pathdata = os.path.join(data_root_path, 'maestro-v2.0.0')
        pathdata_csv = os.path.join(pathdata, 'maestro-v2.0.0.csv')
        wave_paths = pd.read_csv(pathdata_csv, delimiter=',')
        wave_paths = sorted([os.path.join(pathdata, path) for idx, path in enumerate(wave_paths['audio_filename']) if
                      wave_paths['split'][idx] == phase])
        return wave_paths

    def load_piano_paths():
        wave_paths = [f for f in os.listdir('../../data/piano/train') if f.endswith('.wav')]
        wave_paths = sorted([os.path.join('../../data/piano/train',f) for f in wave_paths])
        return wave_paths

    def normalize(x):
        m = np.max(np.abs(x))
        return x / m

    def tomono(x):
        return (x[:, 0] + x[:, 1]) / 2

    def downsample3(sig, Nwin=32):
        win = firwin(numtaps=Nwin, cutoff=0.55)
        new_sig = sig.copy()
        new_sig = np.convolve(new_sig, win, 'same')
        new_sig = new_sig[2::3]
        return new_sig

    def toint16(x):
        return np.int16(x * (2 ** 15))

    def transform(x):
        x = x/(2**15)
        x = (0.99*x.T/np.max(np.abs(x), axis=1)).T
        return x

    # Load path to wave files

    wave_paths = load_maestro_paths(phase)
    #wave_paths = load_piano_paths()

    for wave_path in wave_paths:
        wavobj = read(wave_path)
        fs = wavobj.rate
        assert (fs == 44100)
        waveform = wavobj.data.copy()
        waveform = normalize(waveform)
        waveform = tomono(waveform)
        waveform = downsample3(waveform)
        sig = toint16(waveform)

        if len(sig.shape) == 1:
            sig = np.reshape(sig, [1, len(sig)])

        # Transform data
        sig = transform(sig)

        # Downsample
        Nwin = 32
        if scaling > 1:
            # sig = blocks.downsample(sig, scaling)
            sig = transformation.downsample_1d(sig, scaling, Nwin=Nwin)

        if smooth is not None:
            sig = sig[:, :(sig.shape[1] // smooth) * smooth]
            sig_down = transformation.downsample_1d(sig, smooth, Nwin=Nwin)
            sig_smooth = transformation.upsamler_1d(sig_down, smooth, Nwin=Nwin)

            sig = np.concatenate((np.expand_dims(sig, axis=2), np.expand_dims(sig_smooth, axis=2)), axis=2)
        print('Get data {}'.format(wave_path))
        time.sleep(0.5)
        yield sig


def queued_generator(data, maxsize=2):
    """
    Transform a generator in a threaded generator
    :param data: data
    :param maxsize: maximum number of data stored in the queue
    :return: preprocess data
    """
    def put_data_in_queue(queue, data):
        for dat in data:
            queue.put(dat)
        queue.put('DONE')

    queue = multiprocessing.Queue(maxsize=maxsize)

    reader_p = multiprocessing.Process(target=put_data_in_queue, args=(queue, data))
    reader_p.daemon = True  # have to be set before .start(); when script ends its job will kill all subprocess.
    reader_p.start()    # Launch reader_proc() as a separate python process

    while True:
        out = queue.get()
        if out == 'DONE':
            return
        else:
            yield out


class Dataset_maestro(object):
    ''' Dataset oject for GAN and CosmoGAN classes

        Transform should probably be False for a classical GAN.
    '''

    def __init__(self, phase='train', scaling=1, smooth=None, shuffle=True, patch=False, spix=None, augmentation=False, maxsize=2, dtype=np.float32):
        ''' Initialize a Dataset object

        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to the sliced dataset
                      This allows extend the dataset.
        * slice_fn : Slicing function to cut the data into smaller parts
        '''
        self.dtype = dtype
        self.scaling = scaling
        self.smooth = smooth
        self.phase = phase
        self.maxsize = maxsize

        self.my_generator = queued_generator(get_data(self.scaling, self.smooth, self.phase), maxsize=self.maxsize)
        self._X = next(self.my_generator).astype(self.dtype)
        self.X = self._X
        self._shuffle = shuffle
        if patch:

            self._slice_fn = partial(transformation.slice_1d_patch, spix=spix)
        else:
            if spix is not None:
                self._slice_fn = partial(transformation.slice_1d, spix=spix)
            else:
                self._slice_fn = do_nothing

        if augmentation:
            self._transform = partial(transformation.random_shift_1d, roll=False, spix=spix)
        else:
            self._transform = do_nothing

        self._data_process = compose2(self._transform, self._slice_fn)
        self._N = len(self._data_process(self._X))
        self.n = self._N
        if shuffle:
            self._p = np.random.permutation(self._N)
        else:
            self._p = np.arange(self._N)

    def get_all_data(self):
        ''' Return all the data (shuffled) '''
        return self._data_process(self._X)[self._p]

    def get_samples(self, N=100):
        ''' Get the `N` first samples '''
        return self._data_process(self._X)[self._p[:N]]

    # TODO: kwargs to be removed
    def iter(self, batch_size=1, **kwargs):
        return self.__iter__(batch_size, **kwargs)

    # TODO: kwargs to be removed
    def __iter__(self, batch_size=1, **kwargs):
        self.my_gen = queued_generator(get_data(self.scaling, self.smooth, self.phase), maxsize=self.maxsize)
        for data_gen in self.my_gen:

            self._X = data_gen.astype(self.dtype)
            self._N = len(self._data_process(self._X))
            self._p = np.arange(self._N)

            # Batch size greater than total number of samples available -> concatenate data
            while batch_size > self.N:

                self._X = np.concatenate((self._X, next(self.my_gen).astype(self.dtype)), axis=1)
                self._N = len(self._data_process(self._X))
                self._p = np.arange(self._N)

            # Reshuffle the data
            if self.shuffle:
                self._p = np.random.permutation(self._N)
            nel = (self._N // batch_size) * batch_size
            transformed_data = self._data_process(self._X)[self._p[range(nel)]]
            for data in grouper(transformed_data, batch_size):
                yield np.array(data)


    @property
    def shuffle(self):
        ''' Is the dataset suffled? '''
        return self._shuffle

    @property
    def N(self):
        ''' Number of element in the dataset '''
        return self._N


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks. This function commes
    from itertools
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)



