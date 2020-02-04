
#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../')

import os
# No GPU because working locally
os.environ["CUDA_VISIBLE_DEVICES"]=""

import numpy as np
import pandas as pd
from scipy.io import wavfile
import tensorflow as tf
import matplotlib.pyplot as plt
from gantools import utils
from gantools import plot
from gantools.gansystem import GANsystem
from gantools import blocks
from audioinpainting.load_generator import Dataset_maestro

import ltfatpy
from ltfatpy import plotdgtreal

## Trained model
downscale = 2
model = 'extend'
type = 'solo'

## Used dataset
phase = 'test'
path = '../../data/piano'   # Path to the dataset
data = 'piano' # Dataset to test model
fs_rate = 48000 # 44100

## Define parameters for the WGAN
time_str = '{}_{}'.format(model, type)
global_path = '../../../saved_results'
name = 'WGAN' + '_' + time_str
if not os.path.exists(os.path.join(global_path, name + '_checkpoints/')):
    raise ValueError('Path to the checkpoints of the model is wrong')

# # Define parameters to generate fake samples
N_f = 16 # Number of generated samples

# # Data handling
print('Load the data for model {}'.format(model))
if model=='extend':
    from audioinpainting.model_extend import InpaintingGAN
    spix = 1024*52
    signal_length = 1024 * 52
    signal_split = [1024 * 18, 1024 * 6, 1024 * 4, 1024 * 6, 1024 * 18]
elif model =='basic':
    from audioinpainting.model_basic import InpaintingGAN
    spix = 1024*16
    signal_length = 1024 * 16
    signal_split = [1024 * 6, 1024 * 4, 1024 * 6]
else:
    raise ValueError('Incorrect model; choose either "extend" or "basic"')

dataset = Dataset_maestro(scaling=downscale, spix=spix, augmentation=True, maxsize=2, phase=phase, type=data, path=path, fs_rate=fs_rate, files=8, preprocessing=False)
# Check whether number of generated samples is consistent with total number of samples
if N_f > dataset.N:
    N_f = dataset.N
print('Number of samples: {}'.format(dataset.N))


# ## Parameters
bn = False
md = 64

params_discriminator = dict()
params_discriminator['stride'] = [4,4,4,4,4]
params_discriminator['nfilter'] = [md, 2*md, 4*md, 8*md, 16*md]
params_discriminator['shape'] = [[25], [25], [25], [25], [25]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = [md*4]
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 1
params_discriminator['apply_phaseshuffle'] = True 
params_discriminator['spectral_norm'] = True
params_discriminator['activation'] = blocks.lrelu


params_generator = dict()
params_generator['stride'] = [4, 4, 4, 4, 4]
params_generator['latent_dim'] = 100
params_generator['nfilter'] = [8*md, 4*md, 2*md, md, 1]
params_generator['shape'] = [[25], [25], [25], [25], [25]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [64*md]
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.tanh
params_generator['activation'] = tf.nn.relu
params_generator['data_size'] = 1
params_generator['spectral_norm'] = True 
params_generator['in_conv_shape'] =[4]

params_generator['borders'] = dict()
params_generator['borders']['nfilter'] = [md, 2*md, 4*md, 8*md, 2*md]
params_generator['borders']['batch_norm'] = [bn, bn, bn, bn, bn]
params_generator['borders']['shape'] = [[25], [25], [25], [25], [25]]
params_generator['borders']['stride'] = [4, 4, 4, 4, 4]
params_generator['borders']['data_size'] = 1
params_generator['borders']['width_full'] = 128
params_generator['borders']['activation'] = tf.nn.relu


params_optimization = dict()
params_optimization['batch_size'] = 64
params_optimization['epoch'] = 10000
params_optimization['n_critic'] = 5
params_optimization['generator'] = dict()
params_optimization['generator']['optimizer'] = 'adam'
params_optimization['generator']['kwargs'] = {'beta1':0.5, 'beta2':0.9}
params_optimization['generator']['learning_rate'] = 1e-4
params_optimization['discriminator'] = dict()
params_optimization['discriminator']['optimizer'] = 'adam'
params_optimization['discriminator']['kwargs'] = {'beta1':0.5, 'beta2':0.9}
params_optimization['discriminator']['learning_rate'] = 1e-4


# all parameters
params = dict()
params['net'] = dict() # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['discriminator'] = params_discriminator
params['net']['prior_distribution'] = 'gaussian'
params['net']['shape'] = [signal_length, 1] # Shape of the image
params['net']['inpainting'] = dict()
params['net']['inpainting']['split'] = signal_split
params['net']['gamma_gp'] = 10 # Gradient penalty
params['net']['fs'] = 16000//downscale
params['net']['loss_type'] ='wasserstein'

params['optimization'] = params_optimization
params['summary_every'] = 100 # Tensorboard summaries every ** iterations
params['print_every'] = 50 # Console summaries every ** iterations
params['save_every'] = 1000 # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 0


resume, params = utils.test_resume(True, params)

# Build the model
print('Load the model')
wgan = GANsystem(InpaintingGAN, params)


# Generate new samples
print('Generate new samples')
real_signals = dataset.get_samples(N=N_f)
if model == 'extend':
    border1 = real_signals[:,signal_split[0]:(signal_split[0]+signal_split[1])]
    border2 = real_signals[:,-(signal_split[3]+signal_split[4]):-signal_split[4]]
    border3 = real_signals[:,:(signal_split[0]+signal_split[1])]
    border4 = real_signals[:,-(signal_split[3]+signal_split[4]):]
    borders1 = np.stack([border1, border2], axis=2)
    borders2 = np.stack([border3, border4], axis=2)
    fake_signals = np.squeeze(wgan.generate(N=N_f, borders1=borders1, borders2=borders2)[1], axis=2)
elif model == 'basic':
    border1 = real_signals[:, :signal_split[0]]
    border2 = real_signals[:,-signal_split[2]:]
    borders = np.stack([border1, border2], axis=2)
    fake_signals = np.squeeze(wgan.generate(N=N_f, borders=borders))


# Calculating Signal to Noise Ratio (SNR)
def _pavlovs_SNR(y_orig, y_inp, onAxis=(1,)):
    norm_y_orig = _squaredEuclideanNorm(y_orig, onAxis)
    norm_y_orig_minus_y_inp = _squaredEuclideanNorm(y_orig - y_inp, onAxis)
    SNRs = 10 * np.log10(norm_y_orig / norm_y_orig_minus_y_inp)
    SNRs[np.isinf(SNRs)] = 0
    SNRs_pd = pd.DataFrame(columns=['SNR'])
    SNRs_pd['SNR'] = SNRs
    SNRs_pd.loc['mean'] = np.mean(SNRs, axis=0)
    SNRs_pd.loc['std'] = np.std(SNRs, axis=0)
    return SNRs_pd


def _squaredEuclideanNorm(vector, onAxis=(1,)):
    squared = np.square(vector)
    summed = np.sum(squared, axis=onAxis)
    return summed


def plot_sgram(signal, a = 256, M = 512, g='itersine', dynrange=80, **kwargs):
    c = ltfatpy.gabor.dgtreal.dgtreal(signal, g, a, M)[0]
    return plotdgtreal(c, a, M, dynrange=dynrange,**kwargs)


def save_sound(x, fs, filename):
    wavfile.write(filename, np.int(fs), (x * (2 ** 15)).astype(np.int16))


# Calculating Signal to Noise Ratio (SNR)
print('Calculate SNRs')
path_snr = os.path.join(path,'results_{}_{}/snr/'.format(model, type))
if not os.path.exists(path_snr):
    os.makedirs(path_snr)
SNRs = _pavlovs_SNR(real_signals, fake_signals)
SNRs.to_csv('{}/SNRs.csv'.format(path_snr), index=True)


# Save sound file
print('Save sound files')
path_wav = os.path.join(path,'results_{}_{}/wav/'.format(model, type))
if not os.path.exists(path_wav):
    os.makedirs(path_wav)
for i in range(N_f):
    # Real
    save_sound(real_signals[i,:], fs=16000//downscale, filename='{}/real_{}.wav'.format(path_wav, i))
    # Fake
    save_sound(fake_signals[i,:], fs=16000//downscale, filename='{}/fake_{}.wav'.format(path_wav, i))


# Display a few fake samples
print('Display a few real and fake samples')
path_fig = os.path.join(path,'results_{}_{}/fig/'.format(model, type))
if not os.path.exists(path_fig):
    os.makedirs(path_fig)
plot.audio.plot_signals(real_signals,nx=4,ny=4);
plt.suptitle("Real samples")
plt.savefig('{}/real.png'.format(path_fig))
plot.audio.plot_signals(fake_signals,nx=4,ny=4);
plt.suptitle("Fake samples")
plt.savefig('{}/fake.png'.format(path_fig))


# Display magnitude spectrogram
print('Display a few real and fake magnitude spectrogram')
path_sgram = os.path.join(path,'results_{}_{}/sgram/'.format(model, type))
if not os.path.exists(path_sgram):
    os.makedirs(path_sgram)
for i in range(N_f):
    plt.figure(figsize=(15, 4))
    plt.subplot(121)
    plot_sgram(fake_signals[i].astype(np.float64), fs=16000//downscale);
    plt.title('Inpainted')
    plt.subplot(122)
    plot_sgram(real_signals[i].astype(np.float64), fs=16000//downscale);
    plt.title('Original')
    plt.savefig('{}/sgram_{}.png'.format(path_sgram, i))
    plt.close()

