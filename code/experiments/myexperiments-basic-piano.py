
#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../')

# No GPU because working locally
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""

import time
import numpy as np
import tensorflow as tf
from gantools import utils
from gantools import plot
#from gantools.model_extend import InpaintingGAN
from gantools.gansystem import GANsystem

import matplotlib.pyplot as plt
from copy import deepcopy
from gantools import blocks
from audioinpainting.load import load_audio_dataset
from audioinpainting.model_basic import InpaintingGAN

# # Parameters

downscale = 2

# # Data handling
# Load the data 
start = time.time()
# dataset = data.load.load_audio_dataset(scaling=downscale)
dataset = load_audio_dataset(scaling=downscale, type='piano', spix=1024*16, augmentation=True)

print('Number of samples: {}'.format(dataset.N))


# =============================================================================
# # The dataset can return an iterator.
# it = dataset.iter(10)
# print(next(it).shape)
# del it
# 
# # Get all the data
# X = dataset.get_all_data().flatten()
# 
# plt.hist(X, 100)
# print('min: {}'.format(np.min(X)))
# print('max: {}'.format(np.max(X)))
# plt.yscale('log')
# 
# # to free some memory
# del X
# 
# # Let us plot 16 samples
# 
# plot.audio.plot_signals(dataset.get_samples(N=16),nx=4,ny=4);
# plt.suptitle("Real samples");
# 
# plot.audio.play_sound(dataset.get_samples(16)[0,:], fs=14700//downscale)
# =============================================================================

#%%
# # Define parameters for the WGAN

time_str = 'basic_piano'
#global_path = '../saved_results'
global_path = '/scratch/snx3000/aeltelt/saved_results_basic'

name = 'WGAN' + '_' + time_str

#%%
# ## Parameters

bn = False
signal_length = 1024*16
signal_split = [1024*6, 1024*4, 1024*6]
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


resume, params = utils.test_resume(False, params)

#%%
# # Build the model

wgan = GANsystem(InpaintingGAN, params)

# # Train the model

wgan.train(dataset, resume=resume)

end = time.time()
print('Elapse time: {} minutes'.format((end - start)/60))

# =============================================================================
# #%%
# # # Generate new samples
# # To have meaningful statistics, be sure to generate enough samples
# # * 2000 : 32 x 32
# # * 500 : 64 x 64
# # * 200 : 128 x 128
# # 
# 
# N = 16 # Number of samples
# real_signals = dataset.get_samples(N=N)
# border1 = real_signals[:,:signal_split[0]]
# border2 = real_signals[:,-signal_split[2]:]
# borders = np.stack([border1, border2], axis=2)
# fake_signals = np.squeeze(wgan.generate(N=N, borders=borders))
# 
# #%%
# # Display a few fake samples
# 
# import ltfatpy
# from ltfatpy import plotdgtreal
# def plot_sgram(signal, a = 256, M = 512, g='itersine', dynrange=80, **kwargs):
#     c = ltfatpy.gabor.dgtreal.dgtreal(signal, g, a, M)[0]
#     return plotdgtreal(c, a, M, dynrange=dynrange,**kwargs)
# 
# for i in range(4):
#     print('Real')
#     plot.audio.play_sound(real_signals[i,:], fs=14700//downscale)    
#     print('Fake')
#     plot.audio.play_sound(fake_signals[i,:], fs=14700//downscale)
# 
# plot.audio.plot_signals(fake_signals,nx=4,ny=4);
# plt.suptitle("Fake samples");
# 
# plot.audio.plot_signals(real_signals,nx=4,ny=4);
# plt.suptitle("Real samples");
# 
# for i in range(4):
#     plt.figure(figsize=(15, 4))
#     plt.subplot(121)
#     plot_sgram(fake_signals[i].astype(np.float64), fs=14700//downscale);
#     plt.title('Inpainted')
#     plt.subplot(122)
#     plot_sgram(real_signals[i].astype(np.float64), fs=14700//downscale);
#     plt.title('Original')
# =============================================================================





