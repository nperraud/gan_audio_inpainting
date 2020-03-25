
#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../')

# No GPU because working locally
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""

import time
import tensorflow as tf
from gantools import utils
from gantools.model import InpaintingGAN
from gantools.gansystem import GANsystem

from gantools import blocks
from audioinpainting.load_generator import Dataset_maestro

# # Parameters

downscale = 2

# # Data handling
# Load the data

start = time.time()
# dataset = data.load.load_audio_dataset(scaling=downscale)
dataset = Dataset_maestro(scaling=downscale, spix=1024*16, augmentation=True, maxsize=2, type='maestro', path='../data', fs_rate=48000, files=8, preprocessing=False)
print('Number of samples: {}'.format(dataset.N))

#%%
# # Define parameters for the WGAN
time_str = 'basic_maestro'
global_path = '/saved_results_basic'

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
