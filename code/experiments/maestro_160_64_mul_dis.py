import sys
sys.path.insert(0, '../')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
import scipy.io
from gantools import data
from gantools import utils
from gantools import plot
from gantools.model import WGAN, MultipleDiscrimnatorInpaintingGAN
from gantools.data.Dataset import Dataset
from gantools.gansystem import GANsystem
from gantools.data import fmap
import functools
import matplotlib.pyplot as plt
from copy import deepcopy
downscale = 1


global_path = '../saved_results'

name = 'maestro_160_64_multiple_dis_ncritic5'

from gantools import blocks
bn = False
signal_split = [160, 64, 160]
md = 32

params_discriminator = dict()
params_discriminator['stride'] = [2,2,2,2,2]
params_discriminator['nfilter'] = [md, 2*md, 4*md, 8*md, 16*md]
params_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = []
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 2
params_discriminator['apply_phaseshuffle'] = True
params_discriminator['spectral_norm'] = True
params_discriminator['activation'] = blocks.lrelu

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 2]
params_generator['latent_dim'] = 100
params_generator['nfilter'] = [8*md, 4*md, 2*md, md, 1]
params_generator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [256*md]
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.tanh
params_generator['activation'] = tf.nn.relu
params_generator['data_size'] = 2
params_generator['spectral_norm'] = True 
params_generator['in_conv_shape'] =[8, 2]
params_generator['borders'] = dict()
params_generator['borders']['nfilter'] = [md, 2*md, md, md/2]
params_generator['borders']['batch_norm'] = [bn, bn, bn, bn]
params_generator['borders']['shape'] = [[5, 5],[5, 5],[5, 5],[5, 5]]
params_generator['borders']['stride'] = [2, 2, 3, 4]
params_generator['borders']['data_size'] = 2
# This does not work because of flipping, border 2 need to be flipped tf.reverse(l, axis=[1]), ask Nathanael 
params_generator['borders']['width_full'] = None 
params_generator['borders']['activation'] = tf.nn.relu


# Optimization parameters inspired from 'Self-Attention Generative Adversarial Networks'
# - Spectral normalization GEN DISC
# - Batch norm GEN
# - TTUR ('GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium')
# - ADAM  beta1=0 beta2=0.9, disc lr 0.0004, gen lr 0.0001
# - Hinge loss
# Parameters are similar to the ones in those papers...
# - 'PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION'
# - 'LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS'
# - 'CGANS WITH PROJECTION DISCRIMINATOR'

params_optimization = dict()
params_optimization['batch_size'] = 64*2
params_optimization['epoch'] = 600
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
params['net']['shape'] = [256, 128*3, 1] # Shape of the image
params['net']['inpainting']=dict()
params['net']['inpainting']['split']=signal_split
params['net']['gamma_gp'] = 10 # Gradient penalty
params['net']['fs'] = 16000//downscale
params['net']['loss_type'] ='wasserstein'

params['optimization'] = params_optimization
params['summary_every'] = 250 # Tensorboard summaries every ** iterations
params['print_every'] = 50 # Console summaries every ** iterations
params['save_every'] = 1000 # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 500

resume, params = utils.test_resume(True, params)
from gantools.model import MultipleDiscrimnatorInpaintingGAN

wgan = GANsystem(MultipleDiscrimnatorInpaintingGAN, params)


def read_tfrecord(serialized_example):
    feature_description = {
        'train/window': tf.io.FixedLenFeature((), tf.string)}
    example = tf.io.parse_single_example(serialized_example, feature_description)
    spectrogram = tf.reshape(tf.decode_raw(example['train/window'], tf.float32), [256, 384])

    return spectrogram

num_epochs = 10

dataset = tf.data.TFRecordDataset("../data/Maestro_train_inpainting_w384_h32.tfrecords")
dataset = dataset.shuffle(buffer_size=40000)
dataset = dataset.repeat(num_epochs)
dataset = dataset.map(map_func=read_tfrecord)#, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.batch(batch_size=128)
dataset = dataset.prefetch(buffer_size=1) # this should be the last transformation
dataset.N = 2837745

wgan.train(dataset, resume=resume)

