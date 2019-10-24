import sys
sys.path.insert(0, '../')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import scipy.io

from gantools import data
from gantools import utils
from gantools import plot
from gantools.model import WGAN, InpaintingGAN
from gantools.data.Dataset import Dataset
from gantools.gansystem import GANsystem
from gantools.data import fmap
import functools
import matplotlib.pyplot as plt
from copy import deepcopy
downscale = 1

#mat_path = "../data/piano_yiruma_spectrograms_256.mat"
#raw_data = scipy.io.loadmat(mat_path)
#preprocessed_images = raw_data['logspecs']
#print(preprocessed_images.shape)
#print(np.max(preprocessed_images[:256, :]))
#print(np.min(preprocessed_images[:256, :]))
#print(np.mean(preprocessed_images[:256, :]))

#def slice_function(images):
#    hop_size = 32
#    print('count: ', int(images.shape[1]/hop_size))
#    extended_array = np.zeros([int((images.shape[1]-128*3)/hop_size+1), 256, 128*3])
#    for array_index, index_shift in enumerate(np.arange(0, (images.shape[1]-128*3), hop_size)):
#        extended_array[array_index] = images[:256, index_shift:index_shift+128*3]
#    print(extended_array.shape)
#    print(np.mean(extended_array[-1, :]))
#    print(np.std(extended_array[-1, :]))
#    return extended_array

#song_count = 21
#cut_every = preprocessed_images.shape[1]/song_count

#train_cut_length = preprocessed_images.shape[1]*9/10/song_count
#valid_cut_length = preprocessed_images.shape[1]*1/10/song_count

#train_examples = np.zeros([257, 0])
#valid_examples = np.zeros([257, 0])

#for i in range(song_count):
#    train_examples = np.append(train_examples, np.append(preprocessed_images[:, int(i*(train_cut_length+valid_cut_length)):int(i*(train_cut_length+valid_cut_length)+train_cut_length)], np.zeros([257, 128*2])-1, axis=1), axis=1)
#    valid_examples = np.append(valid_examples, np.append(preprocessed_images[:, int(i*(train_cut_length+valid_cut_length)+train_cut_length):int((i+1)*(train_cut_length+valid_cut_length))], np.zeros([257, 128*2])-1, axis=1), axis=1)


#train_dataset = Dataset(slice_function(train_examples))
#valid_dataset = Dataset(slice_function(valid_examples))

global_path = '../saved_results'

name = 'yiruna_160_64_tfrecord'

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
from gantools.model import InpaintingGAN

wgan = GANsystem(InpaintingGAN, params)


def read_tfrecord(serialized_example):
    feature_description = {
        'train/window': tf.io.FixedLenFeature((), tf.string)}
    example = tf.io.parse_single_example(serialized_example, feature_description)
    spectrogram = tf.reshape(tf.decode_raw(example['train/window'], tf.float32), [256, 384])#/(10/2)+1

    return spectrogram

num_epochs = 300

dataset = tf.data.TFRecordDataset("../data/yiruma_train_inpainting_w384_h32_27261.tfrecords")
dataset = dataset.shuffle(buffer_size=128*20)
dataset = dataset.repeat(num_epochs)
dataset = dataset.map(map_func=read_tfrecord)#, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.batch(batch_size=128)
dataset = dataset.prefetch(buffer_size=1) # this should be the last transformation
dataset.N = 27261

wgan.train(dataset, resume=resume)





