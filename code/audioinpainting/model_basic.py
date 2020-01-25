import tensorflow as tf
import numpy as np
# The next import should be changed
from gantools.blocks import * 
from gantools import utils
from tfnntools.model import BaseNet, rprint
from gantools.plot import colorize
from gantools.metric import ganlist
from gantools.data.transformation import tf_flip_slices, tf_patch2img, get_attenuation_weights
from gantools.plot.plot_summary import PlotSummaryPlot
from copy import deepcopy

class BaseGAN(BaseNet):
    """Abstract class for the model."""
    def __init__(self, params=None, name='BaseGAN'):
        if params is None:
            params = {}
        self.G_fake = None
        self.D_real = None
        self.D_fake = None
        self._D_loss = None
        self._G_loss = None
        self._summary = None
        self._constraints = []
        super().__init__(params=params, name=name)
        self._loss = (self.D_loss, self.G_loss)

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss

    @property
    def summary(self):
        return self._summary

    @property
    def has_encoder(self):
        return False

    @property
    def constraints(self):
        return self._constraints
    

    def sample_latent(self, N):
        raise NotImplementedError("This is a an abstract class.")


class WGAN(BaseGAN):
    def default_params(self):
        d_params = deepcopy(super().default_params())
        d_params['shape'] = [16, 16, 1] # Shape of the image
        d_params['prior_distribution'] = 'gaussian' # prior distribution
        d_params['gamma_gp'] = 10 
        d_params['loss_type'] = 'wasserstein'  # 'hinge' or 'wasserstein'
        d_params['fs'] = 16000  # only for 1d signal
        

        bn = False

        d_params['generator'] = dict()
        d_params['generator']['latent_dim'] = 100
        d_params['generator']['full'] = [2*8 * 8]
        d_params['generator']['nfilter'] = [2, 32, 32, 1]
        d_params['generator']['batch_norm'] = [bn, bn, bn]
        d_params['generator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
        d_params['generator']['stride'] = [1, 2, 1, 1]
        d_params['generator']['in_conv_shape'] = None
        d_params['generator']['summary'] = True
        d_params['generator']['data_size'] = 2 # 1 for 1D signal, 2 for images, 3 for 3D
        d_params['generator']['inception'] = False # Use inception module
        d_params['generator']['residual'] = False # Use residual connections
        d_params['generator']['activation'] = lrelu # leaky relu
        d_params['generator']['one_pixel_mapping'] = [] # One pixel mapping
        d_params['generator']['non_lin'] = tf.nn.relu # non linearity at the end of the generator
        d_params['generator']['spectral_norm'] = False # use spectral norm

        d_params['discriminator'] = dict()
        d_params['discriminator']['full'] = [32]
        d_params['discriminator']['nfilter'] = [16, 32, 32, 32]
        d_params['discriminator']['batch_norm'] = [bn, bn, bn, bn]
        d_params['discriminator']['shape'] = [[5, 5], [5, 5], [5, 5], [3, 3]]
        d_params['discriminator']['stride'] = [2, 2, 2, 1]
        d_params['discriminator']['summary'] = True
        d_params['discriminator']['data_size'] = 2 # 1 for 1D signal, 2 for images, 3 for 3D
        d_params['discriminator']['inception'] = False # Use inception module
        d_params['discriminator']['activation'] = lrelu # leaky relu
        d_params['discriminator']['one_pixel_mapping'] = [] # One pixel mapping
        d_params['discriminator']['non_lin'] = None # non linearity at the beginning of the discriminator
        d_params['discriminator']['cdf'] = None # cdf
        d_params['discriminator']['cdf_block'] = None # non linearity at the beginning of the discriminator
        d_params['discriminator']['moment'] = None # non linearity at the beginning of the discriminator
        d_params['discriminator']['minibatch_reg'] = False # Use minibatch regularization
        d_params['discriminator']['spectral_norm'] = False # use spectral norm
        d_params['discriminator']['fft_features'] = False
        d_params['discriminator']['psd_features'] = False


        return d_params

    def __init__(self, params, name='wgan'):
        super().__init__(params=params, name=name)
        self._summary = tf.summary.merge(tf.get_collection("model"))


    def _build_generator(self):
        shape = self._params['shape']
        self.X_real = tf.placeholder(tf.float32, shape=[None, *shape], name='Xreal')
        self.z = tf.placeholder(
            tf.float32,
            shape=[None, self.params['generator']['latent_dim']],
            name='z')
        self.X_fake = self.generator(self.z, reuse=False)

    def _build_net(self):
        self._data_size = self.params['generator']['data_size']
        assert(self.params['discriminator']['data_size'] == self.data_size)
        
        reduction = stride2reduction(self.params['generator']['stride'])
        if self.params['generator']['in_conv_shape'] is None:
            in_conv_shape = [el//reduction for el in self.params['shape'][:-1]]
            self._params['generator']['in_conv_shape'] = in_conv_shape
  
        self._build_generator()
        self._D_fake = self.discriminator(self.X_fake, reuse=False)
        self._D_real = self.discriminator(self.X_real, reuse=True)
        self._D_loss_f = tf.reduce_mean(self._D_fake)
        self._D_loss_r = tf.reduce_mean(self._D_real)

        if self.params['loss_type'] == 'wasserstein':
            # Wasserstein loss
            gamma_gp = self.params['gamma_gp']
            print(' Wasserstein loss with gamma_gp={}'.format(gamma_gp))
            self._D_gp = self.wgan_regularization(gamma_gp, [self.X_fake], [self.X_real])
            self._D_loss = -(self._D_loss_r - self._D_loss_f) + self._D_gp
            self._G_loss = -self._D_loss_f
        elif self.params['loss_type'] == 'hinge':
            # Hinge loss
            print(' Hinge loss.')
            self._D_loss = tf.nn.relu(1-self._D_loss_r) + tf.nn.relu(self._D_loss_f+1)
            self._G_loss = -self._D_loss_f
        elif self.params['loss_type'] == 'normalized_wasserstein':            # Wasserstein loss
            gamma_gp = self.params['gamma_gp']
            print(' Wasserstein loss with gamma_gp={}'.format(gamma_gp))
            self._D_gp = self.wgan_regularization(gamma_gp, [self.X_fake], [self.X_real])
            reg = tf.nn.relu(self._D_loss_r*self._D_loss_f)
            self._D_loss = -(self._D_loss_r - self._D_loss_f) + self._D_gp + reg
            self._G_loss = -self._D_loss_f
            tf.summary.scalar("Disc/reg", reg, collections=["train"])

        else:
            raise ValueError('Unknown loss type!')    
        self._inputs = (self.z)
        self._outputs = (self.X_fake)


    def _add_summary(self):
        tf.summary.histogram('Prior/z', self.z, collections=['model'])
        self._build_image_summary()
        self._build_stat_summary()
        self._wgan_summaries()

    def generator(self, z, **kwargs):
        return generator(z, params=self.params['generator'], **kwargs)

    def discriminator(self, X, **kwargs):
        return discriminator(X, params=self.params['discriminator'], **kwargs) 

    def sample_latent(self, bs=1):
        latent_dim = self.params['generator']['latent_dim']
        return utils.sample_latent(bs, latent_dim, self._params['prior_distribution'])

    def wgan_regularization(self, gamma, list_fake, list_real):
        if not gamma:
            # I am not sure this part or the code is still useful
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            D_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
            self._constraints.append(D_clip)
            D_gp = tf.constant(0, dtype=tf.float32)
            print(" [!] Using weight clipping")
        else:
            # calculate `x_hat`
            assert(len(list_fake) == len(list_real))
            bs = tf.shape(list_fake[0])[0]
            eps = tf.random_uniform(shape=[bs], minval=0, maxval=1)

            x_hat = []
            for fake, real in zip(list_fake, list_real):
                singledim = [1]* (len(fake.shape.as_list())-1)
                eps = tf.reshape(eps, shape=[bs,*singledim])
                x_hat.append(eps * real + (1.0 - eps) * fake)

            D_x_hat = self.discriminator(*x_hat, reuse=True)

            # gradient penalty
            gradients = tf.gradients(D_x_hat, x_hat)
            norm_gradient_pen = tf.norm(gradients[0], ord=2)
            D_gp = gamma * tf.square(norm_gradient_pen - 1.0)
            tf.summary.scalar("Disc/GradPen", D_gp, collections=["train"])
            tf.summary.scalar("Disc/NormGradientPen", norm_gradient_pen, collections=["train"])
            print(" Using gradients penalty")

        return D_gp

    def _wgan_summaries(self):
        tf.summary.scalar("Disc/Neg_Loss", -self._D_loss, collections=["train"])
        tf.summary.scalar("Disc/Neg_Critic", self._D_loss_f - self._D_loss_r, collections=["train"])
        tf.summary.scalar("Disc/Loss_f", self._D_loss_f, collections=["train"])
        tf.summary.scalar("Disc/Loss_r", self._D_loss_r, collections=["train"])
        tf.summary.scalar("Gen/Loss", self._G_loss, collections=["train"])
   
    def _build_stat_summary(self):
        self._stat_list_real = ganlist.gan_stat_list('real')
        self._stat_list_fake = ganlist.gan_stat_list('fake')

        for stat in self._stat_list_real:
            stat.add_summary(collections="model")

        for stat in self._stat_list_fake:
            stat.add_summary(collections="model")

        self._metric_list = ganlist.gan_metric_list(size=self.data_size)
        for met in self._metric_list:
            met.add_summary(collections="model")

    def preprocess_summaries(self, X_real, **kwargs):
        for met in self._metric_list:
            met.preprocess(X_real, **kwargs)

    def compute_summaries(self, X_real, X_fake, feed_dict={}):
        for stat in self._stat_list_real:
            feed_dict = stat.compute_summary(X_real, feed_dict)
        for stat in self._stat_list_fake:
            feed_dict = stat.compute_summary(X_fake, feed_dict)
        for met in self._metric_list:
            feed_dict = met.compute_summary(X_fake, X_real, feed_dict)
        if self.data_size==1:
            feed_dict = self._plot_real.compute_summary(np.squeeze(X_real), feed_dict=feed_dict)
            feed_dict = self._plot_fake.compute_summary(np.squeeze(X_fake), feed_dict=feed_dict)
        return feed_dict

    def _build_image_summary(self):
        vmin = tf.reduce_min(self.X_real)
        vmax = tf.reduce_max(self.X_real)
        if self.data_size==3:
            X_real = utils.tf_cube_slices(self.X_real)
            X_fake = utils.tf_cube_slices(self.X_fake)
            # Plot some slices
            sl = self.X_real.shape[3]//2
            tf.summary.image(
                "images/Real_Image_slice_middle",
                colorize(self.X_real[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
            tf.summary.image(
                "images/Fake_Image_slice_middle",
                colorize(self.X_fake[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
            sl = self.X_real.shape[3]-1
            tf.summary.image(
                "images/Real_Image_slice_end",
                colorize(self.X_real[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
            tf.summary.image(
                "images/Fake_Image_slice_end",
                colorize(self.X_fake[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
            sl = (self.X_real.shape[3]*3)//4
            tf.summary.image(
                "images/Real_Image_slice_3/4",
                colorize(self.X_real[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
            tf.summary.image(
                "images/Fake_Image_slice_3/4",
                colorize(self.X_fake[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
        elif self.data_size==2:
            X_real = self.X_real
            X_fake = self.X_fake
        elif self.data_size==1:
            self._plot_real = PlotSummaryPlot(4, 4, "real", "signals", collections=['model'])
            self._plot_fake = PlotSummaryPlot(4, 4, "fake", "signals", collections=['model'])
            fs = self.params.get('fs', 16000)
            tf.summary.audio(
                'audio/Real', self.X_real, fs, max_outputs=4, collections=['model'])
            tf.summary.audio(
                'audio/Fake', self.X_fake, fs, max_outputs=4, collections=['model'])
            return None
        tf.summary.image(
            "images/Real_Image",
            colorize(X_real, vmin, vmax),
            max_outputs=4,
            collections=['model'])
        tf.summary.image(
            "images/Fake_Image",
            colorize(X_fake, vmin, vmax),
            max_outputs=4,
            collections=['model'])

    def assert_image(self, x):
        dim = self.data_size + 1
        if len(x.shape) < dim:
            raise ValueError('The size of the data is wrong')
        elif len(x.shape) < (dim +1):
            x = np.expand_dims(x, dim)
        return x

    def batch2dict(self, batch):
        d = dict()
        d['X_real'] = self.assert_image(batch)
        d['z'] = self.sample_latent(len(batch))
        return d

    @property
    def data_size(self):
        return self._data_size

class InpaintingGAN(WGAN):
    def default_params(self):
        bn = False
        d_params = deepcopy(super().default_params())
        bn = False
        signal_length = 1024*16
        signal_split = [1024*6, 1024*4, 1024*6]
        md = 16
        d_params['shape'] = [signal_length, 1] # Shape of the image
        d_params['inpainting'] = dict()
        d_params['inpainting']['split'] = signal_split
        d_params['discriminator']['stride'] = [4,4,4,4,4]
        d_params['discriminator']['nfilter'] = [md, 2*md, 4*md, 8*md, 16*md]
        d_params['discriminator']['shape'] = [[25], [25], [25], [25], [25]]
        d_params['discriminator']['batch_norm'] = [bn, bn, bn, bn, bn]
        d_params['discriminator']['full'] = [md*4]
        d_params['discriminator']['minibatch_reg'] = False
        d_params['discriminator']['summary'] = True
        d_params['discriminator']['data_size'] = 1
        d_params['discriminator']['apply_phaseshuffle'] = True 
        d_params['discriminator']['spectral_norm'] = True
        d_params['discriminator']['activation'] = tf.nn.relu
        d_params['generator']['stride'] = [4, 4, 4, 4, 4]
        d_params['generator']['latent_dim'] = 100
        d_params['generator']['nfilter'] = [8*md, 4*md, 2*md, md, 1]
        d_params['generator']['shape'] = [[25], [25], [25], [25], [25]]
        d_params['generator']['batch_norm'] = [bn, bn, bn, bn]
        d_params['generator']['full'] = [64*md]
        d_params['generator']['summary'] = True
        d_params['generator']['non_lin'] = tf.nn.tanh
        d_params['generator']['activation'] = tf.nn.relu
        d_params['generator']['data_size'] = 1
        d_params['generator']['spectral_norm'] = True 
        d_params['generator']['in_conv_shape'] =[4]
        d_params['generator']['borders'] = dict()
        d_params['generator']['borders']['nfilter'] = [md, 2*md, 4*md, 8*md, 2*md]
        d_params['generator']['borders']['batch_norm'] = [bn, bn, bn, bn, bn]
        d_params['generator']['borders']['shape'] = [[25], [25], [25], [25], [25]]
        d_params['generator']['borders']['stride'] = [4, 4, 4, 4, 4]
        d_params['generator']['borders']['data_size'] = 1
        d_params['generator']['borders']['width_full'] = 128
        d_params['generator']['borders']['activation'] = tf.nn.relu
        return d_params

    def __init__(self, params, name='inpaint_gan'):
        # Only works with 1D signal for now
        assert(params['generator']['data_size'] in [1,2])
        super().__init__(params=params, name=name)
        self._inputs = (self.z, self.borders)
        self._outputs = (self.X_fake)

    def batch2dict(self, batch):
        d = dict()
        d['X_real'] = self.assert_image(batch[:len(batch)//2])
        d['X_to_inpaint'] = self.assert_image(batch[len(batch)//2:])
        d['z'] = self.sample_latent(len(batch))
        return d

    def sample_latent(self, bs=1):
        return super().sample_latent(int(bs//2))
	
    def _build_generator(self):
        shape = self._params['shape']
        self.X_real = tf.placeholder(tf.float32, shape=[None, *shape], name='Xreal')
        self.z = tf.placeholder(
            tf.float32,
            shape=[None, self.params['generator']['latent_dim']],
            name='z')

        self.X_to_inpaint = tf.placeholder(tf.float32, shape=[None, *shape], name='XtoInpaint')
        borderleft, self.center_real, borderright = tf.split(self.X_to_inpaint, self.params['inpainting']['split'], axis=1)
        borders = tf.concat([borderleft,borderright], axis=self.data_size+1)
        inshape = borders.shape.as_list()[1:]

        self.borders = tf.placeholder_with_default(borders, shape=[None, *inshape], name='borders')

        self.X_fake_center = self.generator(self.z,  y=self.borders, reuse=False)
        # Those line should be done in a better way
        if self.data_size == 1:
            self.X_fake = tf.concat([self.borders[:,:,0:1], self.X_fake_center, self.borders[:,:,1:2]], axis=1)
        elif self.data_size == 2:
            self.X_fake = tf.concat([self.borders[:,:,:,0:1], self.X_fake_center, self.borders[:,:,:,1:2]], axis=1)
        else:
            raise NotImplementedError()

    def generator(self, z, y, **kwargs):
        return generator_border(z, y=y, params=self.params['generator'], **kwargs)

# Conditional WGAN
# input parameters are encoded as the length of the latent vector and
# concatenated after the convolutional layers of the discriminator
class ConditionalParamWGAN(WGAN):

    def __init__(self, params, name='Conditional_Params_WGAN'):
        super().__init__(params=params, name=name)

    def default_params(self):
        d_params = super().default_params()
#         d_params['switch_p'] = 0 # Switch real and fake images with this probability
        d_params['cond_params'] = 1
        # This parammeter indicates that the gansystem has to provide the conditioning parameters when calling sample_latent
        d_params['conditional'] = True
        d_params['prior_distribution'] = "gaussian_length"
        d_params['prior_normalization'] = False
        # Note: init range has to always be specified
        d_params['init_range'] = [[0, 1]]
        # Note: final range is the same for every parameter
        d_params['final_range'] = [0.8, 1.2]
        return d_params

    def _build_generator(self):
        shape = self._params['shape']
        self.X_real = tf.placeholder(tf.float32, shape=[None, *shape], name='Xreal')
        self.z = tf.placeholder(tf.float32, shape=[None, self.params['generator']['latent_dim'] * self.params['cond_params']], name='z')
        self.X_param = tf.placeholder(tf.float32, shape=[None, self._params['cond_params']], name='Xparam')
        self.X_fake = self.generator(self.z, reuse=False, model=self)
#         self.switch = tf.placeholder(tf.float32, shape=[None, 1], name='switch')s

    def discriminator(self, X, **kwargs):
        return discriminator(X, params=self.params['discriminator'], z=self.X_param, model=self, **kwargs)

#     def _build_stat_summary(self):
#         super()._build_stat_summary()
#         self._params_metric_list = ganlist.parameters_metric_list()
#         for met in self._params_metric_list:
#             met.add_summary(collections="model")      

#     def compute_summaries(self, X_real, X_fake, feed_dict={}):
#         feed_dict = super().compute_summaries(X_real, X_fake, feed_dict)
#         for met in self._params_metric_list:
#             feed_dict = met.compute_summary(X_fake, X_real, feed_dict)
#         return feed_dict

    def preprocess_summaries(self, data, **kwargs):
        super().preprocess_summaries(data[0], **kwargs)
#         for met in self._params_metric_list:
#             met.preprocess(data[0], **kwargs)

    def sample_latent(self, bs=1, params=None):
        latent_dim = self.params['generator']['latent_dim']
        if params is None:
            raise ValueError("Using a conditional GAN but not providing conditioning parameters to the generator")
#             z = np.zeros([bs,latent_dim])
#             z[:] = np.nan
#             return z
        # Sample latent vector for every params as a different channel
        latent = np.zeros((bs, latent_dim, self.params['cond_params']))
        for c in range(self.params['cond_params']):
            kwargs = {'x': params[:, c], 'init_range': self.params['init_range'][c], 'final_range': self.params['final_range']}
            latent[:, :, c] = utils.sample_latent(bs, latent_dim, self.params['prior_distribution'], self.params['prior_normalization'], **kwargs)
        return latent.reshape((bs, latent_dim * self.params['cond_params']))

    def batch2dict(self, batch):
        d = dict()
        d['X_real'] = self.assert_image(np.array([batch[i][0] for i in range(len(batch))]))
        d['X_param'] = np.array([batch[i][1][:self.params['cond_params']] for i in range(len(batch))])
#         scaled_vec = scale2range(kwargs['x'], kwargs['init_range'], kwargs['final_range'])
#         z = (z.T * np.sqrt((scaled_vec * scaled_vec) / np.sum(z * z, axis=1))).T
        d['z'] = self.sample_latent(len(batch), d['X_param'])

#         d['switch'] = self.sample_switch_real_fake()
        return d

#     def sample_switch_real_fake(self, bs=1):
#         return int(np.random.rand() < self.params['switch_p']) * np.ones((bs, 1))

class LapWGAN(WGAN):
    # TODO add summaries for the 1D case...
    def default_params(self):
        d_params = deepcopy(super().default_params())
        d_params['shape'] = [16, 16, 1] # Shape of the image
        bn = False
        d_params['upscaling'] = 2
        d_params['generator']['latent_dim'] = 16*16
        d_params['generator']['full'] = []
        d_params['generator']['nfilter'] = [16, 32, 32, 1]
        d_params['generator']['batch_norm'] = [bn, bn, bn]
        d_params['generator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
        d_params['generator']['stride'] = [1, 1, 1, 1]

        return d_params

    def __init__(self, params, name='lap_wgan'):
        super().__init__(params=params, name=name)

    def _build_generator(self):
        shape = self._params['shape']
        self.X_real = tf.placeholder(tf.float32, shape=[None, *shape], name='Xreal')
        self.upscaling = self.params['upscaling']
        X_down = down_sampler(self.X_real, s=self.upscaling)
        inshape = X_down.shape.as_list()[1:]
        self.X_down = tf.placeholder_with_default(X_down, shape=[None, *inshape], name='y')
        self.X_smooth = up_sampler(self.X_down, s=self.upscaling, smoothout=True)
        self.z = tf.placeholder(
            tf.float32,
            shape=[None, self.params['generator']['latent_dim']],
            name='z')
        self.X_fake = self.generator(self.z, X=self.X_smooth, reuse=False)

    def discriminator(self, X, **kwargs):
        axis = self.data_size + 1
        v = tf.concat([X, self.X_smooth, X-self.X_smooth], axis=axis)
        return discriminator(v, params=self.params['discriminator'], **kwargs)

    def _build_image_summary(self):
        super()._build_image_summary()
        vmin = tf.reduce_min(self.X_real)
        vmax = tf.reduce_max(self.X_real)
        if self.data_size==3:
            X_smooth = utils.tf_cube_slices(self.X_smooth)
        elif self.data_size==2:
            X_smooth = self.X_smooth

        elif self.data_size==1:
            self._plot_down = PlotSummaryPlot(4, 4, "down", "signals", collections=['model'])
            fs = self.params.get('fs', 16000)
            tf.summary.audio(
                'audio/down', self.X_smooth, fs, max_outputs=4, collections=['model'])
            return None
        tf.summary.image(
            "images/Down_sampled_image",
            colorize(X_smooth, vmin, vmax),
            max_outputs=4,
            collections=['model'])

    def compute_summaries(self, X_real, X_fake, feed_dict={}):
        feed_dict = super().compute_summaries(X_real, X_fake, feed_dict)
        if self.data_size==1:
            X_smooth = self.X_smooth.eval(feed_dict=feed_dict)
            feed_dict = self._plot_down.compute_summary(np.squeeze(X_smooth), feed_dict=feed_dict)
        return feed_dict

class UpscalePatchWGAN(WGAN):
    '''
    Generate blocks, using top, left and top-left border information
    '''

    def default_params(self):
        d_params = deepcopy(super().default_params())
        d_params['shape'] = [16, 16, 4] # Shape of the input data (1 image and 3 borders)
        bn = False
        d_params['upscaling'] = None
        d_params['use_symmetry'] = False
        d_params['generator']['latent_dim'] = 16*16
        d_params['generator']['full'] = []
        d_params['generator']['nfilter'] = [16, 32, 32, 1]
        d_params['generator']['batch_norm'] = [bn, bn, bn]
        d_params['generator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
        d_params['generator']['stride'] = [1, 1, 1, 1]
        d_params['generator']['use_Xdown'] = False
        d_params['generator']['latent_dim_split'] = None
        d_params['generator']['weights_border'] = False
        d_params['generator']['borders'] = None

        return d_params

    def __init__(self, params, name='upscale_patch_wgan'):
        super().__init__(params=params, name=name)
        assert(not(self.params['upscaling']==1))
        if self.params['upscaling']:
            self._inputs = (self.z, self.borders)
        self._outputs = (self.X_fake_corner)


    def _build_generator(self):
        shape = self.params['shape']
        self.X_data = tf.placeholder(tf.float32, shape=[None, *shape], name='X_data')
        self.z = tf.placeholder(
            tf.float32,
            shape=[None, self.params['generator']['latent_dim']],
            name='z')
        # A) Separate real data and border information
        if self.data_size==3:
            axis = 4
            o = 7
        elif self.data_size==2:
            axis = 3
            o = 3
        else:
            axis=2
            o = 1
        la = shape[-1]
        mult = la//(1+o)
        self.X_real_corner, borders = tf.split(self.X_data, [mult,o*mult], axis=axis)
        inshape = borders.shape.as_list()[1:]
        self.borders = tf.placeholder_with_default(borders, shape=[None, *inshape], name='borders')
        
        # B) Split the borders
        border_list = tf.split(self.borders, o, axis=axis)
        if self.params['generator']['weights_border']:
            print('Apply attenuation to borders')
            ns = self.params['shape'][0]
            np_weights = get_attenuation_weights(ns, self.data_size)
            tf_weights = tf.constant(np_weights, dtype=tf.float32)
            border_list_weighted = tf.split(self.borders*tf_weights, o, axis=axis)
            # D) Flip the borders
            flipped_border_list = tf_flip_slices(*border_list_weighted, size=self.data_size)            
        else:    
            # D) Flip the borders
            flipped_border_list = tf_flip_slices(*border_list, size=self.data_size)
        

        # C) Handling downsampling
        if self.params['upscaling']:
            self.upscaling = self.params['upscaling']
            X_down = down_sampler(self.X_real_corner, s=self.upscaling)
            inshape = X_down.shape.as_list()[1:]
            self.X_down = tf.placeholder_with_default(X_down, shape=[None, *inshape], name='y')
            self.X_smooth = up_sampler(self.X_down, s=self.upscaling)
        else:
            self.X_smooth = None
            self.X_down = None

        if self.params['generator']['use_Xdown']:
            print('Using X_down instead of X_smooth')
            X=self.X_down
        else:
            X = self.X_smooth
        z = self.z
        if self.params['generator']['latent_dim_split'] is not None:
            lts = self.params['generator']['latent_dim_split']
            ltv = np.prod(np.array(lts))
            z = self.z[:,ltv:]
            bs = tf.shape(self.z)[0]
            imgz = tf.reshape(self.z[:,:ltv],[bs, *lts])

            if X is None:
                X = imgz
            else:
                X = tf.concat((X, imgz), axis=len(imgz.shape)-1)

        # E) Generater the corner
        self.X_fake_corner = self.generator(z=z, y=flipped_border_list,X=X, reuse=False)


        
        #F) Recreate the big images
        self.X_real = tf_patch2img(self.X_real_corner, *border_list, size=self.data_size, use_symmetry=self.params['use_symmetry'])
        self.X_fake = tf_patch2img(self.X_fake_corner, *border_list, size=self.data_size, use_symmetry=self.params['use_symmetry'])

        if self.params['upscaling']:
            self.X_down_up = up_sampler(down_sampler(self.X_real, s=self.upscaling),s=self.upscaling)

    def discriminator(self, X, **kwargs):
        if self.params['upscaling']:
            axis = self.data_size + 1
            v = tf.concat([X, self.X_down_up, X-self.X_down_up], axis=axis)
        else:
            v = X
#         if self.params['discriminator']['fft_features']:
#             print('Use FFT features')
#             X = tf.cast(X, dtype=tf.complex64)
#             if self.data_size==2:
#                 fftX = tf.abs(tf.fft2d(X))
#             elif self.data_size==3:
#                 fftX = tf.abs(tf.fft3d(X))
#             else:
#                 raise NotImplementedError()
#             axis = self.data_size + 1
#             v = tf.concat([v, tf.cast(fftX, dtype=tf.float32)], axis=axis)
                
        return discriminator(v, params=self.params['discriminator'], **kwargs)

    def _get_corner(self, X):
        if X is not None:
            axis = self.data_size + 1
            slc = [slice(None)] * len(X.shape)
            slc[axis] = 0
            return X[slc]

    def preprocess_summaries(self, X_real, **kwargs):
        super().preprocess_summaries(self._get_corner(self.assert_image(X_real)), **kwargs)

    def compute_summaries(self, X_real, X_fake, feed_dict={}):
        feed_dict = super().compute_summaries(self._get_corner(X_real), self._get_corner(X_fake), feed_dict)
        if self.data_size==1 and self.params['upscaling']:
            X_smooth = self.X_smooth.eval(feed_dict=feed_dict)
            feed_dict = self._plot_down.compute_summary(np.squeeze(X_smooth), feed_dict=feed_dict)
        return feed_dict

    def batch2dict(self, batch):
        d = dict()
        d['X_data'] = self.assert_image(batch)
        d['z'] = self.sample_latent(len(batch))
        return d


    def _build_image_summary(self):
        super()._build_image_summary()
        if self.params['upscaling']:
            vmin = tf.reduce_min(self.X_real)
            vmax = tf.reduce_max(self.X_real)
            if self.data_size==3:
                X_smooth = utils.tf_cube_slices(self.X_smooth)
            elif self.data_size==2:
                X_smooth = self.X_smooth
            elif self.data_size==1:
                self._plot_down = PlotSummaryPlot(4, 4, "down", "signals", collections=['model'])
                fs = self.params.get('fs', 16000)
                tf.summary.audio(
                    'audio/down', self.X_smooth, fs, max_outputs=4, collections=['model'])
                return None
            tf.summary.image(
                "images/Down_sampled_image",
                colorize(X_smooth, vmin, vmax),
                max_outputs=4,
                collections=['model'])

    def generator(self, z, y, X=None, **kwargs):
        if self.params['generator']['borders'] is not None:
            axis = self.data_size + 1
            newy = tf.concat(y, axis=axis)
            return generator_border(z, X=X, y=newy, params=self.params['generator'], **kwargs)
        else:
            if self.params['generator']['use_Xdown']:
                raise ValueError('X_down has to be used with borders parameters')
            axis = self.data_size +1
            if self.params['upscaling']:
                if self.data_size==1:
                    # y = remove_center(y, self.data_size)
                    newX = tf.concat([X, y], axis=axis)
                else:
                    newX = tf.concat([X, *y], axis=axis)
            else:
                newX = tf.concat(y, axis=axis)
            return generator(z, X=newX, params=self.params['generator'], **kwargs)

# def remove_center(X, data_size):
#     '''Only keep the last pixel and set the center to 0.'''
#     zt = np.zeros([1,*X.shape[1:data_size+1], 1], dtype=tf.float32)
#     for i in range(data_size):
#         axis = i + 1
#         slc = [slice(None)] * len(X.shape)
#         slc[axis] = slice(0,X.shape[1+i], X.shape[1+i]-1)
#         zt[slc] = 1
#     zt = tf.convert_to_tensor(zt, np.float32)
#     return X*zt





def js_gan_summaries(D_out_f, D_out_r):
    tf.summary.scalar("Disc/Out_f", tf.reduce_mean(D_out_f), collections=["Training"])
    tf.summary.scalar("Disc/Out_r", tf.reduce_mean(D_out_r), collections=["Training"])
    tf.summary.scalar("Disc/Out_f-r", tf.reduce_mean(D_out_f - D_out_r), collections=["Training"])

def wgan_summaries(D_loss, G_loss, D_loss_f, D_loss_r):
    tf.summary.scalar("Disc/Neg_Loss", -D_loss, collections=["Training"])
    tf.summary.scalar("Disc/Neg_Critic", D_loss_f - D_loss_r, collections=["Training"])
    tf.summary.scalar("Disc/Loss_f", D_loss_f, collections=["Training"])
    tf.summary.scalar("Disc/Loss_r", D_loss_r, collections=["Training"])
    tf.summary.scalar("Gen/Loss", G_loss, collections=["Training"])


def fisher_gan_regularization(D_real, D_fake, rho=1):
    with tf.variable_scope("discriminator", reuse=False):
        phi = tf.get_variable('lambda', shape=[],
            initializer=tf.initializers.constant(value=1.0, dtype=tf.float32))
        D_real2 = tf.reduce_mean(tf.square(D_real))
        D_fake2 = tf.reduce_mean(tf.square(D_fake))
        constraint = 1.0 - 0.5 * (D_real2 + D_fake2)

        # Here phi should be updated using another opotimization scheme
        reg_term = phi * constraint + 0.5 * rho * tf.square(constraint)
        print(D_real.shape)
        print(D_real2.shape)        
        print(constraint.shape)
        print(reg_term.shape)
    tf.summary.scalar("Disc/constraint", reg_term, collections=["Training"])
    tf.summary.scalar("Disc/reg_term", reg_term, collections=["Training"])
    return reg_term


def wgan_regularization(gamma, discriminator, list_fake, list_real):
    if not gamma:
        # I am not sure this part or the code is still useful
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        D_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
        D_gp = tf.constant(0, dtype=tf.float32)
        print(" [!] Using weight clipping")
    else:
        D_clip = tf.constant(0, dtype=tf.float32)
        # calculate `x_hat`
        assert(len(list_fake) == len(list_real))
        bs = tf.shape(list_fake[0])[0]
        eps = tf.random_uniform(shape=[bs], minval=0, maxval=1)

        x_hat = []
        for fake, real in zip(list_fake, list_real):
            singledim = [1]* (len(fake.shape.as_list())-1)
            eps = tf.reshape(eps, shape=[bs,*singledim])
            x_hat.append(eps * real + (1.0 - eps) * fake)

        D_x_hat = discriminator(*x_hat, reuse=True)

        # gradient penalty
        gradients = tf.gradients(D_x_hat, x_hat)
        norm_gradient_pen = tf.norm(gradients[0], ord=2)
        D_gp = gamma * tf.square(norm_gradient_pen - 1.0)
        tf.summary.scalar("Disc/GradPen", D_gp, collections=["Training"])
        tf.summary.scalar("Disc/NormGradientPen", norm_gradient_pen, collections=["Training"])
    return D_gp


#Roth et al. 2017, see https://github.com/rothk/Stabilizing_GANs
def js_regularization(D1_logits, D1_arg, D2_logits, D2_arg, batch_size):
    #print("In shapes: {}, {}, {}, {}".format(D1_logits.shape, D1_arg.shape, D2_logits.shape, D2_arg.shape))
    D1 = tf.nn.sigmoid(D1_logits)
    D2 = tf.nn.sigmoid(D2_logits)
    bs = tf.shape(D1)[0]
    grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
    #print(grad_D1_logits.shape)
    grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
    grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, [bs,-1]), axis=1, keep_dims=True)
    grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, [bs,-1]), axis=1, keep_dims=True)

    #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
    print("Shapes: {}=?={} and {}=?={}".format(grad_D1_logits_norm.shape, D1.shape,
                                               grad_D2_logits_norm.shape, D2.shape))
    #assert grad_D1_logits_norm.shape == D1.shape
    #assert grad_D2_logits_norm.shape == D2.shape

    reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
    reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
    return disc_regularizer


def get_conv(data_size):
    if data_size == 3:
        return conv3d
    elif data_size == 2:
        return conv2d
    elif data_size == 1:
        return conv1d
    else:
        raise ValueError("Wrong data_size")


def deconv(in_tensor, bs, sx, n_filters, shape, stride, summary, conv_num, use_spectral_norm, sy=None, sz=None, data_size=2):
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    if data_size==3:
        output_shape = [bs, sx, sy, sz, n_filters]
        out_tensor = deconv3d(in_tensor,
                              output_shape=output_shape,
                              shape=shape,
                              stride=stride,
                              name='{}_deconv_3d'.format(conv_num),
                              use_spectral_norm=use_spectral_norm,
                              summary=summary)
    elif data_size==2:
        output_shape = [bs, sx, sy, n_filters]
        out_tensor = deconv2d(in_tensor,
                              output_shape=output_shape,
                              shape=shape,
                              stride=stride,
                              name='{}_deconv_2d'.format(conv_num),
                              use_spectral_norm=use_spectral_norm,
                              summary=summary)
    elif data_size==1:
        output_shape = [bs, sx, n_filters]
        out_tensor = deconv1d(in_tensor,
                              output_shape=output_shape,
                              shape=shape,
                              stride=stride,
                              name='{}_deconv_1d'.format(conv_num),
                              use_spectral_norm=use_spectral_norm,
                              summary=summary)
    else:
        raise ValueError("Wrong data_size")

    return out_tensor


def apply_non_lin(non_lin, x, reuse):
    if non_lin:
        if type(non_lin)==str:
            non_lin_f = getattr(tf, params['non_lin'])
            x = non_lin_f(x)
            rprint('    Non lienarity: {}'.format(non_lin), reuse)
        else:
            x = non_lin(x)   
            rprint('    Costum non linearity: {}'.format(non_lin), reuse)

    return x


def legacy_cdf_block(x, params, reuse):
    cdf = tf_cdf(x, params['cdf'])
    rprint('    Cdf layer: {}'.format(params['cdf']), reuse)
    rprint('         Size of the cdf variables: {}'.format(cdf.shape), reuse)
    if params['channel_cdf']:
        lst = []
        for i in range(x.shape[-1]):
            lst.append(tf_cdf(x, params['channel_cdf'],
                              name="cdf_weight_channel_{}".format(i)))
            rprint('        Channel Cdf layer: {}'.format(params['cdf']), reuse)
        lst.append(cdf)
        cdf = tf.concat(lst, axis=1)
        rprint('         Size of the cdf variables: {}'.format(cdf.shape), reuse)
    cdf = linear(cdf, 2 * params['cdf'], 'cdf_full', summary=params['summary'])
    cdf = params['activation'](cdf)
    rprint('     CDF Full layer with {} outputs'.format(2 * params['cdf']), reuse)
    rprint('         Size of the CDF variables: {}'.format(cdf.shape), reuse)
    return cdf


def cdf_block(x, params, reuse):
    assert ('cdf_block' in params.keys())
    block_params = params['cdf_block']
    assert ('cdf_in' in block_params.keys() or 'channel_cdf' in block_params.keys())
    use_first = block_params.get('use_first_channel', False)
    cdf = None
    if block_params.get('cdf_in', None):
        cdf = tf_cdf(x, block_params['cdf_in'], use_first_channel=use_first)
        rprint('    Cdf layer: {}'.format(block_params['cdf_in']), reuse)
        rprint('         Size of the cdf variables: {}'.format(cdf.shape), reuse)
    if block_params.get('channel_cdf', None):
        lst = []
        for i in range(x.shape[-1]):
            lst.append(tf_cdf(x[:,:,:,i], block_params['channel_cdf'], use_first_channel=False,
                              name="cdf_weight_channel_{}".format(i)))
            rprint('        Channel Cdf layer: {}'.format(block_params['channel_cdf']), reuse)
        if block_params.get('cdf_in', None):
            lst.append(cdf)
        cdf = tf.concat(lst, axis=1)
        rprint('         Size of the cdf variables: {}'.format(cdf.shape), reuse)
    out_dim = block_params.get('cdf_out', 2 * block_params.get('cdf_in',8))
    cdf = linear(cdf, out_dim, 'cdf_full', summary=params['summary'])
    cdf = params['activation'](cdf)
    rprint('     CDF Full layer with {} outputs'.format(out_dim), reuse)
    rprint('         Size of the CDF variables: {}'.format(cdf.shape), reuse)
    return cdf


def histogram_block(x, params, reuse):
    hist = learned_histogram(x, params['histogram'])
    out_dim = params['histogram'].get('full', 32)
    hist = linear(hist, out_dim, 'hist_full', summary=params['summary'])
    hist = params['activation'](hist)
    rprint('     Histogram full layer with {} outputs'.format(out_dim), reuse)
    rprint('         Size of the histogram variables: {}'.format(hist.shape), reuse)
    return hist


def discriminator(x, params, z=None, reuse=True, scope="discriminator", model=None):
    conv = get_conv(params['data_size'])

    assert(len(params['stride']) ==
           len(params['nfilter']) ==
           len(params['batch_norm']))
    nconv = len(params['stride'])
    nfull = len(params['full'])

    for it, st in enumerate(params['stride']):
        if not(isinstance(st ,list) or isinstance(st ,tuple)):
            params['stride'][it] = [st]*params['data_size']


    with tf.variable_scope(scope, reuse=reuse):
        if params['fft_features'] or params['psd_features']:
            ns = x.shape.as_list()[1]
            if params['data_size']==2:
                X = tf.cast(x[:,:,:,0], dtype=tf.complex64)
                fftX = tf.abs(tf.fft2d(X))/tf.constant(ns, dtype=tf.float32)
            elif params['data_size']==3:
                X = tf.cast(x[:,:,:,:,0], dtype=tf.complex64)
                fftX = tf.abs(tf.fft3d(X))/tf.constant(ns**(3/2), dtype=tf.float32)
            else:
                raise NotImplementedError()
            fftX = tf.expand_dims(fftX,axis=params['data_size']+1)
                
        if params['fft_features']:
            rprint('Use FFT features', reuse)
            axis = params['data_size'] + 1
            x = tf.concat([x, tf.cast(fftX, dtype=tf.float32)], axis=axis)
        
        if params['psd_features']:
            rprint('Use PSD features', reuse)
            S = get_fourier_sum_matrix(ns, params['data_size']).astype(np.float32)
            tfS = tf.SparseTensor(
                indices=np.array([S.row, S.col]).T,
                values=S.data,
                dense_shape=S.shape)
            fftx = reshape2d(fftX)
            psd_features = tf.transpose(tf.sparse_tensor_dense_matmul(tfS, fftx, adjoint_a=False, adjoint_b=True))
        
        rprint('Discriminator \n'+''.join(['-']*50), reuse)
        rprint('     The input is of size {}'.format(x.shape), reuse)
        if len(params['one_pixel_mapping']):
            x = one_pixel_mapping(x,
                                  params['one_pixel_mapping'],
                                  summary=params['summary'],
                                  reuse=reuse)
        if params['non_lin']:
            non_lin_f = getattr(tf, params['non_lin'])
            x = non_lin_f(x)
            rprint('    Non lienarity: {}'.format(params['non_lin']), reuse)

        if params['cdf']:
            cdf = legacy_cdf_block(x, params, reuse)
        if params['cdf_block']:
            assert(not params['cdf'])
            cdf = cdf_block(x, params, reuse)
        if params.get('histogram', None):
            print('generating histogram block')
            hist = histogram_block(x, params, reuse)
        if params['moment']:
            rprint('    Covariance layer with {} shape'.format(params['moment']), reuse)
            cov = tf_covmat(x, params['moment'])
            rprint('        Layer output {} shape'.format(cov.shape), reuse)
            cov = reshape2d(cov)
            rprint('        Reshape output {} shape'.format(cov.shape), reuse)
            nel = np.prod(params['moment'])**2
            cov = linear(cov, nel, 'cov_full', summary=params['summary'])
            cov = params['activation'](cov)
            rprint('     Covariance Full layer with {} outputs'.format(nel), reuse)
            rprint('         Size of the CDF variables: {}'.format(cov.shape), reuse)

        for i in range(nconv):
            # TODO: this really needs to be cleaned uy...

            if params['data_size']==1 and not(i==0):
                if params.get('apply_phaseshuffle', False):
                    rprint('     Phase shuffle', reuse)               
                    x=apply_phaseshuffle(x)
            if params['inception']:
                x = inception_conv(in_tensor=x, 
                                    n_filters=params['nfilter'][i], 
                                    stride=params['stride'][i], 
                                    summary=params['summary'], 
                                    num=i,
                                    data_size=params['data_size'],
                                    use_spectral_norm=params['spectral_norm'],
#                                     merge=(i == (nconv-1))
                                    merge=True
                                    )
                rprint('     {} Inception(1x1,2x2,4x4) layer with {} channels'.format(i, params['nfilter'][i]), reuse)
            elif params.get('separate_first', False) and i == 0:
                n_out = params['nfilter'][i] // (int(x.shape[3]) + 1)
                lst = []
                for j in range(x.shape[3]):
                    lst.append(conv(x[:,:,:,j:j+1],
                        nf_out=n_out,
                        shape=params['shape'][i],
                        stride=params['stride'][i],
                        use_spectral_norm=params['spectral_norm'],
                        name='{}_conv{}'.format(i,j),
                        summary=params['summary']))
                lst.append(conv(x[:,:,:,:],
                        nf_out=params['nfilter'][i] - (n_out * int(x.shape[3])),
                        shape=params['shape'][i],
                        stride=params['stride'][i],
                        use_spectral_norm=params['spectral_norm'],
                        name='{}_conv_full'.format(i),
                        summary=params['summary']))
                x = tf.concat(lst, axis=3)
            else:
                x = conv(x,
                         nf_out=params['nfilter'][i],
                         shape=params['shape'][i],
                         stride=params['stride'][i],
                         use_spectral_norm=params['spectral_norm'],
                         name='{}_conv'.format(i),
                         summary=params['summary'])
                rprint('     {} Conv layer with {} channels'.format(i, params['nfilter'][i]), reuse)

            if params['batch_norm'][i]:
                x = batch_norm(x, name='{}_bn'.format(i), train=True)
                rprint('         Batch norm', reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

            x = params['activation'](x)
            if model is not None:
                setattr(model, '_D_conv_activation_' + str(i), x)
                
        x = reshape2d(x, name='img2vec')
        rprint('     Reshape to {}'.format(x.shape), reuse)
        
        if  model is not None:
            model._D_features = x
        if z is not None:
            x = tf.concat([x, z], axis=1)
            rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)
        if params['cdf'] or params['cdf_block']:
            x = tf.concat([x, cdf], axis=1)
            rprint('     Contenate with CDF variables to {}'.format(x.shape), reuse)
        if params.get('histogram', None):
            x = tf.concat([x, hist], axis=1)
            rprint('     Contenate with Histogram variables to {}'.format(x.shape), reuse)
        if params['moment']:
            x = tf.concat([x, cov], axis=1)
            rprint('     Contenate with covairance variables to {}'.format(x.shape), reuse)           
        if params['psd_features']:
            x = tf.concat([x, psd_features], axis=1)
            rprint('     Contenate with psd_features to {}'.format(x.shape), reuse)    
        for i in range(nfull):
            x = linear(x,
                       params['full'][i],
                       '{}_full'.format(i+nconv),
                       summary=params['summary'])
            x = params['activation'](x)
            if model is not None:
                setattr(model, '_D_full_activation_' + str(i), x)
            rprint('     {} Full layer with {} outputs'.format(nconv+i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)
        if params['minibatch_reg']:
            rprint('     Minibatch regularization', reuse)
            x = mini_batch_reg(x, n_kernels=150, dim_per_kernel=30)
            rprint('       Size of the variables: {}'.format(x.shape), reuse)

        x = linear(x, 1, 'out', summary=params['summary'])
        # x = tf.sigmoid(x)
        rprint('     {} Full layer with {} outputs'.format(nconv+nfull, 1), reuse)
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)
    return x

def generator(x, params, X=None, y=None, reuse=True, scope="generator", model=None):
    assert(len(params['stride']) == len(params['nfilter'])
           == len(params['batch_norm'])+1)
    nconv = len(params['stride'])
    nfull = len(params['full'])
    for it, st in enumerate(params['stride']):
        if not(isinstance(st ,list) or isinstance(st ,tuple)):
            params['stride'][it] = [st]*params['data_size']


    with tf.variable_scope(scope, reuse=reuse):
        rprint('Generator \n'+''.join(['-']*50), reuse)
        rprint('     The input is of size {}'.format(x.shape), reuse)
        if y is not None:
            x = tf.concat([x, y], axis=1)
            rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)
        for i in range(nfull):
            x = linear(x,
                       params['full'][i],
                       '{}_full'.format(i),
                       summary=params['summary'])
            x = params['activation'](x)
            rprint('     {} Full layer with {} outputs'.format(i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

        bs = tf.shape(x)[0]  # Batch size

        # The following code can probably be much more beautiful.
        if params['data_size']==3:
            # nb pixel
            # if params.get('in_conv_shape', None) is not None:
            sx, sy, sz = params['in_conv_shape']
            # else:
            #     if X is not None:
            #         sx, sy, sz = X.shape.as_list()[1:4]
            #     else:
            #         sx = np.int(np.round((np.prod(x.shape.as_list()[1:]))**(1/3)))
            #         sy, sz = sx, sx
            c = np.int(np.round(np.prod(x.shape.as_list()[1:])))//(sx*sy*sz)
            x = tf.reshape(x, [bs, sx, sy, sz, c], name='vec2img')
            rprint('     Reshape to {}'.format(x.shape), reuse)
            if X is not None:
                x = tf.concat([x, X], axis=4)
                rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)
        elif params['data_size']==2:
            # if params.get('in_conv_shape', None) is not None:
            sx, sy = params['in_conv_shape']
            sz = None
            # else:
            #     # nb pixel
            #     if X is not None:
            #         sx, sy = X.shape.as_list()[1:3]
            #     else:
            #         sx = np.int(np.round(
            #             np.sqrt(np.prod(x.shape.as_list()[1:]))))
            #         sy = sx
            c = np.int(np.round(np.prod(x.shape.as_list()[1:])))//(sx*sy)
            x = tf.reshape(x, [bs, sx, sy, c], name='vec2img')
            rprint('     Reshape to {}'.format(x.shape), reuse)
            if X is not None:
                x = tf.concat([x, X], axis=3)
                rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)
        else:
            # if params.get('in_conv_shape', None) is not None:
            sx = params['in_conv_shape'][0]
            sy, sz = None, None
            # else:
            #     if X is not None:
            #         sx = X.shape.as_list()[1]
            #     else:
            #         sx = np.int(np.round(np.prod(x.shape.as_list()[1:])))
            c = np.int(np.round(np.prod(x.shape.as_list()[1:])))//sx
            x = tf.reshape(x, [bs, sx, c], name='vec2img')
            rprint('     Reshape to {}'.format(x.shape), reuse)

            if X is not None:
                x = tf.concat([x, X], axis=2)
                rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)

        if model is not None:
            setattr(model, '_G_deconv_activation_0', x)
        
        
        if params.get('use_conv_over_deconv', True):
            conv_over_deconv = stride2reduction(params['stride'])==1 # If true use conv, else deconv
        else:
            conv_over_deconv = False

        for i in range(nconv):
            sx = sx * params['stride'][i][0]
            if params['data_size']>1:
                sy = sy * params['stride'][i][1]
            if params['data_size']>2:
                sz = sz * params['stride'][i][2]
            if params['residual'] and (i%2 != 0) and (i < nconv-2): # save odd layer inputs for residual connections
                residue = x

            if params['inception']:
                if conv_over_deconv:
                    x = inception_conv(in_tensor=x, 
                                    n_filters=params['nfilter'][i], 
                                    stride=params['stride'][i], 
                                    summary=params['summary'], 
                                    num=i,
                                    data_size=params['data_size'], 
                                    use_spectral_norm=params['spectral_norm'],
#                                     merge= (True if params['residual'] else (i == (nconv-1)) )
                                    merge= True
                                    )
                    rprint('     {} Inception conv(1x1,2x2,4x4) layer with {} channels'.format(i, params['nfilter'][i]), reuse)

                else:
                    x = inception_deconv(in_tensor=x, 
                                        bs=bs, 
                                        sx=sx, 
                                        n_filters=params['nfilter'][i], 
                                        stride=params['stride'][i], 
                                        summary=params['summary'], 
                                        num=i, 
                                        data_size=params['data_size'],
                                        use_spectral_norm=params['spectral_norm'],
#                                         merge= (True if params['residual'] else (i == (nconv-1)) )
                                        merge= True
                                        )
                    rprint('     {} Inception deconv(1x1,2x2,4x4) layer with {} channels'.format(i, params['nfilter'][i]), reuse)

            else:       
                x = deconv(in_tensor=x, 
                           bs=bs, 
                           sx=sx,
                           n_filters=params['nfilter'][i],
                           shape=params['shape'][i],
                           stride=params['stride'][i],
                           summary=params['summary'],
                           conv_num=i,
                           use_spectral_norm=params['spectral_norm'],
                           sy = sy,
                           sz = sz,
                           data_size=params['data_size']
                           )
                rprint('     {} Deconv layer with {} channels'.format(i+nfull, params['nfilter'][i]), reuse)
            # residual connections before ReLU of every even layer, except 0th and last layer
            if params['residual'] and (i != 0) and (i != nconv-1) and (i%2 == 0):
                x = x + residue
                rprint('         Residual connection', reuse)


            if i < nconv-1:
                if params['batch_norm'][i]:
                    x = batch_norm(x, name='{}_bn'.format(i), train=True)
                    rprint('         Batch norm', reuse)

                x = params['activation'](x)
                if model is not None:
                    setattr(model, '_G_deconv_activation_' + str(i+1), x)
                rprint('         Non linearity applied', reuse)

            rprint('         Size of the variables: {}'.format(x.shape), reuse)

        if len(params['one_pixel_mapping']):
            x = one_pixel_mapping(x,
                                  params['one_pixel_mapping'],
                                  summary=params['summary'],
                                  reuse=reuse)

        x = apply_non_lin(params['non_lin'], x, reuse)

        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)
    return x


def encoder(x, params, latent_dim, reuse=True, scope="encoder"):

    assert(len(params['stride']) ==
           len(params['nfilter']) ==
           len(params['batch_norm']))
    nconv = len(params['stride'])
    nfull = len(params['full'])

    with tf.variable_scope(scope, reuse=reuse):
        rprint('Encoder \n'+''.join(['-']*50), reuse)
        rprint('     The input is of size {}'.format(x.shape), reuse)
        for i in range(nconv):
            x = conv2d(x,
                       nf_out=params['nfilter'][i],
                       shape=params['shape'][i],
                       stride=params['stride'][i],
                       name='{}_conv'.format(i),
                       summary=params['summary'])
            rprint('     {} Conv layer with {} channels'.format(i, params['nfilter'][i]), reuse)
            if params['batch_norm'][i]:
                x = batch_norm(x, name='{}_bn'.format(i), train=True)
                rprint('         Batch norm', reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

            x = lrelu(x)

        x = conv2d(x,
                   nf_out=64,
                   shape=[1,1],
                   stride=1,
                   name='out',
                   summary=params['summary'])
        x = reshape2d(x, name='img2vec')
        rprint('     Reshape to {}'.format(x.shape), reuse)
        for i in range(nfull):
            x = linear(x,
                       params['full'][i],
                       '{}_full'.format(i+nconv),
                       summary=params['summary'])
            x = lrelu(x)
            rprint('     {} Full layer with {} outputs'.format(nconv+i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

        #x = linear(x, latent_dim, 'out', summary=params['summary'])

        # x = tf.sigmoid(x)
        rprint('     {} Full layer with {} outputs'.format(nconv+nfull, 1), reuse)
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)
    return x


def one_pixel_mapping(x, n_filters, summary=True, reuse=False):
    """One pixel mapping."""
    rprint('  Begining of one Pixel Mapping '+''.join(['-']*20), reuse)
    xsh = tf.shape(x) 

    rprint('     The input is of size {}'.format(x.shape), reuse)
    x = tf.reshape(x, [xsh[0], prod(x.shape.as_list()[1:]), 1, 1])
    rprint('     Reshape x to size {}'.format(x.shape), reuse)
    nconv = len(n_filters)
    for i, n_filter in enumerate(n_filters):
        x = conv2d(x,
                   nf_out=n_filter,
                   shape=[1, 1],
                   stride=1,
                   name='{}_1x1conv'.format(i),
                   summary=summary)

        rprint('     {} 1x1 Conv layer with {} channels'.format(i, n_filter), reuse)    
        x = lrelu(x)
        rprint('         Size of the variables: {}'.format(x.shape), reuse)

    x = conv2d(x,
               nf_out=1,
               shape=[1, 1],
               stride=1,
               name='final_1x1conv',
               summary=summary)
    x = tf.reshape(x, xsh)
    rprint('     Reshape x to size {}'.format(x.shape), reuse)
    rprint('  End of one Pixel Mapping '+''.join(['-']*20)+'\n', reuse)
    return x

def generator_border(x, params, X=None, y=None, reuse=True, scope="generator"):
    params_border = params['borders']
    conv = get_conv(params_border['data_size'])

    assert(len(params_border['stride']) == len(params_border['nfilter'])
           == len(params_border['batch_norm']))
    nconv_border = len(params_border['stride'])
    with tf.variable_scope(scope, reuse=reuse):
        rprint('Border block', reuse)
        rprint('\n'+(''.join(['-']*50)), reuse)
        rprint('     BORDER:  The input is of size {}'.format(y.shape), reuse)
        imgt = y
        for i in range(nconv_border):
            imgt = conv(imgt,
                       nf_out=params_border['nfilter'][i],
                       shape=params_border['shape'][i],
                       stride=params_border['stride'][i],
                       name='{}_conv'.format(i),
                       summary=params['summary'])
            rprint('     BORDER: {} Conv layer with {} channels'.format(i, params_border['nfilter'][i]), reuse)
            if params_border['batch_norm'][i]:
                imgt = batch_norm(imgt, name='{}_border_bn'.format(i), train=True)
                rprint('         Batch norm', reuse)
            rprint('         BORDER:  Size of the conv variables: {}'.format(imgt.shape), reuse)
            imgt = lrelu(imgt)
        imgt = reshape2d(imgt, name='border_conv2vec')
	        
        wf = params_border['width_full']
        if wf is not None:
            st = y.shape.as_list()
            if params_border['data_size']==1:
                # We take the begining or the signal as it is flipped.
                border = reshape2d(tf.slice(y, [0, 0, 0], [-1, wf, st[2]]), name='border2vec')
            elif params_border['data_size']==2:
                print('Warning slicing only on side')
                # This is done for the model inpaintingGAN that is supposed to work with spectrograms...
                # We take the begining or the signal as it is flipped.
                border = reshape2d(tf.slice(y, [0, 0, 0, 0], [-1, wf, st[2], -1]), name='border2vec')
                # border = reshape2d(tf.slice(img, [0, st[1]-wf, 0, 0], [-1, wf, st[2], st[3]]), name='border2vec')
            elif params_border['data_size']==3:
                raise NotImplementedError()
            else:
                raise ValueError('Incorrect data_size')
            rprint('     BORDER:  Size of the border variables: {}'.format(border.shape), reuse)
            # rprint('     Latent:  Size of the Z variables: {}'.format(x.shape), reuse)
            y = tf.concat([imgt, border], axis=1)
        else:
            y = imgt

        rprint('     BORDER:  Size of the conv variables: {}'.format(imgt.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)

        return generator(x, params, X=X, y=y, reuse=reuse, scope=scope)

def stride2reduction(stride):
    # This code works with array and single element in stride
    reduction = 1
    for st in stride:
        reduction *= np.array([st]).flatten()[0]
    return reduction
