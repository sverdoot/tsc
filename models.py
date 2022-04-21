import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import datetime
import os
from pathlib import Path
import pickle
from typing import Callable, Optional, Tuple, Union, Any

from util import *
from dist import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class VI_KLqp:

    def __init__(self, dataset='funnel', v_fam='gaussian', num_dims=2, 
                 num_samp=1, batch_size=5000, train_size=5000, 
                 loc_init=[2.,5.], scale_init=[1.,1.], cis=0):
        self.v_fam = v_fam.lower()
        self.dataset = dataset.lower()
        self.num_dims = num_dims
        self.num_samp = num_samp
        self.batch_size = batch_size
        self.train_size = train_size
        self.cis = cis
        if self.dataset == 'funnel' or self.dataset == 'banana':
            self.num_dims = num_dims
            self.loc_init = tf.zeros(self.num_dims)
            self.scale_init = tf.ones(self.num_dims)
        elif self.dataset == 'survey':
            self.num_dims = 123
            self.loc_init = tf.zeros(self.num_dims)
            self.scale_init = tf.ones(self.num_dims)/3
        
        self.likelihood = self.define_likelihood()
        self.prior = self.define_prior()
        self.define_var_dist()
        
        if self.dataset == 'funnel' or self.dataset == 'banana':
            self.trainable_var = self.q.trainable_variables
        elif self.dataset == 'survey':
            self.trainable_var = []
            self.trainable_var.extend(self.q.trainable_variables)
            self.trainable_var.extend([self.gamma_0,
                                       self.gamma, 
                                       self.sigma])
            
    def define_var_dist(self):
        self.base_distribution = tfd.Sample(
                tfd.Normal(0., 1.), sample_shape=[self.num_dims])
        if self.v_fam == 'iaf' or self.v_fam == 'flow':
            self.p_weight = 1
            self.made = tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=[self.num_dims*5, self.num_dims*5],
                event_shape=(self.num_dims,),
                activation='elu',
                kernel_initializer=tfk.initializers.GlorotNormal())
            self.make_model()
        elif self.v_fam == 'gaussian':
            self.p_weight = 1
            self.phi_m = tf.Variable(
                self.loc_init, 
                name='phi_m')
            self.phi_s = tfp.util.TransformedVariable(
                self.scale_init, 
                tfb.Softplus(),
                name='phi_s')
            self.q = tfd.MultivariateNormalDiag(
                loc=self.phi_m, 
                scale_diag=self.phi_s)

    def define_likelihood(self):
        if self.dataset == 'funnel':
            return Funnel().get_dist()
        elif self.dataset == 'banana':
            return Banana().get_dist()
        elif self.dataset == 'survey':
            self.x = tf.zeros((self.batch_size, 128)) # A placeholder to become data
            self.gamma_0 = tf.Variable(0., dtype=tf.float32, name='gamma_0')
            self.gamma = tf.Variable(tf.zeros(5), name='gamma')
            return self.survey_likelihood_lpdf

    def define_prior(self):
        if self.dataset == 'funnel' or self.dataset == 'banana':
            return None
        elif self.dataset == 'survey':
            self.sigma = tf.Variable(
                tf.zeros(7), 
                name='sigma')
            return self.survey_prior_lpdf
        
    def survey_likelihood_lpdf(self, alpha):
        splitted_x = tf.split(self.x, [123, 5], axis=1)
        term1 = tf.matmul(splitted_x[0], tf.transpose(alpha)) # has shape (batch_size, chains)
        term2 = self.gamma_0 # is scaler
        term3 = tf.matmul(splitted_x[1], tf.reshape(self.gamma, (5, 1))) # has shape (batch_size, 1)
        logits = term1 + term2 + term3 # has shape (batch_size, chains)
        print(logits.shape)
        likelihoods = -tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, 
            labels=tf.tile(tf.expand_dims(self.y, axis=1), [alpha.shape[0], self.num_samp]))
        return tf.reduce_sum(likelihoods, axis=0)
    
    def survey_prior_lpdf(self, alpha):
        # alpha must be chains-by-123
        splitted_alpha = tf.split(alpha, [50, 6, 4, 5, 8, 30, 20], axis=1)
        prior_lpdf = 0.
        for i in range(7):
            priors = log_normal_pdf(
                splitted_alpha[i],
                0.,
                tf.gather(self.sigma, i))
            prior_lpdf = prior_lpdf + priors
        return prior_lpdf 
    
    def make_model(self):
        x_in = tfkl.Input(shape=(self.num_dims,), dtype=tf.float32) # eps
        x_ = self.made(x_in)
        self.model = tfk.Model(x_in, x_)
        self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q = tfd.TransformedDistribution(
            self.base_distribution, 
            self.bij)

    def load_model(self, path):
        self.model.load_weights(path)
        self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q = tfd.TransformedDistribution(
            self.base_distribution, 
            self.bij)
    
    def kl_loss(self):
        eps = self.base_distribution.sample(self.num_samp)
        if self.v_fam == 'flow' or self.v_fam == 'iaf':
            logqz_x = tf.reduce_mean(log_normal_pdf(eps, 0., 0.) - self.bij.forward_log_det_jacobian(eps, 1))
            z = self.bij.forward(eps)
        elif self.v_fam == 'gaussian':
            z = self.phi_m + self.phi_s * eps
            logqz_x = tf.reduce_mean(log_normal_pdf(z, self.phi_m, 2 * tf.math.log(self.phi_s)))
        if self.dataset == 'funnel' or self.dataset == 'banana':
            loss = tf.reduce_mean(logqz_x - 
                self.likelihood.log_prob(z))
        elif self.dataset == 'survey':
            loss = tf.reduce_mean(logqz_x - 
                self.likelihood(z) - self.prior(z))
        return loss
    
    def record_data(self, lst):
        if self.dataset == 'funnel' or self.dataset == 'banana':
            if self.v_fam == 'gaussian':
                lst[0].append(self.phi_m.numpy())
                lst[1].append(self.phi_s.numpy())
        elif self.dataset == 'survey':
            if self.v_fam == 'gaussian':
                lst[0].append(self.phi_m.numpy())
                lst[1].append(self.phi_s.numpy())
                lst[2].append(self.gamma_0.numpy())
                lst[3].append(self.gamma.numpy())
                lst[4].append(self.sigma.numpy())
            elif self.v_fam == 'iaf' or self.v_fam == 'flow':
                lst[2].append(self.gamma_0.numpy())
                lst[3].append(self.gamma.numpy())
                lst[4].append(self.sigma.numpy())
        return lst
    
    def train(self, epochs=int(1e5), lr=0.001, decay_rate=0.001, save=True, path=None):

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(lr, 1, decay_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        phi_m = []
        phi_s = []
        gamma_0 = []
        gamma = []
        sigma = []
        params = [phi_m, phi_s, gamma_0, gamma, sigma]
        params_savenames = ['phi_m.csv', 'phi_s.csv', 'gamma_0.csv', 'gamma.csv', 'sigma.csv']
        elbo = []
        posterior_stds = []
        if save:
            tm = str(datetime.datetime.now())
            tm_str = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
            if path is None:
                path = 'results/' + self.dataset + '/' + 'vi_klqp_' + self.v_fam + '/' + tm_str + '/'
            else:
                path += self.dataset + '/' + 'vi_klqp_' + self.v_fam + '/' + tm_str + '/'
            if not os.path.exists(path): 
                os.makedirs(path)

        for epoch in range(1, epochs + 1):
            
            params = self.record_data(params)

            with tf.GradientTape() as tape:
                loss_value = self.kl_loss()
                elbo.append(-loss_value.numpy())
                
                grads = tape.gradient(loss_value, self.trainable_var)

            optimizer.apply_gradients(zip(grads, self.trainable_var))
            
            if epoch % 50 == 0 or epoch == 1:
                if self.dataset == 'funnel' or self.dataset == 'banana':
                    if self.v_fam == 'iaf' or self.v_fam == 'flow':
                        print('Epoch', epoch, 
                              'Loss', loss_value.numpy())
                    elif self.v_fam == 'gaussian':
                        print('Epoch', epoch, 
                              'Loss', np.round(loss_value.numpy(),3),
                              'phi_s', np.round(self.phi_s.numpy(),3))
                elif self.dataset == 'survey':
                    if self.v_fam == 'iaf' or self.v_fam == 'flow':
                        print('Epoch', epoch, 
                              'Loss', loss_value.numpy(),  
                              'gamma', np.round(self.gamma.numpy(),3),)
                    elif self.v_fam == 'gaussian':
                        print('Epoch', epoch, 
                              'Loss', np.round(loss_value.numpy(),3),
                              'gamma', np.round(self.gamma.numpy(),3),
                              'phi_s', np.round(np.mean(self.phi_s.numpy()),3))
            if epoch % 100 == 0:
                # Monitor convergence for flow-based distributions:
                if self.v_fam == 'iaf' or self.v_fam == 'flow':
                    iaf_qp_samp = self.q.sample(int(1e6))
                    s = tf.math.reduce_std(iaf_qp_samp, axis=0).numpy()
                    m = tf.reduce_mean(iaf_qp_samp, axis=0).numpy()
                    posterior_stds.append(s)
            if epochs - epoch < 100:
                i = epochs - epoch
                if self.v_fam == 'iaf' or self.v_fam == 'flow':
                    self.model.save_weights(path + 'models/model{}/model'.format(i))
                    if epoch == epochs:
                        model_agg = combine_nn_params(self.model, path, 100)
                        model_agg.save_weights(path + 'models/model_agg/model')
            if (epoch % 10000 == 0 or epoch == 1) and save:
                np.savetxt(path+'elbo.csv', np.array(elbo))
                if self.v_fam == 'iaf' or self.v_fam == 'flow':
                    self.model.save_weights(path + 'models/model/model')
                    np.savetxt(path+'posterior_stds.csv', np.array(posterior_stds))
                for i in range(len(params)):
                    if len(params[i]) != 0:
                        np.savetxt(path + params_savenames[i], np.array(params[i]))
                if epoch == 1:
                    rp = open(path + "run_parameters.txt", "w")
                    rp.write('dataset: ' + str(self.dataset) + '\n')
                    rp.write('variational family: ' + self.v_fam + '\n')
                    rp.write('number of samples: ' + str(self.num_samp) + '\n')
                    rp.write('epochs: ' + str(epochs) + '\n')
                    rp.write('learning rate: ' + str(lr) + '\n')
                    rp.write('decay rate: ' + str(decay_rate) + '\n')
                    rp.close()



class VI_KLpq:

    def __init__(self, 
            dataset='funnel', 
            v_fam='gaussian', 
            space='eps', 
            num_dims:int = 2, 
            num_samp: int = 1, 
            chains: int = 1, 
            hmc_e: float = 0.25, 
            hmc_L: int = 4, 
            batch_size: int = 5000, 
            train_size: int = 5000, 
            pt_init: tf.Tensor = tf.constant([[2,10]], 
            dtype=tf.float32), 
            loc_init: Tuple = [2.,5.], 
            scale_init: Tuple = [1.,1.], 
            cis: int = 0, 
            corr_coef: float = 0., 
            bernoulli_prob_corr: float = 0., 
            rejuvenation: bool = False,
            rao_blackwell: bool = False):
        self.space = space.lower()
        self.v_fam = v_fam.lower()
        self.dataset = dataset.lower()
        if self.dataset == 'funnel' or self.dataset == 'banana':
            self.num_dims = num_dims
            # self.pt_init = tf.tile(pt_init, [chains, 1])
            # self.loc_init = loc_init
            # self.scale_init = scale_init
            self.pt_init = tfd.Sample(tfd.Normal(0,1), self.num_dims).sample(chains)
            self.loc_init = tf.zeros(self.num_dims)
            self.scale_init = tf.ones(self.num_dims)
        elif self.dataset == 'survey':
            self.num_dims = 123
            self.pt_init = tfd.Sample(tfd.Normal(0,1), self.num_dims).sample(chains)
            self.loc_init = tf.zeros(self.num_dims)
            self.scale_init = tf.ones(self.num_dims)/3
        self.hmc_e = hmc_e
        self.hmc_L = hmc_L
        self.batch_size = batch_size
        self.train_size = train_size
        self.chains = chains
        self.num_samp = num_samp
        self.cis = cis
        self.corr_coef = corr_coef
        self.bernoulli_prob_corr = bernoulli_prob_corr
        self.rejuvenation = rejuvenation
        self.rao_blackwell = rao_blackwell
        
        self.likelihood = self.define_likelihood()
        self.prior = self.define_prior()
        self.define_var_dist()
        self.log_hmc_target = self.define_log_hmc_target()
        self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.log_hmc_target,
            step_size=np.float32(self.hmc_e),
            num_leapfrog_steps=self.hmc_L,
            state_gradients_are_stopped=True)
        
        if self.dataset == 'funnel' or self.dataset == 'banana':
            self.trainable_var = self.q.trainable_variables
        elif self.dataset == 'survey':
            self.trainable_var_q = list(self.q.trainable_variables)
            self.trainable_var_p = [self.gamma_0,
                                    self.gamma, 
                                    self.sigma]
            self.trainable_var = self.trainable_var_q + self.trainable_var_p
        
    def define_var_dist(self):
        if self.v_fam == 'iaf' or self.v_fam == 'flow':
            self.base_distribution = tfd.Sample(
                tfd.Normal(0., 1.), sample_shape=[self.num_dims])
            self.made = tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=[self.num_dims*5, self.num_dims*5],
                event_shape=(self.num_dims,),
                activation='elu',
                kernel_initializer=tfk.initializers.GlorotNormal())
            self.make_model()
        elif self.v_fam == 'gaussian':
            self.phi_m = tf.Variable(
                self.loc_init, 
                name='phi_m')
            self.phi_s = tfp.util.TransformedVariable(
                self.scale_init, 
                tfb.Softplus(),
                name='phi_s')
            self.q = tfd.MultivariateNormalDiag(
                loc=self.phi_m, 
                scale_diag=self.phi_s)
            if self.space == 'eps' or self.space == 'warped':
                self.bij = tfb.Affine(
                    shift=self.phi_m,
                    scale_diag=self.phi_s)

    def define_likelihood(self) -> Callable:
        if self.dataset == 'funnel':
            return Funnel().get_dist().log_prob
        elif self.dataset == 'banana':
            return Banana().get_dist().log_prob
        elif self.dataset == 'survey':
            self.x = tf.zeros((self.batch_size, 128)) # A placeholder to become data
            self.gamma_0 = tf.Variable(0., dtype=tf.float32, name='gamma_0')
            self.gamma = tf.Variable(tf.zeros(5), name='gamma')
            return self.survey_likelihood_lpdf

    def define_prior(self) -> Union[Callable, Any]:
        if self.dataset == 'funnel' or self.dataset == 'banana':
            return None
        elif self.dataset == 'survey':
            self.sigma = tf.Variable(
                tf.zeros(7), 
                name='sigma')
            return self.survey_prior_lpdf
        
    def define_log_hmc_target(self) -> Callable:
        if self.space == 'eps' or self.space == 'warped':
            self.current_state = self.bij.inverse(self.pt_init)
            return lambda x: self.log_hmc_target_warped_space(x)[0]
        else:
            self.current_state = self.pt_init
            return lambda x: self.log_hmc_target_original_space(x)[0]
        
    def survey_likelihood_lpdf(self, alpha: tf.Tensor) -> tf.Tensor:
        splitted_x = tf.split(self.x, [123, 5], axis=1)
        term1 = tf.matmul(splitted_x[0], tf.transpose(alpha)) # has shape (batch_size, chains)
        term2 = self.gamma_0 # is scaler
        term3 = tf.matmul(splitted_x[1], tf.reshape(self.gamma, (5, 1))) # has shape (batch_size, 1)
        logits = term1 + term2 + term3 # has shape (batch_size, chains)
        likelihoods = -tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, 
            labels=tf.tile(tf.expand_dims(self.y, axis=1), [1, alpha.shape[0]]))
        return tf.reduce_sum(likelihoods, axis=0)
    
    def survey_prior_lpdf(self, alpha: tf.Tensor) -> tf.Tensor:
        # alpha must be chains-by-123
        splitted_alpha = tf.split(alpha, [50, 6, 4, 5, 8, 30, 20], axis=1)
        prior_lpdf = 0.
        for i in range(7):
            priors = log_normal_pdf(
                splitted_alpha[i],
                0.,
                tf.gather(self.sigma, i))
            prior_lpdf = prior_lpdf + priors
        return prior_lpdf 
    
    def log_hmc_target_warped_space(self, eps: tf.Tensor) -> Tuple[tf.Tensor, Any, tf.Tensor]:
        # Unnormalized density p(epsilon | data)
        if self.dataset == 'funnel' or self.dataset == 'banana':
            z = self.bij.forward(eps)
            # z = self.phi_m + self.phi_s * eps
            part1 = self.likelihood(z)
            part2 = self.bij.forward_log_det_jacobian(eps, 1)
            # part2 = tf.reduce_sum(tf.math.log(self.phi_s))
            return part1 + part2, part2, z
        elif self.dataset == 'survey':
            z = self.bij.forward(eps)
            part1 = self.likelihood(z) + self.prior(z)
            part2 = self.bij.forward_log_det_jacobian(eps, 1)
            return part1 + part2, part2, z
        else:
            raise KeyError
    
    def log_hmc_target_original_space(self, z) -> Tuple[tf.Tensor, Any]:
        # Unnormalized density p(z | data)
        if self.dataset == 'funnel' or self.dataset == 'banana':
            return self.likelihood(z), None
        elif self.dataset == 'survey':
            return self.likelihood(z) + self.prior(z), None
        else:
            raise KeyError
    
    def loss(self, z: tf.Tensor, weights: Optional[tf.Tensor] = None) -> Tuple[Any, tf.Tensor]: 
        if tf.rank(z) != 2:
            raise Exception
        if weights is None:
            weights = tf.ones(z.shape[0]) / float(tf.shape(z)[0])
        if weights.shape != z.shape[:1]:
            raise Exception

        if self.v_fam == 'iaf' or self.v_fam == 'flow':
            z0 = self.bij.inverse(z)
            logqz_x = tf.reduce_sum((log_normal_pdf(z0, 0., 0.) - self.bij.forward_log_det_jacobian(z0, 1)) * weights)
        elif self.v_fam == 'gaussian':
            logqz_x = tf.reduce_sum((log_normal_pdf(z, self.phi_m, 2 * tf.math.log(self.phi_s))) * weights)
        else:
            raise KeyError

        if self.dataset == 'funnel' or self.dataset == 'banana':
            return 0, -logqz_x # Just KL(pq)
        elif self.dataset == 'survey':
            return -tf.reduce_sum((self.likelihood(z) + self.prior(z)) * weights), -logqz_x     
        else:
            raise KeyError    
    
    def make_model(self):
        x_in = tfkl.Input(shape=(self.num_dims,), dtype=tf.float32) # eps
        x_ = self.made(x_in)
        self.model = tfk.Model(x_in, x_)
        self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q = tfd.TransformedDistribution(
            self.base_distribution, 
            self.bij)

    def load_model(self, path):
        self.model.load_weights(path)
        self.bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q = tfd.TransformedDistribution(
            self.base_distribution, 
            self.bij)

    def reset_hmc_kernel(self):
        if self.is_accepted > 0.9:
            self.hmc_e = min(self.hmc_e * 1.01, 1.)
        elif self.is_accepted < 0.67:
            self.hmc_e = self.hmc_e * 0.99
        self.hmc_L = min(max(1, int(1 / self.hmc_e)), 33)

        self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.log_hmc_target,
            step_size=np.float32(self.hmc_e),
            num_leapfrog_steps=self.hmc_L,
            state_gradients_are_stopped=True)
        
    def record_data(self, lst):
        if self.dataset == 'funnel' or self.dataset == 'banana':
            if self.v_fam == 'gaussian':
                lst[0].append(self.phi_m.numpy())
                lst[1].append(self.phi_s.numpy())
        elif self.dataset == 'survey':
            if self.v_fam == 'gaussian':
                lst[0].append(self.phi_m.numpy())
                lst[1].append(self.phi_s.numpy())
                lst[2].append(self.gamma_0.numpy())
                lst[3].append(self.gamma.numpy())
                lst[4].append(self.sigma.numpy())
            elif self.v_fam == 'iaf' or self.v_fam == 'flow':
                lst[2].append(self.gamma_0.numpy())
                lst[3].append(self.gamma.numpy())
                lst[4].append(self.sigma.numpy())
        return lst

    def train(self, epochs=int(1e5), lr=0.001, decay_rate=0.001, natural_gradient=False, save=True, path=None, load_path=None, load_epoch=1):    
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(lr, 1, decay_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        if self.dataset == 'survey' and (self.v_fam == 'iaf' or self.v_fam == 'flow'):
            lr_schedule_q_flow = tf.keras.optimizers.schedules.InverseTimeDecay(lr/30, 1, decay_rate)
            optimizer_q = tf.keras.optimizers.Adam(learning_rate=lr_schedule_q_flow)
        if natural_gradient:
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    
        phi_m = []
        phi_s = []
        gamma_0 = []
        gamma = []
        sigma = []
        params = [phi_m, phi_s, gamma_0, gamma, sigma]
        params_savenames = ['phi_m.csv', 'phi_s.csv', 'gamma_0.csv', 'gamma.csv', 'sigma.csv']
        losses = []
        hmc_points = [] # record points directly from HMC, not passed to transport map
        posterior_stds = [] # record posterior stds (approx) in flow, to monitor convergence
        is_accepted = 0
        if save:
            tm = str(datetime.datetime.now())
            tm_str = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
            sub_path = Path(self.dataset, f'vi_klpq_N{self.cis}_{self.space}_{self.v_fam}' + ('_rao' if self.rao_blackwell else ''), tm_str)
            if path is None:
                path = Path('results', sub_path)
            else:
                path += Path(path,sub_path)
            # if self.rao_blackwell:
            #     path = Path(path.parent, f'{path.name}_rao')

            path.mkdir(exist_ok=True, parents=True)
            path = path.as_posix() + '/'

        if load_path is not None and self.dataset == 'survey':
            gamma_0 = list(np.genfromtxt(load_path+'gamma_0.csv', dtype='float32'))
            gamma = list(np.genfromtxt(load_path+'gamma.csv', dtype='float32'))
            sigma = list(np.genfromtxt(load_path+'sigma.csv', dtype='float32'))
            losses = list(np.genfromtxt(load_path+'losses.csv'))
            hmc_points = list(np.genfromtxt(load_path+'hmc_points.csv'))
            self.gamma_0.assign(gamma_0[-1])
            self.gamma.assign(gamma[-1])
            self.sigma.assign(sigma[-1])
            if self.v_fam == 'iaf' or self.v_fam == 'flow':
                self.load_model(load_path+'flow_model/model')
            elif self.v_fam == 'gaussian':
                phi_m = list(np.genfromtxt(load_path+'phi_m.csv', dtype='float32'))
                phi_s = list(np.genfromtxt(load_path+'phi_s.csv', dtype='float32'))
                self.phi_m.assign(phi_m[-1])
                self.phi_s.assign(phi_s[-1])
            params = [phi_m, phi_s, gamma_0, gamma, sigma]
            if self.space == 'eps' or self.space == 'warped':
                if self.v_fam == 'gaussian':
                    self.bij = tfb.Affine(
                        shift=self.phi_m,
                        scale_diag=self.phi_s)
                self.current_state = self.bij.inverse(np.genfromtxt(load_path+'hmc_points.csv'))
            else:
                self.current_state = np.genfromtxt(load_path+'hmc_points.csv')

        begin = datetime.datetime.now()
        for epoch in range(load_epoch, epochs+1):
            # --- Get sample ---+
            if self.cis > 0: # Ex2MCMC
                z0_Sm1 = tfd.Sample(tfd.Normal(0,1), self.num_dims).sample((self.cis - 1) * self.chains)
                z0_Sm1 = tf.reshape(z0_Sm1, (self.cis - 1, self.chains, -1))

                alphas = tfd.Sample(
                    tfd.Bernoulli(probs=self.bernoulli_prob_corr * tf.ones(1))
                    ).sample(self.chains * (self.cis - 1))
                alphas = self.corr_coef * tf.cast(alphas, tf.float32)
                alphas = tf.reshape(alphas, (self.cis - 1, self.chains))
                alphas = tf.tile(tf.expand_dims(alphas, 2), [1, 1, self.num_dims])

                z0 = self.current_state
                alpha0 = tfd.Sample(tfd.Bernoulli(probs=self.bernoulli_prob_corr)).sample(self.chains)
                alpha0 = self.corr_coef * tf.cast(alpha0, tf.float32)
                alpha0 =  tf.tile(tf.expand_dims(alpha0, 1), [1, self.num_dims])
                noise = tfd.Sample(tfd.Normal(0,1), self.num_dims).sample(self.chains)
                xi = alpha0 * z0 + (1. - alpha0**2) ** .5 * noise
                xi = tf.tile(tf.expand_dims(xi, 0), [self.cis - 1, 1, 1])
                
                z0_Sm1 = alphas * xi + (1. - alphas**2) ** .5 * z0_Sm1
                z0_S = tf.concat([tf.reshape(z0, (1, self.chains, -1)), z0_Sm1], axis=0)

                z0_S_flat = tf.reshape(z0_S, (self.chains * self.cis, -1))
                if self.space == 'eps' or self.space == 'warped':
                    log_w_flat, log_jacs, zt_S_flat = self.log_hmc_target_warped_space(z0_S_flat)
                else:
                    log_w_flat, log_jacs = self.log_hmc_target_original_space(z0_S_flat)
                    zt_S_flat = z0_S_flat
                zt_S = tf.reshape(zt_S_flat, (self.cis, self.chains, -1))

                log_w = tf.transpose(tf.reshape(log_w_flat, (self.cis, self.chains)))
                if tf.rank(log_jacs) > 0:
                    log_jacs = tf.transpose(tf.reshape(log_jacs, (self.cis, self.chains)))
                # Sample from categorical (J is batch_size-long vector)
                J = tf.random.categorical(log_w, 1)

                # Set z[k], or z of this iteration
                J = tf.tile(tf.expand_dims(J, 2), [1, self.num_dims, 1])
                z = tf.gather_nd(params=tf.transpose(zt_S, (1, 2, 0)), indices=J, batch_dims=2)
                z0 = tf.gather_nd(params=tf.transpose(z0_S, (1, 2, 0)), indices=J, batch_dims=2)

                if self.rejuvenation:
                    if self.space == 'warped':
                        hmc_input = z0
                    else:
                        hmc_input = z
            else:
                hmc_input = self.current_state

            if self.cis == 0 or self.rejuvenation:
                hmc_input = tf.stop_gradient(hmc_input)
                out = tfp.mcmc.sample_chain(self.num_samp, hmc_input, 
                    previous_kernel_results=None, kernel=self.hmc_kernel,
                    num_burnin_steps=0, num_steps_between_results=0, 
                    trace_fn=(lambda current_state, kernel_results: kernel_results.is_accepted), 
                    parallel_iterations=1000,
                    return_final_kernel_results=False, seed=None, name=None)
                results_is_accepted = out[1].numpy()
                out = out[0]
                if len(out.shape) > 2:
                    out = tf.squeeze(out, axis=0)
                if len(out.shape) < 2:
                    out = tf.expand_dims(out, axis=0)
                if self.space == 'warped':
                    eps = out
                    z = self.bij.forward(eps)
                else:
                    z = out
            else:
                results_is_accepted = np.array([1.])
            z = tf.stop_gradient(z)

            params = self.record_data(params)
            hmc_points.append(z.numpy())
            is_accepted += np.mean(np.squeeze(results_is_accepted))
            self.is_accepted = is_accepted / (epoch+1)

            # --- Training ---+
            if self.cis > 0 and self.rao_blackwell: #Ex2MCMC, Rao-Blackwellized
                zt_S_flat = tf.stop_gradient(zt_S_flat)
                z_in = zt_S_flat
                weights = tf.reshape(
                    tf.nn.softmax(tf.transpose(log_w - log_jacs), axis=0) / log_w.shape[0], 
                    self.chains * self.cis)
                weights = tf.stop_gradient(weights)
            else:    
                z_in = z
                weights = None

            with tf.GradientTape(persistent=True) as tape:
                loss_p, loss_q = self.loss(z_in, weights)
                loss_value = loss_p + loss_q
                
            if self.dataset == 'survey' and (self.v_fam == 'iaf' or self.v_fam == 'flow'):
                grads_p = tape.gradient(loss_p, self.trainable_var_p)
                optimizer.apply_gradients(zip(grads_p, self.trainable_var_p))
                grads_q = tape.gradient(loss_q, self.trainable_var_q)
                optimizer_q.apply_gradients(zip(grads_q, self.trainable_var_q))
            else:
                grads = tape.gradient(loss_value, self.trainable_var)

                if natural_gradient:
                    grads = tf.expand_dims(tf.concat(list(grads), axis=0), axis=0)
                    F = tf.stop_gradient(
                        - E_log_normal_hessian(self.phi_m, self.phi_s))
                    grads = tf.matmul(tf.linalg.inv(F), tf.transpose(grads))
                    grads = tf.squeeze(grads)
                    grads = tf.split(grads, [self.num_dims, self.num_dims])

                optimizer.apply_gradients(zip(grads, self.trainable_var))
            del tape

            losses.append(loss_value.numpy())
            
            # --- Update current state for hmc kernel ---+
            if self.space == 'eps' or self.space == 'warped':
                if self.v_fam == 'gaussian':
                    self.bij = tfb.Affine(
                        shift=self.phi_m,
                        scale_diag=self.phi_s)
                self.current_state = self.bij.inverse(z)
            else:
                self.current_state = z 

            if epoch > 10000 and self.dataset == 'survey':
                self.reset_hmc_kernel()

            end = datetime.datetime.now()

            # --- Print and save results ---+    
            if epoch % 100 == 0:
                print(f'Time: {end-begin}\n')
                if self.dataset == 'funnel' or self.dataset == 'banana':
                    if self.v_fam == 'iaf' or self.v_fam == 'flow':
                        print('Epoch', epoch, 
                              'Loss', loss_value.numpy(),  
                              'point', np.round(tf.gather(z, 0).numpy(),3), 
                              'acceptance rate', round(self.is_accepted, 3))
                    elif self.v_fam == 'gaussian':
                        print('Epoch', epoch, 
                              'Loss', np.round(loss_value.numpy(),3),
                              'point', np.round(tf.gather(z, 0).numpy(),3), 
                              'phi_s', np.round(self.phi_s.numpy(),3),
                              'acceptance rate', round(self.is_accepted, 3))
                elif self.dataset == 'survey':
                    if self.v_fam == 'iaf' or self.v_fam == 'flow':
                        print('Epoch', epoch, 
                              'Loss', loss_value.numpy(),  
                              'acceptance rate', round(self.is_accepted, 3),
                              'gamma', np.round(self.gamma.numpy(),3),
                              'hmc_e', np.round(self.hmc_e, 3))
                    elif self.v_fam == 'gaussian':
                        print('Epoch', epoch, 
                              'Loss', np.round(loss_value.numpy(),3),
                              'acceptance rate', round(self.is_accepted, 3),
                              'gamma', np.round(self.gamma.numpy(),3),
                              'phi_s', np.round(np.mean(self.phi_s.numpy()),3),
                              'hmc_e', np.round(self.hmc_e, 3))
            if epoch % 100 == 0:
                # Monitor convergence for flow-based distributions:
                if self.v_fam == 'iaf' or self.v_fam == 'flow':
                    iaf_pq_samp = self.q.sample(int(1e6))
                    s = tf.math.reduce_std(iaf_pq_samp, axis=0).numpy()
                    m = tf.reduce_mean(iaf_pq_samp, axis=0).numpy()
                    posterior_stds.append(s)
            if epochs - epoch < 100:
                i = epochs - epoch
                if self.v_fam == 'iaf' or self.v_fam == 'flow':
                    self.model.save_weights(path + 'models/model{}/model'.format(i))
                    if epoch == epochs:
                        model_agg = combine_nn_params(self.model, path, 100)
                        model_agg.save_weights(path + 'models/model_agg/model')
            if (epoch % 10000 == 0 or epoch == 1) and save:
                np.savetxt(path+'losses.csv', np.array(losses))
                with open(path+'hmc_points.pickle', 'wb') as handle:
                    pickle.dump(hmc_points, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if self.v_fam == 'iaf' or self.v_fam == 'flow':
                    self.model.save_weights(path + 'flow_model/model')
                    np.savetxt(path+'posterior_stds.csv', np.array(posterior_stds))
                for i in range(len(params)):
                    if len(params[i]) != 0:
                        np.savetxt(path + params_savenames[i], np.array(params[i]))
                if epoch == 1:
                    rp = open(path + "run_parameters.txt", "w")
                    rp.write('dataset: ' + str(self.dataset) + '\n')
                    rp.write('variational family: ' + self.v_fam + '\n')
                    rp.write('epochs: ' + str(epochs) + '\n')
                    rp.write('learning rate: ' + str(lr) + '\n')
                    rp.write('decay rate: ' + str(decay_rate) + '\n')
                    rp.write('number of samples: ' + str(self.num_samp) + '\n')
                    rp.write('HMC space: ' + self.space + '\n')
                    rp.write('HMC step size e: ' + str(self.hmc_e) + '\n')
                    rp.write('HMC number of leapfrog steps L: ' + str(self.hmc_L) + '\n')
                    rp.write('HMC number of chains: ' + str(self.chains) + '\n')
                    rp.close()




class HMC:
    def __init__(self, space='eps', num_dims=2, iters=int(1e4), chains=1,
        hmc_e=0.25, hmc_L=4):
        self.space = space
        self.iters = iters
        self.chains = chains
        self.hmc_e = hmc_e
        self.hmc_L = hmc_L
        self.target = Funnel(num_dims).get_dist()
        self.base_distribution = tfd.Sample(
                tfd.Normal(0., 1.), sample_shape=[num_dims])
        if self.space == 'eps':
            self.made = tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=[20, 20],
                event_shape=(2,),
                activation='elu',
                kernel_initializer=tfk.initializers.GlorotNormal())
            self.make_model()
        self.current_state = self.base_distribution.sample(self.chains)

    def make_model(self):
        x_in = tfkl.Input(shape=(2,), dtype=tf.float32) # eps
        x_ = self.made(x_in)
        self.model = tfk.Model(x_in, x_)
        self.iaf_bijector_qp = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
        self.q_iaf_qp = tfd.TransformedDistribution(
            self.base_distribution, 
            self.iaf_bijector_qp)

    def load_model(self, path):
        if self.space == 'eps':
            self.model.load_weights(path)
            self.iaf_bijector_qp = tfb.Invert(tfb.MaskedAutoregressiveFlow(self.model))
            self.q_iaf_qp = tfd.TransformedDistribution(
                self.base_distribution, 
                self.iaf_bijector_qp)
        else:
            print('This is only valid with HMC on warped space (epsilon-space).')

    def log_hmc_target(self, eps):
        # Unnormalized density, p(epsilon | data)
        # Needs global variables iaf_bijector and target
        theta = self.iaf_bijector_qp.forward(eps)
        part1 = self.target.log_prob(theta)
        part2 = self.iaf_bijector_qp.forward_log_det_jacobian(eps, 1)

        return part1 + part2

    def run(self, load_path=None, save=True, path=None):
        if self.space == 'eps':
            if load_path is not None:
                self.load_model(path)
            self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.log_hmc_target,
                step_size=np.float32(self.hmc_e),
                num_leapfrog_steps=self.hmc_L,
                state_gradients_are_stopped=True)
        else:
            self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.target.log_prob,
                step_size=np.float32(self.hmc_e),
                num_leapfrog_steps=self.hmc_L,
                state_gradients_are_stopped=True)

        out = tfp.mcmc.sample_chain(
            self.iters, self.current_state, previous_kernel_results=None, kernel=self.hmc_kernel,
            num_burnin_steps=0, num_steps_between_results=0, parallel_iterations=self.chains, 
            trace_fn=(lambda current_state, kernel_results: kernel_results.is_accepted), 
            return_final_kernel_results=False, seed=None, name=None)
        
        accept_rate = np.sum(np.squeeze(out[1].numpy()))/self.iters/self.chains

        if save:
            tm = str(datetime.datetime.now())
            tm_str = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
            if path is None:
                path = 'results/' + 'hmc_' + self.space + '/' + tm_str + '/'
            else:
                path += 'hmc_' + self.space + '/' + tm_str + '/'
            if not os.path.exists(path): 
                os.makedirs(path)

            if self.space == 'eps':
                out_eps = out[0].numpy() # iters x chains x num_dims
                samps = []
                for i in range(self.chains):
                    samps.append(self.iaf_bijector_qp.forward(out_eps[:,i,:]))
                samps = np.array(samps)
                np.savetxt(path + 'hmc.csv', np.reshape(samps,(-1,2)))
            else:
                samps = out[0].numpy()
                np.savetxt(path + 'hmc.csv', np.reshape(samps,(-1,2)))

            rp = open(path + "run_parameters.txt", "w")
            rp.write('iters: ' + str(self.iters) + '\n')
            rp.write('chains: ' + str(self.chains) + '\n')
            rp.write('HMC step size e: ' + str(self.hmc_e) + '\n')
            rp.write('HMC number of leapfrog steps L: ' + str(self.hmc_L) + '\n')
            rp.write('acceptance rate: ' +str( round(accept_rate,3) ) + '\n')
            rp.close()



















