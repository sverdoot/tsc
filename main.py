import numpy as np
import argparse

import util
import models
import models_vae

def parse_args():
    parser = argparse.ArgumentParser(
                        description='VI and HMC')
    parser.add_argument('--dataset',
                        help='benchmark dataset to use.',
                        default='mnist')
    parser.add_argument('--method',
                        help='method, please choose from: {vi_klqp, vi_klpq, hmc, vae, vae_hsc}.',
                        default='vi_klpq')
    parser.add_argument('--v_fam',
                        help='variational family, please choose from: {gaussian, flow, iaf} (the last two are the same here).',
                        default='gaussian')
    parser.add_argument('--space',
                        help='for HSC (VI with KLpq) and HMC only, please choose from: {eps, theta}.',
                        default='eps')
    parser.add_argument('--vae_num_flow',
                       help='Number of flows to stack in VAE. For no flow (i.e. Gaussian), type 0.',
                       type=int,
                       default=0)
    parser.add_argument('--epochs',
                       help='VI total training epochs.',
                       type=int,
                       default=100)
    parser.add_argument('--lr',
                       help='learning rate (of the initial epoch).',
                       type=float,
                       default=0.1)
    parser.add_argument('--batch_size',
                       help='batch size.',
                       type=int,
                       default=100)
    parser.add_argument('--latent_dim',
                        help='VAE latent dimension.',
                        type=int,
                        default=2)
    parser.add_argument('--num_samp',
                       help='number of samples in VI methods.',
                       type=int,
                       default=1)
    parser.add_argument('--iters',
                       help='HMC iterations.',
                       type=int,
                       default=1000)
    parser.add_argument('--chains',
                       help='HMC chains.',
                       type=int,
                       default=1)
    parser.add_argument('--hmc_e',
                       help='HMC step size.',
                       type=float,
                       default=0.25)
    parser.add_argument('--hmc_L',
                       help='HMC number of leapfrog steps.',
                       type=int,
                       default=4)
    parser.add_argument('--stop_idx',
                       help='Index after which training data is not used.',
                       type=int,
                       default=1e8)
    parser.add_argument('--shear', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--reinitialize_from_q', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warm_up', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_encoder', 
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--default_path',
                       help='Redefine default_path to save results (if omitted, there will already be a default path).',
                       default='na')
    parser.add_argument('--load_path',
                       help='Define load_path if you want to load a trained model.',
                       default='na')
    parser.add_argument('--load_epoch',
                       help='Which epoch do you want the loaded model to start trainig at.',
                       type=int,
                       default=1)
    args = parser.parse_args()
    return args

args = parse_args()
if args.default_path.lower() == 'na':
    path = None
else:
    path = args.default_path

if args.method.lower() == 'vi_klqp':
    model = models.VI_KLqp(
        dataset=args.dataset, 
        v_fam=args.v_fam.lower(),
        num_samp=args.num_samp)
    model.train(epochs=args.epochs, lr=args.lr, save=True, path=path)

if args.method.lower() == 'vi_klpq':
    model = models.VI_KLpq(
        dataset=args.dataset,
        v_fam=args.v_fam.lower(), 
        space=args.space.lower(), 
        num_samp=args.num_samp, 
        chains=args.chains,
        hmc_e=args.hmc_e, 
        hmc_L=args.hmc_L)
    model.train(epochs=args.epochs, lr=args.lr, save=True, path=path)

if args.method.lower() == 'hmc':
    model = models.HMC(
        space=args.space.lower(), 
        iters=args.iters, 
        chains=args.chains, 
        hmc_e=args.hmc_e, 
        hmc_L=args.hmc_L)
    model.run(load_path='results/checkpoints/iaf_qp', save=True, path=path)

if args.method.lower() == 'vae' or args.method.lower() == 'vae_hsc':
    batch_size = args.batch_size
    if args.dataset.lower() == 'mnist':
        train_size = 60000
        test_size = 10000
        train_dataset, test_dataset = util.load_mnist(batch_size)
    random_vector_for_generation = np.genfromtxt(
        'data/vae_random_vector_' + str(args.latent_dim) + '.csv').astype('float32')
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:16, :, :, :]

    if args.method.lower() == 'vae':
        model = models_vae.VAE(
            args.latent_dim,
            num_flow=args.vae_num_flow,
            batch_size=batch_size)
        model.train(train_dataset, test_dataset, epochs=args.epochs, lr=args.lr,
            test_sample=test_sample, random_vector_for_generation=random_vector_for_generation)

    if args.method.lower() == 'vae_hsc':
        if args.load_path.lower() == 'na':
            load_path = None
        else:
            load_path = args.load_path
        model = models_vae.VAE_HSC(
            args.latent_dim, 
            num_flow=args.vae_num_flow,
            num_samp=args.num_samp, 
            chains=args.chains,
            hmc_e=args.hmc_e, 
            hmc_L=args.hmc_L,
            batch_size=batch_size,
            train_size=train_size,
            reinitialize_from_q=args.reinitialize_from_q,
            shear=args.shear)         
        model.train(train_dataset, test_dataset, epochs=args.epochs, lr=args.lr, stop_idx=args.stop_idx, warm_up=args.warm_up,
            train_encoder=args.train_encoder,
            test_sample=test_sample, random_vector_for_generation=random_vector_for_generation, generation=True,
            load_path=load_path, load_epoch=args.load_epoch)






