#!/usr/bin/env bash

# Affine transform


#VI_pq HMC, gaussian var family
python main.py --dataset=survey \
    --method=vi_klpq \
    --v_fam=gaussian \
    --space=warped \
    --epochs=10000 \
    --lr=0.01 \
    --hmc_e=0.05 \
    --hmc_L=20 \
    --cis=0

# # VI_pq Ex2MCMC (HMC), gaussian var family
# python main.py --dataset=survey \
#     --method=vi_klpq \
#     --v_fam=gaussian \
#     --space=eps \
#     --epochs=10000 \
#     --lr=0.01 \
#     --hmc_e=0.05 \
#     --hmc_L=20 \
#     --cis=10 \
#     --corr_coef=0.95 \
#     --bernoulli_prob_corr=0.5 \
#     --rejuvenation

# VI_pq Ex2MCMC (HMC), gaussian var family, HMC in warped space
python main.py --dataset=survey \
    --method=vi_klpq \
    --v_fam=gaussian \
    --space=warped \
    --epochs=10000 \
    --lr=0.01 \
    --hmc_e=0.05 \
    --hmc_L=20 \
    --cis=10 \
    --corr_coef=1.0 \
    --bernoulli_prob_corr=1.0 \
    --rejuvenation \
    --rao_blackwell

# Flow


# # VI_pq HMC, flow var family
# python main.py --dataset=survey \
#     --method=vi_klpq \
#     --v_fam=flow \
#     --space=warped \
#     --epochs=10000 \
#     --lr=0.001 \
#     --hmc_e=0.03 \
#     --hmc_L=33 \
#     --cis=0

# # VI_pq Ex2MCMC (HMC), flow var family
# python main.py --dataset=survey \
#     --method=vi_klpq \
#     --v_fam=flow \
#     --space=eps \
#     --epochs=10000 \
#     --lr=0.001 \
#     --hmc_e=0.03 \
#     --hmc_L=33 \
#     --cis=10 \
#     --corr_coef=0.95 \
#     --bernoulli_prob_corr=0.5 \
#     --rejuvenation

# # VI_pq Ex2MCMC (HMC), flow var family, HMC in warped space
# python main.py --dataset=survey \
#     --method=vi_klpq \
#     --v_fam=flow \
#     --space=warped \
#     --epochs=10000 \
#     --lr=0.001 \
#     --hmc_e=0.03 \
#     --hmc_L=33 \
#     --cis=10 \
#     --corr_coef=0.95 \
#     --bernoulli_prob_corr=0.5 \
#     --rejuvenation
