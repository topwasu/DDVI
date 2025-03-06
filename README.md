# Denoising Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors

Accepted to AAAI 2025

Arxiv link: [https://arxiv.org/abs/2401.02739](https://arxiv.org/abs/2401.02739)

# Installation
```
pip install -r requirements.txt
```

# Run

Bash script to run DDVI (model=diff_vae_warmup):
```
dataset=mnist # Options are [mnist, cifar]
model=diff_vae_warmup # Options are [vae, iaf_vae, h_iaf_vae, diff_vae_warmup]
prior=pin_wheel # Options are [pinwheel, swiss_roll, less_noisy_square]

mkdir experimental_results/
mkdir experimental_results/$dataset_$model_$prior
python run.py dataset=$dataset model=$model prior=$prior save_folder=${save_folder}/run0 seed=0
```