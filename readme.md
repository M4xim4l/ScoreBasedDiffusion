# Score-based diffusion models

The goal of this repo is to reimplement some score-based generative models that currently are SOTA in many tasks, suchs 
as Text-2-Image generation, on simpler datasets.

### 01 - Exact Score Matching
Demonstration of Langevin sampling using the exact score function calculated as the gradient of the log-data density

### 02 - Learned Score Sampling
Similar to the first step, we use Langevin sampling to sample from a Gaussian Mixture distribution, 
however this time we use the connection between score-matching and denoising to estimate the score function from samples of the
distribution

### TODO - Denoising Diffusion Probabilistic Models

### TODO - Score-Based Generative Modeling through Stochastic Differential Equations