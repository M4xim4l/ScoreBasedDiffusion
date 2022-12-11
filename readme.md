# Score-Based Diffusion Generative Models

The goal of this repo is to reimplement score-based generative models on smaller datasets in PyTorch 
to help us understand the methods without worrying too much about training time and implementation
details.
Score-based generative methods currently are state-of-the-art in many generative tasks, such
as [image generation](https://arxiv.org/abs/2105.05233) and power many of the text-to-image models
such as [Dall-E](https://openai.com/dall-e-2/), [Imagen](https://imagen.research.google/) and 
[Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release). 

### 01 - Sampling Using The Exact Score
We start by implementing Langevin Sampling using the ground-truth score function of a 
Gaussian Mixture and compare the generated samples to ground truth samples before moving on
to learned score functions in the next example.

![Langevin Samples](/readme_imgs/01.png)


### 02 - Learned Score Sampling
In real applications, we typically do not have direct acces to the groudn-truth score function.
In this example, we will therefore use a neural network to learn the score function from data. 
This is done by using the [connection between score matching and denoising](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf).
As the loss function is an expectation over data samples, learning the score on low data density regions
can be ill-posed. We therefore use ideas from ['Generative Modeling by Estimating Gradients of the Data Distribution'](https://arxiv.org/abs/1907.05600)
by Yang Song and Stefano Ermon and learn to predict the score on increasingly smoothed data distributions.

![Langevin Samples](/readme_imgs/02_noisy_distributions.png)

As we use one single conditional network for all noise scales, the strongly perturbed distributions will regularize
the network on the finer scales and give us better score estimates across all scales.
During Langevin Sampling, we start by sampling from the noisiest distribution and gradually move back towards
less and less perturbed distributions. 

![Langevin Samples](/readme_imgs/02.png)

### 03 - Noise-Conditional Score-Based Sampling On CIFAR10

In this example, we will take the ideas from the previous section and apply them to CIFAR10. 
However, to generate realistic samples, it is important to chose hyperparameters that match the data distribution.
In particular, it is not obvious how one would chose the standard deviation sigma of the largest noise scale, the total
number of noise scales or the stepsize in Langevin Sampling. We implement some ideas from ['Improved Techniques for Training Score-Based Generative Models
'](https://arxiv.org/abs/2006.09011) that allow us to automatically chose those parameters from the data
without the need for extensive parameter tuning. 

![Langevin Samples](/readme_imgs/03.png)


### TODO - Denoising Diffusion Probabilistic Models

Implement both the original formulation of Diffusion Generative Models from ['Deep Unsupervised Learning using Nonequilibrium Thermodynamics'
](https://arxiv.org/abs/1503.03585) by Sohl-Dickstein et al. which optimizes the Evidence Lower Bound as well
as the more recent [Denoising Diffusion](https://arxiv.org/abs/2006.11239) formulation from Ho et al.


### TODO - Score-Based Generative Modeling through Stochastic Differential Equations
Implement continuous time diffusion as proposed in [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456).