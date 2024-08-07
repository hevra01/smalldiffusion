import math
from itertools import pairwise

import torch
import numpy as np
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace
from typing import Optional

class Schedule:
    '''Diffusion noise schedules parameterized by sigma'''
    def __init__(self, sigmas: torch.FloatTensor):
        self.sigmas = sigmas

    def __getitem__(self, i) -> torch.FloatTensor:
        return self.sigmas[i]

    def __len__(self) -> int:
        return len(self.sigmas)

    def sample_sigmas(self, steps: int) -> torch.FloatTensor:
        '''Called during sampling to get a decreasing sigma schedule with a
        specified number of sampling steps:
          - Spacing is "trailing" as in Table 2 of https://arxiv.org/abs/2305.08891
          - Includes initial and final sigmas
            i.e. len(schedule.sample_sigmas(steps)) == steps + 1

        High Noise at the Beginning:
        Initial large steps in noise reduction.

        Gradually Reducing:
        Smaller steps in noise reduction towards the end.

        e.g. sigmas = [20, 15, 10, 8, 6, 4, 2, 0]

        '''
        indices = list((len(self) * (1 - np.arange(0, steps)/steps))
                       .round().astype(np.int64) - 1)
        
        return self[indices + [0]]

    def sample_batch(self, x0: torch.FloatTensor) -> torch.FloatTensor:
        '''Called during training to get a batch of randomly sampled sigma values
        '''
        batchsize = x0.shape[0]

        # generates "batchsize" number of random integers max value of len(self), which will be used
        # as an index to self.sigmas.
        # e.g. is self.sigmas has values from [0.005, 10], then we will randomly
        # sample "batchsize" values from here.
        return self[torch.randint(len(self), (batchsize,))].to(x0)

def sigmas_from_betas(betas: torch.FloatTensor):
    return (1/torch.cumprod(1.0 - betas, dim=0) - 1).sqrt()

# Simple log-linear schedule works for training many diffusion models
class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float=0.02, sigma_max: float=10):
        super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N))

# Default parameters recover schedule used in most diffusion models
class ScheduleDDPM(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.0001, beta_end: float=0.02):
        super().__init__(sigmas_from_betas(torch.linspace(beta_start, beta_end, N)))

# Default parameters recover schedule used in most latent diffusion models, e.g. Stable diffusion
class ScheduleLDM(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.00085, beta_end: float=0.012):
        super().__init__(sigmas_from_betas(torch.linspace(beta_start**0.5, beta_end**0.5, N)**2))

# Sigmoid schedule used in GeoDiff
class ScheduleSigmoid(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.0001, beta_end: float=0.02):
        betas = torch.sigmoid(torch.linspace(-6, 6, N)) * (beta_end - beta_start) + beta_start
        super().__init__(sigmas_from_betas(betas))

# Cosine schedule used in Nichol and Dhariwal 2021
class ScheduleCosine(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.0001, beta_end: float=0.02, max_beta: float=0.999):
        alpha_bar = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        betas = [min(1 - alpha_bar((i+1)/N)/alpha_bar(i/N), max_beta)
                 for i in range(N)]
        super().__init__(sigmas_from_betas(torch.tensor(betas, dtype=torch.float32)))

# Given a batch of data x0, returns:
#   eps  : i.i.d. normal with same shape as x0
#   sigma: uniformly sampled from schedule, with shape Bx1x..x1 for broadcasting
def generate_train_sample(x0: torch.FloatTensor, schedule: Schedule):
    sigma = schedule.sample_batch(x0)
    
    # check the dimensionality of the data.
    # if it is larger than sigma, then unsqueeze it so that
    # all the dimensions get the sigma. 
    while len(sigma.shape) < len(x0.shape):
        sigma = sigma.unsqueeze(-1)
    
    # This line generates a tensor of noise values (eps) with the same shape as x0, where each element is 
    # independently sampled from a standard normal distribution N(0,1).
    eps = torch.randn_like(x0)
    return sigma, eps

# Model objects
# Always called with (x, sigma):
#   If x.shape == [B, D1, ..., Dk], sigma.shape == [] or [B, 1, ..., 1].
#   If sigma.shape == [], model will be called with the same sigma for each x0
#   Otherwise, x[i] will be paired with sigma[i] when calling model
# Have a `rand_input` method for generating random xt during sampling

def training_loop(loader     : DataLoader,
                  model      : nn.Module,
                  schedule   : Schedule,
                  accelerator: Optional[Accelerator] = None,
                  epochs     : int = 10000,
                  lr         : float = 1e-3):
    accelerator = accelerator or Accelerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    for _ in (pbar := tqdm(range(epochs))):
        for x0 in loader:
            optimizer.zero_grad()
            # Generates a batch of sigma values (sigma) and corresponding noise (eps).
            # Each sigma value is used to scale the noise eps.
            sigma, eps = generate_train_sample(x0, schedule)
    
            # The model is given the noisy data points (x0 + sigma * eps) and the sigma values.
            # The model's task is to predict the noise (eps_hat).
            eps_hat = model(x0 + sigma * eps, sigma)
            
            # The loss is computed between the predicted noise (eps_hat) and the actual noise (eps).
            loss = nn.MSELoss()(eps_hat, eps)
            yield SimpleNamespace(**locals()) # For extracting training statistics
            accelerator.backward(loss)
            optimizer.step()

# Generalizes most commonly-used samplers:
#   DDPM       : gam=1, mu=0.5
#   DDIM       : gam=1, mu=0
#   Accelerated: gam=2, mu=0
@torch.no_grad()
def samples(model      : nn.Module,
            sigmas     : torch.FloatTensor, # Iterable with N+1 values for N sampling steps
            gam        : float = 1.,        # Suggested to use gam >= 1
            mu         : float = 0.,        # Requires mu in [0, 1)
            xt         : Optional[torch.FloatTensor] = None,
            accelerator: Optional[Accelerator] = None,
            batchsize  : int = 1):
    

    """
    Given noise, this function will iteratively remove the noise and bring the data
    to the learned distribution. 

    The noise can either be given to the function or the function can create
    random noise.
    """
    
    # xt is random noise
    #print("sigmas", sigmas)
    accelerator = accelerator or Accelerator()
    if xt is None:
        xt = model.rand_input(batchsize).to(accelerator.device) * sigmas[0]
        #print("xt", xt[:10]) we get the same result here 
    else:
        batchsize = xt.shape[0]
    eps = None


    # Yield the initial noise before any updates.
    # this wasn't here in the original code
    yield xt

    # Get a bunch of (depending on batch size) pure noise and clean
    # out the noise in N steps. The model has learned the distribution
    # in the training data, hence, we hope that from noise it will arrive to
    # a similar distribution of the data. 

    # Iterative Denoising:
    # The model iteratively refines this noise tensor. At each step, it uses the predicted noise 
    # (learned during training) to progressively remove noise from the tensor.
    # Each iteration moves the tensor closer to a data point that resembles the original training data.
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps, eps_prev = model(xt, sig.to(xt)), eps
        eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
        sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
        eta = (sig_prev**2 - sig_p**2).sqrt()
        xt = xt - (sig - sig_p) * eps_av + eta * model.rand_input(batchsize).to(xt)
        
        # Yield the noise after each update
        yield xt

@torch.no_grad()
def DDIM_inversion(model      : nn.Module,
            sigmas     : torch.FloatTensor, # Iterable with N+1 values for N sampling steps
            xt         : torch.FloatTensor,
            gam        : float = 1.,        # Suggested to use gam >= 1
            mu         : float = 0.,        # Requires mu in [0, 1)
            accelerator: Optional[Accelerator] = None,
            batchsize  : int = 1):
    

    """
    Given noise, this function will iteratively remove the noise and bring the data
    to the learned distribution. 

    The noise can either be given to the function or the function can create
    random noise.
    """
    
    # xt is random noise
    accelerator = accelerator or Accelerator()
    if xt is None:
        xt = model.rand_input(batchsize).to(accelerator.device) * sigmas[0]
    else:
        batchsize = xt.shape[0]
    eps = None

    yield xt

    # Get a bunch of (depending on batch size) pure noise and clean
    # out the noise in N steps. The model has learned the distribution
    # in the training data, hence, we hope that from noise it will arrive to
    # a similar distribution of the data. 

    # Iterative Denoising:
    # The model iteratively refines this noise tensor. At each step, it uses the predicted noise 
    # (learned during training) to progressively remove noise from the tensor.
    # Each iteration moves the tensor closer to a data point that resembles the original training data.
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps, eps_prev = model(xt, sig.to(xt)), eps
        eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
        sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
        eta = (sig_prev**2 - sig_p**2).sqrt()
        xt = xt - (sig - sig_p) * eps_av + eta * model.rand_input(batchsize).to(xt)
        yield xt