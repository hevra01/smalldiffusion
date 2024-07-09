import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import HTML
import torch

def animate_diffusion(samples, filename="", save=False):
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], marker='.')

    # Concatenate all samples along a new dimension 
    all_samples = torch.cat(samples, dim=0)

    # Calculate global min and max for each dimension separately
    min_x = torch.min(all_samples[:, 0])
    max_x = torch.max(all_samples[:, 0])
    min_y = torch.min(all_samples[:, 1])
    max_y = torch.max(all_samples[:, 1])

    min_x = min_x.cpu()
    min_y = min_y.cpu()
    max_x = max_x.cpu()
    max_y = max_y.cpu()
        
    def init():
        ax.set_xlim(min_x-1, max_x+1)
        ax.set_ylim(min_y-1, max_y+1)
        return scat,

    def update(frame):
        data = samples[frame].cpu()
        local_max_x = torch.max(samples[frame][:, 0]).cpu()
        local_max_y = torch.max(samples[frame][:, 1]).cpu()
        local_min_x = torch.min(samples[frame][:, 0]).cpu()
        local_min_y = torch.min(samples[frame][:, 1]).cpu()

        data -= np.array([local_min_x, local_min_y])
        data *= (np.array([max_x, max_y]) - np.array([min_x, min_y])) / (np.array([local_max_x, local_max_y]) - np.array([local_min_x, local_min_y]))
        data += np.array([min_x, min_y])

        scat.set_offsets(data)
        return scat,

    anim = FuncAnimation(fig, update, frames=len(samples), init_func=init, blit=True)

    if save:
        anim.save(filename + ".gif", writer='pillow')
     
    # Display the animation in the notebook
    return HTML(anim.to_jshtml())


def animate_diffusion_2(samples, filename="", save=False):
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], marker='.')

        
    def init():
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        return scat,

    def update(frame):
        data = samples[frame].cpu()

        scat.set_offsets(data)
        return scat,

    anim = FuncAnimation(fig, update, frames=len(samples), init_func=init, blit=True)

    if save:
        anim.save(filename + ".gif", writer='pillow')
     
    # Display the animation in the notebook
    return HTML(anim.to_jshtml())


def animate_diffusion_3(samples1, samples2, samples3, filename="", save=False):
    fig, ax = plt.subplots()
    scat1 = ax.scatter([], [], label='Set 1')
    scat2 = ax.scatter([], [], label='Set 2', marker='x')
    scat3 = ax.scatter([], [], label='Set 3', marker='+')

    def init():
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.legend()
        return scat1, scat2, scat3

    def update(frame):
        data1 = samples1[frame].cpu()
        data2 = samples2[frame].cpu()
        data3 = samples3[frame].cpu()

        scat1.set_offsets(data1)
        scat2.set_offsets(data2)
        scat3.set_offsets(data3)
        return scat1, scat2, scat3

    anim = FuncAnimation(fig, update, frames=len(samples1), init_func=init, blit=True)

    if save:
        anim.save(filename + ".gif", writer='pillow')
     
    # Display the animation in the notebook
    return HTML(anim.to_jshtml())

# Example usage
# animate_diffusion_3(samples1, samples2, samples3, filename="combined_animation", save=True)

def animate_diffusion_4(samples1, samples2, samples3, filename="", save=False, interval=200):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    scat1 = axs[0].scatter([], [])
    scat2 = axs[1].scatter([], [])
    scat3 = axs[2].scatter([], [])

    def init():
        for ax in axs:
            ax.set_xlim(-25, 25)
            ax.set_ylim(-25, 25)
        return scat1, scat2, scat3

    def update(frame):
        data1 = samples1[frame].cpu()
        data2 = samples2[frame].cpu()
        data3 = samples3[frame].cpu()

        scat1.set_offsets(data1)
        scat2.set_offsets(data2)
        scat3.set_offsets(data3)
        return scat1, scat2, scat3

    anim = FuncAnimation(fig, update, frames=len(samples1), init_func=init, blit=True, interval=interval)

    if save:
        anim.save(filename + ".gif", writer='pillow')
     
    # Display the animation in the notebook
    return HTML(anim.to_jshtml())

# Example usage
# animate_diffusion_3(samples1, samples2, samples3, filename="combined_animation", save=True)
