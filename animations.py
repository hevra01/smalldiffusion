import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import HTML
import torch

def animate_diffusion(samples, filename="", save=False):
    fig, ax = plt.subplots()
    scat = ax.scatter([], [])
    print("len(samples)", len(samples))

    # initialize the min and max to be min and max of the first sample. 
    # note that samples represent the different denoising step
    # furthermore, in each sample, there are several other data points.
    # Concatenate all samples along a new dimension to aggregate in one go
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
