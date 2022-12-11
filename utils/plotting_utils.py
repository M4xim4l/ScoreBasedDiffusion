import numpy as np
import matplotlib.pyplot as plt

def create_3d_hist(ax, data, x_min, x_max, y_min, y_max, n_bins=50):
    data_np = data.detach().numpy()
    hist, xedges, yedges = np.histogram2d(data_np[:,0], data_np[:,1], bins=n_bins, range=[[x_min, x_max], [y_min, y_max]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    #https://stackoverflow.com/questions/44895117/colormap-for-3d-bar-plot-in-matplotlib-applied-to-every-bar
    max_height = np.max(dz)
    min_height = np.min(dz)

    cmap = plt.cm.get_cmap('plasma')
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=rgba)


def create_2d_hist(ax, data, x_min, x_max, y_min, y_max, n_bins=50):
    data_np = data.detach().numpy()
    ax.hist2d(data_np[:,0], data_np[:,1], bins=n_bins, range=[[x_min, x_max], [y_min, y_max]], cmap='plasma')
    #ax.set_xticks([])
    #ax.set_yticks([])
