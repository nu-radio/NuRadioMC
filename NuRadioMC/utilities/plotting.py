from __future__ import absolute_import, division, print_function
import numpy as np
from matplotlib import pyplot as plt
from NuRadioReco.utilities import units

def plot_vertex_distribution(xx, yy, zz, weights=None,
                            rmax=None, zmin=None,
                            trigger_name=""):
    """
    produces beautiful plot of the vertex distribution

    Parameters:
    xx: array
        the x positions of the vertices
    yy: array
        the y positions of the vertices
    zz: array
        the z positions of the vertices

    Returns
    -------
    fig, ax
    """
    if(weights is None):
        weights = np.ones_like(xx)
    fig, ax = plt.subplots(1, 1)
    rr = (xx ** 2 + yy ** 2) ** 0.5
    mask_weight = weights > 1e-2
    max_r = rr[mask_weight].max()
    max_z = np.abs(zz[mask_weight]).max()
    if(rmax is None):
        rmax = max_r
    if(zmin is None):
        zmin = zz.min()
    h = ax.hist2d(rr / units.m, zz / units.m, bins=[np.linspace(0, max_r, 50), np.linspace(-max_z, 0, 50)],
                  cmap=plt.get_cmap('Blues'), weights=weights)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(h[3], ax=ax, cax=cax)
    cb.set_label("# (weighted)")
    ax.set_aspect('equal')
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_xlim(0, rmax)
    ax.set_ylim(zmin, 0)
    if(trigger_name != ""):
        ax.set_title("trigger: {}".format(trigger_name))
    fig.tight_layout()
    return fig, ax
