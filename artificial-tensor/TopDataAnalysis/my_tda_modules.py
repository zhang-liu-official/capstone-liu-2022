import persim
# Basic imports 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
# from IPython.display import Video

# scikit-tda imports..... Install all with -> pip install scikit-tda
#--- this is the main persistence computation workhorse
import ripser
# from persim import plot_diagrams
import persim
# import persim.plot

# teaspoon imports...... Install with -> pip install teaspoon
#---these are for generating data and some drawing tools 
import teaspoon.MakeData.PointCloud as makePtCloud
import teaspoon.TDA.Draw as Draw

#---these are for generating time series network examples
from teaspoon.SP.network import ordinal_partition_graph
from teaspoon.TDA.PHN import PH_network
from teaspoon.SP.network_tools import make_network
from teaspoon.parameter_selection.MsPE import MsPE_tau
import teaspoon.MakeData.DynSysLib.DynSysLib as DSL

import kmapper as km
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from kmapper import jupyter 
import pydiffmap
from pydiffmap import diffusion_map as dm
from pydiffmap.visualization import embedding_plot, data_plot

# import dionysus as ds
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import pylab as pl
from matplotlib import collections  as mc
from sklearn.datasets import load_digits
from skimage.morphology import skeletonize
import math
import sys

from gudhi.wasserstein import wasserstein_distance
from gudhi.hera import wasserstein_distance as hera

class Barcode:
    __doc__ = """
        Barcode visualisation made easy!
        Note that this convenience class requires instantiation as the number
        of subplots produced depends on the dimension of the data.
        """

    def __init__(self, diagrams, verbose=False):
        """
        Parameters
        ===========
        diagrams: list-like
            typically the output of ripser(nodes)['dgms']
        verbose: bool
            Execute print statemens for extra information; currently only echoes
            number of bars in each dimension (Default=False).
        Examples
        ===========
        >>> n = 300
        >>> t = np.linspace(0, 2 * np.pi, n)
        >>> noise = np.random.normal(0, 0.1, size=n)
        >>> data = np.vstack([((3+d) * np.cos(t[i]+d), (3+d) * np.sin(t[i]+d)) for i, d in enumerate(noise)])
        >>> diagrams = ripser(data)
        >>> bc = Barcode(diagrams['dgms'])
        >>> bc.plot_barcode()
        """
        if not isinstance(diagrams, list):
            diagrams = [diagrams]

        self.diagrams = diagrams
        self._verbose = verbose
        self._dim = len(diagrams)

    def plot_barcode(self, figsize=None, show=True, export_png=False, dpi=100, **kwargs):
        """Wrapper method to produce barcode plot
        Parameters
        ===========
        figsize: tuple
            figure size, default=(6,6) if H0+H1 only, (6,4) otherwise
        show: boolean
            show the figure via plt.show()
        export_png: boolean
            write image to png data, returned as io.BytesIO() instance,
            default=False
        **kwargs: artist paramters for the barcodes, defaults:
            c='grey'
            linestyle='-'
            linewidth=0.5
            dpi=100 (for png export)
        Returns
        ===========
        out: list or None
            list of png exports if export_png=True, otherwise None
        """
        if self._dim == 2:
            if figsize is None:
                figsize = (6, 6)

            return self._plot_H0_H1(
                figsize=figsize,
                show=show,
                export_png=export_png,
                dpi=dpi,
                **kwargs
            )

        else:
            if figsize is None:
                figsize = (6, 4)

            return self._plot_Hn(
                figsize=figsize,
                show=show,
                export_png=export_png,
                dpi=dpi,
                **kwargs
            )

    def _plot_H0_H1(self, *, figsize, show, export_png, dpi, **kwargs):
        out = []

        fig, ax = plt.subplots(2, 1, figsize=figsize)

        for dim, diagram in enumerate(self.diagrams):
            self._plot_many_bars(dim, diagram, dim, ax, **kwargs)

        if export_png:
            fp = io.BytesIO()
            plt.savefig(fp, dpi=dpi)
            fp.seek(0)

            out += [fp]

        if show:
            plt.show()
        else:
            plt.close()

        if any(out):
            return out

    def _plot_Hn(self, *, figsize, show, export_png, dpi, **kwargs):
        out = []

        for dim, diagram in enumerate(self.diagrams):
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            self._plot_many_bars(dim, diagram, 0, [ax], **kwargs)

            if export_png:
                fp = io.BytesIO()
                plt.savefig(fp, dpi=dpi)
                fp.seek(0)

                out += [fp]

            if show:
                plt.show()
            else:
                plt.close()

        if any(out):
            return out

    def _plot_many_bars(self, dim, diagram, idx, ax, **kwargs):
        number_of_bars = len(diagram)
        if self._verbose:
            print("Number of bars in dimension %d: %d" % (dim, number_of_bars))

        if number_of_bars > 0:
            births = np.vstack([(elem[0], i) for i, elem in enumerate(diagram)])
            deaths = np.vstack([(elem[1], i) for i, elem in enumerate(diagram)])

            inf_bars = np.where(np.isinf(deaths))[0]
            max_death = deaths[np.isfinite(deaths[:, 0]), 0].max()

            number_of_bars_fin = births.shape[0] - inf_bars.shape[0]
            number_of_bars_inf = inf_bars.shape[0]

            _ = [self._plot_a_bar(ax[idx], birth, deaths[i], max_death, **kwargs) for i, birth in enumerate(births)]

        # the line below is to plot a vertical red line showing the maximal finite bar length
        ax[idx].plot(
            [max_death, max_death],
            [0, number_of_bars - 1],
            c='r',
            linestyle='--',
            linewidth=0.7
        )

        title = "H%d barcode: %d finite, %d infinite" % (dim, number_of_bars_fin, number_of_bars_inf)
        ax[idx].set_title(title, fontsize=9)
        ax[idx].set_yticks([])

        for loc in ('right', 'left', 'top'):
            ax[idx].spines[loc].set_visible(False)

    @staticmethod
    def _plot_a_bar(ax, birth, death, max_death, c='gray', linestyle='-', linewidth=0.5):
        if np.isinf(death[0]):
            death[0] = 1.05 * max_death
            ax.plot(
                death[0],
                death[1],
                c=c,
                markersize=4,
                marker='>',
            )

        ax.plot(
            [birth[0], death[0]],
            [birth[1], death[1]], 
            c=c,
            linestyle=linestyle,
            linewidth=linewidth,
        )


def drawTDAtutorial(P,diagrams, R):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (20,5))

    # Draw diagrams
    plt.sca(axes[0])
    plt.title('0-dim Diagram')
    Draw.drawDgm(diagrams[0])
    plt.axis([0,R,0,R])
    
    plt.sca(axes[1])
    plt.title('1-dim Diagram')
    Draw.drawDgm(diagrams[1])
    plt.axis([0,R,0,R])
    
#     plt.sca(axes[2])
#     plt.title('2-dim Diagram')
#     Draw.drawDgm(diagrams[2])
#     plt.axis([0,R,0,R])

    
def diffusion_tda(X):
  ## diffusion map with automatic epsilon detection:
    mydmap = dm.DiffusionMap.from_sklearn(n_evecs = 3, alpha = 1, epsilon =  1.0 , k=100)

  # Fit to and transform the data
    X_dmap = mydmap.fit_transform(X)
  
    embedding_plot(mydmap, dim=3, scatter_kwargs = {'c': X_dmap[:,0], 'cmap': 'Spectral'})
    data_plot(mydmap, dim=3, scatter_kwargs = {'cmap': 'Spectral'})
    plt.show()
    print("SHAPE",X_dmap.shape)

    ax = plt.axes(projection ="3d")
    ax.scatter3D(X_dmap[:,0], X_dmap[:,1], X_dmap[:,2])
    plt.show()

    X_diagrams = ripser.ripser(X_dmap, maxdim = 1)['dgms']

  ## draw persistence diagrams
    drawTDAtutorial(X_dmap,X_diagrams,R=0.1) 

  ## draw persistence barcodes
    Barcode(X_diagrams).plot_barcode()
    
    return X_diagrams, Barcode(X_diagrams)

def pairwise_Wasserstein(X_diagrams):
# X_diagrams = []
# X_diagrams.append(X1_diagrams)
# X_diagrams.append(X2_diagrams)
# X_diagrams.append(X3_diagrams)
# X_diagrams.append(X4_diagrams)
# X_diagrams.append(X5_diagrams)
# X_diagrams.append(X6_diagrams)

    pd_gudhi = np.zeros((6,6))
    for dim in range(3):
        for i in range(6):
            for j in range(6):
                if i != j:
                    pd_gudhi[i,j] = hera(X_diagrams[i][dim], X_diagrams[j][dim],internal_p=2)
                    print('Wasserstein distance between persistence diagrams for' + str(i+1) +' and '+ str(j+1) + ' (H' + str(dim) + ') is: \n')
                    print(pd_gudhi[i,j] )
                    