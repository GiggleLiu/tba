from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

__all__=['plot_e','pcolor_e','scatter_e']

def plot_e(klist,elist,**kwargs):
    '''
    Show the dispersion relation.

    Parameters:
        :klist: 1d array, flat k-list holding this elist.
        :elist: 1d, 2d array, list of energies.
    '''
    elist=elist if ndim(elist)==2 else elist[:,newaxis]
    nband=elist.shape[-1]
    if klist is None:
        klist=arange(len(elist))
    plot(klist,elist,**kwargs)
    legend(arange(nband))

def pcolor_e(kmesh,Emesh,**kwargs):
    '''
    Show the dispersion relation.

    Parameters:
        :kmesh: 3d array, k-mesh holding this elist.
        :Emesh: 2d array, list of energies.
    '''
    assert(ndim(Emesh)==2)
    N1,N2=Emesh.shape
    if kmesh is None:
        kx,ky=meshgrid(arange(N1),arange(N2),indexing='ij')
    else:
        kx,ky=kmesh[...,0],kmesh[...,1]
    pcolor(kx,ky,Emesh,**kwargs)

def scatter_e(kmesh,Emesh,**kwargs):
    '''
    Show the dispersion relation.

    Parameters:
        :kmesh: 3d array, k-mesh holding this emesh.
        :Emesh: 3d array, multi-band energies.
    '''
    assert(ndim(Emesh)==3)
    nband=Emesh.shape[-1]
    colors=cm.get_cmap('autumn')(linspace(0,1,nband))
    N1,N2=Emesh.shape[:2]
    if kmesh is None:
        kx,ky=meshgrid(arange(N1),arange(N2),indexing='ij')
    else:
        kx,ky=kmesh[...,0],kmesh[...,1]
    fig=gcf()
    ax=fig.add_subplot(111,projection='3d')
    for i in xrange(nband):
        ax.scatter(kx,ky,Emesh[...,i],s=5,edgecolor='none',c=colors[i],**kwargs)

def plotconfig(spaceconfig,config,offset=zeros(2)):
    '''
    Display a config of electron.

    Parameters:
        :config: 1D array/2D array, len-nsite array with items 0,1(state without/with electron).
    '''
    cfg=config.reshape(spaceconfig.config)
    cfg=swapaxes(cfg,-1,-2).reshape([-1,spaceconfig.natom])
    x,y=meshgrid(spaceconfig.get_indexer('atom'),arange(spaceconfig.nsite/spaceconfig.natom))
    colors=cm.get_cmap('rainbow')(float64(cfg))
    scatter(x+offset[0],y+offset[1],s=20,c=colors.reshape([-1,4]))


