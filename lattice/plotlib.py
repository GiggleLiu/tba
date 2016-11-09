from numpy import *
from matplotlib.pyplot import *
from matplotlib.collections import LineCollection
from matplotlib import patches
import time,pdb

from bond import BondCollection

__all__=['show_bonds','scatter_sites','show_kspace','show_bzone']

def show_bonds(bonds,offsets=0,lw=1,**kwargs):
    '''
    Display a collection of bonds.

    Parameters:
        :bonds: list/<BondCollection> instance.
        :offsets: number/2d array, the location of starting atoms.
        :lw,**kwargs: line width of bonds and key word arguments for 
    '''
    if len(bonds)==0: return
    if isinstance(bonds,list): bonds=BondCollection(bonds)
    vdim=bonds.vdim

    #calculate the start and end of bond vectors.
    bvs=[]
    if ndim(offsets)==0:
        offsets=offsets*ones([bonds.N,vdim])
    elif ndim(start)==1:
        bvs=zip(offsets,bonds.bondvs+offsets)
    else:
        bvs=zip(offsets,bonds.bondvs+offsets)
    if vdim==1:
        bvs=[(append(start,[0]),append(end,[0])) for start,end in bvs]  #???

    lc=LineCollection(bvs,**kwargs)
    lc.set_linewidth(lw)
    ax=gca()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

def scatter_sites(sites,offset=0,**kwargs):
    '''
    Show the sites.

    Parameters:
        :structure: <Structure>,
        :offset: 1d array, the vector offset.
        :color: str, the color.
    '''
    sites=sites+offset
    if sites.shape[-1]==2:
        x,y=sites[:,0],sites[:,1]
    else:
        x=sites[:,0]
        y=zeros(len(sites))
    scatter(x,y,edgecolor='none',**kwargs)

def showcell(lt,bondindex=(1,2),plane=(0,1),color='r',offset=None):
    '''
    Plot the cell structure.

    bondindex:
        the bondindex-th nearest bonds are plotted. It should be a tuple.
        Default is (1,2) - the nearest, and second nearest neightbors.
    plane:
        project to the specific plane if it is a 3D structre.
        Default is (0,1) - 'x-y' plane.
    color:
        color, default is 'r' -red.
    offset:
        The offset of the sample cell.
    '''
    p1,p2=plane
    nnnbonds=lt.cbonds
    if nnnbonds is None or len(nnnbonds)<=1:
        warnings.warn('Trivial bondindex@Structure.plotstructure, plot on-site terms only.')
        bondindex=()
    if offset is None:
        offset=[2]*lt.dimension
    for nb in bondindex:
        cb=nnnbonds[nb]
        cellindex=lt.l2index(append(offset,[0]))
        show_bonds(cb,start=lt.sites[cellindex+cb.atom1s])
    #plot sites
    if lt.vdim>1:
        p1,p2=plane
        x,y=lt.sites[:,p1],lt.sites[:,p2]
    else:
        x=lt.sites[:,0]
        y=zeros(lt.N)
    scatter(x,y,s=50,c=color,edgecolor='none',vmin=-0.5,vmax=1.5)

def show_kspace(kspace,**kwargs):
    '''
    Show the k-mesh of this <KSpace> instance.
    '''
    if kspace.vdim>2:
        raise NotImplementedError()
    if kspace.vdim==1:
        scatter(kspace.kmesh[...,0],zeros(kspace.N[0]),**kwargs)
    elif kspace.vdim==2:
        scatter(kspace.kmesh[...,0],kspace.kmesh[...,1],**kwargs)
    for pname,ps in kspace.special_points.iteritems():
        if ndim(ps)==1:
            ps=[ps]
        for i in xrange(len(ps)):
            x,y=ps[i] if kspace.vdim==2 else (ps[i],0)
            text(x,y,pname)
            scatter(x,y,edgecolor='none',color='r')

def show_bzone(bzone,**kwargs):
    '''
    Plot the border or Brillouin zone.
    '''
    patch=patches.PathPatch(bzone._path,facecolor='none',**kwargs)
    gca().add_patch(patch)
