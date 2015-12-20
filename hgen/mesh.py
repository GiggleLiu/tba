'''
Meshes for hamitonian Generator
'''
from numpy import *
from multithreading import mpido
from scipy.linalg import eigh,eigvalsh
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utils import H2G
import pdb,time

__all__=['Emesh','Hmesh','Gmesh']

class Emesh(object):
    '''
    Energy Mesh.

    Construct
    ----------------
    Emesh(data), data is an N-D(N>=2) array or the filename of data file.

    Attributes
    ---------------
    data:
        The mesh data of energy arrange as [dim_1,...dim_n,nband].
    '''
    def __init__(self,data):
        if type(data)==str:
            self.data=loadtxt(data)
        else:
            self.data=data

    def save(filename):
        '''Save the hamiltonian data.'''
        save(filename,self.data)

    @property
    def nband(self):
        '''The number of bands'''
        return self.data.shape[-1]

    @property
    def size(self):
        '''The size of mesh.'''
        return self.data.shape[:-1]

    def show(self,kmesh=None):
        '''
        Show the dispersion relation.

        kmesh:
            N-D(N=1,3) array, The k-mesh holding this <Emesh>.
        '''
        dim=ndim(self.data)
        nband=self.nband
        colors=cm.get_cmap('autumn')(linspace(0,1,nband))
        if dim>3 or dim<=1:
            raise Exception('Not Implemented for mesh with dimension %s!'%(dim))
        elif dim==2:
            if kmesh is None:
                kmesh=arange(len(self.data))
            plot(self.data.T)
        else:
            N1,N2=self.size
            if kmesh is None:
                kx,ky=meshgrid(arange(N1),arange(N2),indexing='ij')
            else:
                kx,ky=kmesh[...,0],kmesh[...,1]
            fig=gcf()
            ax=fig.add_subplot(111,projection='3d')
            for i in xrange(nband):
                ax.scatter(kx,ky,self.data[...,i],s=5,edgecolor='none',c=colors[i])
        legend(arange(nband))

    def show_dos(self,wlist,geta=3e-2,inverse_axis=False,weights=None):
        '''
        Get a List instance of dos.

        wlist:
            the frequency space.
        geta: 
            smearing factor.
        inverse_axis:
            inverse x-y axis.
        weights:
            weights of energies.
        '''
        nw=len(wlist)
        dos=List(shape=[nw],dtype='float64')
        dos[...]=(weights/(wlist[:,newaxis]+1j*geta-reshape(self.data,[1,-1]))).imag.sum(axis=-1)
        dos*=-1./pi/prod(self.data.shape)
        if inverse_axis:
            wlist,dos=dos,wlist
        plot(wlist,dos)
        return wlist,dos

class Hmesh(object):
    '''
    Mesh of Hamiltonians.

    Construct
    ----------------
    Hmesh(data), data is an N-D(N>=3) array or the filename of data file.

    Attributes
    ---------------
    data:
        The mesh data of Hamiltonian arrange as [dim_1,...dim_n,nband,nband].
    '''
    def __init__(self,data):
        if type(data)==str:
            self.data=loadtxt(data)
        else:
            self.data=data

    @property
    def size(self):
        '''The size of mesh.'''
        return self.data.shape[:-2]

    @property
    def nband(self):
        '''The number of bands'''
        return self.data.shape[-1]

    def save(filename):
        '''Save the hamiltonian data.'''
        save(filename,self.data)

    def getemesh(self,evalvk=False):
        '''
        Get an Ek(with or without vk) mesh.

        evalvk:
            Evaluate vkmesh if True.
        '''
        nband=self.nband
        dmesh=mpido(func=eigh if evalvk else eigvalsh,inputlist=self.data.reshape([-1,nband,nband]))
        if evalvk:
            ekl,vkl=[],[]
            for ek,vk in dmesh:
                ekl.append(ek)
                vkl.append(vk)
            return reshape(ekl,self.data.shape[:,-1]),reshape(vkl,self.data.shape)
        else:
            return reshape(dmesh,self.data.shape[:-1])

    def getgmesh(self,w,sigma=None,tp='r',geta=1e-2,**kwargs):
        '''
        Get the Green's function mesh(Gwmesh) instance.

        w:
            an array(or a float number) of energy(frequency).
        sigma:
            self energy correction.
        tp:
            type of green's function.

            * 'r' - retarded.(default)
            * 'a' - advanced.(default)
            * 'matsu' - finite temperature.
        geta:
            smearing factor, default is 1e-2.
        '''
        if ndim(w)==0:
            #only 1 w is to be computed
            return H2G(w=w,h=self.data,sigma=sigma,tp=tp,geta=geta)
        else:
            #generate a mesh on w-list
            gmesh=H2G(w=w[[slice(None)]+[newaxis]*ndim(self.data)],h=self.data,sigma=sigma,tp=tp,geta=geta)
        return gmesh

class Gmesh(object):
    '''
    Mesh of Green's function

    Construct
    --------------------
    Gmesh(data,geta,tp,T=None)

    Attributes
    ---------------------
    data:
        A array of Green's function data
    geta:
        The smearing factor.
    tp: 
        The type of Green's function.

            * 'r' - retarded.(default)
            * 'a' - advanced.
            * 'matsu' - matsubara Green's function.
    T:
        The temperature for matsubara Green's function.
    '''
    def __init__(self,data,geta,tp,T=None):
        assert(tp=='r' or tp=='a' or (tp=='matsu' and not T is None))
        self.data=data
        self.tp=tp
        self.geta=geta
        self.T=T

    @property
    def dimension(self):
        return ndim(self.data)-2

    @property
    def Amesh(self):
        '''
        Get the spectrum function Ak-mesh
        '''
        data=self.data
        GH=swapaxes(data.conj(),axis1=-1,axis2=-1)
        if self.tp=='r':
            res= 1j/2./pi*(data-GH)
        elif self.tp=='a':
            res= -1j/2./pi*(data-GH)
        else:
            raise Exception('Error','Rules for spectrum of matsubara Green\'s function is not set.')
        return res

    def show_dos(self,lw=3,inverse=False,**kwargs):
        '''
        show dos.
        '''
        nflv=self.data.shape[-1]
        dos=trace(self.Amesh[...,:nflv,:nflv],axis1=-1,axis2=-2)
        ddim=ndim(self.data)
        if ddim!=3:
            kaxis=range(self.dimension)
            kaxis.remove(self.iaxis)
            dos=mean(dos,axis=kaxis)
