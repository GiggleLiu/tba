'''
Meshes for hamitonian Generator
'''

from numpy import *
from scipy.linalg import eigh,eigvalsh
from numpy.linalg import inv
import pdb,time

from multithreading import mpido
from utils import bcast_dot

__all__=['random_H','E2DOS','H2E','G2A','A2DOS','H2G','EU2C','C2H']

def random_H(nband,size=()):
    '''
    Generate random Hamiltonian.

    Parameters:
        :nband: int, the number of bands.
        :size: tuple, the size of hamiltonian.

    Return:
        Hamiltonian mesh.
    '''
    shape=concatenate([size,[nband,nband]])
    H=random.random(shape)+1j*random.random(shape)
    H=H+H.swapaxes(-1,-2).conj()
    return H

def E2DOS(Emesh,wlist,weights=1.,geta=3e-2):
    '''
    Get density of states from energy list(not normalized).
    
    Parameters:
        :Emesh: ndarray, the energy mesh.
        :wlist: 1d array, the spectrum space.
        :weights: number/ndarray, the weight of each energy.
        :geta: float, the smearing factor.

    Return:
        1d array.
    '''
    nw=len(wlist)
    dos=(weights/(wlist[:,newaxis]+1j*geta-reshape(Emesh,[1,-1]))).imag.sum(axis=-1)
    dos*=-1./pi
    return dos

def H2E(Hmesh,evalvk=False):
    '''
    Get an Ek(with or without vk) mesh.

    evalvk:
        Evaluate vkmesh if True.
    '''
    nband=Hmesh.shape[-1]
    dmesh=mpido(func=eigh if evalvk else eigvalsh,inputlist=Hmesh.reshape([-1,nband,nband]))
    if evalvk:
        ekl,vkl=[],[]
        for ek,vk in dmesh:
            ekl.append(ek)
            vkl.append(vk)
        return reshape(ekl,Hmesh.shape[:-1]),reshape(vkl,Hmesh.shape)
    else:
        return reshape(dmesh,Hmesh.shape[:-1])

def H2G(Hmesh,w,sigma=None,tp='r',geta=1e-2):
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
    if ndim(w)!=0:
        w=w[[slice(None)]+[newaxis]*ndim(Hmesh)]
    #only 1 w is to be computed
    if tp=='r':
        z=w+1j*geta
    elif tp=='a':
        z=w-1j*geta
    elif tp=='matsu':
        z=1j*w
    if sigma!=None:
        Hmesh=Hmesh+sigma

    hdim=ndim(Hmesh)
    if hdim>0:
        return inv(z*identity(Hmesh.shape[-1])-Hmesh)
    else:
        return 1./(z-Hmesh)

def G2A(Gmesh,tp='r'):
    '''
    Get the spectrum function Ak-mesh
    '''
    GH=swapaxes(Gmesh.conj(),axis1=-1,axis2=-1)
    if tp=='r':
        res= 1j/2./pi*(Gmesh-GH)
    elif tp=='a':
        res= -1j/2./pi*(Gmesh-GH)
    else:
        raise Exception('Error','Rules for spectrum of matsubara Green\'s function is not set.')
    return res

def A2DOS(Amesh):
    '''
    Get the density of states from spectrum function.

    Parameters:
        :Amesh: ndarray, mesh of spectrums.
    '''
    dos=trace(Amesh,axis1=-1,axis2=-2)
    return dos

def EU2C(E,U,T=0):
    '''
    Get the expectation matrix in k-space.

    Parameters:
        :E,U: ndarray, the eigenvalues and eigenvectors defined on real space.
        ndim(E)>=2 and ndim(U)=ndim(E)+1.

    Return:
        ndarray, the expectation matrix(the expectation mesh of <ck^\dag,ck>)
    '''
    assert(ndim(E)>=1 and ndim(U)==ndim(E)+1)
    fm=fermi(E,T=T)
    C=bcast_dot((U.conj()*fm[...,newaxis,:]),swapaxes(U,-1,-2))
    return C

def C2H(C,T=1.):
    '''
    Get the entanglement hanmiltonian from expectation matrix.

    Parameters:
        :C: ndarray, the expectation matrix.
        :T: float, the temperature.

    Return:
        ndarray, the hamiltonian.
    '''
    CE,CU=eigh(C)
    print 'Checking for the range fermionic occupation numbers, min -> %s, max -> %s.'%(CE.min(),CE.max())
    assert(all(CE>0) and all(CE<1))
    H=bcast_dot(CU.conj()*(log(1./CE-1)*T)[...,newaxis,:],swapaxes(CU,-1,-2))
    return H


