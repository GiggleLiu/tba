'''
Author: Giggle Leo
Date : 8 September 2014
Description : physics library
'''

from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *
import scipy.sparse as sps
import pdb,time
import cPickle as pickle

__all__=['sx','sy','sz','s','s1x','s1y','s1z','s1','fermi','s2vec','vec2s',
        'ind2c','c2ind','perm_parity','bcast_dot','quicksave','quickload',
        'inherit_docstring_from']
############################ DEFINITIONS ##############################
# pauli spin
sx = array([[0, 1],[ 1, 0]])
sy = array([[0, -1j],[1j, 0]])
sz = array([[1, 0],[0, -1]])
s=[identity(2),sx,sy,sz]

# spin 1 matrices.
s1x=array([[0,1,0],[1,0,1],[0,1,0]])/sqrt(2)
s1y=array([[0,-1j,0],[1j,0,-1j],[0,1j,0]])/sqrt(2)
s1z=array([[1,0,0],[0,0,0],[0,0,-1]])
s1=[identity(3),s1x,s1y,s1z]
############################ FUNCTIONS ##############################

def bcast_dot(A,B):
    '''
    dot product broadcast version.
    '''
    return einsum('...jk,...kl->...jl', A, B)

def fermi(elist,T=0):
    '''
    Fermi statistics, python implimentation.

    Parameters:
        :elist: float/ndarray, the energy.
        :T: float, the temperature.

    Return:
        float/ndarray, Fermionic disctribution.
    '''
    elist=asarray(elist)
    if T<0.:
        raise ValueError('Negative temperature is not allowed!')
    elif T==0:
        if ndim(elist)!=0:
            f=zeros(elist.shape,dtype='float64')
            f[elist<0]=1.
            f[elist==0]=0.5
            return f
        else:
            if elist>0:
                return 0.
            elif elist==0:
                return 0.5
            else:
                return 1.
    else:
        f=1./(1.+exp(-abs(elist)/T))
        if ndim(elist)!=0:
            posmask=elist>0
            f[posmask]=1.-f[posmask]
        elif elist>0:
            f=1.-f
        return f

def s2vec(s):
    '''
    Transform a spin to a 4 dimensional vector, corresponding to s0,sx,sy,sz component.

    s: 
        the spin.
    '''
    res=array([trace(s),trace(dot(sx,s)),trace(dot(sy,s)),trace(dot(sz,s))])/2
    return res

def vec2s(n):
    '''
    Transform a vector of length 3 or 4 to a pauli matrix.

    n: 
        a 1-D array of length 3 or 4 to specify the `direction` of spin.
    *return*:
        2 x 2 matrix.
    '''
    if len(n)<=3:
        res=zeros([2,2],dtype='complex128')
        for i in xrange(len(n)):
            res+=s[i+1]*n[i]
        return res
    elif len(n)==4:
        return identity(2)*n[0]+sx*n[1]+sy*n[2]+sz*n[3]
    else:
        raise Exception('length of vector %s too large.'%len(n))


def c2ind(c,N):
    '''
    Get the index of the total space N from the index of the subspace exression (n1 x n2 x n3...)

    Parameters:
        :c: 1D array/2D array, a list of indexes like [i,j,k,...]
        :N: 1D array, the space config [n1,n2,n3...].

    Return:
        integer/1D array, indices.
    '''
    assert(shape(c)[-1]==len(N))
    c=array(c)
    n=c.shape[-1]
    cc=c[...,0]
    for i in xrange(n-1):
        cc=cc*N[i+1]+c[...,i+1]
    return cc

def ind2c(ind,N):
    '''
    Trun global index into sub-indices.

    Parameters:
        :ind: integer, the index of total space.
        :N: 1D array, the space config [n1,n2,n3...].

    Return:
        1D array, the subindices.
    '''
    dim=len(N)
    indl=ndarray(list(shape(ind))+[dim],dtype='int32')
    for i in xrange(dim):
        indl[...,-1-i]=ind%N[-1-i]
        ind=ind/N[-1-i]
    return indl

def perm_parity(perm):
    """ 
    Returns the parity of the perm(0 or 1). 
    """
    size=len(perm)
    unchecked=ones(size,dtype='bool')
    #c counts the number of cycles in the perm including 1 cycles
    c=0
    for j in xrange(size):
        if unchecked[j]:
            c=c+1
            unchecked[j]=False
            i=j
            while perm[i]!=j:
                i=perm[i]
                unchecked[i]=False
    return (size-c)%2

def quicksave(filename,obj):
    '''Save an instance.'''
    f=open(filename,'wb')
    pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    f.close()

def quickload(filename):
    '''Load an instance.'''
    f=open(filename,'rb')
    obj=pickle.load(f)
    f.close()
    return obj

def inherit_docstring_from(cls):
    def docstring_inheriting_decorator(fn):
        fn.__doc__ = getattr(cls, fn.__name__).__doc__
        return fn
    return docstring_inheriting_decorator


