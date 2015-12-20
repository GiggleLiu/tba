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

__all__=['sx','sy','sz','s','H2G','s2vec','vec2s','ind2c','c2ind']

############################ DEFINITIONS ##############################
# pauli spin
sx = array([[0, 1],[ 1, 0]])
sy = array([[0, -1j],[1j, 0]])
sz = array([[1, 0],[0, -1]])
s=[identity(2),sx,sy,sz]
############################ FUNCTIONS ##############################
def H2G(h,w,tp='r',geta=1e-2,sigma=None):
    '''
    Get Green's function g from Hamiltonian h.

    h: 
        an array of hamiltonian.
    w:
        the energy(frequency).
    tp:
        the type of Green's function.
        'r': retarded Green's function.(default)
        'a': advanced Green's function.
        'matsu': finite temperature Green's function.
    geta:
        smearing factor. default is 1e-2.
    sigma:
        additional self energy.
    *return*:
        a Green's function.
    '''
    if tp=='r':
        z=w+1j*geta
    elif tp=='a':
        z=w-1j*geta
    elif tp=='matsu':
        z=1j*w
    if sigma!=None:
        h=h+sigma

    hdim=ndim(h)
    if hdim>0:
        return inv(z*identity(h.shape[-1])-h)
    else:
        return 1./(z-h)

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

    c: 
        a list of indexes like [i,j,k,...]
    N: 
        the space config [n1,n2,n3...].
    '''
    n=c.shape[-1]
    cc=c[...,0]
    for i in xrange(n-1):
        cc=cc*N[i+1]+c[...,i+1]
    return cc

def ind2c(ind,N):
    '''
    Get the index of the index of the subspace from the total space N.

    ind:
        the index of total space.
    N:
        the space config [n1,n2,n3...].
    '''
    dim=len(N)
    indl=ndarray(list(shape(ind))+[dim],dtype='int32')
    for i in xrange(dim):
        indl[...,-1-i]=ind%N[-1-i]
        ind=ind/N[-1-i]
    return indl
