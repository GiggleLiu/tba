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

__all__=['sx','sy','sz','s','s1x','s1y','s1z','s1','fermi','H2G','s2vec','vec2s',
        'ind2c','c2ind','perm_parity','EU2C','C2H','bcast_dot']

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


