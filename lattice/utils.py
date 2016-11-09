'''
Author: Giggle Leo
Date : 8 November 2015
Description : physics library
'''
from numpy import *
from numpy.linalg import inv

__all__=['toreciprocal','c2ind','ind2c','meshgrid_v','bisect']

def toreciprocal(a):
    '''
    Get the reciprocal lattice vectors.

    a:
        matrix of [a1,a2].T.
    '''
    res=2*pi*inv(a).T
    return res

def c2ind(c,N):
    '''
    Get the index of the total space N from the index of the subspace exression (n1 x n2 x n3...)

    c: 
        a list of indexes like [i,j,k,...]
    N: 
        the space config [n1,n2,n3...].
    '''
    c=array(c)
    n=shape(c)[-1]
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

def meshgrid_v(siteconfig,vecs):
    '''
    Get the meshgrid in a vector space.

    Parameters:
        :siteconfig: list of 1D array, the configuration of sites.
        :evcs: list of 1D array, the vectors spanning this space.

    Return:
        The ND+1 meshgrid defined on this vector space.
    '''
    mg=ix_(*siteconfig)
    mr=[g[...,newaxis]*vec for vec,g in zip(vecs,mg)]
    return sum(mr,axis=0)

