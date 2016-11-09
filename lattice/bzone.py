#!/usr/bin/python
'''
The objects and utilities related to Brillouin zone.
'''
from numpy import *
from numpy.linalg import inv
from matplotlib import path
import pdb,time

__all__=['BZone']

class BZone(object):
    '''
    Brilouin zone Class.

    Attributes:
        :vertices: 2darray, vertices of the Brillouin zone.
    
    Note:
        To now, it is only valid for 2D system.
    '''
    def __init__(self,vertices):
        self.vertices=vertices
        self._path=path.Path(vertices)

    def inbzone(self,k,tol=1e-8):
        '''
        Decide whether a k-point is in bzone.

        Parameters:
            :k: ndarray, momentum vector(s).
            :tol: float, points out of border within tol is be regarded as in.

        Return:
            bool
        '''
        assert(shape(k)[-1]==2)
        if ndim(k)==1:
            return self._path.contains_point(k,radius=tol)
        oshape=shape(k)[:-1]
        kl=k.reshape([-1,2])
        return reshape(array([self._path.contains_point(ki,radius=tol) for ki in kl],dtype='bool'),oshape)

    def k2bzone(self,k,b):
        '''
        Move k point to bzone by group action.

        Parameters:
            :k: 1d array, momentum vector(s).
            :b: 1d array, The displace vectors.
        '''
        raise NotImplementedError()

