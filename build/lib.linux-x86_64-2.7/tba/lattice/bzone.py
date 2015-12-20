#!/usr/bin/python
'''
The objects and utilities related to Brillouin zone.
'''
from numpy import *
from numpy.linalg import inv
from matplotlib.pyplot import *
from matplotlib import path,patches
import pdb,time

__all__=['BZone']

class BZone(object):
    '''
    Brilouin zone Class.

    Construct
    --------------
    BZone(vertices)

    Attributes
    -----------------
    vertices:
        Vertices of the Brillouin zone.
    
    Note
    -----------------
    Up to now, it is only valid for 2D system.
    '''
    def __init__(self,vertices):
        self.vertices=path.Path(vertices)

    def inbzone(self,k,radius=1e-8):
        '''
        Decide whether a k-point is in bzone.

        k:
            A(or an array of) momentum vector.
        radius:
            Count points within radius into accound.
        '''
        assert(shape(k)[-1]==2)
        if ndim(k)==1:
            return self.vertices.contains_point(k,radius=radius)
        oshape=shape(k)[:-1]
        kl=k.reshape([-1,2])
        return reshape(array([self.vertices.contains_point(ki,radius=radius) for ki in kl],dtype='bool'),oshape)

    def k2bzone(self,k,b):
        '''
        Transfrom a k point to bzone.

        k:
            A(or array of) momentum vector(s).
        b:
            The displace vectors.
        '''
        raise Exception('Not Implemented!')

    def show(self,**kwargs):
        '''
        Plot the border or Brillouin zone.
        '''
        patch=patches.PathPatch(self.vertices,facecolor='none',**kwargs)
        gca().add_patch(patch)


