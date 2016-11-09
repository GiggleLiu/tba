#!/usr/bin/python
'''
Momentum-space related classes and operations.
'''
import time,pdb
from numpy import *
from group import TranslationGroup
from utils import toreciprocal,ind2c,c2ind,meshgrid_v
from numpy.linalg import norm
from scipy.spatial import cKDTree
from bzone import BZone

__all__=['KSpace']

class KSpace(object):
    '''
    Lattice in K space(reciprocal lattice).

    Attributes:
        :b: 2d array, reciprocal lattice vectors.
        :N: 1d array, number of samples in each direction.
        :kmesh: ndarray, mesh of momentum vectors.
        :groups: ndarray, point groups used by this KSpace.
        :special_points: dict, special points

            * `G`(center), can also be refer as <KSpace>.G
            * `K`(vertices), can also be refer as <KSpace>.K
            * `M`(edgecenter), can also be refer as <KSpace>.M
            * `TRI`(Time Reversal invarient) points, can also be refer as <KSpace>.TRI
    '''
    def __init__(self,b,N):
        self.b=array(b)
        self.kmesh=meshgrid_v([arange(n) for n in N],vecs=b/array(N)[:,newaxis])
        self.groups={}
        G=zeros(self.b.shape[-1],dtype='float64')
        self.special_points={'G':G}

        #add transplation symmetry group
        #tg=TranslationGroup(self.b,per=arange(self.dimension))
        #self.usegroup(tg)

    @property
    def N(self):
        '''The mesh size of momentum vectors.'''
        return self.kmesh.shape[:-1]

    @property
    def dimension(self):
        '''The dimension of the k-mesh.'''
        return ndim(self.kmesh)-1

    @property
    def vdim(self):
        '''The vector dimension of this K-space.'''
        return self.b.shape[-1]

    @property
    def G(self):
        '''
        The `G` point if k space.
        '''
        return self.special_points.get('G')

    @property
    def K(self):
        '''
        The `K` point if k space.
        '''
        return self.special_points.get('K')

    @property
    def M(self):
        '''
        The `M` point if k space.
        '''
        return self.special_points.get('M')

    @property
    def TRI(self):
        '''Time reversal invarient points.'''
        return meshgrid_v([arange(-0.5,1.,0.5)]*len(N),vecs=b).reshape([-1,vdim])

    def usegroup(self,g):
        '''
        Use specific point group.

        Parameters:
            :g: <Group>,
        '''
        if g.tp=='translation': pass
        self.groups[g.tp]=g
        print self.M,self.G
        vm=(self.M[0]-self.G)[...,newaxis]
        vk=(self.K[0]-self.G)[...,newaxis]
        gkm=concatenate([vm,vk],axis=1) #gmk is matrix representation of the triangular irredicible zone (vm,vk).
        g.irzone=gkm

    def get_bzone(self):
        '''
        Get Brillouin zone.
        '''
        bzoneborder=concatenate([self.K,self.K[0:1]])
        return BZone(bzoneborder)

