#!/usr/bin/python
'''
Lattice related classes.

<Lattice> is a drivative class of <Structure>, it is known as the periodic structure.
'''
import time,os,pickle
from numpy import *
from group import TranslationGroup
from utils import toreciprocal,ind2c,c2ind,meshgrid_v
import itertools
from bond import Bond,BondCollection
from numpy.linalg import norm,solve
from scipy.spatial import cKDTree
from kspace import KSpace
from structure import Structure
import warnings,pdb

__all__=['Lattice']

class Lattice(Structure):
    '''
    Lattice Structure, which contains tranlation of cells.

    Attributes:
        :a: 2d array, lattice vectors in cols.
        :N: 1d array, number of cells for each lattice vector.
        :catoms: 2d array, atoms in one unit cell.
        :lmesh: ndarray, sites in lattice config (Nx,Ny, ..., ncatom).

    Read Only Attributes:
        :dimension: int, the lattice dimension, # of a vectors.
        :ncatom: int, the # of atoms if a unit cell.
        :siteconfig: 1d array, the configuration of space [n1,n2,...,ncatom].
    '''
    def __init__(self,a,N,catoms=[(0.,0.)]):
        self.catoms=array(catoms)
        self.a=array(a)
        self.N=array(N)
        
        mr=meshgrid_v([arange(n) for n in self.N],vecs=a)
        self.lmesh=(mr[...,newaxis,:]+self.catoms)

        vdim=self.a.shape[-1]
        items=self.lmesh.reshape([-1,vdim])
        super(Lattice,self).__init__(items)

    def __str__(self):
        return '''<Lattice> \na = %s,\nN = %s,\n# of atoms in one cell = %s.'''\
                %(list(self.a),' x '.join([str(n) for n in self.N]),self.ncatom)

    @property
    def dimension(self):
        '''The dimension of lattice.'''
        return len(self.N)

    @property
    def ncatom(self):
        '''number of atoms within a unit cell.'''
        return self.catoms.shape[0]

    @property
    def siteconfig(self):
        '''The site configuration, taking catoms into account.'''
        return list(self.N)+[self.ncatom]

    def index2l(self,index):
        '''
        Get lattice indices (n1,n2,...,atom index in cell) from site index.

        Parameters:
            :index: int, the site index.

        Return:
            1d array, lattice index - (n1,n2,...,cindex)
        '''
        return ind2c(index,N=self.siteconfig)

    def l2index(self,lindex):
        '''
        Get the site index from lattice indices.

        Parameters:
            :lindex: 1d array, lattice index - (n1,n2,...,cindex)

        Return:
            site index, int
        '''
        return c2ind(lindex,N=self.siteconfig)

    def findsite(self,pos,tol=1e-5):
        '''
        Get the lattice indices from the position.

        Parameters:
            :pos: 1d array, the position of atom.
            :tol: float, the tolerence of atom position.

        Return:
            1d array/None, lattice index or None for not found.
        '''
        for iatom in xrange(self.ncatom):
            r=pos-self.catoms[iatom]
            inds=solve(self.a.T,r)
            inds_int=int32(inds.round())
            if abs(inds-inds_int).sum()<tol:
                #bingo!
                inds=append(inds_int,[iatom])
                return inds

    def set_periodic(self,per):
        '''
        Set periodic boundary condition.

        Parameters:
            :per: 1d array of bool, periodicity for each dimension.
        '''
        self.usegroup(TranslationGroup(self.N[:,newaxis]*self.a,per=per))

    def get_cbonds(self,K=30):
        '''
        Get a list of bonds within a unit cell.

        Parameters:
            :K: int, the detection span.
        '''
        dimension=self.dimension
        st=Lattice(a=self.a,N=[5]*dimension,catoms=self.catoms)
        st.set_periodic([True]*dimension)
        cbonds=[]
        stbonds=st.initbonds(K=K)
        for i,bi in enumerate(stbonds):
            bc=bi.query(atom1=arange(self.ncatom))
            bc.atom2s=st.index2l(bc.atom2s)[:,-1]
            cbonds.append(bc)
        return cbonds

    def get_kspace(self):
        '''
        Get the KSpace instance correspond to this lattice.
        '''
        return KSpace(b=toreciprocal(self.a),N=self.N)

if __name__=='__main__':
    c=ChainStructure(catoms=[(0.,0.),(0.5,sqrt(3.)/6)],N1=10,a1=[1.,0])
    c.plotstructure()
    pdb.set_trace()

