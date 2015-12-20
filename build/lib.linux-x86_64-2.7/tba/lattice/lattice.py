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
from bond import Bond,show_bonds,BondCollection
from numpy.linalg import norm,solve
from scipy.spatial import cKDTree
from kspace import KSpace
from structure import Structure
from matplotlib.pyplot import *
import warnings,pdb

__all__=['Lattice']

class Lattice(Structure):
    '''
    Lattice Structure, which contains tranlation of cells.

    Construct
    ----------------
    Lattice(name,a,N,catoms=[(0.,0.)])

    Attributes
    ------------------
    name:
        The name of this latice.
    a:
        Lattice vector
    N: 
        Number of cells
    catoms:
        Atoms in one cell.
    lmesh:
        The sites reshaped according to the lattice config (Nx,Ny, ..., ncatom).
    '''
    def __init__(self,name,a,N,catoms=[(0.,0.)]):
        self.name=name
        self.catoms=array(catoms)
        self.a=array(a)
        self.N=array(N)
        
        mr=meshgrid_v([arange(n) for n in self.N],vecs=a)
        self.lmesh=(mr[...,newaxis,:]+self.catoms)

        vdim=self.a.shape[-1]
        items=self.lmesh.reshape([-1,vdim])
        super(Lattice,self).__init__(items)

    def __str__(self):
        return '''<Lattice %s>
    Base           ->  %s,
    Lattice Size   ->  %s,
    Cell Size      ->  %s.
'''%(self.name,', '.join([str(tuple(a)) for a in self.a]),' x '.join([str(n) for n in self.N]),self.ncatom)

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

    @property
    def cbonds(self):
        '''
        Get a list of bonds within a unit cell.
        '''
        dimension=self.dimension
        st=Lattice(name='anonymous',a=self.a,N=[5]*dimension,catoms=self.catoms)
        st.usegroup(TranslationGroup(st.N[:,newaxis]*st.a,per=ones(st.dimension,dtype='bool')))
        cbonds=[]
        stbonds=st.initbonds(K=30)
        for i,bi in enumerate(stbonds):
            bc=bi.query(atom1=arange(self.ncatom))
            bc.atom2s=st.index2l(bc.atom2s)[:,-1]
            cbonds.append(bc)
        return cbonds

    @property
    def kspace(self):
        '''
        Get the KSpace instance correspond to this lattice.
        '''
        return KSpace(b=toreciprocal(self.a),N=self.N)

    def showcell(self,bondindex=(1,2),plane=(0,1),color='r',offset=None):
        '''
        Plot the cell structure.

        bondindex:
            the bondindex-th nearest bonds are plotted. It should be a tuple.
            Default is (1,2) - the nearest, and second nearest neightbors.
        plane:
            project to the specific plane if it is a 3D structre.
            Default is (0,1) - 'x-y' plane.
        color:
            color, default is 'r' -red.
        offset:
            The offset of the sample cell.
        '''
        p1,p2=plane
        nnnbonds=self.cbonds
        if nnnbonds is None or len(nnnbonds)<=1:
            warnings.warn('Trivial bondindex@Structure.plotstructure, plot on-site terms only.')
            bondindex=()
        if offset is None:
            offset=[2]*self.dimension
        for nb in bondindex:
            cb=nnnbonds[nb]
            cellindex=self.l2index(append(offset,[0]))
            show_bonds(cb,start=self.sites[cellindex+cb.atom1s])
        #plot sites
        if self.vdim>1:
            p1,p2=plane
            x,y=self.sites[:,p1],self.sites[:,p2]
        else:
            x=self.sites[:,0]
            y=zeros(self.N)
        scatter(x,y,s=50,c=color,edgecolor='none',vmin=-0.5,vmax=1.5)

    def index2l(self,index):
        '''
        Get lattice indices (n1,n2,...,atom index in cell) from site index.

        index:
            the site index.
        '''
        return ind2c(index,N=self.siteconfig)

    def l2index(self,lindex):
        '''
        Get the site index from lattice indices.
        lindex:
            lattice index - (n1,n2,...,atom index in cell)
        '''
        return c2ind(lindex,N=self.siteconfig)

    def findsite(self,pos,tol=1e-5):
        '''
        Get the lattice indices from the position.

        pos: 
            the position r.
        tol:
            the tolerence of atom position.

        *return*:
            an array of lattice index.
        '''
        for iatom in xrange(self.ncatom):
            r=pos-self.catoms[iatom]
            inds=solve(self.a.T,r)
            inds_int=int32(inds.round())
            if abs(inds-inds_int).sum()<tol:
                #bingo!
                inds=append(inds,[iatom])
                return inds
        print 'Can not find this site!'


if __name__=='__main__':
    c=ChainStructure(catoms=[(0.,0.),(0.5,sqrt(3.)/6)],N1=10,a1=[1.,0])
    c.plotstructure()
    pdb.set_trace()

