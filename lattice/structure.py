'''
<Structure> class.
'''

from numpy import *
from utils import toreciprocal,ind2c,c2ind
import itertools,warnings
from numpy.linalg import norm
from scipy.spatial import cKDTree
import time,pdb,pickle,warnings

from bond import Bond,BondCollection

__all__=['NotInitializedError','Structure']

class NotInitializedError(Exception):
    pass

class Structure(object):
    '''
    Structure Base class. A collection of sites.

    Attributes:
        :sites: 2d array, positions of sites.
        :groups: dict, the translation group and point groups imposed on this lattice.

    Read Only Attributes:
        :nsite: int, the number of sites.
        :vdim: int, dimension vector space.
        :b1s, b2s, b3s: <BondCollection>, first,second,third nearest neighbors.
        :bonds_initialized: bool, True if the bonds are initialized.

    Private Attributes:
        :_kdt/_kdt_map: KDTree/dict, the kd-tree for this structure, and the mapping for kdt index and site index.
        :_nnnbonds: list, the bonds grouped by neighboring ranks.
    '''
    def __init__(self,sites):
        '''
        sites:
            The positions of sites in this structure.
        '''
        self.sites=array(sites)
        self.groups={}
        self._nnnbonds=None
        self._kdt=None
        self._kdt_map=None

    @property
    def nsite(self):
        '''Number of sites'''
        return len(self.sites)

    @property
    def vdim(self):
        '''Dimention of vector space.'''
        return self.sites.shape[1]

    @property
    def b1s(self):
        '''The nearest neighbor bonds.'''
        return self.getbonds(1)

    @property
    def b2s(self):
        '''The nearest neighbor bonds.'''
        return self.getbonds(2)

    @property
    def b3s(self):
        '''The nearest neighbor bonds.'''
        return self.getbonds(3)

    @property
    def bonds_initialized(self):
        '''Whether bonds are initialized.'''
        return self._nnnbonds is not None

    def get_neighbors(self,i,k,**kwargs):
        '''
        Get the k-nearest sites near specific site.

        Parameters:
            :i: int, the index of target site.
            :k: int, the number of neighbors.
            *kwargs*: Key word arguments for cKDTree.

        Return:
            (site, distance), the indices of neighbor sites, int32 array of length k, 
        '''
        if not self.bonds_initialized: raise NotInitializedError()
        pos=self.sites[i]
        k=min(len(self._kdt.data),k+1)
        distance,ind=self._kdt.query(pos,k=k,**kwargs)
        sites=self._kdt_map[ind[1:]]
        return sites,distance[1:]

    def findsite(self,pos,tol=1e-5):
        '''
        Find the site at specific position.

        Parameters:
            :pos: 1d array, the position of the site.
            :tol: float, the position tolerence.
        
        Return:
            int/None, the site index, or None if not found.
        '''
        if not self.bonds_initialized: raise NotInitializedError()
        distance,ind=self._kdt.query(pos,k=1)
        site=self._kdt_map[ind]
        if distance<tol:
            return site

    def usegroup(self,g):
        '''
        Apply group `g` on this lattice.
        '''
        self.groups[g.tp]=g

    def get_distance(self,i,j,k=2):
        '''
        Measure the 'true' distance between sites at ri and rj. 
        Here, the main problem is the periodic bondary condition.

        Parameters:
            :i/j: int, index of start/end atom.
            :k: int, the maximum times of translation group imposed on r2.

        Return:
            (|r|,r), absolute distance and vector distance.
        '''
        tg=self.groups.get('translation')
        ri,rj=self.sites[i],self.sites[j]
        if tg is not None:
            return tg.measure(ri,rj)
        else:
            r=rj-ri
            return norm(r,axis=-1),r

    def initbonds(self,K=20,tol=1e-5,leafsize=30,nth_neighbor=Inf):
        '''
        Initialize the distance(bond) mesh, and classify it by onsite,1st,2ed,3rd ... nearest neighbours.
        
        Parameters:
            :K: int, number of neighbors to be detected(>= number of sites up to nmax-th neighbor).
            :tol: float, the bond vector tolerence in grouping.
            :leafsize: int, the leafsize of kdtree.
            :nmax: int, the number of neighbors considered,
            :nth_neighbor: int, up to nth neighbor are considered.

        Return:
            <BondCollection>, bond vectors, the elements are 0,1,2...-th neightbor bonds.
        '''
        #expand boundary if it is periodic
        print 'Search Nearest Neighbors through Method of cKD Tree.'
        t0=time.time()
        tg=self.groups.get('translation')
        if tg is not None:
            sites=tg.get_equiv(self.sites).reshape([-1,self.vdim])
            self._kdt_map=repeat(arange(self.nsite),tg.tmatrix_seq.shape[0])
        else:
            sites=self.sites
            self._kdt_map=arange(self.nsite)
        kdt=cKDTree(sites,leafsize=leafsize)
        self._kdt=kdt

        #calculate the mesh of neighbor informations.
        atom1s=[] #start atom
        atom2s=[] #end atom
        datas=[]  #the distance.
        bondvs=[] #vector
        K=min(len(kdt.data),K)
        for i,site in enumerate(self.sites):
            distance,ind2=kdt.query(site,k=K)
            atom2=self._kdt_map[ind2]
            datas.append(distance)
            atom2s.append(atom2)
            atom1s.append(i*ones(len(ind2),dtype='int32'))
            bondvs.append(kdt.data[ind2]-site)
        datas=concatenate(datas)
        atom1s=concatenate(atom1s)
        atom2s=concatenate(atom2s)
        bondvs=concatenate(bondvs,axis=0)

        #group by distances.
        count=0
        t1=time.time()
        nnnbonds=[]
        while any(datas!=Inf):
            minval=datas.min()
            dindex=(datas-minval)<tol
            nnnbonds.append(BondCollection((atom1s[dindex],atom2s[dindex],bondvs[dindex])))
            datas[dindex]=Inf
            count+=1
            if count==nth_neighbor:
                break
        self._nnnbonds=nnnbonds
        t2=time.time()
        print 'Elapse -> %s, %s'%(t1-t0,t2-t1)
        return nnnbonds

    def getbonds(self,n):
        '''
        Get n-th nearest bonds.
        
        n: 
            Specify which set of bonds.

        *return*:
            A <BondCollection> instance.
        '''
        if not self.bonds_initialized: raise NotInitializedError()
        bonds=self._nnnbonds
        if n>len(bonds)-1:
            return BondCollection([])
        else:
            return bonds[n]
