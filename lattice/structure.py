#!/usr/bin/python
'''
<Structure> related classes and operations.
'''
from numpy import *
from utils import toreciprocal,ind2c,c2ind
import itertools,warnings
from bond import Bond,show_bonds,BondCollection
from numpy.linalg import norm
from scipy.spatial import cKDTree
from matplotlib.pyplot import *
import time,pdb,pickle,warnings

__all__=['Structure']

FOLDER='data'
class Structure(object):
    '''
    Structure Base class. A collection of sites.

    Construct
    ---------------------
    Structure(sites)

    Attributes
    ----------------------
    sites:
        Positions of sites.
    groups:
        The translation group and point groups imposed on this lattice.
    __kdt__/__kdt_map__:
        The kd-tree for this structure, and the mapping for kdt index and site index.
    '''
    def __init__(self,sites):
        '''
        sites:
            The positions of sites in this structure.
        '''
        self.sites=array(sites)
        self.groups={}
        self.__nnnbonds__=None
        self.__kdt__=None
        self.__kdt_map__=None

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
        return self.__nnnbonds__ is not None

    def query_neighbors(self,i,k,**kwargs):
        '''
        Query the K-nearest sites near specific site.

        Parameters:
            :i: int, the index of target site.
            :k: int, the number of neighbors.
            *kwargs*: Key word arguments for cKDTree.

        Return:
            int32 array of length k, the indices of neighbor sites.
        '''
        if self.__kdt__ is None:
            warnings.warn('Please initialize bonds before querying sites.')
            return None
        pos=self.sites[i]
        k=min(len(self.__kdt__.data),k)
        distance,ind=self.__kdt__.query(pos,k=k,**kwargs)
        site=self.__kdt_map__[ind]
        return distance,site

    def findsite(self,pos,tol=1e-5):
        '''
        Find the site at specific position.

        pos:
            The position of the site.
        tol:
            The position tolerence.
        
        *return*:
            The site index.
        '''
        distance,ind=self.__kdt__.query(pos,k=1,tol=1e-5)
        site=self.__kdt_map__[ind]
        if distance<tol:
            return site

    def usegroup(self,g):
        '''
        Apply a group on this lattice.

        g: 
            A <Group> instance.
        '''
        self.groups[g.tp]=g

    def measure(self,i,j,k=2):
        '''
        Measure the 'true' distance between sites at ri and rj. 
        Here, the main problem is the periodic bondary condition.

        i/j: 
            index of start/end atom.
        k:
            The maximum times of translation group imposed on r2.

        *return*:
            (|r|,r), absolute distance and vector distance.
        '''
        tg=self.groups.get('translation')
        ri,rj=self.sites[i],self.sites[j]
        if tg is not None:
            return tg.measure(ri,rj)
        else:
            r=ri-rj
            return norm(r,axis=-1),r

    def initbonds(self,nmax=3,K=None,tol=1e-5,leafsize=30):
        '''
        Initialize the distance(bond) mesh, and classify it by onsite,1st,2ed,3rd ... nearest neighbours.
        
        nmax:
            Up to nmax-th neighbor will be considered.

            Note: if nmax<0, it will initialize the tree for query but will not initialize bonds.
        K: 
            Number of neighbors calculated through cKD Tree, should be >= number of sites up to nmax-th neighbor.
        tol:
            The bond vector tolerence.
        leafsize:
            The leafsize of kdtree.

        *return*:
            A list of bond vectors, the elements are 0,1,2...-th neightbors.
        '''
        #expand boundary if it is periodic
        print 'Search Nearest Neighbors through Method of cKD Tree.'
        t0=time.time()
        tg=self.groups.get('translation')
        if tg is not None:
            sites=tg.get_equiv(self.sites).reshape([-1,self.vdim])
            self.__kdt_map__=repeat(arange(self.nsite),tg.tmatrix_seq.shape[0])
        else:
            sites=self.sites
            self.__kdt_map__=arange(self.nsite)
        kdt=cKDTree(sites,leafsize=leafsize)
        self.__kdt__=kdt

        if nmax<0: return
        if K is None: K=6*nmax+1
        atom1s=[]
        atom2s=[]
        datas=[]
        bondvs=[]
        K=min(len(kdt.data),K)
        for i,site in enumerate(self.sites):
            distance,ind2=kdt.query(site,k=K)
            atom2=self.__kdt_map__[ind2]
            datas.append(distance)
            atom2s.append(atom2)
            atom1s.append(i*ones(len(ind2),dtype='int32'))
            bondvs.append(kdt.data[ind2]-site)
        datas=concatenate(datas)
        atom1s=concatenate(atom1s)
        atom2s=concatenate(atom2s)
        bondvs=concatenate(bondvs,axis=0)
        count=0
        t1=time.time()
        nnnbonds=[]
        while any(datas!=Inf):
            minval=datas.min()
            dindex=(datas-minval)<tol
            nnnbonds.append(BondCollection(atom1s[dindex],atom2s[dindex],bondvs[dindex]))
            datas[dindex]=Inf
            count+=1
            if count>nmax:
                break
        self.__nnnbonds__=nnnbonds
        t2=time.time()
        print 'Elapse -> %s, %s'%(t1-t0,t2-t1)
        return nnnbonds

    def save_bonds(self,filename=None,nmax=3):
        '''
        Save bonds.

        filename:
            The target filename.
        nmax:
            Up to nmax-th neighnors are saved.
        '''
        if filename is None: filename='nnnbonds_%s.dat'%self.nsite
        bonds=self.__nnnbonds__
        if bonds is None:
            warnings.warn('Bond not Initialized, not saved.')
            return
        nmax=min(len(bonds),nmax)

        atom1s=concatenate([bonds[i].atom1s for i in xrange(nmax)])
        atom2s=concatenate([bonds[i].atom2s for i in xrange(nmax)])
        bondvs=concatenate([bonds[i].bondvs for i in xrange(nmax)])
        nnnfile=FOLDER+'/%s'%filename
        Nptr=append([0],cumsum([bonds[i].N for i in xrange(nmax)]))
        savetxt(nnnfile[:-4]+'.atom.dat',concatenate([atom1s[:,newaxis],atom2s[:,newaxis]],axis=1),fmt='%d')
        savetxt(nnnfile[:-4]+'.bond.dat',bondvs)
        savetxt(nnnfile[:-4]+'.info.dat',Nptr,fmt='%d')

    def load_bonds(self,filename=None):
        '''
        Load bonds.

        filename:
            The target filename.
        '''
        if filename is None: filename='nnnbonds_%s.dat'%self.nsite
        nnnfile=FOLDER+'/%s'%filename
        atom1s,atom2s=loadtxt(nnnfile[:-4]+'.atom.dat',dtype='int32').T
        bondvs=loadtxt(nnnfile[:-4]+'.bond.dat')
        Ns=loadtxt(nnnfile[:-4]+'.info.dat',dtype='int32')
        self.__nnnbonds__=[BondCollection(atom1s[ni:nj],atom2s[ni:nj],bondvs[ni:nj]) for ni,nj in zip(Ns[:-1],Ns[1:])]

    def getbonds(self,n):
        '''
        Get n-th nearest bonds.
        
        n: 
            Specify which set of bonds.

        *return*:
            A <BondCollection> instance.
        '''
        if self.__nnnbonds__ is None:
            warnings.warn('Bond not initialized, please initialize the bonds using @initbonds method.')
            return
        bonds=self.__nnnbonds__
        if n>len(bonds)-1:
            return []
        else:
            return bonds[n]

    def show_bonds(self,nth=(1,),plane=(0,1),color='r',offset=(0,0)):
        '''
        Plot the structure.

        Parameters:
            :plane: tuple, project to the specific plane if it is a 3D structre. Default is (0,1) - 'x-y' plane.
            :nth: int, the n-th nearest bonds are plotted. It should be a tuple. Default is (1,) - the nearest neightbor.
            :color: str, color, default is 'r' -red.
            :offset: len-2 tuple, offset in x,y direction.
        '''
        if ndim(nth)==0:
            nth=[nth]
        nnnbonds=self.__nnnbonds__
        if nnnbonds is None or len(nnnbonds)<=1:
            warnings.warn('Trivial bonds @Structure.show_bond, give up.')
            return
        for nb in nth:
            bs=nnnbonds[nb]
            show_bonds(bs,start=self.sites[bs.atom1s]+offset)

    def show_sites(self,plane=(0,1),color='r',offset=(0,0)):
        '''
        Show the sites in this structure.

        Parameters:
            :color: str, the color.
        '''
        if self.vdim>1:
            p1,p2=plane
            x,y=self.sites[:,p1],self.sites[:,p2]
        else:
            x=self.sites[:,0]
            y=zeros(self.nsite)
        scatter(x+offset[0],y+offset[1],s=50,c=color,edgecolor='none',vmin=-0.5,vmax=1.5)

