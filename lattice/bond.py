'''
Bond related objects and functions.
'''

from numpy import *
from numpy.linalg import norm
import pdb
import cPickle as pickle

__all__=['Bond','BondCollection','load_bonds']

class Bond(object):
    '''A simple Bond class'''
    def __init__(self,atom1,atom2,bondv):
        self.atom1=atom1
        self.atom2=atom2
        self.bondv=asarray(bondv)

    def __str__(self):
        return self.bondv.__str__()+' '+str(self.atom1)+' '+str(self.atom2)

    def __eq__(self,bond2):
        return (all(self.bondv==bond2.bondv)) & (self.atom1==bond2.atom1) & (self.atom2==bond2.atom2)

    def __neg__(self):
        return Bond(self.atom2,self.atom1,-self.bondv)


class BondCollection(object):
    '''
    A collection of Bond objects, pack it into compact form.

    Construction:
        :BondCollection(bonds=[bond1,bond2,...]): a list <Bond> instances.
        :BondCollection(bonds=(atom1s,atom2s,bondvs)): len-3 tuple of starting/ending sites and bond vectors.

    Attributes:
        :atom1s/atom2s: 1d array, the starting/ending sites of bonds.
        :bondvs: 2d array, the bond vectors.
    '''
    def __init__(self,bonds):
        if isinstance(bonds,tuple):
            atom1s,atom2s,bondvs=bonds
            self.atom1s=asarray(atom1s)
            self.atom2s=asarray(atom2s)
            self.bondvs=asarray(bondvs)
        elif isinstance(bonds,list):
            self.atom1s=array([b.atom1 for b in bonds])
            self.atom2s=array([b.atom2 for b in bonds])
            self.bondvs=array([b.bondv for b in bonds])
        elif isinstance(bonds,BondCollection):
            self.atom1s=bonds.atom1s
            self.atom2s=bonds.atom2s
            self.bondvs=bonds.bondvs
        else:
            raise ValueError()

    def __getitem__(self,index):
        if isinstance(index,int):
            return Bond(self.atom1s[index],self.atom2s[index],self.bondvs[index])
        elif isinstance(index,slice) or ndim(index)==1:
            return BondCollection((self.atom1s[index],self.atom2s[index],self.bondvs[index]))
        else:
            raise KeyError()

    def __iter__(self):
        for item in zip(self.atom1s,self.atom2s,self.bondvs):
            yield Bond(*item)

    def __str__(self):
        return '<BondCollection(%s)>\n'%self.N+'\n'.join([b.__str__() for b in self])

    def __eq__(self,bc2):
        if isinstance(bc2,list): bc2=BondCollection(bc2)
        if self.N!=bc2.N: return False
        #around is used to avoid rounding errors.
        pm1=lexsort(list(around(self.bondvs,decimals=3).T)+[self.atom2s,self.atom1s])
        pm2=lexsort(list(around(bc2.bondvs,decimals=3).T)+[bc2.atom2s,bc2.atom1s])
        return allclose(self.atom1s[pm1],bc2.atom1s[pm2])\
                and allclose(self.atom2s[pm1],bc2.atom2s[pm2])\
                and allclose(self.bondvs[pm1],bc2.bondvs[pm2],atol=1e-5)

    def __len__(self):
        return self.N

    def __add__(self,target):
        if isinstance(target,BondCollection):
            return BondCollection((append(self.atom1s,target.atom1s),append(self.atom2s,target.atom2s),concatenate([self.bondvs,target.bondvs],axis=0)))
        elif isinstance(target,list):
            return self+BondCollection(target)
        else:
            raise ValueError()

    def __radd__(self,target):
        if isinstance(target,list):
            return BondCollection(target).__add__(self)
        elif isinstance(target,BondCollection):
            return target+self
        else:
            raise ValueError()

    @property
    def N(self):
        '''The number of bonds'''
        return len(self.atom1s)

    @property
    def vdim(self):
        '''The dimension of bonds.'''
        return self.bondvs.shape[-1]

    def query(self,atom1=None,atom2=None,bondv=None,condition='and'):
        '''
        Get specific bonds meets given requirements.

        Parameters:
            :atom1/atom2: int/list, atom1,atom2.
            :bondv: 1d array/list, bond vectors.
            :condition:

                * `and` -> meets all requirements.
                * `or`  -> meets one of the requirements.
                * `xor` -> xor condition.

        Return:
            <BondCollection>.
        '''
        tol=1e-5
        masks=[]
        assert(condition in ['or','and','xor'])
        if atom1 is not None:
            if ndim(atom1)==0: atom1=[atom1]
            masks.append(any(abs(self.atom1s-asarray(atom1)[:,newaxis])<0.5,axis=0))
        if atom2 is not None:
            if ndim(atom2)==0: atom2=[atom2]
            masks.append(any(abs(self.atom2s-asarray(atom2)[:,newaxis])<0.5,axis=0))
        if bondv is not None:
            if ndim(bondv)==1: bondv=[bondv]
            masks.append(any(norm(self.bondvs-asarray(bondv)[:,newaxis,:],axis=-1)<tol,axis=0))
        mask=reduce(lambda x,y:getattr(x,'__%s__'%condition)(y),masks)
        return BondCollection((self.atom1s[mask],self.atom2s[mask],self.bondvs[mask]))

    def save(self,filename):
        '''
        Save bonds.

        Parameters:
            :filename: str, the target filename.
        '''
        data=concatenate([self.atom1s[:,newaxis],self.atom2s[:,newaxis],self.bondvs],axis=1)
        savetxt(filename,data)

def load_bonds(filename):
    '''
    Load bonds.

    Parameters:
        :filename: str, the target filename.
    '''
    data=loadtxt(filename)
    return BondCollection((data[:,0].astype('int32'),data[:,1].astype('int32'),data[:,2:]))

