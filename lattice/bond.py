#/usr/bin/python
'''
Bond related objects and functions.
'''
from numpy import *
from matplotlib.pyplot import *
from matplotlib.collections import LineCollection
import pdb

__all__=['Bond','BondCollection','show_bonds']

class Bond(object):
    '''A simple Bond class'''
    def __init__(self,bondv,atom1,atom2):
        self.bondv=bondv
        self.atom1=atom1
        self.atom2=atom2

    def getreverse(self):
        '''reverse the bond'''
        return Bond(-self.bondv,self.atom2,self.atom1)

    def __str__(self):
        return self.bondv.__str__()+' '+str(self.atom1)+' '+str(self.atom2)

    def __eq__(self,bond2):
        return (all(self.bondv==bond2.bondv)) & (self.atom1==bond2.atom1) & (self.atom2==bond2.atom2)


class BondCollection(object):
    '''
    A collection of Bond objects, pack it into compact form.

    *Attributes*

    atom1s/atom2s:
        the starting/ending sites of bonds.
    bondvs:
        the bond vectors.
    '''
    def __init__(self,atom1s,atom2s,bondvs):
        self.atom1s=array(atom1s)
        self.atom2s=array(atom2s)
        self.bondvs=array(bondvs)
        self.bonds=[Bond(bv,a1,a2) for bv,a1,a2 in zip(bondvs,atom1s,atom2s)]
        self.__finalized__=True

    def __getitem__(self,index):
        return self.bonds[index]

    def __setattr__(self,name,value):
        finalized=hasattr(self,'__finalized__')
        if not finalized:
            return super(BondCollection,self).__setattr__(name,value)
        super(BondCollection,self).__setattr__(name,value)
        if name!='bonds':
            self.bonds=[Bond(bv,a1,a2) for bv,a1,a2 in zip(self.bondvs,self.atom1s,self.atom2s)]

    def __iter__(self):
        return iter(self.bonds)

    def __str__(self):
        return '<BondCollection(%s)>\n'%self.N+'\n'.join([b.__str__() for b in self.bonds])

    def __len__(self):
        return self.N

    @property
    def N(self):
        '''The number of bonds'''
        return len(self.bonds)

    @property
    def vdim(self):
        '''The dimension of bonds.'''
        return self.bondvs.shape[-1]

    def query(self,atom1=None,atom2=None,bondv=None,condition='and'):
        '''
        Get specific bonds meets given requirements.

        atom1/atom2:
            One or a list of atom1,atom2.
        bondv:
            One or a list of bond vectors.
        condition:

            * `and` -> meets all requirements.
            * `or`  -> meets one of the requirements.
            * `xor` -> xor condition.

        *return*:
            A <BondCollection> instance.
        '''
        tol=1e-5
        masks=[]
        assert(condition in ['or','and','xor'])
        if atom1 is not None:
            if ndim(atom1)==0: atom1=array([atom1])
            masks.append(any(abs(self.atom1s-atom1[:,newaxis])<0.5,axis=0))
        if atom2 is not None:
            if ndim(atom2)==0: atom2=array([atom2])
            masks.append(any(abs(self.atom2s-atom2[:,newaxis])<0.5,axis=0))
        if bondv is not None:
            if ndim(bondv)==1: bondv=array([bondv])
            masks.append(any(norm(self.bondvs-bondv[:,newaxis,:],axis=-1)<tol,axis=0))
        mask=ones(self.N,dtype='bool')
        for mk in masks:
            if condition=='or':
                mask|=mk
            elif condition=='and':
                mask&=mk
            else:
                mask^=mk
        return BondCollection(self.atom1s[mask],self.atom2s[mask],self.bondvs[mask])

def show_bonds(bonds,start=None,lw=1,**kwargs):
    '''
    Display a collection of bonds.

    bonds:
        A <BondCollection> instance.
    start:
        the location of starting atoms.
    lw,**kwargs:
        line width of bonds and key word arguments for 

    *return*:
        None
    '''
    vdim=bonds.vdim
    bvs=[]
    if start is None:
        start=zeros([bonds.N,vdim])
    elif ndim(start)==1:
        bvs=zip(start,bonds.bondvs+start)
    else:
        bvs=zip(start,bonds.bondvs+start)
    if vdim==1:
        bvs=[(append(start,[0]),append(end,[0])) for start,end in bvs]
    lc=LineCollection(bvs,**kwargs)
    lc.set_linewidth(lw)
    ax=gca()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

