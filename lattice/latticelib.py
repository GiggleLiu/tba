#!/usr/bin/python
#-*-coding:utf-8-*-
#By Giggle Liu

from numpy import *
from lattice import Lattice
from bzone import BZone
from group import C6vGroup,C4vGroup,C3vGroup

__all__=['Honeycomb_Lattice','Square_Lattice','Triangular_Lattice','Chain','construct_lattice','resize_lattice']

class Honeycomb_Lattice(Lattice):
    '''
    HoneyComb Lattice class.

    Construct
    ----------------
    Honeycomb_Lattice(N,form=1.)

    form:
        The form of lattice.
        `1` -> traditional one with 0 point at a vertex, using C3v group.
        `2` -> the C6v form with 0 point at the center of hexagon, using C6v group.
    '''
    def __init__(self,N,form=1):
        if form==1:
            catoms=[(0.,0.),(0.5,sqrt(3.)/6)]
            pg=C3vGroup()
        elif form==2:
            catoms=array[(0.,1./sqrt(3.)),(0.5,sqrt(3.)/6)]
            pg=C6vGroup()
        else:
            raise ValueError('Form %s not defined.'%form)
        super(Honeycomb_Lattice,self).__init__(name='honeycomb',a=array([(1.,0),(0.5,sqrt(3.)/2)]),N=N,catoms=catoms)
        self.usegroup(pg)

    @property
    def kspace(self):
        '''
        Get the <KSpace> instance.
        '''
        ks=super(Honeycomb_Lattice,self).kspace

        M0=ks.b[1]/2.0
        K0=(ks.b[0]+2*ks.b[1])/3.0
        c6vg=C6vGroup()

        M=[]
        K=[]
        for i in xrange(6):
            M.append(c6vg.actonK(M0,i))
            K.append(c6vg.actonK(K0,i))
        ks.special_points['M']=M
        ks.special_points['K']=K
        ks.usegroup(c6vg)
        return ks

class Square_Lattice(Lattice):
    '''
    Square Lattice, using C4v Group.

    Construct
    ----------------
    Square_Lattice(N,catoms=[(0.,0.)])
    '''
    def __init__(self,N,catoms=[(0.,0.)]):
        a=array([(1.,0),(0.,1.)])
        super(Square_Lattice,self).__init__(N=N,a=a,catoms=catoms,name='square')
        c4vg=C4vGroup()
        self.usegroup(c4vg)

    @property
    def kspace(self):
        '''
        Get the <KSpace> instance.
        '''
        ks=super(Square_Lattice,self).kspace
        M0=ks.b[1]/2.0
        K0=(ks.b[0]+ks.b[1])/2.0
        c4vg=C4vGroup()

        M=[]
        K=[]
        for i in xrange(4):
            M.append(c4vg.actonK(M0,i))
            K.append(c4vg.actonK(K0,i))
        ks.special_points['M']=M
        ks.special_points['K']=K
        ks.usegroup(c4vg)
        return ks

class Triangular_Lattice(Lattice):
    '''
    Triangular Lattice, using C6v Group.

    Construct
    ----------------
    Triangular_Lattice(N,catoms=[(0.,0.)])
    '''
    def __init__(self,N,catoms=[(0.,0.)]):
        '''Basic information of Triangular Lattice'''
        a=array([(1.,0),(0.5,sqrt(3.)/2)])
        super(Triangular_Lattice,self).__init__(a=a,catoms=catoms,name='triangular',N=N)
        c6vg=C6vGroup()
        self.usegroup(c6vg)

    @property
    def kspace(self):
        '''
        Get the <KSpace> instance.
        '''
        ks=super(Triangular_Lattice,self).kspace
        M0=ks.b[1]/2.0
        K0=(ks.b[0]+2*ks.b[1])/3.0
        c6vg=C6vGroup()

        M=[]
        K=[]
        for i in xrange(6):
            M.append(c6vg.actonK(M0,i))
            K.append(c6vg.actonK(K0,i))
        ks.special_points['M']=M
        ks.special_points['K']=K
        ks.usegroup(c6vg)
        return ks



class Chain(Lattice):
    '''
    Lattice of Chain.

    Construct
    ----------------
    Chain(N,a=(1.),catoms=[(0.,0.)])
    '''
    def __init__(self,N,a=(1.),catoms=[(0.)]):
        '''
        N:
            Number of cells, integer.
        a:
            Lattice vector, 1D array.
        catoms:
            Atom positions in a unit cell.
        '''
        super(Chain,self).__init__(a=[a],N=[N],name='chain',catoms=catoms)

    @property
    def kspace(self):
        '''The <KSpace> instance correspond to a chain.'''
        a=self.a[0]
        b=2*pi*a/a.dot(a)
        ks=KSpace(N=self.N,b=b)
        ks.special_points['K']=array([-b/2.,b/2.])
        return ks


def construct_lattice(N,lattice_shape='',a=None,catoms=None,args={}):
    '''
    Uniform construct method for lattice.

    N:
        The size of lattice.
    lattice_shape:
        The shape of lattice.

        * ''            -> the anonymous lattice.
        * 'square'      -> square lattice.
        * 'honeycomb'   -> honeycomb lattice.
        * 'triangular'  -> triangular lattice.
        * 'chain'       -> a chain.
    a:
        The unit vector.
    catoms:
        The atoms in a unit cell.
    args:
        Other arguments,

        * `form` -> the form used in constructing honeycomb lattice.
    '''
    if lattice_shape=='':
        assert(a is not None)
        if catoms is None: catoms=zeros(shape(a)[-1])
        return Lattice(name='anonymous',N=N,a=a,catoms=catoms)
    elif lattice_shape=='honeycomb':
        return Honeycomb_Lattice(N=N,form=args.get('form',1))
    elif lattice_shape=='square':
        if catoms is None: catoms=zeros([1,2])
        return Square_Lattice(N=N,catoms=catoms)
    elif lattice_shape=='triangular':
        if catoms is None: catoms=zeros([1,2])
        return Triangular_Lattice(N=N,catoms=catoms)
    elif lattice_shape=='chain':
        if a is None: a=[1.]
        if catoms is None: catoms=zeros([1,1])
        if ndim(N)==1:
            N=N[0]
        return Chain(N=N,catoms=catoms)


def resize_lattice(lattice,N):
    '''
    Resize the lattice to specific size.

    lattice:
        The target lattice.
    N:
        1D - array, the size of new lattice.
    '''
    return construct_lattice(a=lattice.a,N=N,catoms=lattice.catoms,args={'form':getattr(lattice,'form',None)})
