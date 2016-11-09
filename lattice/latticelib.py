#!/usr/bin/python
#-*-coding:utf-8-*-
#By Giggle Liu

from numpy import *
from lattice import Lattice
from kspace import KSpace
from bzone import BZone
from group import PointGroup

__all__=['Honeycomb_Lattice','Square_Lattice','Triangular_Lattice','Chain','construct_lattice','resize_lattice','plot_lattice']

class Honeycomb_Lattice(Lattice):
    '''
    HoneyComb Lattice class.

    Attributes:
        :form: 1/2, The form of lattice.

            * `1` -> traditional one with 0 point at a vertex, using C3v group.
            * `2` -> the C6v form with 0 point at the center of hexagon, using C6v group.
    '''
    def __init__(self,N,form=1):
        self.form=form
        if form==1:
            catoms=[(0.,0.),(0.5,sqrt(3.)/6)]
            pg=PointGroup('C3v')
        elif form==2:
            catoms=array[(0.,1./sqrt(3.)),(0.5,sqrt(3.)/6)]
            pg=PointGroup('C6v')
        else:
            raise ValueError('Form %s not defined.'%form)
        super(Honeycomb_Lattice,self).__init__(a=array([(1.,0),(0.5,sqrt(3.)/2)]),N=N,catoms=catoms)
        self.usegroup(pg)

    def get_kspace(self):
        '''
        Get the <KSpace> instance.
        '''
        ks=super(Honeycomb_Lattice,self).get_kspace()

        M0=ks.b[1]/2.0
        K0=(ks.b[0]+2*ks.b[1])/3.0
        c6vg=PointGroup('C6v')

        M=[]
        K=[]
        for i in xrange(6):
            M.append(c6vg.acton_vec(M0,i))
            K.append(c6vg.acton_vec(K0,i))
        ks.special_points['M']=M
        ks.special_points['K']=K
        ks.usegroup(c6vg)
        return ks

class Square_Lattice(Lattice):
    '''
    Square Lattice, using C4v Group.
    '''
    def __init__(self,N,catoms=[(0.,0.)]):
        a=array([(1.,0),(0.,1.)])
        super(Square_Lattice,self).__init__(N=N,a=a,catoms=catoms)
        c4vg=PointGroup('C4v')
        self.usegroup(c4vg)

    def get_kspace(self):
        '''
        Get the <KSpace> instance.
        '''
        ks=super(Square_Lattice,self).get_kspace()
        M0=ks.b[1]/2.0
        K0=(ks.b[0]+ks.b[1])/2.0
        c4vg=PointGroup('C4v')

        M=[]
        K=[]
        for i in xrange(4):
            M.append(c4vg.acton_vec(M0,i))
            K.append(c4vg.acton_vec(K0,i))
        ks.special_points['M']=M
        ks.special_points['K']=K
        ks.usegroup(c4vg)
        return ks

class Triangular_Lattice(Lattice):
    '''
    Triangular Lattice, using C6v Group.
    '''
    def __init__(self,N,catoms=[(0.,0.)]):
        '''Basic information of Triangular Lattice'''
        a=array([(1.,0),(0.5,sqrt(3.)/2)])
        super(Triangular_Lattice,self).__init__(a=a,catoms=catoms,N=N)
        c6vg=PointGroup('C6v')
        self.usegroup(c6vg)

    def get_kspace(self):
        '''
        Get the <KSpace> instance.
        '''
        ks=super(Triangular_Lattice,self).get_kspace()
        M0=ks.b[1]/2.0
        K0=(ks.b[0]+2*ks.b[1])/3.0
        c6vg=PointGroup('C6v')

        M=[]
        K=[]
        for i in xrange(6):
            M.append(c6vg.acton_vec(M0,i))
            K.append(c6vg.acton_vec(K0,i))
        ks.special_points['M']=M
        ks.special_points['K']=K
        ks.usegroup(c6vg)
        return ks

class Chain(Lattice):
    '''
    Lattice of Chain.
    '''
    def __init__(self,N,a=array([1.]),catoms=[(0.)]):
        super(Chain,self).__init__(a=[a],N=[N],catoms=catoms)

    def get_kspace(self):
        '''The <KSpace> instance correspond to a chain.'''
        a=self.a[0]
        b=2*pi*a/dot(a,a)
        ks=KSpace(N=self.N,b=[b])
        ks.special_points['K']=array([-b/2.,b/2.])
        return ks


def construct_lattice(N,lattice_type='',a=None,catoms=None,args={}):
    '''
    Uniform construct method for lattice.

    Parameter:
        :N: int, the size of lattice.
        :lattice_type: str, the type of lattice.

            * ''            -> the anonymous lattice.
            * 'square'      -> square lattice.
            * 'honeycomb'   -> honeycomb lattice.
            * 'triangular'  -> triangular lattice.
            * 'chain'       -> a chain.
        :a: 2d array, unit vectors.
        :catoms: 2d array, atoms in a unit cell.
        :args: dict, other arguments,

            * `form` -> the form used in constructing honeycomb lattice.
    '''
    if lattice_type=='':
        assert(a is not None)
        if catoms is None: catoms=zeros([1,shape(a)[-1]])
        return Lattice(N=N,a=a,catoms=catoms)
    elif lattice_type=='honeycomb':
        return Honeycomb_Lattice(N=N,form=args.get('form',1))
    elif lattice_type=='square':
        if catoms is None: catoms=zeros([1,2])
        return Square_Lattice(N=N,catoms=catoms)
    elif lattice_type=='triangular':
        if catoms is None: catoms=zeros([1,2])
        return Triangular_Lattice(N=N,catoms=catoms)
    elif lattice_type=='chain':
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

def plot_lattice(lattice,bondcolor='k',sitecolor='b',offset=(0,0)):
    '''
    Plot the lattice.
    '''
    lattice.show_sites(plane=(0,1),color=sitecolor,offset=offset)
    lattice.show_bonds(nth=(1,),plane=(0,1),color=bondcolor,offset=offset)
