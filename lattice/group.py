#!/usr/bin/python
'''
Group Operations for Lattice/KSpace and Hamiltonian.
'''
from numpy import *
from utils import meshgrid_v
from numpy.linalg import norm
import re,pdb

__all__=['Group','TranslationGroup','PointGroup','vector_rotate','vector_image','pauli_rotate_matrix']

pgroup_types=[r'^C(\d)$',r'^C(\d)v$']
xyz_axes=dict(x=[1.,0,0],y=[0,1.,0],z=[0,0,1.])

def vector_rotate(v,theta):
    '''
    Rotate vector(s) anti-clock wise for theta.

    Parameters:
        :v: ndarray, len-2 vectors.
        :theta: float, the angle.

    Return:
        ndarray, rotated vectors.
    '''
    vector=asarray(v)
    assert(vector.shape[-1]==2)
    rotatematrix=array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]],dtype='float64')
    return (rotatematrix*vector[...,newaxis,:]).sum(axis=-1)


def vector_image(v,axis):
    '''
    Reflect vector(s) with respect to specific plane.

    Parameters:
        :v: ndarray, vectors.
        :axis: float/str, the axis perpendicular to the image plane.

    Return:
        ndarray, reflected vectors.
    '''
    if isinstance(axis,str): axis=xyz_axes[axis]
    axis,v=asarray(axis),asarray(v)
    axis=axis/norm(axis)
    return v-2*v.dot(axis[:,newaxis])*axis

def pauli_rotate_matrix(theta,axis):
    '''
    Pauli rotation matrix exp(-j*theta/2*sigma(n))

    Parameters:
        :theta: float, the angle.
        :axis: 1d array,str, the pivot axis.
    '''
    if isinstance(axis,str): axis=xyz_axes(axis)
    sigman=vec2s(axis,unit=True)
    return identity(2)*cos(theta/2)-1j*sin(theta/2)*sigman

def spin_rotate(spaceconfig,A,theta,axis):
    '''
    Rotate spin for theta along a specific axis.

    Parameters:
        :spaceconfig: <SpaceConfig>, the configuration of hamiltonian space.
        :A: matrix,
        :theta: float, the angle to rotate.
        :axis: 1d array/str, the pivoting axis.

    Return:
        matrix, the rotated spin.
    '''
    dim2=spaceconfig.norbit*spaceconfig.natom
    U0=kron(pauli_rotate_matrix(theta=theta,axis=axis),identity(dim2))
    if spaceconfig.nnambu==1:
        U=U0
    else:
        nflv=2*dim2
        U=zeros([nflv*2,nflv*2],dtype='complex128')
        U[:nflv,:nflv]=U0
        U[nflv:,nflv:]=conj(U0)
    return dot(conj(transpose(U)),dot(A,U))

def spin_image(spaceconfig,A,axis):
    '''
    Image spin about plane n(indicated by the perpendicular vector).

    Parameters:
        :spaceconfig: <SpaceConfig>, the configuration of hamiltonian space.
        :A: matrix,
        :axis: 1d array/str, the imagine axis.

    Return:
        matrix, the rotated spin.
    '''
    return spin_rotate(spaceconfig,A=A,axis=axis,theta=pi)

class Group(object):
    '''
    The Base class for groups.

    Attributes:
        :tp: str, the type of group.

            * 'point': the point group.
            * 'tranlation': the translation group.
            * 'tr': time reversal group.
    '''
    def __init__(self,tp):
        self.tp=tp

class TranslationGroup(Group):
    '''
    Translation group(with type `translation`).

    Attributes:
        :Rs: 2d array, A list of R, the periodic vectors in real space.
        :per: 1d array, A list of boolean indicating periodic(True) or not(False) in specific direction.
            Or, equivalently, an array of integer indicating the index of periodic axes.
        :k: int, the order of this translation group, the maximum translation operation times,
            the larger, the slower, the more reliable result will be got.
    '''
    def __init__(self,Rs,per,k=1):
        super(TranslationGroup,self).__init__('translation')
        self.Rs=asarray(Rs)
        self.per=zeros(len(Rs),dtype='bool')
        self.per[asarray(per)]=True
        self.k=k

        #initialize the translation matrix.
        self.tmatrix=meshgrid_v([arange(-k,k+1)]*sum(self.per),vecs=self.Rs[self.per])

    def __setattr__(self,name,value):
        super(TranslationGroup,self).__setattr__(name,value)
        if name=='per' or name=='k' or name=='Rs':
            if hasattr(self,'tmatrix'):
                self.tmatrix=meshgrid_v([arange(-self.k,self.k+1)]*sum(self.per),vecs=self.Rs[self.per])

    @property
    def nper(self):
        '''The number of periodic axes.'''
        return self.per.sum()

    @property
    def vdim(self):
        '''The vector dimension.'''
        return self.Rs.shape[1]

    @property
    def tmatrix_seq(self):
        '''translation matrix in sequential form.'''
        return reshape(self.tmatrix,[-1,self.vdim])

    def is_equiv(self,r1,r2,tol=1e-5):
        '''
        Decide whether two points are equivalent under group actions.

        r1/r2:
            1D array, the real space positions.
        tol:
            The distance tolerence.

        *return*:
            True if equivalent else False.
        '''
        for R in self.Rs[self.per]:
            dr1=(r1-r2)%R
            if abs(norm(dr))<tol or abs(norm(dr-R))<tol:
                return True
        return False

    def get_equiv(self,r):
        '''
        Get the equivalent points of specific position under group action.

        r:
            The position vector.
        '''
        if ndim(r)==1:
            return r+self.tmatrix_seq
        else:
            return r[:,newaxis]+self.tmatrix_seq

    def measure(self,ri,rj,k=2):
        '''
        Measure the distance between ri and rj.

        k:
            The maximum times of translation group imposed on r2.
        '''
        rjs=self.get_equiv(rj)
        dr=rjs-ri
        absdr=norm(dr,axis=-1)
        ind=argmin(absdr)
        return absdr[ind],dr[ind]

class PointGroup(Group):
    '''
    Point group class.

    Attributes:
        :name: str, name of group, like C6v, C3v ...
        :irzone: 2d array/None, 2 vector defining the irreducible zone.
        :ng: int, number of group operations.

    Private Attributes:
        :_type: int, the index of type in pgroup_types list.

            * 0: Cn
            * 1: Cnv
    '''
    def __init__(self,name):
        super(PointGroup,self).__init__('point')
        self.name=name
        self.irzone=None
        for i,tpstr in enumerate(pgroup_types):
            mch=re.match(tpstr,self.name)
            if mch is not None:
                self._type=i
                self.ng=int(mch.group(1))*(1 if i==0 else 2)
                break
            if i==len(pgroup_types)-1:
                raise ValueError('No Such Group!')

    def is_reducible(self,k):
        '''
        Test whether k can be reduced to region-G-K-M instead of in the region-G-K-M

        k: 
            the momentum.
        '''
        am,ak=solve(self.irzone,k)
        if am>=-1e-12 and ak>=-1e-12 and (ak+am)<=1+1e-12:
            return False
        else:
            return True

    def reduce(self,k):
        '''
        Reduce k to irreducible zone.

        k:
            Convert k to reducible zone. k should be in brillouin zone!
        '''
        for ig in xrange(self.ng):
            ki=self.acton_vec(k,ig)
            if not self.is_reducible(ki):
                return ki
        print 'warning: irreducible k=',k,'. k not in bzone?'
        return k

    def acton_vec(self,k,ig=None):
        '''
        Get the vector after group action specified by ig

        Parameters:
            :k: ndarray, the input k-vectors.
            :ig: int, specify the group element to act on k.
        '''
        if ig is None: ig=range(self.ng)
        if ndim(ig)==1:
            return array([self.acton_vec(k,iig) for iig in ig])
        if ig>=self.ng: raise ValueError()
        #get the transformation to k: G^-1(ig)*k
        n=self.ng if self._type==0 else self.ng/2
        k1=vector_rotate(k,ig*pi*2/n)
        if ig>=n:
            #image about the y axis
            k1=vector_image(k1,axis=array([1.0,0.0]))
        k1=vector_rotate(k,ig*pi*2/n)
        return k1

    def acton_H(self,spaceconfig,Hmat,ig):
        '''
        Act the group operation on H(spin action and orbit action).
        Please, pay attention to the order of operation here.
        rotate the wave function for angle theta, and then image about yz plane is equivalent to:
        image the Hamiltonian first, then rotate the spin operator for an angle -theta!

        Parameters:
            :spaceconfig: <SpaceConfig>, the configuration of hamiltonian space.
            :Hmat: matrix, the Hamiltonian to perform this action.
            :ig: int, specify the group element.

        Return:
            matrix,
        '''
        if ig is None:
            return array([self.acton_H(spaceconfig,Hmat,ig) for ig in xrange(self.ng)])
        if ig>=self.ng: raise ValueError()
        n=self.ng if self._type==0 else self.ng/2
        if ig >= self.ng/2:
            Hmat=spin_image(spaceconfig,Hmat,axis='x')
        theta=ig*pi*2/n
        Hmat=spin_rotate(spaceconfig,Hmat,theta,axis='x')
        return Hmat
