#!/usr/bin/python
'''
Group Operations for Lattice/KSpace and Hamiltonian.
'''
from numpy import *
from utils import meshgrid_v,rotate
from numpy.linalg import norm
import pdb

__all__=['Group','TranslationGroup','PointGroup','CnvGroup','CnGroup','C3vGroup','C4vGroup','C6vGroup']

class Group(object):
    '''
    The Base class for groups.

    Construct
    --------------------
    Group(tp)

    Attributes
    ----------------
    tp:
        the type of group.

        * 'point': the point group.
        * 'tranlation': the translation group.
        * 'tr': time reversal group.
    '''
    def __init__(self,tp):
        self.tp=tp

class TranslationGroup(Group):
    '''
    Translation group(with type `translation`).

    Construct
    ----------------------
    TranslationGroup(Rs,per,k)

    Attributes
    ----------------------
    Rs:
        A list of R, the periodic vectors in real space.
    per:
        A list of boolean indicating periodic(True) or not(False) in specific direction.
        Or, equivalently, an array of integer indicating the index of periodic axes.
    k:
        The order of this translation group, the maximum translation operation times,
        the larger, the slower, the more reliable result will be got.
    '''
    def __init__(self,Rs,per,k=1):
        super(TranslationGroup,self).__init__('translation')
        self.Rs=array(Rs)
        self.per=zeros(len(Rs),dtype='bool')
        self.per[array(per)]=True
        self.per=tuple(per)
        self.k=k

        #initialize the translation matrix.
        self.tmatrix=meshgrid_v([arange(-k,k+1)]*sum(self.per),vecs=self.Rs[array(self.per)])

    def __setattr__(self,name,value):
        super(TranslationGroup,self).__setattr__(name,value)
        if name=='per' or name=='k' or name=='Rs':
            if hasattr(self,'tmatrix'):
                self.tmatrix=meshgrid_v([arange(-self.k,self.k+1)]*sum(self.per),vecs=self.Rs[array(self.per)])

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
        dr=ri-rjs
        absdr=norm(dr,axis=-1)
        ind=argmin(absdr)
        return absdr[ind],dr[ind]

class PointGroup(Group):
    '''
    Point group class.

    Construct
    -----------------
    Group(name)

    Attributes
    ----------------
    name:
        Name of group, like C6v, C3v ...
    irzone:
        The 2 vector defining the irreducible zone.
    '''
    def __init__(self,name):
        super(PointGroup,self).__init__('pointgroup')
        self.name=name
        self.irzone=None

    def spinrotate(self,spaceconfig,Hmat,theta,n=array([0,0,1.])):
        '''
        Rotate spin for theta along a specific axis.

        spaceconfig: 
            The configuration of hamiltonian space.
        Hmat:
            the Hamiltonian to perform this action.
        theta:
            the angle to rotate.
        n: 
            the axis(default is the z axis).
        '''
        dim2=spaceconfig.norbit*spaceconfig.natom
        U0=kron(Srot(theta=theta,n=n),identity(dim2))
        if spaceconfig.nnambu==1:
            U=U0
        else:
            nflv=2*dim2
            U=zeros([nflv*2,nflv*2],dtype='complex128')
            U[:nflv,:nflv]=U0
            U[nflv:,nflv:]=conj(U0)
        return dot(conj(transpose(U)),dot(Hmat,U))

    def spinimage(self,spaceconfig,Hmat,n=array([1.,0,0])):
        '''
        Image spin about plane n(indicated by the perpendicular vector).

        spaceconfig: 
            The configuration of hamiltonian space.
        Hmat:
            the Hamiltonian to perform this action.
        n: 
            the vector(default is the x axis).
        '''
        return self.spinrotate(spaceconfig,Hmat=Hmat,n=n,theta=pi)

    def actonK(self,k,ig,*args,**kwargs):
        '''
        Act this group on specific k-point.

        k:
            The k-vector.
        ig:
            The index of group members.
        
        *return*:
            The k-vector after group action.
        '''
        raise Exception('Not Implemented!')

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
            ki=self.actonK(k,ig)
            if not self.is_reducible(ki):
                return ki
        print 'warning: irreducible k=',k,'. k not in bzone?'
        return k


class CnvGroup(PointGroup):
    '''
    class of Cnv Group.

    Construct
    -----------------
    GnvGroup(n)

    Attributes
    -------------------
    n:
        The n of Cnv.
    '''
    def __init__(self,n):
        super(CnvGroup,self).__init__(name='C'+str(n)+'v')
        self.n=n
        self.ng=2*n

    def actonK(self,k,ig):
        '''
        Get the k' vector after group action on k specified by ig

        k:
            the input k-vector.
        ig:
            specify the group element to act on k.
        '''
        #get the transformation to k: G^-1(ig)*k
        k1=rotate(k,ig*pi*2/self.n)
        if ig>=self.ng/2:
            #image about the y axis
            k1=sv(k1,n=array([1.0,0.0]))
        return k1

    def actonH(self,spaceconfig,Hmat,ig):
        '''
        Act the group operation on H(spin action and orbit action).
        Please, pay attention to the order of operation here.
        rotate the wave function for angle theta, and then image about yz plane is equivalent to:
        image the Hamiltonian first, then rotate the spin operator for an angle -theta!

        spaceconfig: 
            The configuration of hamiltonian space.
        Hmat:
            the Hamiltonian to perform this action.
        ig:
            specify the group element.
        '''
        if ig >= self.ng/2:
            Hmat=self.spinimage(model,Hmat) #keep n default x axis.
        theta=ig*pi*2/self.n
        Hmat=self.spinrotate(model,Hmat,theta)
        return Hmat

class CnGroup(Group):
    '''
    Class of Cn group.

    Construct
    -----------------
    GnGroup(n)

    Attributes
    -------------------
    n:
        the n of Cn.
    '''
    def __init__(self,n):
        super(CnGroup,self).__init__(name='C'+str(n))
        self.n=n
        self.ng=n

    def actonK(self,k,ig):
        '''
        Get the k' vector after group action on k specified by ig

        k:
            the input k-vector.
        ig:
            specify the group element to act on k.
        '''
        #get the transformation to k: G^-1(ig)*k
        k1=rotate(k,ig*pi*2/self.n)
        return k1

    def actonH(self,model,Hmat,ig):
        '''
        Act the group operation on H(spin action and orbit action).
        Please, pay attention to the order of operation here.
        rotate the wave function for angle theta, and then image about yz plane is equivalent to:
        image the Hamiltonian first, then rotate the spin operator for an angle -theta!

        k:
            the input k-vector.
        ig:
            specify the group element to act on k.
        '''
        theta=ig*pi/3
        Hmat=self.spinrotate(model,Hmat,theta) #keep n default z axis.
        return Hmat

def C6vGroup():
    '''
    get a C6v group.
    '''
    return CnvGroup(6)

def C4vGroup():
    '''
    get a C4v group.
    '''
    return CnvGroup(4)

def C3vGroup():
    '''
    get a C3v group.
    '''
    return CnvGroup(3)
