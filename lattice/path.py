'''
Path in 2D space.
'''
from numpy import *
from numpy.linalg import norm
from matplotlib.pyplot import *
import warnings,pdb

__all__=['KPath','path_k','opath_k']

class KPath(object):
    '''
    A path in k space

    Attributes:
        :data: 2d array, a list of k vectors.
    '''
    def __init__(self,klist=zeros([0,2])):
        self.data=klist

    def __getitem__(self,index):
        return self.data[index]

    def __len__(self):
        return self.N

    def __add__(self,kp):
        return KPath(concatenate([self.data,kp.data]))

    def __radd__(self,kp):
        return KPath(concatenate([kp.data,self.data]))

    def __iadd__(self,kp):
        self.data=concatenate([self.data,kp.data])
        return self

    def __iter__(self):
        for i in xrange(self.N):
            yield self.data[i]

    @property
    def N(self):
        '''number of points.'''
        return len(self.data)

    def as_1d(self,mode='abs'):
        '''
        ravel this path into 1D.
        
        Parameters:
            :mode:

                * 'abs': the absolute length.
                * 'linear': the linear.
                * 'norm': the distance from 0.
        '''
        if mode=='abs':
            return concatenate([array([0.]),cumsum(norm(diff(self.data,axis=0),axis=1))])
        elif mode=='norm':
            return norm(self.data,axis=1)
        elif mode=='linear':
            return linspace(0,1,self.N)
        else:
            raise ValueError()

    def eval(self,func):
        '''
        eval func on this path, return a copy of data.

        func:
            function of k.
        '''
        data=array([func(k) for k in self.data])
        return data

def path_k(vertices,N,mode='equalstep'):
    '''
    Create a KPath instance along specific route.

    Parameters:
        :vertices: 2d array, turnning points.
        :N: int, # of points.
        :mode:

            * 'equalstep' -> equal step length.
            * 'equaldot' -> equal number of dots for each interval.
    Return:
        2d array, the path
    '''
    vertices=array(vertices)
    npoint=len(vertices)
    llist=[norm(end-start) for start,end in zip(vertices[:-1],vertices[1:])]
    L=sum(llist)
    kl=[]
    for i in xrange(npoint-1):
        if mode=='equalstep':
            ni=int(llist[i]/L*N)
        elif mode=='equaldot':
            ni=N/(npoint-1)
        point1=vertices[i]
        point2=vertices[i+1]
        step=(point2-point1)/ni
        kl+=[i*step+point1 for i in xrange(ni)]
    kl=array(kl,dtype='float64')
    return kl

def opath_k(R=1.,N=240):
    '''
    circle path,

    Parameters:
        :R: the radius of this path.
        :N: total number of points.

    Return:
        2d array, the path
    '''
    dphi=2*pi/N
    kl=[(cos(i*dphi),sin(i*dphi)) for i in xrange(N)]
    kl=array(kl,dtype='float64')
    return kl
