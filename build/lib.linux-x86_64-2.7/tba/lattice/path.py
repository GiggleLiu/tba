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

    klist:
        a list of k vectors.
    label:
        the label for this path.
    '''
    def __init__(self,klist=zeros([0,2]),label=''):
        self.label=label
        self.data=klist
        self.info={}

    def __getitem__(self,index):
        return self.data[index]

    def __len__(self):
        return self.N

    def __add__(self,kp):
        return KPath(concatenate([self.data,kp.data]),label=self.label)

    def __radd__(self,kp):
        return KPath(concatenate([kp.data,self.data]),label=self.label)

    def __iadd__(self,kp):
        self.data=concatenate([self.data,kp.data])
        return self

    def __iter__(self):
        for i in xrange(self.N):
            yield self.data[i]

    def __get_xdata__(self,mode='abs'):
        if mode=='abs':
            return self.abspath
        elif mode=='norm':
            return norm(self.data,axis=1)
        elif mode=='lin':
            return linspace(0,1,self.N)
        elif mode=='x':
            return self.data[:,0]
        else:
            raise Exception('Error','Unknown mode '+mode)

    @property
    def abspath(self):
        '''Absolute distance of path.'''
        return concatenate([array([0.]),cumsum(norm(diff(self.data,axis=0),axis=1))])

    @property
    def N(self):
        '''number of points.'''
        return len(self.data)

    def eval(self,func):
        '''
        eval func on this path, return a copy of data.

        func:
            function of k.
        '''
        data=array([func(k) for k in self.data])
        return data

    def scatter(self,data,zoom=1.,extradata=None,**kwargs):
        '''
        plot func(defined on k vectors) on k-space.

        data:
            the dataset.
        zoom:
            zoom path in k-space.
        offset:
            expand the fermi surface k points. with a factor of `offset`
        '''
        colormap=cm.get_cmap('RdYlBu')
        if extradata==None:
            ax=scatter(self.data[:,0]*zoom,self.data[:,1]*zoom,c=data,edgecolor='none',cmap=colormap,**kwargs)
        else:
            ax=scatter(self.abspath,self.data,c=extradata,s=20,edgecolor='none',cmap=colormap,**kwargs)
            xlim([self.abspath.min(),self.abspath.max()])
        return ax

    def plot(self,data,mode='abs',**kwargs):
        '''
        show data.

        mode:
            'abs' - plot with x-axis equals to path length.
            'x' - x-axis equals to kx
            'lin' - x-axis equals to N(the number of dot)
        normalize: scale x-axis to unit.'''
        xlabel('k')
        ylabel('Energy (eV)')
        x=self.__get_xdata__(mode)
        xlim([x.min(),x.max()])
        plot(x,data,**kwargs)

    def show(self,mode='scatter',**kwargs):
        '''
        show this path.
        '''
        if mode=='plot':
            plot(self.data[:,0],self.data[:,1],lw=3,**kwargs)
        else:
            scatter(self.data[:,0],self.data[:,1],s=10,edgecolor='none',**kwargs)

def path_k(vertices,N,mode='equalstep'):
    '''
    Create a KPath instance along specific route.

    vertices:
        a list of turnning points.
    N:
        total points.
    mode:
        `equalstep` -> equal step length.
        `equaldot` -> equal number of dots for each interval.
    *return*:
        a KPath instance.
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
    return KPath(kl)


def opath_k(R=1.,N=240):
    '''
    circle path,

    R: 
        the radius of this path.
    N:
        total number of points.
    *return*:
        a KPath instance.
    '''
    dphi=2*pi/N
    kl=[(cos(i*dphi),sin(i*dphi)) for i in xrange(N)]
    kl=array(kl,dtype='float64')
    return KPath(kl)
