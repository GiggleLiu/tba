#!/usr/bin/python
'''
Fermi surface related utilities.
'''
from numpy import *
from numpy.linalg import norm
from matplotlib.pyplot import *
from path import path_k,KPath
from utils import bisect

__all__=['FermiPiece','FermiSurface','FSHandler']

class FermiPiece(KPath):
    '''
    A piece of fermi surface.

    Construct
    ------------------------
    FermiPiece(centerpoint,klist)

    centerpoint:
        Center point for this pocket.
    klist:
        The k-route of this fermi surface piece.
    '''
    def __init__(self,centerpoint,klist):
        self.centerpoint=centerpoint
        super(FermiPiece,self).__init__(klist)

class FermiSurface(object):
    '''
    Fermi surface. It is a collection of <FermiPiece> instances.

    Construct
    -------------------
    FermiSurface(pieces=None)

    Attributes
    ---------------------
    pieces:
        A dict of pockets aranged in the order {token:<FermiPiece>}.
    '''
    def __init__(self,pieces=None):
        assert(pieces is None or type(pieces)==dict)
        self.pieces={} if pieces is None else pieces

    def __str__(self):
        s='FermiSurface -> Number of pieces %s.\n'%self.npiece
        s+=str(self.pieces.keys())
        return s

    def __getitem__(self,index):
        return self.pieces[index]

    def __add__(self,target):
        fs=FermiSurface()
        fs.pieces.update(self.pieces)
        fs.pieces.update(target.pieces)
        return fs

    def __radd__(self,target):
        return target.__add__(self)

    def __iadd__(self,target):
        self.pieces.update(target.pieces)

    @property
    def tokens(self):
        '''get piece tokens(or center points).'''
        return self.pieces.keys()

    @property
    def npiece(self):
        '''The number of fermi surface pieces.'''
        return len(self.pieces)

    def add_piece(self,token,piece):
        '''
        Add a piece of fermi surface to this system.

        token:
            The token.
        piece:
            A FermiPiece instance.
        '''
        self.pieces[token]=piece

    def eval(self,func):
        '''
        Evaluate func on the k-points on This fermi surface.
        
        function:
            a function defined on fermi surfaces.

        *return*:
            A list of evaluated data defined on fermi pieces.
        '''
        res=[]
        for piece in self.pieces.values():
            res.append(piece.eval(func))
        return res

    def show(self,method='plot',offset=0.,color='k',**kwargs):
        '''
        Plot func(defined on k vectors) on fermi surface.

        offset:
            Expand the fermi surface k points. with a factor of `offset`
        '''
        nfs=self.npiece
        ax=[]
        for i,piece in enumerate(self.pieces.values()):
            axis('equal')
            colormap=cm.get_cmap('RdYlBu')
            if method=='scatter':
                ax.append(scatter(piece.data[:,0],pieces.data[:,1],edgecolor='none',facecolor=color))
            elif method=='plot':
                ax.append(piece.show(color=color))
            else:
                raise ValueError('Undefined method %s for show.'%method)
        return ax

    def plot(self,datas,fsindices=None,**kwargs):
        '''
        Plot func(defined on k vectors) on fermi surface

        fsindex:
            the index of the fermi surface piece(s).
        '''
        if fsindices is None:
            for i,ps in enumerate(self.pieces.values()):
                ps.plot(datas[i],normalize=True,withfigure=False,**kwargs)
        
        for fsi in enumerate(fsindices):
            self.pieces[fsi].plot(datas[i],normalize=True,withfigure=False,**kwargs)

class FSHandler(object):
    '''
    Fermi surface handler class.

    Construct
    ------------------------
    FSHandler(efunc,resolution=0.02,tol=1e-4)

    Attributes
    -----------------------
    efunc:
        Energy function.
    resolution:
        The resolution of fermi-surface. dk~b/resolution, here, b is the k-lattice constant. default resolution is 0.02.
    tol:
        The accuracy of k point data.
    '''
    def __init__(self,efunc,resolution=0.02,tol=1e-4):
        self.resolution=resolution
        self.tol=tol
        self.efunc=efunc

    def findzero(self,start,end,eshift=0.):
        '''
        find zero point on a path from center point to k.

        start/end:
            The start/end point.
        efunc:
            Energy function.
        *return*:
            The k point at the zero point of energy.
        '''
        resolution=self.resolution
        tol=self.tol
        efunc=lambda k:self.efunc(k)-eshift
        dk=(end-start)*resolution
        Nmax=int(1./resolution)
        ei_old=efunc(start)
        for i in xrange(Nmax):
            ki=start+dk*(i+1)
            ei=efunc(ki)
            if abs(ei)<1e-15:
                return ki
            elif ei*ei_old<0:
                if ei<0:
                    klow,khigh=ki,ki-dk
                else:
                    klow,khigh=ki-dk,ki
                return bisect(efunc,klow,khigh,tol=tol)

    def get_ps(self,centerpoint,nseg,peripheral=None,eshift=0.):
        '''
        Get a piece of fermi surface.

        centerpoint:
            Center points.
        nseg:
            Number of k vectors.
        peripheral:
            Peripheral region, to limit the searching region..
        eshift:
            Fermi surface for E=eshift.

        *return*:
            A <FermiPiece> instance.
        '''
        if peripheral is None:
            x,y=centerpoint
            peripheral=array([[x-pi,y-pi],[x+pi,y-pi],[x+pi,y+pi],[x-pi,y+pi],[x-pi,y-pi]])
        kp=path_k(peripheral,nseg)
        kl=kp.data
        for ii in xrange(len(kl)):
            k=kl[ii]
            ki=self.findzero(centerpoint,k,eshift)
            if ki is None:
                return None
            kl[ii]=ki
        return FermiPiece(centerpoint,kl)

    def get_fs_byname(self,name,kspace,nseg,eshift=0):
        '''
        Fermi Surface decided by `G` `K` `M`.

        name:
            The name code of fermi surface type.
            `G` -> G pockets.
            `K` -> K pockets.
            `M` -> M pockets.
        kspace:
            A <KSpace> instance.

        *return*:
            A list of <FermiPiece> instance.
        '''
        self.kspace=kspace

        Kl=kspace.K
        Ml=kspace.M
        G0=kspace.G
        nK,nM=len(Kl),len(Ml)

        fs=FermiSurface()
        if 'G' in name:
            centralpoints=[]
            peripherals=[]
            centralpoints.append(G0)
            peripherals.append(kspace.K+[kspace.K[0]])
            for cp,peri in zip(centralpoints,peripherals):
                ps=self.get_ps(cp,nseg=nseg,peripheral=peri,eshift=eshift)
                if ps is not None:
                    fs.add_piece('G',ps)
                else:
                    print 'This band does not have a `G` pocket!'

        if 'K' in name:
            centralpoints=[]
            peripherals=[]
            for i in xrange(nK):
                GK=Kl[i]-G0
                GM=Ml[i]-G0
                n=GK/norm(GK)
                Mi2=2*dot(n,GM)*n-GM
                centralpoints.append(Kl[i])
                peripherals.append(array([Ml[i],G0,Mi2]))
            for i,(cp,peri) in enumerate(zip(centralpoints,peripherals)):
                ps=self.get_ps(cp,nseg=nseg,peripheral=peri,eshift=eshift)
                if ps is not None:
                    fs.add_piece('K%s'%i,ps)
                else:
                    print 'This band does not have a `K%s` pocket!'%i
        if 'M' in name:
            centralpoints=[]
            peripherals=[]
            for i in xrange(nM):
                GK=Kl[i]-G0
                GM=Ml[i]-G0
                n=GM/norm(GM)
                Ki2=2*dot(n,GK)*n-GK
                centralpoints.append(Ml[i])
                peripherals.append(array([Kl[i],G0,Ki2]))
            for i,(cp,peri) in enumerate(zip(centralpoints,peripherals)):
                ps=self.get_ps(cp,nseg=nseg,peripheral=peri,eshift=eshift)
                if ps is not None:
                    fs.add_piece('M%s'%i,ps)
                else:
                    print 'This band does not have a `M%s` pocket!'%i
        return fs
