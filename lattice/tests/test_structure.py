import time,pdb,sys
sys.path.insert(0,'../')
from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose

from latticelib import *
from structure import *
from group import TranslationGroup
from plotlib import scatter_sites
from bond import *

class TestStructure():
    def setup(self,N,vdim):
        self.N=N
        self.vdim=vdim

    def get_structure(self):
        xs=[cumsum([0.9,1.1]*(ni/2)) for ni in self.N]
        sites=array(meshgrid(*xs)).reshape([len(self.N),-1]).transpose()
        return Structure(sites)

    def test_bonds(self):
        print 'Test initialize bonds.'
        st=self.get_structure()
        ion()
        assert_(not st.bonds_initialized)
        assert_raises(NotInitializedError,st.getbonds,1)
        st.initbonds(K=15)
        assert_(st.bonds_initialized)
        print 'Test getting bonds.'
        assert_(st.getbonds(100)==[])
        v0,vx,vy=array([0.,0]),array([0.9,0]),array([0,0.9])
        b0s=[Bond(i,i,v0) for i in xrange(st.nsite)]
        b1s=[Bond(i,i+1,vx) for i in [1,5,9,13]]+[Bond(i,i+4,vy) for i in [4,5,6,7]]
        b1s=[-b for b in b1s]+b1s
        assert_(st.getbonds(0)==b0s)
        assert_(st.getbonds(1)==b1s)
        print 'Test measuring distance.'
        matom1,matom2=4,11
        r,rv=st.get_distance(matom1,matom2)
        assert_almost_equal(r,sqrt(3.1**2+0.9**2))
        assert_allclose(rv,[3.1,0.9])
        print 'Test query neighbors.'
        sites,dis=st.get_neighbors(4,k=5)
        pm=argsort(sites)
        print sites
        assert_allclose(sites[pm],[0,1,5,8,9])
        assert_allclose(dis[pm],[1.1,1.1*sqrt(2),1.1,0.9,sqrt(0.9**2+1.1**2)])

        print '-------- Test using translation group. -----------'
        tgroup=TranslationGroup(Rs=array([[self.N[0],0],[0,self.N[1]]]),per=(True,False))
        st.usegroup(tgroup)
        st.initbonds(K=15)
        b1ps=[Bond(i,i-3,vx) for i in [3,7,11,15]]
        b1s=b1s+b1ps+[-b for b in b1ps]
        assert_(st.getbonds(1)==b1s)
        print 'Test measuring distance.'
        matom1,matom2=4,11
        r,rv=st.get_distance(matom1,matom2)
        assert_almost_equal(r,sqrt(0.9**2+0.9**2))
        assert_allclose(rv,[-0.9,0.9])
        print 'Test query neighbors.'
        sites,dis=st.get_neighbors(4,k=4)
        pm=argsort(sites)
        assert_allclose(sites[pm],[0,5,7,8])
        assert_allclose(dis[pm],[1.1,1.1,0.9,0.9])

        print 'Test find site number by position.'
        isite=array([2.9]*st.vdim)
        assert_(st.findsite(isite)==10)
        isite=array([3.]*st.vdim)
        assert_(st.findsite(isite) is None)

    def test_all(self):
        self.test_bonds()

def test_lattice(N,tp):
    '''
    Test functions for lattice class.
    '''
    print '############ TYPE = %s #############'%type
    if tp=='':
        lt=construct_lattice(N=N,lattice_type=tp,a=[(1,0),(0,0.8)])
    else:
        lt=construct_lattice(N=N,lattice_type=tp)
    lt.initbonds()
    print 'Test index2l, l2index'
    ntest=100
    inds=random.randint(0,lt.nsite,ntest)
    inds2=lt.l2index(lt.index2l(inds))
    assert_allclose(inds2,inds)
    print 'Test findsite'
    pos=concatenate([lt.sites[inds],random.random([ntest,lt.vdim])*10],axis=0)
    for psi in pos:
        lind=lt.findsite(psi)
        sind=Structure.findsite(lt,psi)
        if lind is None:
            assert_(sind is None)
        else:
            assert_(lt.l2index(lind)==sind)
    print 'For periodic condition.'
    lt.set_periodic([True]*lt.dimension)
    lt.initbonds()
    print 'Test get bonds for a unit cell, and group action of vectors.'
    cbonds=lt.get_cbonds()
    v0={'honeycomb':[0.5,sqrt(3)/6],
            'square':[1.0,0],
            'triangular':[1.0,0],
            }.get(tp)
    if v0 is not None:
        g=lt.groups['point']
        vs=g.acton_vec(v0,ig=arange(g.ng if g._type==0 else g.ng/2))
        if tp=='honeycomb':
            cbonds_nn=[Bond(0,1,v) for v in vs]; cbonds_nn=cbonds_nn+[-b for b in cbonds_nn]
        else:
            cbonds_nn=[Bond(0,0,v) for v in vs]
    else:
        if tp=='chain':
            cbonds_nn=[Bond(0,0,v) for v in [[-1],[1]]]
        elif tp=='':
            cbonds_nn=[Bond(0,0,v) for v in [[0,0.8],[0,-0.8]]]
    assert_(cbonds[1]==cbonds_nn)

def test_kspace(N,lt):
    '''
    Test functions for lattice class.
    '''
    lt=construct_lattice(N=N,lattice_shape=lt)
    tgroup=TranslationGroup(Rs=lt.a*lt.N[:,newaxis],per=(True,False))
    lt.usegroup(tgroup)
    ks=lt.kspace
    bzone=ks.get_bzone()
    ion()
    bzone.show()
    ks.show()
    km=ks.kmesh
    #kmbzone=array(filter(bzone.inbzone,km.reshape([-1,km.shape[-1]])))
    kmbzone=km[bzone.inbzone(km)]
    scatter(kmbzone[:,0],kmbzone[:,1],color='r')
    axis('equal')
    pdb.set_trace()
 
def test_all():
    ts=TestStructure()
    ts.setup((4,4),2)
    ts.test_all()
    for tp in ['','honeycomb','square','triangular','chain']:
        test_lattice((4,4),tp)

if __name__=='__main__':
    test_all()
    #test_lattice((30,20),'honeycomb')
    #test_kspace((30,20),'honeycomb')
