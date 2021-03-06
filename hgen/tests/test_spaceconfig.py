from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.linalg import eigh
import sys,pdb
sys.path.insert(0,'../')

#from utils import perm_parity
from utils import *
from op import *
from oplib import *
from spaceconfig import *
sys.path.insert(0,'../../lattice')
from bond import Bond

SpaceConfig.SPACE_TOKENS=['nambu','atom','spin','orbit']

class Test_superconfig():
    def __init__(self):
        config=[5,2,2]
        self.sscfg=SuperSpaceConfig(config)
        self.sscfg_ne=SuperSpaceConfig(config,ne=10)
        self.scfgs=[self.sscfg,self.sscfg_ne]
        for scfg in self.scfgs:
            assert_allclose([scfg.natom,scfg.nspin,scfg.norbit],config)
            assert_allclose(scfg.config[1:],config)

    def test_indices_occ(self):
        spaceconfig=SuperSpaceConfig([2,2,1])
        #the first digree of freedom(atom1,up) occupied, means [1,x,x,x]
        assert_allclose(spaceconfig.indices_occ(occ=[0]),arange(1,16,2))
        #the first two digree of freedom(atom1,up^dn) occupied, means [1,1,x,x]
        assert_allclose(spaceconfig.indices_occ(occ=[0,1]),arange(3,16,4))
        #the first two digree of freedom(atom1,up^dn) occupied/unoccupied, means [1,0,x,x]
        assert_allclose(spaceconfig.indices_occ(occ=[0],unocc=[1]),arange(1,16,4))

    def test_parse(self):
        '''test for ind-config parsing.'''
        for scfg in self.scfgs[:2]:
            print 'Running ind-config parsing test for %s.'%scfg
            print 'Test for 1D array ind.'
            ind=random.randint(0,scfg.hndim,10)
            config=scfg.ind2config(ind)
            ind2=scfg.config2ind(config)
            assert_(all(ind2==ind))
            print 'test for integer ind.'
            ind=random.randint(0,scfg.hndim)
            config=scfg.ind2config(ind)
            ind2=scfg.config2ind(config)
            assert_(ind2==ind)

    def test_all(self):
        self.test_indices_occ()
        self.test_parse()

class Test_spinconfig():
    def __init__(self):
        config1=[1,2]
        config2=[1,3]
        self.scfg=SpinSpaceConfig(config1)
        self.scfg1=SpinSpaceConfig(config2)
        self.scfgs=[self.scfg,self.scfg1]
        for scfg,config in zip(self.scfgs,[config1,config2]):
            assert_allclose([scfg.natom,scfg.nspin],config)
            assert_allclose(scfg.config,config)

    def test_sigma(self):
        print 'Test spin matrices.'
        for i in [1,2,3]:
            for scfg in self.scfgs:
                si=scfg.sigma(i)
                #test hermicity
                assert_allclose(si.T.conj(),si)
                evals=eigh(si)[0]
                tvals=arange(scfg.nspin)-(scfg.nspin-1)/2.
                assert_allclose(evals,tvals,atol=1e-10)

    def test_config(self):
        config=SpinSpaceConfig([4,2])
        print 'Testing config-ind parsing 1D.'
        assert_(config.config2ind([1,1,0,1])==13)
        assert_allclose(config.ind2config(5),[0,1,0,1])
        print 'Testing config ind parsing 2D.'
        assert_allclose(config.config2ind([[1,1,0,1],[0,0,1,0]]),[13,2])
        assert_allclose(config.ind2config([5,10]),[[0,1,0,1],[1,0,1,0]])

    def test_all(self):
        self.test_sigma()
        self.test_config()

Test_spinconfig().test_all()
Test_superconfig().test_all()
