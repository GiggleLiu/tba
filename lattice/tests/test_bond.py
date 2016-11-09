import time,pdb,sys
sys.path.insert(0,'../')
from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose

from bond import *

vdim=2

############# utilities ##################
def _setup_globals(dim):
    global vdim
    vdim=dim

def _assert_same_collection(bc1,bc2):
    assert_(bc1==bc2)

def _random_bond(nsite=10):
    atom1=random.randint(0,nsite)
    atom2=random.randint(0,nsite)
    bondv=random.random(vdim)
    return Bond(atom1,atom2,bondv)

def _random_bc(nsite=10,nbond=100):
    atom1s=random.randint(0,nsite,nbond)
    atom2s=random.randint(0,nsite,nbond)
    bondvs=random.random([nbond,vdim])
    #check for validity
    #atoms=concatenate([atom1s[:,newaxis],atom2s[:,newaxis]],axis=1).view([('',atom1s.dtype)]*2)
    #ar,indices=unique(atoms,return_index=True)
    #atom1s,atom2s,bondvs=atom1s[indices],atom2s[indices],bondvs[indices]
    return BondCollection((atom1s,atom2s,bondvs))

############## start ##################

def test_bond():
    print 'Testing Bond.'
    b1=_random_bond()
    print 'reverse'
    b1_r=-b1
    assert_(b1.atom1==b1_r.atom2)
    assert_(b1.atom2==b1_r.atom1)
    assert_allclose(b1.bondv,-b1_r.bondv)
    print '=='
    b2=Bond(b1.atom1,b1.atom2,b1.bondv)
    assert_(b1_r==-b2)
    assert_(b1_r!=b2)
    print b1

def test_construction():
    filename='test.dat'
    print 'test construction'
    bc=_random_bc()
    pm=arange(bc.N); random.shuffle(pm)
    bc2=bc[pm]
    _assert_same_collection(bc2,bc)
    print 'test saveload.'
    bc.save(filename)
    bc2=load_bonds(filename)
    _assert_same_collection(bc2,bc)

def test_add():
    print 'test __len__, __(r)add__, __itter__, __getitem__'
    nsite=10
    bc1=_random_bc(nsite=10)
    bc2=_random_bc(nsite=10)
    bc3=bc1+bc2
    for i,b1 in enumerate(bc1):
        assert_(b1==bc3[i])
    _assert_same_collection(bc3[len(bc1):],bc2)

    print 'test property N, vdim'
    assert_(bc3.N==bc2.N+bc1.N)
    assert_(bc3.vdim==vdim)

def test_query():
    print 'test query.'
    nsite=10
    bc1=BondCollection(((0,1,2,3,4,2,6),(0,0,0,4,3,1,0),([0,1],[2,3],[1,3],[0,1.],[1,1],[1,1],[2,2])))
    xs=[2,5]
    _assert_same_collection(bc1.query(atom1=2),BondCollection((bc1.atom1s[xs],bc1.atom2s[xs],bc1.bondvs[xs])))
    xs=[0,2,5]
    _assert_same_collection(bc1.query(atom1=(0,2)),BondCollection((bc1.atom1s[xs],bc1.atom2s[xs],bc1.bondvs[xs])))
    xs=[2]
    _assert_same_collection(bc1.query(atom1=2,atom2=0),BondCollection((bc1.atom1s[xs],bc1.atom2s[xs],bc1.bondvs[xs])))
    xs=[0,1,2,5,6]
    _assert_same_collection(bc1.query(atom1=2,atom2=0,condition='or'),BondCollection((bc1.atom1s[xs],bc1.atom2s[xs],bc1.bondvs[xs])))
    xs=[2,4]
    _assert_same_collection(bc1.query(atom1=2,bondv=[1,1],condition='xor'),BondCollection((bc1.atom1s[xs],bc1.atom2s[xs],bc1.bondvs[xs])))

def test_all():
    test_bond()
    test_construction()
    test_add()
    test_query()

if __name__=='__main__':
    for i in xrange(3):
        _setup_globals(dim=i+1)
        test_all()
