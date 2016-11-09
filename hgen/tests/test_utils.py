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

def test_perm():
    print 'Testing parity for permutations'
    def perm(N,times):
        t=0
        base=arange(N)
        for i in xrange(times):
            ri=random.randint(0,N-1)
            base[ri:ri+2]=base[ri+1],base[ri]
        return base
    for k in xrange(10):
        N=random.randint(2,20)
        times=random.randint(0,50)
        assert_(perm_parity(perm(N,times))==times%2)

test_perm()
