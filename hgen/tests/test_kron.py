from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
from numpy.linalg import eigh,eigvalsh
from scipy import sparse as sps
from scipy.sparse.linalg import eigsh
import time,pdb,sys
sys.path.insert(0,'../')

from tba.hgen import SpaceConfig,SuperSpaceConfig,SpinSpaceConfig,RHGenerator,op_simple_hopping,op_U,op_simple_onsite
from tba.lattice import Structure
from blockmatrix.blocklib import eigbsh,eigbh,get_blockmarker,tobdmatrix,SimpleBMG
import kron

def test_kron():
    for shpA,shpB in [((0,0),(0,0)),((3,3),(6,3)),\
            ((2,0),(0,2)),((100,200),(100,200))]:
        print shpA,'x',shpB
        A=random.random(shpA)
        B=random.random(shpB)
        A[A<0.8]=0
        B[B<0.8]=0
        A=sps.csr_matrix(A)
        B=sps.csr_matrix(B)
        A_coo,B_coo=A.tocoo(),B.tocoo()
        t0=time.time()
        res1=sps.kron(A_coo,B_coo)
        t1=time.time()
        res2=kron.kron_csr(A,B)
        t2=time.time()
        res3=kron.kron_coo(A_coo,B_coo)
        t3=time.time()
        assert_allclose((res1-res2).data,0)
        print 'Test kron time used %s(csr),%s(coo) compared to scipy %s'%(t2-t1,t3-t2,t1-t0)
    pdb.set_trace()

def test_kron_takerow():
    for shpA,shpB in [((2,0),(2,0)),((100,200),(100,200))]:
        print shpA,'x',shpB
        A=random.random(shpA)
        B=random.random(shpB)
        rows=random.randint(0,A.shape[0]*B.shape[0],A.shape[0]*B.shape[0]/2)
        A[A<0.8]=0
        B[B<0.8]=0
        A=sps.csr_matrix(A)
        B=sps.csr_matrix(B)
        t0=time.time()
        res1=sps.kron(A,B).asformat('csr')[rows]
        t1=time.time()
        res2=kron.kron_csr(A,B,takerows=rows)
        t2=time.time()
        assert_allclose((res1-res2).data,0)
        print 'Test kron(take-row) time used %s(csr) compared to scipy %s'%(t2-t1,t1-t0)
    pdb.set_trace()

test_kron_takerow()
test_kron()
