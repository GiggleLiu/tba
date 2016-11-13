from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
from numpy.linalg import eigh,eigvalsh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import time,pdb,sys
sys.path.insert(0,'../')

from spaceconfig import SpaceConfig,SuperSpaceConfig,SpinSpaceConfig
from generator import RHGenerator
from oplib import op_simple_hopping,op_U,op_simple_onsite,op_from_mats,op_on_bond
from tba.lattice import Structure
from blockmatrix.blocklib import eigbsh,eigbh,get_blockmarker,tobdmatrix
from utils import sx,sy,sz

#SpaceConfig.SPACE_TOKENS=['nambu','atom','spin','orbit']

def test_single_site():
    print 'Test single site operator.'
    spaceconfig=SuperSpaceConfig([2,1,1])
    print 'Hubbard U'
    op_ninj=op_U(label='ninj',spaceconfig=spaceconfig)
    udata=zeros([spaceconfig.hndim]*2,dtype='complex128')
    udata[3,3]=1
    assert_allclose(op_ninj(),udata)
    print 'Spin'
    op_sy=op_from_mats(label='Sy',spaceconfig=spaceconfig,mats=[sy],bonds=None)
    sydata=zeros([spaceconfig.hndim]*2,dtype='complex128')
    sydata[1,2]=-1j
    sydata[2,1]=1j
    assert_allclose(op_sy(),sydata)

def test_all():
    test_single_site()

if __name__=='__main__':
    test_all()
