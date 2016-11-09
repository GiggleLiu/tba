import time,pdb,sys
sys.path.insert(0,'../')
from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose

from latticelib import *
from structure import Structure
from group import TranslationGroup
from plotlib import *

def get_kspace(tp):
    N=(10,10)
    lt=construct_lattice(N=N,lattice_type=tp)
    ks=lt.get_kspace()
    print 'Test for reciprocal vectors.'
    if tp=='honeycomb' or tp=='triangular':
        assert_allclose(ks.b,[[2*pi,-2*pi/sqrt(3)],[0,4*pi/sqrt(3)]],atol=1e-5)
    elif tp=='square':
        assert_allclose(ks.b,[[2*pi,0],[0,2*pi]],atol=1e-5)
    elif tp=='chain':
        assert_allclose(ks.b,[[2*pi]],atol=1e-5)
    return ks

def test_bzone(tp):
    '''
    Test functions for lattice class.
    '''
    print 'Test %s lattice'%tp
    ks=get_kspace(tp)
    kl=array([[0,0],[4,0],[4,4],[6,6]])
    if tp!='chain':
        bzone=ks.get_bzone()
    if tp=='honeycomb' or tp=='triangular':
        assert_allclose(bzone.inbzone(kl),[1,1,0,0])
    elif tp=='square':
        assert_allclose(bzone.inbzone(kl),[1,0,0,0])
        #show_bzone(bzone)
 
if __name__=='__main__':
    for tp in ['honeycomb','triangular','square','chain']:
        test_bzone(tp)
