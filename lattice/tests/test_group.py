import time,pdb,sys
sys.path.insert(0,'../')
from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose

from group import *

def test_rv():
    print 'Test rotate, 1 vector.'
    v=[1,1.]
    angle=pi/2
    v2=vector_rotate(v,angle)
    assert_allclose(v2,[-1,1])
    print 'Test rotate, multiple vector'
    vs=[[1,1],[1,0],[0,0]]
    vs2=vector_rotate(vs,angle)
    assert_allclose(vs2,[[-1,1],[0,1],[0,0]],atol=1e-10)
    print 'Test image, 1 vector.'
    axis=[0,2]
    v3=vector_image(v,axis)
    assert_allclose(v3,[1,-1])
    print 'Test image, multiple vector.'
    v3s=vector_image(vs,axis)
    assert_allclose(v3s,[[1,-1],[1,0],[0,0]])
    pdb.set_trace()


def test_rs():
    print 'Test rotate, spin'

def test_all():
    test_rv()

if __name__=='__main__':
    test_all()
