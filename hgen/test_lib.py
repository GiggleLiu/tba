'''Test libs.'''

from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.linalg import inv,eigh,eigvalsh
from matplotlib.pyplot import *
import pdb

from lib import *
from plotlib import *

#__all__=['random_H','E2DOS','H2E','G2A','A2DOS','H2G','EU2C','C2H']
def test_H2E():
    nband=3
    H=random_H(nband,size=())
    print 'testing single H2E'
    E1,V1=H2E(H,evalvk=True)
    assert_allclose((V1*E1).dot(V1.T.conj()),H)
    print 'testing multiple H2E'
    size=(10,12)
    H=random_H(nband,size=size)
    E,V=H2E(H,evalvk=True)
    assert_allclose([[(V[i,j]*E[i,j]).dot(V[i,j].T.conj()) for j in xrange(size[1])] for i in xrange(size[0])],H)
    print 'Testing for displaying.'
    ion()
    plot_e(None,E1)
    pause(1)
    cla()
    pcolor_e(None,E[:,:,0])
    pause(1)
    cla()
    scatter_e(None,E)
    pause(1)

def test_H2G():
    nband=3
    size=()
    H=random_H(nband,size=size)
    w=random.random()
    print 'testing single-w H2G'
    assert_allclose(H2G(H,w,geta=0.1,tp='r'),inv(-H+(w+0.1j)*identity(nband)))
    assert_allclose(H2G(H,w,geta=0.1,tp='a'),inv(-H+(w-0.1j)*identity(nband)))
    assert_allclose(H2G(H,w,geta=0.1,tp='matsu'),inv(-H+1j*w*identity(nband)))
    print 'testing multi-w H2G'
    w=random.random(10)
    for tp in ['r','a','matsu']:
        assert_allclose(H2G(H,w,geta=0.1,tp=tp),[H2G(H,wi,geta=0.1,tp=tp) for wi in w])
    print 'testing multi-w-H H2G'
    size=(10,)
    H=random_H(nband,size=size)
    for tp in ['r','a','matsu']:
        assert_allclose(H2G(H,w,geta=0.1,tp=tp),[[H2G(hi,wi,geta=0.1,tp=tp) for hi in H] for wi in w])

def test_todos():
    wlist=linspace(-2,2,100)
    nband=3
    H=random_H(nband,size=(10,12))
    geta=0.1
    print 'Tesing generating dos.'
    dos1=E2DOS(H2E(H),wlist=wlist,geta=geta)
    dos2=A2DOS(G2A(H2G(w=wlist,Hmesh=H,geta=geta,tp='r'),tp='r'))
    assert_allclose(dos1,dos2.sum(axis=(1,2)))

test_H2G()
test_todos()
test_H2E()
