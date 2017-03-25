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

class TestLinear():
    def __init__(self):
        self.spaceconfig=SuperSpaceConfig([1,4,2,1])

    def test_qlinear(self):
        print 'Test for QLinear'
        ndim=self.spaceconfig.ndim
        for i in xrange(10):
            indices=random.randint(0,ndim,4)
            if len(unique(indices))!=4:
                continue
            factor=random.random()
            l1=Qlinear(self.spaceconfig,indices=indices,factor=factor)
            assert_allclose((l1.index1,l1.index2,l1.index3,l1.index4),indices)
            print l1

    def get_random_xlinear(self,n,nbody):
        '''get n random nbody linears'''
        res=[]
        config=self.spaceconfig.config
        scfg=self.spaceconfig
        for i in xrange(n):
            cs=[[random.randint(0,nmax) for nmax in config] for i in xrange(nbody)]
            indices=[scfg.c2ind(ci) for ci in cs]
            if nbody==2:
                item=Bilinear(scfg,index1=indices[0],index2=indices[1],factor=random.random())
            elif nbody==4:
                i=indices[0]
                j=indices[1]
                if i==j:
                    j=j-1 if j!=0 else j+1
                item=Qlinear_ninj(scfg,i=i,j=j,factor=random.random())
            else:
                item=Xlinear(scfg,indices=indices,factor=random.random())
            item*=random.random()
            res.append(item)
        return res

    def get_random_operator(self,n,nbody=None):
        '''
        get random operator with n xlinears.
        '''
        scfg=self.spaceconfig
        nbody=random.choice([1,2,4]) if nbody is None else nbody
        xl=self.get_random_xlinear(n,nbody)
        op=sum(xl)
        op=op*random.random()
        return op

    def test_site_shift(self):
        print 'Test site shift.'
        shift=2
        new_spaceconfig=SuperSpaceConfig([1,7,2,self.spaceconfig.norbit])
        op=self.get_random_operator(n=5,nbody=2)
        nop=site_shift(op,shift,new_spaceconfig)
        op2=site_shift(nop,-shift,self.spaceconfig)
        assert_(op.factor==op2.factor)
        for o1,o2 in zip(op.suboperators,op2.suboperators):
            assert_(o1.factor==o2.factor)
            assert_(o1.factor==o2.factor)
            assert_allclose(o1.indices,o2.indices)

    def test_op_fusion(self):
        print 'Test join two operators.'
        shift=2
        n1,n2,n3=2,3,4
        op1=self.get_random_operator(n=n1,nbody=2)
        op2=self.get_random_operator(n=n2,nbody=2)
        op3=self.get_random_operator(n=n3,nbody=2)
        nop=op_fusion(label='newlabel',operators=[op1,op2,op3])
        assert_(nop.label=='newlabel')
        scfg=nop.spaceconfig
        atomaxis=scfg.get_axis('atom')
        for o1,o2 in zip(op2.suboperators,nop.suboperators[n1:n1+n2]):
            assert_(all(scfg.ind2c(o2.indices)[:,atomaxis]==self.spaceconfig.ind2c(o1.indices)[:,atomaxis]+self.spaceconfig.natom))
        for o1,o2 in zip(op3.suboperators,nop.suboperators[n1+n2:]):
            assert_(all(scfg.ind2c(o2.indices)[:,atomaxis]==self.spaceconfig.ind2c(o1.indices)[:,atomaxis]+2*self.spaceconfig.natom))

    def test_oponbond(self):
        print 'Test op_on_bond'
        natom=self.spaceconfig.natom
        dim1=self.spaceconfig.ndim/natom
        atom_axis=self.spaceconfig.get_axis('atom')
        print 'test on-site'
        mats=[0.5*identity(dim1)]*natom
        bonds=[Bond(i,i,zeros(2)) for i in xrange(natom)]
        op=op_on_bond(label='E',spaceconfig=self.spaceconfig,mats=mats,bonds=bonds)
        for i in xrange(natom):
            for j in xrange(dim1):
                assert_(all(self.spaceconfig.ind2c(op.suboperators[dim1*i+j].indices)[:,atom_axis]==ones(2)*i))
                assert_(all(op.suboperators[dim1*i+j].factor==0.5))
        print 'test for nearest hopping'
        mats=[0.5*identity(dim1)]*(natom-1)
        bonds=[Bond(i,i+1,array([1,0])) for i in xrange(natom-1)]
        op=op_on_bond(label='T',spaceconfig=self.spaceconfig,mats=mats,bonds=bonds)
        for i in xrange(natom-1):
            for j in xrange(dim1):
                assert_(all(self.spaceconfig.ind2c(op.suboperators[dim1*i+j].indices)[:,atom_axis]==array([i,i+1])))
                assert_(all(op.suboperators[dim1*i+j].factor==0.5))

    def test_xlinear(self):
        print 'Test X linear.'
        bl1,bl2=self.get_random_xlinear(2,4)
        print '%s + %s = %s'%(bl1,bl2,bl1+bl2)
        print '%s - %s = %s'%(bl1,bl2,bl1-bl2)
        data1,data2=bl1(),bl2()
        data1/=2.
        bl1/=2.
        data2*=3.
        bl2*=3
        if isinstance(bl1.spaceconfig,SuperSpaceConfig):
            assert_allclose((data1+data2-0.3*data1+0.5*(data2/3.+data1*2.)).toarray(),(bl1+bl2-0.3*bl1+0.5*(bl2/3.+bl1*2.))(dense=True))
        else:
            assert_allclose(data1+data2-0.3*data1+0.5*(data2/3.+data1*2.),(bl1+bl2-0.3*bl1+0.5*(bl2/3.+bl1*2.))(dense=True))

    def test_all(self):
        self.test_qlinear()
        self.test_xlinear()
        self.test_site_shift()
        self.test_op_fusion()
        self.test_oponbond()

TestLinear().test_all()
