from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb

from utils import perm_parity
from op import *
from oplib import *
from spaceconfig import *
sys.path.insert(0,'../lattice')
from bond import Bond

SpaceConfig.SPACE_TOKENS=['nambu','atom','spin','orbit']
class TestLinear():
    def __init__(self):
        self.spaceconfig=SuperSpaceConfig([1,4,2,1])

    def test_qlinear(self):
        ndim=self.spaceconfig.ndim
        for i in xrange(10):
            indices=random.randint(0,ndim,4)
            if len(unique(indices))!=4:
                continue
            factor=random.random()
            l1=Qlinear(self.spaceconfig,indices=indices,factor=factor)
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
        shift=2
        new_spaceconfig=SuperSpaceConfig([1,7,2,self.spaceconfig.norbit])
        op=self.get_random_operator(n=5,nbody=2)
        nop=site_shift(op,shift,new_spaceconfig)
        op2=site_shift(nop,-shift,self.spaceconfig)
        print 'old-operator',op
        print 'shifted-operator',nop
        assert_(op.factor==op2.factor)
        for o1,o2 in zip(op.suboperators,op2.suboperators):
            assert_(o1.factor==o2.factor)
            assert_(o1.factor==o2.factor)
            assert_allclose(o1.indices,o2.indices)

    def test_op_fusion(self):
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
        natom=self.spaceconfig.natom
        dim1=self.spaceconfig.ndim/natom
        atom_axis=self.spaceconfig.get_axis('atom')
        #test on-site
        mats=[0.5*identity(dim1)]*natom
        bonds=[Bond(zeros(2),i,i) for i in xrange(natom)]
        op=op_on_bond(label='E',spaceconfig=self.spaceconfig,mats=mats,bonds=bonds)
        print 'Get operator,',op
        for i in xrange(natom):
            for j in xrange(dim1):
                assert_(all(self.spaceconfig.ind2c(op.suboperators[dim1*i+j].indices)[:,atom_axis]==ones(2)*i))
                assert_(all(op.suboperators[dim1*i+j].factor==0.5))
        #test for nearest hopping
        mats=[0.5*identity(dim1)]*(natom-1)
        bonds=[Bond(array([1,0]),i,i+1) for i in xrange(natom-1)]
        op=op_on_bond(label='T',spaceconfig=self.spaceconfig,mats=mats,bonds=bonds)
        print 'Get operator,',op
        for i in xrange(natom-1):
            for j in xrange(dim1):
                assert_(all(self.spaceconfig.ind2c(op.suboperators[dim1*i+j].indices)[:,atom_axis]==array([i,i+1])))
                assert_(all(op.suboperators[dim1*i+j].factor==0.5))

    def test_xlinear(self):
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

class Test_config():
    def __init__(self):
        self.sscfg=SuperSpaceConfig([2,5,2])
        self.sscfg_ne=SuperSpaceConfig([2,5,2],ne=10)
        self.scfgs=[self.sscfg,self.sscfg_ne]

    def test_indconfig(self):
        '''test for ind-config parsing.'''
        for scfg in self.scfgs[:2]:
            print 'Running test for ',scfg
            #test for 1D array of ind.
            ind=random.randint(0,scfg.hndim,10)
            config=scfg.ind2config(ind)
            ind2=scfg.config2ind(config)
            assert_(all(ind2==ind))
            #test for integer ind.
            ind=random.randint(0,scfg.hndim)
            config=scfg.ind2config(ind)
            ind2=scfg.config2ind(config)
            assert_(ind2==ind)

def test_perm():
    def perm(N,times):
        t=0
        base=arange(N)
        for i in xrange(times):
            ri=random.randint(0,N-1)
            base[ri:ri+2]=base[ri+1],base[ri]
        return base
    for k in xrange(10):
        print 'running sample %s'%k
        N=random.randint(2,20)
        times=random.randint(0,50)
        assert_(perm_parity(perm(N,times))==times%2)

#test_perm()
TestLinear().test_qlinear()
TestLinear().test_xlinear()
TestLinear().test_site_shift()
TestLinear().test_op_fusion()
TestLinear().test_oponbond()
Test_config().test_indconfig()
