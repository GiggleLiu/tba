from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys

from op import *
from oplib import *
from spaceconfig import *

class TestLinear():
    def __init__(self):
        self.spaceconfig=SuperSpaceConfig([1,2,4,1])

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
                item=Qlinear_ninj(scfg,i=indices[0],j=indices[1],factor=random.random())
            else:
                item=Xlinear(scfg,indices=indices,factor=random.random())
            res.append(item)
        return res

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

#TestLinear().test_xlinear()
Test_config().test_indconfig()
