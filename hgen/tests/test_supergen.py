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
from oplib import op_simple_hopping,op_U,op_simple_onsite
from tba.lattice import Structure
from blockmatrix.blocklib import eigbsh,eigbh,get_blockmarker,tobdmatrix

class FermiHTest(object):
    '''
    Test fermionic hamiltonian generator.
    
    Using standard hexgon as a test.
    '''
    def __init__(self):
        self.setup(1.,0.2,0.,0.2)

    def setup(self,t,t2,U,mu):
        '''setup parameters'''
        self.model_exact=Hex6(t=t,t2=t2,U=U,mu=mu,occ=False)
        self.model_occ=Hex6(t=t,t2=t2,U=U,mu=mu,occ=True)
        scfg=self.model_occ.hgen.spaceconfig
        spaceconfig=SuperSpaceConfig([scfg.nspin,1,scfg.norbit])

    def test_directsolve(self):
        '''test for directly construct and solve the ground state energy.'''
        modelocc=self.model_occ
        rlattice=modelocc.hgen.rlattice
        h_occ=modelocc.hgen.H()
        h_exact=self.model_exact.hgen.H(dense=True)
        Emin=eigsh(h_occ,which='SA',k=1)[0]
        E_excit=eigvalsh(h_exact)
        Emin_exact=sum(E_excit[E_excit<0])
        print E_excit
        print 'The Ground State Energy for hexagon(t = %s, t2 = %s) is %s, tolerence %s.'%(modelocc.t,modelocc.t2,Emin,Emin-Emin_exact)
        assert_almost_equal(Emin,Emin_exact)

    def test_all(self):
        for i in xrange(10):
            t,t2,mu=random.random(3)
            self.setup(t,t2,0,mu)
            self.test_directsolve()

class Hex6(object):
    '''This is a tight-binding model for a hexagon.'''
    def __init__(self,t,t2,mu,U=0,occ=True):
        self.t,self.t2,self.U,self.mu=t,t2,U,mu
        self.occ=occ

        #occupation representation will use <SuperSpaceConfig>, otherwise <SpaceConfig>.
        if self.occ:
            spaceconfig=SuperSpaceConfig([1,6,2,1])
        else:
            spaceconfig=SpaceConfig([1,6,2,1],kspace=False)
            if abs(U)>0: warnings.warn('U is ignored in non-occupation representation.')
        hgen=RHGenerator(spaceconfig=spaceconfig)

        #define the operator of the system
        hgen.register_params({
            't1':self.t,
            't2':self.t2,
            'U':self.U,
            '-mu':self.mu,
            })

        #define a structure and initialize bonds.
        rlattice=Structure(sites=[(0.,0),(0,sqrt(3.)/3),(0.5,sqrt(3.)/2),(1.,0),(1.,sqrt(3.)/3),(0.5,-sqrt(3.)/6)])
        hgen.uselattice(rlattice)

        b1s=rlattice.getbonds(1)  #the nearest neighbor
        b2s=rlattice.getbonds(2)  #the nearest neighbor

        #add the hopping term.
        op_t1=op_simple_hopping(label='hop1',spaceconfig=spaceconfig,bonds=b1s)
        hgen.register_operator(op_t1,param='t1')
        op_t2=op_simple_hopping(label='hop2',spaceconfig=spaceconfig,bonds=b2s)
        hgen.register_operator(op_t2,param='t2')
        op_mu=op_simple_onsite(label='n',spaceconfig=spaceconfig)
        hgen.register_operator(op_mu,param='-mu')

        #add the hubbard interaction term if it is in the occupation number representation.
        if self.occ:
            op_ninj=op_U(label='ninj',spaceconfig=spaceconfig)
            hgen.register_operator(op_ninj,param='U')

        self.hgen=hgen

FermiHTest().test_all()
