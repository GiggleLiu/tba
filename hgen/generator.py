'''
Hamiltonian generator classes.
'''
from numpy import *
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.linalg import solve,eigvalsh
from matplotlib.pyplot import *
import os,time,re,pdb

from multithreading import mpido

__all__=['HGeneratorBase','KHGenerator','RHGenerator']

class HGeneratorBase(object):
    '''
    This is a Basic Model contains only the most elementary lattice and energy informations.

    Construct:
        HGeneratorBase(spaceconfig)

    Attributes:
        :spaceconfig: Space configuration(<SpaceConfig> instance).
        :operatordict:  A dict of operators used in this model.
        :params: The parameters used in this model, to set the hamiltonian orz ...
        :op_param_map: The mapping of operator and parameters.
    '''
    def __init__(self,spaceconfig):
        self.spaceconfig=spaceconfig
        self.operatordict={}
        self.params={'one':1}
        self.op_param_map={}

    def __str__(self):
        opstr=''
        for opname in self.operatordict:
            paramname=self.op_param_map[opname]
            opstr+='%s*<%s>, %s=%s\n'%(paramname,opname,paramname,self.params[paramname])
        return '''%s
---------------
%s
%s'''%(self.__class__,self.spaceconfig.__str__(),opstr)

    def register_operator(self,operator,param='one'):
        '''
        Register/modify a operator.

        operator:
            the operator.
        param: 
            the param label correspond to this operator. default is nullparam 1.
        '''
        self.op_param_map[operator.label]=param
        self.operatordict[operator.label]=operator

    def register_params(self,paramdict):
        '''
        Register/Modify params.

        paramdict:
            a dict of Param instances.
        '''
        self.params.update(paramdict)

    def uselattice(self,lattice):
        '''
        Use the lattice.

        lattice:
            the target lattice.
        '''
        self.rlattice=lattice

    def H(self,params={},dense=False):
        '''
        Get the hamiltonian.

        params:
            asign params instead of registerd ones.
        dense:
            use dense matrix or sparse(default).
        '''
        hndim=self.spaceconfig.hndim
        if not dense:
            h=csr_matrix((hndim,hndim),dtype='complex128')
        else:
            h=zeros((hndim,hndim))
            h[...]=0.

        for op in self.operatordict.keys():
            operator=self.operatordict[op]
            paramname=self.op_param_map[operator.label]
            param=params.get(paramname,self.params.get(paramname))
            if param==None or abs(param)<1e-8:
                #print 'Warning! Small parameter ',param,'! Ignored'
                continue
            h=h+operator(param=param,dense=dense)
        return h

class KHGenerator(HGeneratorBase):
    '''
    HGenerator defined on a k-mesh(specified by self.uselattice).

    Construct
    ------------------
    RHGenerator(spaceconfig)

    Parameters
    -------------------
    propergauge: 
        use proper gauge if True.(propergauge: keep k-dependant phases exp(ikr) within a sublattice the same.)
    '''

    def __init__(self,spaceconfig,propergauge,**kwargs):
        super(KHGenerator,self).__init__(spaceconfig,**kwargs)
        self.propergauge=propergauge

    def H(self,*args,**kwargs):
        '''
        alias for gethkmesh.
        '''
        return self.gethkmesh(*args,**kwargs)

    def gethkmesh(self,kmesh=None,append=False,params={}):
        '''
        Get hk mesh.

        kmesh: 
            specify the k-mesh.
        append: 
            append one if True.
        params:
            asign params instead of registered ones.
        '''
        if kmesh is not None:
            hkmesh=mpido(inputlist=reshape(kmesh,(-1,kmesh.shape[-1])),func=lambda k:self.Hk(k))
            return hkmesh
        hkmesh=0
        for op in self.operatordict.keys():
            operator=self.operatordict[op]
            paramname=self.op_param_map[operator.label]
            param=params.get(paramname,self.params.get(paramname))
            if param==None or abs(param)<1e-8:
                #print 'Warning! Small parameter ',param,'! Ignored'
                continue
            hkmesh+=param*operator.Hk(self.kspace.kmesh)
        if self.propergauge and self.spaceconfig.natom>1:
            hkmesh=self.properize(hkmesh)
        return hkmesh

    def Hk(self,k,params={}):
        '''
        Return the H(k).

        k:
            the momentum.
        params:
            asign params instead of registerd ones.
        '''
        hk=zeros([self.spaceconfig.hndim,self.spaceconfig.hndim],dtype='complex128')
        for op in self.operatordict.keys():
            operator=self.operatordict[op]
            paramname=self.op_param_map[operator.label]
            param=params.get(paramname,self.params.get(paramname))
            if param==None or abs(param*operator.factor)<1e-8:
                #print 'Warning! Small parameter ',param,'! Ignored'
                continue
            hk=hk+operator.Hk(k=k,param=param)
        if self.propergauge and self.spaceconfig.natom>1:
            hk[...]=self.properize(hk,k)
        return hk

    def uselattice(self,lattice):
        '''
        Use the lattice.

        lattice:
            the target lattice.
        '''
        super(KHGenerator,self).uselattice(lattice)
        #self.kspace=lattice.getReciprocalLattice()
        self.kspace=lattice.kspace
        if self.propergauge:
            catoms=self.rlattice.catoms[self.spaceconfig.atomindexer]
            self.propermesh=array([[diag(exp([-1j*dot(self.kspace.kmesh[i,j],catoms[l]) for l in xrange(self.spaceconfig.ndim)])) for j in xrange(self.kspace.N[1])] for i in xrange(self.kspace.N[0])])

    def properize(self,inmesh,k=None,reverse=False):
        '''
        Properize inmesh, canceling the phase difference induced by sub-lattice position difference.

        inmesh:
            it should be a Hamiltonian.
        k:
            the momentum
        '''
        dim=inmesh.shape[-1]
        if ndim(inmesh)==4:
            propermesh=self.propermesh[:,:,:dim,:dim]
            propermeshH=conj(propermesh)  #note that P is diagonal
            if reverse:
                propermesh,propermeshH=propermeshH,propermesh
            return array([[dot(dot(propermeshH[i,j],inmesh[i,j]),propermesh[i,j]) for j in xrange(self.kspace.N[1])] for i in xrange(self.kspace.N[0])])
        elif ndim(inmesh)==2:
            pmat=diag(exp([-1j*dot(k,self.rlattice.catoms[self.spaceconfig.atomindexer[l]]) for l in xrange(dim)]))
            pmatH=conj(pmat)
            if reverse:
                pmat,pmatH=pmatH,pmat
            return dot(pmatH,dot(inmesh,pmat))

    def antiproperize(self,*args):
        '''
        Antiproperize inmesh, the inverse process as properize.

        inmesh:
            it should be a Hamiltonian.
        k:
            the momentum
        '''
        return self.properize(reverse=True,*args)

    def Gk(self,w,k,tp='r',geta=1e-2):
        '''
        Get the Green's function

        w: 
            the energy level
        k: 
            momentum k.
        tp:
            type of green's function.

            'r': retarded(default).

            'a': advanced
        geta:
            smearing factor. default is 1e-2.
        '''
        return H2G(self.Hk(k),w,tp,geta)

    def Ek(self,k):
        '''
        Get the E(k).

        k:
            the momentum.
        '''
        ek=eigvalsh(self.Hk(k))
        return ek

class RHGenerator(HGeneratorBase):
    '''
    HGenerator defined on a realspace-mesh(specified by self.uselattice) assuming no translational invarience.

    Construct
    ------------------
    RHGenerator(spaceconfig)
    '''

    def __init__(self,spaceconfig):
        super(RHGenerator,self).__init__(spaceconfig)

    def uselattice(self,lattice):
        '''
        Use the lattice.

        lattice:
            the target lattice.
        '''
        self.rlattice=lattice
        if not lattice.bonds_initialized:
            self.rlattice.initbonds(3)

