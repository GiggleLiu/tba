'''
Tree structured Operator classes.

Operator -> Base class for operators.

    * Bilinear -> Elemental(leaf) Operator in the form of { factor * c^\dag c }.
        * BBilinear -> Bilinear defined on the bond.

    * Qlinear -> Elemental(leaf) Operator in the form of { factor * c^\dag c^\dag c c }.
    * Operator_C -> Elemental(leaf) Operator in the form of { factor * c(^\dag) }.
'''

from numpy import *
from numpy.linalg import norm
from matplotlib import patches
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix,csr_matrix
import copy
import time,pdb,numbers,re

from utils import ind2c,perm_parity
from spaceconfig import SuperSpaceConfig,SpaceConfig

__all__=['OperatorBase','Operator','Bilinear','BBilinear','Operator_C','Qlinear','Qlinear_ninj','Xlinear']

class OperatorBase(object):
    '''
    Base class for operators, Xlinear and it's conbination operator.

    Construct:
        Operator(label,spaceconfig,factor=1.)

    Attributes:
        :spaceconfig: <SpaceConfig>, the configuration of Hamiltonian, an instance of SpaceConfig(or it's derived classes).
        :factor: float, multiplication factor. 
        :suboperators: list of <Xlinear>, sibling-operators.
        :meta_info: str, displayed in the __str__ as bonus infomation.
        :HC: <OperatorBase>, the hermitian conjugate.
        
        Mathematic operations like `op+op`, `op-op`, `op*num`, `num*op`, `op/num`, `-op` are supported.
    '''
    def __init__(self,spaceconfig,factor=1.):
        self.spaceconfig=spaceconfig
        self.factor=factor
        self.meta_info=''

    def __mul__(self,target):
        if isinstance(target,numbers.Number):
            op=copy.copy(self)
            op.factor*=target
            return op
        else:
            raise ValueError('Can not multiply type %s and %s'%(type(self),type(target)))

    def __rmul__(self,target):
        if isinstance(target,numbers.Number):
            return self.__mul__(target)
        else:
            raise ValueError('Can not multiply type %s and %s'%(type(target),type(self)))

    def __imul__(self,target):
        if isinstance(target,numbers.Number):
            self.factor*=target
            return self
        else:
            raise ValueError('Can not multiply type %s and %s'%(type(self),type(target)))

    def __div__(self,target):
        if isinstance(target,numbers.Number):
            op=copy.copy(self)
            op.factor/=target
            return op
        else:
            raise ValueError('Can not divide type %s and %s'%(type(self),type(target)))

    def __idiv__(self,target):
        if isinstance(target,numbers.Number):
            self.factor/=target
            return self
        else:
            raise ValueError('Can not divide type %s and %s'%(type(self),type(target)))

    def __neg__(self):
        op=copy.copy(self)
        op.factor*=-1
        return op

    def __sub__(self,target):
        return self.__add__(-target)

    def __rsub__(self,target):
        return target.__sub__(-self)

    def __isub__(self,target):
        return self.__iadd__(-target)

    def __add__(self,target):
        raise NotImplementedError()

    def __radd__(self,target):
        raise NotImplementedError()

    def __iadd__(self,target):
        raise NotImplementedError()

    def __call__(self,*args,**kwargs):
        raise NotImplementedError()

    @property
    def HC(self):
        '''get the hermition conjugate'''
        raise NotImplementedError()


class Operator(OperatorBase):
    '''
    An operator is consisted of <Xlinear>s.

    Attributes:
        :label: The label of this operator.
        :suboperators: sub-<Xlinear>s.
        :nsubop: integer, number of suboperators.
    '''
    def __init__(self,label,spaceconfig,suboperators=None,factor=1.):
        self.suboperators=[] if suboperators is None else suboperators
        self.label=label
        super(Operator,self).__init__(spaceconfig,factor)

    def __call__(self,param=1.,dense=True):
        '''
        Get a matrix(Hamiltonian) representation of this operator.

        param: 
            bonus multiplication factor of this operator.
        '''
        hndim=self.spaceconfig.hndim
        if dense:
            H=zeros((hndim,hndim),dtype='complex128')
        else:
            H=csr_matrix((hndim,hndim),dtype='complex128')
        for bl in self.suboperators:
            H=H+bl(param=param*self.factor,dense=dense)
        return H

    def __str__(self):
        txt='%s -> %s*('%(self.label,('%.4f'%self.factor).rstrip('0'))
        for i in xrange(len(self.suboperators)):
            operator=self.suboperators[i]
            if i!=0:
                txt+=' + '
            txt+=operator.__str__()
        return txt+self.meta_info+')\n'

    def __repr__(self):
        return '<Operator %s>'%self.label

    def __add__(self,target):
        if isinstance(target,Xlinear):
            newops=self.suboperators+[target/self.factor]
            return Operator(label=self.label,spaceconfig=self.spaceconfig,suboperators=newops,factor=self.factor)
        elif isinstance(target,Operator):
            if self.factor==1 and target.factor==1:
                tops=target.suboperators
            else:
                if self.factor==0:
                    return copy.copy(target)
                tops=[op*(target.factor/self.factor) for op in target.suboperators]
            newops=self.suboperators+tops
            return Operator(label=self.label+'+'+target.label,spaceconfig=self.spaceconfig,suboperators=newops,factor=self.factor)
        elif target==0:
            return copy.copy(self)
        else:
            raise TypeError('Can not add for <Operator> and %s'%target.__class__)

    def __radd__(self,target):
        if isinstance(target,Xlinear):
            return self.__add__(target)
        elif isinstance(target,Operator):
            return target.__add__(self)
        elif target==0:
            return copy.copy(self)
        else:
            raise TypeError('Can not add for %s and <Operator>'%target.__class__)

    def __iadd__(self,target):
        if isinstance(target,Xlinear):
            self.suboperators.append(target/self.factor)
            return self
        elif isinstance(target,Operator):
            if self.factor==1 and target.factor==1:
                tops=target.suboperators
            else:
                tops=[op*(target.factor/self.factor) for op in target.suboperators]
            self.suboperators.extend(tops)
            return self
        else:
            raise TypeError('Can not add for <Operator> and %s'%target.__class__)

    @property
    def nsubop(self):
        '''Number of suboperators'''
        return len(self.suboperators)

    def addsubop(self,subop,weight=None):
        '''
        Add a sub-operator to this operator.

        subop: 
            The operator to be added, and <Operator>(or it's derivative classes) instance.
        weight: 
            The weight of suboperator.
        '''
        if weight!=None:
            subop.factor*=weight
        self.suboperators.append(subop)

    def Hk(self,k,param=1.):
        '''
        Get a matrix(Hamiltonian) representation of this operator.

        k: 
            Momentum vector or a mesh of momentum vector,
        param: 
            Bonus parameter as a multiplication factor for this operator.
        '''
        if ndim(k)==1:
            H=zeros([self.spaceconfig.hndim,self.spaceconfig.hndim],dtype='complex128')
        elif ndim(k)>1:
            H=zeros(list(k.shape[:-1])+[self.spaceconfig.hndim]*2,dtype='complex128')
        else:
            raise Exception('Error','Invalid k vector!')
        for operator in self.suboperators:
            H+=operator.Hk(k=k,param=param*self.factor)
        return H

    @property
    def HC(self):
        '''
        Get the hermitian conjugate.
        '''
        return Operator(label=self.label,spaceconfig=self.spaceconfig,factor=conj(self.factor),suboperators=[op.HC for op in self.suboperators])

class Xlinear(OperatorBase):
    '''
    Xlinear class(N body operator).

    Construct:
        Qlinear(spaceconfig,indices,factor=1.)

    Attributes:
        :indices: N indices for Xlinear. sorted in the normal order "c+c+...cc...".
        *see also <Operator> class.*

    Note:
        * indices are aranged in the normal order, dag before non-dag.
    '''
    def __init__(self,spaceconfig,indices,indices_ndag=None,factor=1.):
        super(Xlinear,self).__init__(spaceconfig=spaceconfig,factor=factor)
        self.indices=array(indices)
        if indices_ndag is None: indices_ndag=len(indices)/2
        self.indices_ndag=indices_ndag

    def __add__(self,target):
        if isinstance(target,Operator):
            return target.__radd__(self)
        elif isinstance(target,Xlinear):
            return Operator(label='OP',spaceconfig=self.spaceconfig,suboperators=[self,target])
        elif isinstance(target,numbers.Number) and target==0:
            return copy.copy(self)
        else:
            raise TypeError('Can not add %s with %s.'%(self.__class__,target))

    def __radd__(self,target):
        if isinstance(target,OperatorBase):
            return target.__add__(self)
        elif isinstance(target,numbers.Number) and target==0:
            return copy.copy(self)
        else:
            raise TypeError('Can not add %s with %s.'%(target.__class__,self.__class__))

    def __getattr__(self,name):
        mch=re.match('^index(\d+)$',name)
        if mch:
            n=int(mch.group(1))
            if n<self.nbody:
                return self.indices[n]
            else:
                raise ValueError('Index exceeded %s/%s.'%(n,self.nbody))
        else:
            raise AttributeError('Can not find attribute %s'%name)

    def __str__(self):
        txt=('%.4f'%self.factor).rstrip('0')+'*'
        inds=[self.spaceconfig.ind2c(ind) for ind in self.indices]
        c=['C_']*self.nbody
        for i in xrange(self.nbody):
            if self.spaceconfig.nspin==2:
                c[i]+=('dn' if inds[i][1] else 'up')
            if self.spaceconfig.natom>=2:
                #c[i]+=chr(65+inds[i][2])
                c[i]+=str(inds[i][2])
            if self.spaceconfig.norbit>=2:
                c[i]+='o'+str(inds[i][3])
            if i<self.indices_ndag:
                c[i]+='+'
        txt+=''.join(c)+self.meta_info
        return txt

    def __repr__(self):
        return self.__str__()

    def __call__(self,param=1.,dense=False,**kwargs):
        '''
        Refer Operator.__call__
        '''
        if len(unique(self.indices))!=self.nbody:
            raise NotImplementedError('Indices are not allowed to overlap at this moment.')
        indices1,indices2,ne=self.spaceconfig.indices_occ(unocc=self.indices[:self.indices_ndag],occ=self.indices[self.indices_ndag:],count_e=True,getreverse=True)
        parity=perm_parity(argsort(self.indices))
        sign=1-2*((sum(ne,axis=0)+parity)%2)
        data=self.factor*param*sign
        if not dense:
            h=coo_matrix((data,(indices1,indices2)),shape=[self.spaceconfig.hndim]*2)
            return h
        else:
            H=zeros(shape=[self.spaceconfig.hndim]*2,dtype='complex128')
            H[indices1,indices2]+=data
            return H

    @property
    def nbody(self):
        '''N body operator.'''
        return len(self.indices)

    @property
    def HC(self):
        '''
        Get the hermitian conjugate.
        '''
        return self.__class__(indices=self.indices[::-1],spaceconfig=self.spaceconfig,indices_ndag=self.nbody-self.indices_ndag,factor=conj(self.factor))

class Operator_C(Xlinear):
    '''
    Anilation(Creation if dag set True) Operator class.

    Construct:
        Operator_C(spaceconfig,index,dag=False,factor=1.)

    Attributes:
        :index: The index(in spaceconfig) of this c operator.
        :dag: c^\dag if True else c.
        *see also <Operator> class*
    '''
    def __init__(self,spaceconfig,index,dag=False,factor=1.):
        super(Operator_C,self).__init__(spaceconfig=spaceconfig,indices=[index],factor=factor)
        self.dag=dag

    @property
    def index(self):
        '''The index of this c operator.'''
        return self.indices[0]

    @property
    def HC(self):
        '''
        Get the hermitian conjugate.
        '''
        return Operator_C(spaceconfig=self.spaceconfig,index=self.index,dag=not self.dag,factor=conj(self.factor))

    def __str__(self):
        txt=('%.4f'%self.factor).rstrip('0')+'*'
        inds=self.spaceconfig.ind2c(self.index)
        c='C_'
        if self.spaceconfig.nspin==2:
            c+=('dn' if inds[1] else 'up')
        if self.spaceconfig.natom>=2:
           # c+=chr(65+inds[2])
            c+=str(inds[2])
        if self.spaceconfig.norbit>=2:
            c+='o'+str(inds[3])
        if self.dag:
            c+='+'
        txt+=c+self.meta_info
        return txt

    def __call__(self,param=1.,dense=True):
        '''
        Refer Operator.__call__ for useage.
        '''
        param=param*self.factor
        if isinstance(self.spaceconfig,SuperSpaceConfig):
            nflv=self.spaceconfig.nflv
            ind1=self.index
            occ,unocc=[ind1],[]
            if self.dag:
                occ,unocc=unocc,occ
            #notice that this state is used as index2, and the reverse is index1
            index2,index1,count_e=self.spaceconfig.indices_occ(occ=occ,unocc=unocc,getreverse=True,count_e=True)

            #coping the sign problem
            sparam=array([param]*len(index1))
            #ind1 passes eletrons [0,ind1)
            #the sign is equal to the electrons site<ind1
            sparam[count_e[0]%2==1]*=-1
            if dense:
                h=zeros(shape=[self.spaceconfig.hndim]*2,dtype='complex128')
                h[index1,index2]=sparam
            else:
                h=coo_matrix((sparam,(index1,index2)),shape=[self.spaceconfig.hndim]*2,dtype='complex128')
            return h
        else:
            print 'You can not transform this operator into Hamiltonian with spaceconfig other than SuperSpaceConfig.'
            pdb.set_trace()

class Bilinear(Xlinear):
    '''
    Bilinear class, it is a special Operator with only one bilinear. It consititue the lowest layer of Operator tree.

    H = C1^dag C2

    Construct:
        Bilinear(spaceconfig,index1,index2,factor=1.)

    Attributes:
        :index1/index2: Arrays indicating sub-block of hamiltonian H[index1,index2](readonly).
        *see also <Operator> class*
    '''
    def __init__(self,spaceconfig,index1,index2,factor=1.,indices_ndag=1):
        super(Bilinear,self).__init__(spaceconfig=spaceconfig,indices=[index1,index2],factor=factor,indices_ndag=indices_ndag)

    def __str__(self):
        txt=('%.4f'%self.factor).rstrip('0')+'*'
        inds=[self.spaceconfig.ind2c(self.index1),self.spaceconfig.ind2c(self.index2)]
        c=['C_','C_']
        for i in xrange(2):
            if self.spaceconfig.nspin==2:
                c[i]+=('dn' if inds[i][1] else 'up')
            elif self.spaceconfig.smallnambu:
                c[i]+=('dn' if inds[i][0] else 'up')
            if self.spaceconfig.natom>=2:
                c[i]+=str(inds[i][2])
            if self.spaceconfig.norbit>=2:
                c[i]+='o'+str(inds[i][3])
            if self.spaceconfig.nnambu==2:
                c[i]+='+' if (i==inds[i][0]) else ''
            else:
                c[i]+='+' if (i<self.indices_ndag) else ''
        txt+=''.join(c)+self.meta_info
        return txt

    def __call__(self,param=1.,dense=True):
        '''
        Refer Operator.__call__ for useage.
        '''
        if isinstance(self.spaceconfig,SuperSpaceConfig):
            nflv=self.spaceconfig.nflv
            ind1,ind2=self.index1,self.index2
            if ind2!=ind1:
                return super(Bilinear,self).__call__(param,dense)
            else:
                if self.index1<nflv:
                    index2=index1=self.spaceconfig.indices_occ(occ=[self.index1])[0]
                else:
                    index2=index1=self.spaceconfig.indices_occ(unocc=[self.index1])[0]
                sparam=(param*self.factor)*ones(len(index1))
            if not dense:
                h=coo_matrix((sparam,(index1,index2)),shape=[self.spaceconfig.hndim]*2,dtype='complex128')
                h=h.tocsr()
            else:
                h=zeros(shape=[self.spaceconfig.hndim]*2,dtype='complex128')
                h[index1,index2]=sparam
            return h
        else:
            #for hamiltonian in sencond quantized representation.
            H=zeros([self.spaceconfig.hndim]*2,dtype='complex128')
            H[self.index1,self.index2]=self.factor*param
            return H

    @property
    def index1(self):
        '''Get the first index.'''
        return self.indices[0]

    @property
    def index2(self):
        '''Get the first index.'''
        return self.indices[1]

    @property
    def issc(self):
        '''
        Is a c+c+ type bilinear if true.
        '''
        return (self.index2>=self.spaceconfig.nflv) & (self.index1<self.spaceconfig.nflv)

    @property
    def HC(self):
        '''
        Get the hermitian conjugate.
        '''
        return Bilinear(spaceconfig=self.spaceconfig,index1=self.index2,index2=self.index1,factor=conj(self.factor))

    def Hk(self,k,param=1.):
        '''
        Get Hamiltonian defined on k(mesh).

        k:
            The k-vector, or mesh of k vectors.
        param:
            The multiplication factor.
        '''
        param=param*self.factor
        index1,index2=self.index1,self.index2
        if ndim(k)>=1:
            H=zeros(list(k.shape[:-1])+[self.spaceconfig.hndim]*2,dtype='complex128')
            H[...,index1,index2]+=param
            return H
        else:
            raise ValueError('Shape of k not correct!')


class BBilinear(Bilinear):
    '''
    Bilinear defined on a bond.

    Construct:
        BBilinear(spaceconfig,index1,index2,bondv,factor=1.)

    Attributes:
        :bondv: The bond vector.

    *see also <Bilinear> class*
    '''
    def __init__(self,spaceconfig,index1,index2,bondv,factor=1.):
        super(BBilinear,self).__init__(spaceconfig=spaceconfig,index1=index1,index2=index2,factor=factor)
        self.bondv=array(bondv)
        self.meta_info='' if norm(self.bondv)<1e-5 else ' <%s>'%(','.join(['%.2f'%b for b in self.bondv]))

    def Hk(self,k,param=1.):
        '''
        Get Hamiltonian defined on k(mesh).

        k:
            The k-vector, or mesh of k vectors.
        param:
            The multiplication factor.
        '''
        return super(BBilinear,self).Hk(k,param*exp(1j*(k*self.bondv).sum(axis=-1)))

    @property
    def HC(self):
        '''
        Get the hermitian conjugate.
        '''
        return BBilinear(spaceconfig=self.spaceconfig,index1=self.index2,index2=self.index1,bondv=-self.bondv,factor=conj(self.factor))

class Qlinear(Xlinear):
    '''
    Qlinear class(4 body operator), unlike bilinear, qlinear does not support factor!

    Construct:
        Qlinear(spaceconfig,indices,factor=1.)

    *see also <Operator> class.*
    '''
    def __init__(self,spaceconfig,indices,indices_ndag=2,factor=1.):
        if not isinstance(spaceconfig,SuperSpaceConfig):
            raise TypeError('Using wrong type of spaceconfig, <SuperSpaceConfig> is required!')
        super(Qlinear,self).__init__(spaceconfig=spaceconfig,indices=indices,indices_ndag=indices_ndag,factor=factor)

    def __getattr__(self,name):
        if name=='index1':
            return self.indices[0]
        elif name=='index2':
            return self.indices[1]
        elif name=='index3':
            return self.indices[2]
        elif name=='index4':
            return self.indices[3]
        elif name=='nbody':
            return len(self.indices)
        else:
            return super(Qlinear,self).__getattr__(name)

class Qlinear_ninj(Qlinear):
    '''
    V*ni*nj type operator. e.g. hubbard U and coloumb V.

    Construct:
        Qlinear_ninj(spaceconfig,i,j,factor=1.)

    Attributes:
        *see Qlinear*
    '''
    def __init__(self,spaceconfig,i,j,factor=1.):
        if i==j:
            raise ValueError('Operator ni^2 should be reduced to bilinear!')
        #note that it will take a '-' sign for (ni,nj) to transform to normal order
        factor*=-1
        super(Qlinear_ninj,self).__init__(spaceconfig=spaceconfig,indices=[i,j,i,j],factor=factor)

    def __getattr__(self,name):
        if name=='i':
            return self.indices[0]
        elif name=='j':
            return self.indices[1]
        else:
            return super(Qlinear_ninj,self).__getattr__(name)

    def __call__(self,param=1.,dense=False,**kwargs):
        '''
        Refer Operator.__call__
        '''
        #first get all posible distribution of eletrons with site2 and site1 both occupied/unoccupied(reverse).
        indices=self.spaceconfig.indices_occ([self.i,self.j])[0]
        #set the H matrix, note the '-' sign befor self.factor.
        data=param*(-self.factor)*ones(indices.shape,dtype='complex128')
        if not dense:
            h=coo_matrix((data,(indices,indices)),shape=[self.spaceconfig.hndim]*2)
            return h
        else:
            H=zeros(shape=[self.spaceconfig.hndim]*2,dtype='complex128')
            H[indices,indices]+=data
            return H

    @property
    def HC(self):
        '''
        Get the hermitian conjugate.
        '''
        res=copy.copy(self)
        res.factor=conj(res.factor)
        return res
