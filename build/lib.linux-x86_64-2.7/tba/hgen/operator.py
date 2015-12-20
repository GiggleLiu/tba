#!/usr/bin/python
'''
Tree structured Operator classes.

Operator -> Base class for operators.

    * Bilinear -> Elemental(leaf) Operator in the form of { factor * c^\dag c }.
        * BBilinear -> Bilinear defined on the bond.

    * Qlinear -> Elemental(leaf) Operator in the form of { factor * c^\dag c^\dag c c }.
    * Operator_C -> Elemental(leaf) Operator in the form of { factor * c(^\dag) }.
'''
__all__=['Operator','Bilinear','BBilinear','Operator_C','Qlinear','Qlinear_ninj','formop']
from utils import ind2c
from numpy.linalg import norm
from spaceconfig import SuperSpaceConfig,SpaceConfig
from matplotlib import patches
from matplotlib.collections import LineCollection
from numpy import *
from scipy.sparse import coo_matrix,csr_matrix
from copy import deepcopy
import time,pdb

class Operator(object):
    '''
    Operators are basic constituents of a Hamiltonian.

    A quantum operator can be consist of one or more suboperators(or it's drived classed, like (B)Bilinear/Qlinear).

    Construct
    -------------------
    Operator(label,spaceconfig,factor=1.)

    Attributes
    -------------------
    label:
        The label of this operator.
    spaceconfig:
        The configuration of Hamiltonian, an instance of SpaceConfig(or it's derived classes).
    factor:
        Multiplication factor. 
    father/weight:
        Superoperator and Multiplication factor as a Node operator.
    suboperators:
        Sibling-operators.
    meta_info:
        Displayed in the __str__ as bonus infomation.
        
    Note
    ----------------
        Operators are structured in the form of Tree, this is what father, suboperators and weight are for.
        The leaves of this Tree must be (B)Bilinear, Operator_C or Qlinear instance.
        
        The following Tree Operations are available:

            * addsubop -> add a sibling.
            * ravel    -> flatten this tree into root-leaves(or star) structure.

        For the difference between `factor` and `weight`, `factor` is more like an intrinsic property of this operator compared with the
        extrinsic multiplication factor `weight` which only make sence as a sibling.

        Mathematic operations like `op+op`, `op-op`, `op*num`, `num*op`, `op/num`, `-op` are supported.
    '''
    def __init__(self,label,spaceconfig,factor=1.):
        if label is None:
            label=id(self)
        self.label=label
        self.spaceconfig=spaceconfig
        self.factor=factor
        self.suboperators=[]
        self.father=None
        self.weight=1.
        self.meta_info=''

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
        for operator in self.suboperators:
            H=H+operator(param=param*self.factor*self.weight,dense=dense)
        return H

    def __str__(self):
        txt=(self.label+' = ' if self.father==None else '')+'%s*('%(self.factor*self.weight,)
        for i in xrange(len(self.suboperators)):
            operator=self.suboperators[i]
            if i!=0:
                txt+=' + '
            txt+=operator.__str__()
        return txt+self.meta_info+')\n'

    def __mul__(self,target):
        if isinstance(target,Operator):
            pass
        elif isinstance(x, (int, long, float, complex)):
            op=deepcopy(self)
            op.weight*=target
            return op
        else:
            raise ValueError('Target should be type <Num> or <Operator>, but got %s'%type(target))

    def __rmul__(self,target):
        if isinstance(target,Operator):
            return target.__mul__(self)
        elif isinstance(x, (int, long, float, complex)):
            op=deepcopy(self)
            op.weight*=target
            return op
        else:
            raise ValueError('Target should be type <Num> or <Operator>, but got %s'%type(target))

    def __imul__(self,target):
        if isinstance(x, (int, long, float, complex)):
            self.weight*=target
            return self
        else:
            raise ValueError('Target should be type <Num>, but got %s'%type(target))

    def __add__(self,target):
        op=deepcopy(self)
        newops=deepcopy(target.suboperators)
        factor=target.factor  #weight is extrinsic
        for cop in newops:
            cop.factor*=factor
        op.suboperators+=newops
        return op

    def __radd__(self,target):
        return target.__add__(self)

    def __iadd__(self,target):
        newops=deepcopy(target.suboperators)
        factor=target.factor  #weight is extrinsic
        for cop in newops:
            cop.factor*=factor
        self.suboperators+=newops
        return self

    def __div__(self,target):
        op=deepcopy(self)
        op.weight/=target
        return op

    def __idiv__(self,target):
        self.weight/=target
        return self

    def __sub__(self,target):
        op=deepcopy(self)
        newops=deepcopy(target.suboperators)
        factor=-target.factor  #weight is extrinsic
        for cop in newops:
            cop.factor*=factor
        op.suboperators+=newops
        return op

    def __rsub__(self,target):
        return target.__sub__(self)

    def __isub__(self,target):
        newops=deepcopy(target.suboperators)
        factor=-target.factor  #weight is extrinsic
        for cop in newops:
            cop.factor*=factor
        self.suboperators+=newops
        return self

    def __neg__(self):
        op=deepcopy(self)
        op.weight*=-1
        return op

    def addsubop(self,subop,weight=None):
        '''
        Add a sub-operator to this operator.

        subop: 
            The operator to be added, and <Operator>(or it's derivative classes) instance.
        weight: 
            The weight of suboperator.
        '''
        if weight!=None:
            subop.weight=weight
        subop.father=self
        self.suboperators.append(subop)

    def ravel(self,factor=1.):
        '''
        ravel the operator into a set of elementary bilinears instead of a 'tree-hierachy'.

        factor: 
            The bonus multiplication factor.
        '''
        bllist=[]
        for op in self.suboperators:
            bllist+=op.ravel(factor=self.weight*self.factor*factor)
        return bllist

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
            H+=operator.Hk(k=k,param=param*self.factor*self.weight)
        return H

    def show_structure(self,offset=(0,0),width=10.,ax=None):
        '''
        Diplay The Graphical structure of this operator.
        '''
        facecolor='k'
        edgecolor='k'
        if ax is None: ax=gca()
        nsub=len(self.suboperators)
        is_leaf=nsub==0
        is_root=self.father is None
        r=min(0.2,width/3.)
        a=min(0.4,width/1.5)
        subwidth=self.width/nsub
        if self.is_leaf:
            ax.add_patch(patches.Circle(xy=(offset[0],offset[1]),radius=r,facecolor=facecolor,edgecolor=edgecolor))
        else:
            ax.add_patch(patches.Rectangle(xy=(offset[0],offset[1]),width=a,height=a,facecolor=facecolor,edgecolor=edgecolor))
        offsety=offset[1]+width*2
        for i,op in enumerate(self.suboperators):
            offsetx=subwidth*(i+0.5)
            offseti=(offsetx,offsety)
            lc.append([offset,offseti])
            op.show_structure(offseti,width=subwidth,ax=ax)
        lc=LineCollection(lc,color=edgecolor)
        ax.add_collection(lc)
        if is_root:
            ax.autoscale()
            axis('equal')

class Operator_C(Operator):
    '''
    Anilation(Creation if dag set True) Operator class.

    Construct
    ------------------
    Operator_C(spaceconfig,index,dag=False,factor=1.,label=None), label is the `id` of this instance if not specified.

    Attributes
    ------------------
    index:
        The index(in spaceconfig) of this c operator.
    dag:
        c^\dag if True else c.

    *see also <Operator> class*
    '''
    def __init__(self,spaceconfig,index,dag=False,factor=1.,label=None):
        if label==None:
            label='cdag' if dag else 'c'
        super(Operator_C,self).__init__(label=label,spaceconfig=spaceconfig,factor=factor)
        self.index=array(arange(spaceconfig.ndim)[index]).reshape([-1])
        self.dag=dag

    def __str__(self):
        txt=''
        for k in xrange(self.nind):
            if type(self.factor)==ndarray:
                factor=self.factor[0,0]
            else:
                factor=self.factor
            txt+='%s*('%(factor*self.weight,)
            inds=self.spaceconfig.ind2c(self.index1[k])
            c='C_'
            if self.spaceconfig.nspin==2:
                c+=('dn' if inds[1] else 'up')
            if self.spaceconfig.natom>=2:
                c+=chr(65+inds[2])
            if self.spaceconfig.norbit>=2:
                c+='o'+str(inds[3])
            if self.dag:
                c+='+'
            txt+=c+self.meta_info
            if k!=self.nind-1:
                txt+=' + '
        return txt+')'

    def __getattr__(self,name):
        return super(Operator_C,self).__getattr__(name)

    def __call__(self,param=1.,dense=True):
        '''
        Refer Operator.__call__ for useage.
        '''
        param=param*self.factor*self.weight
        if isinstance(self.spaceconfig,SuperSpaceConfig):
            nflv=self.spaceconfig.nflv
            if self.nind!=1:
                print 'Error, multiple indices not supported yet for occupation representaion.'
                pdb.set_trace()
            ind1=self.index[0]
            occ,unocc=[ind1],[]
            if self.dag:
                occ,unocc=unocc,occ
            #notice that this state is used as index2, and the reverse is index1
            index2,index1,count_e=self.spaceconfig.indices_occ(occ=occ,unocc=unocc,getreverse=True,count_e=True)

            #coping the sign problem
            sparam=array([param]*len(index1))
            #ind1 passes eletrons [0,ind1)
            #the sign is equal to the electrons site<ind1
            sparam[count_e%2==1]*=-1
            if dense:
                h=zeros(shape=[self.spaceconfig.hndim]*2,dtype='complex128')
                h[index1,index2]=sparam
            else:
                h=coo_matrix((sparam,(index1,index2)),shape=[self.spaceconfig.hndim]*2,dtype='complex128')
            return h
        else:
            print 'You can not transform this operator into Hamiltonian with spaceconfig other than SuperSpaceConfig.'
            pdb.set_trace()

    @property
    def nind(self):
        '''
        The number of indices.
        '''
        return len(self.index)

    @property
    def multiple_indexing(self):
        '''
        True if  this operator uses multiple indexing.
        '''
        return self.nind>1

    def ravel(self,factor=1.,**kwargs):
        '''
        Ravel an Bilinear to elementary ones without multiple indexing.

        factor:
            the bonus factor.
        '''
        clist=[]
        ind1=arange(self.spaceconfig.ndim)[self.index]
        factor=factor*self.factor*self.weight
        for i in xrange(len(ind1)):
            f=factor[i] if ndim(factor)>0 else factor
            if abs(f)>1e-5:
                clist.append(type(self)(spaceconfig=self.spaceconfig,index=ind1[i],dag=self.dag,factor=f,**kwargs))
        return clist

class Bilinear(Operator):
    '''
    Bilinear class, it is a special Operator with only one bilinear. It consititue the lowest layer of Operator tree.

    Construct
    ------------------
    Bilinear(spaceconfig,index1,index2,factor=1.,label=None), label is the `id` of this instance if not specified.

    Attributes
    ------------------
    index1/index2:
        Arrays indicating sub-block of hamiltonian H[index1,index2].

    *see also <Operator> class*
    '''
    def __init__(self,spaceconfig,index1,index2,factor=1.,label=None):
        if label is None: label=id(self)
        super(Bilinear,self).__init__(label=label,spaceconfig=spaceconfig,factor=factor)
        self.indices=(array(arange(spaceconfig.nflv*2)[index1]).reshape([-1]),array(arange(spaceconfig.nflv*2)[index2]).reshape([-1]))
        if len(self.index1)!=len(self.index2) or len(self.index1)==0:
            raise Exception('Indices - (%s,%s) is not valid! @Bilinear'%(self.index1,self.index2))

    def __str__(self):
        txt=''
        for k in xrange(self.nind):
            if type(self.factor)==ndarray:
                factor=self.factor[0,0]
            else:
                factor=self.factor
            txt+='%s*('%(factor*self.weight,)
            inds=[self.spaceconfig.ind2c(self.index1[k]),self.spaceconfig.ind2c(self.index2[k])]
            c=['C_','C_']
            for i in xrange(2):
                if self.spaceconfig.nspin==2:
                    c[i]+=('dn' if inds[i][1] else 'up')
                elif self.spaceconfig.smallnambu:
                    c[i]+=('dn' if inds[i][0] else 'up')
                if self.spaceconfig.natom>=2:
                    c[i]+=chr(65+inds[i][2])
                if self.spaceconfig.norbit>=2:
                    c[i]+='o'+str(inds[i][3])
                if self.spaceconfig.nnambu==2:
                    c[i]+='+' if (i==inds[i][0]) else ''
                else:
                    c[i]+='+' if (i==0 or self.issc) else ''
            txt+=''.join(c)+self.meta_info
            if k!=self.nind-1:
                txt+=' + '
        return txt+')'

    def __call__(self,param=1.,dense=True):
        '''
        Refer Operator.__call__ for useage.
        '''
        param=param*self.weight*self.factor
        if isinstance(self.spaceconfig,SuperSpaceConfig):
            nflv=self.spaceconfig.nflv
            if self.nind!=1:
                print 'Error, multiple indices not supported yet for occupation representaion.'
                pdb.set_trace()
            ind1,ind2=self.index1[0],self.index2[0]
            if ind2!=ind1:
                if ind1<nflv:
                    if ind2<nflv:
                        occ,unocc=[ind2],[ind1]
                    else:
                        ind2-=nflv
                        occ,unocc=[],[ind2,ind1]
                else:
                    if ind2<nflv:
                        ind1-=nflv
                        occ,unocc=[ind2,ind1],[]
                    else:
                        ind1-=nflv
                        ind2-=nflv
                        occ,unocc=[ind1],[ind2]
                #note that return values is [initial states,final states,electrons between them]
                index2,index1,count_e=self.spaceconfig.indices_occ(occ=occ,unocc=unocc,getreverse=True,count_e=True)
                #coping the sign problem
                sparam=array([param]*len(index1))
                if ind1<ind2:
                    #ind2 passes eletrons [0,ind2)
                    #ind1 passes eletrons [0,ind1)
                    #the sign is equal to the electrons ind1<=site<ind2, note that ind1 is not occupied here.
                    sparam[count_e%2==1]*=-1
                else:
                    #ind2 passes eletrons [0,ind2)
                    #ind1 passes eletrons [0,ind1)+(-1 if ind2 has e else +1, change sgn)
                    #the sign is equal to the electrons ind2<site<ind1
                    if not self.issc:
                        sparam[count_e%2==1]*=-1
                    else:
                        sparam[count_e%2==0]*=-1  #the first C2+ will take effect on C1+!
            else:
                if self.index1<nflv:
                    index2=index1=self.spaceconfig.indices_occ(occ=[self.index1[0]])[0]
                else:
                    index2=index1=self.spaceconfig.indices_occ(unocc=[self.index1[0]])[0]
                sparam=array([param]*len(index1))
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
            H[ix_(self.index1,self.index2)]=param
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
    def nbody(self): 
        '''number of indices'''
        return 2

    @property
    def nind(self):
        '''
        The number of indices.
        '''
        return len(self.index1)

    @property
    def multiple_indexing(self):
        '''
        True if  this operator uses multiple indexing.
        '''
        return self.nind>1

    @property
    def issc(self):
        '''
        Is a c+c+ type bilinear if true.
        '''
        return (self.index2>=self.spaceconfig.nflv) & (self.index1<self.spaceconfig.nflv)

    def Hk(self,k,param=1.):
        '''
        Get Hamiltonian defined on k(mesh).

        k:
            The k-vector, or mesh of k vectors.
        param:
            The multiplication factor.
        '''
        param=param*self.factor*self.weight
        if ndim(k)==1:
            #single k
            H=zeros([self.spaceconfig.hndim]*2,dtype='complex128')
            H[ix_(self.index1,self.index2)]+=param
            return H
        elif ndim(k)>1:
            H=zeros(list(k.shape[:-1])+[self.spaceconfig.hndim]*2,dtype='complex128')
            subind=ix_(self.index1,self.index2)
            if ndim(param)==0:
                H[...,subind[0],subind[1]]+=param
            else:
                H[...,subind[0],subind[1]]+=param[:,:,newaxis,newaxis]
            return H
        else:
            raise Exception('Error','Unkown kmesh')


    def ravel(self,factor=1.,**kwargs):
        '''
        Ravel an Bilinear to elementary ones without multiple indexing.

        factor:
            The bonus multiplication factor.
        '''
        bllist=[]
        ind1=arange(self.spaceconfig.ndim)[self.index1]
        ind2=arange(self.spaceconfig.ndim)[self.index2]
        factor=factor*self.factor*self.weight
        for i in xrange(len(ind1)):
            for j in xrange(len(ind2)):
                f=factor[i,j] if ndim(factor)>0 else factor
                if abs(f)>1e-5:
                    bllist.append(type(self)(spaceconfig=self.spaceconfig,index1=ind1[i],index2=ind2[j],factor=f,**kwargs))
        return bllist

class BBilinear(Bilinear):
    '''
    Bilinear defined on a bond.

    Construct
    ------------------
    BBilinear(spaceconfig,index1,index2,bondv,factor=1.,label=None), label is the `id` of this instance if not specified.

    Attributes
    ---------------
    bondv:
        The bond vector.

    *see also <Bilinear> class*
    '''
    def __init__(self,spaceconfig,index1,index2,bondv,factor=1.,label=None):
        super(BBilinear,self).__init__(spaceconfig=spaceconfig,index1=index1,index2=index2,factor=factor,label=label)
        self.bondv=array(bondv)
        self.meta_info='' if norm(self.bondv)<1e-5 else ' <%s>'%(','.join(['%.2f'%b for b in self.bondv]))

    def ravel(self,factor=1.):
        '''
        Ravel BBilinear instance.

        factor: the bonus factor.
        '''
        return super(BBilinear,self).ravel(factor=factor,bondv=self.bondv)

    def Hk(self,k,param=1.):
        '''
        Get Hamiltonian defined on k(mesh).

        k:
            The k-vector, or mesh of k vectors.
        param:
            The multiplication factor.
        '''
        return super(BBilinear,self).Hk(k,param*exp(1j*(k*self.bondv).sum(axis=-1)))

class Qlinear(Operator):
    '''
    Qlinear class(4 body operator), unlike bilinear, qlinear does not support factor!

    Construct
    ------------------
    Qlinear(spaceconfig,indices,factor=1.,label=None), label is the `id` of this instance if not specified.

    Attributes
    ------------------
    indices:
        4 indices for qlinear. sorted in the normal order "c+c+cc".

    *see also <Operator> class.*
    '''
    def __init__(self,spaceconfig,indices,factor=1.,label=None):
        super(Qlinear,self).__init__(label=label,spaceconfig=spaceconfig,factor=factor)
        self.indices=array(indices)

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

    def __str__(self):
        txt=''
        if type(self.factor)==ndarray:
            factor=self.factor[0,0]
        else:
            factor=self.factor
        txt+='%s*('%(factor*self.weight,)
        inds=[self.spaceconfig.ind2c(ind) for ind in self.indices]
        c=['C_']*self.nbody
        for i in xrange(self.nbody):
            if self.spaceconfig.nspin==2:
                c[i]+=('dn' if inds[i][1] else 'up')
            elif self.spaceconfig.smallnambu:
                c[i]+=('dn' if inds[i][0] else 'up')
            if self.spaceconfig.natom>=2:
                c[i]+=chr(65+inds[i][2])
            if self.spaceconfig.norbit>=2:
                c[i]+='o'+str(inds[i][3])
            if i<2:
                c[i]+='+'
        txt+=''.join(c)+self.meta_info
        return txt+')'

class Qlinear_ninj(Qlinear):
    '''
    V*ni*nj type operator. e.g. hubbard U and coloumb V.

    Construct
    ------------------
    Qlinear_ninj(spaceconfig,i,j,factor=1.,label=None), label is the `id` of this instance if not specified.

    Attributes:
        *see Qlinear*
    '''
    def __init__(self,spaceconfig,i,j,factor=1.,label=None):
        if i==j:
            print 'Warning: Reducing it to Blinear!'
            self=Bilinear(spaceconfig=spaceconfig,index1=i,index2=j,label=label,factor=factor)
            return

        #note that it will take a '-' sign for (ni,nj) to transform to normal order
        factor*=-1
        super(Qlinear_ninj,self).__init__(spaceconfig=spaceconfig,indices=[i,j,i,j],label=label,factor=factor)

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
        data=param*self.weight*(-self.factor)*ones(indices.shape,dtype='complex128')
        if not dense:
            h=coo_matrix((data,(indices,indices)),shape=[self.spaceconfig.hndim]*2)
            return h
        else:
            H=zeros(shape=[self.spaceconfig.hndim]*2,dtype='complex128')
            H[indices,indices]+=data
            return H

class Nlinear(Operator):
    '''
    Nlinear class(N body operator).

    Construct
    ------------------
    Qlinear(spaceconfig,indices,factor=1.,label=None), label is the `id` of this instance if not specified.

    Attributes
    ------------------
    indices:
        N indices for Nlinear. sorted in the normal order "c+c+...cc...".

    *see also <Operator> class.*
    '''
    def __init__(self,spaceconfig,indices,indices_ndag=None,factor=1.,label=None):
        super(Nlinear,self).__init__(label=label,spaceconfig=spaceconfig,factor=factor)
        self.indices=array(indices)
        if indices_ndag is None: indices_ndag=len(indices)/2
        self.indices_ndag=indices_ndag

    def __getattr__(self,name):
        mch=re.match('^index(\d+)$',name)
        if mch:
            n=int(mch.group(1))
            if n<self.nbody:
                return self.indices[n]
            else:
                raise ValueError('Index exceeded %s/%s.'%(n,self.nbody))
        else:
            return super(Nlinear,self).__getattr__(name)

    def __str__(self):
        txt=''
        if type(self.factor)==ndarray:
            factor=self.factor[0,0]
        else:
            factor=self.factor
        txt+='%s*('%(factor*self.weight,)
        inds=[self.spaceconfig.ind2c(ind) for ind in self.indices]
        c=['C_']*self.nbody
        for i in xrange(self.nbody):
            if self.spaceconfig.nspin==2:
                c[i]+=('dn' if inds[i][1] else 'up')
            if self.spaceconfig.natom>=2:
                c[i]+=chr(65+inds[i][2])
            if self.spaceconfig.norbit>=2:
                c[i]+='o'+str(inds[i][3])
            if i<self.indices_ndag:
                c[i]+='+'
        txt+=''.join(c)+self.meta_info
        return txt+')'

    def __call__(self,param,dense=False,**kwargs):
        raise Exception('Not Implemented!')
        #first get all posible distribution of eletrons with site2 and site1 both occupied/unoccupied(reverse).
        indices=self.spaceconfig.indices_occ(occ=self.indices[self.indices_ndag:])[0]
        #set the H matrix, note the '-' sign befor self.factor.
        data=param*self.weight*(-self.factor)*ones(indices.shape,dtype='complex128')
        if not dense:
            h=coo_matrix((data,(indices,indices)),shape=[self.spaceconfig.hndim]*2)
            return h
        else:
            H=zeros(shape=[self.spaceconfig.hndim]*2,dtype='complex128')
            H[indices,indices]+=data
            return H

    @property
    def nbody(self):
        '''N body operator.'''
        return len(self.indices)

def formop(oplist,label=None):
    '''
    Construct a new operator from a list of operators.

    oplist: 
        A list of operator.
    label:
        The label of new operator.
    '''
    op=Operator(label=label,spaceconfig=oplist[0].spaceconfig)
    for item in oplist:
        op.addsubop(item)
    return op

if __name__=='__main__':
    pass
