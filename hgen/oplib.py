'''
Operator Construction methods.
'''
from numpy import *
import pdb,copy,time

from op import *
from utils import sx,sy,sz
from spaceconfig import *

__all__=['op_from_mats','op_on_bond','op_U','op_V','op_c','op_cdag','op_simple_onsite','op_simple_hopping','op_M','site_shift','op_fusion']

def op_from_mats(label,spaceconfig,mats,bonds=None):
    '''
    get operator from mats.

    Parameters:
        :label: str, the label of this operator.
        :spaceconfig: <SpaceConfig>, the spaceconfig of this operator.
        :mats: list of 2D array, the hopping matrices.
        :bonds: <Bond>, the bonds.
    Return:
        <Operator>, the operator.
    '''
    tol=1e-12
    opt=Operator(label,spaceconfig,factor=1.)
    #add bilinears
    if bonds is None:
        for mat in mats:
            nzmask=abs(mat)>tol
            xs,ys=where(nzmask)
            for x,y in zip(xs,ys):
                opt.addsubop(Bilinear(spaceconfig,index1=x,index2=y),weight=mat[x,y])
    else:
        assert(len(mats)==len(bonds))
        for mat,bond in zip(mats,bonds):
            nzmask=abs(mat)>tol
            xs,ys=where(nzmask)
            for x,y in zip(xs,ys):
                opt.addsubop(BBilinear(spaceconfig,index1=x,index2=y,bondv=bond.bondv),weight=mat[x,y])
    return opt

def op_on_bond(label,spaceconfig,mats,bonds):
    '''
    Get operator on specific link.

    Parameters:
        :label: str, label of operator.
        :mats: 3D array, the operator matrix.
        :spaceconfig: <SpaceConfig>, the spaceconfig for the operator.
        :bonds: <BondCollection>/list of <Bond>, the bonds.

    Return:
        <Operator>, the operator.
    '''
    tol=1e-12
    opt=Operator(label,spaceconfig,factor=1.)
    config=list(spaceconfig.config)
    atomaxis=spaceconfig.get_axis('atom')
    config[atomaxis]=1
    if isinstance(spaceconfig,SuperSpaceConfig):
        spaceconfig1=SuperSpaceConfig(config,ne=spaceconfig.ne)
    elif isinstance(spaceconfig,SpaceConfig):
        spaceconfig1=SpaceConfig(config,kspace=spaceconfig.kspace)
    elif isinstance(spaceconfig,SpinSpaceConfig):
        spaceconfig1=SpinSpaceConfig(config)
    else:
        raise TypeError()

    #add bilinears
    assert(len(mats)==len(bonds))
    for mat,bond in zip(mats,bonds):
        nzmask=abs(mat)>tol
        xs,ys=where(nzmask)
        for x,y in zip(xs,ys):
            c1,c2=spaceconfig1.ind2c(x),spaceconfig1.ind2c(y)
            c1[atomaxis]=bond.atom1
            c2[atomaxis]=bond.atom2
            index1,index2=spaceconfig.c2ind(c1),spaceconfig.c2ind(c2)
            if bond.atom1==bond.atom2:
                opt.addsubop(Bilinear(spaceconfig,index1=index1,index2=index2),weight=mat[x,y])
            else:
                opt.addsubop(BBilinear(spaceconfig,index1=index1,index2=index2,bondv=bond.bondv),weight=mat[x,y])
    return opt

def op_U(spaceconfig,sites=None,label='U'):
    '''
    Get an Hubbard type operator like U*niup*nidn.

    spaceconfig: 
        the configuration of hamiltonian space.
    sites:
        the sites to hold this Hubbard interaction, default is all sites.
    label: 
        the label, default is 'U'.
    '''
    #check spaceconfig
    if spaceconfig.nspin<2:
        print 'To use op_U, you must take a spaceconfig with 2 spins.'

    op=Operator(label=label,spaceconfig=spaceconfig)
    if sites==None:
        sites=arange(spaceconfig.natom)
    for i in sites:
        indup=where(op.spaceconfig.subspace(atomindex=i,spinindex=0,nambuindex=0))[0]
        inddn=where(op.spaceconfig.subspace(atomindex=i,spinindex=1,nambuindex=0))[0]
        for i1,i2 in zip(indup,inddn):
            op.addsubop(Qlinear_ninj(spaceconfig=spaceconfig,i=i1,j=i2))
    return op

def op_V(spaceconfig,bonds,label='V'):
    '''Get an coloumb interaction operator like V*niup*njdn.

    spaceconfig: 
        the configuration of hamiltonian space.
    bonds: 
        the bonds to hold this coloumb interaction.
    label: 
        the label, default is 'V'.
    '''
    op=Operator(label=label,spaceconfig=spaceconfig)
    for bond in bonds:
        ind1=op.spaceconfig.subspace(atomindex=bond.atom1,nambuindex=0)
        ind2=op.spaceconfig.subspace(atomindex=bond.atom2,nambuindex=0)
        for i1,i2 in zip(ind1,ind2):
            op.addsubop(Qlinear_ninj(spaceconfig=spaceconfig,i=i1,j=i2))
    return op

def op_c(spaceconfig,index):
    '''
    Anilation Operator.
    
    spaceconfig: 
        the configuration of hamiltonian space.
    index:
        the index of electron to anilate.
    '''
    return Operator_C(spaceconfig=spaceconfig,index=index)

def op_cdag(spaceconfig,index):
    '''
    Creation Operator.
    
    spaceconfig: 
        the configuration of hamiltonian space.
    index:
        the index of electron to create.
    '''
    return Operator_C(spaceconfig=spaceconfig,index=index,dag=True)

def op_simple_onsite(label,spaceconfig,index=None):
    '''
    electron number operator.

    label:
        the label.
    spaceconfig: 
        the configuration of hamiltonian space.
    index:
        the index of this number operator(0 to ndim-1),

        Leave None to make it all indices.
    '''
    hndim=spaceconfig.ndim
    if spaceconfig.nnambu==1:
        if index is None:
            dg=ones(hndim)
        else:
            dg=zeros(hndim)
            dg[index]=1
    else:
        nflv=hndim/2
        if index is None:
            dg=append(ones(nflv),-ones(nflv))
        else:
            dg=zeros(hndim)
            dg[index]=1 if index<nflv else -1
    return op_from_mats(label=label,spaceconfig=spaceconfig,mats=[diag(dg)],bonds=None)


def op_M(spaceconfig,label='m',direction='z'):
    '''
    magnetic momentum operator.

    spaceconfig: 
        the configuration of hamiltonian space.
    label:
        the label.
    direction:
        the spin polarization direction.
    '''
    if spaceconfig.nspin==1 and spaceconfig.nnambu==1:
        raise Exception('You can not defined a magnetic operator in spinless system!')
    hndim=spaceconfig.ndim
    if direction=='x':
        smat=sx
    elif direction=='y':
        smat=sy
    elif direction=='z':
        smat=sz
    else:
        raise ValueError('Direction should be `x`,`y` or `z`! get %s.'%direction)

    if spaceconfig.nnambu==1:
        return op_from_mats(label,spaceconfig,[kron(smat,identity(hndim/2))],bonds=None)
    elif spaceconfig.smallnambu:
        if direction!='z':
            raise Exception('Spin direction %s is not allowed in small nambu space!'%direction)
        return op_from_mats(label,spaceconfig,[identity(hndim)],bonds=None)
    else:
        nflv=hndim/2
        smat=kron(sz,smat)
        return op_from_mats(label,spaceconfig,[kron(smat,identity(hndim/4))],bonds=None)

class Operator_D(Operator):
    '''
    Superconduct paring Operator.

    spaceconfig: 
        space configuration Instance.
    factor: 
        the factor for this magnetic unit. default 1.
    label:
        the label. default is 'D'.
    '''
    def __init__(self,spaceconfig,factor=1.,label='D'):
        super(Operator_D,self).__init__(spaceconfig=spaceconfig,factor=factor,label=label)
        self.cache['meta']=' + h.c.'

    def __call__(self,param=1.,dense=True):
        D=super(Operator_D,self).__call__(param=param,dense=dense)
        H=copy(D)
        nflv=self.spaceconfig.nflv
        torder=arange(ndim(D))
        torder[-2:]=torder[-2:][::-1]
        if dense:
            H[...,nflv:,:nflv]=H[...,nflv:,:nflv]+conj(transpose(D[...,:nflv,nflv:],axes=torder))
        else:
            H[...,nflv:,:nflv]+=conj(transpose(D[...,:nflv,nflv:],axes=torder))
        return H

    def Hk(self,k,param=1.,dense=True):
        '''
        get Superconducting hamiltonian defined in k-space.

        k:
            the momentum.
        param:
            the parameter.
        '''
   
        D=super(Operator_D,self).Hk(k=k,param=param)
        H=D
        nflv=self.spaceconfig.nflv
        torder=arange(ndim(D))
        torder[-2:]=torder[-2:][::-1]
        if dense:
            H[...,nflv:,:nflv]=H[...,nflv:,:nflv]+conj(transpose(D[...,:nflv,nflv:],axes=torder))
        else:
            H[...,nflv:,:nflv]+=conj(transpose(D[...,:nflv,nflv:],axes=torder))
        return H

    def D(self,k=None,param=1.):
        '''
        Get the superconducting part of hamiltonian.

        k:
            the momentum.
        '''
        nflv=self.spaceconfig.nflv
        if k==None:
            D=super(Operator_D,self).__call__(param=param)[:nflv,nflv:]
        else:
            D=super(Operator_D,self).Hk(k,param=param)[...,:nflv,nflv:]
        return D

    def readbonds(self,bonds,bondfunc=lambda b:1.,spinconfig='singlet'):
        '''Read a set of bonds to form a set of sub-operators.

        bonds:
            the Bond instances to read.
        bondfunc:
            a function defined on bond to determine the weight of sub-operators.
        spinconfig:
            configurations of spin, default is 'upup+dndn'(s0), refer utils.spindict for detail.
        '''
        spins=spin_dict[spinconfig]
        spaceconfig=self.spaceconfig #spaceconfig is used as the config to find indexer
        if (spaceconfig.smallnambu and spins!=spin_dict['singlet']):
            print 'Error, please check spinconfig or the dimension of you nambu spacec. @readbonds'
            pdb.set_trace()
        if isinstance(spaceconfig,SuperSpaceConfig):
            spaceconfig=deepcopy(self.spaceconfig)
            spaceconfig.config[0]=2
        if not hasattr(bondfunc,'__call__'):
            bondfunc=bfunc_dict[bondfunc]

        for b in bonds:
            if spaceconfig.nspin==1:
                spins=[(0,0,1.)]
            for s1,s2,factor in spins:
                index1=spaceconfig.subspace(atomindex=b.atom1,spinindex=s1,nambuindex=0)
                index2=spaceconfig.subspace(atomindex=b.atom2,spinindex=s2,nambuindex=1)
                bl=BBilinear(spaceconfig=self.spaceconfig,index1=index1,index2=index2,bondv=b.bondv)
                self.addsubop(bl,weight=bondfunc(b)*factor)

    def gk(self,k):
        '''
        Get the Geometric factor of pairing - g(k).

        k: 
            the momentum
        '''
        if self.spaceconfig.smallnambu:
            return self.Hk(k)[:self.spaceconfig.nflv,self.spaceconfig.nflv:]
        sgmy=self.spaceconfig.sigma(2,nambu=False)
        gk=dot(self.Hk(k),-1j*sgmy)
        return gk[:self.spaceconfig.nflv,self.spaceconfig.nflv:]

    def setbondweight(self,atom1,atom2,weight,mutual=True):
        '''
        Set the bond between atom1 and atom2.

        atom1/atom2:
            the start/end atom of the bond.
        weight:
            the new weight.
        mutual:
            set the conterpart at the same time.
        '''
        for dcell in self.suboperators:
            b=dcell.bond
            if b.atom1==atom1 and b.atom2==atom2:
                dcell.weight=weight
            if mutual and  (b.atom1==atom2 and b.atom2==atom1):
                dcell.weight=weight

    def breakbond(self,atom1,atom2,mutual=True):
        '''
        Break the bond between specified atoms.

        atom1/atom2: 
            the start/end atom of the bond.
        mutual:
            set the conterpart at the same time.
        '''
        self.setbondweight(atom1,atom2,0.,mutual=mutual)


def op_D(spaceconfig,bonds,bondfunc=lambda b:1.,spinconfig='singlet',label='D'):
    '''
    Get an cooper pairing operator.

    spinconfig: 
        the configuration of spin-pairing, default is singlet.
    bonds:
        the bonds to hold pairing.
    bondfunc:
        the weight of D as a function of bond. default is uniform.
    spinconfig:
        the pairing information, default is 'singlet' pairing. refer utils.spindict for detail.
    label: 
        the label, default is 'D'.
    '''
    op=Operator_D(label=label,spaceconfig=spaceconfig)
    op.readbonds(bonds,bondfunc=bondfunc,spinconfig=spinconfig)
    return op


def op_simple_hopping(label,spaceconfig,bonds):
    '''
    Get simple hopping terms.

    label:
        the label.
    spaceconfig: 
        the configuration of hamiltonian space.
    bonds:
        the bonds to hold this hopping term.
    '''
    mats=[]
    nflv=spaceconfig.nflv
    atomindexer=[where(spaceconfig.subspace(atomindex=i,nambuindex=0))[0] for i in xrange(spaceconfig.natom)]
    opt=Operator(label,spaceconfig,factor=1.)
    for bond in bonds:
        ind1s=atomindexer[bond.atom1]
        ind2s=atomindexer[bond.atom2]
        for x,y in zip(ind1s,ind2s):
            opt.addsubop(BBilinear(spaceconfig,index1=x,index2=y,bondv=bond.bondv),weight=1.)
    return opt

def site_shift(op,n,new_spaceconfig):
    '''
    Shift site of the specific opertor by changing indices.

    Parameters:
        :op: <Operator>/<Xlinear>, the operator.
        :n: integer, the sites to shift.
        :new_spaceconfig: <SpaceConfig>/None, the new space configuration, leave it `None` to use the old one.
    '''
    if new_spaceconfig is None:
        new_spaceconfig=operator.spaceconfig
    if isinstance(op,Operator):
        nop=0
        for xl in op.suboperators:
            nop=nop+site_shift(xl,n,new_spaceconfig=new_spaceconfig)
        nop.label=op.label
        return op.factor*nop
    elif isinstance(op,Xlinear):
        op=copy.copy(op)
        old_c=op.spaceconfig.ind2c(op.indices)
        old_c[:,op.spaceconfig.get_axis('atom')]+=n
        new_indices=new_spaceconfig.c2ind(old_c)
        op.indices=new_indices
        op.spaceconfig=new_spaceconfig
        return op
    else:
        raise TypeError()

def op_fusion(label,operators):
    '''
    Combine operators in cluster into blocks.

    Parameters:
        :label: The label of fusioned operator.
        :operators: The operators of cluster.

    Return:
        <Operator>, the operator after fusion.
    '''
    scfg=operators[0].spaceconfig
    natoms=[]
    is_super=isinstance(scfg,SuperSpaceConfig)
    ne=0
    for operator in operators:
        natoms.append(operator.spaceconfig.natom)
        if is_super and operator.spaceconfig.ne is not None:
            ne+=ne
    atomshift=append([0],cumsum(natoms))

    #get the new spaceconfig
    totalnatom=sum(natoms)
    config=list(scfg.config)
    if is_super:
        if ne==0:ne=None
        config[scfg.get_axis('atom')]=totalnatom
        spaceconfig=SuperSpaceConfig(config,ne=ne)
    elif isinstance(scfg,SpaceConfig):
        config[scfg.get_axis('atom')]=totalnatom
        spaceconfig=SpaceConfig(config,kspace=scfg.kspace)
    elif isinstance(scfg,SpinSpaceConfig):
        config[scfg.get_axis('atom')]=totalnatom
        spaceconfig=SpinSpaceConfig(config)
    else:
        raise ValueError()

    #fusion operators
    nop=0
    for op,shift in zip(operators,atomshift[:-1]):
        nop=nop+site_shift(op,n=shift,new_spaceconfig=spaceconfig)
    nop.label=label
    return nop
