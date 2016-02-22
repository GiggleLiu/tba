'''
SpaceConfiguration for hamiltonian.
'''
from numpy import *
from numpy.linalg import *
from scipy.misc import factorial
from utils import ind2c,c2ind,s
from itertools import combinations
from matplotlib.pyplot import *
import pdb,time

__all__=['SpaceConfig','SuperSpaceConfig','SpinSpaceConfig']

standard_order=['nambu','spin','atom','orbit']

def parse_order(config,order):
    '''
    Parse config to standard order.

    Parameters:
        :config: len-4 list, the configuration.
        :order: len-4 list, the target order string.

    Return:
        len-4 list, the configuration in target order.
    '''
    cfg=[]
    for s in order:
        cfg.append(config[standard_order.index(s)])
    return cfg

class SpaceConfig(object):
    SPACE_TOKENS=['nambu','spin','atom','orbit']
    '''
    Space configuration class.
    A typical hamiltonian space config consist of 4 part:

        1. Use Nambu space?
        2. How many spins?
        3. Number of sites.
            the sites in a unit cell for momentum being a good quantum number.
            the total sites for k not a good quantum number.
        4. How many orbital to be considerd.

    config:
        an len-4 array/list, the elements are:

        1. nnambu, 2 if use Nambu space else 1.
        2. nspin, number of spins.
        3. natom, number of atoms.
        4. norbit, number of orbits.
    kspace:
        True if k is a good quantum number and you decided to work in k space.
    '''
    def __init__(self,config,kspace=True):
        self.config=array(config)
        self.kspace=kspace

    def __str__(self):
        return ' x '.join(['%s(%s)'%(token,n) for token,n in zip(self.SPACE_TOKENS,self.config)])

    def __getattr__(self,name):
        if name=='smallnambu':
            return self.nnambu==2 and self.nspin==1
        elif name=='superconduct':
            return self.nnambu==2
        elif name[-7:]=='indexer':
            s=name[:-7]
            s_axis=self.get_axis(s)
            nl=prod(self.config[:s_axis])
            nr=prod(self.config[s_axis+1:])
            return kron(ones(nl,dtype='int32'),kron(arange(self.config[s_axis]),ones(nr,dtype='int32')))
        elif name[0] is 'n':
            substr=name[1:]
            if substr=='flv':
                if 'nambu' in self.SPACE_TOKENS:
                    return self.ndim/self.nnambu
                else:
                    return self.ndim
            elif substr in self.SPACE_TOKENS:
                return self.config[self.SPACE_TOKENS.index(substr)]
        else:
            raise AttributeError('Can not find attibute %s'%name)

    @property
    def ndim(self):
        '''
        The dimension of space.
        '''
        return self.config.prod()

    @property
    def hndim(self):
        '''
        The dimension of hamiltonian.
        '''
        return self.ndim

    def get_axis(self,token):
        '''
        Get the axis index of specific token.
        '''
        assert(token in self.SPACE_TOKENS)
        return self.SPACE_TOKENS.index(token)

    def setnatom(self,N):
        '''
        Set atom number.

        N:
            the new atom number.
        '''
        self.config[-2]=N

    def ind2c(self,index):
        '''
        Parse index into space len-4 config indices like (nambui,spini,atomi,orbiti).

        index:
            the index(0 ~ ndim-1).
        '''
        return ind2c(index,N=self.config)

    def c2ind(self,indices=None,nambuindex=0,spinindex=0,atomindex=0,orbitindex=0):
        '''
        Parse space config index into index.

        indices:
            a len-4 array for all the above indices.
        nambuindex/spinindex/atomindex/orbitindex:
            index for nambuspace/spinspace/atomspace/orbitspace
        '''
        if self.smallnambu:
            spinindex=0
        if indices is None:
            indices=parse_order([nambuindex,spinindex,atomindex,orbitindex],order=self.SPACE_TOKENS)
        return c2ind(indices,N=self.config[-shape(indices)[-1]:])

    def subspace2(self,nambus=None,spins=None,atoms=None,orbits=None):
        '''
        Get a matrix of mask on a subspace.

        nambus/spins/atoms/orbits:
            array as indices of nambu/spin/atom/orbit, default is all indices.
        '''
        if nambus is None:
            nambus=arange(self.nnambu)
        if spins is None:
            spins=arange(self.nspin)
        if atoms is None:
            atoms=arange(self.natom)
        if orbits is None:
            orbits=arange(self.norbit)
        if self.smallnambu:
            spins=[0]
        subspace=zeros(self.config,dtype='bool')
        subspace[ix_(parse_order(nambus,spins,atoms,orbits,self.SPACE_TOKENS))]=True
        subspace=subspace.ravel()
        return subspace

    def subspace(self,nambuindex=None,spinindex=None,atomindex=None,orbitindex=None):
        '''
        Get the subspace mask. the single index version of subspace2.

        nambuindex/spinindex/atomindex/orbitindex:
            nambu/spin/atomindex/orbitindex index, default is all indices.
        '''
        if self.smallnambu:
            spinindex=0
        mask=ones(self.ndim,dtype='bool')
        if nambuindex!=None:
            mask=mask & (self.nambuindexer==nambuindex)
        if spinindex!=None:
            mask=mask & (self.spinindexer==spinindex)
        if atomindex!=None:
            mask=mask & (self.atomindexer==atomindex)
        if orbitindex!=None:
            mask=mask & (self.orbitindexer==orbitindex)
        return mask

    def sigma(self,index,onlye=False):
        '''
        Pauli matrices for spin.
        It is defined in whole space by default, which makes a different in superconductng case.

        index: 
            = 1,2,3 for spin x/y/z.
        onlye:
            return a spin operator define in eletron space only if True.
        '''
        spin_axis=self.get_axis('spin')
        if not onlye:
            nl=prod(self.config[1:spin_axis])
        else:
            nl=prod(self.config[:spin_axis])
        nr=prod(self.config[spin_axis+1:])
        return kron(kron(identity(nl),s[index]),identity(nr))

    def tau(self,index):
        '''
        Pauli matrices for nambu.

        index:
            = 1,2,3 for tau_x/y/z.
        '''
        nl=1
        nr=self.nflv
        return kron(kron(identity(nl),s[index]),identity(nr))

    @property
    def I(self):
        '''
        Identity matrix.
        '''
        dim=self.hndim
        return identity(dim)

class SuperSpaceConfig(SpaceConfig):
    '''
    Space config in the occupation representation.
    Note that the dimension of space here are different from dimension of hamiltonian.
    Notice, Notice, Notice -> we arange space as |0,1,2,3 ... >, here higher number means lower power.
    e.g.
        |up=0,dn=0> -> 0
        |up=1,dn=0> -> 1
        |up=0,dn=1> -> 2
        |up=1,dn=1> -> 3

    Attributes:
        :ne: integer/None, number of electrons, None for unrestricted.
        :ne_conserve: bool, True if electron numbers are conserved else False(readonly).
        :hndim: integer, the dimension of hilbert space.
        :nsite: the number of sites(flavor).
    '''
    def __init__(self,config,ne=None):
        super(SuperSpaceConfig,self).__init__([1]+list(config)[-3:],kspace=False)
        if ne is not None:
            self.setne(ne)
        else:
            self.ne=None
            self._table=None

        #_binaryparser is a parser from config to index, use '&' operation to parse.
        self._binaryparser=1 << arange(self.nsite)#-1,-1,-1)


    def __str__(self):
        return super(SuperSpaceConfig,self).__str__()+(', with %s electrons.'%self.ne if self.ne_conserve else '')

    def __getattr__(self,name):
        return super(SuperSpaceConfig,self).__getattr__(name)

    @property
    def ne_conserve(self):
        '''
        Electron number conservative.
        '''
        return self.ne is not None

    @property
    def nsite(self):
        '''
        The number of sites, note that it equivalent to nflv, superconducting space is needless.
        '''
        res=self.ndim
        if self.nnambu==2 and (not self.smallnambu):
            res/=2
        return res

    @property
    def hndim(self):
        '''
        The dimension of hamiltonian.
        '''
        nsite=self.nsite
        ne=self.ne
        if self.ne_conserve:
            hndim=int(factorial(nsite)/factorial(ne)/factorial(nsite-ne))
        else:
            hndim=pow(2,nsite)
        return hndim

    def setnatom(self,natom):
        '''
        Set the number of atoms.

        natom:
            the number of atoms.
        '''
        super(SuperSpaceConfig,self).setnatom(natom)
        self._binaryparser=1 << np.arange(self.nsite) #-1,-1,-1)

    def setne(self,N):
        '''
        Set eletron number.

        N: 
            the new eletron number.
        '''
        self.ne=N
        comb=fromiter(combinations(arange(self.nsite),N),dtype=[('',int32)]*N).view((int32,N))
        self._table=sort(sum(2**comb,axis=-1))

    def ind2config(self,index):
        '''
        Parse index into eletron configuration.

        Parameters:
            :index: integer/1D array, the index(0 ~ hndim-1).

        Return:
            1D array/2D array(dtype bool), A electron configuration.
        '''
        if self.ne_conserve:
            id=self._table[index]
            return self.id2config(id)
        else:
            return self.id2config(index)

    def config2ind(self,config):
        '''
        Parse eletron configuration to index.

        Parameters:
            :config: 1D/2D array, A electron configuration is len-nsite array with items 0,1(state without/with electron).

        Return: 
            a index(0 ~ hndim-1).
        '''
        if self.ne_conserve:
            #parse config to id - the number representation
            id=self.config2id(config)
            return searchsorted(self._table,id)
        else:
            return self.config2id(config)

    def plotconfig(self,config,offset=zeros(2)):
        '''
        Display a config of electron.

        Parameters:
            :config: 1D array/2D array, len-nsite array with items 0,1(state without/with electron).
        '''
        cfg=config.reshape(self.config)
        cfg=swapaxes(cfg,-1,-2).reshape([-1,self.natom])
        x,y=meshgrid(self.atomindexer,arange(self.nsite/self.natom))
        colors=cm.get_cmap('rainbow')(float64(cfg))
        scatter(x+offset[0],y+offset[1],s=20,c=colors.reshape([-1,4]))

    def config2id(self,config):
        '''
        Convert config to id

        Parameters:
            :config: 1D array/2D array, an array of 0 and 1 that indicates the config of electrons.

        Return:
            integer/1D array, indices.
        '''
        return sum(config*self._binaryparser,axis=-1)

    def id2config(self,id):
        '''
        Convert id to config.

        Parameters:
            :id: a number indicate the whole space index.

        Return:
            1D array/2D array, the configuration of electrons.
        '''
        if ndim(id)>0:
            id=id[...,newaxis]
        return (id & self._binaryparser)>0

    def indices_occ(self,occ=[],unocc=[],getreverse=False,count_e=False):
        '''
        Find the indices with specific sites occupied.

        Parameters:
            :occ: list, the sites with electron. default is no site.
            :unocc: list, unoccupied sites. default is no site.
            :getreverse: bool, get a reversed state(with occ unoccupied and unocc occupied) at the same time.
            :count_e: bool, count electron numbers before used sites.

        Return:
            list, [states meets requirements, final states(if get reverse), electrons before used sites]
        '''
        occ=unique(occ).astype('int32')
        unocc=unique(unocc).astype('int32')
        no=len(occ)
        nn=len(unocc)
        #get remained sites
        usedsites=concatenate([occ,unocc])
        order=argsort(usedsites)
        usedsites=usedsites[order]
        usedcounter=append(ones(len(occ),dtype='int32'),zeros(len(unocc),dtype='int32'))[order]
        remainsites=delete(arange(self.nsite),usedsites)
        if no+nn!=self.nsite-len(remainsites):
            raise ValueError('no match for indices occ-%s,unocc-%s! @indices_occ'%(occ,unocc))
        #get possible combinations
        if self.ne_conserve:
            distri=array(list(combinations(remainsites,self.ne-no)))
            if count_e:
                ecounter=[(distri<site).sum(axis=-1) for site in usedsites]
            #turn into indices.
            ids0=(2**distri).sum(axis=-1)
            ids=ids0+(2**occ).sum()
            indices=searchsorted(self._table,ids)
            res=[indices]
            if getreverse:
                rids=ids0+(2**unocc).sum()
                res.append(searchsorted(self._table,rids))
            if count_e:
                res.append(ecounter)
            return tuple(res)
        else:
            distri=self.id2config(arange(2**(self.nsite-no-nn)))[...,:-no-nn]
            if count_e:
                #ecounter=distri[...,usedsites[0]:usedsites[1]-1].sum(axis=-1)
                ecounter=[distri[...,:site-i].sum(axis=-1)+sum(usedcounter[:i]) for i,site in enumerate(usedsites)]
            parser=delete(self._binaryparser,usedsites)
            ids0=(distri*parser).sum(axis=-1)
            ids=ids0+(2**occ).sum()
            res=[ids]
            if getreverse:
                res.append(ids0+(2**unocc).sum())
            if count_e:
                res.append(ecounter)
            return tuple(res)

    def show_basis(self):
        '''
        display the whole basis.
        '''
        for i in xrange(self.hndim):
            print self.ind2config(i).astype(uint8)

class SpinSpaceConfig(SpaceConfig):
    '''
    Space configuration for spin system.

    spin number and atom number are considered.
    '''
    SPACE_TOKENS=['spin','atom']
    def __init__(self,config):
        assert(len(config)==2 and config[0]>1)
        super(SpinSpaceConfig,self).__init__(config,kspace=False)

    @property
    def hndim(self):
        '''Hamiltonian dimension.'''
        return self.config[0]**self.config[1:].prod()

    def sigma(self,index):
        '''
        Pauli operators.
        '''
        if self.nspin==2:
            return s[index]
        else:
            raise NotImplementedError()

    def config2ind(self,config):
        '''
        Convert config to id

        Parameters:
            :config: 1D array/2D array, an array of 0 and 1 that indicates the config of electrons.

        Return:
            integer/1D array, indices.
        '''
        nspin=self.nspin
        return sum([config[...,i]*self.natom**i for i in xrange(self.natom)],axis=0)

    def ind2config(self,id):
        '''
        Convert id to config.

        Parameters:
            :id: a number indicate the whole space index.

        Return:
            1D array/2D array, the configuration of electrons.
        '''
        nspin=self.nspin
        config=[]
        for i in xrange(self.natom):
            ci=id%nspin
            id=(id-ci)/nspin
            config.append(ci[...,newaxis])
        return concatenate(config[::-1],axis=-1)
