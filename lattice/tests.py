#!/usr/bin/python

from numpy import *
from matplotlib.pyplot import *
from latticelib import *
from structure import Structure
from group import TranslationGroup
import time,pdb
def test_structure(N,lt):
    '''
    N:
        The size of the lattice.
    lt:
        The lattice type.
    '''
    lts=construct_lattice(N=N,lattice_shape=lt)
    tgroup=TranslationGroup(Rs=lts.a*lts.N[:,newaxis],per=(True,False))
    lt=Structure(lts.sites)
    #test for bonds
    lt.usegroup(tgroup)
    lt.initbonds(K=17)
    #finding neighbors
    print 'find nearest neighbor %s'%lt.b1s.N
    print 'find second nearest neighbor %s'%lt.b2s.N
    print 'find third nearest neighbor %s'%lt.b3s.N
    isite=array([1]*lt.vdim)
    #find a site at specific position.
    print 'finding - %s'%isite,lt.findsite(isite)   
    i=0
    if lts.dimension==2:
        j=lts.l2index((0,N[1]-1,0))
        print 'measureing distance between site %s and %s -> %s'%(i,j,lt.measure(i,j))
        j=lts.l2index((N[0]-1,0,0))
        print 'measureing distance between site %s and %s -> %s'%(i,j,lt.measure(i,j))
    else:
        j=lt.nsite-1
        print 'measureing distance between site %s and %s -> %s'%(i,j,lt.measure(i,))
    ion()
    lt.show_bonds()
    lt.show_sites()
    pdb.set_trace()
    cla()
    #test for save and load functionality.
    lt.save_bonds()
    lt2=Structure(lts.sites)
    lt2.load_bonds()
    lt2.show_bonds()
    lt2.show_sites()
    axis('equal')
    pdb.set_trace()

def test_lattice(N,lt):
    '''
    Test functions for lattice class.
    '''
    ion()
    lt=construct_lattice(N=N,lattice_shape=lt)
    tgroup=TranslationGroup(Rs=lt.a*lt.N[:,newaxis],per=(True,False))
    print lt
    l0=[random.randint(n) for n in lt.siteconfig]
    index=lt.l2index(l0)
    print 'l2index l0 = %s -> %s, index2l %s -> %s'%(l0,index,index,lt.index2l(index))
    l=copy(l0)
    l[0]=l[0]+lt.siteconfig[0]
    index=lt.l2index(l)
    print 'l2index l = %s -> %s, index2l %s -> %s'%(l,index,index,lt.index2l(index))
    lpos=lt.lmesh[tuple(l0)]+tgroup.Rs[0]
    print 'finding l at %s, get %s.'%(lpos,lt.findsite(lpos))
    lt.usegroup(tgroup)
    print 'For periodic condition.'
    print 'finding l at %s, get %s.'%(lpos,lt.findsite(lpos))
    pdb.set_trace()
    cla()
    cbonds=lt.cbonds
    print 'get cell bonds,'
    for i in xrange(1,4):
        print '%s -> %s'%(i,lt.cbonds[i])
    lt.showcell((1,2,3))
    axis('equal')
    pdb.set_trace()

def test_kspace(N,lt):
    '''
    Test functions for lattice class.
    '''
    lt=construct_lattice(N=N,lattice_shape=lt)
    tgroup=TranslationGroup(Rs=lt.a*lt.N[:,newaxis],per=(True,False))
    lt.usegroup(tgroup)
    ks=lt.kspace
    bzone=ks.get_bzone()
    ion()
    bzone.show()
    ks.show()
    km=ks.kmesh
    #kmbzone=array(filter(bzone.inbzone,km.reshape([-1,km.shape[-1]])))
    kmbzone=km[bzone.inbzone(km)]
    scatter(kmbzone[:,0],kmbzone[:,1],color='r')
    axis('equal')
    pdb.set_trace()
 
if __name__=='__main__':
    test_structure((100,100),'honeycomb')
    #test_lattice((30,20),'honeycomb')
    #test_kspace((30,20),'honeycomb')
