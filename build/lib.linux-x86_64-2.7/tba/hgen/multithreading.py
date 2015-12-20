#!/usr/bin/python
from numpy import *
from mpi4py import MPI
from matplotlib.pyplot import *
#MPI setting
try:
    COMM=MPI.COMM_WORLD
    SIZE=COMM.Get_size()
    RANK=COMM.Get_rank()
except:
    COMM=None
    SIZE=1
    RANK=0

__all__=['mpido']

def mpido(func,inputlist,bcastouputmesh=True):
    '''
    MPI for list input.

    func:
        The function defined on inputlist.
    inputlist:
        The input list.
    bcastouputmesh:
        broadcast output mesh if True.
    '''
    N=len(inputlist)
    ntask=(N-1)/SIZE+1
    datas=[]
    for i in xrange(N):
        if i/ntask==RANK:
            datas.append(func(inputlist[i]))
    datal=COMM.gather(datas,root=0)
    if RANK==0:
        datas=[]
        for datai in datal:
            datas+=datai
    #broadcast mesh
    if bcastouputmesh:
        datas=COMM.bcast(datas,root=0)
    return datas

def test_mpido():
    x=linspace(0,1,100)
    y=mpido(func=lambda x:x**2,inputlist=x)
    if RANK==0:
        plot(x,y)
        show()

if __name__=='__main__':
    test_mpido()
