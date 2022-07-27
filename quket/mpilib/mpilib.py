# Copyright 2022 The Quket Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# limitations under the License.
"""
#######################
#        quket        #
#######################

mpilib.py

Initiating MPI and setting relevant arguments.

"""
import numpy as np
import qulacs
from qulacs import DensityMatrix
try:
    from mpi4py import MPI
    
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    name = MPI.Get_processor_name()

except ImportError as error:
    print('mpi4py is not imported. no MPI.')
    class MPI():
        SUM = 'SUM'
        MAX = 'MAX'
        COMM_WORLD = None
        def Get_processor_name(self):
            return 'dummy'
        def Finalize(self):
            pass
    class Comm():
        def Bcast(self, buf, root):
            return None
        def bcast(self, buf, root):
            pass
        def Allreduce(self, sendbuf, recvbuf, op):
            raise Exception("This function is not meant for no MPI version!")
        def allreduce(self, sendbuf, op):
            return sendbuf
        def Gather(self, sendobj, recvobj, root):
            raise Exception("This function is not meant for no MPI version!")
        def gather(self, sendobj, root):
            return sendobj
        def send(self, obj, dest, tag): 
            pass
        def recv(self, buf, source, tag): 
            return None
        def Barrier(self):
            pass
    comm = Comm()
    MPI = MPI()
    rank = 0
    nprocs = 1

if rank == 0:
    main_rank = 1
else:
    main_rank = 0

# Wrappers
def bcast(buf, root=0):
    from quket.lib import QuantumState
    if nprocs == 1:
        return buf
    if isinstance(buf, np.ndarray): 
        #comm.Bcast(buf, root)
        #return buf
        return comm.bcast(buf, root)
    elif isinstance(buf, (QuantumState, qulacs.QuantumState)):
        vec = buf.get_vector()
        vec = comm.bcast(vec, root)
        buf.load(vec)
        return buf
    else:
        return comm.bcast(buf, root)
    
def allreduce(sendbuf, op=MPI.SUM):
    if nprocs == 1:
        return sendbuf
    else:
        if isinstance(sendbuf, np.ndarray): 
            recvbuf = np.zeros_like(sendbuf)
            comm.Allreduce(sendbuf, recvbuf, op)
            return recvbuf
        else:
            return comm.allreduce(sendbuf, op)
def gather(sendobj, root=0): 
    if nprocs == 1:
        return [sendobj]
    if isinstance(sendobj, np.ndarray):
        recvobj = np.zeros_like(sendobj)
        comm.Gather(sendobj, recvobj, root)
    else:
        return comm.gather(sendobj, root)
def allgather(sendobj, root=0):
    if nprocs == 1:
        return sendobj
    data = gather(sendobj,root=root)
    if rank == root:
        recvobj = [x for l in data for x in l]
    else:
        recvobj = None
    return bcast(recvobj, root=root)

def send(obj, dest, tag=0):
    if nprocs == 1:
        pass
    comm.send(obj, dest, tag)

def recv(buf=None, source=0, tag=0):
    if nprocs == 1:
        return obj
    return comm.recv(source, tag)

def barrier():    
    comm.Barrier()
    
def myrange(ndim, backward=False):
    """
        Calculate process-dependent range for MPI distribution of `ndim` range.
        Image
        ----------
        Process 0        :       0         ---     ndim/nprocs
        Process 1        :   ndim/procs+1  ---   2*ndim/nprocs
        Process 2        : 2*ndim/procs+1  ---   3*ndim/nprocs
          ...
        Process i        :     ipos        ---   ipos + my_ndim
          ...
        Process nprocs-1 :      ...        ---      ndim-1

        Returns `ipos` and `my_ndim`

        If backward = True, Process nprocs-1 gains the largest my_ndim.

    Author(s): Takashi Tsuchimochi
    """
    nrem = ndim%nprocs
    nblk = (ndim-nrem)//nprocs
    if not backward:
        if rank < nrem:
            my_ndim = nblk + 1
            ipos = my_ndim*rank
        else:
            my_ndim = nblk
            ipos = my_ndim*rank + nrem
    else:
        if rank < nprocs-nrem:
            my_ndim = nblk
            ipos = my_ndim*rank
        else:
            my_ndim = nblk + 1
            ipos = nblk*(nprocs-nrem)  + (rank - nprocs + nrem) * nrem
        
    return ipos, my_ndim

def mem_proc_dict():
    import quket.utils.memory as mem
    mem_list = gather(mem.available())
    proc_list = gather(MPI.Get_processor_name())
    mem_list = bcast(mem_list)
    proc_list = bcast(proc_list)
    mem_dict = dict(zip(proc_list,mem_list))
    proc_dict = {}
    for proc, mem in mem_dict.items(): 
        proc_dict[proc] = proc_list.count(proc)
    return mem_dict, proc_dict
