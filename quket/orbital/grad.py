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

orbital/grad.py

Functions related to orbital gradiens and hessians

"""
import numpy as np
import time

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.linalg import vectorize_skew, skew
from quket.fileio import error, prints, printmat, print_state
from .misc import *

def get_total_energy(Quket, DA=None, DB=None, Daaaa=None, Dbbbb=None, Dbaab=None, state=None):
    """Function
    Get energy by 1- and 2-RDMs.

    By default, gradientis computed by using 1- and 2-RDMs of Quket.state (stored as Quket.DA, Quket.DB, etc.).
    If Quket.DA etc. are None, then they are computed here.
    
    If 1- and 2-RDMs of a specific state are passed, then the code will use them.
    This is useful for state-average calculations by substituting state-averaged density matrices. 
    Note that DA, DB are (norbs, norbs) whereas Daaaa, Dbbbb, Dbaaab are (nact, nact, nact, nact), where nact is the number of active orbitals (n_qubits/2).
    Also, the definitions of DA and Daaaa here are
    DA[p,q] = <p^ q>
    Daaaa[p,q,r,s] = <p^ q^ r s>
    
    If 'state' is given, its 1- and 2-RDMs are explicitly re-computed and used.
    'state' takes a higher preference over 1- and 2-RDMs.

    Args:
        Quket (QuketData): 
        DA (2darray, optional): Alpha 1-particle density matrix (norbs, norbs)
        DB (2darray, optional): Beta 1-particle density matrix (norbs, norbs)
        Daaaa (4darray, optional): Alpha 2-particle density matrix in the active space (nact, nact, nact, nact)
        Dbbbb (4darray, optional): Beta 2-particle density matrix (nact, nact, nact, nact)
        Dbaab (4darray, optional): Beta-Alpha 2-particle density matrix (nact, nact, nact, nact)
        state (QuantumState, optional): QuantumState for which orbital Hessian is computed.
    Returns:
        Energy (float): Total Energy
    
    Author(s): Takashi Tsuchimochi
    """
    from quket.post import get_1RDM, get_2RDM
    ncore = Quket.n_frozen_orbitals + Quket.n_core_orbitals
    nact = Quket.n_active_orbitals
    norbs = Quket.n_orbitals
    nsec = norbs - ncore - nact 
    if state is not None:
        # Use the inserted quantum state 
        DA, DB = get_1RDM(Quket, state=state)
        Daaaa, Dbaab, Dbbbb = get_2RDM(Quket, state=state)
    elif DA is DB is Daaaa is Dbbbb is Dbaab is None:
        if Quket.DA is None:
            Quket.get_1RDM()
        if Quket.Daaaa is None:
            Quket.get_2RDM()
        DA = Quket.DA
        DB = Quket.DB
        Daaaa = Quket.Daaaa
        Dbaab = Quket.Dbaab
        Dbbbb = Quket.Dbbbb
    elif DA is None or DB is None or Daaaa is None or Dbbbb is None or Dbaab is None:
        raise ValueError(f'Some density matrices are plugged in but not all that are required:'
                         f'DA = {type(DA)}   DB = {type(DB)}  Daaaa = {type(Daaaa)}  Dbbbb = {type(Dbbbb)}  Dbaab = {type(Dbaab)}')

    # Active Space 1RDM
    Daa  = DA[ncore:ncore+nact,ncore:ncore+nact]
    Dbb  = DB[ncore:ncore+nact,ncore:ncore+nact]

    ### Get htilde
    Ecore, htilde = get_htilde(Quket) 
    h_wxyz = Quket.two_body_integrals[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    E1 = np.einsum('pq,pq->',htilde, Daa+Dbb)
    E2 = 0.5*np.einsum('prqs, pqsr->',h_wxyz, Daaaa+2*Dbaab+Dbbbb)
    return Ecore+E1+E2
    

def get_orbital_gradient(Quket, DA=None, DB=None, Daaaa=None, Dbbbb=None, Dbaab=None, state=None):
    """Function
    Form orbital gradient by density matrix,
      Jpq = < [H, Epq - Eqp] >
    where p and q run over all spatial orbitals (alpha and beta).
    
    By default, gradientis computed by using 1- and 2-RDMs of Quket.state (stored as Quket.DA, Quket.DB, etc.).
    If Quket.DA etc. are None, then they are computed here.
    
    If 1- and 2-RDMs of a specific state are passed, then the code will use them.
    This is useful for state-average calculations by substituting state-averaged density matrices. 
    Note that DA, DB are (norbs, norbs) whereas Daaaa, Dbbbb, Dbaaab are (nact, nact, nact, nact), where nact is the number of active orbitals (n_qubits/2).
    Also, the definitions of DA and Daaaa here are
    DA[p,q] = <p^ q>
    Daaaa[p,q,r,s] = <p^ q^ r s>
    
    If 'state' is given, its 1- and 2-RDMs are explicitly re-computed and used.
    'state' takes a higher preference over 1- and 2-RDMs.

    Args:
        Quket (QuketData): 
        DA (2darray, optional): Alpha 1-particle density matrix (norbs, norbs)
        DB (2darray, optional): Beta 1-particle density matrix (norbs, norbs)
        Daaaa (4darray, optional): Alpha 2-particle density matrix in the active space (nact, nact, nact, nact)
        Dbbbb (4darray, optional): Beta 2-particle density matrix (nact, nact, nact, nact)
        Dbaab (4darray, optional): Beta-Alpha 2-particle density matrix (nact, nact, nact, nact)
        state (QuantumState, optional): QuantumState for which orbital Hessian is computed.
    Returns:
        general_Ja (2Darray): (Skew) gradient matrix for alpha, Jpq = -Jqp
        general_Jb (2Darray): (Skew) gradient matrix for beta, Jpq = -Jqp
    
    Author(s): Takashi Tsuchimochi
    """
    from quket.post import get_1RDM, get_2RDM
    ncore = Quket.n_frozen_orbitals + Quket.n_core_orbitals
    nact = Quket.n_active_orbitals
    norbs = Quket.n_orbitals
    nsec = norbs - ncore - nact 
    if state is not None:
        # Use the inserted quantum state 
        DA, DB = get_1RDM(Quket, state=state)
        Daaaa, Dbaab, Dbbbb = get_2RDM(Quket, state=state)
    elif DA is DB is Daaaa is Dbbbb is Dbaab is None:
        if Quket.DA is None:
            Quket.get_1RDM()
        if Quket.Daaaa is None:
            Quket.get_2RDM()
        DA = Quket.DA
        DB = Quket.DB
        Daaaa = Quket.Daaaa
        Dbaab = Quket.Dbaab
        Dbbbb = Quket.Dbbbb
    elif DA is None or DB is None or Daaaa is None or Dbbbb is None or Dbaab is None:
        raise ValueError(f'Some density matrices are plugged in but not all that are required:'
                         f'DA = {type(DA)}   DB = {type(DB)}  Daaaa = {type(Daaaa)}  Dbbbb = {type(Dbbbb)}  Dbaab = {type(Dbaab)}')

    # Active Space 1RDM
    Daa  = DA[ncore:ncore+nact,ncore:ncore+nact]
    Dbb  = DB[ncore:ncore+nact,ncore:ncore+nact]

    ### Get htilde
    htilde = Quket.one_body_integrals.copy()
    h_pqIJ = Quket.two_body_integrals[:,:,:ncore,:ncore]
    h_pIJq = Quket.two_body_integrals[:,:ncore,:ncore,:]
    h_wxyz = Quket.two_body_integrals[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    h_Ixyz = Quket.two_body_integrals[:ncore,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    h_Axyz = Quket.two_body_integrals[ncore+nact:norbs,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    h_IAxy = Quket.two_body_integrals[:ncore,ncore+nact:,ncore:ncore+nact, ncore:ncore+nact]
    h_IyxA = Quket.two_body_integrals[:ncore,ncore:ncore+nact,ncore:ncore+nact, ncore+nact:]
    h2_full = Quket.two_body_integrals
    for p in range(norbs):
        for q in range(norbs):
            for K in range(ncore):
                htilde[p,q] += 2*h_pqIJ[p,q,K,K] - h_pIJq[p,K,K,q] 


    ### Get Jxy (active-active)
    # alpha: <Daa[x,z] htilde[y+ncore, z+ncore]
    # beta : <Dbb[x,z] htilde[y+ncore, z+ncore]
    Jxy_a = np.einsum('xz,yz->xy',Daa, htilde[ncore:ncore+nact, ncore:ncore+nact])
    Jxy_b = np.einsum('xz,yz->xy',Dbb, htilde[ncore:ncore+nact, ncore:ncore+nact])
    # alpha: (pq|ry) (Daaaa[p,r,x,q] + Dbaab[p,r,x,q])
    # beta:  (pq|ry) (Dbbbb[p,r,x,q] + Dabba[p,r,x,q]) = (pq|ry) (Dbbbb[r,p,q,x] + Dbaab[r,p,q,x])
    Jxy_a += np.einsum('pqry,prxq->xy', h_wxyz, (Daaaa+Dbaab))
    Jxy_b += np.einsum('pqry,rpqx->xy', h_wxyz, (Dbbbb+Dbaab))

    Jxy_a = Jxy_a.T - Jxy_a 
    Jxy_b = Jxy_b.T - Jxy_b 

    ### Get JxI (active-core) 
    #   JxI = 2 (  
    #             htilde[I,x]  - sum_x D1[x,y] htilde[y,I]
    #           + sum_yz [ (yz|Ix) - (Iz|yx) ] D1[z,y] 
    #           - sum_yzw (yz|wI) D2[wy,zx]

    JxI_a = htilde[ncore:ncore+nact, :ncore].copy() \
           - np.einsum('xy,yI->xI',Daa, htilde[ncore:ncore+nact,:ncore]) \
           + np.einsum('Ixyz,zy->xI', h_Ixyz, (Daa+Dbb)) \
           - np.einsum('Izyx,zy->xI', h_Ixyz, Daa) \
           - np.einsum('Iwyz,wyzx->xI', h_Ixyz, (Daaaa+Dbaab))
    JxI_b = htilde[ncore:ncore+nact, :ncore].copy() \
           - np.einsum('xy,yI->xI',Dbb, htilde[ncore:ncore+nact,:ncore]) \
           + np.einsum('Ixyz,zy->xI', h_Ixyz, (Daa+Dbb)) \
           - np.einsum('Izyx,zy->xI', h_Ixyz, Dbb) \
           - np.einsum('Iwyz,ywxz->xI', h_Ixyz, (Dbbbb+Dbaab))

    ### Get JAx (frozenv-active) 
    #   JAx = 2 (  
    #             sum_y D1[x,y] htilde[y,A]
    #           + sum_yzw (yz|wA) D2[yw,zx]

    JAx_a = np.einsum('xy,yA->Ax', Daa, htilde[ncore:ncore+nact, ncore+nact:]) \
            + np.einsum('Awyz,wyzx->Ax', h_Axyz, (Daaaa+Dbaab))
    JAx_b = np.einsum('xy,yA->Ax', Dbb, htilde[ncore:ncore+nact, ncore+nact:]) \
            + np.einsum('Awyz,ywxz->Ax', h_Axyz, (Dbbbb+Dbaab))

    ### Get JAI (frozenv-frozenc) 
    #   JAI = 2 ( htilde[I,A] 
    #           + sum_xy ((xy|IA) - (Iy|xA)) D1[y,x]
    #           )
    JAI_a = htilde[ncore+nact:, :ncore] \
            + np.einsum('IAxy,yx->AI',h_IAxy, Daa+Dbb) \
            - np.einsum('IyxA,yx->AI',h_IyxA, Daa)
    JAI_b = htilde[ncore+nact:, :ncore] \
            + np.einsum('IAxy,yx->AI',h_IAxy, Daa+Dbb) \
            - np.einsum('IyxA,yx->AI',h_IyxA, Dbb)
    #
    J_IJ = np.zeros((ncore,ncore))
    J_AB = np.zeros((nsec,nsec))

    general_Ja = 2* np.block([[J_IJ, -JxI_a.T,-JAI_a.T],
                              [JxI_a, Jxy_a,  -JAx_a.T],
                              [JAI_a, JAx_a,   J_AB   ]])
    general_Jb = 2* np.block([[J_IJ, -JxI_b.T,-JAI_b.T],
                              [JxI_b, Jxy_b,  -JAx_b.T],
                              [JAI_b,JAx_b,   J_AB   ]])
    Ja_vec = vectorize_skew(general_Ja) 
    Jb_vec = vectorize_skew(general_Jb) 
    return Ja_vec, Jb_vec

def orbital_gradient(Quket, DA=None, DB=None, Daaaa=None, Dbbbb=None, Dbaab=None, state=None):
    norbs = Quket.n_orbitals
    grad = np.zeros((norbs*(norbs-1)//2),dtype=float)
    ### Get residual R
    Quket.get_2RDM()
    Quket.get_1RDM()
    JA, JB = get_orbital_gradient(Quket)
    R = (JA+JB)
    # Force symmetry
    pq = 0
    for p in range(norbs):
        for q in range(p):
            if Quket.irrep_list[2*p] != Quket.irrep_list[2*q]:
                if abs(R[p,q]) > 1e-4:
                    print(f"Warning! Orbitals p={p} and q={q} have different symmetries but orbital derivative is {R[p,q]}")
                R[p,q] = 0    
            grad[pq] = R[p,q]
            pq += 1
    print(f'Orbital gradient  ||g|| = {np.linalg.norm(grad)}')
    print(grad)
    return grad
