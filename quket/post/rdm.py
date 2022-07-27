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

rdm.py

Functions related to reduced density matrix.

"""
import copy
from typing import List
from dataclasses import dataclass, field, InitVar, make_dataclass

import numpy as np
from scipy.special import comb
import time
from qulacs.state import inner_product

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.linalg import symm, skew, skew3, skew4
from quket.fileio import error, prints, printmat, print_state
from quket.opelib import evolve
from quket.lib import FermionOperator

def calc_RDM(state, n_qubits, string, mapping='jordan_wigner'):
    """Function
    Calculate elements of each RDM in order to compute RDM rapidly.
    Author(s) : Taisei Nishimaki
    """
    from quket.opelib import OpenFermionOperator2QulacsGeneralOperator
    E = FermionOperator(string)
    E_qu = OpenFermionOperator2QulacsGeneralOperator(E, n_qubits, mapping=mapping)
    E_expect = E_qu.get_expectation_value(state).real
    return E_expect

def get_1RDM(Quket, state=None):
    """Function
    Compute 1RDM of QuantmState in QuketData (whole orbital space).
    No symmetry is enforced. They are tensors of the dimensions
    [n_orbitals, n_orbitals]

    Args:
        Quket (QuketData): QuketData instance
        state (QuantumState, optional): Target state for which RDM is computed
    Returns:
        DA (2darray): Alpha 1-particle density matrix
        DB (2darray): Beta 1-particle density matrix

    Author(s): Taisei Nishimaki, Takashi Tsuchimochi
    """
    prints(" === Computing 1RDM === ")
    if Quket.tapered["states"]:
        Quket.taper_off(backtransform=True)

    n_qubits = Quket.n_qubits
    norbs = Quket.n_active_orbitals
    if state is None:
        state = Quket.state
    mapping = Quket.cf.mapping

    # set for frozen_core #
    ncore = Quket.n_frozen_orbitals + Quket.n_core_orbitals
    n_orbitals =Quket.n_orbitals
    nsec  = n_orbitals - ncore - norbs

    my_DA = np.zeros((norbs, norbs))
    my_DB = np.zeros((norbs, norbs))

    # MPI parallel
    ipos, my_n_qubits = mpi.myrange(n_qubits)

    for i in range(ipos, ipos+my_n_qubits):
        for j in range(n_qubits):
            ii = i//2
            jj = j//2

            string = f"{i}^ {j}"
            Dpq = calc_RDM(state, n_qubits, string, mapping=mapping)

            if i%2 == 0 and j%2 == 0:
                my_DA[jj, ii] = Dpq
            elif i%2 == 1 and j%2 == 1:
                my_DB[jj, ii] = Dpq
    DA = mpi.allreduce(my_DA, mpi.MPI.SUM)
    DB = mpi.allreduce(my_DB, mpi.MPI.SUM)

    # frozen-core and virtual#
    if ncore != 0:
        hf_rdm = np.identity(ncore, float)
        zeros_adj  = np.zeros((norbs, ncore))
        DA = np.block([[hf_rdm, zeros_adj.T],
                        [zeros_adj, DA]])
        DB = np.block([[hf_rdm, zeros_adj.T],
                        [zeros_adj, DB]])

    if nsec != 0:
        zero_rdm = np.zeros((nsec, nsec), float)
        zeros_adj  = np.zeros((nsec, norbs+ncore))
        DA = np.block([[DA, zeros_adj.T],
                    [zeros_adj, zero_rdm]])
        DB = np.block([[DB, zeros_adj.T],
                    [zeros_adj, zero_rdm]])


    return DA, DB

def has_duplicates(seq):
    return len(seq) != len(set(seq))


def rdm2_to_rdm1(Quket):
    ### check ###
    print("Check 2RDM -> 1RDM \n")
    norbs = Quket.n_orbitals
    N = Quket.n_electrons
    Daaaa = Quket.Daaaa
    Dbbbb = Quket.Dbbbb
    Dbaab = Quket.Dbaab
    Dabba = Dbaab.transpose(1,0,3,2)
    Daa2 = np.zeros((norbs,norbs))
    Dbb2 = np.zeros((norbs,norbs))
    for p in range(norbs):
        for r in range(norbs):
            for q in range(norbs):
                Daa2[p,r] += (Daaaa[p,q,q,r] + Dabba[p,q,q,r])
                Dbb2[p,r] += (Dbbbb[p,q,q,r] + Dbaab[p,q,q,r])
    Daa2 = Daa2/(N-1)
    Dbb2 = Dbb2/(N-1)
    prints("== check 2RDM -> 1RDM ==")
    printmat(Quket.DA, name="Da")
    printmat(Daa2, name="Da from 2RDM")
    printmat(Quket.DB, name="Db")
    printmat(Dbb2, name="Db from 2RDM")

def rdm3_to_rdm2(Quket, Daaaaaa = None, Dbbbbbb = None,
                 Dbaaaab = None, Dbbaabb = None):
    ### check ###
    print("Check 3RDM -> 2RDM \n")
    norbs = Quket.n_orbitals
    N = Quket.n_electrons

    if (Daaaaaa is None ):
        Daaaaaa = Quket.Daaaaaa
        Dbbbbbb = Quket.Dbbbbbb
        Dbbaabb = Quket.Dbbaabb
        Dbaaaab = Quket.Dbaaaab

    Dbabbab = Dbbaabb.transpose(0,2,1,4,3,5)
    Daabbaa = Dbaaaab.transpose(2,1,0,5,4,3)
    Daaaa2 = np.zeros((norbs,norbs,norbs,norbs))
    Dbbbb2 = np.zeros((norbs,norbs,norbs,norbs))
    Dbaab2 = np.zeros((norbs,norbs,norbs,norbs))
    for p in range(norbs):
        for q in range(norbs):
            for s in range(norbs):
                for t in range(norbs):
                    for r in range(norbs):
                        Daaaa2[p,q,s,t] += (Daaaaaa[p,q,r,r,s,t] + Daabbaa[p,q,r,r,s,t])
                        Dbbbb2[p,q,s,t] += (Dbbbbbb[p,q,r,r,s,t] + Dbbaabb[p,q,r,r,s,t])
                        Dbaab2[p,q,s,t] += (Dbaaaab[p,q,r,r,s,t] + Dbabbab[p,q,r,r,s,t])
    Daaaa2 = Daaaa2/(N-2)
    Dbbbb2 = Dbbbb2/(N-2)
    Dbaab2 = Dbaab2/(N-2)
    #print("Daaaa - Daaaa(3RDM) \n \n",Quket.Daaaa - Daaaa2)
    #print("\n Daaaa from 3RDM \n \n",Daaaa2)
    prints("== check 3RDM -> 2RDM ==")

def rdm4_to_rdm3(Quket):
    ### check ###
    norbs = Quket.n_orbitals
    N = Quket.n_electrons
    print("Check 4RDM -> 3RDM \n")
    Daaaaaaaa = Quket.Daaaaaaaa
    Dbbbbbbbb = Quket.Dbbbbbbbb
    Dbaaaaaab = Quket.Dbaaaaaab
    Dbbaaaabb = Quket.Dbbaaaabb
    Dbbbaabbb = Quket.Dbbbaabbb

    Daaabbaaa = Dbaaaaaab.transpose(3, 1, 2, 0, 5, 6, 7, 4)
    Dbaabbaab = Dbbaaaabb.transpose(0, 3, 2, 1, 6, 5, 4, 7)
    Dbbabbabb = Dbbbaabbb.transpose(0, 1, 3, 2, 5, 4, 6, 7)

    Daaaaaa2 = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs))
    Dbbbbbb2 = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs))
    Dbaaaab2 = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs))
    Dbbaabb2 = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs))
    for p in range(norbs):
        for q in range(norbs):
            for r in range(norbs):
                for t in range(norbs):
                    for u in range(norbs):
                        for v in range(norbs):
                            for s in range(norbs):
                                Daaaaaa2[p,q,r,t,u,v] += (Daaaaaaaa[p,q,r,s,s,t,u,v] + Daaabbaaa[p,q,r,s,s,t,u,v])
                                Dbbbbbb2[p,q,r,t,u,v] += (Dbbbbbbbb[p,q,r,s,s,t,u,v] + Dbbbaabbb[p,q,r,s,s,t,u,v])
                                Dbaaaab2[p,q,r,t,u,v] += (Dbaaaaaab[p,q,r,s,s,t,u,v] + Dbaabbaab[p,q,r,s,s,t,u,v])
                                Dbbaabb2[p,q,r,t,u,v] += (Dbbaaaabb[p,q,r,s,s,t,u,v] + Dbbabbabb[p,q,r,s,s,t,u,v])
    Daaaaaa2 = Daaaaaa2/(N-3)
    Dbbbbbb2 = Dbbbbbb2/(N-3)
    Dbbaabb2 = Dbbaabb2/(N-3)
    Dbaaaab2 = Dbaaaab2/(N-3)
    prints("Check 4RDM -> 3RDM \n")
    rdm3_to_rdm2(Quket, Daaaaaa2, Dbbbbbb2, Dbaaaab2, Dbbaabb2)

def get_Generalized_Fock_Matrix(Quket):
    if Quket.Dbaab is None:
        Quket.get_2RDM()

    Daaaa = Quket.Daaaa
    Dbbbb = Quket.Dbbbb
    Dbaab = Quket.Dbaab
    Daa = Quket.DA
    Dbb = Quket.DB
    nact = Quket.n_active_orbitals
    ncore = Quket.n_frozen_orbitals
    hpq = Quket.one_body_integrals
    hpqrs = Quket.two_body_integrals
    f_ut_a = np.zeros((norbs, norbs))
    f_ut_b = np.zeros((norbs, norbs))
    Dabba = Dbaab.transpose(1,0,3,2)

    for u in range(norbs):
        for t in range(norbs):
            for p in range(norbs):
                f_ut_a[u, t] += Daa[u, p]*hpq[p, t]
                f_ut_b[u, t] += Dbb[u, p]*hpq[p, t]
    for u in range(norbs):
        for t in range(norbs):
            for q in range(norbs):
                for r in range(norbs):
                    for p in range(norbs):
                        f_ut_a[u,t] += hpqrs[p,q,r,t]*Daaaa[p,r,u,q] + hpqrs[p,q,r,t]*Dbaab[p,r,u,q]
                        f_ut_b[u,t] += hpqrs[p,q,r,t]*Dbbbb[p,r,u,q] + hpqrs[p,q,r,t]*Dabba[p,r,u,q]

    return f_ut_a.T, f_ut_b.T

def get_Generalized_Fock_Matrix_one_body(Quket, act=True):
    Daa = Quket.DA
    Dbb = Quket.DB
    #Daa = Quket.RelDA
    #Dbb = Quket.RelDB
    if Daa is None or Dbb is None:
        raise ValueError('Daa is None in get_Generalized_Fock_Matrix_one_body')
    norbs = Quket.n_orbitals
    n_active_orbs = Quket.n_active_orbitals
    n_frozen_orbs = Quket.n_frozen_orbitals
    hpq = Quket.one_body_integrals
    hpqrs = Quket.two_body_integrals
    fa = copy.deepcopy(hpq)
    fb = copy.deepcopy(hpq)

    for p in range(norbs):
        for q in range(norbs):
            for r in range(norbs):
                for s in range(norbs):
                    fa[p,q] += hpqrs[p,q,r,s]*(Daa[r,s] + Dbb[r,s]) - hpqrs[p,s,r,q]*Daa[r,s]
                    fb[p,q] += hpqrs[p,q,r,s]*(Daa[r,s] + Dbb[r,s]) - hpqrs[p,s,r,q]*Dbb[r,s]
    if act:
        fa = fa[n_frozen_orbs:n_frozen_orbs+n_active_orbs, n_frozen_orbs:n_frozen_orbs+n_active_orbs]
        fb = fb[n_frozen_orbs:n_frozen_orbs+n_active_orbs, n_frozen_orbs:n_frozen_orbs+n_active_orbs]
    return fa, fb

def get_1RDM_full(state, mapping='jordan_wigner'):
    """Function
    Compute full-spin 1RDM of QuantmState `state` in QuketData.
    Indices correspond to qubit number.
    No MPI implementation.

    Author(s): Takashi Tsuchimochi
    """
    n_qubits = state.get_qubit_count()

    D1 = np.zeros((n_qubits, n_qubits))
    state_list = []
    for p in range(n_qubits):
        string = f"{p}"
        op = FermionOperator(string)
        state_list.append(evolve(op, state, mapping=mapping))
        for q in range(n_qubits):
            if p < q or (p%2!=q%2):
                continue
            val = inner_product(state_list[p], state_list[q]).real
            D1[p, q] = val
            D1[q, p] = val
    return D1

def get_2RDM_full(state, mapping='jordan_wigner'):
    """Function
    Compute full-spin 2RDM of QuantmState `state` in QuketData.
    Indices correspond to qubit number.
    To be explicit,
    D2[p,q,r,s] = < p^ q^ r s >

    Args:
        state (QuantumState): State for which 2RDM is computed
    Returns:
        D2 (4darray): D2[p,q,r,s] = < p^ q^ r s > in qubit basis

    Author(s): Takashi Tsuchimochi
    """
    n_qubits = state.get_qubit_count()

    D2 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    my_D2 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    state_list=[]
    pqrs = -1
    pq = -1
    for p in range(n_qubits):
        for q in range(p):
            pq += 1
            string = f"{p} {q}"
            op = FermionOperator(string)
            state_list.append(evolve(op, state, mapping=mapping))

            rs = -1
            for r in range(n_qubits):
                for s in range(r):
                    rs += 1
                    if pq < rs or (p%2+q%2!=r%2+s%2):
                        continue
                    pqrs += 1
                    if pqrs % mpi.nprocs == mpi.rank:
                        # Here we take the inner-product between
                        # p q|phi>  and  r s|phi>
                        # -> <phi| q^ p^ r s |phi> = - <phi|p^ q^ r s|phi>
                        val = -inner_product(state_list[pq], state_list[rs]).real
                        my_D2[p, q, r, s] = val
                        my_D2[p, q, s, r] = -val
                        my_D2[q, p, r, s] = -val
                        my_D2[q, p, s, r] = val
                        my_D2[ r, s, p, q] = val
                        my_D2[ r, s, q, p] = -val
                        my_D2[ s, r, p, q] = -val
                        my_D2[ s, r, q, p] = val
    D2 = mpi.allreduce(my_D2, mpi.MPI.SUM)
    return D2

def get_3RDM_full(state, mapping='jordan_wigner'):
    """Function
    Compute 3RDM of QuantmState `state` in QuketData.
    Indices correspond to qubit number.
    To be explicit,
    D3[p,q,r,s,t,u] = < p^ q^ r^ s t u >

    Args:
        state (QuantumState): State for which 3RDM is computed
    Returns:
        D3 (6darray): D3[p,q,r,s,t,u] = < p^ q^ r^ s t u > in qubit basis


    Author(s): Takashi Tsuchimochi
    """
    n_qubits = state.get_qubit_count()

    D3 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits))
    my_D3 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits))
    t1 = time.time()
    state_list = []
    pqrstu = -1
    pqr = -1
    for p in range(n_qubits):
        #for q in range(n_qubit):
        for q in range(p):
            for r in range(q):
                pqr += 1
                string = f"{p} {q} {r}"
                op = FermionOperator(string)
                state_list.append(evolve(op, state, mapping=mapping))
                stu = -1
                for s in range(n_qubits):
                    for t in range(s):
                        for u in range(t):
                            stu += 1
                            if pqr < stu or (p%2+q%2+r%2!=s%2+t%2+u%2):
                                continue
                            pqrstu += 1
                            if pqrstu % mpi.nprocs == mpi.rank:
                                # Here we take the inner-product between
                                # p q r|phi>  and  s t u|phi>
                                # -> <phi| r^ q^ p^ s t u |phi> = - <phi|p^ q^ r^ s t u|phi>
                                val = -inner_product(state_list[pqr], state_list[stu]).real
                                my_D3[p, q, r, s, t, u] = val
                                my_D3[p, q, r, s, u, t] = -val
                                my_D3[p, q, r, u, t, s] = -val
                                my_D3[p, q, r, u, s, t] = val
                                my_D3[p, q, r, t, s, u] = -val
                                my_D3[p, q, r, t, u, s] = val

                                my_D3[p, r, q, s, t, u] = -val
                                my_D3[p, r, q, s, u, t] = val
                                my_D3[p, r, q, u, t, s] = val
                                my_D3[p, r, q, u, s, t] = -val
                                my_D3[p, r, q, t, s, u] = val
                                my_D3[p, r, q, t, u, s] = -val

                                my_D3[r, p, q, s, t, u] = val
                                my_D3[r, p, q, s, u, t] = -val
                                my_D3[r, p, q, u, t, s] = -val
                                my_D3[r, p, q, u, s, t] = val
                                my_D3[r, p, q, t, s, u] = -val
                                my_D3[r, p, q, t, u, s] = val

                                my_D3[r, q, p, s, t, u] = -val
                                my_D3[r, q, p, s, u, t] = val
                                my_D3[r, q, p, u, t, s] = val
                                my_D3[r, q, p, u, s, t] = -val
                                my_D3[r, q, p, t, s, u] = val
                                my_D3[r, q, p, t, u, s] = -val

                                my_D3[q, r, p, s, t, u] = val
                                my_D3[q, r, p, s, u, t] = -val
                                my_D3[q, r, p, u, t, s] = -val
                                my_D3[q, r, p, u, s, t] = val
                                my_D3[q, r, p, t, s, u] = -val
                                my_D3[q, r, p, t, u, s] = val

                                my_D3[q, p, r, s, t, u] = -val
                                my_D3[q, p, r, s, u, t] = val
                                my_D3[q, p, r, u, t, s] = val
                                my_D3[q, p, r, u, s, t] = -val
                                my_D3[q, p, r, t, s, u] = val
                                my_D3[q, p, r, t, u, s] = -val
### pqr <-> stu
                                my_D3[s, t, u, p, q, r] = val
                                my_D3[s, u, t, p, q, r] = -val
                                my_D3[u, t, s, p, q, r] = -val
                                my_D3[u, s, t, p, q, r] = val
                                my_D3[t, s, u, p, q, r] = -val
                                my_D3[t, u, s, p, q, r] = val

                                my_D3[s, t, u, p, r, q] = -val
                                my_D3[s, u, t, p, r, q] = val
                                my_D3[u, t, s, p, r, q] = val
                                my_D3[u, s, t, p, r, q] = -val
                                my_D3[t, s, u, p, r, q] = val
                                my_D3[t, u, s, p, r, q] = -val

                                my_D3[s, t, u, r, p, q] = val
                                my_D3[s, u, t, r, p, q] = -val
                                my_D3[u, t, s, r, p, q] = -val
                                my_D3[u, s, t, r, p, q] = val
                                my_D3[t, s, u, r, p, q] = -val
                                my_D3[t, u, s, r, p, q] = val

                                my_D3[s, t, u, r, q, p] = -val
                                my_D3[s, u, t, r, q, p] = val
                                my_D3[u, t, s, r, q, p] = val
                                my_D3[u, s, t, r, q, p] = -val
                                my_D3[t, s, u, r, q, p] = val
                                my_D3[t, u, s, r, q, p] = -val

                                my_D3[s, t, u, q, r, p] = val
                                my_D3[s, u, t, q, r, p] = -val
                                my_D3[u, t, s, q, r, p] = -val
                                my_D3[u, s, t, q, r, p] = val
                                my_D3[t, s, u, q, r, p] = -val
                                my_D3[t, u, s, q, r, p] = val

                                my_D3[s, t, u, q, p, r] = -val
                                my_D3[s, u, t, q, p, r] = val
                                my_D3[u, t, s, q, p, r] = val
                                my_D3[u, s, t, q, p, r] = -val
                                my_D3[t, s, u, q, p, r] = val
                                my_D3[t, u, s, q, p, r] = -val

    D3 = mpi.allreduce(my_D3, mpi.MPI.SUM)
    t2 = time.time()
    prints(t2-t1)
    return D3

def get_4RDM_full(state, mapping='jordan_wigner'):
    """Function
    Compute 4RDM of QuantmState `state` in QuketData.
    Indices correspond to qubit number.
    To be explicit,
    D4[p,q,r,s,t,u,v,w] = < p^ q^ r^ s^ t u v w >

    Args:
        state (QuantumState): State for which 4RDM is computed
    Returns:
        D4 (8darray): D4[p,q,r,s,t,u,v,w] = < p^ q^ r^ s^ t u v w > in qubit basis


    Author(s): Takashi Tsuchimochi
    """
    n_qubits = state.get_qubit_count()

    D4 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits))
    my_D4 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits))
    state_list = []
    pqrstuvw = -1
    pqrs = -1
    for p in range(n_qubits):
        for q in range(p):
            for r in range(q):
                for s in range(r):
                    pqrs += 1
                    string = f"{p} {q} {r} {s}"
                    op = FermionOperator(string)
                    state_list.append(evolve(op, state, mapping=mapping))
                    tuvw = -1
                    for t in range(n_qubits):
                        for u in range(t):
                            for v in range(u):
                                for w in range(v):
                                    tuvw += 1
                                    if pqrs < tuvw or (p%2+q%2+r%2+s%2!=t%2+u%2+v%2+w%2):
                                        continue
                                    pqrstuvw += 1
                                    if pqrstuvw % mpi.nprocs == mpi.rank:
                                        # Here we take the inner-product between
                                        # p q r s|phi>  and  t u v w|phi>
                                        # -> <phi| s^ r^ q^ p^ t u v w |phi> = <phi|p^ q^ r^ s^ t u v w|phi>
                                        val = inner_product(state_list[pqrs], state_list[tuvw]).real
                                        if abs(val) < 1e-8:
                                            continue
# p,q,r,s
                                        par = 1
                                        my_D4[p, q, r, s, t, u, v, w] = val*par
                                        my_D4[p, q, r, s, t, u, w, v] =-val*par
                                        my_D4[p, q, r, s, t, v, u, w] =-val*par
                                        my_D4[p, q, r, s, t, v, w, u] = val*par
                                        my_D4[p, q, r, s, t, w, v, u] =-val*par
                                        my_D4[p, q, r, s, t, w, u, v] = val*par
                                        my_D4[p, q, r, s, u, t, v, w] =-val*par
                                        my_D4[p, q, r, s, u, t, w, v] =+val*par
                                        my_D4[p, q, r, s, u, v, t, w] =+val*par
                                        my_D4[p, q, r, s, u, v, w, t] =-val*par
                                        my_D4[p, q, r, s, u, w, v, t] =+val*par
                                        my_D4[p, q, r, s, u, w, t, v] =-val*par
                                        my_D4[p, q, r, s, v, u, t, w] =-val*par
                                        my_D4[p, q, r, s, v, u, w, t] = val*par
                                        my_D4[p, q, r, s, v, t, u, w] = val*par
                                        my_D4[p, q, r, s, v, t, w, u] =-val*par
                                        my_D4[p, q, r, s, v, w, t, u] = val*par
                                        my_D4[p, q, r, s, v, w, u, t] =-val*par
                                        my_D4[p, q, r, s, w, u, v, t] =-val*par
                                        my_D4[p, q, r, s, w, u, t, v] = val*par
                                        my_D4[p, q, r, s, w, v, u, t] = val*par
                                        my_D4[p, q, r, s, w, v, t, u] =-val*par
                                        my_D4[p, q, r, s, w, t, v, u] = val*par
                                        my_D4[p, q, r, s, w, t, u, v] =-val*par

                                        par = -1
                                        my_D4[p, q, s, r, t, u, v, w] = val*par
                                        my_D4[p, q, s, r, t, u, w, v] =-val*par
                                        my_D4[p, q, s, r, t, v, u, w] =-val*par
                                        my_D4[p, q, s, r, t, v, w, u] = val*par
                                        my_D4[p, q, s, r, t, w, v, u] =-val*par
                                        my_D4[p, q, s, r, t, w, u, v] = val*par
                                        my_D4[p, q, s, r, u, t, v, w] =-val*par
                                        my_D4[p, q, s, r, u, t, w, v] =+val*par
                                        my_D4[p, q, s, r, u, v, t, w] =+val*par
                                        my_D4[p, q, s, r, u, v, w, t] =-val*par
                                        my_D4[p, q, s, r, u, w, v, t] =+val*par
                                        my_D4[p, q, s, r, u, w, t, v] =-val*par
                                        my_D4[p, q, s, r, v, u, t, w] =-val*par
                                        my_D4[p, q, s, r, v, u, w, t] = val*par
                                        my_D4[p, q, s, r, v, t, u, w] = val*par
                                        my_D4[p, q, s, r, v, t, w, u] =-val*par
                                        my_D4[p, q, s, r, v, w, t, u] = val*par
                                        my_D4[p, q, s, r, v, w, u, t] =-val*par
                                        my_D4[p, q, s, r, w, u, v, t] =-val*par
                                        my_D4[p, q, s, r, w, u, t, v] = val*par
                                        my_D4[p, q, s, r, w, v, u, t] = val*par
                                        my_D4[p, q, s, r, w, v, t, u] =-val*par
                                        my_D4[p, q, s, r, w, t, v, u] = val*par
                                        my_D4[p, q, s, r, w, t, u, v] =-val*par

                                        par = -1
                                        my_D4[p, r, q, s, t, u, v, w] = val*par
                                        my_D4[p, r, q, s, t, u, w, v] =-val*par
                                        my_D4[p, r, q, s, t, v, u, w] =-val*par
                                        my_D4[p, r, q, s, t, v, w, u] = val*par
                                        my_D4[p, r, q, s, t, w, v, u] =-val*par
                                        my_D4[p, r, q, s, t, w, u, v] = val*par
                                        my_D4[p, r, q, s, u, t, v, w] =-val*par
                                        my_D4[p, r, q, s, u, t, w, v] =+val*par
                                        my_D4[p, r, q, s, u, v, t, w] =+val*par
                                        my_D4[p, r, q, s, u, v, w, t] =-val*par
                                        my_D4[p, r, q, s, u, w, v, t] =+val*par
                                        my_D4[p, r, q, s, u, w, t, v] =-val*par
                                        my_D4[p, r, q, s, v, u, t, w] =-val*par
                                        my_D4[p, r, q, s, v, u, w, t] = val*par
                                        my_D4[p, r, q, s, v, t, u, w] = val*par
                                        my_D4[p, r, q, s, v, t, w, u] =-val*par
                                        my_D4[p, r, q, s, v, w, t, u] = val*par
                                        my_D4[p, r, q, s, v, w, u, t] =-val*par
                                        my_D4[p, r, q, s, w, u, v, t] =-val*par
                                        my_D4[p, r, q, s, w, u, t, v] = val*par
                                        my_D4[p, r, q, s, w, v, u, t] = val*par
                                        my_D4[p, r, q, s, w, v, t, u] =-val*par
                                        my_D4[p, r, q, s, w, t, v, u] = val*par
                                        my_D4[p, r, q, s, w, t, u, v] =-val*par

                                        par = 1
                                        my_D4[p, r, s, q, t, u, v, w] = val*par
                                        my_D4[p, r, s, q, t, u, w, v] =-val*par
                                        my_D4[p, r, s, q, t, v, u, w] =-val*par
                                        my_D4[p, r, s, q, t, v, w, u] = val*par
                                        my_D4[p, r, s, q, t, w, v, u] =-val*par
                                        my_D4[p, r, s, q, t, w, u, v] = val*par
                                        my_D4[p, r, s, q, u, t, v, w] =-val*par
                                        my_D4[p, r, s, q, u, t, w, v] =+val*par
                                        my_D4[p, r, s, q, u, v, t, w] =+val*par
                                        my_D4[p, r, s, q, u, v, w, t] =-val*par
                                        my_D4[p, r, s, q, u, w, v, t] =+val*par
                                        my_D4[p, r, s, q, u, w, t, v] =-val*par
                                        my_D4[p, r, s, q, v, u, t, w] =-val*par
                                        my_D4[p, r, s, q, v, u, w, t] = val*par
                                        my_D4[p, r, s, q, v, t, u, w] = val*par
                                        my_D4[p, r, s, q, v, t, w, u] =-val*par
                                        my_D4[p, r, s, q, v, w, t, u] = val*par
                                        my_D4[p, r, s, q, v, w, u, t] =-val*par
                                        my_D4[p, r, s, q, w, u, v, t] =-val*par
                                        my_D4[p, r, s, q, w, u, t, v] = val*par
                                        my_D4[p, r, s, q, w, v, u, t] = val*par
                                        my_D4[p, r, s, q, w, v, t, u] =-val*par
                                        my_D4[p, r, s, q, w, t, v, u] = val*par
                                        my_D4[p, r, s, q, w, t, u, v] =-val*par

                                        par = -1
                                        my_D4[p, s, r, q, t, u, v, w] = val*par
                                        my_D4[p, s, r, q, t, u, w, v] =-val*par
                                        my_D4[p, s, r, q, t, v, u, w] =-val*par
                                        my_D4[p, s, r, q, t, v, w, u] = val*par
                                        my_D4[p, s, r, q, t, w, v, u] =-val*par
                                        my_D4[p, s, r, q, t, w, u, v] = val*par
                                        my_D4[p, s, r, q, u, t, v, w] =-val*par
                                        my_D4[p, s, r, q, u, t, w, v] =+val*par
                                        my_D4[p, s, r, q, u, v, t, w] =+val*par
                                        my_D4[p, s, r, q, u, v, w, t] =-val*par
                                        my_D4[p, s, r, q, u, w, v, t] =+val*par
                                        my_D4[p, s, r, q, u, w, t, v] =-val*par
                                        my_D4[p, s, r, q, v, u, t, w] =-val*par
                                        my_D4[p, s, r, q, v, u, w, t] = val*par
                                        my_D4[p, s, r, q, v, t, u, w] = val*par
                                        my_D4[p, s, r, q, v, t, w, u] =-val*par
                                        my_D4[p, s, r, q, v, w, t, u] = val*par
                                        my_D4[p, s, r, q, v, w, u, t] =-val*par
                                        my_D4[p, s, r, q, w, u, v, t] =-val*par
                                        my_D4[p, s, r, q, w, u, t, v] = val*par
                                        my_D4[p, s, r, q, w, v, u, t] = val*par
                                        my_D4[p, s, r, q, w, v, t, u] =-val*par
                                        my_D4[p, s, r, q, w, t, v, u] = val*par
                                        my_D4[p, s, r, q, w, t, u, v] =-val*par

                                        par = 1
                                        my_D4[p, s, q, r, t, u, v, w] = val*par
                                        my_D4[p, s, q, r, t, u, w, v] =-val*par
                                        my_D4[p, s, q, r, t, v, u, w] =-val*par
                                        my_D4[p, s, q, r, t, v, w, u] = val*par
                                        my_D4[p, s, q, r, t, w, v, u] =-val*par
                                        my_D4[p, s, q, r, t, w, u, v] = val*par
                                        my_D4[p, s, q, r, u, t, v, w] =-val*par
                                        my_D4[p, s, q, r, u, t, w, v] =+val*par
                                        my_D4[p, s, q, r, u, v, t, w] =+val*par
                                        my_D4[p, s, q, r, u, v, w, t] =-val*par
                                        my_D4[p, s, q, r, u, w, v, t] =+val*par
                                        my_D4[p, s, q, r, u, w, t, v] =-val*par
                                        my_D4[p, s, q, r, v, u, t, w] =-val*par
                                        my_D4[p, s, q, r, v, u, w, t] = val*par
                                        my_D4[p, s, q, r, v, t, u, w] = val*par
                                        my_D4[p, s, q, r, v, t, w, u] =-val*par
                                        my_D4[p, s, q, r, v, w, t, u] = val*par
                                        my_D4[p, s, q, r, v, w, u, t] =-val*par
                                        my_D4[p, s, q, r, w, u, v, t] =-val*par
                                        my_D4[p, s, q, r, w, u, t, v] = val*par
                                        my_D4[p, s, q, r, w, v, u, t] = val*par
                                        my_D4[p, s, q, r, w, v, t, u] =-val*par
                                        my_D4[p, s, q, r, w, t, v, u] = val*par
                                        my_D4[p, s, q, r, w, t, u, v] =-val*par

### q, [p, r, s] ...
                                        par = -1
                                        my_D4[q, p, r, s, t, u, v, w] = val*par
                                        my_D4[q, p, r, s, t, u, w, v] =-val*par
                                        my_D4[q, p, r, s, t, v, u, w] =-val*par
                                        my_D4[q, p, r, s, t, v, w, u] = val*par
                                        my_D4[q, p, r, s, t, w, v, u] =-val*par
                                        my_D4[q, p, r, s, t, w, u, v] = val*par
                                        my_D4[q, p, r, s, u, t, v, w] =-val*par
                                        my_D4[q, p, r, s, u, t, w, v] =+val*par
                                        my_D4[q, p, r, s, u, v, t, w] =+val*par
                                        my_D4[q, p, r, s, u, v, w, t] =-val*par
                                        my_D4[q, p, r, s, u, w, v, t] =+val*par
                                        my_D4[q, p, r, s, u, w, t, v] =-val*par
                                        my_D4[q, p, r, s, v, u, t, w] =-val*par
                                        my_D4[q, p, r, s, v, u, w, t] = val*par
                                        my_D4[q, p, r, s, v, t, u, w] = val*par
                                        my_D4[q, p, r, s, v, t, w, u] =-val*par
                                        my_D4[q, p, r, s, v, w, t, u] = val*par
                                        my_D4[q, p, r, s, v, w, u, t] =-val*par
                                        my_D4[q, p, r, s, w, u, v, t] =-val*par
                                        my_D4[q, p, r, s, w, u, t, v] = val*par
                                        my_D4[q, p, r, s, w, v, u, t] = val*par
                                        my_D4[q, p, r, s, w, v, t, u] =-val*par
                                        my_D4[q, p, r, s, w, t, v, u] = val*par
                                        my_D4[q, p, r, s, w, t, u, v] =-val*par

                                        par = 1
                                        my_D4[q, p, s, r, t, u, v, w] = val*par
                                        my_D4[q, p, s, r, t, u, w, v] =-val*par
                                        my_D4[q, p, s, r, t, v, u, w] =-val*par
                                        my_D4[q, p, s, r, t, v, w, u] = val*par
                                        my_D4[q, p, s, r, t, w, v, u] =-val*par
                                        my_D4[q, p, s, r, t, w, u, v] = val*par
                                        my_D4[q, p, s, r, u, t, v, w] =-val*par
                                        my_D4[q, p, s, r, u, t, w, v] =+val*par
                                        my_D4[q, p, s, r, u, v, t, w] =+val*par
                                        my_D4[q, p, s, r, u, v, w, t] =-val*par
                                        my_D4[q, p, s, r, u, w, v, t] =+val*par
                                        my_D4[q, p, s, r, u, w, t, v] =-val*par
                                        my_D4[q, p, s, r, v, u, t, w] =-val*par
                                        my_D4[q, p, s, r, v, u, w, t] = val*par
                                        my_D4[q, p, s, r, v, t, u, w] = val*par
                                        my_D4[q, p, s, r, v, t, w, u] =-val*par
                                        my_D4[q, p, s, r, v, w, t, u] = val*par
                                        my_D4[q, p, s, r, v, w, u, t] =-val*par
                                        my_D4[q, p, s, r, w, u, v, t] =-val*par
                                        my_D4[q, p, s, r, w, u, t, v] = val*par
                                        my_D4[q, p, s, r, w, v, u, t] = val*par
                                        my_D4[q, p, s, r, w, v, t, u] =-val*par
                                        my_D4[q, p, s, r, w, t, v, u] = val*par
                                        my_D4[q, p, s, r, w, t, u, v] =-val*par

                                        par = 1
                                        my_D4[q, r, p, s, t, u, v, w] = val*par
                                        my_D4[q, r, p, s, t, u, w, v] =-val*par
                                        my_D4[q, r, p, s, t, v, u, w] =-val*par
                                        my_D4[q, r, p, s, t, v, w, u] = val*par
                                        my_D4[q, r, p, s, t, w, v, u] =-val*par
                                        my_D4[q, r, p, s, t, w, u, v] = val*par
                                        my_D4[q, r, p, s, u, t, v, w] =-val*par
                                        my_D4[q, r, p, s, u, t, w, v] =+val*par
                                        my_D4[q, r, p, s, u, v, t, w] =+val*par
                                        my_D4[q, r, p, s, u, v, w, t] =-val*par
                                        my_D4[q, r, p, s, u, w, v, t] =+val*par
                                        my_D4[q, r, p, s, u, w, t, v] =-val*par
                                        my_D4[q, r, p, s, v, u, t, w] =-val*par
                                        my_D4[q, r, p, s, v, u, w, t] = val*par
                                        my_D4[q, r, p, s, v, t, u, w] = val*par
                                        my_D4[q, r, p, s, v, t, w, u] =-val*par
                                        my_D4[q, r, p, s, v, w, t, u] = val*par
                                        my_D4[q, r, p, s, v, w, u, t] =-val*par
                                        my_D4[q, r, p, s, w, u, v, t] =-val*par
                                        my_D4[q, r, p, s, w, u, t, v] = val*par
                                        my_D4[q, r, p, s, w, v, u, t] = val*par
                                        my_D4[q, r, p, s, w, v, t, u] =-val*par
                                        my_D4[q, r, p, s, w, t, v, u] = val*par
                                        my_D4[q, r, p, s, w, t, u, v] =-val*par

                                        par = -1
                                        my_D4[q, r, s, p, t, u, v, w] = val*par
                                        my_D4[q, r, s, p, t, u, w, v] =-val*par
                                        my_D4[q, r, s, p, t, v, u, w] =-val*par
                                        my_D4[q, r, s, p, t, v, w, u] = val*par
                                        my_D4[q, r, s, p, t, w, v, u] =-val*par
                                        my_D4[q, r, s, p, t, w, u, v] = val*par
                                        my_D4[q, r, s, p, u, t, v, w] =-val*par
                                        my_D4[q, r, s, p, u, t, w, v] =+val*par
                                        my_D4[q, r, s, p, u, v, t, w] =+val*par
                                        my_D4[q, r, s, p, u, v, w, t] =-val*par
                                        my_D4[q, r, s, p, u, w, v, t] =+val*par
                                        my_D4[q, r, s, p, u, w, t, v] =-val*par
                                        my_D4[q, r, s, p, v, u, t, w] =-val*par
                                        my_D4[q, r, s, p, v, u, w, t] = val*par
                                        my_D4[q, r, s, p, v, t, u, w] = val*par
                                        my_D4[q, r, s, p, v, t, w, u] =-val*par
                                        my_D4[q, r, s, p, v, w, t, u] = val*par
                                        my_D4[q, r, s, p, v, w, u, t] =-val*par
                                        my_D4[q, r, s, p, w, u, v, t] =-val*par
                                        my_D4[q, r, s, p, w, u, t, v] = val*par
                                        my_D4[q, r, s, p, w, v, u, t] = val*par
                                        my_D4[q, r, s, p, w, v, t, u] =-val*par
                                        my_D4[q, r, s, p, w, t, v, u] = val*par
                                        my_D4[q, r, s, p, w, t, u, v] =-val*par

                                        par = 1
                                        my_D4[q, s, r, p, t, u, v, w] = val*par
                                        my_D4[q, s, r, p, t, u, w, v] =-val*par
                                        my_D4[q, s, r, p, t, v, u, w] =-val*par
                                        my_D4[q, s, r, p, t, v, w, u] = val*par
                                        my_D4[q, s, r, p, t, w, v, u] =-val*par
                                        my_D4[q, s, r, p, t, w, u, v] = val*par
                                        my_D4[q, s, r, p, u, t, v, w] =-val*par
                                        my_D4[q, s, r, p, u, t, w, v] =+val*par
                                        my_D4[q, s, r, p, u, v, t, w] =+val*par
                                        my_D4[q, s, r, p, u, v, w, t] =-val*par
                                        my_D4[q, s, r, p, u, w, v, t] =+val*par
                                        my_D4[q, s, r, p, u, w, t, v] =-val*par
                                        my_D4[q, s, r, p, v, u, t, w] =-val*par
                                        my_D4[q, s, r, p, v, u, w, t] = val*par
                                        my_D4[q, s, r, p, v, t, u, w] = val*par
                                        my_D4[q, s, r, p, v, t, w, u] =-val*par
                                        my_D4[q, s, r, p, v, w, t, u] = val*par
                                        my_D4[q, s, r, p, v, w, u, t] =-val*par
                                        my_D4[q, s, r, p, w, u, v, t] =-val*par
                                        my_D4[q, s, r, p, w, u, t, v] = val*par
                                        my_D4[q, s, r, p, w, v, u, t] = val*par
                                        my_D4[q, s, r, p, w, v, t, u] =-val*par
                                        my_D4[q, s, r, p, w, t, v, u] = val*par
                                        my_D4[q, s, r, p, w, t, u, v] =-val*par

                                        par = -1
                                        my_D4[q, s, p, r, t, u, v, w] = val*par
                                        my_D4[q, s, p, r, t, u, w, v] =-val*par
                                        my_D4[q, s, p, r, t, v, u, w] =-val*par
                                        my_D4[q, s, p, r, t, v, w, u] = val*par
                                        my_D4[q, s, p, r, t, w, v, u] =-val*par
                                        my_D4[q, s, p, r, t, w, u, v] = val*par
                                        my_D4[q, s, p, r, u, t, v, w] =-val*par
                                        my_D4[q, s, p, r, u, t, w, v] =+val*par
                                        my_D4[q, s, p, r, u, v, t, w] =+val*par
                                        my_D4[q, s, p, r, u, v, w, t] =-val*par
                                        my_D4[q, s, p, r, u, w, v, t] =+val*par
                                        my_D4[q, s, p, r, u, w, t, v] =-val*par
                                        my_D4[q, s, p, r, v, u, t, w] =-val*par
                                        my_D4[q, s, p, r, v, u, w, t] = val*par
                                        my_D4[q, s, p, r, v, t, u, w] = val*par
                                        my_D4[q, s, p, r, v, t, w, u] =-val*par
                                        my_D4[q, s, p, r, v, w, t, u] = val*par
                                        my_D4[q, s, p, r, v, w, u, t] =-val*par
                                        my_D4[q, s, p, r, w, u, v, t] =-val*par
                                        my_D4[q, s, p, r, w, u, t, v] = val*par
                                        my_D4[q, s, p, r, w, v, u, t] = val*par
                                        my_D4[q, s, p, r, w, v, t, u] =-val*par
                                        my_D4[q, s, p, r, w, t, v, u] = val*par
                                        my_D4[q, s, p, r, w, t, u, v] =-val*par
# r, [p,q,s]
                                        par = -1
                                        my_D4[r, q, p, s, t, u, v, w] = val*par
                                        my_D4[r, q, p, s, t, u, w, v] =-val*par
                                        my_D4[r, q, p, s, t, v, u, w] =-val*par
                                        my_D4[r, q, p, s, t, v, w, u] = val*par
                                        my_D4[r, q, p, s, t, w, v, u] =-val*par
                                        my_D4[r, q, p, s, t, w, u, v] = val*par
                                        my_D4[r, q, p, s, u, t, v, w] =-val*par
                                        my_D4[r, q, p, s, u, t, w, v] =+val*par
                                        my_D4[r, q, p, s, u, v, t, w] =+val*par
                                        my_D4[r, q, p, s, u, v, w, t] =-val*par
                                        my_D4[r, q, p, s, u, w, v, t] =+val*par
                                        my_D4[r, q, p, s, u, w, t, v] =-val*par
                                        my_D4[r, q, p, s, v, u, t, w] =-val*par
                                        my_D4[r, q, p, s, v, u, w, t] = val*par
                                        my_D4[r, q, p, s, v, t, u, w] = val*par
                                        my_D4[r, q, p, s, v, t, w, u] =-val*par
                                        my_D4[r, q, p, s, v, w, t, u] = val*par
                                        my_D4[r, q, p, s, v, w, u, t] =-val*par
                                        my_D4[r, q, p, s, w, u, v, t] =-val*par
                                        my_D4[r, q, p, s, w, u, t, v] = val*par
                                        my_D4[r, q, p, s, w, v, u, t] = val*par
                                        my_D4[r, q, p, s, w, v, t, u] =-val*par
                                        my_D4[r, q, p, s, w, t, v, u] = val*par
                                        my_D4[r, q, p, s, w, t, u, v] =-val*par

                                        par = 1
                                        my_D4[r, q, s, p, t, u, v, w] = val*par
                                        my_D4[r, q, s, p, t, u, w, v] =-val*par
                                        my_D4[r, q, s, p, t, v, u, w] =-val*par
                                        my_D4[r, q, s, p, t, v, w, u] = val*par
                                        my_D4[r, q, s, p, t, w, v, u] =-val*par
                                        my_D4[r, q, s, p, t, w, u, v] = val*par
                                        my_D4[r, q, s, p, u, t, v, w] =-val*par
                                        my_D4[r, q, s, p, u, t, w, v] =+val*par
                                        my_D4[r, q, s, p, u, v, t, w] =+val*par
                                        my_D4[r, q, s, p, u, v, w, t] =-val*par
                                        my_D4[r, q, s, p, u, w, v, t] =+val*par
                                        my_D4[r, q, s, p, u, w, t, v] =-val*par
                                        my_D4[r, q, s, p, v, u, t, w] =-val*par
                                        my_D4[r, q, s, p, v, u, w, t] = val*par
                                        my_D4[r, q, s, p, v, t, u, w] = val*par
                                        my_D4[r, q, s, p, v, t, w, u] =-val*par
                                        my_D4[r, q, s, p, v, w, t, u] = val*par
                                        my_D4[r, q, s, p, v, w, u, t] =-val*par
                                        my_D4[r, q, s, p, w, u, v, t] =-val*par
                                        my_D4[r, q, s, p, w, u, t, v] = val*par
                                        my_D4[r, q, s, p, w, v, u, t] = val*par
                                        my_D4[r, q, s, p, w, v, t, u] =-val*par
                                        my_D4[r, q, s, p, w, t, v, u] = val*par
                                        my_D4[r, q, s, p, w, t, u, v] =-val*par

                                        par = 1
                                        my_D4[r, p, q, s, t, u, v, w] = val*par
                                        my_D4[r, p, q, s, t, u, w, v] =-val*par
                                        my_D4[r, p, q, s, t, v, u, w] =-val*par
                                        my_D4[r, p, q, s, t, v, w, u] = val*par
                                        my_D4[r, p, q, s, t, w, v, u] =-val*par
                                        my_D4[r, p, q, s, t, w, u, v] = val*par
                                        my_D4[r, p, q, s, u, t, v, w] =-val*par
                                        my_D4[r, p, q, s, u, t, w, v] =+val*par
                                        my_D4[r, p, q, s, u, v, t, w] =+val*par
                                        my_D4[r, p, q, s, u, v, w, t] =-val*par
                                        my_D4[r, p, q, s, u, w, v, t] =+val*par
                                        my_D4[r, p, q, s, u, w, t, v] =-val*par
                                        my_D4[r, p, q, s, v, u, t, w] =-val*par
                                        my_D4[r, p, q, s, v, u, w, t] = val*par
                                        my_D4[r, p, q, s, v, t, u, w] = val*par
                                        my_D4[r, p, q, s, v, t, w, u] =-val*par
                                        my_D4[r, p, q, s, v, w, t, u] = val*par
                                        my_D4[r, p, q, s, v, w, u, t] =-val*par
                                        my_D4[r, p, q, s, w, u, v, t] =-val*par
                                        my_D4[r, p, q, s, w, u, t, v] = val*par
                                        my_D4[r, p, q, s, w, v, u, t] = val*par
                                        my_D4[r, p, q, s, w, v, t, u] =-val*par
                                        my_D4[r, p, q, s, w, t, v, u] = val*par
                                        my_D4[r, p, q, s, w, t, u, v] =-val*par

                                        par = -1
                                        my_D4[r, p, s, q, t, u, v, w] = val*par
                                        my_D4[r, p, s, q, t, u, w, v] =-val*par
                                        my_D4[r, p, s, q, t, v, u, w] =-val*par
                                        my_D4[r, p, s, q, t, v, w, u] = val*par
                                        my_D4[r, p, s, q, t, w, v, u] =-val*par
                                        my_D4[r, p, s, q, t, w, u, v] = val*par
                                        my_D4[r, p, s, q, u, t, v, w] =-val*par
                                        my_D4[r, p, s, q, u, t, w, v] =+val*par
                                        my_D4[r, p, s, q, u, v, t, w] =+val*par
                                        my_D4[r, p, s, q, u, v, w, t] =-val*par
                                        my_D4[r, p, s, q, u, w, v, t] =+val*par
                                        my_D4[r, p, s, q, u, w, t, v] =-val*par
                                        my_D4[r, p, s, q, v, u, t, w] =-val*par
                                        my_D4[r, p, s, q, v, u, w, t] = val*par
                                        my_D4[r, p, s, q, v, t, u, w] = val*par
                                        my_D4[r, p, s, q, v, t, w, u] =-val*par
                                        my_D4[r, p, s, q, v, w, t, u] = val*par
                                        my_D4[r, p, s, q, v, w, u, t] =-val*par
                                        my_D4[r, p, s, q, w, u, v, t] =-val*par
                                        my_D4[r, p, s, q, w, u, t, v] = val*par
                                        my_D4[r, p, s, q, w, v, u, t] = val*par
                                        my_D4[r, p, s, q, w, v, t, u] =-val*par
                                        my_D4[r, p, s, q, w, t, v, u] = val*par
                                        my_D4[r, p, s, q, w, t, u, v] =-val*par

                                        par = 1
                                        my_D4[r, s, p, q, t, u, v, w] = val*par
                                        my_D4[r, s, p, q, t, u, w, v] =-val*par
                                        my_D4[r, s, p, q, t, v, u, w] =-val*par
                                        my_D4[r, s, p, q, t, v, w, u] = val*par
                                        my_D4[r, s, p, q, t, w, v, u] =-val*par
                                        my_D4[r, s, p, q, t, w, u, v] = val*par
                                        my_D4[r, s, p, q, u, t, v, w] =-val*par
                                        my_D4[r, s, p, q, u, t, w, v] =+val*par
                                        my_D4[r, s, p, q, u, v, t, w] =+val*par
                                        my_D4[r, s, p, q, u, v, w, t] =-val*par
                                        my_D4[r, s, p, q, u, w, v, t] =+val*par
                                        my_D4[r, s, p, q, u, w, t, v] =-val*par
                                        my_D4[r, s, p, q, v, u, t, w] =-val*par
                                        my_D4[r, s, p, q, v, u, w, t] = val*par
                                        my_D4[r, s, p, q, v, t, u, w] = val*par
                                        my_D4[r, s, p, q, v, t, w, u] =-val*par
                                        my_D4[r, s, p, q, v, w, t, u] = val*par
                                        my_D4[r, s, p, q, v, w, u, t] =-val*par
                                        my_D4[r, s, p, q, w, u, v, t] =-val*par
                                        my_D4[r, s, p, q, w, u, t, v] = val*par
                                        my_D4[r, s, p, q, w, v, u, t] = val*par
                                        my_D4[r, s, p, q, w, v, t, u] =-val*par
                                        my_D4[r, s, p, q, w, t, v, u] = val*par
                                        my_D4[r, s, p, q, w, t, u, v] =-val*par

                                        par = -1
                                        my_D4[r, s, q, p, t, u, v, w] = val*par
                                        my_D4[r, s, q, p, t, u, w, v] =-val*par
                                        my_D4[r, s, q, p, t, v, u, w] =-val*par
                                        my_D4[r, s, q, p, t, v, w, u] = val*par
                                        my_D4[r, s, q, p, t, w, v, u] =-val*par
                                        my_D4[r, s, q, p, t, w, u, v] = val*par
                                        my_D4[r, s, q, p, u, t, v, w] =-val*par
                                        my_D4[r, s, q, p, u, t, w, v] =+val*par
                                        my_D4[r, s, q, p, u, v, t, w] =+val*par
                                        my_D4[r, s, q, p, u, v, w, t] =-val*par
                                        my_D4[r, s, q, p, u, w, v, t] =+val*par
                                        my_D4[r, s, q, p, u, w, t, v] =-val*par
                                        my_D4[r, s, q, p, v, u, t, w] =-val*par
                                        my_D4[r, s, q, p, v, u, w, t] = val*par
                                        my_D4[r, s, q, p, v, t, u, w] = val*par
                                        my_D4[r, s, q, p, v, t, w, u] =-val*par
                                        my_D4[r, s, q, p, v, w, t, u] = val*par
                                        my_D4[r, s, q, p, v, w, u, t] =-val*par
                                        my_D4[r, s, q, p, w, u, v, t] =-val*par
                                        my_D4[r, s, q, p, w, u, t, v] = val*par
                                        my_D4[r, s, q, p, w, v, u, t] = val*par
                                        my_D4[r, s, q, p, w, v, t, u] =-val*par
                                        my_D4[r, s, q, p, w, t, v, u] = val*par
                                        my_D4[r, s, q, p, w, t, u, v] =-val*par
# s, [p,q,r]
                                        par = -1
                                        my_D4[s, q, r, p, t, u, v, w] = val*par
                                        my_D4[s, q, r, p, t, u, w, v] =-val*par
                                        my_D4[s, q, r, p, t, v, u, w] =-val*par
                                        my_D4[s, q, r, p, t, v, w, u] = val*par
                                        my_D4[s, q, r, p, t, w, v, u] =-val*par
                                        my_D4[s, q, r, p, t, w, u, v] = val*par
                                        my_D4[s, q, r, p, u, t, v, w] =-val*par
                                        my_D4[s, q, r, p, u, t, w, v] =+val*par
                                        my_D4[s, q, r, p, u, v, t, w] =+val*par
                                        my_D4[s, q, r, p, u, v, w, t] =-val*par
                                        my_D4[s, q, r, p, u, w, v, t] =+val*par
                                        my_D4[s, q, r, p, u, w, t, v] =-val*par
                                        my_D4[s, q, r, p, v, u, t, w] =-val*par
                                        my_D4[s, q, r, p, v, u, w, t] = val*par
                                        my_D4[s, q, r, p, v, t, u, w] = val*par
                                        my_D4[s, q, r, p, v, t, w, u] =-val*par
                                        my_D4[s, q, r, p, v, w, t, u] = val*par
                                        my_D4[s, q, r, p, v, w, u, t] =-val*par
                                        my_D4[s, q, r, p, w, u, v, t] =-val*par
                                        my_D4[s, q, r, p, w, u, t, v] = val*par
                                        my_D4[s, q, r, p, w, v, u, t] = val*par
                                        my_D4[s, q, r, p, w, v, t, u] =-val*par
                                        my_D4[s, q, r, p, w, t, v, u] = val*par
                                        my_D4[s, q, r, p, w, t, u, v] =-val*par

                                        par = 1
                                        my_D4[s, q, p, r, t, u, v, w] = val*par
                                        my_D4[s, q, p, r, t, u, w, v] =-val*par
                                        my_D4[s, q, p, r, t, v, u, w] =-val*par
                                        my_D4[s, q, p, r, t, v, w, u] = val*par
                                        my_D4[s, q, p, r, t, w, v, u] =-val*par
                                        my_D4[s, q, p, r, t, w, u, v] = val*par
                                        my_D4[s, q, p, r, u, t, v, w] =-val*par
                                        my_D4[s, q, p, r, u, t, w, v] =+val*par
                                        my_D4[s, q, p, r, u, v, t, w] =+val*par
                                        my_D4[s, q, p, r, u, v, w, t] =-val*par
                                        my_D4[s, q, p, r, u, w, v, t] =+val*par
                                        my_D4[s, q, p, r, u, w, t, v] =-val*par
                                        my_D4[s, q, p, r, v, u, t, w] =-val*par
                                        my_D4[s, q, p, r, v, u, w, t] = val*par
                                        my_D4[s, q, p, r, v, t, u, w] = val*par
                                        my_D4[s, q, p, r, v, t, w, u] =-val*par
                                        my_D4[s, q, p, r, v, w, t, u] = val*par
                                        my_D4[s, q, p, r, v, w, u, t] =-val*par
                                        my_D4[s, q, p, r, w, u, v, t] =-val*par
                                        my_D4[s, q, p, r, w, u, t, v] = val*par
                                        my_D4[s, q, p, r, w, v, u, t] = val*par
                                        my_D4[s, q, p, r, w, v, t, u] =-val*par
                                        my_D4[s, q, p, r, w, t, v, u] = val*par
                                        my_D4[s, q, p, r, w, t, u, v] =-val*par

                                        par = 1
                                        my_D4[s, r, q, p, t, u, v, w] = val*par
                                        my_D4[s, r, q, p, t, u, w, v] =-val*par
                                        my_D4[s, r, q, p, t, v, u, w] =-val*par
                                        my_D4[s, r, q, p, t, v, w, u] = val*par
                                        my_D4[s, r, q, p, t, w, v, u] =-val*par
                                        my_D4[s, r, q, p, t, w, u, v] = val*par
                                        my_D4[s, r, q, p, u, t, v, w] =-val*par
                                        my_D4[s, r, q, p, u, t, w, v] =+val*par
                                        my_D4[s, r, q, p, u, v, t, w] =+val*par
                                        my_D4[s, r, q, p, u, v, w, t] =-val*par
                                        my_D4[s, r, q, p, u, w, v, t] =+val*par
                                        my_D4[s, r, q, p, u, w, t, v] =-val*par
                                        my_D4[s, r, q, p, v, u, t, w] =-val*par
                                        my_D4[s, r, q, p, v, u, w, t] = val*par
                                        my_D4[s, r, q, p, v, t, u, w] = val*par
                                        my_D4[s, r, q, p, v, t, w, u] =-val*par
                                        my_D4[s, r, q, p, v, w, t, u] = val*par
                                        my_D4[s, r, q, p, v, w, u, t] =-val*par
                                        my_D4[s, r, q, p, w, u, v, t] =-val*par
                                        my_D4[s, r, q, p, w, u, t, v] = val*par
                                        my_D4[s, r, q, p, w, v, u, t] = val*par
                                        my_D4[s, r, q, p, w, v, t, u] =-val*par
                                        my_D4[s, r, q, p, w, t, v, u] = val*par
                                        my_D4[s, r, q, p, w, t, u, v] =-val*par

                                        par = -1
                                        my_D4[s, r, p, q, t, u, v, w] = val*par
                                        my_D4[s, r, p, q, t, u, w, v] =-val*par
                                        my_D4[s, r, p, q, t, v, u, w] =-val*par
                                        my_D4[s, r, p, q, t, v, w, u] = val*par
                                        my_D4[s, r, p, q, t, w, v, u] =-val*par
                                        my_D4[s, r, p, q, t, w, u, v] = val*par
                                        my_D4[s, r, p, q, u, t, v, w] =-val*par
                                        my_D4[s, r, p, q, u, t, w, v] =+val*par
                                        my_D4[s, r, p, q, u, v, t, w] =+val*par
                                        my_D4[s, r, p, q, u, v, w, t] =-val*par
                                        my_D4[s, r, p, q, u, w, v, t] =+val*par
                                        my_D4[s, r, p, q, u, w, t, v] =-val*par
                                        my_D4[s, r, p, q, v, u, t, w] =-val*par
                                        my_D4[s, r, p, q, v, u, w, t] = val*par
                                        my_D4[s, r, p, q, v, t, u, w] = val*par
                                        my_D4[s, r, p, q, v, t, w, u] =-val*par
                                        my_D4[s, r, p, q, v, w, t, u] = val*par
                                        my_D4[s, r, p, q, v, w, u, t] =-val*par
                                        my_D4[s, r, p, q, w, u, v, t] =-val*par
                                        my_D4[s, r, p, q, w, u, t, v] = val*par
                                        my_D4[s, r, p, q, w, v, u, t] = val*par
                                        my_D4[s, r, p, q, w, v, t, u] =-val*par
                                        my_D4[s, r, p, q, w, t, v, u] = val*par
                                        my_D4[s, r, p, q, w, t, u, v] =-val*par

                                        par = 1
                                        my_D4[s, p, r, q, t, u, v, w] = val*par
                                        my_D4[s, p, r, q, t, u, w, v] =-val*par
                                        my_D4[s, p, r, q, t, v, u, w] =-val*par
                                        my_D4[s, p, r, q, t, v, w, u] = val*par
                                        my_D4[s, p, r, q, t, w, v, u] =-val*par
                                        my_D4[s, p, r, q, t, w, u, v] = val*par
                                        my_D4[s, p, r, q, u, t, v, w] =-val*par
                                        my_D4[s, p, r, q, u, t, w, v] =+val*par
                                        my_D4[s, p, r, q, u, v, t, w] =+val*par
                                        my_D4[s, p, r, q, u, v, w, t] =-val*par
                                        my_D4[s, p, r, q, u, w, v, t] =+val*par
                                        my_D4[s, p, r, q, u, w, t, v] =-val*par
                                        my_D4[s, p, r, q, v, u, t, w] =-val*par
                                        my_D4[s, p, r, q, v, u, w, t] = val*par
                                        my_D4[s, p, r, q, v, t, u, w] = val*par
                                        my_D4[s, p, r, q, v, t, w, u] =-val*par
                                        my_D4[s, p, r, q, v, w, t, u] = val*par
                                        my_D4[s, p, r, q, v, w, u, t] =-val*par
                                        my_D4[s, p, r, q, w, u, v, t] =-val*par
                                        my_D4[s, p, r, q, w, u, t, v] = val*par
                                        my_D4[s, p, r, q, w, v, u, t] = val*par
                                        my_D4[s, p, r, q, w, v, t, u] =-val*par
                                        my_D4[s, p, r, q, w, t, v, u] = val*par
                                        my_D4[s, p, r, q, w, t, u, v] =-val*par

                                        par = -1
                                        my_D4[s, p, q, r, t, u, v, w] = val*par
                                        my_D4[s, p, q, r, t, u, w, v] =-val*par
                                        my_D4[s, p, q, r, t, v, u, w] =-val*par
                                        my_D4[s, p, q, r, t, v, w, u] = val*par
                                        my_D4[s, p, q, r, t, w, v, u] =-val*par
                                        my_D4[s, p, q, r, t, w, u, v] = val*par
                                        my_D4[s, p, q, r, u, t, v, w] =-val*par
                                        my_D4[s, p, q, r, u, t, w, v] =+val*par
                                        my_D4[s, p, q, r, u, v, t, w] =+val*par
                                        my_D4[s, p, q, r, u, v, w, t] =-val*par
                                        my_D4[s, p, q, r, u, w, v, t] =+val*par
                                        my_D4[s, p, q, r, u, w, t, v] =-val*par
                                        my_D4[s, p, q, r, v, u, t, w] =-val*par
                                        my_D4[s, p, q, r, v, u, w, t] = val*par
                                        my_D4[s, p, q, r, v, t, u, w] = val*par
                                        my_D4[s, p, q, r, v, t, w, u] =-val*par
                                        my_D4[s, p, q, r, v, w, t, u] = val*par
                                        my_D4[s, p, q, r, v, w, u, t] =-val*par
                                        my_D4[s, p, q, r, w, u, v, t] =-val*par
                                        my_D4[s, p, q, r, w, u, t, v] = val*par
                                        my_D4[s, p, q, r, w, v, u, t] = val*par
                                        my_D4[s, p, q, r, w, v, t, u] =-val*par
                                        my_D4[s, p, q, r, w, t, v, u] = val*par
                                        my_D4[s, p, q, r, w, t, u, v] =-val*par
### pqrs <-> tuvw
# p,q,r,s
                                        par = 1
                                        my_D4[t, u, v, w, p, q, r, s] = val*par
                                        my_D4[t, u, w, v, p, q, r, s] =-val*par
                                        my_D4[t, v, u, w, p, q, r, s] =-val*par
                                        my_D4[t, v, w, u, p, q, r, s] = val*par
                                        my_D4[t, w, v, u, p, q, r, s] =-val*par
                                        my_D4[t, w, u, v, p, q, r, s] = val*par
                                        my_D4[u, t, v, w, p, q, r, s] =-val*par
                                        my_D4[u, t, w, v, p, q, r, s] =+val*par
                                        my_D4[u, v, t, w, p, q, r, s] =+val*par
                                        my_D4[u, v, w, t, p, q, r, s] =-val*par
                                        my_D4[u, w, v, t, p, q, r, s] =+val*par
                                        my_D4[u, w, t, v, p, q, r, s] =-val*par
                                        my_D4[v, u, t, w, p, q, r, s] =-val*par
                                        my_D4[v, u, w, t, p, q, r, s] = val*par
                                        my_D4[v, t, u, w, p, q, r, s] = val*par
                                        my_D4[v, t, w, u, p, q, r, s] =-val*par
                                        my_D4[v, w, t, u, p, q, r, s] = val*par
                                        my_D4[v, w, u, t, p, q, r, s] =-val*par
                                        my_D4[w, u, v, t, p, q, r, s] =-val*par
                                        my_D4[w, u, t, v, p, q, r, s] = val*par
                                        my_D4[w, v, u, t, p, q, r, s] = val*par
                                        my_D4[w, v, t, u, p, q, r, s] =-val*par
                                        my_D4[w, t, v, u, p, q, r, s] = val*par
                                        my_D4[w, t, u, v, p, q, r, s] =-val*par

                                        par = -1
                                        my_D4[t, u, v, w, p, q, s, r] = val*par
                                        my_D4[t, u, w, v, p, q, s, r] =-val*par
                                        my_D4[t, v, u, w, p, q, s, r] =-val*par
                                        my_D4[t, v, w, u, p, q, s, r] = val*par
                                        my_D4[t, w, v, u, p, q, s, r] =-val*par
                                        my_D4[t, w, u, v, p, q, s, r] = val*par
                                        my_D4[u, t, v, w, p, q, s, r] =-val*par
                                        my_D4[u, t, w, v, p, q, s, r] =+val*par
                                        my_D4[u, v, t, w, p, q, s, r] =+val*par
                                        my_D4[u, v, w, t, p, q, s, r] =-val*par
                                        my_D4[u, w, v, t, p, q, s, r] =+val*par
                                        my_D4[u, w, t, v, p, q, s, r] =-val*par
                                        my_D4[v, u, t, w, p, q, s, r] =-val*par
                                        my_D4[v, u, w, t, p, q, s, r] = val*par
                                        my_D4[v, t, u, w, p, q, s, r] = val*par
                                        my_D4[v, t, w, u, p, q, s, r] =-val*par
                                        my_D4[v, w, t, u, p, q, s, r] = val*par
                                        my_D4[v, w, u, t, p, q, s, r] =-val*par
                                        my_D4[w, u, v, t, p, q, s, r] =-val*par
                                        my_D4[w, u, t, v, p, q, s, r] = val*par
                                        my_D4[w, v, u, t, p, q, s, r] = val*par
                                        my_D4[w, v, t, u, p, q, s, r] =-val*par
                                        my_D4[w, t, v, u, p, q, s, r] = val*par
                                        my_D4[w, t, u, v, p, q, s, r] =-val*par

                                        par = -1
                                        my_D4[t, u, v, w, p, r, q, s] = val*par
                                        my_D4[t, u, w, v, p, r, q, s] =-val*par
                                        my_D4[t, v, u, w, p, r, q, s] =-val*par
                                        my_D4[t, v, w, u, p, r, q, s] = val*par
                                        my_D4[t, w, v, u, p, r, q, s] =-val*par
                                        my_D4[t, w, u, v, p, r, q, s] = val*par
                                        my_D4[u, t, v, w, p, r, q, s] =-val*par
                                        my_D4[u, t, w, v, p, r, q, s] =+val*par
                                        my_D4[u, v, t, w, p, r, q, s] =+val*par
                                        my_D4[u, v, w, t, p, r, q, s] =-val*par
                                        my_D4[u, w, v, t, p, r, q, s] =+val*par
                                        my_D4[u, w, t, v, p, r, q, s] =-val*par
                                        my_D4[v, u, t, w, p, r, q, s] =-val*par
                                        my_D4[v, u, w, t, p, r, q, s] = val*par
                                        my_D4[v, t, u, w, p, r, q, s] = val*par
                                        my_D4[v, t, w, u, p, r, q, s] =-val*par
                                        my_D4[v, w, t, u, p, r, q, s] = val*par
                                        my_D4[v, w, u, t, p, r, q, s] =-val*par
                                        my_D4[w, u, v, t, p, r, q, s] =-val*par
                                        my_D4[w, u, t, v, p, r, q, s] = val*par
                                        my_D4[w, v, u, t, p, r, q, s] = val*par
                                        my_D4[w, v, t, u, p, r, q, s] =-val*par
                                        my_D4[w, t, v, u, p, r, q, s] = val*par
                                        my_D4[w, t, u, v, p, r, q, s] =-val*par

                                        par = 1
                                        my_D4[t, u, v, w, p, r, s, q] = val*par
                                        my_D4[t, u, w, v, p, r, s, q] =-val*par
                                        my_D4[t, v, u, w, p, r, s, q] =-val*par
                                        my_D4[t, v, w, u, p, r, s, q] = val*par
                                        my_D4[t, w, v, u, p, r, s, q] =-val*par
                                        my_D4[t, w, u, v, p, r, s, q] = val*par
                                        my_D4[u, t, v, w, p, r, s, q] =-val*par
                                        my_D4[u, t, w, v, p, r, s, q] =+val*par
                                        my_D4[u, v, t, w, p, r, s, q] =+val*par
                                        my_D4[u, v, w, t, p, r, s, q] =-val*par
                                        my_D4[u, w, v, t, p, r, s, q] =+val*par
                                        my_D4[u, w, t, v, p, r, s, q] =-val*par
                                        my_D4[v, u, t, w, p, r, s, q] =-val*par
                                        my_D4[v, u, w, t, p, r, s, q] = val*par
                                        my_D4[v, t, u, w, p, r, s, q] = val*par
                                        my_D4[v, t, w, u, p, r, s, q] =-val*par
                                        my_D4[v, w, t, u, p, r, s, q] = val*par
                                        my_D4[v, w, u, t, p, r, s, q] =-val*par
                                        my_D4[w, u, v, t, p, r, s, q] =-val*par
                                        my_D4[w, u, t, v, p, r, s, q] = val*par
                                        my_D4[w, v, u, t, p, r, s, q] = val*par
                                        my_D4[w, v, t, u, p, r, s, q] =-val*par
                                        my_D4[w, t, v, u, p, r, s, q] = val*par
                                        my_D4[w, t, u, v, p, r, s, q] =-val*par

                                        par = -1
                                        my_D4[t, u, v, w, p, s, r, q] = val*par
                                        my_D4[t, u, w, v, p, s, r, q] =-val*par
                                        my_D4[t, v, u, w, p, s, r, q] =-val*par
                                        my_D4[t, v, w, u, p, s, r, q] = val*par
                                        my_D4[t, w, v, u, p, s, r, q] =-val*par
                                        my_D4[t, w, u, v, p, s, r, q] = val*par
                                        my_D4[u, t, v, w, p, s, r, q] =-val*par
                                        my_D4[u, t, w, v, p, s, r, q] =+val*par
                                        my_D4[u, v, t, w, p, s, r, q] =+val*par
                                        my_D4[u, v, w, t, p, s, r, q] =-val*par
                                        my_D4[u, w, v, t, p, s, r, q] =+val*par
                                        my_D4[u, w, t, v, p, s, r, q] =-val*par
                                        my_D4[v, u, t, w, p, s, r, q] =-val*par
                                        my_D4[v, u, w, t, p, s, r, q] = val*par
                                        my_D4[v, t, u, w, p, s, r, q] = val*par
                                        my_D4[v, t, w, u, p, s, r, q] =-val*par
                                        my_D4[v, w, t, u, p, s, r, q] = val*par
                                        my_D4[v, w, u, t, p, s, r, q] =-val*par
                                        my_D4[w, u, v, t, p, s, r, q] =-val*par
                                        my_D4[w, u, t, v, p, s, r, q] = val*par
                                        my_D4[w, v, u, t, p, s, r, q] = val*par
                                        my_D4[w, v, t, u, p, s, r, q] =-val*par
                                        my_D4[w, t, v, u, p, s, r, q] = val*par
                                        my_D4[w, t, u, v, p, s, r, q] =-val*par

                                        par = 1
                                        my_D4[t, u, v, w, p, s, q, r] = val*par
                                        my_D4[t, u, w, v, p, s, q, r] =-val*par
                                        my_D4[t, v, u, w, p, s, q, r] =-val*par
                                        my_D4[t, v, w, u, p, s, q, r] = val*par
                                        my_D4[t, w, v, u, p, s, q, r] =-val*par
                                        my_D4[t, w, u, v, p, s, q, r] = val*par
                                        my_D4[u, t, v, w, p, s, q, r] =-val*par
                                        my_D4[u, t, w, v, p, s, q, r] =+val*par
                                        my_D4[u, v, t, w, p, s, q, r] =+val*par
                                        my_D4[u, v, w, t, p, s, q, r] =-val*par
                                        my_D4[u, w, v, t, p, s, q, r] =+val*par
                                        my_D4[u, w, t, v, p, s, q, r] =-val*par
                                        my_D4[v, u, t, w, p, s, q, r] =-val*par
                                        my_D4[v, u, w, t, p, s, q, r] = val*par
                                        my_D4[v, t, u, w, p, s, q, r] = val*par
                                        my_D4[v, t, w, u, p, s, q, r] =-val*par
                                        my_D4[v, w, t, u, p, s, q, r] = val*par
                                        my_D4[v, w, u, t, p, s, q, r] =-val*par
                                        my_D4[w, u, v, t, p, s, q, r] =-val*par
                                        my_D4[w, u, t, v, p, s, q, r] = val*par
                                        my_D4[w, v, u, t, p, s, q, r] = val*par
                                        my_D4[w, v, t, u, p, s, q, r] =-val*par
                                        my_D4[w, t, v, u, p, s, q, r] = val*par
                                        my_D4[w, t, u, v, p, s, q, r] =-val*par

### q, [p, r, s] ...
                                        par = -1
                                        my_D4[t, u, v, w, q, p, r, s] = val*par
                                        my_D4[t, u, w, v, q, p, r, s] =-val*par
                                        my_D4[t, v, u, w, q, p, r, s] =-val*par
                                        my_D4[t, v, w, u, q, p, r, s] = val*par
                                        my_D4[t, w, v, u, q, p, r, s] =-val*par
                                        my_D4[t, w, u, v, q, p, r, s] = val*par
                                        my_D4[u, t, v, w, q, p, r, s] =-val*par
                                        my_D4[u, t, w, v, q, p, r, s] =+val*par
                                        my_D4[u, v, t, w, q, p, r, s] =+val*par
                                        my_D4[u, v, w, t, q, p, r, s] =-val*par
                                        my_D4[u, w, v, t, q, p, r, s] =+val*par
                                        my_D4[u, w, t, v, q, p, r, s] =-val*par
                                        my_D4[v, u, t, w, q, p, r, s] =-val*par
                                        my_D4[v, u, w, t, q, p, r, s] = val*par
                                        my_D4[v, t, u, w, q, p, r, s] = val*par
                                        my_D4[v, t, w, u, q, p, r, s] =-val*par
                                        my_D4[v, w, t, u, q, p, r, s] = val*par
                                        my_D4[v, w, u, t, q, p, r, s] =-val*par
                                        my_D4[w, u, v, t, q, p, r, s] =-val*par
                                        my_D4[w, u, t, v, q, p, r, s] = val*par
                                        my_D4[w, v, u, t, q, p, r, s] = val*par
                                        my_D4[w, v, t, u, q, p, r, s] =-val*par
                                        my_D4[w, t, v, u, q, p, r, s] = val*par
                                        my_D4[w, t, u, v, q, p, r, s] =-val*par

                                        par = 1
                                        my_D4[t, u, v, w, q, p, s, r] = val*par
                                        my_D4[t, u, w, v, q, p, s, r] =-val*par
                                        my_D4[t, v, u, w, q, p, s, r] =-val*par
                                        my_D4[t, v, w, u, q, p, s, r] = val*par
                                        my_D4[t, w, v, u, q, p, s, r] =-val*par
                                        my_D4[t, w, u, v, q, p, s, r] = val*par
                                        my_D4[u, t, v, w, q, p, s, r] =-val*par
                                        my_D4[u, t, w, v, q, p, s, r] =+val*par
                                        my_D4[u, v, t, w, q, p, s, r] =+val*par
                                        my_D4[u, v, w, t, q, p, s, r] =-val*par
                                        my_D4[u, w, v, t, q, p, s, r] =+val*par
                                        my_D4[u, w, t, v, q, p, s, r] =-val*par
                                        my_D4[v, u, t, w, q, p, s, r] =-val*par
                                        my_D4[v, u, w, t, q, p, s, r] = val*par
                                        my_D4[v, t, u, w, q, p, s, r] = val*par
                                        my_D4[v, t, w, u, q, p, s, r] =-val*par
                                        my_D4[v, w, t, u, q, p, s, r] = val*par
                                        my_D4[v, w, u, t, q, p, s, r] =-val*par
                                        my_D4[w, u, v, t, q, p, s, r] =-val*par
                                        my_D4[w, u, t, v, q, p, s, r] = val*par
                                        my_D4[w, v, u, t, q, p, s, r] = val*par
                                        my_D4[w, v, t, u, q, p, s, r] =-val*par
                                        my_D4[w, t, v, u, q, p, s, r] = val*par
                                        my_D4[w, t, u, v, q, p, s, r] =-val*par

                                        par = 1
                                        my_D4[t, u, v, w, q, r, p, s] = val*par
                                        my_D4[t, u, w, v, q, r, p, s] =-val*par
                                        my_D4[t, v, u, w, q, r, p, s] =-val*par
                                        my_D4[t, v, w, u, q, r, p, s] = val*par
                                        my_D4[t, w, v, u, q, r, p, s] =-val*par
                                        my_D4[t, w, u, v, q, r, p, s] = val*par
                                        my_D4[u, t, v, w, q, r, p, s] =-val*par
                                        my_D4[u, t, w, v, q, r, p, s] =+val*par
                                        my_D4[u, v, t, w, q, r, p, s] =+val*par
                                        my_D4[u, v, w, t, q, r, p, s] =-val*par
                                        my_D4[u, w, v, t, q, r, p, s] =+val*par
                                        my_D4[u, w, t, v, q, r, p, s] =-val*par
                                        my_D4[v, u, t, w, q, r, p, s] =-val*par
                                        my_D4[v, u, w, t, q, r, p, s] = val*par
                                        my_D4[v, t, u, w, q, r, p, s] = val*par
                                        my_D4[v, t, w, u, q, r, p, s] =-val*par
                                        my_D4[v, w, t, u, q, r, p, s] = val*par
                                        my_D4[v, w, u, t, q, r, p, s] =-val*par
                                        my_D4[w, u, v, t, q, r, p, s] =-val*par
                                        my_D4[w, u, t, v, q, r, p, s] = val*par
                                        my_D4[w, v, u, t, q, r, p, s] = val*par
                                        my_D4[w, v, t, u, q, r, p, s] =-val*par
                                        my_D4[w, t, v, u, q, r, p, s] = val*par
                                        my_D4[w, t, u, v, q, r, p, s] =-val*par

                                        par = -1
                                        my_D4[t, u, v, w, q, r, s, p] = val*par
                                        my_D4[t, u, w, v, q, r, s, p] =-val*par
                                        my_D4[t, v, u, w, q, r, s, p] =-val*par
                                        my_D4[t, v, w, u, q, r, s, p] = val*par
                                        my_D4[t, w, v, u, q, r, s, p] =-val*par
                                        my_D4[t, w, u, v, q, r, s, p] = val*par
                                        my_D4[u, t, v, w, q, r, s, p] =-val*par
                                        my_D4[u, t, w, v, q, r, s, p] =+val*par
                                        my_D4[u, v, t, w, q, r, s, p] =+val*par
                                        my_D4[u, v, w, t, q, r, s, p] =-val*par
                                        my_D4[u, w, v, t, q, r, s, p] =+val*par
                                        my_D4[u, w, t, v, q, r, s, p] =-val*par
                                        my_D4[v, u, t, w, q, r, s, p] =-val*par
                                        my_D4[v, u, w, t, q, r, s, p] = val*par
                                        my_D4[v, t, u, w, q, r, s, p] = val*par
                                        my_D4[v, t, w, u, q, r, s, p] =-val*par
                                        my_D4[v, w, t, u, q, r, s, p] = val*par
                                        my_D4[v, w, u, t, q, r, s, p] =-val*par
                                        my_D4[w, u, v, t, q, r, s, p] =-val*par
                                        my_D4[w, u, t, v, q, r, s, p] = val*par
                                        my_D4[w, v, u, t, q, r, s, p] = val*par
                                        my_D4[w, v, t, u, q, r, s, p] =-val*par
                                        my_D4[w, t, v, u, q, r, s, p] = val*par
                                        my_D4[w, t, u, v, q, r, s, p] =-val*par

                                        par = 1
                                        my_D4[t, u, v, w, q, s, r, p] = val*par
                                        my_D4[t, u, w, v, q, s, r, p] =-val*par
                                        my_D4[t, v, u, w, q, s, r, p] =-val*par
                                        my_D4[t, v, w, u, q, s, r, p] = val*par
                                        my_D4[t, w, v, u, q, s, r, p] =-val*par
                                        my_D4[t, w, u, v, q, s, r, p] = val*par
                                        my_D4[u, t, v, w, q, s, r, p] =-val*par
                                        my_D4[u, t, w, v, q, s, r, p] =+val*par
                                        my_D4[u, v, t, w, q, s, r, p] =+val*par
                                        my_D4[u, v, w, t, q, s, r, p] =-val*par
                                        my_D4[u, w, v, t, q, s, r, p] =+val*par
                                        my_D4[u, w, t, v, q, s, r, p] =-val*par
                                        my_D4[v, u, t, w, q, s, r, p] =-val*par
                                        my_D4[v, u, w, t, q, s, r, p] = val*par
                                        my_D4[v, t, u, w, q, s, r, p] = val*par
                                        my_D4[v, t, w, u, q, s, r, p] =-val*par
                                        my_D4[v, w, t, u, q, s, r, p] = val*par
                                        my_D4[v, w, u, t, q, s, r, p] =-val*par
                                        my_D4[w, u, v, t, q, s, r, p] =-val*par
                                        my_D4[w, u, t, v, q, s, r, p] = val*par
                                        my_D4[w, v, u, t, q, s, r, p] = val*par
                                        my_D4[w, v, t, u, q, s, r, p] =-val*par
                                        my_D4[w, t, v, u, q, s, r, p] = val*par
                                        my_D4[w, t, u, v, q, s, r, p] =-val*par

                                        par = -1
                                        my_D4[t, u, v, w, q, s, p, r] = val*par
                                        my_D4[t, u, w, v, q, s, p, r] =-val*par
                                        my_D4[t, v, u, w, q, s, p, r] =-val*par
                                        my_D4[t, v, w, u, q, s, p, r] = val*par
                                        my_D4[t, w, v, u, q, s, p, r] =-val*par
                                        my_D4[t, w, u, v, q, s, p, r] = val*par
                                        my_D4[u, t, v, w, q, s, p, r] =-val*par
                                        my_D4[u, t, w, v, q, s, p, r] =+val*par
                                        my_D4[u, v, t, w, q, s, p, r] =+val*par
                                        my_D4[u, v, w, t, q, s, p, r] =-val*par
                                        my_D4[u, w, v, t, q, s, p, r] =+val*par
                                        my_D4[u, w, t, v, q, s, p, r] =-val*par
                                        my_D4[v, u, t, w, q, s, p, r] =-val*par
                                        my_D4[v, u, w, t, q, s, p, r] = val*par
                                        my_D4[v, t, u, w, q, s, p, r] = val*par
                                        my_D4[v, t, w, u, q, s, p, r] =-val*par
                                        my_D4[v, w, t, u, q, s, p, r] = val*par
                                        my_D4[v, w, u, t, q, s, p, r] =-val*par
                                        my_D4[w, u, v, t, q, s, p, r] =-val*par
                                        my_D4[w, u, t, v, q, s, p, r] = val*par
                                        my_D4[w, v, u, t, q, s, p, r] = val*par
                                        my_D4[w, v, t, u, q, s, p, r] =-val*par
                                        my_D4[w, t, v, u, q, s, p, r] = val*par
                                        my_D4[w, t, u, v, q, s, p, r] =-val*par
# r, [p,q,s]
                                        par = -1
                                        my_D4[t, u, v, w, r, q, p, s] = val*par
                                        my_D4[t, u, w, v, r, q, p, s] =-val*par
                                        my_D4[t, v, u, w, r, q, p, s] =-val*par
                                        my_D4[t, v, w, u, r, q, p, s] = val*par
                                        my_D4[t, w, v, u, r, q, p, s] =-val*par
                                        my_D4[t, w, u, v, r, q, p, s] = val*par
                                        my_D4[u, t, v, w, r, q, p, s] =-val*par
                                        my_D4[u, t, w, v, r, q, p, s] =+val*par
                                        my_D4[u, v, t, w, r, q, p, s] =+val*par
                                        my_D4[u, v, w, t, r, q, p, s] =-val*par
                                        my_D4[u, w, v, t, r, q, p, s] =+val*par
                                        my_D4[u, w, t, v, r, q, p, s] =-val*par
                                        my_D4[v, u, t, w, r, q, p, s] =-val*par
                                        my_D4[v, u, w, t, r, q, p, s] = val*par
                                        my_D4[v, t, u, w, r, q, p, s] = val*par
                                        my_D4[v, t, w, u, r, q, p, s] =-val*par
                                        my_D4[v, w, t, u, r, q, p, s] = val*par
                                        my_D4[v, w, u, t, r, q, p, s] =-val*par
                                        my_D4[w, u, v, t, r, q, p, s] =-val*par
                                        my_D4[w, u, t, v, r, q, p, s] = val*par
                                        my_D4[w, v, u, t, r, q, p, s] = val*par
                                        my_D4[w, v, t, u, r, q, p, s] =-val*par
                                        my_D4[w, t, v, u, r, q, p, s] = val*par
                                        my_D4[w, t, u, v, r, q, p, s] =-val*par

                                        par = 1
                                        my_D4[t, u, v, w, r, q, s, p] = val*par
                                        my_D4[t, u, w, v, r, q, s, p] =-val*par
                                        my_D4[t, v, u, w, r, q, s, p] =-val*par
                                        my_D4[t, v, w, u, r, q, s, p] = val*par
                                        my_D4[t, w, v, u, r, q, s, p] =-val*par
                                        my_D4[t, w, u, v, r, q, s, p] = val*par
                                        my_D4[u, t, v, w, r, q, s, p] =-val*par
                                        my_D4[u, t, w, v, r, q, s, p] =+val*par
                                        my_D4[u, v, t, w, r, q, s, p] =+val*par
                                        my_D4[u, v, w, t, r, q, s, p] =-val*par
                                        my_D4[u, w, v, t, r, q, s, p] =+val*par
                                        my_D4[u, w, t, v, r, q, s, p] =-val*par
                                        my_D4[v, u, t, w, r, q, s, p] =-val*par
                                        my_D4[v, u, w, t, r, q, s, p] = val*par
                                        my_D4[v, t, u, w, r, q, s, p] = val*par
                                        my_D4[v, t, w, u, r, q, s, p] =-val*par
                                        my_D4[v, w, t, u, r, q, s, p] = val*par
                                        my_D4[v, w, u, t, r, q, s, p] =-val*par
                                        my_D4[w, u, v, t, r, q, s, p] =-val*par
                                        my_D4[w, u, t, v, r, q, s, p] = val*par
                                        my_D4[w, v, u, t, r, q, s, p] = val*par
                                        my_D4[w, v, t, u, r, q, s, p] =-val*par
                                        my_D4[w, t, v, u, r, q, s, p] = val*par
                                        my_D4[w, t, u, v, r, q, s, p] =-val*par

                                        par = 1
                                        my_D4[t, u, v, w, r, p, q, s] = val*par
                                        my_D4[t, u, w, v, r, p, q, s] =-val*par
                                        my_D4[t, v, u, w, r, p, q, s] =-val*par
                                        my_D4[t, v, w, u, r, p, q, s] = val*par
                                        my_D4[t, w, v, u, r, p, q, s] =-val*par
                                        my_D4[t, w, u, v, r, p, q, s] = val*par
                                        my_D4[u, t, v, w, r, p, q, s] =-val*par
                                        my_D4[u, t, w, v, r, p, q, s] =+val*par
                                        my_D4[u, v, t, w, r, p, q, s] =+val*par
                                        my_D4[u, v, w, t, r, p, q, s] =-val*par
                                        my_D4[u, w, v, t, r, p, q, s] =+val*par
                                        my_D4[u, w, t, v, r, p, q, s] =-val*par
                                        my_D4[v, u, t, w, r, p, q, s] =-val*par
                                        my_D4[v, u, w, t, r, p, q, s] = val*par
                                        my_D4[v, t, u, w, r, p, q, s] = val*par
                                        my_D4[v, t, w, u, r, p, q, s] =-val*par
                                        my_D4[v, w, t, u, r, p, q, s] = val*par
                                        my_D4[v, w, u, t, r, p, q, s] =-val*par
                                        my_D4[w, u, v, t, r, p, q, s] =-val*par
                                        my_D4[w, u, t, v, r, p, q, s] = val*par
                                        my_D4[w, v, u, t, r, p, q, s] = val*par
                                        my_D4[w, v, t, u, r, p, q, s] =-val*par
                                        my_D4[w, t, v, u, r, p, q, s] = val*par
                                        my_D4[w, t, u, v, r, p, q, s] =-val*par

                                        par = -1
                                        my_D4[t, u, v, w, r, p, s, q] = val*par
                                        my_D4[t, u, w, v, r, p, s, q] =-val*par
                                        my_D4[t, v, u, w, r, p, s, q] =-val*par
                                        my_D4[t, v, w, u, r, p, s, q] = val*par
                                        my_D4[t, w, v, u, r, p, s, q] =-val*par
                                        my_D4[t, w, u, v, r, p, s, q] = val*par
                                        my_D4[u, t, v, w, r, p, s, q] =-val*par
                                        my_D4[u, t, w, v, r, p, s, q] =+val*par
                                        my_D4[u, v, t, w, r, p, s, q] =+val*par
                                        my_D4[u, v, w, t, r, p, s, q] =-val*par
                                        my_D4[u, w, v, t, r, p, s, q] =+val*par
                                        my_D4[u, w, t, v, r, p, s, q] =-val*par
                                        my_D4[v, u, t, w, r, p, s, q] =-val*par
                                        my_D4[v, u, w, t, r, p, s, q] = val*par
                                        my_D4[v, t, u, w, r, p, s, q] = val*par
                                        my_D4[v, t, w, u, r, p, s, q] =-val*par
                                        my_D4[v, w, t, u, r, p, s, q] = val*par
                                        my_D4[v, w, u, t, r, p, s, q] =-val*par
                                        my_D4[w, u, v, t, r, p, s, q] =-val*par
                                        my_D4[w, u, t, v, r, p, s, q] = val*par
                                        my_D4[w, v, u, t, r, p, s, q] = val*par
                                        my_D4[w, v, t, u, r, p, s, q] =-val*par
                                        my_D4[w, t, v, u, r, p, s, q] = val*par
                                        my_D4[w, t, u, v, r, p, s, q] =-val*par

                                        par = 1
                                        my_D4[t, u, v, w, r, s, p, q] = val*par
                                        my_D4[t, u, w, v, r, s, p, q] =-val*par
                                        my_D4[t, v, u, w, r, s, p, q] =-val*par
                                        my_D4[t, v, w, u, r, s, p, q] = val*par
                                        my_D4[t, w, v, u, r, s, p, q] =-val*par
                                        my_D4[t, w, u, v, r, s, p, q] = val*par
                                        my_D4[u, t, v, w, r, s, p, q] =-val*par
                                        my_D4[u, t, w, v, r, s, p, q] =+val*par
                                        my_D4[u, v, t, w, r, s, p, q] =+val*par
                                        my_D4[u, v, w, t, r, s, p, q] =-val*par
                                        my_D4[u, w, v, t, r, s, p, q] =+val*par
                                        my_D4[u, w, t, v, r, s, p, q] =-val*par
                                        my_D4[v, u, t, w, r, s, p, q] =-val*par
                                        my_D4[v, u, w, t, r, s, p, q] = val*par
                                        my_D4[v, t, u, w, r, s, p, q] = val*par
                                        my_D4[v, t, w, u, r, s, p, q] =-val*par
                                        my_D4[v, w, t, u, r, s, p, q] = val*par
                                        my_D4[v, w, u, t, r, s, p, q] =-val*par
                                        my_D4[w, u, v, t, r, s, p, q] =-val*par
                                        my_D4[w, u, t, v, r, s, p, q] = val*par
                                        my_D4[w, v, u, t, r, s, p, q] = val*par
                                        my_D4[w, v, t, u, r, s, p, q] =-val*par
                                        my_D4[w, t, v, u, r, s, p, q] = val*par
                                        my_D4[w, t, u, v, r, s, p, q] =-val*par

                                        par = -1
                                        my_D4[t, u, v, w, r, s, q, p] = val*par
                                        my_D4[t, u, w, v, r, s, q, p] =-val*par
                                        my_D4[t, v, u, w, r, s, q, p] =-val*par
                                        my_D4[t, v, w, u, r, s, q, p] = val*par
                                        my_D4[t, w, v, u, r, s, q, p] =-val*par
                                        my_D4[t, w, u, v, r, s, q, p] = val*par
                                        my_D4[u, t, v, w, r, s, q, p] =-val*par
                                        my_D4[u, t, w, v, r, s, q, p] =+val*par
                                        my_D4[u, v, t, w, r, s, q, p] =+val*par
                                        my_D4[u, v, w, t, r, s, q, p] =-val*par
                                        my_D4[u, w, v, t, r, s, q, p] =+val*par
                                        my_D4[u, w, t, v, r, s, q, p] =-val*par
                                        my_D4[v, u, t, w, r, s, q, p] =-val*par
                                        my_D4[v, u, w, t, r, s, q, p] = val*par
                                        my_D4[v, t, u, w, r, s, q, p] = val*par
                                        my_D4[v, t, w, u, r, s, q, p] =-val*par
                                        my_D4[v, w, t, u, r, s, q, p] = val*par
                                        my_D4[v, w, u, t, r, s, q, p] =-val*par
                                        my_D4[w, u, v, t, r, s, q, p] =-val*par
                                        my_D4[w, u, t, v, r, s, q, p] = val*par
                                        my_D4[w, v, u, t, r, s, q, p] = val*par
                                        my_D4[w, v, t, u, r, s, q, p] =-val*par
                                        my_D4[w, t, v, u, r, s, q, p] = val*par
                                        my_D4[w, t, u, v, r, s, q, p] =-val*par
# s, [p,q,r]
                                        par = -1
                                        my_D4[t, u, v, w, s, q, r, p] = val*par
                                        my_D4[t, u, w, v, s, q, r, p] =-val*par
                                        my_D4[t, v, u, w, s, q, r, p] =-val*par
                                        my_D4[t, v, w, u, s, q, r, p] = val*par
                                        my_D4[t, w, v, u, s, q, r, p] =-val*par
                                        my_D4[t, w, u, v, s, q, r, p] = val*par
                                        my_D4[u, t, v, w, s, q, r, p] =-val*par
                                        my_D4[u, t, w, v, s, q, r, p] =+val*par
                                        my_D4[u, v, t, w, s, q, r, p] =+val*par
                                        my_D4[u, v, w, t, s, q, r, p] =-val*par
                                        my_D4[u, w, v, t, s, q, r, p] =+val*par
                                        my_D4[u, w, t, v, s, q, r, p] =-val*par
                                        my_D4[v, u, t, w, s, q, r, p] =-val*par
                                        my_D4[v, u, w, t, s, q, r, p] = val*par
                                        my_D4[v, t, u, w, s, q, r, p] = val*par
                                        my_D4[v, t, w, u, s, q, r, p] =-val*par
                                        my_D4[v, w, t, u, s, q, r, p] = val*par
                                        my_D4[v, w, u, t, s, q, r, p] =-val*par
                                        my_D4[w, u, v, t, s, q, r, p] =-val*par
                                        my_D4[w, u, t, v, s, q, r, p] = val*par
                                        my_D4[w, v, u, t, s, q, r, p] = val*par
                                        my_D4[w, v, t, u, s, q, r, p] =-val*par
                                        my_D4[w, t, v, u, s, q, r, p] = val*par
                                        my_D4[w, t, u, v, s, q, r, p] =-val*par

                                        par = 1
                                        my_D4[t, u, v, w, s, q, p, r] = val*par
                                        my_D4[t, u, w, v, s, q, p, r] =-val*par
                                        my_D4[t, v, u, w, s, q, p, r] =-val*par
                                        my_D4[t, v, w, u, s, q, p, r] = val*par
                                        my_D4[t, w, v, u, s, q, p, r] =-val*par
                                        my_D4[t, w, u, v, s, q, p, r] = val*par
                                        my_D4[u, t, v, w, s, q, p, r] =-val*par
                                        my_D4[u, t, w, v, s, q, p, r] =+val*par
                                        my_D4[u, v, t, w, s, q, p, r] =+val*par
                                        my_D4[u, v, w, t, s, q, p, r] =-val*par
                                        my_D4[u, w, v, t, s, q, p, r] =+val*par
                                        my_D4[u, w, t, v, s, q, p, r] =-val*par
                                        my_D4[v, u, t, w, s, q, p, r] =-val*par
                                        my_D4[v, u, w, t, s, q, p, r] = val*par
                                        my_D4[v, t, u, w, s, q, p, r] = val*par
                                        my_D4[v, t, w, u, s, q, p, r] =-val*par
                                        my_D4[v, w, t, u, s, q, p, r] = val*par
                                        my_D4[v, w, u, t, s, q, p, r] =-val*par
                                        my_D4[w, u, v, t, s, q, p, r] =-val*par
                                        my_D4[w, u, t, v, s, q, p, r] = val*par
                                        my_D4[w, v, u, t, s, q, p, r] = val*par
                                        my_D4[w, v, t, u, s, q, p, r] =-val*par
                                        my_D4[w, t, v, u, s, q, p, r] = val*par
                                        my_D4[w, t, u, v, s, q, p, r] =-val*par

                                        par = 1
                                        my_D4[t, u, v, w, s, r, q, p] = val*par
                                        my_D4[t, u, w, v, s, r, q, p] =-val*par
                                        my_D4[t, v, u, w, s, r, q, p] =-val*par
                                        my_D4[t, v, w, u, s, r, q, p] = val*par
                                        my_D4[t, w, v, u, s, r, q, p] =-val*par
                                        my_D4[t, w, u, v, s, r, q, p] = val*par
                                        my_D4[u, t, v, w, s, r, q, p] =-val*par
                                        my_D4[u, t, w, v, s, r, q, p] =+val*par
                                        my_D4[u, v, t, w, s, r, q, p] =+val*par
                                        my_D4[u, v, w, t, s, r, q, p] =-val*par
                                        my_D4[u, w, v, t, s, r, q, p] =+val*par
                                        my_D4[u, w, t, v, s, r, q, p] =-val*par
                                        my_D4[v, u, t, w, s, r, q, p] =-val*par
                                        my_D4[v, u, w, t, s, r, q, p] = val*par
                                        my_D4[v, t, u, w, s, r, q, p] = val*par
                                        my_D4[v, t, w, u, s, r, q, p] =-val*par
                                        my_D4[v, w, t, u, s, r, q, p] = val*par
                                        my_D4[v, w, u, t, s, r, q, p] =-val*par
                                        my_D4[w, u, v, t, s, r, q, p] =-val*par
                                        my_D4[w, u, t, v, s, r, q, p] = val*par
                                        my_D4[w, v, u, t, s, r, q, p] = val*par
                                        my_D4[w, v, t, u, s, r, q, p] =-val*par
                                        my_D4[w, t, v, u, s, r, q, p] = val*par
                                        my_D4[w, t, u, v, s, r, q, p] =-val*par

                                        par = -1
                                        my_D4[t, u, v, w, s, r, p, q] = val*par
                                        my_D4[t, u, w, v, s, r, p, q] =-val*par
                                        my_D4[t, v, u, w, s, r, p, q] =-val*par
                                        my_D4[t, v, w, u, s, r, p, q] = val*par
                                        my_D4[t, w, v, u, s, r, p, q] =-val*par
                                        my_D4[t, w, u, v, s, r, p, q] = val*par
                                        my_D4[u, t, v, w, s, r, p, q] =-val*par
                                        my_D4[u, t, w, v, s, r, p, q] =+val*par
                                        my_D4[u, v, t, w, s, r, p, q] =+val*par
                                        my_D4[u, v, w, t, s, r, p, q] =-val*par
                                        my_D4[u, w, v, t, s, r, p, q] =+val*par
                                        my_D4[u, w, t, v, s, r, p, q] =-val*par
                                        my_D4[v, u, t, w, s, r, p, q] =-val*par
                                        my_D4[v, u, w, t, s, r, p, q] = val*par
                                        my_D4[v, t, u, w, s, r, p, q] = val*par
                                        my_D4[v, t, w, u, s, r, p, q] =-val*par
                                        my_D4[v, w, t, u, s, r, p, q] = val*par
                                        my_D4[v, w, u, t, s, r, p, q] =-val*par
                                        my_D4[w, u, v, t, s, r, p, q] =-val*par
                                        my_D4[w, u, t, v, s, r, p, q] = val*par
                                        my_D4[w, v, u, t, s, r, p, q] = val*par
                                        my_D4[w, v, t, u, s, r, p, q] =-val*par
                                        my_D4[w, t, v, u, s, r, p, q] = val*par
                                        my_D4[w, t, u, v, s, r, p, q] =-val*par

                                        par = 1
                                        my_D4[t, u, v, w, s, p, r, q] = val*par
                                        my_D4[t, u, w, v, s, p, r, q] =-val*par
                                        my_D4[t, v, u, w, s, p, r, q] =-val*par
                                        my_D4[t, v, w, u, s, p, r, q] = val*par
                                        my_D4[t, w, v, u, s, p, r, q] =-val*par
                                        my_D4[t, w, u, v, s, p, r, q] = val*par
                                        my_D4[u, t, v, w, s, p, r, q] =-val*par
                                        my_D4[u, t, w, v, s, p, r, q] =+val*par
                                        my_D4[u, v, t, w, s, p, r, q] =+val*par
                                        my_D4[u, v, w, t, s, p, r, q] =-val*par
                                        my_D4[u, w, v, t, s, p, r, q] =+val*par
                                        my_D4[u, w, t, v, s, p, r, q] =-val*par
                                        my_D4[v, u, t, w, s, p, r, q] =-val*par
                                        my_D4[v, u, w, t, s, p, r, q] = val*par
                                        my_D4[v, t, u, w, s, p, r, q] = val*par
                                        my_D4[v, t, w, u, s, p, r, q] =-val*par
                                        my_D4[v, w, t, u, s, p, r, q] = val*par
                                        my_D4[v, w, u, t, s, p, r, q] =-val*par
                                        my_D4[w, u, v, t, s, p, r, q] =-val*par
                                        my_D4[w, u, t, v, s, p, r, q] = val*par
                                        my_D4[w, v, u, t, s, p, r, q] = val*par
                                        my_D4[w, v, t, u, s, p, r, q] =-val*par
                                        my_D4[w, t, v, u, s, p, r, q] = val*par
                                        my_D4[w, t, u, v, s, p, r, q] =-val*par

                                        par = -1
                                        my_D4[t, u, v, w, s, p, q, r] = val*par
                                        my_D4[t, u, w, v, s, p, q, r] =-val*par
                                        my_D4[t, v, u, w, s, p, q, r] =-val*par
                                        my_D4[t, v, w, u, s, p, q, r] = val*par
                                        my_D4[t, w, v, u, s, p, q, r] =-val*par
                                        my_D4[t, w, u, v, s, p, q, r] = val*par
                                        my_D4[u, t, v, w, s, p, q, r] =-val*par
                                        my_D4[u, t, w, v, s, p, q, r] =+val*par
                                        my_D4[u, v, t, w, s, p, q, r] =+val*par
                                        my_D4[u, v, w, t, s, p, q, r] =-val*par
                                        my_D4[u, w, v, t, s, p, q, r] =+val*par
                                        my_D4[u, w, t, v, s, p, q, r] =-val*par
                                        my_D4[v, u, t, w, s, p, q, r] =-val*par
                                        my_D4[v, u, w, t, s, p, q, r] = val*par
                                        my_D4[v, t, u, w, s, p, q, r] = val*par
                                        my_D4[v, t, w, u, s, p, q, r] =-val*par
                                        my_D4[v, w, t, u, s, p, q, r] = val*par
                                        my_D4[v, w, u, t, s, p, q, r] =-val*par
                                        my_D4[w, u, v, t, s, p, q, r] =-val*par
                                        my_D4[w, u, t, v, s, p, q, r] = val*par
                                        my_D4[w, v, u, t, s, p, q, r] = val*par
                                        my_D4[w, v, t, u, s, p, q, r] =-val*par
                                        my_D4[w, t, v, u, s, p, q, r] = val*par
                                        my_D4[w, t, u, v, s, p, q, r] =-val*par
    D4 = mpi.allreduce(my_D4, mpi.MPI.SUM)
    return D4


def debug_RDM(Quket):
    """
    Debug code for 1, 2, 3, and 4-RDMs.
    This code also illustrates how to contract with integrals.
    """
    n_qubits = Quket.n_qubits
    state = Quket.state
    n_orbs = Quket.n_active_orbitals
    n_active_electrons = Quket.n_active_electrons
    t0 = time.time()
    D1 = get_1RDM_full(state, mapping=Quket.cf.mapping)
    t1 = time.time()
    D2 = get_2RDM_full(state, mapping=Quket.cf.mapping)
    t2 = time.time()
    D3 = get_3RDM_full(state, mapping=Quket.cf.mapping)
    t3 = time.time()
    D4 = get_4RDM_full(state, mapping=Quket.cf.mapping)
    t4 = time.time()

    prints(f'Total time for RDMs in {n_qubits}-qubit simulation.')
    prints(f'D1 : {t1-t0}')
    prints(f'D2 : {t2-t1}')
    prints(f'D3 : {t3-t2}')
    prints(f'D4 : {t4-t3}')
    ### Debugging D2, D3, D4
    X1 = np.zeros((n_qubits,n_qubits))
    X2 = np.zeros((n_qubits,n_qubits,n_qubits,n_qubits))
    X3 = np.zeros((n_qubits,n_qubits,n_qubits,n_qubits,n_qubits,n_qubits))
    my_X3 = np.zeros((n_qubits,n_qubits,n_qubits,n_qubits,n_qubits,n_qubits))
    pqruv = -1
    for p in range(n_qubits):
        for q in range(n_qubits):
            for r in range(n_qubits):
                X1[p,q] += D2[p,r,r,q]/(n_active_electrons-1)
                if n_active_electrons > 2:
                    for u in range(n_qubits):
                        for v in range(n_qubits):
                            X2[p,q,r,u] += D3[p,q,v,v,r,u]/(n_active_electrons-2)
                            pqruv += 1
                            if n_active_electrons > 3:
                                if pqruv % mpi.nprocs == mpi.rank:
                                    for w in range(n_qubits):
                                        for s in range(n_qubits):
                                            my_X3[p,q,r,u,v,w] += D4[p,q,r,s,s,u,v,w]/(n_active_electrons-3)

    if n_active_electrons > 3:
        X3 = mpi.allreduce(my_X3, mpi.MPI.SUM)
    prints('||D1-X1||', np.linalg.norm(D1-X1))
    if n_active_electrons > 2:
        prints('||D2-X2||', np.linalg.norm(D2-X2))
    if n_active_electrons > 3:
        prints('||D3-X3||', np.linalg.norm(D3-X3))

    ### Compute energy from 1-, 2-RDMs by contracting with integrals.
    E0 = Quket.zero_body_integrals_active
    h1 = Quket.one_body_integrals_active
    h2 = Quket.two_body_integrals_active   # h2[p,q,r,s] = 1/2<pq|rs> with p^ q^ r s
    E1 = 0
    E2 = 0
    for p in range(n_qubits):
        for q in range(n_qubits):
            E1 += h1[p,q] * D1[p, q]
            for s in range(n_qubits):
                for r in range(n_qubits):
                    E2 += h2[p,q,r,s] * D2[p,q,r,s]
    prints('Energy contributions computed with 1-, 2-RDMs.')
    prints(f' E0 = {E0}')
    prints(f' E1 = {E1}')
    prints(f' E2 = {E2}')
    prints(f' E0 + E1 + E2 = {E0 + E1 + E2}')
    prints(f' {Quket.ansatz} energy = {Quket.energy}')
    if abs(E0+E1+E2 - Quket.energy) < 1e-8:
        prints('Agreed.')
    else:
        prints('Disagreed.')



def get_relax_delta_full(Quket, print_level=0):
    """Function
    Compute the contributions from relaxed term of the density matrix.
    Usually, Hartree-Fock canonical orbitals are used, and we have to consider the non-Hellmann-Feynman
    term of the (variational) wave function derivative, by enforcing Fpq = 0 under small perturbation.
    (For MP2 FNO or CASSCF, we do not support the relaxed term.)

    For methods that are unitary invariant w.r.t. occ-occ and vir-vir rotations, such as UCC, or PHF,
    it suffices to consider only the occ-vir block (accordingly, only Aaibj).
    However, for more general VQE methods such as ADAPT or hardware-efficient VQE, one has to
    incorporate occ-occ and vir-vir rotations because they are not invariant w.r.t. these rotations.

    Author(s): Taisei Nishimaki, Takashi Tsuchimochi
    """
    from quket.orbital import get_orbital_gradient, get_HF_orbital_hessian
    noa = Quket.noa  # active alpha occupied
    nob = Quket.nob  # active beta occupied
    nva = Quket.nva  # active alpha virtual
    nvb = Quket.nvb  # active beta virtual
    ncore = Quket.n_frozen_orbitals + Quket.n_core_orbitals
    nact = Quket.n_active_orbitals
    norbs = Quket.n_orbitals
    nsec = norbs - ncore - nact
    NOA = noa + ncore
    NOB = nob + ncore
    NVA = nva + nsec
    NVB = nvb + nsec
    NOAVA = NOA*NVA
    NOBVB = NOB*NVB
    # Note: With frozen-core orbitals, we have to 'shift' the index by ncore
    #       in order to actually handle the target orbitals i,j,a,b
    #       (i.e., i <- i + ncore)
    h_pqrs = Quket.two_body_integrals
    mo_energy = Quket.mo_energy

    # active i > j
    ij_size_a = (NOA) * (NOA-1) // 2
    ij_size_b = (NOB) * (NOB-1) // 2
    # active i, inactive I
    iI_size_a = noa * ncore
    iI_size_b = noa * ncore
    # general virtual c (C), general occupied k (K)
    ck_size_a = (NVA) * (NOA)
    ck_size_b = (NVB) * (NOB)
    # active a > b
    ab_size_a = NVA * (NVA-1) // 2
    ab_size_b = NVB * (NVB-1) // 2
    # inactive A, active a
    Aa_size_a = nsec * nva
    Aa_size_b = nsec * nvb

    z_ij_a = np.zeros((1,ij_size_a))
    z_ab_a = np.zeros((1,ab_size_a))
    z_ij_b = np.zeros((1,ij_size_b))
    z_ab_b = np.zeros((1,ab_size_b))

    ### Core + Active space
    Ja, Jb = get_orbital_gradient(Quket)
    Ja = skew(Ja)
    Jb = skew(Jb)
    if cf.debug:
        printmat(Ja, name='Ja')
        printmat(Jb, name='Jb')

    Jck_a = np.zeros(NVA*NOA)
    Jck_b = np.zeros(NVB*NOB)
    ### Jck ###
    ck = 0
    for k in range(NOA):
        for c in range(NVA):
            Jck_a[ck] = Ja[c+NOA,k]
            ck += 1
    ck = 0
    for k in range(NOB):
        for c in range(NVB):
            Jck_b[ck] = Jb[c+NOB,k]
            ck += 1
    J_ck = np.concatenate([Jck_a, Jck_b], 0)

    delta_ij_a = np.zeros((NOA, NOA))
    delta_ij_b = np.zeros((NOB, NOB))
    delta_ab_a = np.zeros((NVA, NVA))
    delta_ab_b = np.zeros((NVB, NVB))

    ij = 0
    for i in range(NOA):
        for j in range(i):
            # print("i,j,ij:",i,j,ij)
            if (mo_energy[i] - mo_energy[j]) < 10e-6:
                z_ij_a[0,ij] = 0.0
                delta_ij_a[i,j] = 0.0
            else:
                z_ij_a[0,ij] = -1*Ja[i,j]/(mo_energy[i] - mo_energy[j])
                delta_ij_a[i,j] = z_ij_a[0,ij]
            # print("zij:",z_ij_a[0,ij])
            ij+=1

    ij = 0
    for i in range(NOB):
        for j in range(i):
            if (mo_energy[i] - mo_energy[j]) < 10e-6:
                z_ij_b[0,ij] = 0.0
                delta_ij_b[i,j] = 0.0
            else:
                z_ij_b[0,ij] = -1*Jb[i,j]/(mo_energy[i] - mo_energy[j])
                delta_ij_b[i,j] = z_ij_b[0,ij]
            ij+=1
    ### z_ab ###
    ab=0
    for a in range(NVA):
        for b in range(a):
            a_index = a+NOA
            b_index = b+NOA
            if (mo_energy[a_index] - mo_energy[b_index]) < 10e-6:
                z_ab_a[0,ab] = 0.0
                delta_ab_a[a,b] = 0.0
            else:
                z_ab_a[0,ab] = -1*Ja[a_index,b_index]/(mo_energy[a_index] - mo_energy[b_index])
                delta_ab_a[a,b] = -1*Ja[a_index,b_index]/(mo_energy[a_index] - mo_energy[b_index])
            ab+=1
    ab = 0
    for a in range(NVB):
        for b in range(a):
            a_index = a+NOB
            b_index = b+NOB
            if (mo_energy[a_index] - mo_energy[b_index]) < 10e-6:
                z_ab_b[0,ab] = 0.0
                delta_ab_b[a,b] = 0.0
            else:
                z_ab_b[0,ab] = -1*Jb[a_index,b_index]/(mo_energy[a_index] - mo_energy[b_index])
                delta_ab_b[a,b] = -1*Jb[a_index,b_index]/(mo_energy[a_index] - mo_energy[b_index])
            ab+=1

    zij = np.hstack([z_ij_a,z_ij_b])
    zab = np.hstack([z_ab_a,z_ab_b])
    if cf.debug:
        printmat(zij,name="Zij")
        printmat(zab,name="Zab")

    Aijck_a = np.zeros((ij_size_a,(NVA)*(NOA)))
    Aijck_b = np.zeros((ij_size_b,(NVB)*(NOB)))
    Aijck_ba = np.zeros((ij_size_b,(NVA)*(NOA)))
    Aijck_ab = np.zeros((ij_size_a,(NVB)*(NOB)))

    Aabck_a = np.zeros((ab_size_a,(NVA)*(NOA)))
    Aabck_b = np.zeros((ab_size_b,(NVB)*(NOB)))
    Aabck_ba = np.zeros((ab_size_b,(NVA)*(NOA)))
    Aabck_ab = np.zeros((ab_size_a,(NVB)*(NOB)))
    ### Aij,ck ###
    ### spin: aa
    ij = 0
    for i in range(NOA):
        for j in range(i):
            ck = 0
            for k in range(NOA):
                for c in range(NVA):
                    # <ic||jk> + <ik||jc> = <ic|jk> - <ic|kj> + <ik|jc> - <ik|cj>
                    #                     = (ij|ck) - (ik|cj) + (ij||kc) - (ic|kj)
                    Aijck_a[ij,ck] = (h_pqrs[i,j,c+NOA,k]
                                     - h_pqrs[i,k,j,c+NOA]
                                     + h_pqrs[i,j,c+NOA,k]
                                     - h_pqrs[i,c+NOA,j,k])
                    ck += 1
            ij+=1
    ### spin: bb
    ij=0
    for i in range(NOB):
        for j in range(i):
            ck = 0
            for k in range(NOB):
                for c in range(NVB):
                    # <ic||jk> + <ik||jc> = <ic|jk> - <ic|kj> + <ik|jc> - <ik|cj>
                    #                     = (ij|ck) - (ik|cj) + (ij||kc) - (ic|kj)
                    Aijck_b[ij,ck] = (h_pqrs[i,j,c+NOB,k]
                                     - h_pqrs[i,k,c+NOB,j]
                                     + h_pqrs[i,j,k,c+NOB]
                                     - h_pqrs[i,c+NOB,k,j])
                    ck+=1
            ij+=1
    ### spin: ba
    ij = 0
    for i in range(NOB):
        for j in range(i):
            ck=0
            for k in range(NOA):
                for c in range(NVA):
                    # <ic||jk> + <ik||jc> = <ic|jk> - <ic|kj> + <ik|jc> - <ik|cj>
                    #                     = (ij|ck) + (ij||kc)
                    Aijck_ba[ij,ck] = (h_pqrs[i,j,c+NOA,k]
                                     + h_pqrs[i,j,k,c+NOA])
                    ck +=1
            ij +=1
    ### spin: ab
    ij = 0
    for i in range(NOA):
        for j in range(i):
            ck=0
            for k in range(NOB):
                for c in range(NVB):
                    # <ic||jk> + <ik||jc> = <ic|jk> - <ic|kj> + <ik|jc> - <ik|cj>
                    #                     = (ij|ck) + (ij||kc)
                    Aijck_ab[ij,ck] = (h_pqrs[i,j,c+NOB,k]
                                     + h_pqrs[i,j,k,c+NOB])
                    ck +=1
            ij +=1

    Aijck = np.block([[Aijck_a , Aijck_ab],
                      [Aijck_ba, Aijck_b]])
    if print_level > 1:
        printmat(Aijck,name="Aijck")

    ### Aab,ck ###
    ### spin: aa
    ab = 0
    for a in range(NVA):
        for b in range(a):
            ck = 0
            for k in range(NOA):
                for c in range(NVA):
                    Aabck_a[ab,ck] = (h_pqrs[a+NOA,b+NOA,c+NOA,k]
                                     - h_pqrs[a+NOA,k,b+NOA,c+NOA]
                                     + h_pqrs[a+NOA,b+NOA,k,c+NOA]
                                     - h_pqrs[a+NOA,c+NOA,k,b+NOA])
                    ck+=1
            ab+=1
    # spin: bb
    ab = 0
    for a in range(NVB):
        for b in range(a):
            ck = 0
            for k in range(NOB):
                for c in range(NVB):
                    Aabck_b[ab,ck] = (h_pqrs[a+NOB,b+NOB,c+NOB,k]
                                     - h_pqrs[a+NOB,k,b+NOB,c+NOB]
                                     + h_pqrs[a+NOB,b+NOB,k,c+NOB]
                                     - h_pqrs[a+NOB,c+NOB,k,b+NOB])
                    ck+=1
            ab+=1
    ### spin:ba
    ab = 0
    for a in range(NVB):
        for b in range(a):
            ck = 0
            for k in range(NOA):
                for c in range(NVA):
                    Aabck_ba[ab,ck] = (h_pqrs[a+NOB,b+NOB,c+NOA,k]
                                     + h_pqrs[a+NOB,b+NOB,k,c+NOA])
                    ck+=1
            ab+=1

    ### spin:ab
    ab = 0
    for a in range(NVA):
        for b in range(a):
            ck = 0
            for k in range(NOB):
                for c in range(NVB):
                    Aabck_ab[ab,ck] = (h_pqrs[a+NOA,b+NOA,c+NOB,k]
                                     + h_pqrs[a+NOA,b+NOA,k,c+NOB])
                    ck+=1
            ab+=1

    Aabck = np.block([[Aabck_a , Aabck_ab],
                      [Aabck_ba, Aabck_b]])
    if print_level > 1:
        printmat(Aabck, name="Aabck")

    J_ijck = np.zeros((NOAVA+NOBVB))
    J_abck = np.zeros((NOAVA+NOBVB))

    ### (82),(83) ###
    for i in range(ij_size_a + ij_size_b):
        for j in range((NVA)*(NOA)+(NVB)*(NOB)):
            temp = zij[0,i]*Aijck[i,j]
            J_ijck[j] += temp
            #print(temp)
    if print_level > 1:
        printmat(J_ijck, name="Jijck")

    for i in range(ab_size_a + ab_size_b):
        for j in range((NVA)*(NOA)+(NVB)*(NOB)):
            temp = zab[0,i]*Aabck[i,j]
            J_abck[j] += temp
            #print(temp)
    if print_level > 1:
        printmat(J_abck, name="Jabck")

    # J_ck_1  = np.vstack([J_ijck_a, J_ijck_b])
    # J_ck_11 = np.vstack([J_abck_a, J_abck_b])
    # printmat(J_ck_1 ,name="J'ck")
    # printmat(J_ck_11,name='J"ck')

    ### symmetrization of zij,zab matrix
    rel_ij_a = (delta_ij_a + delta_ij_a.T)
    rel_ij_b = (delta_ij_b + delta_ij_b.T)
    rel_ab_a = (delta_ab_a + delta_ab_a.T)
    rel_ab_b = (delta_ab_b + delta_ab_b.T)

    if print_level > 0 :
        printmat(rel_ij_a,name="delta_ij_a")
        printmat(rel_ij_b,name="delta_ij_b")
        printmat(rel_ab_a,name="delta_ab_a")
        printmat(rel_ab_b,name="delta_ab_b")


    Aaick = get_HF_orbital_hessian(Quket)
    Jtilde = J_ck.copy()
    Jtilde += J_abck.copy()
    Jtilde += J_ijck.copy()
    A_inv = np.linalg.pinv(Aaick)
    z = A_inv@Jtilde
    if cf.debug:
        printmat(z,name="Zai")

    index = (NOAVA+NOBVB)//2
    z_aa = z[:index]
    z_bb = z[index : NOAVA+NOBVB]
    z_aa_rdm = -z_aa.reshape(NOA, NVA)
    z_bb_rdm = -z_bb.reshape(NOB, NVB)
    delta_pq_aa = np.block([[rel_ij_a, z_aa_rdm],
                              [z_aa_rdm.T, rel_ab_a]])
    delta_pq_bb = np.block([[rel_ij_b, z_bb_rdm],
                              [z_bb_rdm.T, rel_ab_b]])

    if Quket.multiplicity == 1 and Quket.projection.SpinProj:
        delta_pq_aa = (delta_pq_aa + delta_pq_bb)/2
        delta_pq_bb = delta_pq_aa
    return delta_pq_aa/2, delta_pq_bb/2

def get_2RDM(Quket, state=None, symmetry=False):
    """Function
    Compute 2RDM of QuantmState in QuketData (active space only).
    To be explicit,
    if symmetry = True
    Daaaa[pq,rs] = < pA^ qA^ rA sA >   (p>q, r>s)
    Dbaba[p,q,r,s] = < pB^ qA^ rB sA >
    Dbbbb[pq,rs] = < pB^ qB^ rB sB >   (p>q, r>s)

    if symmetry = False
    Daaaa[p,q,r,s] = < pA^ qA^ rA sA >
    Dbaab[p,q,r,s] = < pB^ qA^ rA sB >
    Dbbbb[p,q,r,s] = < pB^ qB^ rB sB >

    Args:
        Quket (QuketData): QuketData instance
        state (QuantumState, optional): Target state for which RDM is computed
        symmetry (bool): If symmetry is true, matrix is folded by anti-symmetry.
    Returns:
        Daaaa (4darray): Alpha 2-particle density matrix in the active space
        Dbbbb (4darray): Beta 2-particle density matrix in the active space
        Dbaab (4darray): Beta-Alpha 2-particle density matrix in the active space

    Author(s): Taisei Nishimaki
    """
    prints(" === Computing 2RDM === ")
    n_qubits = Quket.n_qubits
    if state is None:
        state = Quket.state
    mapping = Quket.cf.mapping
    norbs = Quket.n_active_orbitals
    n_frozen_orbitals = Quket.n_frozen_orbitals
    nott = norbs*(norbs-1)//2

    Daaaa = np.zeros((nott * (nott+1)//2), float)
    Dbbbb = np.zeros((nott * (nott+1)//2), float)
    Dbaba = np.zeros((norbs**2 * (norbs**2 + 1)//2), float)


    t1 = time.time()
    cyc = 0
    pq = 0
    states = {}
    for p in range(norbs):
        pa = p * 2
        pb = pa + 1
        for q in range(p):
            qa = q * 2
            qb = qa + 1
            string_pa = f"{qa} {pa}"
            states[string_pa] = evolve(FermionOperator(string_pa),state,parallel=False,mapping=mapping)
            string_pb = f"{qb} {pb}"
            states[string_pb] = evolve(FermionOperator(string_pb),state,parallel=False,mapping=mapping)
            rs = 0
            for r in range(norbs):
                ra = r * 2
                rb = ra + 1
                for s in range(r):
                    sa = s * 2
                    sb = sa + 1
                    string_qa = f"{sa} {ra}"
                    string_qb = f"{sb} {rb}"
                    if pq < rs:
                        continue
                    cyc += 1
                    if cyc % mpi.nprocs == mpi.rank:
                        #string = f"{pa}^ {qa}^ {ra} {sa}"
                        val = -inner_product(states[string_pa], states[string_qa]).real
                        Daaaa[pq*(pq+1)//2 + rs] = val
                        #string = f"{pb}^ {qb}^ {rb} {sb}"
                        val = -inner_product(states[string_pb], states[string_qb]).real
                        Dbbbb[pq*(pq+1)//2 + rs] = val
                    rs += 1
            pq += 1

    cyc = 0
    pq = 0
    states = {}
    for p in range(norbs):
        pa = p * 2
        pb = pa + 1
        for q in range(norbs):
            qa = q * 2
            qb = qa + 1
            string_p = f"{qa} {pb}"
            states[string_p] = evolve(FermionOperator(string_p),state,parallel=False,mapping=mapping)
            rs = 0
            for r in range(norbs):
                ra = r * 2
                rb = ra + 1
                for s in range(norbs):
                    sa = s * 2
                    sb = sa + 1
                    string_q = f"{sa} {rb}"
                    if pq < rs:
                        continue
                    cyc += 1
                    if cyc % mpi.nprocs == mpi.rank:
                        string = f"{pb}^ {qa}^ {rb} {sa}"
                        val = -inner_product(states[string_p], states[string_q]).real
                        Dbaba[pq*(pq+1)//2 + rs] = +val

                    rs += 1
            pq += 1


    Daaaa = mpi.allreduce(Daaaa, mpi.MPI.SUM)
    Dbbbb = mpi.allreduce(Dbbbb, mpi.MPI.SUM)
    Dbaba = mpi.allreduce(Dbaba, mpi.MPI.SUM)


    if symmetry:
        return Daaaa, Dbaba, Dbbbb
    from quket.linalg import symm, skew
    Daaaa = symm(Daaaa)  # Daaaa[pq, rs]
    Dbbbb = symm(Dbbbb)  # Dbbbb[pq, rs]
    Dbaba = symm(Dbaba)  # Dbaba[pq, rs]

    ### Full tensor
    tmp = np.zeros((nott, norbs**2), float)
    for pq in range(nott):
        tmp[pq, :] = skew(Daaaa[pq, :]).reshape(norbs**2)
    tmp1 = tmp.T.copy()
    Daaaa = np.zeros((norbs**2, norbs**2), float)
    for rs in range(norbs**2):
        Daaaa[rs, :] = skew(tmp1[rs, :]).reshape(norbs**2)
    Daaaa = Daaaa.reshape(norbs, norbs, norbs, norbs)

    for pq in range(nott):
        tmp[pq, :] = skew(Dbbbb[pq, :]).reshape(norbs**2)
    tmp1 = tmp.T.copy()
    Dbbbb = np.zeros((norbs**2, norbs**2), float)
    for rs in range(norbs**2):
        Dbbbb[rs, :] = skew(tmp1[rs, :]).reshape(norbs**2)
    Dbbbb = Dbbbb.reshape(norbs, norbs, norbs, norbs)

    Dbaba = Dbaba.reshape(norbs, norbs, norbs, norbs)

    return Daaaa, -Dbaba.transpose(0,1,3,2).copy(), Dbbbb
    ###    Daaaa  Dbaab  Dbbbb



def get_3RDM(Quket, state=None, symmetry=False):
    """Function
    Compute 3RDM of QuantmState in QuketData (active space only).
    To be explicit,
    if symmetry = True,
    Daaaaaa[pqr,stu] = < pA^ qA^ rA^ sA tA uA>     (p>q>r,  s>t>u)
    Dbaabaa[p,qr,s,tu] = < pB^ qA^ rA^ sB tA uA>   (q>r, t>u)
    Dbbabba[pq,r,st,u] = < pB^ qB^ rA^ sB tB uA> (p>q, s>t)
    Dbbbbbb[pqr,stu] = < pB^ qB^ rB^ sB tB uB >    (p>q>r,  s>t>u)

    if symmetry = False,
    Daaaaaa[p,q,r,s,t,u] = < pA^ qA^ rA^ sA tA uA >
    Dbaaaab[p,q,r,s,t,u] = < pB^ qA^ rA^ sA tA uB >
    Dbbaabb[p,q,r,s,t,u] = < pB^ qB^ rA^ sA tB uB >
    Dbbbbbb[p,q,r,s,t,u] = < pB^ qB^ rB^ sB tB uB >

    Args:
        Quket (QuketData): QuketData instance
        state (QuantumState, optional): Target state for which RDM is computed
        symmetry (bool): If symmetry is true, matrix is folded by anti-symmetry.
    Author(s): Takashi Tsuchimochi
    """
    prints(" === Computing 3RDM === ")
    n_qubits = Quket.n_qubits
    if state is None:
        state = Quket.state
    mapping = Quket.cf.mapping

    norbs = Quket.n_active_orbitals
    n_frozen_orbitals = Quket.n_frozen_orbitals
    nott = norbs*(norbs-1)//2
    nottt = norbs*(norbs-1)*(norbs-2)//6

    Daaaaaa = np.zeros((nottt * (nottt+1)//2), float)
    Dbbbbbb = np.zeros((nottt * (nottt+1)//2), float)
    Dbaabaa = np.zeros((norbs * nott * (norbs * nott + 1)//2), float)
    Dbbabba = np.zeros((norbs * nott * (norbs * nott + 1)//2), float)

    ### AAA
    ### BBB
    t0 = time.time()
    cyc = 0
    p123 = 0
    states = {}
    for p1 in range(norbs):
        #
        p1a = p1 * 2
        p1b = p1a + 1
        for p2 in range(p1):
            p2a = p2 * 2
            p2b = p2a + 1
            for p3 in range(p2):
                p3a = p3 * 2
                p3b = p3a + 1
                p3a_state = evolve(FermionOperator(f"{p3a} {p2a} {p1a}"), state, parallel=False, mapping=mapping)
                p3b_state = evolve(FermionOperator(f"{p3b} {p2b} {p1b}"), state, parallel=False, mapping=mapping)
                states[f"{p3a} {p2a} {p1a}"] = p3a_state
                states[f"{p3b} {p2b} {p1b}"] = p3b_state
                q123 = 0
                for q1 in range(norbs):
                    q1a = q1 * 2
                    q1b = q1a + 1
                    for q2 in range(q1):
                        q2a = q2 * 2
                        q2b = q2a + 1
                        for q3 in range(q2):
                            if p123 < q123:
                                continue
                            q3a = q3 * 2
                            q3b = q3a + 1
                            cyc += 1
                            if cyc % mpi.nprocs == mpi.rank:
                                #string = f"{p1a}^ {p2a}^ {p3a}^ {q1a} {q2a} {q3a}"
                                #val = inner_product(p3a_state, q3a_state).real
                                #val = inner_product(states_a[p123], states_a[q123]).real
                                val = inner_product(states[f"{p3a} {p2a} {p1a}"], states[f"{q3a} {q2a} {q1a}"]).real
                                Daaaaaa[p123*(p123+1)//2 + q123] = -val
                                #string = f"{p1b}^ {p2b}^ {p3b}^ {q1b} {q2b} {q3b}"
                                val = inner_product(states[f"{p3b} {p2b} {p1b}"], states[f"{q3b} {q2b} {q1b}"]).real
                                Dbbbbbb[p123*(p123+1)//2 + q123] = -val
                                #prints('val ', val + val_)
                                #prints('val ', abs(val) - abs(val_))
                            q123 += 1
                p123 += 1


    ### BAA
    cyc = 0
    p123q123 = 0
    p123 = 0
    states = {}
    for p1 in range(norbs):
        p1a = p1 * 2
        p1b = p1a + 1
        for p2 in range(norbs):
            p2a = p2 * 2
            p2b = p2a + 1
            for p3 in range(p2):
                p3a = p3 * 2
                p3b = p3a + 1
                string = f"{p3a} {p2a} {p1b}"
                p3_state = evolve(FermionOperator(string), state, parallel=False, mapping=mapping)
                states[string] = p3_state
                q123 = 0
                for q1 in range(norbs):
                    q1a = q1 * 2
                    q1b = q1a + 1
                    for q2 in range(norbs):
                        q2a = q2 * 2
                        q2b = q2a + 1
                        for q3 in range(q2):
                            q3a = q3 * 2
                            q3b = q3a + 1
                            if p123 < q123:
                                continue
                            cyc += 1
                            if cyc % mpi.nprocs == mpi.rank:
                                #string = f"{p1b}^ {p2a}^ {p3a}^ {q1b} {q2a} {q3a}"
                                #q3_state = evolve(FermionOperator(f"{q3a} {q2a} {q1b}"), state, parallel=False)
                                #val = -inner_product(p3_state, q3_state).real
                                val = -inner_product(states[f"{p3a} {p2a} {p1b}"], states[f"{q3a} {q2a} {q1b}"]).real
                                Dbaabaa[p123q123] = val
                            p123q123 += 1
                            q123 += 1
                p123 += 1

    ### BBA
    cyc = 0
    p123q123 = 0
    p123 = 0
    states = {}
    for p1 in range(norbs):
        p1a = p1 * 2
        p1b = p1a + 1
        for p2 in range(p1):
            p2a = p2 * 2
            p2b = p2a + 1
            for p3 in range(norbs):
                p3a = p3 * 2
                p3b = p3a + 1
                string = f"{p3a} {p2b} {p1b}"
                p3_state = evolve(FermionOperator(string), state, parallel=False, mapping=mapping)
                states[string] = p3_state
                q123 = 0
                for q1 in range(norbs):
                    q1a = q1 * 2
                    q1b = q1a + 1
                    for q2 in range(q1):
                        q2a = q2 * 2
                        q2b = q2a + 1
                        for q3 in range(norbs):
                            q3a = q3 * 2
                            q3b = q3a + 1
                            if p123 < q123:
                                continue
                            cyc += 1
                            if cyc % mpi.nprocs == mpi.rank:
                                #string = f"{p1b}^ {p2b}^ {p3a}^ {q1b} {q2b} {q3a}"
                                #val = -inner_product(p3_state, q3_state).real
                                val = -inner_product(states[f"{p3a} {p2b} {p1b}"], states[f"{q3a} {q2b} {q1b}"]).real
                                Dbbabba[p123q123] = val
                            p123q123 += 1
                            q123 += 1
                p123 += 1

    Daaaaaa = mpi.allreduce(Daaaaaa, mpi.MPI.SUM)
    Dbbbbbb = mpi.allreduce(Dbbbbbb, mpi.MPI.SUM)
    Dbaabaa = mpi.allreduce(Dbaabaa, mpi.MPI.SUM)
    Dbbabba = mpi.allreduce(Dbbabba, mpi.MPI.SUM)

    t1 = time.time()

    if symmetry:
        return Daaaaaa, Dbaabaa, Dbbabba, Dbbbbbb


    Daaaaaa = symm(Daaaaaa)  # Daaaaaa[pqr, stu]
    Dbbbbbb = symm(Dbbbbbb)  # Dbbbbbb[pqr, stu]
    Dbaabaa = symm(Dbaabaa).reshape(norbs, nott, norbs, nott)  # Dbaabaa[p, qr, s, tu]
    Dbbabba = symm(Dbbabba).reshape(nott, norbs, nott, norbs)  # Dbaabaa[pq, r, st, u]

    ### Full tensor
    tmp = np.zeros((nottt, norbs**3), float)
    if nottt > 0:
        for pqr in range(nottt):
            tmp[pqr, :] = skew3(Daaaaaa[pqr, :]).reshape(norbs**3)
        tmp1 = tmp.T.copy()
        Daaaaaa = np.zeros((norbs**3, norbs**3), float)
        for stu in range(norbs**3):
            Daaaaaa[stu, :] = skew3(tmp1[stu, :]).reshape(norbs**3)
        Daaaaaa = Daaaaaa.reshape(norbs, norbs, norbs, norbs, norbs, norbs)

        tmp = np.zeros((nottt, norbs**3), float)
        for pqr in range(nottt):
            tmp[pqr, :] = skew3(Dbbbbbb[pqr, :]).reshape(norbs**3)
        tmp1 = tmp.T.copy()
        Dbbbbbb = np.zeros((norbs**3, norbs**3), float)
        for stu in range(norbs**3):
            Dbbbbbb[stu, :] = skew3(tmp1[stu, :]).reshape(norbs**3)
        Dbbbbbb = Dbbbbbb.reshape(norbs, norbs, norbs, norbs, norbs, norbs)
    else:
        Daaaaaa = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs), float)
        Dbbbbbb = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs), float)

    # bbabba [p, qr, s, tu]
    #         b  aa  b  aa
    if nott > 0:
        tmp = np.zeros((norbs, nott, norbs, norbs**2), float)
        for p in range(norbs):
            for qr in range(nott):
                for s in range(norbs):
                    tmp[p, qr, s, :] = skew(Dbaabaa[p, qr, s, :]).reshape(norbs**2)

        tmp1 = tmp.transpose(2, 3, 0, 1).reshape(norbs**3, norbs, nott)
        Dbaabaa = np.zeros((norbs**3, norbs, norbs**2), float)
        for stu in range(norbs**3):
            for p in range(norbs):
                Dbaabaa[stu, p, :] = skew(tmp1[stu, p, :]).reshape(norbs**2)
        Dbaabaa = Dbaabaa.reshape(norbs, norbs, norbs, norbs, norbs, norbs)

        # bbabba [pq, r, st, u]
        #         bb  a  bb  a
        tmp = np.zeros((nott, norbs, norbs, norbs**2), float)
        for pq in range(nott):
            for r in range(norbs):
                for u in range(norbs):
                    tmp[pq, r, u, :] = skew(Dbbabba[pq, r, :, u]).reshape(norbs**2)

        ### pq, r, u, [s, t]--> [s,t,u], r, pq
        tmp1 = tmp.transpose(3, 2, 1, 0).reshape(norbs**3, norbs, nott)
        Dbbabba = np.zeros((norbs**3, norbs, norbs**2), float)
        for stu in range(norbs**3):
            for r in range(norbs):
                Dbbabba[stu, r, :] = skew(tmp1[stu, r, :]).reshape(norbs**2)
        Dbbabba = Dbbabba.transpose(0, 2, 1).reshape(norbs, norbs, norbs, norbs, norbs, norbs)
    else:
        Dbaabaa = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs), float)
        Dbbabba = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs), float)
    t2 = time.time()
    prints(f"{t2-t0:.3f} sec")

    return Daaaaaa, Dbaabaa.transpose(0,1,2,4,5,3).copy(), Dbbabba.transpose(0,1,2,5,3,4).copy(), Dbbbbbb
    #####  Daaaaaa  Dbaaaab  Dbbaabb  Dbbbbbb


def get_4RDM(Quket, state=None, symmetry=False):
    """Function
    Compute 4RDM of QuantmState in QuketData (active space only).
    To be explicit,
    if symmetry = True,
    Daaaaaaaa[pqrs,tuvw] = < pA^ qA^ rA^ sA^ tA uA vA wA >     (p>q>r>s,  t>u>v>w)
    Dbbbbbbbb[pqrs,tuvw] = < pB^ qB^ rB^ sB^ tB uB vB wB >    (p>q>r>w,  t>u>v>w)
    Dbaaabaaa[p,qrs,tuv, w] = < pB^ qA^ rA^ sA^ tB uA vA wA>   (q>r>s, u>v>w)
    Dbbaabbaa[pq,rs,tu,vw] = < pB^ qB^ rA^ sA^ tB uB vA wA > (p>q, r>s, t>u, v>w)
    Dbbbabbba[pqr,s,tuv,w] = < pB^ qB^ rB^ sA^ tB uB vB wA > (p>q>r>, t>u>v)

    If symmetry = False,
    Daaaaaaaa[p,q,r,s,t,u,v,w] = < pA^ qA^ rA^ sA^ tA uA vA wA >
    Dbaaaaaab[p,q,r,s,t,u,v,w] = < pB^ qA^ rA^ sA^ tA uA vA wB >
    Dbbaaaabb[p,q,r,s,t,u,v,w] = < pB^ qB^ rA^ sA^ tA uA vB wB >
    Dbbbaabbb[p,q,r,s,t,u,v,w] = < pB^ qB^ rB^ sA^ tA uB vB wB >
    Dbbbbbbbb[p,q,r,s,t,u,v,w] = < pB^ qB^ rB^ sB^ tB uB vB wB >

    Args:
        Quket (QuketData): QuketData instance
        state (QuantumState, optional): Target state for which RDM is computed
        symmetry (bool): If symmetry is true, matrix is folded by anti-symmetry.
    Author(s): Takashi Tsuchimochi
    """
    prints(" === Computing 4RDM === ")
    n_qubits = Quket.n_qubits
    if state is None:
        state = Quket.state
    mapping = Quket.cf.mapping

    norbs = Quket.n_active_orbitals
    n_frozen_orbitals = Quket.n_frozen_orbitals
    nott = norbs*(norbs-1)//2
    nottt = norbs*(norbs-1)*(norbs-2)//6
    notttt = norbs*(norbs-1)*(norbs-2)*(norbs-3)//24

    Daaaaaaaa = np.zeros((notttt * (notttt+1)//2), float)
    Dbbbbbbbb = np.zeros((notttt * (notttt+1)//2), float)
    Dbaaabaaa = np.zeros((norbs * nottt * (norbs * nottt + 1)//2), float)
    Dbbaabbaa = np.zeros((nott * nott * (nott * nott + 1)//2), float)
    Dbbbabbba = np.zeros((norbs * nottt * (norbs * nottt + 1)//2), float)

    ### AAAA
    ### BBBB
    t0 = time.time()
    cyc = 0
    p1234 = 0
    p1234q1234 = 0
    states = {}
    for p1 in range(norbs):
        p1a = p1 * 2
        p1b = p1a + 1
        for p2 in range(p1):
            p2a = p2 * 2
            p2b = p2a + 1
            for p3 in range(p2):
                p3a = p3 * 2
                p3b = p3a + 1
                for p4 in range(p3):
                    p4a = p4 * 2
                    p4b = p4a + 1
                    string_pa = f"{p4a} {p3a} {p2a} {p1a}"
                    p4a_state = evolve(FermionOperator(string_pa), state, parallel=False, mapping=mapping)
                    states[string_pa] = p4a_state
                    string_pb = f"{p4b} {p3b} {p2b} {p1b}"
                    p4b_state = evolve(FermionOperator(string_pb), state, parallel=False, mapping=mapping)
                    states[string_pb] = p4b_state
                    q1234 = 0
                    for q1 in range(norbs):
                        q1a = q1 * 2
                        q1b = q1a + 1
                        for q2 in range(q1):
                            q2a = q2 * 2
                            q2b = q2a + 1
                            for q3 in range(q2):
                                q3a = q3 * 2
                                q3b = q3a + 1
                                for q4 in range(q3):
                                    q4a = q4 * 2
                                    q4b = q4a + 1
                                    string_qa = f"{q4a} {q3a} {q2a} {q1a}"
                                    string_qb = f"{q4b} {q3b} {q2b} {q1b}"
                                    if p1234 < q1234:
                                        continue
                                    cyc += 1
                                    if cyc % mpi.nprocs == mpi.rank:
                                        string = f"{p1a}^ {p2a}^ {p3a}^ {p4a}^ {q1a} {q2a} {q3a} {q4a}"
                                        val = inner_product(states[string_pa], states[string_qa]).real
                                        Daaaaaaaa[p1234q1234] = val
                                        string = f"{p1b}^ {p2b}^ {p3b}^ {p4b}^ {q1b} {q2b} {q3b} {q4b}"
                                        val = inner_product(states[string_pb], states[string_qb]).real
                                        Dbbbbbbbb[p1234q1234] = val
                                    q1234 += 1
                                    p1234q1234 += 1
                    p1234 += 1

    # BAAA
    cyc = 0
    p1234 = 0
    p1234q1234 = 0
    states = {}
    for p1 in range(norbs):
        p1a = p1 * 2
        p1b = p1a + 1
        for p2 in range(norbs):
            p2a = p2 * 2
            p2b = p2a + 1
            for p3 in range(p2):
                p3a = p3 * 2
                p3b = p3a + 1
                for p4 in range(p3):
                    p4a = p4 * 2
                    p4b = p4a + 1
                    string_p = f"{p4a} {p3a} {p2a} {p1b}"
                    p4_state = evolve(FermionOperator(string_p), state, parallel=False, mapping=mapping)
                    states[string_p] = p4_state
                    q1234 = 0
                    for q1 in range(norbs):
                        q1a = q1 * 2
                        q1b = q1a + 1
                        for q2 in range(norbs):
                            q2a = q2 * 2
                            q2b = q2a + 1
                            for q3 in range(q2):
                                q3a = q3 * 2
                                q3b = q3a + 1
                                for q4 in range(q3):
                                    q4a = q4 * 2
                                    q4b = q4a + 1
                                    string_q = f"{q4a} {q3a} {q2a} {q1b}"
                                    if p1234 < q1234:
                                        continue
                                    cyc += 1
                                    if cyc % mpi.nprocs == mpi.rank:
                                        string = f"{p1b}^ {p2a}^ {p3a}^ {p4a}^ {q1b} {q2a} {q3a} {q4a}"
                                        val = inner_product(states[string_p], states[string_q]).real
                                        Dbaaabaaa[p1234q1234] = val
                                    q1234 += 1
                                    p1234q1234 += 1
                    p1234 += 1

    # BBAA
    cyc = 0
    p1234 = 0
    p1234q1234 = 0
    states = {}
    for p1 in range(norbs):
        p1a = p1 * 2
        p1b = p1a + 1
        for p2 in range(p1):
            p2a = p2 * 2
            p2b = p2a + 1
            for p3 in range(norbs):
                p3a = p3 * 2
                p3b = p3a + 1
                for p4 in range(p3):
                    p4a = p4 * 2
                    p4b = p4a + 1
                    string_p = f"{p4a} {p3a} {p2b} {p1b}"
                    p4_state = evolve(FermionOperator(string_p), state, parallel=False, mapping=mapping)
                    states[string_p] = p4_state
                    q1234 = 0
                    for q1 in range(norbs):
                        q1a = q1 * 2
                        q1b = q1a + 1
                        for q2 in range(q1):
                            q2a = q2 * 2
                            q2b = q2a + 1
                            for q3 in range(norbs):
                                q3a = q3 * 2
                                q3b = q3a + 1
                                for q4 in range(q3):
                                    q4a = q4 * 2
                                    q4b = q4a + 1
                                    string_q = f"{q4a} {q3a} {q2b} {q1b}"
                                    if p1234 < q1234:
                                        continue
                                    cyc += 1
                                    if cyc % mpi.nprocs == mpi.rank:
                                        string = f"{p1b}^ {p2b}^ {p3a}^ {p4a}^ {q1b} {q2b} {q3a} {q4a}"
                                        val = inner_product(states[string_p], states[string_q]).real
                                        Dbbaabbaa[p1234q1234] = val
                                    q1234 += 1
                                    p1234q1234 += 1
                    p1234 += 1

    # BBBA
    cyc = 0
    p1234 = 0
    p1234q1234 = 0
    states = {}
    for p1 in range(norbs):
        p1a = p1 * 2
        p1b = p1a + 1
        for p2 in range(p1):
            p2a = p2 * 2
            p2b = p2a + 1
            for p3 in range(p2):
                p3a = p3 * 2
                p3b = p3a + 1
                for p4 in range(norbs):
                    p4a = p4 * 2
                    p4b = p4a + 1
                    string_p = f"{p4a} {p3b} {p2b} {p1b}"
                    p4_state = evolve(FermionOperator(string_p), state, parallel=False, mapping=mapping)
                    states[string_p] = p4_state
                    q1234 = 0
                    for q1 in range(norbs):
                        q1a = q1 * 2
                        q1b = q1a + 1
                        for q2 in range(q1):
                            q2a = q2 * 2
                            q2b = q2a + 1
                            for q3 in range(q2):
                                q3a = q3 * 2
                                q3b = q3a + 1
                                for q4 in range(norbs):
                                    q4a = q4 * 2
                                    q4b = q4a + 1
                                    string_q = f"{q4a} {q3b} {q2b} {q1b}"
                                    if p1234 < q1234:
                                        continue
                                    cyc += 1
                                    if cyc % mpi.nprocs == mpi.rank:
                                        string = f"{p1b}^ {p2b}^ {p3b}^ {p4a}^ {q1b} {q2b} {q3b} {q4a}"
                                        val = inner_product(states[string_p], states[string_q]).real
                                        Dbbbabbba[p1234q1234] = val
                                    q1234 += 1
                                    p1234q1234 += 1
                    p1234 += 1

    Daaaaaaaa = mpi.allreduce(Daaaaaaaa, mpi.MPI.SUM)
    Dbbbbbbbb = mpi.allreduce(Dbbbbbbbb, mpi.MPI.SUM)
    Dbaaabaaa = mpi.allreduce(Dbaaabaaa, mpi.MPI.SUM)
    Dbbaabbaa = mpi.allreduce(Dbbaabbaa, mpi.MPI.SUM)
    Dbbbabbba = mpi.allreduce(Dbbbabbba, mpi.MPI.SUM)

    t1 = time.time()

    Daaaaaaaa = symm(Daaaaaaaa)  # Daaaaaaaa[pqrs, tuvw]
    Dbbbbbbbb = symm(Dbbbbbbbb)  # Dbbbbbbbb[pqrs, tuvw]
    Dbaaabaaa = symm(Dbaaabaaa).reshape(norbs, nottt, norbs, nottt)  # Dbaaabaaa[p, qrs, t, uvw]
    Dbbaabbaa = symm(Dbbaabbaa).reshape(nott, nott, nott, nott)  # Dbaabaa[pq, rs, tu, vw]
    Dbbbabbba = symm(Dbbbabbba).reshape(nottt, norbs, nottt, norbs)  # Dbbbabbba[pqr, s, tuv, w]

    if symmetry:
        return Daaaaaaaa, Dbaaabaaa, Dbbaabbaa, Dbbbabbba, Dbbbbbbbb


    ### Full tensor
    if notttt > 0:
        tmp = np.zeros((notttt, norbs**4), float)
        for pqrs in range(notttt):
            tmp[pqrs, :] = skew4(Daaaaaaaa[pqrs, :]).reshape(norbs**4)
        tmp1 = tmp.T.copy()
        Daaaaaaaa = np.zeros((norbs**4, norbs**4), float)
        for tuvw in range(norbs**4):
            Daaaaaaaa[tuvw, :] = skew4(tmp1[tuvw, :]).reshape(norbs**4)
        Daaaaaaaa = Daaaaaaaa.reshape(norbs, norbs, norbs, norbs, norbs, norbs, norbs, norbs)

        for pqrs in range(notttt):
            tmp[pqrs, :] = skew4(Dbbbbbbbb[pqrs, :]).reshape(norbs**4)
        tmp1 = tmp.T.copy()
        Dbbbbbbbb = np.zeros((norbs**4, norbs**4), float)
        for tuvw in range(norbs**4):
            Dbbbbbbbb[tuvw, :] = skew4(tmp1[tuvw, :]).reshape(norbs**4)
        Dbbbbbbbb = Dbbbbbbbb.reshape(norbs, norbs, norbs, norbs, norbs, norbs, norbs, norbs)
    else:
        Daaaaaaaa = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs, norbs, norbs), float)
        Dbbbbbbbb = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs, norbs, norbs), float)
    # baaabaaa [p, qrs, t, uvw]
    #           b  aaa  b  aaa
    if nottt > 0:
        tmp = np.zeros((norbs, nottt, norbs, norbs**3), float)
        for p in range(norbs):
            for qrs in range(nottt):
                for t in range(norbs):
                    tmp[p, qrs, t, :] = skew3(Dbaaabaaa[p, qrs, t, :]).reshape(norbs**3)
        # [p, qrs, t, [u,v,w]]  ->  [[t,u,v,w], p, qrs]
        tmp1 = tmp.transpose(2, 3, 0, 1).reshape(norbs**4, norbs, nottt)
        Dbaaabaaa = np.zeros((norbs**4, norbs, norbs**3), float)
        for tuvw in range(norbs**4):
            for p in range(norbs):
                Dbaaabaaa[tuvw, p, :] = skew3(tmp1[tuvw, p, :]).reshape(norbs**3)
        Dbaaabaaa = Dbaaabaaa.reshape(norbs, norbs, norbs, norbs, norbs, norbs, norbs, norbs)
    else:
        Dbaaabaaa = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs, norbs, norbs), float)

    # bbaabbaa [pq, rs, tu, vw]
    #           aa  bb  aa  bb
    if nott > 0:
        tmp = np.zeros((nott, nott, norbs**2, norbs**2), float)
        tmp_ = np.zeros(( nott, norbs**2), float)
        for pq in range(nott):
            for rs in range(nott):
                for tu in range(nott):
                    tmp_[tu, :] = skew(Dbbaabbaa[pq, rs, tu, :]).reshape(norbs**2)
                ### pq, rs, tu, [v, w]--> pq, rs, [v,w], tu
                tmp__ = tmp_.T.reshape(norbs**2, nott)
                for vw in range(norbs**2):
                    tmp[pq, rs, vw, :] = skew(tmp__[vw, :]).reshape(norbs**2)

        ### [pq, rs, [v,w] ,[t,u]]   ->   [[v,w,t,u], pq, rs]
        #    aa  bb   b b    a a            b b a a   aa  bb
        tmp1 = tmp.transpose(2, 3, 0, 1).reshape(norbs**4, nott, nott)
        Dbbaabbaa = np.zeros((norbs**4, norbs**2, norbs**2), float)
        for vwtu in range(norbs**4):
            for pq in range(nott):
                tmp_[pq, :] = skew(tmp1[vwtu, pq, :]).reshape(norbs**2)
            ### [v,w,t,u], pq, [r,s] --> [v,w,t,u], [r,s], pq
            tmp__ = tmp_.T.reshape(norbs**2, nott)
            for rs in range(norbs**2):
                Dbbaabbaa[vwtu, rs, :] = skew(tmp__[rs, :]).reshape(norbs**2)
        Dbbaabbaa = Dbbaabbaa.reshape(norbs, norbs, norbs, norbs, norbs, norbs, norbs, norbs)
    else:
        Dbbaabbaa = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs, norbs, norbs), float)

    # bbbabbba [pqr, s, tuv, w]
    #           bbb  a  bbb  a
    if nottt > 0:
        tmp = np.zeros((nottt, norbs, norbs, norbs**3), float)
        tmp_ = np.zeros(( norbs, norbs**3), float)
        for pqr in range(nottt):
            for s in range(norbs):
                tmp_ = Dbbbabbba[pqr, s, :, :].T.copy()
                for w in range(norbs):
                    tmp[pqr, s, w, :] = skew3(tmp_[w, :]).reshape(norbs**3)
        ### [pqr, s, w, [t,u,v]]  -> [[t,u,v], w, s, pqr]
        ###  bbb  a  a   b b b         b b b   a  a  bbb
        tmp1 = tmp.transpose(3, 2, 1, 0).reshape(norbs**3, norbs, norbs, nottt)
        Dbbbabbba = np.zeros((norbs**3, norbs, norbs, norbs**3), float)
        for tuv in range(norbs**3):
            for w in range(norbs):
                for s in range(norbs):
                    Dbbbabbba[tuv, w, s, :] = skew3(tmp1[tuv, w, s, :]).reshape(norbs**3)

        Dbbbabbba = Dbbbabbba.transpose(0, 1, 3, 2).reshape(norbs, norbs, norbs, norbs, norbs, norbs, norbs, norbs)
    else:
        Dbbbabbba = np.zeros((norbs, norbs, norbs, norbs, norbs, norbs, norbs, norbs), float)

    t2 = time.time()
    prints(f"{t2-t0:.3f} sec")

    return Daaaaaaaa, -Dbaaabaaa.transpose(0,1,2,3,5,6,7,4).copy(), Dbbaabbaa.transpose(0,1,2,3,6,7,4,5).copy(), -Dbbbabbba.transpose(0,1,2,3,7,4,5,6).copy(), Dbbbbbbbb
    #####  Daaaaaaaa  Dbaaaaaab  Dbbaaaabb  Dbbbaabbb  Dbbbbbbbb

