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

qse.py

Quantum Subspace Expansion.

"""
import numpy as np
import scipy as sp
import time
from qulacs.state import inner_product

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import error, prints, printmat, print_state
from quket.opelib import OpenFermionOperator2QulacsObservable, OpenFermionOperator2QulacsGeneralOperator, single_operator_gradient, spin_single_grad, Separate_Fermionic_Hamiltonian
from quket.opelib import evolve
from quket.linalg import root_inv
from quket.utils import int2occ
from quket.utils.utils import get_tau
from quket.lib import get_fermion_operator,jordan_wigner,bravyi_kitaev, normal_ordered
from .rdm import get_1RDM_full, get_2RDM_full, get_3RDM_full, get_4RDM_full

def QSE_driver(Quket, method="QSE"):
    """Function:
    Main driver of QSE and CIS(debug for QSE).

    Args:
        Quket (QuketData): Quket data
        method (string): method(QSE or CIS)
    """
    norbs = Quket.n_active_orbitals
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb

    t0 = time.time()
    cf.t_old = t0

    excitation_list = []
    nexcite = 1

    prints(f"\n Performing {method} ")
    if Quket.post_general:
        prints(f'UCCGS subspace')
        ### reference
        if method in ["QSE"]:
            excitation_list.append([0,0])
        if method in ["CIS"]:
            excitation_list.append([])
        ### singles ###
        for p in range(Quket.n_active_orbitals):
            for q in range(p):
                x = [2*p, 2*q]
                excitation_list.append(x)
                nexcite += 1
                x = [2*p+1, 2*q+1]
                excitation_list.append(x)
                nexcite += 1
    else:
        prints("UCCS subspace")
        ## singles ##
        ## reference
        if method in ["QSE"]:
            excitation_list.append([0,0])
        if method in ["CIS"]:
            excitation_list.append([])
        ## alpha ##
        for i in range(noa):
            for a in range(nva):
                x = [2*(a+noa), 2*i]
                excitation_list.append(x)
                nexcite += 1
        ## beta ##
        for j in range(nob):
            for b in range(nvb):
                x = [2*(b+nob)+1, 2*j+1]
                excitation_list.append(x)
                nexcite += 1
    
    prints(f"Number of operators in pool: {nexcite}") 
    if method in ["QSE"]:
        Hmat, Smat = create_HS4QSE(Quket.operators.qubit_Hamiltonian, excitation_list, Quket)
    if method in ["CIS"]:
        hc = False
        Hmat, Smat = get_HS4CIS(Quket.operators.qubit_Hamiltonian, excitation_list, Quket.state, Quket.operators.Hamiltonian, hc=hc)
    # if cf.debug:
        #print(f"check Hqse - Hcis: \n{Hqse - Hcis}")
        #print(f"check Sqse - Scis: \n{Sqse - Scis}")
        #printmat(Hqse-Hcis,name="debug H")
        #printmat(Sqse-Scis,name="debug S")
    
    root_invS = root_inv(Smat)
    rank = root_invS.shape[1]

    H_ortho = root_invS.T @ Hmat @ root_invS
    e, U  = sp.linalg.eigh(H_ortho)

    # Retrieve CI coefficients in the original basis 
    c = root_invS @ U
    # Track the reference-dominant CI vector
    i = -1
    c0 = 0
    while abs(c0) < 0.1:
        i += 1
        t_amp = c[:,i]
        # c0 = (t_amp[0]).real
        c0 = (Smat[0] @ c[:,i]).real
    
    gs_index = i
    E_gs = e[i]

    prints(f"E({method}):",e)
    prints("E_gs:",E_gs)
    #En = e[i] - Quket.energy
    prints('Reference energy = ',Quket.energy)
    ii = -1 
    for i in range(gs_index, nexcite):
        ii += 1
        if ii == 0:
            prints(f"grand state energy from {method}: {e[i]}")
        elif ii == 1:
            prints(f"1st excited state energy from {method}: {e[i]}")
            #prints(f"E[{i}]-E[{i-1}]: {e[i]-e[i-1]}")
        elif ii == 2:
            prints(f"2nd excited state energy from {method}: {e[i]}")
            #prints(f"E[{i}]-E[{i-1}]: {e[i]-e[i-1]}")
        elif ii == 3:
            prints(f"3rd excited state energy from {method}: {e[i]}")
            #prints(f"E[{i}]-E[{i-1}]: {e[i]-e[i-1]}")
        else:
            prints(f"{ii}th excited state energy from {method}: {e[i]}")

    # エネルギーの低い順に並び替え
    # ind = np.argsort(e.real, -1)
    # e = en.real[ind]
    # U = U[:, ind]


    """
    Davidson correction
    EQ = (1 - c0**2) * En
    prints('Reference energy = ',Quket.energy)
    prints('QSE correlation energy = ', En)
    prints('QSE total energy = ', Quket.energy + En)
    prints('QSE+Q total energy = ', Quket.energy + En + EQ)
    """
    t2 = time.time()
    cput = t2 - t0
    prints("\n Done: CPU Time =  ", "%15.4f" % cput)




def cis_driver(Quket, print_level):
    """Function:
    Main driver for CIS . Use to debug QSE
    """
    prints('Entered CIS solver')

    t0 = time.time()
    cf.t_old = t0
    maxiter = Quket.maxiter

    norbs = Quket.n_active_orbitals
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    Quket.energy = Quket.qulacs.Hamiltonian.get_expectation_value(Quket.state)

    ##number of operator is going to dynamically increase
    theta_list = []  ## initialize

    prints("Performing CIS ", end="")
    if Quket.projection.SpinProj:
        prints("(spin-projection) ", end="")
    print_state(Quket.state, name='Reference state')

    nexcite = 1
    excitation_list = []
    if Quket.post_general:
        prints(f'UCCGS subspace')
        ### reference
        excitation_list.append([])
        ### singles ###
        for p in range(Quket.n_active_orbitals):
            for q in range(p):
                x = [2*p, 2*q]
                excitation_list.append(x)
                nexcite += 1
                x = [2*p+1, 2*q+1]
                excitation_list.append(x)
                nexcite += 1
    else:
        prints(f'UCCS subspace')
        # CCS like
        ### reference
        excitation_list.append([])
        ### singles ###
        ## alpha ##
        for i in range(noa):
            for a in range(nva):
                x = [2*(a+noa), 2*i]
                excitation_list.append(x)
                nexcite += 1
        ## beta ##
        for j in range(nob):
            for b in range(nvb):
                x = [2*(b+nob)+1, 2*j+1]
                excitation_list.append(x)
                nexcite += 1
    prints(f"Number of operators in pool: {nexcite}") 
    t1 = time.time()

    #print(excitation_list)

    #################################
    ###  Prepare matrix elements  ###
    ###   H[i,j] = <Ei! H  Ej>    ###
    ###   S[i,j] = <Ei!    Ej>    ###
    #################################
    #print(excitation_list)
    hc = False
    Hmat, Smat = get_HS4CIS(Quket.operators.qubit_Hamiltonian, excitation_list, Quket.state, Quket.operators.Hamiltonian, hc=hc)
    printmat(Smat, name="Smat")
    printmat(Hmat, name="Hmat")

    root_invS = root_inv(Smat)
    rank = root_invS.shape[1]

    H_ortho = root_invS.T @ Hmat @ root_invS
    e, U  = sp.linalg.eigh(H_ortho)
    prints("E(CIS):",e)
    prints(f"grand state energy from CIS: {e[0]}")
    prints(f"1st excited state energy from CIS: {e[1]}")
    prints(f"2nd excited state energy from CIS: {e[2]}")

    # エネルギーの低い順に並び替え
    # ind = np.argsort(e.real, -1)
    # e = en.real[ind]
    # U = U[:, ind]

    # Retrieve CI coefficients in the original basis 
    c = root_invS @ U

    """
    # Track the reference-dominant CI vector
    i = -1
    c0 = 0
    while abs(c0) < 0.1:
        i += 1
        c0 = (Smat[0] @ c[:,i]).real

    En = e[i] - Quket.energy
    # Davidson correction
    EQ = (1 - c0**2) * En
    prints('Reference energy = ',Quket.energy)
    prints('CIS correlation energy = ', En)
    prints('CIS total energy = ', Quket.energy + En)
    prints('CIS+Q total energy = ', Quket.energy + En + EQ)
    """
    
    t2 = time.time()
    cput = t2 - t0
    prints("\n Done: CPU Time =  ", "%15.4f" % cput)


def get_HS4CIS(qubit_H, E, state, H, mapping='jordan_wigner', hc=False):
    """Function
    Given excitation list for anti-hermitized operators E[i],
    perform 
           E[j]!  H  E[i]   for all i,j (nexcite)
           E[j]!     E[i]   for all i,j (nexcite)
    in the Pauli operator basis, get expectation values, store
    them as CI Hamiltonian matrix and overlap matrix.

    Args:
        H : Hamiltonian. Either FermionOperator or QubitOperator
        E (list): [p,q,...,r,s] of anti-hermitized operators p^ q^ ... r s 
        state (QuantumState): reference state

    Returns:
        Hmat ([nexcite, nexcite]): CI matrix
        Smat ([nexcite, nexcite]): Overlap matrix
    """
    n_qubits = state.get_qubit_count()
    size = len(E) 
    # H and S
    sizeT = size*(size+1)//2
    ipos, my_ndim = mpi.myrange(sizeT)
    Hmat = np.zeros((size, size), dtype=float, order='F')
    my_Hmat = np.zeros((size, size), dtype=float, order='F')
    Smat = np.zeros((size, size), dtype=float, order='F')
    my_Smat = np.zeros((size, size), dtype=float, order='F')
    H_obs = OpenFermionOperator2QulacsGeneralOperator(qubit_H, n_qubits, mapping=mapping)

    debug_H0 = np.zeros((size, size), dtype=float, order='F')
    debug_H1 = np.zeros((size, size), dtype=float, order='F')
    debug_H2 = np.zeros((size, size), dtype=float, order='F')

    H0,H1,H2 = Separate_Fermionic_Hamiltonian(H, state)
    #prints("H0(test):",H0)
    #prints("H1(test):",H1)
    #prints("H2(test):",H2)
    if mapping in ("jw", "jordan_wigner"):
        qubit_H0 = jordan_wigner(H0)
    elif mapping in ("bk", "bravyi_kitaev"):
        qubit_H0 = bravyi_kitaev(H0, n_qubits)
    H0_obs = OpenFermionOperator2QulacsGeneralOperator(qubit_H0, n_qubits, mapping=mapping)

    if mapping in ("jw", "jordan_wigner"):
        qubit_H1 = jordan_wigner(H1)
    elif mapping in ("bk", "bravyi_kitaev"):
        qubit_H1 = bravyi_kitaev(H1, n_qubits)
    H1_obs = OpenFermionOperator2QulacsGeneralOperator(qubit_H1, n_qubits, mapping=mapping)

    if mapping in ("jw", "jordan_wigner"):
        qubit_H2 = jordan_wigner(H2)
    elif mapping in ("bk", "bravyi_kitaev"):
        qubit_H2 = bravyi_kitaev(H2, n_qubits)
    H2_obs = OpenFermionOperator2QulacsGeneralOperator(qubit_H2, n_qubits, mapping=mapping)

    #prints("H type:",type(H_obs),type(H0_obs))

    ij = 0
    for i in range(size):
        #print("i=",i)
        tau_i = get_tau(E[i], mapping=mapping, hc=hc, n_qubits=n_qubits)
        state_i = evolve(tau_i, state, mapping=mapping)
        for j in range(i+1):
            if ij % mpi.nprocs == mpi.rank:
                # prints(f'{ij}/{sizeT}')
                #print("j=",j)
                tau_j = get_tau(E[j], mapping=mapping, hc=hc, n_qubits=n_qubits)
                state_j = evolve(tau_j, state, mapping=mapping)
                my_Hmat[i,j] = H_obs.get_transition_amplitude(state_i, state_j).real
                my_Hmat[j,i] = my_Hmat[i,j]
                my_Smat[i,j] = inner_product(state_i, state_j).real
                my_Smat[j,i] = my_Smat[i,j]

                debug_H0[i,j] = H0_obs.get_transition_amplitude(state_i, state_j).real
                debug_H0[j,i] = debug_H0[i,j]

                debug_H1[i,j] = H1_obs.get_transition_amplitude(state_i, state_j).real
                debug_H1[j,i] = debug_H1[i,j]

                debug_H2[i,j] = H2_obs.get_transition_amplitude(state_i, state_j).real
                debug_H2[j,i] = debug_H2[i,j]

                #print("i,j,E[i],E[j]:",i,j,E[i],E[j])
            ij += 1
    mpi.comm.Allreduce(my_Hmat, Hmat, mpi.MPI.SUM)
    mpi.comm.Allreduce(my_Smat, Smat, mpi.MPI.SUM)

    if cf.debug:
        printmat(debug_H0, name="H0(cis)")
        printmat(debug_H1, name="H1(cis)")
        printmat(debug_H2, name="H2(cis)")
        printmat(debug_H0+debug_H1+debug_H2, name="H0+H1+H2(cis)")
        printmat(debug_H0+debug_H1+debug_H2 - Hmat, name='H - Hmat(cis)')

    return Hmat, Smat

def create_HS4QSE(H, E, Quket):
    """Function
    Given excitation list for operators E[i],
    calculate the matrix elements H (CI matrix) and S (Overlap matrix) by using 1,2,3,4RDM.

    Returns:
        Hmat ([nexcite, nexcite]): CI matrix
        Smat ([nexcite, nexcite]): Overlap matrix
    """
    size = len(E) 
    n_qubits = Quket.state.get_qubit_count()
    mapping = Quket.cf.mapping
    E_nuc = Quket.nuclear_repulsion
    norbs = Quket.n_active_orbitals
    D1 = get_1RDM_full(Quket.state, mapping=mapping)
    D2 = get_2RDM_full(Quket.state, mapping=mapping)
    D3 = get_3RDM_full(Quket.state, mapping=mapping)
    D4 = get_4RDM_full(Quket.state, mapping=mapping)

    ncore = Quket.n_frozen_orbitals
    h_pr = Quket.one_body_integrals_active
    h_pqrs = Quket.two_body_integrals_active
    #h_pqrs = 0.5*(h_pqrs - h_pqrs.transpose(0,1,3,2))
    #print(f"h_pr shape:{h_pr.shape}")
    #print(f"h_pqrs shape:{h_pqrs.shape}")

    nvirorbs = Quket.n_secondary_orbitals
    n_all_orbitals = norbs + nvirorbs
    prints(f"n_virtual orbitals:{nvirorbs} n_all_orbitals:{n_all_orbitals}")
    h_pr_2 = Quket.one_body_integrals

    if cf.debug:
        printmat(D1,name="test")
        printmat(h_pr,name="h_pr test")
        printmat(h_pr_2,name="h_pr_2 test")

    S1 = np.zeros((size, size), dtype=float)

    H0 = np.zeros((size, size), dtype=float)
    H1 = np.zeros((size, size), dtype=float)
    H2 = np.zeros((size, size), dtype=float)

    #print(E)
    #print("matrix size :",size)
    ### create S (overlap matrix)
    for i in range(size):
        index_fir = E[i]
        #print("i,E[i]:",i,E[i])
        for j in range(i+1):
            #print("i,j :",i,j)
            index_sec = E[j]
            index_i = index_fir[0]
            index_j = index_fir[1]
            index_k = index_sec[0]
            index_l = index_sec[1]
            #print(f"S1 index: {index_i, index_j, index_k, index_l} i,j: {i, j}")
            if index_i == index_k :
                val = D1[index_j,index_l] - D2[index_j,index_k,index_i,index_l]
            else:
                val = - D2[index_j,index_k,index_i,index_l]
            # #prints(val)
            # if i == j : #debug
            #     prints("i,j:",i,j)
            #     prints("index:",index_i, index_j, index_k, index_l)
            #     prints("val:",val)
            if (index_k == 0 and index_l == 0) :
                S1[i,j] = D1[index_j,index_i]
            else:
                S1[i,j] = val
            S1[j,i] = S1[i,j]
    S1[0,0] = 1.0

    # item 1 #
    # H0 = E_nuc < GS| j^ i k^ l | GS>
    # item 2 #
    # H1 = \sum_{pr} h_pr * F^{ij}_{kl}
    # print(h_pr.shape)
    # print(h_pqrs.shape)

    sum = 0
    ### create H0, H1 (CI matrix, item1, item2)
    for p in range(n_qubits):
        for r in range(n_qubits):
            for i in range(size):
                index_fir = E[i]
                for j in range(i+1):
                    index_sec = E[j]
                    index_i = index_fir[0]
                    index_j = index_fir[1]
                    index_k = index_sec[0]
                    index_l = index_sec[1]
                    # print(f"H0, H1 index:{index_i, index_j, index_k, index_l} i,j:{i, j}")
                    # excite -- grand state -> i,j 
                    if ( index_k == 0 and index_l == 0):
                        # equation(43)
                        if ( i==0 & j==0 ):
                            val = h_pr[p, r] * D1[p, r]
                            H1[i, j] += val
                        elif ( index_i == p ): 
                            val = h_pr[p,r] * (D1[index_j, r] - D2[index_j, p, index_i, r])
                            H1[i, j] += val
                            #print(f"index_i//2, p, i, j:{index_i//2, p, i, j} val:{val}")
                        else:
                            H1[i, j] += h_pr[p,r] * (-D2[index_j, p, index_i, r])
                    else:
                        # linear response -- i,j -> k,l
                        # equation(44)
                        H1[i, j] += h_pr[p,r] * (-D3[index_j, index_k, p, index_i, index_l, r])
                        if ( index_i == index_k ):
                            H1[i, j] += h_pr[p,r] * (-D2[index_j, p, index_l, r])
                        if ( index_i == p ):
                            H1[i, j] += h_pr[p,r] * (D2[index_j, index_k, index_l, r])
                        if ( index_k == r ):
                            H1[i, j] += h_pr[p,r] * (-D2[index_j, p, index_i, index_l])
                        if ( index_i == p and index_k == r ):
                            H1[i, j] += h_pr[p,r] * (D1[index_j, index_l])
                    # calculate H0
                    if ( index_j == index_l ):
                        H0[i, j] =  -D2[index_j,index_k,index_i,index_l] + D1[index_j,index_l]
                    else:
                        H0[i, j] =  -D2[index_j,index_k,index_i,index_l]
                    #
                    H0[j, i] = H0[i, j]
                    H1[j, i] = H1[i, j]
    H0 = S1 * E_nuc

    # item 3 #
    # H2 = \sum_{pqrs} h_pqrs * V^{ij}_{kl}

    ### create H2 (CI matrix, item3)
    sum = 0
    for p in range(n_qubits):
        for q in range(n_qubits):
            for s in range(n_qubits):
                for r in range(n_qubits):
                    for i in range(size):
                        index_fir = E[i]
                        for j in range(i+1):
                            index_sec = E[j]
                            index_i = index_fir[0]
                            index_j = index_fir[1]
                            index_k = index_sec[0]
                            index_l = index_sec[1]
                            # excite -- grand state -> i,j 
                            if ( index_k == 0 and index_l == 0):
                                # equation(45)
                                val = h_pqrs[p,q,r,s] * D3[index_j, p, q, index_i, r, s]
                                if ( i == 0 & j == 0):
                                    H2[i, j] += h_pqrs[p,q,r,s] * D2[p,q,r,s]
                                elif ( index_i == p ):
                                    H2[i, j] += h_pqrs[p,q,r,s] * D2[index_j, q, r, s] + val
                                elif ( index_i == q ):
                                    H2[i, j] += h_pqrs[p,q,r,s] * -D2[index_j, p, r, s] + val
                                else:
                                    H2[i, j] += val
                            else:
                                # linear response -- i,j -> k,l
                                # equation(46)
                                val = h_pqrs[p,q,r,s] * (-D4[index_j, index_k, p, q, index_i, index_l, r, s])
                                H2[i, j] += val
                                if ( index_i == index_k ):
                                    H2[i, j] += h_pqrs[p,q,r,s] * (D3[index_j, p, q, index_l, r, s])
                                if ( index_i == p and index_k  == r ):
                                    H2[i, j] += h_pqrs[p,q,r,s] * D2[index_j, q, index_l, s]
                                if ( index_i == p and index_k  == s ):
                                    H2[i, j] += h_pqrs[p,q,r,s] * -D2[index_j, q, index_l, r]
                                if ( index_i == p ):
                                    H2[i, j] += h_pqrs[p,q,r,s] * -D3[index_j, index_k, q, index_l, r, s]
                                if ( index_i == q and index_k == r ):
                                    H2[i, j] += h_pqrs[p,q,r,s] * -D2[index_j, p, index_l, s]
                                if ( index_i == q and index_k == s ):
                                    H2[i, j] += h_pqrs[p,q,r,s] * D2[index_j, p, index_l, r]
                                if ( index_i == q ):
                                    H2[i, j] += h_pqrs[p,q,r,s] * D3[index_j, index_k, p, index_l, r, s]
                                if ( index_k == r ):
                                    H2[i, j] += h_pqrs[p,q,r,s] * D3[index_j, p, q, index_i, index_l, s]
                                if ( index_k == s ):
                                    H2[i, j] += h_pqrs[p,q,r,s] * -D3[index_j, p, q, index_i, index_l, r]
                            H2[j, i] = H2[i, j]
    # H2 = 1/2 * H2
    H = H0 + H1 + H2
    if cf.debug:
        printmat(S1,name='S(qse)')
        printmat(H0,name='H0(qse)')
        printmat(H1,name='H1(qse)')
        printmat(H2,name='H2(qse)')
        printmat(H,name="H0+H1+H2(qse)")
    return H, S1

def create_HS_VirtualQSE(H, E, Quket):
    size = len(E) 
    n_qubits = Quket.state.get_qubit_count()
    E_nuc = Quket.nuclear_repulsion
    norbs = Quket.n_active_orbitals
    mapping = Quket.cf.mapping
    D1 = get_1RDM_full(Quket.state, mapping=mapping)
    D2 = get_2RDM_full(Quket.state, mapping=mapping)
    D3 = get_3RDM_full(Quket.state, mapping=mapping)
    D4 = get_4RDM_full(Quket.state, mapping=mapping)

    ncore = Quket.n_frozen_orbitals
    h_pr = Quket.one_body_integrals_active
    h_pqrs = Quket.two_body_integrals_active

def reorder_rdm1(Quket):
    "dm[p,q] = < p^ + q >  -> dm[p,q] = < q^ + p >"
    Da = Quket.DA
    Db = Quket.DB
    rdm1a = Da.copy()
    rdm1b = Db.copy()

    norbs = Quket.n_active_orbitals
    rdm_a = np.zeros((norbs,norbs))
    rdm_b = np.zeros((norbs,norbs))

    delta = np.eye(norbs)
    rdm1a = delta - Da
    rdm1b = delta - Db
    printmat(rdm1a, name="reordered Da")
    printmat(rdm1b, name="reordered Db")
    return rdm1a, rdm1b
