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

ct.py

Main driver of Canonical mapping.

"""
import numpy as np
import scipy as sp
import time
import copy
import math

from qulacs import QuantumCircuit
from qulacs.state import inner_product
from qulacs.observable import create_observable_from_openfermion_text
from openfermion.utils import count_qubits

from quket.mpilib import mpilib as mpi
from quket import config as cf

from .prop import prop 
from quket.fileio import prints, print_state, print_amplitudes_adapt, error, SaveAdapt, LoadAdapt, printmat, printmath
from quket.utils import Gdoubles_list
from quket.linalg import root_inv, nullspace
from quket.lib import commutator, hermitian_conjugated, get_fermion_operator,jordan_wigner,bravyi_kitaev, normal_ordered
from .rdm import get_Generalized_Fock_Matrix_one_body
from .cumulant import *
from quket.utils.utils import get_tau

### Control the threshold for removing redundancy (zero singular values)

def ct_solver(Quket, print_level):
    """Function:
    Main driver for Canonical mapping for post-VQE
    
    Args:
        Quket (QuketData): Quket data
        print_level (int): Printing level
        maxiter (int): Maximum number of iterations


    Author(s): Takashi Tsuchimochi 
    """
    from quket.projection import S2Proj
    prints('Entered CT solver')

    t0 = time.time()
    cf.t_old = t0
    maxiter = Quket.maxiter
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    method = Quket.post_method

    ##number of operator is going to dynamically increase
    theta_list = []  ## initialize

    prints("Performing ", Quket.post_method, end="")
    if Quket.projection.SpinProj or Quket.projection.post_SpinProj:
        prints("(spin-projection) ", end="")
    print_state(Quket.state, name='Reference state')
    if Quket.projection.post_SpinProj:
        Quket.state = S2Proj(Quket, Quket.state)
        print_state(Quket.state, name='Reference projected state')
    Quket.energy = Quket.qulacs.Hamiltonian.get_expectation_value(Quket.state)
    prints(f'E[Reference] = {Quket.energy}')
    spinfree = Quket.spinfree
    Quket.get_1RDM()
    Quket.get_2RDM()
    w2 = Quket.regularization
    if method == "luccd":
        do_singles = False
    else:
        do_singles = True
    prints(do_singles)
    nexcite = 0
    nsing = 0
    ndoub = 0
    excitation_list = []
    if Quket.post_general:
        prints(f'UCCGSD subspace')
        r_list, u_list, parity_list = Gdoubles_list(Quket.n_active_orbitals)
        # CCGSD like
        if spinfree:
            # Spin-Free
            excitation_list.append([])
            if do_singles:
                for p in range(Quket.n_active_orbitals):
                    for q in range(p):
                        excitation_list.append([[2*p, 2*q], [2*p+1, 2*q+1]])
                        nexcite += 1
            for k, u in enumerate(u_list):
                if len(u) == 6:
                    ### two possibilities
                    u1 = [u[0], u[1], u[2], u[3]]
                    u2 = [u[0], u[1], u[4], u[5]]
                    excitation_list.append(u1)
                    excitation_list.append(u2)
                    nexcite += 2
                elif len(u) == 2:
                    excitation_list.append(u)
                    nexcite += 1
                elif len(u) == 1:
                    excitation_list.append(u)
                    nexcite += 1

        else:
            # Spin-Dependent #
            ### reference
            excitation_list.append([])
            if do_singles:
                ### singles
                for p in range(Quket.n_active_orbitals):
                    for q in range(p):
                        x = [2*p, 2*q]
                        excitation_list.append(x)
                        nexcite += 1
                        x = [2*p+1, 2*q+1]
                        excitation_list.append(x)
                        nexcite += 1
                        nsing += 2
            ### doubles
            for u in u_list:
                for x in u:
                    excitation_list.append(x)
                    nexcite += 1
                    ndoub += 1
    else:
        prints(f'UCCSD subspace')
        # CCSD like
        if spinfree:
            excitation_list.append([])
            for i in range(noa):
                for a in range(nva):
                    excitation_list.append([[2*(a+noa), 2*i], [2*(a+noa)+1, 2*i+1]])
                    nexcite += 1
            ai = 0
            for i in range(noa):
                for a in range(nva):
                    ai += 1
                    bj = 0
                    for j in range(noa):
                        for b in range(nva):
                            bj += 1
                            if bj <= ai: 
                                print(a+noa,i,b+noa,j)
                                excitation_list.append(get_sf_list([a+noa,i,b+noa,j]))
                                nexcite += 1
            
        else:
            ### reference
            excitation_list.append([])
            # singles
            for i in range(noa):
                for a in range(nva):
                    x = [2*(a+noa), 2*i]
                    excitation_list.append(x)
                    nexcite += 1
                    nsing += 1
            for i in range(nob):
                for a in range(nvb):
                    x = [2*(a+nob)+1, 2*i+1]
                    excitation_list.append(x)
                    nexcite += 1
                    nsing += 1
            # doubles
            for i in range(noa):
                for j in range(i):
                    for a in range(Quket.nva):
                        for b in range(a):
                            x = [2*(a+noa), 2*(b+noa), 2*i, 2*j]
                            excitation_list.append(x)
                            nexcite += 1
                            ndoub += 1
            for i in range(nob):
                for j in range(noa):
                    for a in range(nvb):
                        for b in range(nva):
                            x = [2*(a+nob)+1, 2*(b+noa), 2*i+1, 2*j]
                            excitation_list.append(x)
                            nexcite += 1
                            ndoub += 1
            for i in range(nob):
                for j in range(i):
                    for a in range(nvb):
                        for b in range(a):
                            x = [2*(a+nob)+1, 2*(b+nob)+1, 2*i+1, 2*j+1]
                            excitation_list.append(x)
                            nexcite += 1
                            ndoub += 1
    prints(f"Number of operators in pool: {nexcite}") 
    if cf.debug:
        for k, p in enumerate(excitation_list):
            prints(f'{k:4d}  {p}')
    t1 = time.time()

    t_amp = np.zeros(nexcite, dtype=float)
    A_amp = np.zeros(nexcite, dtype=float)
    H0 = get_fermion_operator(Quket.operators.Hamiltonian)
    n_frozen_orbitals = Quket.n_frozen_orbitals
    DA = Quket.DA[n_frozen_orbitals:, n_frozen_orbitals:]
    DB = Quket.DB[n_frozen_orbitals:, n_frozen_orbitals:]
    D2AA = Quket.Daaaa
    D2BB = Quket.Dbbbb
    D2BA = Quket.Dbaab

    ### Generate D from DA and DB
    norbs = DA.shape[0]
    D1 = np.zeros((2*norbs, 2*norbs), dtype=float)
    D2 = np.zeros((2*norbs, 2*norbs, 2*norbs, 2*norbs), dtype=float)
    for i in range(norbs):
        for j in range(norbs):
            D1[2*i, 2*j] = DA[i,j]
            D1[2*i+1, 2*j+1] = DB[i,j]
            for k in range(norbs):
                for l in range(norbs):
                    D2[2*i, 2*j, 2*k, 2*l] = D2AA[i,j,k,l]
                    D2[2*i+1, 2*j+1, 2*k+1, 2*l+1] = D2BB[i,j,k,l]
                    #D2[2*i+1, 2*j, 2*k, 2*l+1] = D2BA[i,j,k,l]
                    #D2[2*i+1, 2*j, 2*k+1, 2*l] = -D2BA[i,j,l,k]
                    #D2[2*i, 2*j+1, 2*k, 2*l+1] = -D2BA[j,i,k,l]
                    #D2[2*i, 2*j+1, 2*k+1, 2*l] = D2BA[j,i,l,k]
                    D2[2*i+1, 2*j, 2*k, 2*l+1] = D2BA[i,j,k,l]
                    D2[2*j, 2*i+1, 2*k, 2*l+1] = -D2BA[i,j,k,l]
                    D2[2*i+1, 2*j, 2*l+1, 2*k] = -D2BA[i,j,k,l]
                    D2[2*j, 2*i+1, 2*l+1, 2*k] = D2BA[i,j,k,l]
    cum_list = store_Decomposed_3body(D1, D2)
    D3 = cumulant_3RDM(D1, D2)

    #h2 = Quket.two_body_integrals[n_frozen_orbitals:,n_frozen_orbitals:,n_frozen_orbitals:,n_frozen_orbitals:]
    #energy = 0
    #for i in range(norbs):
    #    for j in range(norbs):
    #        energy += h1[i,j] * D[2*i, 2*j]
    #        D[2*i, 2*j] = DA[i,j]
    #        D[2*i+1, 2*j+1] = DB[i,j]
    #        for k in range(norbs):
    #            for l in range(norbs):
    #                D2[2*i, 2*j, 2*k, 2*l] = D2AA[i,j,k,l]
    #                D2[2*i+1, 2*j+1, 2*k+1, 2*l+1] = D2BB[i,j,k,l]

    #t_amp = np.random.rand(nexcite) * 0.1
    Hbar = FormHbar(H0, t_amp, excitation_list, D1, D2, cum_list=cum_list)
    if Quket.cf.mapping == "jordan_wigner":
        qubit_Hbar = jordan_wigner(Hbar)
    elif Quket.cf.mapping == "bravyi_kitaev":
        qubit_Hbar = bravyi_kitaev(Hbar, Quket.n_qubits)
    qubit_Hbar_obs = create_observable_from_openfermion_text(str(qubit_Hbar))
    Ebar = qubit_Hbar_obs.get_expectation_value(Quket.state) 
    Eold = 0
    Ediff = Eold - Ebar
    for icyc in range(10):
        #######################################################
        ### Prepare matrix elements                         ###
        #######################################################
        Dmat, Vvec, Smat  = get_AV_12(Hbar, excitation_list, Quket.state, "lucc", Quket, D1, D2, cum_list)
        t1 = time.time()
        cput = t1 - t0

        prints("\n Matrices ready.   ", "%15.4f" % cput)
        if cf.debug:
            printmath(Dmat, name='Dmat')
            printmath(Smat, name='Smat')
            printmath(Vvec, name='Vvec')
        #    for i in range(Hmat.shape[0]):
        #        for j in range(i+1):
        #            if abs(Hmat[i,j]) > 1e-8: 
        #                prints('H: ', Hmat[i,j], ' : ', excitation_list[i], '   ', excitation_list[j])
        #    if Xmat is not None:
        #        for i in range(Xmat.shape[0]):
        #            for j in range(Xmat.shape[0]):
        #                if abs(Xmat[i,j]) > 1e-8: 
        #                    prints('X: ', Xmat[i,j], ' : ', excitation_list[i], '   ', excitation_list[j])
            for i in range(len(Vvec)):
                if abs(Vvec[i]) > 1e-8: 
                    prints('V ', Vvec[i], ' : ', excitation_list[i])
        prints(f'Norm[V] = {np.linalg.norm(Vvec)}')
        
        hessian = True
        ### Solve A.t = - b by SVD...
        ### However, this does not remove the redundancy completely!
        ### although the energy is quite OK...
        Dmat = Dmat[1:,1:]
        if hessian:
            prints('hessian')
            Dmat = (Dmat + Dmat.T)/2
        Vvec = Vvec[1:]
        u, s, vh = sp.linalg.svd(Dmat)
        ## Moore-Penrose 
        #s_inv = np.zeros((nexcite, nexcite), dtype=float)
        #rank = 0
        #for i in range(nexcite):
        #    if abs(s[i]) > eps:
        #        s_inv[i,i] = 1/s[i]
        #        rank += 1
        #    else:
        #        s_inv[i,i] = 0
        #prints('rank = ',rank)

        ## A.x = u.s.vh.x  = -b
        ##s.vh.x = - u!.b
        ##t_amp = - root_invS @ vh.T @ s_inv @ u.T @ V_ortho
        #t_amp = - vh.T @ s_inv @ u.T @ Vvec
        #Ecorr= (t_amp@Vvec).real
        #EMoore = Quket.energy + Ecorr
        #prints('norm = ',sp.linalg.norm(Amat@t_amp + Vvec))
        
        ### Imaginary level-shift
        s_inv = np.zeros((nexcite, nexcite), dtype=float)
        if cf.debug:
            prints('singular values\n',s)
        for i in range(nexcite):
            s_inv[i,i] = s[i]/(s[i]**2 + w2)

        # A.x = u.s.vh.x  = -b
        #s.vh.x = - u!.b
        t_amp = - vh.T @ s_inv @ u.T @ Vvec
        Ecorr = (t_amp@Vvec).real
        Ereg = Quket.energy + Ecorr
        ### Hylleraas functional
        Ecorr = (t_amp @ Dmat @ t_amp + 2 * t_amp @ Vvec).real
        EHyl = Quket.energy + Ecorr
        imax = np.argmax(abs(t_amp))
        contribution = 0 
        imin = 0
        for i in range(nexcite):
            test = t_amp[i] * Vvec[i]
            if test < contribution:
                imin = i
                contribution = test
        #prints(f'Largest t-amplitude: {t_amp[imax]}')
        #prints(f'Local correlation energy: {t_amp[imax]*Vvec[imax]}')
        #prints(f'Excitation: {excitation_list[imax+1]}')
        #prints(f'\n')
        prints(f'Most contributing t-amplitue: {t_amp[imin]}')
        prints(f'Local correlation energy: {contribution}')
        prints(f'Excitation: {excitation_list[imin+1]}')
        if cf.debug: 
            X = np.block([[Vvec],[t_amp]])
            printmat(X,name='Comparison of V (0) and t (1)')

        ### Orthonormalize the basis of Dmat
        ### This can completely remove the redundancy
        ### but requires 4RDM...
        Smat = Smat[1:,1:]
        Null, Range = nullspace(Smat,eps=w2)
        rank = Range.shape[1]
        A_ortho = Range.T@Dmat@Range
        V_ortho = Range.T@Vvec
        u, s, vh = sp.linalg.svd(A_ortho)
        s_inv = np.zeros((rank, rank), dtype=float)
        if cf.debug:
            prints('singular values\n',s)
        for i in range(rank):
            if s[i] > w2:
                s_inv[i,i] = 1/(s[i])
            else:
                s_inv[i,i] = 0
            #s_inv[i,i] = s[i]/(s[i]**2 + w2)
        t_amp = -Range @ vh.T @ s_inv @ u.T @ V_ortho
        Ecorr = (t_amp @ Dmat @ t_amp + 2 * t_amp @ Vvec).real
        Eortho = Quket.energy + Ecorr
        prints('Using U: Energy is ',Eortho)


        ## Solve instead A.S.S+.t + v = 0 
        ## Get S+
        #s, U  = sp.linalg.eigh(Smat)
        #s_inv = np.zeros((nexcite, nexcite), dtype=float)
        #if cf.debug:
        #    prints('singular values\n',s)
        #for i in range(nexcite):
        #    s_inv[i,i] = s[i]/(s[i]**2 + w2)
        #### Approximate A.S.S+ 
        #ASS = Dmat @ Smat @ U @ s_inv @ U.T
        #u, s, vh = sp.linalg.svd(ASS)
        #s_inv = np.zeros((nexcite, nexcite), dtype=float)
        #if cf.debug:
        #    prints('singular values\n',s)
        #for i in range(nexcite):
        #    s_inv[i,i] = s[i]/(s[i]**2 + w2)
        #t_amp = - vh.T @ s_inv @ u.T @ Vvec
        #Ecorr = (t_amp @ Dmat @ t_amp + 2 * t_amp @ Vvec).real
        #Eortho = Quket.energy + Ecorr


        ### Update Canonical Transformed Hbar ###
        A_amp += t_amp
        Hbar = FormHbar(H0, A_amp, excitation_list, D1, D2, cum_list=cum_list)
        if Quket.cf.mapping == "jordan_wigner":
            qubit_Hbar = jordan_wigner(Hbar)
        elif Quket.cf.mapping == "bravyi_kitaev":
            qubit_Hbar = bravyi_kitaev(Hbar, Quket.n_qubits)
        qubit_Hbar_obs = create_observable_from_openfermion_text(str(qubit_Hbar))
        Ebar = qubit_Hbar_obs.get_expectation_value(Quket.state) 
        Ediff = Eold - Ebar
        if abs(Ediff) < 1e-8:
            break

        Eold = Ebar
        if icyc == 0:
            prints(f'Reference energy = {Quket.energy:.12f}')
            prints(f'{method} total energy (regularized)   = {Ereg:.12f}    (Ecorr = {Ereg - Quket.energy:.12f})')
            prints(f'{method} total energy (Hylleraas)     = {EHyl:.12f}    (Ecorr = {EHyl - Quket.energy:.12f})')
            prints(f'{method} total energy (Orthogonal)    = {Eortho:.12f}    (Ecorr = {Eortho - Quket.energy:.12f})')
        prints(f'CT energy = {Ebar:.12f}  (Ecorr = {Ebar - Quket.energy:.12f})')
    prints(f'Final CT energy = {Ebar:.12f}  (Ecorr = {Ebar - Quket.energy:.12f})')


        

    t2 = time.time()
    cput = t2 - t0

    prints("\n Done: CPU Time =  ", "%15.4f" % cput)

def get_AV_12(H, E, state, method, Quket, D1, D2, cum_list):
    """Function
    Given excitation list for operators E[i],
    evaluate the matrix elements A[i,j] and V[i], and S[i,j] if necessary.
    For unitary methods, we use U[i] = E[i] - E![i] instead.
    A[i,j] is different for different methods.

    lucc               A[i,j] = 1/2 <[[H, Uj], Ui]> 
    cepa               A[i,j] = <Ej! [(H-E0), Ei]> 
    cisd               A[i,j] = <Ej! (H-E0) Ei> 
    ucisd              A[i,j] = <Uj! (H-E0) Ui>

                   V[i] = 1/2 <[H, Ui]> = <Ei! H> 
                    
    in the Pauli operator basis, get expectation values, store
    them in A and V, respectively.

    Args:
        H : Hamiltonian. Either FermionOperator or QubitOperator
        E (list): [p,q,...,r,s] of anti-hermitized operators p^ q^ ... r s 
        state (QuantumState): reference state

    Returns:
        Hmat ([nexcite, nexcite]): Hamiltonian matrix
        Vvec ([nexcite]): V vector
        Smat ([nexcite, nexcite]): Overlap matrix
        Xmat ([nexcite, nexcite]): De-excitation effect (<Uj!Ui H>) for lucc
    """
    if method not in ["lucc",  "cepa", "cepa0", "cisd", "ucisd", "luccsd", "luccd"]:
        raise ValueError('method has to be set in get_AV.')
    if method in ["lucc", "ucisd", "ucepa0", "luccsd", "luccd"]:
        hc = True
    else:
        hc = False
    n_qubits = state.get_qubit_count()
    mapping = Quket.cf.mapping
    size = len(E)

    if '2' in method:
        ## Form generalized fock. 
        fa, fb = get_Generalized_Fock_Matrix_one_body(Quket)
        Fock = create_1body_operator(fa, XB=fb) 
        if mapping == ("jw", "jordan_wigner"):
            Fock = jordan_wigner(Fock)
        elif mapping in ("bk", "bravyi_kitaev"):
            Fock = bravyi_kitaev(Fock, n_qubits)
        Fpsi = evolve(Fock, state) 

    # V
    Vvec = np.zeros(size, dtype=float)
    my_Vvec = np.zeros(size, dtype=float)
    H_obs = OpenFermionOperator2QulacsGeneralOperator(H, n_qubits, mapping=mapping)
    E0 = H_obs.get_expectation_value(state)
    Hpsi = evolve(H, state)
    for i in range(1,size):
        if i % mpi.nprocs == mpi.rank:
            t0 = time.time() 
            tau_i = get_tau(E[i], mapping=None, hc=hc)
            t1 = time.time() 
            HEi = normal_ordered(commutator(H, tau_i))
            t2 = time.time() 
            #HEi = Decompose_3body_CU(HEi, D1)
            HEi = Decompose_3body_MK(HEi, D1, D2, cum_list=cum_list)
            t3 = time.time() 
            HEpsi = evolve(HEi, state)
            t4 = time.time() 
            my_Vvec[i] = inner_product(HEpsi, state).real 
            t5 = time.time() 
            prints('tau : ',t1-t0)
            prints('comm: ',t2-t1)
            prints('cum : ',t3-t2)
            prints('evol: ',t4-t3)
            prints('inne: ',t5-t4)
            prints('')

    Vvec = mpi.allreduce(my_Vvec, mpi.MPI.SUM)


    ## H and S
    Hmat = np.zeros((size, size), dtype=float)
    Smat = np.zeros((size, size), dtype=float)
    my_Hmat = np.zeros((size, size), dtype=float)
    my_Smat = np.zeros((size, size), dtype=float)
    sizeT = size * (size+1)//2
    ij = 0
    E0 = H_obs.get_expectation_value(state).real
    for i in range(1,size):
        tau_i = get_tau(E[i], mapping=None, hc=hc)
        state_i = evolve(tau_i, state)
        HEi = normal_ordered(commutator(H, tau_i))
        # HEi = Decompose_3body_CU(HEi, D1)
        HEi = Decompose_3body_MK(HEi, D1, D2, cum_list=cum_list)

        for j in range(1,size):
            if ij % mpi.nprocs == mpi.rank:
                prints(f'{ij}/{size*size}')
                t0 = time.time()
                tau_j = get_tau(E[j], mapping=None, hc=hc)
                t1 = time.time()
                HEiEj = normal_ordered(commutator(HEi, tau_j))
                t2 = time.time()
                #HEiEj = Decompose_3body_CU(HEiEj, D1)
                #prints('Ei = \n', tau_i)
                #prints('Ej = \n', tau_j)
                #prints('HEiEj = \n', HEiEj)
                HEiEj = Decompose_3body_MK(HEiEj, D1, D2, cum_list=cum_list)
                t3 = time.time()
                state_j = evolve(tau_j, state)
                if HEiEj != 0:
                    state_ij = evolve(HEiEj, state)
                    my_Hmat[i,j] = inner_product(state_ij, state).real
                else:
                    my_Hmat[i,j] = 0

                t4 = time.time()
                my_Smat[i,j] = inner_product(state_i, state_j).real
                t5 = time.time()

                #if HEiEj != 0:
                #    obs = create_observable_from_openfermion_text(str(jordan_wigner(HEiEj)))
                #    x = obs.get_expectation_value(state)
                #t6 = time.time()
                prints('ij tau : ',t1-t0)
                prints('ij comm: ',t2-t1)
                prints('ij cum : ',t3-t2)
                prints('ij evol: ',t4-t3)
                prints('ij inne: ',t5-t4)
                #prints('ij expe: ',t6-t5)
                prints('')
            ij += 1
    Hmat = mpi.allreduce(my_Hmat, mpi.MPI.SUM)
    Smat = mpi.allreduce(my_Smat, mpi.MPI.SUM)

    return Hmat, Vvec, Smat

def FormHbar(H0, Amp, excitation_list, D1, D2, cum_list=None):
    Hbar = copy.deepcopy(H0)
    Hn   = copy.deepcopy(H0)
    nexcite = len(Amp)
    ### Generate A from Amp
    A = FermionOperator('',0)
    printmath(Amp)
    for k in range(nexcite):
        A += Amp[k] * get_tau(excitation_list[k+1], mapping=None, hc=True)
    n = 1
    while norm_Hn(Hn) > 1e-14:
        prints('n = ',n, '  ',norm_Hn(Hn))
        ### get n+1
        HnA = commutator(Hn, A)
        HnA = normal_ordered(HnA)
        #Hn = Decompose_3body_CU(HnA, D1) / n
        mpi.barrier()
        #Hn = Decompose_3body_MK(HnA, D1, D2, cum_list=cum_list, cum_index=cum_index) / n
        Hn = Decompose_3body_MK(HnA, D1, D2) / n
        Hbar += Hn
        n+= 1
    prints(f'convergece of Hbar at n={n}.')
    return normal_ordered(Hbar)


def norm_Hn(Hn):
    if Hn != 0:
        norm = 0
        for op, coef in Hn.terms.items():
            norm += coef**2
    else:
        norm = 0
    return np.sqrt(norm)
