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

lucc.py

Main driver of LUCC and CISD.

"""
import numpy as np
import scipy as sp
import time
import copy
from tqdm import tqdm

from itertools import zip_longest, product, groupby
from math import fsum, fabs
import statistics

from qulacs import QuantumCircuit
from qulacs.state import inner_product
from qulacs.observable import create_observable_from_openfermion_text
from openfermion.utils import count_qubits
from openfermion.ops import InteractionOperator

from quket.mpilib import mpilib as mpi
from quket import config as cf

from quket.fileio import prints, print_state, print_amplitudes_adapt, error, SaveAdapt, LoadAdapt, printmat, printmath
from quket.utils import Gdoubles_list
from quket.utils.utils import get_tau
from quket.linalg import root_inv, nullspace
from quket.opelib import OpenFermionOperator2QulacsGeneralOperator, create_1body_operator
from quket.opelib import evolve
from quket.post.rdm import get_Generalized_Fock_Matrix_one_body
from quket.post.prop import prop
from quket.lib import (
    QubitOperator, FermionOperator, commutator, hermitian_conjugated,
    get_fermion_operator,jordan_wigner,bravyi_kitaev, normal_ordered
    )

### Control the threshold for removing redundancy (zero singular values)

def lucc_solver(Quket, print_level):
    """Function:
    Main driver for LUCC for post-VQE

    Args:
        Quket (QuketData): Quket data
        print_level (int): Printing level
        maxiter (int): Maximum number of iterations


    Author(s): Takashi Tsuchimochi
    """
    from quket.projection import S2Proj
    prints('Entered LUCC solver')

    t0 = time.time()
    cf.t_old = t0
    maxiter = Quket.maxiter
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    method = Quket.post_method
    mapping = Quket.cf.mapping

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
    prints(f'E[Reference] = {Quket.energy}')\

    spinfree = Quket.spinfree
    #Quket.get_1RDM()
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
    symmetry = Quket.tapering.clifford_operators is not None
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
                    if symmetry:
                        ### Check symmetry
                        if chk_sym_tau(Quket.tapering, u1, mapping=mapping):
                            excitation_list.append(u1)
                            excitation_list.append(u2)
                            nexcite += 2
                    else:
                        excitation_list.append(u1)
                        excitation_list.append(u2)
                        nexcite += 2
                elif len(u) == 2:
                    if symmetry:
                        ### Check symmetry
                        if chk_sym_tau(Quket.tapering, u, mapping=mapping):
                            excitation_list.append(u)
                            nexcite += 1
                    else:
                        excitation_list.append(u)
                        nexcite += 1
                elif len(u) == 1:
                    if symmetry:
                        ### Check symmetry
                        if chk_sym_tau(Quket.tapering, u, mapping=mapping):
                            excitation_list.append(u)
                            nexcite += 1
                    else:
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
                for p in range(Quket.n_active_orbitals):
                    for q in range(p):
                        x = [2*p+1, 2*q+1]
                        excitation_list.append(x)
                        nexcite += 1
                        nsing += 2
            ### doubles
            for u in u_list:
                for x in u:
                    if symmetry:
                        ### Check symmetry
                        if chk_sym_tau(Quket.tapering, x, mapping=mapping):
                            excitation_list.append(x)
                            nexcite += 1
                    else:
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


    #######################################################
    ### Prepare matrix elements                         ###
    #######################################################
#    if spinfree:
#        Hmat, Vvec, Smat, Xmat = get_AV_sf(Quket.operators.qubit_Hamiltonian, excitation_list, Quket.state, method, Quket)
#    else:
#        if method in ["lucc2", "luccsd2", "pt2"]:
#            Hmat, Vvec, Smat, Xmat = get_AV_2(Quket.operators.qubit_Hamiltonian, excitation_list, Quket.state, method, Quket, nsing, ndoub)
#        else:
#            Hmat, Vvec, Smat, Xmat = get_AV(Quket.operators.qubit_Hamiltonian, excitation_list, Quket.state, method, Quket)
#    t1 = time.time()
#    cput = t1 - t0
#
#    prints("\n Matrices ready.   ", "%15.4f" % cput)
#    if cf.debug:
#        printmath(Hmat, name='Hmat')
#        printmath(Smat, name='Smat')
#        printmath(Vvec, name='Vvec')
#        if Xmat is not None:
#            printmath(Xmat, name='Xmat')
#        for i in range(Hmat.shape[0]):
#            for j in range(i+1):
#                if abs(Hmat[i,j]) > 1e-8:
#                    prints('H: ', Hmat[i,j], ' : ', excitation_list[i], '   ', excitation_list[j])
#        if Xmat is not None:
#            for i in range(Xmat.shape[0]):
#                for j in range(Xmat.shape[0]):
#                    if abs(Xmat[i,j]) > 1e-8:
#                        prints('X: ', Xmat[i,j], ' : ', excitation_list[i], '   ', excitation_list[j])
#        for i in range(len(Vvec)):
#            if abs(Vvec[i]) > 1e-8:
#                prints('V ', Vvec[i], ' : ', excitation_list[i])
#
#    hessian = False
#    if method in ["lucc", "luccsd", "luccd", "lucc2", "luccsd2", "pt2"]:
#        ### Solve A.t = - b by SVD...
#        ### However, this does not remove the redundancy completely!
#        ### although the energy is quite OK...
#        Amat = Hmat[1:,1:] - Xmat[1:,1:]
#        if hessian:
#            prints('hessian')
#            Amat = (Amat + Amat.T)/2
#        Vvec = Vvec[1:]
#        u, s, vh = sp.linalg.svd(Amat)
#        ## Moore-Penrose
#        #s_inv = np.zeros((nexcite, nexcite), dtype=float)
#        #rank = 0
#        #for i in range(nexcite):
#        #    if abs(s[i]) > eps:
#        #        s_inv[i,i] = 1/s[i]
#        #        rank += 1
#        #    else:
#        #        s_inv[i,i] = 0
#        #prints('rank = ',rank)
#
#        ## A.x = u.s.vh.x  = -b
#        ##s.vh.x = - u!.b
#        ##t_amp = - root_invS @ vh.T @ s_inv @ u.T @ V_ortho
#        #t_amp = - vh.T @ s_inv @ u.T @ Vvec
#        #Ecorr= (t_amp@Vvec).real
#        #EMoore = Quket.energy + Ecorr
#        #prints('norm = ',sp.linalg.norm(Amat@t_amp + Vvec))
#
#        ### Imaginary level-shift
#        s_inv = np.zeros((nexcite, nexcite), dtype=float)
#        if cf.debug:
#            prints('singular values\n',s)
#        for i in range(nexcite):
#            s_inv[i,i] = s[i]/(s[i]**2 + w2)
#
#        # A.x = u.s.vh.x  = -b
#        #s.vh.x = - u!.b
#        t_amp = - vh.T @ s_inv @ u.T @ Vvec
#        Ecorr = (t_amp@Vvec).real
#        Ereg = Quket.energy + Ecorr
#        ### Hylleraas functional
#        Ecorr = (t_amp @ Amat @ t_amp + 2 * t_amp @ Vvec).real
#        EHyl = Quket.energy + Ecorr
#        imax = np.argmax(abs(t_amp))
#        contribution = 0
#        imin = 0
#        for i in range(nexcite):
#            test = t_amp[i] * Vvec[i]
#            if test < contribution:
#                imin = i
#                contribution = test
#        #prints(f'Largest t-amplitude: {t_amp[imax]}')
#        #prints(f'Local correlation energy: {t_amp[imax]*Vvec[imax]}')
#        #prints(f'Excitation: {excitation_list[imax+1]}')
#        #prints(f'\n')
#        prints(f'Most contributing t-amplitue: {t_amp[imin]}')
#        prints(f'Local correlation energy: {contribution}')
#        prints(f'Excitation: {excitation_list[imin+1]}')
#        if cf.debug:
#            X = np.block([[Vvec],[t_amp]])
#            printmat(X,name='Comparison of V (0) and t (1)')
#
#        ### Orthonormalize the basis of Amat
#        ### This can completely remove the redundancy
#        ### but requires 4RDM...
#        Smat = Smat[1:,1:]
#        Null, Range = nullspace(Smat,eps=w2)
#        rank = Range.shape[1]
#        A_ortho = Range.T@Amat@Range
#        V_ortho = Range.T@Vvec
#        u, s, vh = sp.linalg.svd(A_ortho)
#        s_inv = np.zeros((rank, rank), dtype=float)
#        if cf.debug:
#            prints('singular values\n',s)
#        for i in range(rank):
#            if s[i] > w2:
#                s_inv[i,i] = 1/(s[i])
#            else:
#                s_inv[i,i] = 0
#            #s_inv[i,i] = s[i]/(s[i]**2 + w2)
#        t_amp = -Range @ vh.T @ s_inv @ u.T @ V_ortho
#        Ecorr = (t_amp @ Amat @ t_amp + 2 * t_amp @ Vvec).real
#        Eortho = Quket.energy + Ecorr
#        prints('Using U: Energy is ',Eortho)
#
#
#        # Solve instead A.S.S+.t + v = 0
#        # Get S+
#        s, U  = sp.linalg.eigh(Smat)
#        s_inv = np.zeros((nexcite, nexcite), dtype=float)
#        if cf.debug:
#            prints('singular values\n',s)
#        for i in range(nexcite):
#            s_inv[i,i] = s[i]/(s[i]**2 + w2)
#        ### Approximate A.S.S+
#        ASS = Amat @ Smat @ U @ s_inv @ U.T
#        u, s, vh = sp.linalg.svd(ASS)
#        s_inv = np.zeros((nexcite, nexcite), dtype=float)
#        if cf.debug:
#            prints('singular values\n',s)
#        for i in range(nexcite):
#            s_inv[i,i] = s[i]/(s[i]**2 + w2)
#        t_amp = - vh.T @ s_inv @ u.T @ Vvec
#        Ecorr = (t_amp @ Amat @ t_amp + 2 * t_amp @ Vvec).real
#        Eortho = Quket.energy + Ecorr
#
#        #### TEST
#        #SR = Smat@sp.linalg.pinv(Smat)
#        #AR = Amat@sp.linalg.pinv(Amat)
#        #diff = AR - SR
#        #printmat(diff, name='AR-SR')
#
#
#    elif method in ["cisd", "ucisd", "cepa", "ucepa0", "cepa0"]:
#        root_invS = root_inv(Smat)
#        rank = root_invS.shape[1]
#        # Orthonormalize the basis of Hmat and diagonalize it
#        Hmat -= Quket.energy * Smat
#        H_ortho = root_invS.T@Hmat@root_invS
#        e, U  = sp.linalg.eigh(H_ortho)
#        # Retrieve CI coefficients in the original basis
#        c = root_invS @ U
#        #if cf.debug:
#        #    printmat(c, name='c')
#        # Track the reference-dominant CI vector
#        c0 = 0
#        i = -1
#        t_amp = c[:,0]
#        if method in ["cisd", "ucisd", "cepa", "cepa0", "ucepa0"]:
#            while abs(c0) < 0.1:
#                i += 1
#                t_amp = c[:,i]
#                c0 = (t_amp[0]).real
#        Ecorr = e[i]
#        Ecisd = Ecorr
#        # Davidson correction
#        EQ = (1 - c0**2) * Ecorr
#        if method in ["cepa", "ucepa0", "cepa0"]:
#            Qmat = Smat
#            Qmat[0,0] = 0
#            # Iteratively solve with dressded Hamiltonian Hd = (H + Ec Q)
#            icyc = 0
#            dE = Ecorr
#            while icyc < 10 and abs(dE) > 1e-8:
#                Eold = Ecorr
#                Hd = Hmat + Ecorr * Qmat
#                H_ortho = root_invS.T@Hd@root_invS
#                e, U  = sp.linalg.eigh(H_ortho)
#                # Retrieve CI coefficients in the original basis
#                c = root_invS @ U
#                # Track the reference-dominant CI vector
#                c0 = 0
#                i = -1
#                while abs(c0) < 0.1:
#                    i += 1
#                    t_amp = c[:,i]
#                    c0 = (t_amp[0]).real
#                prints('Ecorr = ',Ecorr)
#                Ecorr = e[i]
#                dE = Ecorr - Eold
#                icyc += 1
#
#        prints(f'c0 = {c0}   Ecorr = {Ecorr}')
#    if cf.debug:
#        #printmat(t_amp, name='t_amp')
#        for i in range(len(t_amp)):
#            if abs(t_amp[i]) > 1e-6:
#                prints(t_amp[i], ' : ', excitation_list[i+1])
#
#    prints(f'Reference energy = {Quket.energy:.12f}')
#    if method in ["lucc", "luccsd", "luccd", "lucc2", "luccsd2", "pt2"]:
#        prints(f'{method} total energy (regularized)   = {Ereg:.12f}    (Ecorr = {Ereg - Quket.energy:.12f})')
#        prints(f'{method} total energy (Hylleraas)     = {EHyl:.12f}    (Ecorr = {EHyl - Quket.energy:.12f})')
#        prints(f'{method} total energy (Orthogonal)    = {Eortho:.12f}    (Ecorr = {Eortho - Quket.energy:.12f})')
#    else:
#        prints(f'{method} total energy   = {Quket.energy + Ecorr:.12f}  (Ecorr = {Ecorr:.12f})')
#        if method in ["cisd", "ucisd"]:
#            prints(f'{method}+Q total energy = {Quket.energy + Ecorr + EQ:.12f}')
#
#
#    t2 = time.time()
#    cput = t2 - t0
#
#    prints("\n Done: CPU Time =  ", "%15.4f" % cput)



    #######################################################
    ### TESTBENCH                                       ###
    from time import perf_counter as Time
    from quket.post.lucc_auxiliary.interface import interface as lucc_aux_interface
    from quket.opelib import excitation


    prints(
        "fro cor act vir\n"
        f"{Quket.nf}   {Quket.nc}   {Quket.na}   {Quket.ns}"
    )

    #### Excitation_list Generator ####

    Quket.from_vir = True
    excitation_list = [[0, 0]]
    prints(f'Check from-to spaces {Quket.include}')
    if spinfree:
        excite_dict = excitation.get_excite_dict_sf(Quket)
        # Singles
        for i, a in excitation.singles_sf(excite_dict):
            excitation_list.append((a, i))
        # Doubles
        for i, j, a, b in excitation.doubles_sf(excite_dict):
            excitation_list.append((a, b, i, j))
    else:
        excite_dict = excitation.get_excite_dict(Quket)
        # Singles
        for i, a in excitation.singles(excite_dict):
            excitation_list.append((a, i))
        # Doubles
        for i, j, a, b in excitation.doubles(excite_dict):
            if symmetry:
                ### Check symmetry
                if chk_sym_tau(Quket.tapering, [a,b,i,j], mapping=mapping):
                    excitation_list.append([a,b,i,j])
            else:
                excitation_list.append([a,b,i,j])
            #excitation_list.append((a, b, i, j))
    if cf.debug:
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        prints(f'my excitation_list {excitation_list}')
        for row in excitation_list:
            print(row)
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    prints('LENGTH', len(excitation_list))



    #fake_excitation_list = [ [0,0],
    #    [ 6, 4, 0, 2 ], [4, 0, 6, 2], [6, 2, 0, 4],
    #    [ 4, 0 ], [ 6, 0 ],
    #    [ 4, 2 ], [ 6, 2 ], [ 6, 4 ], [7, 5, 3, 1]
    #]
    #excitation_list = fake_excitation_list

    #excitation_list = excitation_list[:10]


    #                            Frozen || Core | Active || Secondary
    #                              nf   ||  nc  |   na   ||    ns
    #
    #    integrals                 nf   ||  nc  |   na   ||    ns
    #    integrals_active               ||      |   na   ||
    #    H_tlide                   nf   ||  nc  |   na   ||    ns
    #    1234RDM                        ||      |   na   ||
    #    excitation_list                ||  nc  |   na   ||    ns


    lucc_aux_mode = 'general' if Quket.nc + Quket.ns else 'special'
#    t0 = Time()
#    Amat = lucc_aux_interface(lucc_aux_mode, Quket, excitation_list[1:])
#    t1 = Time()
#    prints(f"LUCC auxiliary time = {t1-t0}")

    from quket.post.lucc_auxiliary._interface import interface as lucc_aux_interface
    t0 = Time()
    Vvec = get_V(Quket.operators.qubit_Hamiltonian, excitation_list, Quket.state, method, Quket)
    t1 = Time()
    prints(f"Vvec = {t1-t0}")
    t0 = Time()
    Smat = get_S(excitation_list, Quket.state, method, Quket)
    t1 = Time()
    prints(f"Smat = {t1-t0}")
    t0 = Time()
    Amat = lucc_aux_interface(lucc_aux_mode, Quket, excitation_list[1:])
    t1 = Time()
    prints(f"LUCC auxiliary time = {t1-t0}")
#    prints('Difference : ', np.linalg.norm(Amat - _Amat))
    hessian = False
    if method in ["lucc", "luccsd", "luccd", "lucc2", "luccsd2", "pt2"]:
        ### Solve A.t = - b by SVD...
        ### However, this does not remove the redundancy completely!
        ### although the energy is quite OK...
        if hessian:
            prints('hessian')
            Amat = (Amat + Amat.T)/2
        Vvec = Vvec[1:]
        u, s, vh = sp.linalg.svd(Amat)
        # Moore-Penrose
        s_inv = np.zeros((nexcite, nexcite), dtype=float)
        rank = 0
        eps = 1e-6
        for i in range(nexcite):
            if abs(s[i]) > eps:
                s_inv[i,i] = 1/s[i]
                rank += 1
            else:
                s_inv[i,i] = 0
        prints('rank = ',rank)

        # A.x = u.s.vh.x  = -b
        #s.vh.x = - u!.b
        #t_amp = - root_invS @ vh.T @ s_inv @ u.T @ V_ortho
        t_amp = - vh.T @ s_inv @ u.T @ Vvec
        Ecorr= (t_amp@Vvec).real
        EMoore = Quket.energy + Ecorr
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
        Ecorr = (t_amp @ Amat @ t_amp + 2 * t_amp @ Vvec).real
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

        ### Orthonormalize the basis of Amat
        ### This can completely remove the redundancy
        ### but requires 4RDM...
        Smat = Smat[1:,1:]
        Null, Range = nullspace(Smat,eps=w2)
        rank = Range.shape[1]
        A_ortho = Range.T@Amat@Range
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
        Ecorr = (t_amp @ Amat @ t_amp + 2 * t_amp @ Vvec).real
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
        #ASS = Amat @ Smat @ U @ s_inv @ U.T
        #u, s, vh = sp.linalg.svd(ASS)
        #s_inv = np.zeros((nexcite, nexcite), dtype=float)
        #if cf.debug:
        #    prints('singular values\n',s)
        #for i in range(nexcite):
        #    s_inv[i,i] = s[i]/(s[i]**2 + w2)
        #t_amp = - vh.T @ s_inv @ u.T @ Vvec
        #Ecorr = (t_amp @ Amat @ t_amp + 2 * t_amp @ Vvec).real
        #Eortho = Quket.energy + Ecorr

        prints(f'Reference energy = {Quket.energy:.12f}')
        if method in ["lucc", "luccsd", "luccd", "lucc2", "luccsd2", "pt2"]:
            prints(f'{method} total energy (regularized)   = {Ereg:.12f}    (Ecorr = {Ereg - Quket.energy:.12f})')
            prints(f'{method} total energy (Hylleraas)     = {EHyl:.12f}    (Ecorr = {EHyl - Quket.energy:.12f})')
            prints(f'{method} total energy (Orthogonal)    = {Eortho:.12f}    (Ecorr = {Eortho - Quket.energy:.12f})')
        else:
            prints(f'{method} total energy   = {Quket.energy + Ecorr:.12f}  (Ecorr = {Ecorr:.12f})')
            if method in ["cisd", "ucisd"]:
                prints(f'{method}+Q total energy = {Quket.energy + Ecorr + EQ:.12f}')

    return


def get_V(H, E, state, method, Quket, mapping = None):
   
    """Function
    Returns:
        Vvec ([nexcite]): V vector
    """
    if method not in ["lucc",  "cepa", "cepa0", "cisd", "ucisd", "luccsd", "luccd"]:
        raise ValueError('method has to be set in get_AV.')
    if method in ["lucc", "ucisd", "ucepa0", "luccsd", "luccd"]:
        hc = True
    else:
        hc = False
    n_qubits = state.get_qubit_count()
    size = len(E)
    if mapping is None:
        mapping = Quket.cf.mapping


    if '2' in method:
        ## Form generalized fock.
        fa, fb = get_Generalized_Fock_Matrix_one_body(Quket)
        Fock = create_1body_operator(fa, XB=fb)
        if mapping in ("jw", "jordan_wigner"):
            Fock = jordan_wigner(Fock)
        elif mapping in ("bk", "bravyi_kitaev"):
            Fock = bravyi_kitaev(Fock, n_qubits)
        Fpsi = evolve(Fock, state)

    # V
    t94 = time.time()
    Vvec = np.zeros(size, dtype=float)
    H_obs = OpenFermionOperator2QulacsGeneralOperator(H, n_qubits, mapping=mapping)
    E0 = H_obs.get_expectation_value(state)
    Hpsi = evolve(H, state, mapping=mapping)
    # <phi | [H, U[i]] |phi> = 2<phi | H U[i] |phi>
    for i in tqdm(range(size), desc="Vvec", ncols=100):
        if i % mpi.nprocs == mpi.rank:
            if i > 0:
                tau = get_tau(E[i], mapping=mapping, hc=hc, n_qubits=n_qubits)
                state_i = evolve(tau, state, mapping=mapping)
            else:
                state_i = state
            Vvec[i] = inner_product(Hpsi, state_i).real

    Vvec = mpi.allreduce(Vvec, mpi.MPI.SUM)
    t93 = time.time()
    prints(f"time for Vvec = {t93-t94} s")

    return Vvec

def get_excitation_list(Quket):
    
    from quket.projection import S2Proj

    t0 = time.time()
    cf.t_old = t0
    maxiter = Quket.maxiter
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    method = Quket.post_method
    mapping = Quket.cf.mapping

    ##number of operator is going to dynamically increase
    theta_list = []  ## initialize

    prints("Performing ", Quket.post_method, end="")
    if Quket.projection.SpinProj or Quket.projection.post_SpinProj:
        prints("(spin-projection) ", end="")
    #print_state(Quket.state, name='Reference state')
    if False:#Quket.projection.post_SpinProj:
        Quket.state = S2Proj(Quket, Quket.state)
        print_state(Quket.state, name='Reference projected state')
    Quket.energy = Quket.qulacs.Hamiltonian.get_expectation_value(Quket.state)
    prints(f'E[Reference] = {Quket.energy}')\

    spinfree = Quket.spinfree
    #Quket.get_1RDM()
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
    symmetry = Quket.tapering.clifford_operators is not None
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
                    if symmetry:
                        ### Check symmetry
                        if chk_sym_tau(Quket.tapering, u1, mapping=mapping):
                            excitation_list.append(u1)
                            excitation_list.append(u2)
                            nexcite += 2
                    else:
                        excitation_list.append(u1)
                        excitation_list.append(u2)
                        nexcite += 2
                elif len(u) == 2:
                    if symmetry:
                        ### Check symmetry
                        if chk_sym_tau(Quket.tapering, u, mapping=mapping):
                            excitation_list.append(u)
                            nexcite += 1
                    else:
                        excitation_list.append(u)
                        nexcite += 1
                elif len(u) == 1:
                    if symmetry:
                        ### Check symmetry
                        if chk_sym_tau(Quket.tapering, u, mapping=mapping):
                            excitation_list.append(u)
                            nexcite += 1
                    else:
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
                for p in range(Quket.n_active_orbitals):
                    for q in range(p):
                        x = [2*p+1, 2*q+1]
                        excitation_list.append(x)
                        nexcite += 1
                        nsing += 2
            ### doubles
            for u in u_list:
                for x in u:
                    if symmetry:
                        ### Check symmetry
                        if chk_sym_tau(Quket.tapering, x, mapping=mapping):
                            excitation_list.append(x)
                            nexcite += 1
                    else:
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

    return excitation_list


def get_S(E, state, method, Quket, mapping=None):
    """Function
    Given excitation list for operators E[i],
    evaluate the matrix elements S[i,j] = <E[i]! E[j]>.

    Args:
        E (list): [p,q,...,r,s] of anti-hermitized operators p^ q^ ... r s
        state (QuantumState): reference state

    Returns:
        Smat ([nexcite, nexcite]): Overlap matrix
    """
    if mapping is None:
        mapping = Quket.cf.mapping
    n_qubits = state.get_qubit_count()
    size = len(E)
    ##  S
    Smat = np.zeros((size, size), dtype=float)
    sizeT = size * (size+1)//2
    ij = 0
    for i in tqdm(range(size), desc="Smat", ncols=100):
        if i > 0:
            tau_i = get_tau(E[i], mapping=mapping, hc=hc, n_qubits=n_qubits)
            state_i = evolve(tau_i, state, mapping=mapping)
        else:
            state_i = state
        for j in range(i+1):
            if ij % mpi.nprocs == mpi.rank:
                if j > 0:
                    tau_j = get_tau(E[j], mapping=mapping, hc=hc, n_qubits=n_qubits)
                    state_j = evolve(tau_j, state, mapping=mapping)
                else:
                    state_j = state
                Smat[i,j] =  inner_product(state_i, state_j).real
                Smat[j,i] = Smat[i,j]
            ij += 1
    Smat = mpi.allreduce(Smat, mpi.MPI.SUM)
    t93 = time.time()
    prints(f"time for Smat = {t93-t94} s")

    return Smat

def get_AV(H, E, state, method, Quket, mapping=None):
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
    if mapping is None:
        mapping = Quket.cf.mapping
    n_qubits = state.get_qubit_count()
    size = len(E)


    if '2' in method:
        ## Form generalized fock.
        fa, fb = get_Generalized_Fock_Matrix_one_body(Quket)
        Fock = create_1body_operator(fa, XB=fb)
        if mapping in ("jw", "jordan_wigner"):
            Fock = jordan_wigner(Fock)
        elif mapping in ("bk", "bravyi_kitaev"):
            Fock = bravyi_kitaev(Fock, n_qubits)
        Fpsi = evolve(Fock, state)

    # V
    t94 = time.time()
    Vvec = np.zeros(size, dtype=float)
    H_obs = OpenFermionOperator2QulacsGeneralOperator(H, n_qubits, mapping=mapping)
    E0 = H_obs.get_expectation_value(state)
    Hpsi = evolve(H, state, mapping=mapping)
    # <phi | [H, U[i]] |phi> = 2<phi | H U[i] |phi>
    for i in tqdm(range(size), desc="Vvec", ncols=100):
        if i % mpi.nprocs == mpi.rank:
            if i > 0:
                tau = get_tau(E[i], mapping=mapping, hc=hc, n_qubits=n_qubits)
                state_i = evolve(tau, state)
            else:
                state_i = state
            Vvec[i] = inner_product(Hpsi, state_i).real

    Vvec = mpi.allreduce(Vvec, mpi.MPI.SUM)
    t93 = time.time()
    prints(f"time for Vvec = {t93-t94} s")

    ## H and S
    t94 = time.time()
    Hmat = np.zeros((size, size), dtype=float)
    Smat = np.zeros((size, size), dtype=float)
    if "lucc" in method:
        Xmat = np.zeros((size, size), dtype=float)
    else:
        Xmat = None
    sizeT = size * (size+1)//2
    ij = 0
    E0 = H_obs.get_expectation_value(state).real
    for i in tqdm(range(size), desc="Hmat", ncols=100):
        if i > 0:
            tau_i = get_tau(E[i], mapping=mapping, hc=hc, n_qubits=n_qubits)
            state_i = evolve(tau_i, state)
        else:
            state_i = state
        Hpsi_i = evolve(H, state_i, mapping=mapping)
        #S2psi_i = evolve(Quket.operators.S2, state_i)
        for j in range(i+1):
            if ij % mpi.nprocs == mpi.rank:
                #prints(f'{ij}/{sizeT}')
                if j > 0:
                    tau_j = get_tau(E[j], mapping=mapping, hc=hc, n_qubits=n_qubits)
                    state_j = evolve(tau_j, state)
                else:
                    state_j = state
                Hij = inner_product(Hpsi_i, state_j).real
                Smat[i,j] =  inner_product(state_i, state_j).real
                Smat[j,i] = Smat[i,j]
                if method in ["lucc", "luccsd", "luccd"]:
                    if i > 0:
                        state_ij = evolve(hermitian_conjugated(tau_i), state_j)
                    else:
                        state_ij = state_j
                    if j > 0:
                        state_ji = evolve(hermitian_conjugated(tau_j), state_i)
                    else:
                        state_ji = state_i
                    Xij = inner_product(state_ij, Hpsi).real
                    Xji = inner_product(state_ji, Hpsi).real
                    Xmat[j,i] = Xij
                    Xmat[i,j] = Xji
                Hmat[j,i] = Hij
                Hmat[i,j] = Hij
            ij += 1
    Hmat = mpi.allreduce(Hmat, mpi.MPI.SUM)
    Smat = mpi.allreduce(Smat, mpi.MPI.SUM)
    t93 = time.time()
    prints(f"time for Hmat = {t93-t94} s")

    if method in ["lucc", "luccsd", "luccd"]:
        Xmat = mpi.allreduce(Xmat, mpi.MPI.SUM)

    if method in ["cisd", "ucisd", "cepa", "cepa0", "ucepa0"]:
        for i in range(size-1):
            for j in range(size-1):
                Hmat[i+1,j+1] += - Hmat[0, i+1] * Smat[0, j+1] - Smat[i+1, 0] * Hmat[0, j+1] + E0 * Smat[i+1,0] * Smat[0,j+1]
        for i in range(size-1):
            Hmat[i+1,0] -= E0 * Smat[i+1,0]
            Hmat[0,i+1] -= E0 * Smat[i+1,0]
        Smat = Smat - np.outer(Smat[0], Smat[0])
        Smat[0,0] = 1

    return Hmat, Vvec, Smat, Xmat

def get_AV_2(H, E, state, method, Quket, nsing, ndoub, mapping=None):
    """Function
    Given excitation list for operators E[i],
    evaluate the matrix elements A[i,j] and V[i], and S[i,j] if necessary.
    For unitary methods, we use U[i] = E[i] - E![i] instead.
    A[i,j] is different for different methods.

                A[i,j] = 1/2 <[[H, Uj], Ui]>
                A[i,j] = <[[F, Uj], Ui]>     for  doubles

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
    if method != "lucc2" and method != "luccsd2" and method != "pt2":
        prints(f'method = {method}')
        raise ValueError('method has to be "lucc2" or "pt2" in get_AV_2.')
    hc = True

    if mapping is None:
        mapping = Quket.cf.mapping
    n_qubits = state.get_qubit_count()
    size = len(E)

    ## Form generalized fock.
    fa, fb = get_Generalized_Fock_Matrix_one_body(Quket)
    Fock = create_1body_operator(fa, XB=fb)
    if mapping in ("jw", "jordan_wigner"):
        Fock = jordan_wigner(Fock)
    elif mapping in ("bk", "bravyi_kitaev"):
        Fock = bravyi_kitaev(Fock, n_qubits)
    F_obs = OpenFermionOperator2QulacsGeneralOperator(Fock, n_qubits, mapping=mapping)
    # V
    Vvec = np.zeros(size, dtype=float)
    H_obs = OpenFermionOperator2QulacsGeneralOperator(H, n_qubits, mapping=mapping)
    E0 = H_obs.get_expectation_value(state)
    Hpsi = evolve(H, state)
    Fpsi = evolve(Fock, state)
    for i in range(size):
        if i % mpi.nprocs == mpi.rank:
            if i > 0:
                tau = get_tau(E[i], mapping=mapping, hc=hc, n_qubits=n_qubits)
                state_i = evolve(tau, state)
            else:
                state_i = state
            #if Quket.projection.SpinProj:
            #    from .phflib import S2Proj
            #    state_i = S2Proj(Quket, state_i)
            Vvec[i] = inner_product(Hpsi, state_i).real

    Vvec = mpi.allreduce(Vvec, mpi.MPI.SUM)

    ## H and S
    Hmat = np.zeros((size, size), dtype=float)
    Smat = np.zeros((size, size), dtype=float)
    Xmat = np.zeros((size, size), dtype=float)
    sizeT = size * (size+1)//2
    ij = 0
    E0 = H_obs.get_expectation_value(state).real
    for i in range(size):
        if i > 0:
            tau_i = get_tau(E[i], mapping=mapping, hc=hc, n_qubits=n_qubits)
            state_i = evolve(tau_i, state)
        else:
            state_i = state
        #if Quket.projection.SpinProj:
        #    from .phflib import S2Proj
        #    state_i = S2Proj(Quket, state_i)
        Hpsi_i = evolve(H, state_i)
        Fpsi_i = evolve(Fock, state_i)
        for j in range(i+1):
            if ij % mpi.nprocs == mpi.rank:
                #prints(f'{ij}/{sizeT}')
                if j > 0:
                    tau_j = get_tau(E[j], mapping=mapping, hc=hc, n_qubits=n_qubits)
                    state_j = evolve(tau_j, state)
                else:
                    state_j = state
                if (i > nsing and j > nsing) or method == "pt2":
                    #Hij =  F_obs.get_transition_amplitude(state_i, state_j).real
                    Hij = inner_product(Fpsi_i, state_j).real
                else:
                    #Hij =  H_obs.get_transition_amplitude(state_i, state_j).real
                    Hij = inner_product(Hpsi_i, state_j).real
                Smat[i,j] =  inner_product(state_i, state_j).real
                Smat[j,i] = Smat[i,j]
                if i > 0:
                    state_ij = evolve(hermitian_conjugated(tau_i), state_j)
                else:
                    state_ij = state_j
                if j > 0:
                    state_ji = evolve(hermitian_conjugated(tau_j), state_i)
                else:
                    state_ji = state_i

                if (i > nsing and j > nsing) or method == "pt2":
                    #Xij = F_obs.get_transition_amplitude(state_ij, state).real
                    #Xji = F_obs.get_transition_amplitude(state_ji, state).real
                    Xij = inner_product(state_ij, Fpsi).real
                    Xji = inner_product(state_ji, Fpsi).real
                else:
                    #Xij = H_obs.get_transition_amplitude(state_ij, state).real
                    #Xji = H_obs.get_transition_amplitude(state_ji, state).real
                    Xij = inner_product(state_ij, Hpsi).real
                    Xji = inner_product(state_ji, Hpsi).real
                Hmat[j,i] = Hij
                Hmat[i,j] = Hij
                Xmat[j,i] = Xij
                Xmat[i,j] = Xji
            ij += 1
    Hmat = mpi.allreduce(Hmat, mpi.MPI.SUM)
    Smat = mpi.allreduce(Smat, mpi.MPI.SUM)
    Xmat = mpi.allreduce(Xmat, mpi.MPI.SUM)


    return Hmat, Vvec, Smat, Xmat

def get_sf_taus(excitations, mapping=None, hc=True, n_qubits=None):
    """
    Generate pauli operators of its anti-hermitiian form of excitation operators given in "excitations".
    "excitations" has a set of operator lists, which should be all treated by one coefficient, e.g.,
    excitations = [[8, 4, 2, 0], [9, 4, 2, 1], [8, 5, 3, 0], [9, 5, 3, 1]]
    Each operator list [p,q,r,s] is expected to provide creation operators in the half list and annihilation operators for the rest,
    e.g., E=[p,q,r,s] gives tau=p^ q^ r s
    When hc is true, then this is added by its dagger s^ r^ q p.
    The resulting FermionOperator is transformed into QubitOperator basis.

    Args:
        excitations (list): integer list of p,q,...,r,s to represent the excitation p^ q^ ... r s
        hc (bool): Whether or not hermitian_conjugated is taken for E (E-E! or E).
        mapping : jordan_wigner or bravyi_kitaev.
                         For bravyi_kitaev, n_qubits is required.
    Returns:
        sigma (QubitOperator): Pauli string in QubitOperator class
    """
    if mapping is None:
        mapping = Quket.cf.mapping
    sigma = QubitOperator('',0)
    for E in excitations:
        rank = len(E)
        irank = 0
        excitation_string = ''
        if rank%2 != 0:
            raise ValueError('get_tau needs excitation string, i.e., even creation and annihilation')
        for p in E:
            if irank < rank//2: # creation
                excitation_string = excitation_string + str(p)+'^ '
            else: # annihilation
                excitation_string = excitation_string + str(p)+' '
            irank += 1
        fermi_op = FermionOperator(excitation_string)
        if hc:
            tau = fermi_op - hermitian_conjugated(fermi_op)
        else:
            tau = fermi_op
        if mapping == 'jordan_wigner':
            sigma_i = jordan_wigner(tau)
        elif mapping == 'bravyi_kitaev':
            if n_qubits is None:
                raise ValueError('n_qubits is necessary for bravyi_kitaev')
            sigma_i = bravyi_kitaev(tau, n_qubits)
        sigma += sigma_i
    return sigma

def get_tau(E, mapping='jordan_wigner', hc=True, n_qubits=None):
    """
    From orbital list [p,q], [p,q,r,s], ...,
    generate pauli operators of its anti-hermitiian form
    The first half of the list means creation operators,
    the last half of the list means annihilation operators,
    e.g., E=[p,q,r,s] gives tau=p^ q^ r s - s r q^ p^
    in QubitOperator basis.

    Args:
        E (list): integer list of p,q,...,r,s to represent the excitation p^ q^ ... r s
        hc (bool): Whether or not hermitian_conjugated is taken for E (E-E! or E).
        mapping : jordan_wigner or bravyi_kitaev.
                         For bravyi_kitaev, n_qubits is required.
    Returns:
        sigma (QubitOperator): Pauli string in QubitOperator class
    """
    rank = len(E)
    irank = 0
    excitation_string = ''
    if rank%2 != 0:
        raise ValueError('get_tau needs excitation string, i.e., even creation and annihilation')
    for p in E:
        if irank < rank//2: # creation
            excitation_string = excitation_string + str(p)+'^ '
        else: # annihilation
            excitation_string = excitation_string + str(p)+' '
        irank += 1
    #prints(excitation_string)
    fermi_op = FermionOperator(excitation_string)
    if hc:
        tau = fermi_op - hermitian_conjugated(fermi_op)
    else:
        tau = fermi_op
    if mapping == 'jordan_wigner':
        sigma = jordan_wigner(tau)
    elif mapping == 'bravyi_kitaev':
        if n_qubits is None:
            raise ValueError('n_qubits is necessary for bravyi_kitaev')
        sigma = bravyi_kitaev(tau, n_qubits)
    elif mapping is None:
        sigma = tau
    return sigma


def get_sf_list(r_list, type):
    """
    From a spatial orbital list of double excitations [p, r, q, s],
    which indicates either
        (pA^ rA + pB^ rB) (qA^ sA + qB^ sB)     : type = 1
        (pA^ sA + pB^ sB) (qA^ rA + qB^ rB)     : type = 2
    this function generates a spin orbital list with spin-free excitations.

    """
    # p^ q^ s r = p^ r q^s - delta_qr p^ s
    # we have p^ r q^ s as r_list = [p, r, q, s]

    p = r_list[0]
    r = r_list[1]
    q = r_list[2]
    s = r_list[3]
    pA = 2*p
    qA = 2*q
    rA = 2*r
    sA = 2*s
    pB = 2*p+1
    qB = 2*q+1
    rB = 2*r+1
    sB = 2*s+1
    u_list = []
    if type == 1:
        ### alpha-alpha
        u_list.append([pA, qA, sA, rA])
        ### beta-alpha
        u_list.append([pB, qA, sA, rB])
        ### alpha-beta
        u_list.append([pA, qB, sB, rA])
        ### beta-beta
        u_list.append([pB, qB, sB, rB])
    elif type == 2:
        ### alpha-alpha
        u_list.append([pA, qA, rA, sA])
        ### beta-alpha
        u_list.append([pB, qA, rA, sB])
        ### alpha-beta
        u_list.append([pA, qB, rB, sA])
        ### beta-beta
        u_list.append([pB, qB, rB, sB])

    return u_list

def get_AV_sf(H, E, state, method, Quket, mapping=None):
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
    if mapping is None:
        mapping = Quket.cf.mapping

    if method not in ["lucc", "luccsd", "luccd",  "cepa", "cepa0", "cisd", "ucisd",\
                      "luccsd", "luccd"]:
        raise ValueError('method has to be set in get_AV.')
    if method in ["lucc", "luccsd", "luccd", "ucisd"]:
        hc = True
    else:
        hc = False

    n_qubits = state.get_qubit_count()
    size = len(E)
    # V
    Vvec = np.zeros(size, dtype=float)
    H_obs = OpenFermionOperator2QulacsGeneralOperator(H, n_qubits, mapping=mapping)
    E0 = H_obs.get_expectation_value(state)
    Hpsi = evolve(H, state)
    for i in range(size):
        if i % mpi.nprocs == mpi.rank:
            if i > 0:
                tau = get_sf_taus(E[i], mapping=mapping, hc=hc)
                state_i = evolve(tau, state)
                #print('i = ',i, ' ', E[i], ' S2=', Quket.qulacs.S2.get_expectation_value(state_i))
            else:
                state_i = state
            #if Quket.projection.SpinProj:
            #    from .phflib import S2Proj
            #    state_i = S2Proj(Quket, state_i)
            Vvec[i] = inner_product(Hpsi, state_i).real
            if Vvec[i] != 0:
                print(E[i], ' ->  V=' ,Vvec[i])

    Vvec = mpi.allreduce(Vvec, mpi.MPI.SUM)
    ## H and S
    Hmat = np.zeros((size, size), dtype=float)
    Smat = np.zeros((size, size), dtype=float)
    if "lucc" in method:
        Xmat = np.zeros((size, size), dtype=float)
    else:
        Xmat = None
    sizeT = size * (size+1)//2
    ij = 0
    prints('hc',hc)
    for i in range(size):
        if i > 0:
            tau_i = get_sf_taus(E[i], mapping=mapping, hc=hc)
            state_i = evolve(tau_i, state)
        else:
            state_i = state
        #if Quket.projection.SpinProj:
        #    from .phflib import S2Proj
        #    state_i = S2Proj(Quket, state_i)
        s2 = Quket.qulacs.S2.get_expectation_value(state_i)
        if s2 > 1e-6:
            print('Wrong S2 = ', s2, ' i = ',i)
        Hpsi_i = evolve(H, state_i)
        for j in range(i+1):
            if ij % mpi.nprocs == mpi.rank:
                #prints(f'{ij}/{sizeT}')
                if j > 0:
                    tau_j = get_sf_taus(E[j], mapping=mapping, hc=hc)
                    state_j = evolve(tau_j, state)
                else:
                    state_j = state
                #Hij =  H_obs.get_transition_amplitude(state_i, state_j).real
                Hij = inner_product(Hpsi_i, state_j).real
                Smat[i,j] =  inner_product(state_i, state_j).real
                Smat[j,i] = Smat[i,j]
                if "lucc" in method:
                    if i > 0:
                        state_ij = evolve(hermitian_conjugated(tau_i), state_j)
                    else:
                        state_ij = state_j
                    if j > 0:
                        state_ji = evolve(hermitian_conjugated(tau_j), state_i)
                    else:
                        state_ji = state_i
                    s2_ij = Quket.qulacs.S2.get_expectation_value(state_ij)
                    s2_ji = Quket.qulacs.S2.get_expectation_value(state_ji)
                    if s2_ij > 1e-6 or s2_ji > 1e-6:
                        print('Wrong S2:\n s2_ij = ', s2_ij, '  s2_ji = ',s2_ji, ' ij = ',i,' ',j)
                    Xij = inner_product(state_ij, Hpsi).real
                    Xji = inner_product(state_ji, Hpsi).real
                    Xmat[j,i] = Xij
                    Xmat[i,j] = Xji
                Hmat[j,i] = Hij
                Hmat[i,j] = Hij
            ij += 1
    Hmat = mpi.allreduce(Hmat, mpi.MPI.SUM)
    Smat = mpi.allreduce(Smat, mpi.MPI.SUM)
    if "lucc" in method:
        Xmat = mpi.allreduce(Xmat, mpi.MPI.SUM)


    if method in ["cisd", "ucisd", "cepa", "cepa0", "ucepa0"]:
        for i in range(size-1):
            for j in range(size-1):
                Hmat[i+1,j+1] += - Hmat[0, i+1] * Smat[0, j+1] - Smat[i+1, 0] * Hmat[0, j+1] + E0 * Smat[i+1,0] * Smat[0,j+1]
        for i in range(size-1):
            Hmat[i+1,0] -= E0 * Smat[i+1,0]
            Hmat[0,i+1] -= E0 * Smat[i+1,0]
        Smat = Smat - np.outer(Smat[0], Smat[0])
        Smat[0,0] = 1

    return Hmat, Vvec, Smat, Xmat

def chk_sym_tau(Tapering, E, mapping='jordan_wigner'):
    """Function
    Check if E is symmetry-allowed.
    INEFFICIENT CODE (should be replaced by one using symmetry-table.)
    """
    from quket.tapering import transform_pauli
    # Transform E to pauli
    pauli = get_tau(E, mapping=mapping, n_qubits=n_qubits)
    i = 0
    _, allowed = transform_pauli(Tapering, pauli, reduce=False)
    return allowed
