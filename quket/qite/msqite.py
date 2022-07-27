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

msqite.py

Functions related to model-space QITE.

"""
import time
import scipy as sp
import numpy as np
from numpy import linalg as LA
from qulacs.state import inner_product
from scipy.optimize import minimize

from .qite_function import ( calc_Hpsi, Nonredundant_Overlap_list, Overlap_by_nonredundant_pauli_list, Overlap_by_pauli_list, Overlap_by_pauli_list_IJ, msqlanczos)
from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import prints, print_state, printmat
from quket.utils import fermi_to_str, isfloat
from quket.linalg import lstsq, root_inv, Lowdin_orthonormalization, Lowdin_deriv_d
from quket.opelib import evolve
from quket.opelib import create_exp_state 
from quket.lib import QuantumState, normal_ordered, reverse_jordan_wigner

def msqite(Quket):
    """
    Main driver for MSQITE
    
    Author(s): Yoohee Ryo, Takashi Tsuchimochi
    """
    # Parameter setting
    nstates = len(Quket.multi.weights)
    n = Quket.n_qubits
    ansatz = Quket.ansatz
    db = Quket.dt
    qbit = Quket.current_det
    ntime = Quket.maxiter
    regularization = Quket.regularization
    #observable = Quket.qulacs.Hamiltonian
    H = Quket.qulacs.Hamiltonian
    # spin-shifted operator  H + s2shift * (S**2)
    observable = Quket.qulacs.HS2
    #observable2 = Quket.qulacs.Hamiltonian2
    S2_observable = Quket.qulacs.S2
    Number_observable = Quket.qulacs.Number
    threshold = Quket.ftol
    S2 = 0
    Number = 0
    use_qlanczos = Quket.qlanczos

    if Quket.ansatz == 'cite':
        prints(f"MSQITE")
    else:
        size = len(Quket.pauli_list)
        if Quket.msqite in ("state_average", "sa"):
            independent = False
            prints(f"MSQITE (State-Average): Pauli operator group size = {size}")
        elif Quket.msqite in ("true", "state_specific", "ss"):
            independent = True
            # independent : solve Sa+b=0 for each istate independently 
            prints(f"MSQITE (State-Specific): Pauli operator group size = {size}")
        else: 
            prints(f"No option '{Quket.msqite}' for MSQITE")
            
    if ansatz != "cite":
        nonredundant_sigma, pauli_ij_index, pauli_ij_coef = Nonredundant_Overlap_list(Quket.pauli_list, n)
        len_list = len(nonredundant_sigma)
        prints(f"    Unique sigma list = {len_list}")
    index = np.arange(n)

    energy = [[] for x in range(nstates)]
    #psi_dash = [first_state[x].copy() for x in range(nstates)]
    psi_dash = [Quket.multi.states[x].copy() for x in range(nstates)]
    for istate in range(nstates):
        print_state(psi_dash[istate], name=f'Initial state {istate}')    
    En = [0 for x in range(nstates)]
    S2 = [0 for x in range(nstates)]
    Number = [0 for x in range(nstates)]
    cm = [0 for x in range(nstates)]

    # pramater of lanczos #
    s2 = []
    S_list = []
    H_list = []
    S2_list = []
    d_list = []
    qlanz = []
    ##########
    
    t1 = time.time()
    cf.t_old = t1

    H_eff= np.zeros((nstates,nstates), dtype=complex)
    S2_eff= np.zeros((nstates,nstates), dtype=complex)
    for i in range(nstates):
        for j in range(nstates):
            H_eff[i,j]=H.get_transition_amplitude(psi_dash[i],psi_dash[j])
            if "heisenberg" not in Quket.basis:
                S2_eff[i, j] = Quket.qulacs.S2.get_transition_amplitude(
                    psi_dash[i], psi_dash[j])

            
    H_list.append(H_eff.copy())
    S2_list.append(S2_eff.copy())

    En, c = np.linalg.eigh(H_eff)
    E0_ave = 0
    for istate in range(nstates):
        energy[istate].append(En[istate])
        E0_ave += En[istate]/nstates

    # pramater of lanczos #
    q_en = [En]
    H_q = None
    S_q = None

    for istate in range(nstates):
        if S2_observable is not None:
            S2[istate] = S2_observable.get_expectation_value(psi_dash[istate])
        else:
            S2 = 0
        if Number_observable is not None:
            Number[istate] = Number_observable.get_expectation_value(psi_dash[istate])
    ##########

    # pramater of lanczos #
    q_en = [En]
    H_q = None
    S_q = None
    S2_q = None


    ##########
    # Shift  #
    ##########
    if Quket.shift in ['hf', 'step']:
        shift = [H_eff[x,x] for x in range(nstates)]
    elif Quket.shift == 'true':
        shift = [H_eff[x,x] for x in range(nstates)]
    elif Quket.shift in ['none', 'false']:
        shift = [0 for x in range(nstates)]
    elif isfloat(Quket.shift):
        raise ValueError(f"shift {Quket.shift} cannot be used for MSQITE (multiple shifts are needed)")
    else:
        raise ValueError(f"unknown shift option: {Quket.shift}")
    if ansatz == "cite":
        order = 0
    else:
        order = 1
    prints('Shift = ', shift)
    prints('QLanczos = ', use_qlanczos)
    ref_shift = shift  # Reference shift
    dE = 100
    beta = 0
    Conv = False
    c = np.identity(nstates)
    d = np.zeros((nstates,nstates))
    S_eff= np.identity(nstates, dtype=complex)
    S_list.append(S_eff)
    psi_next=[QuantumState(n) for x in range(nstates)]

    ### 
    # lowdin : perform Lowdin's symmetric orthonormalization
    # use_c : use eigenvectors c instead of Lowdin's d 
    lowdin = True
    #lowdin = False
    use_c = True
    #use_c = False
    ###
    ave_old = 0

    for t in range(ntime):
        t2 = time.time()
        cput = t2 - cf.t_old
        cf.t_old = t2
        if cf.debug:
            for istate in range(nstates):
                print_state(psi_dash[istate], name=f'istate={istate}')
        for istate in range(nstates):
            if istate == 0:
                prints(f"{beta:6.2f}")
            prints(f"      state{istate}: E = {En[istate]:.12f}  "
                f"<S**2> = {S2[istate]:+10.8f}  "
                f"<N> = {Number[istate]:10.8f}  "
                f"Fidelity = {Quket.fidelity(psi_dash[istate]):.6f}  "
                f"CPU Time = {cput: 5.2f}  ")
        """
        for istate in range(nstates):
            if mpi.main_rank:
                prints(f"state{istate}")
                print_state(psi_dash[istate])
        """    
        prints("")
        # Save the QLanczos details in out

        if abs(dE) < threshold:
            Conv = True
            if independent:
            #    independent = False
            #    Conv = False
                break
            else:
                break
        if Quket.shift == 'step':
            shift = [H_eff[x,x] for x in range(nstates)]
        elif Quket.shift == 'none':
            shift = [0 for x in range(nstates)]
        T0 = time.time()
        
        beta += db
        T1 = time.time()
        ####
        #MSQITE
        #ξIとξJからSeffを求める　ξ=psi_dash

        if ansatz!= "cite":
            ### Form H|psi>
            Hpsi = []
            for istate in range(nstates):
                Hpsi.append(calc_Hpsi(observable, psi_dash[istate], shift=shift[istate], db=1, j=-1))

            # determine db
            while True:
                if not independent:
                    S_eff = np.identity(nstates, dtype=complex)
                Quket.multi.states = []
                H1 = np.zeros((nstates,nstates), dtype=complex)
                H1_ = np.zeros((nstates,nstates), dtype=complex)
                for i in range(nstates):
                    Quket.multi.states.append(psi_dash[i].copy())
                    for j in range(nstates):
                        #<ξI(l)|H|ξJ(l)> is Heff from the last cycle
                        H1[i,j] = - 2*H_eff[i,j] + (shift[i] + shift[j]) * S_eff[i,j]
                    
                S_tilde = S_eff + db * H1
                if lowdin:
                    d = Lowdin_orthonormalization(S_tilde)
                    ### Gram-Schmidt ###
                    #Q,R = np.linalg.qr(np.linalg.inv(root_inv(S_tilde)))
                    #X = np.linalg.inv(R)
                    #for i in range(X.shape[0]):
                    #    if X[i,i] < 0:
                    #        X[:,i] *= -1
                    #d = X
                    ### TEST

                elif use_c: 
                    #d = c
                    for i in range(nstates):
                        for j in range(nstates):
                            #H_eff ~ <ξI(l)|H|ξJ(l)> - 2 db <ξI(l)|H^2|ξJ(l)>
                            H_eff[i,j]= H_eff[i,j] - 2*db*(Quket.get_Heff(psi_dash[i],Hpsi[j])) 
                    root_invS = root_inv(S_eff, eps=1e-9)
                    H_ortho = root_invS.T@H_eff@root_invS
                    printmat(H_ortho)
                    eig,c = np.linalg.eigh(H_ortho)
                    d = root_invS @ c
                    d1 = d
                else:
                    # Do not orthogonalize but only normalize
                    d = np.eye(nstates)
                    d1 = d
                    #d *= 0
                    #for istate in range(nstates):
                    #    d[istate, istate] = 1/np.sqrt(S_eff[istate,istate])
                #prints(f'time ell = {t}')
                #printmat(H_list[t], f'Heff({t})')
                #printmat(S_list[t], f'Seff({t})')
                d_ = np.zeros_like(d)
                for J in range(nstates):
                    #d_[J, :] =  d[J, :] * np.exp(db * (shift[J])) 
                    d_[J, :] =  d[J, :] * np.exp(db * (shift[J] - E0_ave)) 
                    #prints(f'exp(db dEj) for {j=}: {np.exp(db * (shift[J] - E0_ave))}') 
                d_list.append(d_)
                
                #printmat(d.real, name=f'd({t})')
                #printmat(d_.real, name=f'd_({t})')

                ### Form \sum_K |K> dKJ  and   \sum_K H|K> dKJ
                psi_d = []
                Hpsi_d = []
                for jstate in range(nstates):
                    xi_j = QuantumState(n)
                    xi_j.multiply_coef(0)
                    Hxi_j = QuantumState(n)
                    Hxi_j.multiply_coef(0)
                    for kstate in range(nstates): 
                        xi_k = psi_dash[kstate].copy()
                        xi_k.multiply_coef(d[kstate,jstate])
                        xi_j.add_state(xi_k)
                        Hxi_k = Hpsi[kstate].copy()
                        Hxi_k.multiply_coef(d[kstate,jstate])
                        Hxi_j.add_state(Hxi_k)

                    psi_d.append(xi_j) 
                    Hpsi_d.append(Hxi_j) 

                T5 = time.time()
                ### Form bI = -2 \sum_K Im <I|sigma H|K> dKJ
                ###         = -2 Im <I|sigma |Hpsi_d[J]>
                ### and  sigmaIJ = 2 \sum_K Im <I|sigma |K> dKJ
                ###          = 2 Im <I|sigma |psi_d[J]>
                ipos, my_ndim = mpi.myrange(len(Quket.pauli_list))
                bI = np.zeros((size, nstates), dtype=float, order='F')
                for istate in range(nstates):
                    for i, pauli in enumerate(Quket.pauli_list):
                        if ipos <= i < ipos + my_ndim:
                            state_i = evolve(pauli, psi_dash[istate], parallel=False)
                            bI[i, istate] = -2 * inner_product(state_i, Hpsi[istate]).imag \
                                            +2 * inner_product(state_i, psi_d[istate]).imag * (1 / db)
                bI = mpi.allreduce(bI)


                ## Compute Sij
                S = []
                for istate in range(nstates):
                    #S.append(Overlap_by_pauli_list(Quket.pauli_list, psi_dash[istate]))
                    S.append(Overlap_by_nonredundant_pauli_list(nonredundant_sigma, pauli_ij_index, pauli_ij_coef, psi_dash[istate]))

                Amat = [2*np.real(S[x]) for x in range(nstates)]
                T5 = time.time()

                b_sum = np.zeros(size)
                A_mat_sum = np.zeros((size, size))
                for istate in range(nstates):
                    b_sum += bI[:, istate]
                    A_mat_sum += Amat[istate]

                #prints(np.linalg.eigh(A_mat_sum)[0])

                b_norm = np.zeros(nstates)
                for istate in range(nstates):
                    val = 0
                    for i in range(size):
                        val += bI[i, istate] **2
                    b_norm[istate] = np.sqrt(val)
                b_sum_norm = 0
                for i in range(size):
                    b_sum_norm += b_sum[i]**2
                b_sum_norm = np.sqrt(b_sum_norm)
                if cf.debug:
                    printmat(d, name='d')
                    printmat(b_norm, name='b_norm')
                    prints('b_sum_norm ',b_sum_norm)

            ### Sum ###
                if not independent:
                    #a, res, rnk, s = lstsq(A_mat_sum, -b_sum, cond=1e-6, damp=0.000001)
                    a =  -np.linalg.pinv(A_mat_sum, rcond=1e-6) @ b_sum
                    a *= db
                    if cf.debug:
                        printmat(A_mat_sum, name='Asum')
                        printmat(bI, name='b')
                        printmat(b_sum, name='b')
                        printmat(a, name='a (average)')
                    for istate in range(nstates):
                        dum = psi_next[istate].copy()  ## Target state
                        dum.multiply_coef(-1)
                        psi_next[istate] = create_exp_state(Quket, init_state=psi_dash[istate], theta_list=-a)
                        ### Check
                        dum.add_state(psi_next[istate])
                        if cf.debug:
                            prints(f'State = {istate}    Diff Norm : = {dum.get_squared_norm()}')
                    break

                else:
            ### Independent ###
                    x = np.zeros((size,nstates))
                    ipos, my_ndim = mpi.myrange(nstates)
                    #for i in range(ipos, ipos+my_ndim):
                    if mpi.main_rank:
                        for i in range(nstates):
                            ### solve Aa = -b
                            a, res, rnk, s = lstsq(Amat[i], -bI[:,i], cond=1e-7, regularization=regularization)
                            #if regularization > 0:
                            #    a, res, rnk, s = lstsq(Amat[i], -bI[:,i], cond=1e-7, regularization=regularization)
                            #else:
                            #    try:
                            #        a =  -np.linalg.pinv(Amat[i], rcond=1e-7) @ bI[:,i]
                            #    except:
                            #        a, res, rnk, s = lstsq(Amat[i], -bI[:,i], cond=1e-7, regularization=regularization)
                            #if cf.debug:
                                #printmat(Amat[i], name=f"Amat {i}")
                                #printmat(np.linalg.pinv(Amat[i], rcond=1e-6), name=f"Ainv")
                            ### If ||a|| is two large compared to ||b||,
                            ### reduce cond (threshold for svd) up to 1e-4 
                            #cond = 1e-6
                            #cont = True
                            #while cont:
                            #    a, res, rnk, s = lstsq(Amat[i], -bI[:,i], cond=cond)
                            #    a_norm = np.linalg.norm(a*db)
                            #    amax = max(abs(a*db))
                            #    prints(f'beta = {beta:5.2f}  state{i}    ||a|| = {a_norm:7.4f}   ||b|| = {b_norm[i]:7.4f}   ||a||/||b|| = {a_norm/b_norm[i]:7.4f}   amax = {amax:7.4f}')
                            #    #### Scale down if x is too large 
                            #    if a_norm/b_norm[i] < 10 or cond >= 1e-4:
                            #        cont = False
                            #        prints(a_norm/b_norm[i], cond)
                            #    else:
                            #        cond *= 2
                            #    cont = False

                            x[:,i] = a*db
                    x = mpi.allreduce(x, mpi.MPI.SUM)
                    
                    ### Scale down if x is too large 
                    a_norm = np.zeros(nstates)
                    for istate in range(nstates):
                        val = 0
                        for i in range(size):
                            val += x[i, istate] **2
                        a_norm[istate] = np.sqrt(val)
#### TEST damping 
                    #if beta > 1:
                    #    for i in range(nstates):
                    #        #if a_norm[i] > b_norm[i]:
                    #        #    x[:,i] *= b_norm[i]/a_norm[i]
                    #        amax = max(abs(x[:,i]))
                    #        if amax > 0.3: 
                    #            x[:,i] *= 0.3/amax

                    for istate in range(nstates):
                        dum = psi_d[istate].copy()  ## Target state
                        dum.multiply_coef(-1)
                        psi_next[istate] = create_exp_state(Quket, init_state=psi_dash[istate], theta_list=-x[:,istate])
                        dum.add_state(psi_next[istate])
                        if cf.debug:
                            prints(f'State = {istate}    Diff Norm : = {dum.get_squared_norm()}')
                    if cf.debug:
                        printmat(a_norm, name='Norm(a)')
                        printmat(bI, name='b')
                        printmat(x, name='a')
                    break
        elif ansatz== "cite":
            ### Form H|psi>
            Hpsi = []
            for istate in range(nstates):
                Hpsi.append(calc_Hpsi(observable, psi_dash[istate], shift=shift[istate], db=db, j=1))
                psi_dash[istate].add_state(Hpsi[istate])
            ### Orthonormalize the state basis again.
            S_eff= np.zeros((nstates,nstates), dtype=complex)
            for i in range(nstates):
                for j in range(nstates):
                    S_eff[i,j]=inner_product(psi_dash[i],psi_dash[j])
            d = Lowdin_orthonormalization(S_eff) 
            for i in range(nstates):
                psi_next[i] = QuantumState(n)
                psi_next[i].multiply_coef(0)
                for j in range(nstates): 
                    chi=psi_dash[j].copy() 
                    chi.multiply_coef(d[j,i])
                    psi_next[i].add_state(chi)
            for i in range(nstates):
                psi_dash[i] = psi_next[i].copy()
#        
#        for istate in range(nstates):
#            En[istate] = observable.get_expectation_value(psi_dash[istate])
        

        #Heff
        H_eff= np.zeros((nstates,nstates), dtype=complex)
        S_eff= np.zeros((nstates,nstates), dtype=complex)
        S2_eff= np.zeros((nstates,nstates), dtype=complex)
        for i in range(nstates):
            for j in range(nstates):
                H_eff[i,j]=H.get_transition_amplitude(psi_next[i],psi_next[j])
                S_eff[i,j]=inner_product(psi_next[i],psi_next[j])
                if "heisenberg" not in Quket.basis :
                    S2_eff[i, j] = Quket.qulacs.S2.get_transition_amplitude(
                        psi_next[i], psi_next[j])


        #if independent:
        #    root_invS = root_inv(S_eff, eps=1e-9)
        #else:
        #    root_invS = np.identity(nstates) 
        #S_list.append(S_tilde_.copy())
        S_list.append(S_eff.copy())
        H_list.append(H_eff.copy())
        S2_list.append(S2_eff.copy())

        if cf.debug:
            printmat(H_eff, name='H_eff')
            printmat(S_eff, name='S_eff')
        root_invS = root_inv(S_eff, eps=1e-9)
        H_ortho = root_invS.T@H_eff@root_invS
        eig,c = np.linalg.eig(H_ortho)

        # Sort 
        ind = np.argsort(eig)
        eig = eig[ind]
        c  = c[:, ind]

        update = True

        ### <S**2> and <N>
        c  = root_invS@c
        for istate in range(nstates):
            if c[istate,istate] < 0:
                c[:,istate] *= -1
            state = QuantumState(n)
            state.multiply_coef(0)
            for jstate in range(nstates):
                temp = psi_next[jstate].copy()
                temp.multiply_coef(c[jstate,istate])
                state.add_state(temp)
            if S2_observable is not None:
                S2[istate] = S2_observable.get_expectation_value(state)
            else:
                S2 = 0
            if Number_observable is not None:
                Number[istate] = Number_observable.get_expectation_value(state)

        ##### TEST
        ##cを用いてξのアップデート
        #for i in range(nstates):
        #    psi_dash[i]=QuantumState(n)
        #    psi_dash[i].multiply_coef(0)
        #    for j in range(nstates): 
        #        chi=psi_next[j].copy() 
        #        chi.multiply_coef(c[j][i])
        #        psi_dash[i].add_state(chi)
        #for i in range(nstates):
        #    for j in range(nstates):
        #        H_eff[i,j]=H.get_transition_amplitude(psi_dash[i],psi_dash[j])
        #        S_eff[i,j]=inner_product(psi_dash[i],psi_dash[j])
        #psi_next = psi_dash.copy()
        #printmat(H_eff, name='Heff (diag?)')
        #printmat(S_eff, name='Seff (diag?)')

        
        ave = 0

        for istate in range(nstates):
            psi_dash[istate] = psi_next[istate].copy()
            En[istate] = eig[istate].real
            energy[istate].append(En[istate])
            ave += En[istate]

        ave /= nstates
        
        if ave > ave_old:
            prints('WARNING: Energy increased during cycles')
            prints(f'Average energy : Present = {ave}     Last = {ave_old}\n')
            #printmat(bI,name='b vector')
            #printmat(x,name='a vector')
            #if beta > 10:
            #    return
        #res_ = np.zeros((size,nstates))
        #for istate in range(nstates):
        #    res_[:, istate] = Amat[istate] @ x[:,istate]/db + bI[:,istate]

        #printmat(res_,name='residual vector')

        ave_old = ave
        ####

        if use_qlanczos and t % 2 == 1:
            S_q, H_q, q_en, S2_q, q_s2 = msqlanczos(
                d_list, nstates, t, S_list, H_list, S2_list, S_q, H_q, S2_q)

            prints(f"  QLanczos:")
            prints('\n'.join(
                f'      E = {e:+.12f} [<S**2> = {s:+.12f}]' for (e, s) in zip(q_en, q_s2)))
            prints("")


        dE = 0
        for istate in range(nstates):
            dE += energy[istate][t+1] - energy[istate][t]
        dE /= nstates
        Quket.state = psi_dash[0]

    if Conv:
        prints(f"CONVERGED at beta={beta:.2f}.")
    else:
        prints(f"CONVERGE FAILED.")

    prints("Final:  ")
    for istate in range(nstates):
        prints(f"      E[{istate}-MSQITE] = {En[istate]:.12f}  "
               f"(<S**2> = {S2[istate]:+7.5f})  ")
    prints("------------------------------------")
    for istate in range(nstates):
        print_state(psi_dash[istate], name=f"QITE Basis   {istate}")
    printmat(c.real, name="Coefficients:")
    prints("------------------------------------\n\n")
    prints("###############################################")
    prints("#               MSQITE   states               #")
    prints("###############################################", end="")

    for istate in range(nstates):
        prints()
        prints(f"State        : {istate}")
        prints(f"E            : {En[istate]:.8f}")
        prints(f"<S**2>       : {S2[istate]:+.5f}")
        spstate = QuantumState(n)
        spstate.multiply_coef(0)
        for jstate in range(nstates):
            state = psi_dash[jstate].copy()
            coef = c[jstate, istate]
            state.multiply_coef(coef)
            spstate.add_state(state)
        print_state(spstate, name="Superposition:")
    prints("###############################################")


