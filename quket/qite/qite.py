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

qite.py

Main driver of QITE.

"""
import scipy as sp
import numpy as np
import time
from numpy import linalg as LA

from qulacs.state import inner_product
from qulacs.observable import create_observable_from_openfermion_text

from quket import config as cf
from quket.mpilib import mpilib as mpi
from .qite_exact import qite_exact
from .qite_inexact import qite_inexact
from .msqite import msqite

from .qite_function import (qlanczos, calc_expHpsi, calc_Hpsi, Nonredundant_Overlap_list, Overlap_by_nonredundant_pauli_list, Overlap_by_pauli_list)
from quket.fileio import prints, print_state, printmat
from quket.utils import fermi_to_str, isfloat
from quket.linalg import lstsq
from quket.opelib import evolve
from quket.opelib import create_exp_state 
from quket.lib import QuantumState, QubitOperator

def QITE_driver(Quket):
    prints("Enetered QITE driver")
    model = Quket.model
    ansatz = Quket.ansatz
    det = Quket.current_det
    n_orbitals = Quket.n_active_orbitals
    n_qubits = Quket._n_qubits
    ftol = Quket.ftol
    truncate = Quket.truncate
    use_msqite = Quket.msqite != "false"

    opt = f"0{n_qubits}b"
    prints(f"Performing QITE for {model} Hamiltonian")
    prints(f"Ansatz = {Quket.ansatz}")
    if type(det) is int:
        prints(f"Initial configuration: | {format(det, opt)} >")
    else:
        prints(f"Initial configuration: ",end='')
        for i, state_ in enumerate(det):
            prints(f" {state_[0]:+.4f} * | {format(state_[1], opt)} >", end='')
            if i > 10:
                prints(" ... ", end='')
                break
        prints('')
    prints(f"Convergence criteria: ftol = {ftol:1.0E}")
    #from .qite_function import hamiltonian_to_str
    #_ = hamiltonian_to_str(Quket.operators.Hamiltonian, True)
    #for x in _:
    #    prints(x)
    if ansatz == "inexact":
        qite_inexact(Quket, cf.nterm, cf.dimension)
    elif ansatz == "exact":
        qite_exact(Quket)
    else:
        # Anti-symmetric group
        if ansatz != "cite":
            Quket.pauli_list = [pauli*-1j for pauli in Quket.pauli_list]
        s = Quket.multiplicity - 1
        qubit_HS2 = Quket.operators.qubit_Hamiltonian +  Quket.s2shift * (Quket.operators.qubit_S2 - QubitOperator('',s*(s+1))) 
        Quket.qulacs.HS2 = create_observable_from_openfermion_text(str(qubit_HS2))

        if use_msqite:
            msqite(Quket)
        else:
            if Quket.folded_spectrum:  
                if not isfloat(Quket.shift):
                    raise ValueError("Need to set shift for excited state calculation")
                qubit_H2 = (qubit_HS2 - QubitOperator('',float(Quket.shift)))**2
                Quket.qulacs.Hamiltonian2 = create_observable_from_openfermion_text(str(qubit_H2))
                Quket.shift = 'none'
            qite(Quket)

def qite(Quket):
    """
    Perform QITE using the pauli strings stored in Quket.pauli_list.
    
    Args:
        Quket (QuketData): QuketData instance

    Returns:
        None: Information stored in Quket.

    Author(s): Yoohee Ryo, Takashi Tsuchimochi
    """
    get_cite = True
    get_cite = False
    # Parameter setting
    ansatz = Quket.ansatz
    n = Quket.n_qubits
    db = Quket.dt
    ntime = Quket.maxiter
    H = Quket.qulacs.Hamiltonian
    regularization = Quket.regularization
    if Quket.folded_spectrum:
        observable = Quket.qulacs.Hamiltonian2
    else:
        observable = Quket.qulacs.HS2
    S2_observable = Quket.qulacs.S2
    Number_observable = Quket.qulacs.Number
    threshold = Quket.ftol
    S2 = 0
    Number = 0
    use_qlanczos = Quket.qlanczos

    if Quket.ansatz == 'cite':
        prints(f"ITE")
    else:
        size = len(Quket.pauli_list)
        prints(f"QITE: Pauli operator group size = {size}")
    if ansatz != "cite":
        nonredundant_sigma, pauli_ij_index, pauli_ij_coef = Nonredundant_Overlap_list(Quket.pauli_list, n)
        len_list = len(nonredundant_sigma)
        prints(f"    Unique sigma list = {len_list}")

    index = np.arange(n)
    delta = QuantumState(n)

    energy = []
    s2 = []
    nm_list = []
    cm_list = []
    psi_dash = Quket.init_state.copy()
    nm_list.append(1)
    cm_list.append(1)
    qlanz = []

    t1 = time.time()
    cf.t_old = t1

    #En = H.get_expectation_value(psi_dash)
    En = Quket.get_E(psi_dash)
    if Quket.folded_spectrum:
        obs = observable.get_expectation_value(psi_dash)
    else:
        obs = En
    energy.append(En)
    q_en = [En]
    H_q = None
    S_q = None
    S2_q = None
    if S2_observable is not None:
        #S2 = S2_observable.get_expectation_value(psi_dash)
        S2 = Quket.get_S2(psi_dash)
    else:
        S2 = 0
    s2.append(S2)
    q_s2 = [S2]
    if Number_observable is not None:
        #Number = Number_observable.get_expectation_value(psi_dash)
        Number = Quket.get_N(psi_dash)

    ##########
    # Shift  #
    ##########
    if Quket.shift in ['hf', 'step']:
        shift = obs
        E0 = obs
    elif Quket.shift == 'true':
        if Quket.shift_value == 0:
            shift = obs
        else:
            shift = Quket.shift_value
        E0 = obs
    elif isfloat(Quket.shift):
        shift = float(Quket.shift)
    elif Quket.shift in ['none', 'false']:
        shift = 0
        E0 = 0
    else:
        raise ValueError(f"unknown shift option: {Quket.shift}")
    if ansatz == "cite":
    #    shift = En
        order = 0
    else:
        order = 1
    prints('Shift = ', shift)
    prints('QLanczos = ',use_qlanczos)
    ref_shift = shift   ## Reference shift
    dE = 100
    beta = 0
    Conv = False
    amax_ = 100

    psis = []
    for t in range(ntime):
        t2 = time.time()
        cput = t2 - cf.t_old
        cf.t_old = t2
        if cf.debug:
            print_state(psi_dash)
        prints(f"{beta:6.2f}: E = {En:.12f}  "
               f"<S**2> = {S2:+10.8f}  "
               f"<N> = {Number:10.8f}  "
               f"Fidelity = {Quket.fidelity(psi_dash):.6f}  "
               f"CPU Time = {cput: 5.2f}", end="")
        if use_qlanczos:
            prints(f"  QLanczos = (", end="")
            prints(', '.join(f'{e:.12f}' for e in q_en[:3]), end="")
            prints(f")", end="")
            if mpi.main_rank:
                print(f"{beta:6.2f}: ", end="")
                for istate in range(len(q_en)):
                    print(f"E[{istate}-QLanczos] = {q_en[istate]:.8f}  "
                          f"(<S**2> = {q_s2[istate]:+7.5f})  ",
                          end="")
                print(f"")
        prints("") 
        ### Save the QLanczos details in out


        if Quket.folded_spectrum:
            obs = observable.get_expectation_value(psi_dash)
        else:
            obs = En
        if abs(dE) < threshold:
            Conv = True
            break
        if Quket.shift == 'step':
            shift = obs
        T0 = time.time()

        ### First-order approximation for  cm ~ 1 - 2db <H>
        cm = 1 - 2*db*obs 
        cm_approximate1 = cm
        #obs2 = Quket.qulacs.H2.get_expectation_value(psi_dash)
        #obs3 = Quket.qulacs.H3.get_expectation_value(psi_dash)
        #cm_approximate2 = 1 - 2*db*obs + 2* db*db*obs2
        #cm_approximate3 = 1 - 2*db*obs + 2* db*db*obs2 - 8/6 * db*db*db*obs3
        cm_shift = cm 
        ### Better approximation for cm
        cm = np.exp(-2 * db * (obs - E0))
        cm_approximate_exp = np.exp(-2 * db * obs)
        #cm_shift = cm 

        beta += db
        T1 = time.time()

        ### CITE
        if get_cite:
            #psi_dash, norm = calc_expHpsi(psi_dash, observable, n, db, shift=shift, order=15, ref_shift=0)
            cite, norm = calc_expHpsi(psi_dash, observable, n, db, shift=0, order=0, ref_shift=0)
            #psi_dash.add_state(Hpsi)
            #print(f'<psi|psi> = {inner_product(psi_dash, psi_dash)}')
            norm = cite.get_squared_norm()
            cite_cm = norm
            #print(f'<psi|psi> = {inner_product(psi_dash, psi_dash)}  and  {cm = }')
            cite.normalize(norm)
            #prints(f'{norm=}  {np.exp(- 2 * db * obs)=}  {(1 - 2* db * obs)=}')

            test, norm = calc_expHpsi(psi_dash, observable, n, db, shift=E0, order=0, ref_shift=0)
            #prints(f'E0 shift: {norm=}  {np.exp(- 2 * db * (obs-E0))=}  {(1 - 2* db * (obs-E0))=}')
            prints('approximate_1 = ', cm_approximate1, 'new = ',cm_approximate_exp, 'exact = ',cite_cm)
            cm = norm
            cm_shift = norm

        if ansatz == "cite":
            #Hpsi = calc_Hpsi(observable, psi_dash, shift=shift, db=db, j=0)
            #psi_dash, norm = calc_expHpsi(psi_dash, observable, n, db, shift=0, order=1, ref_shift=0)
            psi_dash, norm = calc_expHpsi(psi_dash, observable, n, db, shift=E0, order=0, ref_shift=0)
            #psi_dash.add_state(Hpsi)
            #print_state(psi_dash)
            #print(f'<psi|psi> = {inner_product(psi_dash, psi_dash)}')
            norm = psi_dash.get_squared_norm()
            cm = norm
            cm_shift = norm
            #print(f'<psi|psi> = {inner_product(psi_dash, psi_dash)}  and  {cm = }')
            psi_dash.normalize(norm)
            T6 = T1
        else:
            # Compute Sij as expectation values of sigma_list
            t1=time.time()
            S = Overlap_by_nonredundant_pauli_list(nonredundant_sigma, pauli_ij_index, pauli_ij_coef, psi_dash)
            t2=time.time()
            #prints('New',t2 - t1)
            #### Debugging purpose. To be removed. 
            #t1=time.time()
            #S = Overlap_by_pauli_list(Quket.pauli_list, psi_dash, correction = True).real
            #t2=time.time()
            #prints('Old',t2 - t1)
            ##printmat(S,'S')
            ##printmat(S_,'S_')
            ##printmat(S-S_,'S-S_')
            #prints('|S-S_| ', np.linalg.norm(S - S_))
            #prints('|S-S_| ', np.linalg.norm(abs(S) - abs(S_)))
            #t3=time.time()
            Amat = 2*np.real(S) #+ damp * np.eye(size)

            T2 = time.time()

            Hpsi = calc_Hpsi(observable, psi_dash, shift=shift, db=1, j=-1)
            T3 = time.time()

            b_l = np.zeros(size, dtype=float)
            for i, pauli in enumerate(Quket.pauli_list):
                state_i = evolve(pauli, psi_dash)
                if Quket.shift == 'none':
                    b_l[i] = -2 * inner_product(state_i, Hpsi).imag / np.sqrt(cm_shift)
                # Original derivation
                else:
                    b_l[i] = -2 * inner_product(state_i, Hpsi).imag  # * cm_shift
                #    prints(inner_product(state_i, psi_dash))
                #nstates = len(Quket.lower_states)
                #for i in range(nstates):
                #    Ei = Quket.lower_states[i][0]
                #    print_state(psi_dash)
                #    overlap1 = 2 * inner_product(Quket.lower_states[i][1], psi_dash) 
                #    prints(overlap1)
                #    overlap2 =  inner_product(Quket.lower_states[i][1], state_i)
                #    prints(overlap2)
                #    overlap = overlap1 * overlap2
                #    b_l[i] += -  overlap.imag
                #    prints(Quket.get_E())
            T4 = time.time()

            #x, res, rnk, s = lstsq(Amat, -b_l, cond=1e-6)
            if regularization > 0:
                x, res, rnk, s = lstsq(Amat, -b_l, cond=1e-6, regularization=regularization)
            else:
                try:
                    x, res, rnk, s = lstsq(Amat, -b_l, cond=1e-6)
                except:
                    x =  -np.linalg.pinv(Amat, rcond=1e-6) @ b_l

            a = x.copy()
            a *= db
            if beta > 1:
                amax = max(abs(a[:]))
                if amax > amax_: 
                    a[:] *= amax_/amax
                amax_ = amax

            if cf.debug:
                prints(f"cm = {cm_shift}   norm[b] = {np.linalg.norm(b_l)}   norm[a] = {np.linalg.norm(a)}")
                printmat(b_l, name='b_l')
                printmat(a, name='a')
            # Just in case, broadcast a...
            a = mpi.bcast(a, root=0)
            T5 = time.time()

            psi_dash = create_exp_state(Quket, init_state=psi_dash, theta_list=-a)
            if get_cite:
                prints(f'|<cite|psi_dash>|^2 = {inner_product(psi_dash, cite).real ** 2}')
                cite.multiply_coef(-1)
                cite.add_state(psi_dash)
                prints(f'F(a) = {cite.get_squared_norm()}')
            if cf.debug:
                print_state(psi_dash, name='new state')

            T6 = time.time()


            if cf.debug:
                prints(f"T0 -> T1  {T1-T0}")
                prints(f"T1 -> T2  {T2-T1}")
                prints(f"T2 -> T3  {T3-T2}")
                prints(f"T3 -> T4  {T4-T3}")
                prints(f"T4 -> T5  {T5-T4}")
                prints(f"T5 -> T6  {T6-T5}")

        #En = H.get_expectation_value(psi_dash)
        En = Quket.get_E(psi_dash)
        if S2_observable is not None:
            S2 = Quket.get_S2(psi_dash)
        if Number_observable is not None:
            Number = Quket.get_N(psi_dash)
        energy.append(En)
        s2.append(S2)
        dE = energy[t+1] - energy[t]
        Quket.state = psi_dash

        T7 = time.time()
        if cf.debug:
            prints(f"<H> time  {T7-T6}")


        #nm = nm_list[t] * cm
        cm_list.append(cm)
        #nm_list.append(nm)
        if t % 2 and use_qlanczos:
            q_en, q_s2, H_q, S_q, S2_q = qlanczos(cm_list, energy, s2, t+1, H_q=H_q, S_q=S_q, S2_q=S2_q)
            T8 = time.time()
            if cf.debug:
                prints(f"QLanczos time  {T8-T7}")
    if Conv:
        prints(f"CONVERGED at beta={beta:.2f}.")
        Quket.converge = True
    else:
        prints(f"CONVERGE FAILED.")
    prints(f" Final: E[{ansatz}] = {En:.12f}  "
       f"<S**2> = {S2:+17.15f}  ")
    if use_qlanczos:
        prints(f"  QLanczos:")
        prints('\n'.join(f'     E = {e:+.12f} [<S**2> = {s:+.12f}]' for (e, s) in zip(q_en, q_s2)), end="")
        prints("")
    print_state(psi_dash, name="(QITE state)")
