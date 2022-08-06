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

jmucc.py

Multi-reference UCC.
Jeziorski-Monkhorst UCC.

"""
import time
import itertools

import numpy as np
from qulacs import QuantumCircuit
from qulacs.state import inner_product

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import (SaveTheta, print_state, prints, printmat,
                    print_amplitudes_listver)
from quket.opelib import evolve
from quket.utils import int2occ
from quket.linalg import root_inv
from quket.projection import S2Proj
from .hflib import set_circuit_rhf, set_circuit_rohf, set_circuit_uhf
from quket.opelib import single_ope_Pauli, double_ope_Pauli
from quket.lib import QuantumState
from .ucclib import create_uccsd_state


def create_kappalist(ndim1, occ_list, noa, nob, nva, nvb):
    """Function
    Create kappalist from occ_list, which stores the occupied qubits.

    Author(s):  Yuto Mori
    """
    kappa = np.zeros(ndim1)
    occ_hf = set(range(len(occ_list)))
    dup = occ_hf & set(occ_list)
    cre_set = list(set(occ_list)-dup)
    ann_set = list(occ_hf-dup)
    kappalist = []
    for c in cre_set:
        if c%2 == 0:
            for a in ann_set:
                if a%2 == 0:
                    ann_set.remove(a)
                    kappalist.append(int(a/2. + noa*(c/2. - noa)))
        else:
            for a in ann_set:
                if a% 2 == 1:
                    ann_set.remove(a)
                    kappalist.append(
                            int((a-1)/2. + nob*((c-1)/2. - nob) + noa*nva))
    for i in kappalist:
        kappa[i] = np.pi/2.
    return kappa


def create_HS2S(Quket, states):
    X_num = len(states)
    H = np.zeros((X_num, X_num), dtype=complex)
    S2 = np.zeros((X_num, X_num), dtype=complex)
    S = np.zeros((X_num, X_num), dtype=complex)
    for i in range(X_num):
        for j in range(i+1):
            H[i, j] = Quket.qulacs.Hamiltonian.get_transition_amplitude(
                    states[i], states[j])
            S2[i, j] = Quket.qulacs.S2.get_transition_amplitude(
                    states[i], states[j])
            S[i, j] = inner_product(states[i], states[j])
        H[:i, i] = H[i, :i]
        S2[:i, i] = S2[i, :i]
        S[:i, i] = S[i, :i]
    return H, S2, S


## Spin Adapted ##
def create_sa_state(n_qubits, n_electrons, noa, nob, nva, nvb, rho, DS,
                    kappa_list, theta_list, occ_list, vir_list,
                    init_state=None, threshold=1e-4):
    """Function
    Create a spin-free Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """

    if init_state is None:
        state = QuantumState(n_qubits)
        if noa == nob:
            circuit_rhf = set_circuit_rhf(n_qubits, n_electrons)
        else:
            circuit_rhf = set_circuit_rohf(n_qubits, noa, nob)
        circuit_rhf.update_quantum_state(state)
    else:
        state = init_state.copy()
    theta_list_rho = theta_list/rho
    circuit = set_circuit_sauccsdX(n_qubits, noa, nob, nva, nvb, DS,
                                   theta_list_rho, occ_list, vir_list)
    if np.linalg.norm(kappa_list) > threshold:
        circuit_uhf = set_circuit_uhf(n_qubits, noa, nob, nva, nvb, kappa_list)
        circuit_uhf.update_quantum_state(state)
    for i in range(rho):
        circuit.update_quantum_state(state)
    return state


def set_circuit_sauccsdX(n_qubits, noa, nob, nva, nvb, DS, theta_list,
                         occ_list, vir_list):
    """Function
    Prepare a Quantum Circuit for a spin-free Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ndim1 = noa*nva
    circuit = QuantumCircuit(n_qubits)
    if DS:
        ucc_sa_singlesX(circuit, theta_list, occ_list, vir_list, 0)
        ucc_sa_doublesX(circuit, theta_list, occ_list, vir_list, ndim1)
    else:
        ucc_sa_doublesX(circuit, theta_list, occ_list, vir_list, ndim1)
        ucc_sa_singlesX(circuit, theta_list, occ_list, vir_list, 0)
    return circuit


def ucc_sa_singlesX(circuit, theta_list, occ_list, vir_list, ndim2=0):
    """Function
    Prepare a Quantum Circuit for the single exictation part of a spin-free
    Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ia = ndim2
    occ_list_a = [i for i in occ_list if i%2 == 0]
    #occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    #vir_list_b = [i for i in vir_list if i%2 == 1]

    ### alpha (& beta) ###
    ncnot = 0
    for a in vir_list_a:
        for i in occ_list_a:
            single_ope_Pauli(a, i, circuit, theta_list[ia])
            single_ope_Pauli(a+1, i+1, circuit, theta_list[ia])
            ia += 1


def ucc_sa_doublesX(circuit, theta_list, occ_list, vir_list, ndim1=0):
    """Function
    Prepare a Quantum Circuit for the double exictation part of a spin-free
    Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """

    ijab = ndim1
    occ_list_a = [i for i in occ_list if i%2 == 0]
    occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    vir_list_b = [i for i in vir_list if i%2 == 1]

    ### aa or bb ##
    ncnot = 0
    for a, b in itertools.combinations(vir_list_a, 2):
        for i, j in itertools.combinations(occ_list_a, 2):
            double_ope_Pauli(b, a, j, i, circuit, theta_list[ijab])
            double_ope_Pauli(b+1, a+1, j+1, i+1, circuit, theta_list[ijab])
            ijab += 1
    ### ab ###
    no = len(occ_list_a)
    nv = len(vir_list_a)
    for b in vir_list_b:
        for a in vir_list_a:
            for j in occ_list_b:
                for i in occ_list_a:
                    b_i = vir_list_b.index(b)
                    a_i = vir_list_a.index(a)
                    j_i = occ_list_b.index(j)
                    i_i = occ_list_a.index(i)
                    baji = get_baji(b_i, a_i, j_i, i_i, no, nv)
                    double_ope_Pauli(max(b, a), min(b, a),
                                     max(j, i), min(j, i),
                                     circuit, theta_list[ijab+baji])


def get_baji(b, a, j, i, no, nv):
    """Function
    Get index for b,a,j,i

    Author(s):  Yuto Mori
    """
    nov = no*nv
    aa = i*nv + a
    bb = j*nv + b
    baji = nov*(nov-1)//2 - (nov-1-aa)*(nov-aa)//2 + bb
    return baji


def cost_jmucc(Quket, print_level, theta_lists):
    """Function
    Cost function of Jeziorski-Monkhorst UCCSD.

    Author(s):  Yuto Mori, Takashi Tsuchimochi
    """
    t1 = time.time()

    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    nocc = noa + nob
    n_electrons = Quket.n_active_electrons
    n_qubits = Quket.n_qubits
    nstates = len(Quket.multi.weights)
    ndim1 = Quket.ndim1
    ndim2 = Quket.ndim2
    ndim = Quket.ndim
    ndim_i = ndim1 + ndim2
    rho = Quket.rho
    DS = Quket.DS
    det = Quket.current_det

    if len(Quket.multi.states) != nstates:
        Quket.multi.states = []
        for istate in range(nstates):
            Quket.multi.states.append(QuantumState(n_qubits))

    occ_lists = []
    vir_lists = []
    for istate in range(nstates):
        ### Read state integer and extract occupied/virtual info
        if not isinstance(Quket.multi.init_states_info[istate], (int, np.integer)):
            raise TypeError('Model space given by `multi` section has to have only single determinants')
        occ_list_tmp = int2occ(Quket.multi.init_states_info[istate])
        vir_list_tmp = [i for i in range(n_qubits) if i not in occ_list_tmp]
        occ_lists.extend(occ_list_tmp)
        vir_lists.extend(vir_list_tmp)

    ### Prepare kappa_lists
    kappa_lists = []
    for istate in range(nstates):
        kappa_list = create_kappalist(
                ndim1, occ_lists[nocc*istate : nocc*(istate+1)],
                noa, nob, nva, nvb)
        kappa_lists.extend(kappa_list)

    ### Prepare JM basis
    states = []
    t0_ = time.time()
    for istate in range(nstates):
        det = Quket.multi.init_states_info[istate]
        cf.ncnot = 0
        state = create_uccsd_state(
                n_qubits, rho, DS,
                theta_lists[ndim_i*istate : ndim_i*(istate+1)],
                det, ndim1, mapping=Quket.cf.mapping)
        #prints(cf.ncnot)
        #error()
        if Quket.projection.SpinProj:
            state = S2Proj(Quket, state)
        states.append(state)
    t1_ = time.time()
    H, S2, S = create_HS2S(Quket, states)
    t2_ = time.time()
    if cf.debug and print_level == 1:
        prints(f'State preparation time = {t1_-t0_:.2f} s')
        prints(f'H, S    formation time = {t2_-t1_:.2f} s')
    root_invS = root_inv(S.real)
    H_ortho = root_invS.T@H@root_invS
    nstates0 = root_invS.shape[1]
    #prints(nstates0)

    en, dvec = np.linalg.eigh(H_ortho)
    ind = np.argsort(en.real, -1)
    en = en.real[ind]
    dvec = dvec[:, ind]
    cvec = root_invS@dvec

    # Renormalize
    # Compute <S**2> of each state
    S2dig = cvec.T@S2@cvec
    s2 = []
    for istate in range(nstates0):
        s2.append(S2dig[istate, istate].real)

    # compute <S**2> for each state
    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        if cf.debug and cf.grad > 1e-2:
            for istate in range(nstates): 
                string = "State "+str(istate)+" large t-amplitudes (> 0.01)"
                print_amplitudes_listver(theta_lists[ndim_i*istate : ndim_i*(istate+1)], 
                                         noa, nob, nva, nvb, 
                                         occ_lists[nocc*istate : nocc*(istate+1)], 
                                         1e-2, name=string)
                string = "State "+str(istate)+" large gradients (> 0.01)"
                print_amplitudes_listver(cf.gradv[ndim_i*istate : ndim_i*(istate+1)], 
                                         noa, nob, nva, nvb, 
                                         occ_lists[nocc*istate : nocc*(istate+1)], 
                                         1e-2, name=string)

        string = f"{cf.icyc:6d}: "
        for istate in range(nstates0):
            prints(f"{string} E[{istate}-JM-UCC] = {en[istate]:.8f}  "
                   f"(<S**2> = {s2[istate]:+7.5f})  ",
                   end="")
            string = f"\n        "
        prints(f"  Grad = {cf.grad:4.2e}  "
               f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        SaveTheta(ndim, theta_lists, cf.tmp)
        Quket.theta_list = theta_lists.copy()
    if print_level > 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        string = " Final: "
        for istate in range(nstates0):
            prints(f"{string} E[{istate}-JM-UCC] = {en[istate]:.8f}  "
                   f"(<S**2> = {s2[istate]:+7.5f})  ",
                   end="")
            string = f"\n        "
        prints(f"  CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        prints("\n------------------------------------")
        for istate in range(nstates):
            print_state(states[istate], name=f"JM Basis   {istate}")
            print_amplitudes_listver(theta_lists[ndim_i*istate : ndim_i*(istate+1)], 
                                     noa, nob, nva, nvb, 
                                     occ_lists[nocc*istate : nocc*(istate+1)], 
                                     1e-3)
        printmat(cvec.real, name="Coefficients:")
        prints("------------------------------------\n\n")
        prints("###############################################")
        prints("#                  JM states                  #")
        prints("###############################################", end="")

        for istate in range(nstates0):
            prints()
            prints(f"State        : {istate}")
            prints(f"E            : {en[istate]:.8f}")
            prints(f"<S**2>       : {s2[istate]:+.5f}")
            spstate = QuantumState(n_qubits)
            spstate.multiply_coef(0)
            for jstate in range(nstates0):
                state = states[jstate].copy()
                coef = cvec[jstate, istate]
                state.multiply_coef(coef)
                spstate.add_state(state)
            Quket.multi.states[istate] = spstate
            print_state(spstate, name="Superposition:")
            if istate == 0:
                Quket.state = spstate.copy()
                Quket.energy = Quket.get_E()
        prints("###############################################")

    cost = np.sum(Quket.multi.weights[:nstates0]*en)
    norm = np.sum(Quket.multi.weights[:nstates0])
    cost /= norm
    return cost, s2


def deriv_jmucc(Quket, theta_lists):
    """Function
    Derivative function of Jeziorski-Monkhorst UCCSD.
    Because 
    Ei = sum_kl (Cki Hkl Cli) / sum_kl (Cki Skl Cli)
    its derivative is
    dEi/dt = sum_kl Cki (dHkl/dt - E dSkl/dt) Cli 

    dHkl/dt = <dk/dt | H | l>  + <k | H | dl/dt> 

    Author(s):  Yuto Mori, Takashi Tsuchimochi
    """
    t0 = time.time()

    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    nocc = noa + nob
    n_electrons = Quket.n_active_electrons
    n_qubits = Quket.n_qubits
    nstates = len(Quket.multi.weights)
    ndim1 = Quket.ndim1
    ndim2 = Quket.ndim2
    ndim = Quket.ndim
    ndim_i = ndim1 + ndim2
    rho = Quket.rho
    DS = Quket.DS
    det = Quket.current_det

    occ_lists = []
    vir_lists = []
    for istate in range(nstates):
        ### Read state integer and extract occupied/virtual info
        occ_list_tmp = int2occ(Quket.multi.init_states_info[istate])
        vir_list_tmp = [i for i in range(n_qubits) if i not in occ_list_tmp]
        occ_lists.extend(occ_list_tmp)
        vir_lists.extend(vir_list_tmp)

    ### Prepare kappa_lists
    kappa_lists = []
    for istate in range(nstates):
        kappa_list = create_kappalist(
                ndim1, occ_lists[nocc*istate : nocc*(istate+1)],
                noa, nob, nva, nvb)
        kappa_lists.extend(kappa_list)

    t1 = time.time()
    ### Prepare JM basis
    states = []
    t0_ = time.time()
    for istate in range(nstates):
        det = Quket.multi.init_states_info[istate]
        cf.ncnot = 0
        state = create_uccsd_state(
                n_qubits, rho, DS,
                theta_lists[ndim_i*istate : ndim_i*(istate+1)],
                det, ndim1, mapping=Quket.cf.mapping)
        if Quket.projection.SpinProj:
            state = S2Proj(Quket, state)
        states.append(state)
    t2 = time.time()

    ### Prepare Hstates
    Hstates = []
    for istate in range(nstates):
        Hstate = evolve(Quket.operators.qubit_Hamiltonian, states[istate], parallel=True)
        Hstates.append(Hstate)
    t3 = time.time()

    ### Create H and S
    X_num = len(states)
    H = np.zeros((X_num, X_num), dtype=complex)
    S = np.zeros((X_num, X_num), dtype=complex)
    for i in range(X_num):
        for j in range(i+1):
            H[i, j] =  inner_product(states[i], Hstates[j])
            S[i, j] = inner_product(states[i], states[j])
        H[:i, i] = H[i, :i]
        S[:i, i] = S[i, :i]
    t4 = time.time()

    root_invS = root_inv(S.real)
    H_ortho = root_invS.T@H@root_invS
    nstates0 = root_invS.shape[1]

    en, dvec = np.linalg.eigh(H_ortho)
    ind = np.argsort(en.real, -1)
    en = en.real[ind]
    dvec = dvec[:, ind]
    cvec = root_invS@dvec
    cost = np.sum(Quket.multi.weights[:nstates0]*en)
    norm = np.sum(Quket.multi.weights[:nstates0])
    cost /= norm
    t5 = time.time()


    ### Derivatives dHkl/dt and dSkl/dt
    en_d = np.zeros(nstates0)
    my_grad = np.zeros(ndim)
    grad = np.zeros(ndim)
    ipos, my_ndim = mpi.myrange(ndim)

    iloop = 0
    stepsize=1e-8
    #printmat(H,name='H')
    #printmat(S,name='S')
    H_d = np.zeros((X_num, X_num), dtype=complex)
    S_d = np.zeros((X_num, X_num), dtype=complex)
    for istate in range(nstates):
        for itheta in range(ndim_i):
            if iloop >= ipos and iloop < ipos+my_ndim:
                # take d[istate]/d[itheta]
                H_d[:,:] = 0
                S_d[:,:] = 0
                det = Quket.multi.init_states_info[istate]
                theta_lists[ndim_i*istate + itheta] += stepsize
                #if iloop in [25, 101, 209]:
                #    print_level = 2
                #else:
                #    print_level = 2
                #cost_p, dum = cost_jmucc(Quket, print_level, theta_lists)
                #numerical = (cost_p - cost)/stepsize
                dstate_i = create_uccsd_state(
                        n_qubits, rho, DS,
                        theta_lists[ndim_i*istate : ndim_i*(istate+1)],
                        det, ndim1, mapping=Quket.cf.mapping)
                theta_lists[ndim_i*istate + itheta] -= stepsize
                if Quket.projection.SpinProj:
                    dstate_i = S2Proj(Quket, dstate_i)
                #H_test = np.zeros((X_num, X_num), dtype=complex)
                #S_test = np.zeros((X_num, X_num), dtype=complex)
                for jstate in range(nstates):
                    H_d[istate, jstate] = inner_product(dstate_i, Hstates[jstate]) - H[istate,jstate]
                #    H_test[istate, jstate] = inner_product(dstate_i, Hstates[jstate])
                    S_d[istate, jstate] = inner_product(dstate_i, states[jstate]) - S[istate,jstate]
                #    S_test[istate, jstate] = inner_product(dstate_i, states[jstate])
                    H_d[jstate, istate] = H_d[istate, jstate] 
                    S_d[jstate, istate] = S_d[istate, jstate]
                    if jstate == istate:
                    ### For dHii/dt where t belongs to istate, we have to double the number 
                    ###    d<istate|H|istate>/d[itheta] = <d[istate]/dt |H| istate>  + <istate |H| d[istate]/dt>
                        H_d[istate, istate] *= 2 
                        S_d[istate, istate] *= 2
                #    H_test[jstate, istate] = H_test[istate, jstate] 
                #    S_test[jstate, istate] = S_test[istate, jstate]
                #print_state(dstate_i,name='dstate_i',threshold=1e-6)
                #prints(inner_product(dstate_i, Hstates[istate]).real)
                #H_test[istate, istate] = inner_product(dstate_i, Hstates[istate]) 
                #S_test[istate, istate] = inner_product(dstate_i, states[istate]) 
                H_d /=stepsize
                S_d /=stepsize
                #prints(f'\n\n{iloop=}   {numerical=}')
                #printmat(H_d,name='H_d')
                #printmat(S_d,name='S_d')
                #printmat(H_test,name='H_test')
                #printmat(S_test,name='S_test')
                ### Derivative of Ei for nstates0
                for jstate in range(nstates0):
                    en_d[jstate] = (cvec.T@(H_d - en[jstate]*S_d)@cvec)[jstate,jstate]
                deriv = np.sum(Quket.multi.weights[:nstates0]*en_d)
                norm = np.sum(Quket.multi.weights[:nstates0])
                my_grad[iloop] = deriv/norm
                
            iloop += 1

    t6 = time.time()
    grad = mpi.allreduce(my_grad, mpi.MPI.SUM)
    cf.grad = np.linalg.norm(grad)
    cf.gradv = np.copy(grad)
    t7 = time.time()
    if cf.debug:
        prints(f'Timing in deriv_jmucc: {t7-t0}')
        prints(f'------------------------------')
        prints(f'    Parameter setting: {t1-t0}')
        prints(f'   JM state evolution: {t2-t1}')
        prints(f'          H evolution: {t3-t2}')
        prints(f'       Create H and S: {t4-t3}')
        prints(f'         Solve Hc=ScE: {t5-t4}')
        prints(f'          Derivatives: {t6-t5}')
        prints(f'            Allreduce: {t7-t6}')
    return grad


