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
import time
import numpy as np
from qulacs import QuantumCircuit

from quket import config as cf
from quket.fileio import (SaveTheta, print_state, prints, print_amplitudes)
from .ucclib import (ucc_singles, ucc_singles_g, set_circuit_uccsd,
                     set_circuit_uccd, set_circuit_sauccd)
from quket.projection import(
    controlled_Ug_gen
    )
from quket.lib import QuantumState
def set_circuit_rhfZ(n_Pqubits, n_electrons):
    """Function:
    Construct circuit for RHF |0000...1111> with one ancilla

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_Pqubits)
    for i in range(n_electrons):
        circuit.add_X_gate(i)
    return circuit


def set_circuit_rohfZ(n_Pqubits, noa, nob):
    """Function:
    Construct circuit for ROHF |0000...10101111> with one ancilla

    Author(s): Takashi Tsuchimochi
    """
    # generate circuit for rhf
    circuit = QuantumCircuit(n_Pqubits)
    for i in range(noa):
        circuit.add_X_gate(2*i)
    for i in range(nob):
        circuit.add_X_gate(2*i + 1)
    return circuit


def set_circuit_uhfZ(n_Pqubits, noa, nob, nva, nvb, theta_list):
    """Function:
    Construct circuit for UHF with one ancilla

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_Pqubits)
    ucc_singles(circuit, noa, nob, nva, nvb, theta_list)
    return circuit


def set_circuit_ghfZ(n_Pqubits, no, nv, theta_list):
    """Function:
    Construct circuit for GHF with one ancilla

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_Pqubits)
    ucc_singles_g(circuit, no, nv, theta_list)
    return circuit

def cost_proj(Quket, print_level, qulacs_hamiltonianZ, qulacs_s2Z,
              coef0_H, coef0_S2, kappa_list,
              theta_list=0, threshold=0.01):
    """Function:
    Energy functional for projected methods (phf, puccsd, puccd, opt_puccd)

    Author(s): Takashi Tsuchimochi
    """
    t1 = time.time()

    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    n_electrons = Quket.n_active_electrons
    rho = Quket.rho
    DS = Quket.DS
    anc = Quket.anc
    n_qubits = Quket.n_qubits
    n_Pqubits = Quket.n_qubits + 1
# opt_psauccdとかはndimの計算が異なるけどこっちを使う?
    #ndim1 = noa*nva + nob*nvb
    #ndim2aa = noa*(noa-1)*nva*(nva-1)//4
    #ndim2ab = noa*nob*nva*nvb)
    #ndim2bb = nob*(nob-1)*nvb*(nvb-1)//4
    #ndim2 = ndim2aa + ndim2ab + ndim2bb
    ndim1 = Quket.ndim1
    ndim2 = Quket.ndim2
    ndim = Quket.ndim
    ref = Quket.ansatz

    #state = QuantumState(n_Pqubits)
    #if noa == nob:
    #    circuit_rhf = set_circuit_rhfZ(n_Pqubits, n_electrons)
    #else:
    #    circuit_rhf = set_circuit_rohfZ(n_Pqubits, noa, nob)
    #circuit_rhf.update_quantum_state(state)
    state = Quket.init_state.copy()

    if ref == "phf":
        circuit_uhf = set_circuit_uhfZ(n_Pqubits, noa, nob, nva, nvb, kappa_list)
        circuit_uhf.update_quantum_state(state)
    elif ref == "sghf":
        circuit_ghf = set_circuit_ghfZ(n_Pqubits, noa+nob, nva+nvb, kappa_list)
        circuit_ghf.update_quantum_state(state)
    elif ref == "puccsd":
        # First prepare UHF determinant
        circuit_uhf = set_circuit_uhfZ(n_Pqubits, noa, nob, nva, nvb, kappa_list)
        circuit_uhf.update_quantum_state(state)
        # Then prepare UCCSD
        theta_list_rho = theta_list/rho
        circuit = set_circuit_uccsd(n_Pqubits, noa, nob, nva, nvb, 0,
                                    theta_list_rho, ndim1)
        for i in range(rho):
            circuit.update_quantum_state(state)
    elif ref == "puccd":
        # First prepare UHF determinant
        circuit_uhf = set_circuit_uhfZ(n_Pqubits, noa, nob, nva, nvb, kappa_list)
        circuit_uhf.update_quantum_state(state)
        # Then prepare UCCD
        theta_list_rho = theta_list/rho
        circuit = set_circuit_uccd(n_Pqubits, noa, nob, nva, nvb, theta_list_rho)
        for i in range(rho):
            circuit.update_quantum_state(state)
    elif ref == "opt_puccd":
        if DS:
            # First prepare UHF determinant
            circuit_uhf = set_circuit_uhfZ(n_Pqubits, noa, nob, nva, nvb,
                                           theta_list)
            circuit_uhf.update_quantum_state(state)
            # Then prepare UCCD
            theta_list_rho = theta_list[ndim1:]/rho
            circuit = set_circuit_uccd(n_Pqubits, noa, nob, nva, nvb,
                                       theta_list_rho)
            for i in range(rho):
                circuit.update_quantum_state(state)
        else:
            # First prepare UCCD
            theta_list_rho = theta_list[ndim1:]/rho
            circuit = set_circuit_uccd(n_Pqubits, noa, nob, nva, nvb,
                                       theta_list_rho)
            for i in range(rho):
                circuit.update_quantum_state(state)
            # then rotate
            circuit_uhf = set_circuit_uhfZ(n_Pqubits, noa, nob, nva, nvb, theta_list)
            circuit_uhf.update_quantum_state(state)
    elif ref == "opt_psauccd":
# ここが問題
# ndim2が他と異なる
        #theta_list_rho = theta_list[ndim1 : ndim1+ndim2]/rho
        theta_list_rho = theta_list[ndim1:]/rho
        circuit = set_circuit_sauccd(n_Pqubits, noa, nva, theta_list_rho)
        for i in range(rho):
            circuit.update_quantum_state(state)
        circuit_uhf = set_circuit_uhfZ(n_Pqubits, noa, nob, nva, nvb, theta_list)
        circuit_uhf.update_quantum_state(state)

    if print_level > 0:
        if ref in ("uhf", "phf", "suhf", "sghf"):
            SaveTheta(ndim, kappa_list, cf.tmp)
            Quket.kappa_list = kappa_list.copy()
        else:
            SaveTheta(ndim, theta_list, cf.tmp)
            Quket.theta_list = theta_list.copy()
    if print_level > 1:
        prints("State before projection")
        print_state(state, n_qubits=n_qubits)
        if ref in ("puccsd", "opt_puccd"):
            print_amplitudes(theta_list, noa, nob, nva, nvb, threshold)

    ### grid loop ###
    ### a list to compute the probability to observe 0 in ancilla qubit
    ### Array for <HUg>, <S2Ug>, <Ug>
    Ep = S2 = Norm = 0
    nalpha = max(Quket.projection.euler_ngrids[0], 1)
    nbeta = max(Quket.projection.euler_ngrids[1], 1)
    ngamma = max(Quket.projection.euler_ngrids[2], 1)
    HUg = np.zeros(nalpha*nbeta*ngamma)
    S2Ug = np.zeros(nalpha*nbeta*ngamma)
    Ug = np.zeros(nalpha*nbeta*ngamma)
    ig = 0
    for ialpha in range(nalpha):
        alpha = Quket.projection.sp_angle[0][ialpha]
        alpha_coef = Quket.projection.sp_weight[0][ialpha]

        for ibeta in range(nbeta):
            beta = Quket.projection.sp_angle[1][ibeta]
            beta_coef = (Quket.projection.sp_weight[1][ibeta]
                        *Quket.projection.dmm[ibeta])

            for igamma in range(ngamma):
                gamma = Quket.projection.sp_angle[2][igamma]
                gamma_coef = Quket.projection.sp_weight[2][igamma]

                ### Copy quantum state of UHF (cannot be done in real device) ###
                state_g = QuantumState(n_Pqubits)
                state_g.load(state)
                ### Construct Ug test
                circuit_ug = QuantumCircuit(n_Pqubits)
                ### Hadamard on anc
                circuit_ug.add_H_gate(anc)
                #circuit_ug.add_X_gate(anc)
                #controlled_Ug(circuit_ug, n_qubits, anc, beta)
                controlled_Ug_gen(circuit_ug, n_qubits, anc, alpha, beta, gamma)
                #circuit_ug.add_X_gate(anc)
                circuit_ug.add_H_gate(anc)
                circuit_ug.update_quantum_state(state_g)

                ### Compute expectation value <HUg> ###
                HUg[ig] = qulacs_hamiltonianZ.get_expectation_value(state_g)
                ### <S2Ug> ###
                # print_state(state_g)
                S2Ug[ig] = qulacs_s2Z.get_expectation_value(state_g)
                ### <Ug> ###
                p0 = state_g.get_zero_probability(anc)
                p1 = 1 - p0
                Ug[ig] = p0 - p1

                ### Norm accumulation ###
                Norm += alpha_coef*beta_coef*gamma_coef*Ug[ig]
                Ep += alpha_coef*beta_coef*gamma_coef*HUg[ig]
                S2 += alpha_coef*beta_coef*gamma_coef*S2Ug[ig]
                ig += 1
    #        print('p0 : ',p0,'  p1 : ',p1,  '  p0 - p1 : ',p0-p1)
    #    print("Time: ",t2-t1)
    ### Energy calculation <HP>/<P> and <S**2P>/<P> ###
    Ep /= Norm
    S2 /= Norm
    Ep += coef0_H
    S2 += coef0_S2

    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == -1:
        prints(f"Initial E[{ref}] = {Ep:.12f}  <S**2> = {S2:+17.15f}  "
                            f"rho = {rho}")
    elif print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(f"{cf.icyc:6d}: E[{ref}] = {Ep:.12f}  <S**2> = {S2:+17.15f}  "
               f"Grad = {cf.grad:4.2e}  "
                f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
    elif print_level > 1:
        prints(f" Final: E[{ref}] = {Ep:.12f}  <S**2> = {S2:+17.15f}  "
                           f"rho = {rho}")
        print_state(state, n_qubits=n_qubits)
        if ref in ("puccsd", "opt_puccd"):
            print_amplitudes(theta_list, noa, nob, nva, nvb)
        prints("HUg", HUg)
        prints("Ug", Ug)
    return Ep, S2
