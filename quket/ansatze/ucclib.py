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

ucclib.py

Functions preparing UCC-type gates and circuits.
Cost functions are also defined here.

"""
import time
import itertools
import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import PauliRotation
from qulacs.state import inner_product

from quket import config as cf
from quket.fileio import (SaveTheta, print_state, print_amplitudes,
                     print_amplitudes_spinfree, print_amplitudes_listver, 
                     prints)
from quket.opelib import Gdouble_ope, set_exp_circuit, create_exp_state, single_ope_Pauli, double_ope_Pauli
from quket.utils import orthogonal_constraint, get_occvir_lists, get_unique_list, transform_state_jw2bk
from quket.projection import S2Proj
from quket.lib import QuantumState
from .hflib import set_circuit_rhf

def ucc_Gsingles(circuit, norbs, theta_list, ndim2=0):
    """Function:
    Construct circuit for generalized singles
           prod_pq exp[theta (p!q - q!p)]

    Author(s): Yuto Mori
    """
    ia = ndim2
    ### alpha ###
    for a in range(norbs):
        a2 = 2*a
        for i in range(a):
            i2 = 2*i
            single_ope_Pauli(a2, i2, circuit, theta_list[ia])
            ia += 1
    ### beta ###
    for a in range(norbs):
        a2 = 2*a + 1
        for i in range(a):
            i2 = 2*i + 1
            single_ope_Pauli(a2, i2, circuit, theta_list[ia])
            ia += 1

def ucc_singles(circuit, noa, nob, nva, nvb, theta_list, ndim2=0):
    """Function:
    Construct circuit for UCC singles
            prod_ai exp[theta (a!i - i!a)]

    Author(s): Yuto Mori
    """
    ia = ndim2
    ### alpha ###
    for i in range(noa):
        i2 = 2*i
        for a in range(nva):
            a2 = 2*(a+noa)
            single_ope_Pauli(a2, i2, circuit, theta_list[ia])
            ia += 1
    ### beta ###
    for i in range(nob):
        i2 = 2*i + 1
        for a in range(nvb):
            a2 = 2*(a+nob) + 1
            single_ope_Pauli(a2, i2, circuit, theta_list[ia])
            ia += 1


def ucc_doubles(circuit, noa, nob, nva, nvb, theta_list, ndim1=0):
    """Function:
    Construct circuit for UCC doubles
            prod_abij exp[theta (a!b!ji - i!j!ba)]

    Author(s): Yuto Mori
    """
    ### aa -> aa ###
    ijab = ndim1
    for b in range(nva):
        b2 = 2*(b+noa)
        for a in range(b):
            a2 = 2*(a+noa)
            for j in range(noa):
                j2 = 2*j
                for i in range(j):
                    i2 = 2*i
                    double_ope_Pauli(b2, a2, j2, i2, circuit, theta_list[ijab])
                    ijab += 1
    ### ab -> ab ###
    for b in range(nvb):
        b2 = 2*(b+nob) + 1
        for a in range(nva):
            a2 = 2*(a+noa)
            for j in range(nob):
                j2 = 2*j + 1
                for i in range(noa):
                    # b > a, j > i
                    i2 = 2*i
                    double_ope_Pauli(max(b2, a2), min(b2, a2),
                                     max(j2, i2), min(j2, i2),
                                     circuit, theta_list[ijab])
                    ijab += 1
    ### bb -> bb ###
    for b in range(nvb):
        b2 = 2*(b+nob) + 1
        for a in range(b):
            a2 = 2*(a+nob) + 1
            for j in range(nob):
                j2 = 2*j + 1
                for i in range(j):
                    i2 = 2*i + 1
                    double_ope_Pauli(b2, a2, j2, i2, circuit, theta_list[ijab])
                    ijab += 1


def set_circuit_GS(n_qubits, noa, nob, nva, nvb, theta1):
    """Function:
    Construct new circuit for generalized singles,  prod_pq exp[theta (p!q - q!p)]

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    ucc_Gsingles(circuit, norbs, theta1)
    return circuit

def set_circuit_uccsd(n_qubits, noa, nob, nva, nvb, DS, theta_list, ndim1):
    """Function:
    Construct new circuit for UCCSD

    Author(s): Yuto Mori, Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    if DS:
        ucc_singles(circuit, noa, nob, nva, nvb, theta_list, 0)
        ucc_doubles(circuit, noa, nob, nva, nvb, theta_list, ndim1)
    else:
        ucc_doubles(circuit, noa, nob, nva, nvb, theta_list, ndim1)
        ucc_singles(circuit, noa, nob, nva, nvb, theta_list, 0)
    return circuit


#############################
#   Spin-free UCC modules   #
#############################
def get_baji(b, a, j, i, no):
    """Function:
    Search the position for baji in the spin-adapted index

    Author(s): Takashi Tsuchimochi
    """
    bj = b*no + j
    ai = a*no + i
    if bj > ai:
        baji = bj*(bj+1)//2 + ai
    else:
        baji = ai*(ai+1)//2 + bj
    return baji




def set_circuit_uccsdX(n_qubits, DS, theta_list, occ_list, vir_list, ndim1):
    """Function
    Prepare a Quantum Circuit for a UCC state
    from an arbitrary determinant specified by occ_list and vir_list.

    Author(s):  Yuto Mori
    """
    circuit = QuantumCircuit(n_qubits)
    if DS:
        ucc_singlesX(circuit, theta_list, occ_list, vir_list, 0)
        ucc_doublesX(circuit, theta_list, occ_list, vir_list, ndim1)
    else:
        ucc_doublesX(circuit, theta_list, occ_list, vir_list, ndim1)
        ucc_singlesX(circuit, theta_list, occ_list, vir_list, 0)
    return circuit

def ucc_singlesX(circuit, theta_list, occ_list, vir_list, ndim2=0):
    """Function
    Prepare a Quantum Circuit for the single exictation part of
    a Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ia = ndim2
    occ_list_a = [i for i in occ_list if i%2 == 0]
    occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    vir_list_b = [i for i in vir_list if i%2 == 1]

    ### alpha ###
    for a in vir_list_a:
        for i in occ_list_a:
            single_ope_Pauli(a, i, circuit, theta_list[ia])
            ia += 1

    ### beta ###
    for a in vir_list_b:
        for i in occ_list_b:
            single_ope_Pauli(a, i, circuit, theta_list[ia])
            ia += 1

def ucc_doublesX(circuit, theta_list, occ_list, vir_list, ndim1=0):
    """Function
    Prepare a Quantum Circuit for the double exictation part of
    a Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ijab = ndim1
    occ_list_a = [i for i in occ_list if i%2 == 0]
    occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    vir_list_b = [i for i in vir_list if i%2 == 1]

    ### aa -> aa ###
    for a, b in itertools.combinations(vir_list_a, 2):
        for i, j in itertools.combinations(occ_list_a, 2):
            if b > j:
                Gdouble_ope(b, a, j, i, circuit, theta_list[ijab])
            else:
                Gdouble_ope(j, i, b, a, circuit, theta_list[ijab])
            ijab += 1

    ### ab -> ab ###
    for b in vir_list_b:
        for a in vir_list_a:
            for j in occ_list_b:
                for i in occ_list_a:
                    if max(b, a) > max(j, i):
                        Gdouble_ope(max(b, a), min(b, a), max(j, i), min(j, i),
                                    circuit, theta_list[ijab])
                    else:
                        Gdouble_ope(max(j, i), min(j, i), max(b, a), min(b, a),
                                    circuit, theta_list[ijab])
                    ijab += 1

    ### bb -> bb ###
    for a, b in itertools.combinations(vir_list_b, 2):
        for i, j in itertools.combinations(occ_list_b, 2):
            if b > j:
                Gdouble_ope(b, a, j, i, circuit, theta_list[ijab])
            else:
                Gdouble_ope(j, i, b, a, circuit, theta_list[ijab])
            ijab += 1


def create_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1,
                       init_state=None, mapping="jordan_wigner"):
    """Function
    Prepare a UCC state based on theta_list.
    The initial determinant 'det' contains the base-10 integer
    specifying the bit string for occupied orbitals.

    Author(s):  Yuto Mori, Takashi Tsuchimochi
    """
    ### Form RHF bits
    if init_state is None:
        state = QuantumState(n_qubits)
        state.set_computational_basis(det)
        if mapping == "bravyi_kitaev":
           state = transform_state_jw2bk(state)
    else:
        state = init_state.copy()
    occ_list, vir_list = get_occvir_lists(n_qubits, det)

    theta_list_rho = theta_list/rho
    circuit = set_circuit_uccsdX(n_qubits, DS,
                                 theta_list_rho, occ_list, vir_list, ndim1)
    for i in range(rho):
        circuit.update_quantum_state(state)
    return state


def ucc_singles_g(circuit, no, nv, theta_list, ndim2=0):
    """Function:
    Construct circuit for UCC singles in the spin-generalized
            prod_ai exp[theta(a!i - i!a)]

    Author(s): Takashi Tsuchimochi
    """
    ia = ndim2
    ### alpha ###
    for a in range(nv):
        for i in range(no):
            #single_ope(a2, i2, circuit, theta_list[ia])
            single_ope_Pauli(a+no, i, circuit, theta_list[ia])
            ia += 1

### Obsolete functions but used in phflib ###
def set_circuit_uccsd(n_qubits, noa, nob, nva, nvb, DS, theta_list, ndim1):
    """Function:
    Construct new circuit for UCCSD

    Author(s): Yuto Mori, Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    if DS:
        ucc_singles(circuit, noa, nob, nva, nvb, theta_list, 0)
        ucc_doubles(circuit, noa, nob, nva, nvb, theta_list, ndim1)
    else:
        ucc_doubles(circuit, noa, nob, nva, nvb, theta_list, ndim1)
        ucc_singles(circuit, noa, nob, nva, nvb, theta_list, 0)
    return circuit


def set_circuit_sauccd(n_qubits, no, nv, theta_list):
    """Function:
    Construct new circuit for spin-adapted UCCD

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    ucc_doubles_spinfree1(circuit, no, no, nv, nv, theta_list, 0)
    return circuit


def set_circuit_uccd(n_qubits, noa, nob, nva, nvb, theta_list):
    """Function:
    Construct new circuit for UCCD

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    ucc_doubles(circuit, noa, nob, nva, nvb, theta_list)
    return circuit

def count_CNOT_ucc(Quket, theta_list): 
    """
    Number of CNOT gates for UCCSD assuming the c-Ry gate.
    """
    if Quket.ncnot_list is None:
        return 0
    ncnot = 0
    ncnot_ = 0
    for k, pauli in enumerate(Quket.pauli_list):
        if abs(theta_list[k]) > cf.theta_threshold:
            target_list = []
            for op, coef in pauli.terms.items():
                target = []
                lowest = 100
                highest = 0
                for op_ in op:
                    m = op_[0]
                    lowest = min(m,lowest) 
                    highest = max(m,highest) 
                    if op_[1] == 'X' or op_[1] == 'Y':
                        target.append(m)
                target.append([lowest, highest])
                target_list.append(target)
            target_list = get_unique_list(target_list)
            #prints(target_list)
            #prints("\n")
            if len(target_list) == 1:
                # Doubles
                target_list[0].pop(-1)
            elif len(target_list) == 2:
                target_list_ = []
                target_list_.append(target_list[0][:-1])
                target_list_.append(target_list[1][:-1])
                target_list = target_list_
            elif len(target_list) == 3:
                target_list_ = []
                if target_list[0][:-1]== target_list[1][:-1]:
                    target_list_.append(target_list[1][-1])
                    target_list_.append(target_list[2][-1])
                elif target_list[1][:-1]== target_list[2][:-1]:
                    target_list_.append(target_list[0][-1])
                    target_list_.append(target_list[2][-1])
                else:
                    error('strange')
                target_list = target_list_
            elif len(target_list) == 4:
                target_list_ = []
                target_list_.append(target_list[1][-1])
                target_list_.append(target_list[3][-1])
                target_list = target_list_


            for target in target_list:
                if len(target) == 2:
                    # Singles
                    nsf = target[1] - target[0] 
                    ncnot += 2*nsf + 1
                elif len(target) == 4:
                    # Doubles
                    nsf = target[3] - target[2] + target[1] - target[0] 
                    ncnot += 2*nsf + 9
            ncnot += Quket.ncnot_list[k]
    return ncnot

def cost_exp(Quket, print_level, theta_list, parallel=True):
    """Function:
    Energy functional of general exponential ansatz. 
    Generalized to sequential excited state calculations,
    by projecting out lower_states.

    Author(s): Takashi Tsuchimochi
    """
    t1 = time.time()

    ansatz = Quket.ansatz
    rho = Quket.rho
    DS = Quket.DS
    n_qubits = Quket.n_qubits
    ndim = Quket.ndim
    init_state = Quket.init_state

    state = create_exp_state(Quket, init_state =init_state, theta_list=theta_list, rho=Quket.rho)
    #cf.ncnot = count_CNOT_ucc(Quket, theta_list)
    if Quket.projection.SpinProj:
        Quket.state_unproj = state.copy()
        state = S2Proj(Quket, state)

    t2 = time.time()
    # Store the current wave function
    Quket.state = state
    Energy = Quket.get_E(parallel=parallel)
    if Quket.operators.S2 is not None:
        S2 = Quket.get_S2(parallel=parallel)
    else:
        S2 = 0
    
    #prints(f'time for inner_prodcut: {t_inner - t_get}')

    cost = Energy
    t3 = time.time()

    ### Project out the states contained in 'lower_states'
    cost += orthogonal_constraint(Quket, state)

    if Quket.constraint_lambda > 0:
        s = (Quket.spin - 1)/2
        S4 = Quket.qulacs.S4.get_expectation_value(state)
        penalty = Quket.constraint_lambda*(S4 - S2*(s*(s+1) + (s*(s+1))**2))
        cost += penalty

    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == -1:
        prints(f"Initial: E[{ansatz}] = {Energy:+.12f}  "
               f"<S**2> = {S2:+8.6f}  ", end='')
        if Quket.fci_states is not None:
            prints(f"Fidelity = {Quket.fidelity():.6f}  ", end='')
        prints(f"rho = {rho}  ")
               #f"CNOT = {cf.ncnot}")
    if print_level == 1:
        ## cf.constraint_lambda *= 1.1
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(f"{cf.icyc:7d}: E[{ansatz}] = {Energy:+.12f}  "
               f"<S**2> = {S2:+8.6f}  ", end='')
        if Quket.fci_states is not None:
            prints(f"Fidelity = {Quket.fidelity():.6f}  ", end='')
        prints(f"Grad = {cf.grad:4.2e}  "
               #f"CNOT = {cf.ncnot}  "
               f"CPU Time = {cput:10.5f}  ({cpu1:2.2f} / step)")
        if Quket.constraint_lambda != 0:
            prints(f"lambda = {Quket.constraint_lambda}  "
                   f"<S**4> = {S4:17.15f}  "
                   f"Penalty = {penalty:2.15f}")
        SaveTheta(ndim, theta_list, cf.tmp)
        Quket.theta_list = theta_list.copy()
    if print_level > 1:
        istate = len(Quket.lower_states)
        if istate > 0:
            state_str = str(istate) + '-'
        else:
            state_str = ''
        prints(f"  Final: E[{state_str}{ansatz}] = {Energy:+.12f}  "
               f"<S**2> = {S2:+8.6f}  ", end='')
        if Quket.fci_states is not None:
            prints(f"Fidelity = {Quket.fidelity():.6f}  ", end='')
        prints(f"rho = {rho} ")
               #f"CNOT = {cf.ncnot}  ")
        prints(f"\n({ansatz} state)")
        print_state(state)

    Quket.energy = Energy
    Quket.s2 = S2
    return cost, S2
