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

from qulacs import QuantumCircuit
from qulacs.gate import PauliRotation

from quket import config as cf
from quket.fileio import SaveTheta, printmat, print_state, prints
from quket.utils import orthogonal_constraint
from quket.projection import S2Proj, NProj
from quket.lib import QuantumState
from .upcclib import upcc_Gsingles


def set_circuit_bcs(ansatz, n_qubits, n_orbitals, ndim1, ndim, theta_list, k):
    circuit = QuantumCircuit(n_qubits)
    target_list = [0]*2
    pauli_index = [0]*2
    for i in range(k):
        ioff  = i*ndim
        for p in range(n_orbitals):
            pa = 2*p
            pb = 2*p + 1
            target_list = pa, pb

            pauli_index = 1, 2
            gate = PauliRotation(target_list, pauli_index, -theta_list[p+ioff])
            circuit.add_gate(gate)

            pauli_index = 2, 1
            gate = PauliRotation(target_list, pauli_index, -theta_list[p+ioff])
            circuit.add_gate(gate)

            if "ebcs" in ansatz:
                if p < n_orbitals-1:
                    circuit.add_CNOT_gate(pa, pa+2)
                    circuit.add_CNOT_gate(pb, pb+2)
        upcc_Gsingles(circuit, n_orbitals, theta_list, ndim1, n_orbitals, i)
    return circuit


def cost_bcs(Quket, print_level, theta_list, k):
    """Function:
    Energy functional of kBCS

    Author(s): Takahiro Yoshikura
    """
    t1 = time.time()

    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    n_qubits = Quket.n_qubits
    det = Quket.current_det
    ndim1 = Quket.ndim1
    ndim = Quket.ndim

    state = Quket.init_state.copy()
    circuit = set_circuit_bcs(ansatz, n_qubits, n_orbitals, ndim1, ndim,
                              theta_list, k)
    circuit.update_quantum_state(state)

    if Quket.projection.SpinProj:
        state_P = S2Proj(Quket, state)
        state   = state_P.copy()
    if Quket.projection.NumberProj:
        state_P = NProj(Quket, state)
        state   = state_P.copy()
    #print_state(state, threshold=cf.print_amp_thres)

    Ebcs = Quket.qulacs.Hamiltonian.get_expectation_value(state)
    cost = Ebcs
    ### Project out the states contained in 'lower_states'
    cost += orthogonal_constraint(Quket, state)
    S2 = Quket.qulacs.S2.get_expectation_value(state)

    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(f"{cf.icyc:6d}: "
               f"E[{k}-BCS] = {Ebcs:.12f}  <S**2> = {S2:+17.15f}  "
               f"Grad = {cf.grad:4.2e}  "
               f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        SaveTheta(ndim, theta_list, cf.tmp)
        Quket.theta_list = theta_list.copy()
    if print_level > 1:
        prints(f" Final: "
               f"E[{k}-BCS] = {Ebcs:.12f}  <S**2> = {S2:+17.15f}")
        printmat(theta_list, name=f"\n({k}-BCS state)")
        print_state(state, threshold=Quket.print_amp_thres)

    # Store kBCS wave function
    Quket.state = state
    return cost, S2
