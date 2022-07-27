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

from quket import config as cf
from quket.fileio import SaveTheta, print_state, prints
from quket.opelib import Gdouble_ope
from quket.utils import orthogonal_constraint
from .ucclib import ucc_Gsingles
from quket.opelib import single_ope_Pauli
from quket.projection import S2Proj
from quket.lib import QuantumState


def set_circuit_upccgsd(n_qubits, norbs, theta_list, ndim1, ndim2, k):
    """Function:
    Construct new circuit for UpCCGSD

    Author(s): Takahiro Yoshikura
    """
    circuit = QuantumCircuit(n_qubits)
    for i in range(k):
        upcc_Gdoubles(circuit, norbs, theta_list, ndim1, ndim2, i)
        upcc_Gsingles(circuit, norbs, theta_list, ndim1, ndim2, i)
    return circuit


def set_circuit_epccgsd(n_qubits, norbs, theta_list, ndim1, ndim2, k):
    """Function:
    Construct new circuit for EpCCGSD

    Author(s): Takahiro Yoshikura
    """
    circuit = QuantumCircuit(n_qubits)
    for i in range(k-1):
        upcc_Gdoubles(circuit, norbs, theta_list, ndim1, ndim2, i)
        upcc_Gsingles(circuit, norbs, theta_list, ndim1, ndim2, i)
    upcc_Gdoubles(circuit, norbs, theta_list, ndim1, ndim2, k-1)
    ucc_Gsingles(circuit, norbs, theta_list, (ndim1+ndim2)*(k-1) + ndim2)
    return circuit


def upcc_Gdoubles(circuit, norbs, theta_list, ndim1, ndim2, p):
    """Function:
    Construct circuit for UpCC (pair-dobles part)

    Author(s): Takahiro Yoshikura
    """
    ijab = (ndim1+ndim2)*p
    for a in range(norbs):
        a2 = 2*a
        for i in range(a):
            i2 = 2*i
            Gdouble_ope(a2+1, a2, i2+1, i2, circuit, theta_list[ijab])
            ijab += 1


def upcc_Gsingles(circuit, n_orbitals, theta_list, ndim1, ndim2, p):
    """Function:
    Construct circuit for UpCC (singles part)

    Author(s): Takahiro Yoshikura, Takashi Tsuchimochi (spin-free)
    """
    ia = ndim2 + (ndim1+ndim2)*p
    for a in range(n_orbitals):
        a2 = 2*a
        for i in range(a):
            i2 = 2*i
            single_ope_Pauli(a2, i2, circuit, theta_list[ia])
            single_ope_Pauli(a2+1, i2+1, circuit,  theta_list[ia])
            ia += 1


def cost_upccgsd(Quket, print_level, kappa_list, theta_list, k):
    """Function:
    Energy functional of UpCCGSD

    Author(s): Takahiro Yoshikura
    """
    t1 = time.time()

    norbs = Quket.n_active_orbitals
    n_qubits = Quket.n_qubits
    det = Quket.current_det
    ndim1 = Quket.ndim1
    ndim2 = Quket.ndim2
    ndim = Quket.ndim

    #state = QuantumState(n_qubits)
    #state.set_computational_basis(det)
    state = Quket.init_state.copy()
    if "epccgsd" in Quket.ansatz:
        circuit = set_circuit_epccgsd(n_qubits, norbs, theta_list,
                                      ndim1, ndim2, k)
    else:
        circuit = set_circuit_upccgsd(n_qubits, norbs, theta_list,
                                      ndim1, ndim2, k)
    circuit.update_quantum_state(state)
    if Quket.projection.SpinProj:
        state_P = S2Proj(Quket,state)
        state   = state_P.copy()

    #Eupccgsd = Quket.qulacs.Hamiltonian.get_expectation_value(state)
    Eupccgsd = Quket.get_E(state,parallel=False)
    cost = Eupccgsd
    ### Project out the states contained in 'lower_states'
    cost += orthogonal_constraint(Quket, state)
    #S2 = Quket.qulacs.S2.get_expectation_value(state)
    S2 = Quket.get_S2(state,parallel=False)

    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(f"{cf.icyc:5d}: "
               f"E[{k}-UpCCGSD] = {Eupccgsd:.12f}  <S**2> = {S2:+17.15f}  "
               f"Grad = {cf.grad:4.2e}  "
               f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        SaveTheta(ndim, theta_list, cf.tmp)
        Quket.theta_list = theta_list.copy()
    if print_level > 1:
        prints(f"Final: "
               f"E[{k}-UpCCGSD] = {Eupccgsd:.12f}  <S**2> = {S2:+17.15f}")
        prints(f"\n({k}-UpCCGSD state)")
        print_state(state, threshold=Quket.print_amp_thres)

    # Store UpCCGSD wave function
    Quket.state = state
    Quket.energy = Eupccgsd
    Quket.s2 = S2
    return cost, S2

