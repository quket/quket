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


from quket import config as cf
from quket.fileio import (SaveTheta, print_state, print_amplitudes, prints)
from quket.opelib import set_exp_circuit
from quket.utils import transform_state_jw2bk
from quket.lib import QuantumState

def cost_uccgd_forSAOO(Quket, print_level, qulacs_hamiltonian, qulacs_s2,
                       theta_list):
    """Function
    Author(s): Yuto Mori
    """
    
    t1 = time.time()

    nca = ncb = Quket.nc
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    rho = Quket.rho
    DS = Quket.DS
    det = Quket.current_det
    n_qubits = Quket.n_qubits
    ndim1 = Quket.ndim1
    ndim2 = Quket.ndim2
    ndim = Quket.ndim
    nstates = Quket.multi.nstates
    
    nc = nca
    no = noa
    nv = nva

    if len(Quket.multi.states) != nstates:
        Quket.multi.states = []
        for istate in range(nstates):
            Quket.multi.states.append(QuantumState(n_qubits))

    Euccds = []
    S2s = []
    circuit = set_exp_circuit(n_qubits, Quket.pauli_list, theta_list, rho)
    for istate in range(nstates):
        det = Quket.multi.dets[istate]
        cf.ncnot = 0
        state = QuantumState(n_qubits)
        state.set_computational_basis(det)
        if Quket.cf.mapping == "bravyi_kitaev":
            state = transform_state_jw2bk(state)
        for i in range(rho):
            circuit.update_quantum_state(state)
            if Quket.projection.SpinProj:
                from quket.projection import S2Proj
                state = S2Proj(Quket,state)
        Euccd = Quket.qulacs.Hamiltonian.get_expectation_value(state)
        S2 = Quket.qulacs.S2.get_expectation_value(state)
        Euccds.append(Euccd)
        S2s.append(S2)
        Quket.multi.states[istate] = state

    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(f"{cf.icyc:6d}: ", end="")
        for istate in range(nstates):
            prints(f"E[{istate}-SA-OO] = {Euccds[istate]:.8f}  "
                   f"(<S**2> = {S2s[istate]:+7.5f})  ", end="")
        prints(f"Grad = {cf.grad:4.2e}  "
               f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        SaveTheta(ndim, theta_list, cf.tmp)
        Quket.theta_list = theta_list.copy()
    if print_level > 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        prints(" Final: ", end="")
        for istate in range(nstates):
            prints(f"E[{istate}-SA-OO] = {Euccds[istate]:.8f}  "
                   f"(<S**2> = {S2s[istate]:+7.5f})  ", end="")
        prints(f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)\n")
        prints("------------------------------------")
        prints("###############################################")
        prints("#               SA-OO states                  #")
        prints("###############################################", end="")

        for istate in range(nstates):
            prints()
            prints(f"State         : {istate}")
            prints(f"E             : {Euccds[istate]:.8f}")
            prints(f"<S**2>        : {S2s[istate]:+.5f}")
            print_state(Quket.multi.states[istate])
        prints("###############################################")

    cost = norm = 0
    norm = np.sum(Quket.multi.weights)
    for istate in range(nstates):
        cost = cost + Quket.multi.weights[istate] * Euccds[istate]
    cost /= norm
    Quket.energy = cost
    return cost, S2s

