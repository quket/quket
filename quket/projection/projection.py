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

projection.py

Functions related to spin-projection.

"""
import time
import math

import numpy as np
from qulacs import QuantumCircuit

from quket import config as cf
from quket.utils import int2occ
from quket.opelib import set_exp_circuit 
from quket.fileio import SaveTheta, print_state, print_amplitudes, prints, error
from quket.lib import QuantumState


def trapezoidal(x0, x1, n):
    """Function
    Return points and weights based on trapezoidal rule
    """
    if n == 1:
        h = 0
    else:
        h = (x1-x0)/(n-1)

    w = np.zeros(n)
    x = np.zeros(n)
    w[0] = w[n-1] = h/2.
    w[1 : n-1] = h
    x[0] = x0
    x[n-1] = x1
    x[1 : n-1] = x0 + np.arange(1, n-1)*h
    return x.tolist(), w.tolist()


def simpson(x0, x1, n):
    """Function
    Return points and weights based on simpson's rule
    """
    if n%2 == 0:
        error("Simpson's rule cannot be applied with even grids.")

    if n == 1:
        h = 0
    else:
        h = (x1-0)/(n-1)

    w = np.zeros(n)
    x = np.zeros(n)
    w[0] = w[n-1] = h/3.
    w[1 : n-1 : 2] = 2./3.*h
    w[2 : n-1 : 2] = 4./3.*h
    x[0] = x0
    x[n-1] = x1
    x[1 : n-1] = x0 + np.arange(1, n-1)*h
    return x.tolist(), w.tolist()


def weightspin(nbeta, spin, m, n, beta):
    """Function
    Calculae Wigner small d-matrix d^j_{mn}(beta)
    """
    j = spin - 1
    dmm = [wigner_d_matrix(j, m, n, beta[irot])*(j+1)/2.
           for irot in range(nbeta)]
    return dmm


def wigner_d_matrix(j, m, n, angle):
    i1 = (j+n)//2
    i2 = (j-n)//2
    i3 = (j+m)//2
    i4 = (j-m)//2
    i5 = (n-m)//2
    f1 = 1 if i1 == -1 else math.factorial(i1)
    f2 = 1 if i2 == -1 else math.factorial(i2)
    f3 = 1 if i3 == -1 else math.factorial(i3)
    f4 = 1 if i4 == -1 else math.factorial(i4)
    min_k = max(0, i5)
    max_k = min(i1, i4)
    root = np.sqrt(f1*f2*f3*f4)
    cosB = np.cos(angle/2)
    sinB = np.sin(angle/2)

    d_matrix = 0
    for k in range(min_k, max_k+1):
        x1 = 1 if i1 - k == -1 else math.factorial(i1-k)
        x4 = 1 if i4 - k == -1 else math.factorial(i4-k)
        x5 = 1 if k - i5 == -1 else math.factorial(k-i5)
        x = 1 if k == -1 else math.factorial(k)

        denominator = (-1)**(k-i5) * x1 * x * x4 * x5
        numerator = cosB**(i1 + i4 - 2*k) * sinB**(2*k - i5) * root
        d_matrix += numerator/denominator
    return d_matrix


def set_circuit_Ug(circuit, n_Pqubits, beta):
    """Function:
    Construct circuit for Ug in spin-projection (only exp[-i beta Sy])

    Author(s): Takashi Tsuchimochi
    """
    ### Ug
    for i in range(0, n_Pqubits, 2):
        circuit.add_H_gate(i)
        circuit.add_RX_gate(i+1, np.pi/2)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_RZ_gate(i, -beta/2)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_H_gate(i)
        circuit.add_RX_gate(i+1, -np.pi/2)

        circuit.add_H_gate(i+1)
        circuit.add_RX_gate(i, np.pi/2)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_RZ_gate(i, beta/2)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_H_gate(i+1)
        circuit.add_RX_gate(i, -np.pi/2)


def set_circuit_ExpSy(circuit, n_Pqubits, angle, bs_orbitals):
    """Function
    Construct circuit Exp[ -i angle Sy ]

    Author(s): Takashi Tsuchimochi
    """
    from quket.opelib import single_ope_Pauli
    #for i in range(0, n_Pqubits, 2):
    #    single_ope_Pauli(i+1, i, circuit, angle/2, approx=False)
    for i in bs_orbitals:
        single_ope_Pauli(2*i+1, 2*i, circuit, angle/2, approx=False)


def set_circuit_ExpSz(circuit, n_Pqubits, angle, bs_orbitals):
    """Function
    Construct circuit Exp[ -i angle Sz ]
    (20210205) Bug fixed
            Sz = 1/4 (-Z0 +Z1 -Z2 +Z3 ...)

    Author(s): Takashi Tsuchimochi
    """
    #for i in range(n_Pqubits):
    #    if i%2 == 0:
    #        circuit.add_RZ_gate(i, angle/2)
    #    else:
    #        circuit.add_RZ_gate(i, -angle/2)
    for i in bs_orbitals:
            circuit.add_RZ_gate(2*i, angle/2)
            circuit.add_RZ_gate(2*i+1, -angle/2)
        


def set_circuit_ExpNa(circuit, n_Pqubits, angle):
    """Function
    Construct circuit Exp[ -i angle Na ] Exp[ i angle M/2]
    Na = Number operator for alpha spin
       = M/2  -  1/2 ( Z0 + Z2 + Z4 + ...)
    The phase Exp[ -i angle M/2 ] is canceled out here,
    and treated elsewhere.

    Author(s): Takashi Tsuchimochi
    """

    for i in range(0, n_Pqubits, 2):
        circuit.add_RZ_gate(i, angle)


def set_circuit_ExpNb(circuit, n_Pqubits, angle):
    """Function
    Construct circuit Exp[ -i angle Nb ] Exp[ i angle M/2]
            Nb = Number operator for beta spin
               = M/2  -  1/2 ( Z1 + Z3 + Z5 + ...)
    The phase Exp[ -i angle M/2 ] is canceled out here,
    and treated elsewhere.

    Author(s): Takashi Tsuchimochi
    """
    for i in range(1, n_qubitsm, 2):
        circuit.add_RZ_gate(i, angle)


def set_circuit_Rg(circuit, n_Pqubits, alpha, beta, gamma, bs_orbitals):
    """Function
    Construct circuit Rg for complete spin-projection

    Author(s): Takashi Tsuchimochi
    """
    set_circuit_ExpSz(circuit, n_Pqubits, gamma, bs_orbitals)
    set_circuit_ExpSy(circuit, n_Pqubits, beta, bs_orbitals)
    set_circuit_ExpSz(circuit, n_Pqubits, alpha, bs_orbitals)


def controlled_Ug_gen(circuit, n_qubits, anc, alpha, beta, gamma,
                      threshold=1e-6):
    """Function:
    Construct circuit for controlled-Ug in general spin-projection

    Args:
        circuit (QuantumCircuit): Circuit to be updated in return
        n_qubits (int): Total number of qubits (including ancilla)
        anc (int): The index number of ancilla (= n_qubits - 1)
        alpha (float): alpha angle for spin-rotation
        beta (float): beta angle for spin-rotation
        gamma (float): gamma angle for spin-rotation

    Author(s): Takashi Tsuchimochi
    """
    ### Controlled Ug(alpha, beta, gamma)
    if gamma > threshold:
        for i in range(n_qubits):
            if i % 2 == 0:
                circuit.add_RZ_gate(i, gamma/4)
                circuit.add_CNOT_gate(anc, i)
                circuit.add_RZ_gate(i, -gamma/4)
                circuit.add_CNOT_gate(anc, i)
            else:
                circuit.add_RZ_gate(i, -gamma/4)
                circuit.add_CNOT_gate(anc, i)
                circuit.add_RZ_gate(i, gamma/4)
                circuit.add_CNOT_gate(anc, i)

    for i in range(0, n_qubits, 2):
        circuit.add_H_gate(i)
        circuit.add_RX_gate(i+1, np.pi/2)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_RZ_gate(i, -beta/4)
        circuit.add_CNOT_gate(anc, i)
        circuit.add_RZ_gate(i, beta/4)
        circuit.add_CNOT_gate(anc, i)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_H_gate(i)
        circuit.add_RX_gate(i+1, -np.pi/2)

        circuit.add_H_gate(i+1)
        circuit.add_RX_gate(i, np.pi/2)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_RZ_gate(i, beta/4)
        circuit.add_CNOT_gate(anc, i)
        circuit.add_RZ_gate(i, -beta/4)
        circuit.add_CNOT_gate(anc, i)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_H_gate(i+1)
        circuit.add_RX_gate(i, -np.pi/2)

    if alpha > threshold:
        for i in range(n_qubits):
            if i % 2 == 0:
                circuit.add_RZ_gate(i, alpha/4)
                circuit.add_CNOT_gate(anc, i)
                circuit.add_RZ_gate(i, -alpha/4)
                circuit.add_CNOT_gate(anc, i)
            else:
                circuit.add_RZ_gate(i, -alpha/4)
                circuit.add_CNOT_gate(anc, i)
                circuit.add_RZ_gate(i, alpha/4)
                circuit.add_CNOT_gate(anc, i)


def controlled_Ug(circuit, n_qubits, anc, beta):
    """Function:
    Construct circuit for controlled-Ug in spin-projection

    Author(s): Takashi Tsuchimochi
    """
    ### Controlled Ug
    for i in range(0, n_qubits, 2):
        circuit.add_H_gate(i)
        circuit.add_RX_gate(i+1, np.pi/2)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_RZ_gate(i, -beta/4)
        circuit.add_CNOT_gate(anc, i)
        circuit.add_RZ_gate(i, beta/4)
        circuit.add_CNOT_gate(anc, i)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_H_gate(i)
        circuit.add_RX_gate(i+1, -np.pi/2)

        circuit.add_H_gate(i+1)
        circuit.add_RX_gate(i, np.pi/2)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_RZ_gate(i, beta/4)
        circuit.add_CNOT_gate(anc, i)
        circuit.add_RZ_gate(i, -beta/4)
        circuit.add_CNOT_gate(anc, i)
        circuit.add_CNOT_gate(i+1, i)
        circuit.add_H_gate(i+1)
        circuit.add_RX_gate(i, -np.pi/2)



def S2Proj(Quket, Q, threshold=1e-8, bs_orbitals=None, normalize=True, nalpha=None, nbeta=None, ngamma=None):
    """Function
    Perform spin-projection to QuantumState |Q>
            |Q'>  =  Ps |Q>
    where Ps is a spin-projection operator (non-unitary).
            Ps = \sum_i^ng   wg[i] Ug[i]
    This function provides a shortcut to |Q'>, which is unreal.
    One actually needs to develop a quantum circuit for this
    (See PRR 2, 043142 (2020)).

    Author(s): Takashi Tsuchimochi
    """
    spin = Quket.projection.spin
    s = (spin-1)/2.
    ### Check Ms is consistent with current_det
    occ = int2occ(Quket.current_det)
    occ = [i +  Quket.nc*2 for i in occ]
    occ_a = [i for i in occ if i%2 == 0]
    occ_b = [i for i in occ if i%2 == 1]
    noa = len(occ_a)
    nob = len(occ_b)
    if (noa-nob) != Quket.projection.Ms:
        prints(f'WARNING! Initial product state {bin(Quket.current_det)[2:]} has noa={noa} and nob={nob} but Projection.Ms = {Quket.projection.Ms}.\nForce to overwrite.')
        Quket.projection.Ms = noa-nob
        Quket.set_projection()
    Ms = Quket.projection.Ms/2.

    n_qubits = Q.get_qubit_count()
    state_P = QuantumState(n_qubits)
    state_P.multiply_coef(0)
    sp_angle = []
    sp_weight = []
    if nalpha is None:
        nalpha = max(Quket.projection.euler_ngrids[0], 1)
        sp_angle.append(Quket.projection.sp_angle[0])
        sp_weight.append(Quket.projection.sp_weight[0])
    else:
        ### Prepare alpha angles and weights
        if nalpha > 1:
            alpha, wg_alpha = trapezoidal(0, 2*np.pi, nalpha)
        else:
            nalpha = 1
            alpha = [0]
            wg_alpha = [1]
        sp_angle.append(alpha)
        sp_weight.append(wg_alpha)

    if nbeta is None:
        nbeta = max(Quket.projection.euler_ngrids[1], 1)
        sp_angle.append(Quket.projection.sp_angle[1])
        sp_weight.append(Quket.projection.sp_weight[1])
        dmm = Quket.projection.dmm
    else:
        ### Prepare beta angles and weights
        if nbeta > 1:
            beta, wg_beta \
                    = np.polynomial.legendre.leggauss(nbeta)
            beta = np.arccos(beta)
            beta = beta.tolist()
            dmm = weightspin(nbeta, spin, Quket.projection.Ms, Quket.projection.Ms, beta)
        else:
            nbeta = 1
            beta = [0]
            wg_beta = [1]
            dmm = [1]
        sp_angle.append(beta)
        sp_weight.append(wg_beta)

    if ngamma is None:
        ngamma = max(Quket.projection.euler_ngrids[2], 1)
        sp_angle.append(Quket.projection.sp_angle[2])
        sp_weight.append(Quket.projection.sp_weight[2])
    else:
        ### Prepare gamma angles and weights
        if ngamma > 1:
            gamma, wg_gamma = trapezoidal(0, 2*np.pi, ngamma) 
        else:
            ngamma = 1
            gamma = [0]
            wg_gamma = [1]
        sp_angle.append(gamma)
        sp_weight.append(wg_gamma)

    if bs_orbitals is None:
        bs_orbitals = [i for i in range(Quket.n_active_orbitals)]

    for ialpha in range(nalpha):
        alpha = sp_angle[0][ialpha]
        alpha_coef = sp_weight[0][ialpha]*np.exp(1j*alpha*Ms)

        for ibeta in range(nbeta):
            beta = sp_angle[1][ibeta]
            beta_coef = (sp_weight[1][ibeta] * dmm[ibeta])

            for igamma in range(ngamma):
                gamma = sp_angle[2][igamma]
                gamma_coef = (sp_weight[2][igamma] * np.exp(1j*gamma*Ms))

                angle_list = [gamma, beta, alpha]

                # Total Weight
                #coef = (2*s + 1)/(8*np.pi)*(alpha_coef*beta_coef*gamma_coef)
                coef = (alpha_coef*beta_coef*gamma_coef)

                state_g = QuantumState(n_qubits)
                state_g.load(Q)
                ### If tapering is exercised...
                #if Quket.tapered["states"]: 
                #    circuit_Rg = set_exp_circuit(n_qubits, Quket.projection.Rg_pauli, angle_list)
                #else:
                #    circuit_Rg = QuantumCircuit(n_qubits)
                #    set_circuit_Rg(circuit_Rg, n_qubits, alpha, beta, gamma, bs_orbitals)
                #circuit_Rg = QuantumCircuit(n_qubits)
                #set_circuit_Rg(circuit_Rg, n_qubits, alpha, beta, gamma, bs_orbitals)

                circuit_Rg = set_exp_circuit(n_qubits, Quket.projection.Rg_pauli_list, angle_list)

                circuit_Rg.update_quantum_state(state_g)
                state_g.multiply_coef(coef)
                state_P.add_state(state_g)

    if normalize:
        # Normalize
        norm2 = state_P.get_squared_norm()
        if norm2 < threshold:
            error(f"Norm of spin-projected state is too small! "
                  f"(norm={norm2:4.4E} < {threshold})\n",
                  f"This usually means the broken-symmetry state has NO component "
                  f"of the target spin.")
        state_P.normalize(norm2)
    # print_state(state_P,name="P|Q>",threshold=1e-6)
    return state_P


def NProj(Quket, Q, threshold=1e-8):
    """Function
    Perform number-projection to QuantumState |Q>
            |Q'>  =  PN |Q>
    where PN is a number-projection operator (non-unitary).
            PN = \sum_i^ng   wg[i] Ug[i]
    This function provides a shortcut to |Q'>, which is unreal.
    One actually needs to develop a quantum circuit for this
    (See QST 6, 014004 (2021)).

    Author(s): Takashi Tsuchimochi
    """
    n_qubits = Q.get_qubit_count()
    state_P = QuantumState(n_qubits)
    state_P.multiply_coef(0)
    state_g = QuantumState(n_qubits)
    nphi = max(Quket.projection.number_ngrids, 1)
    #print_state(Q)
    for iphi in range(nphi):
        coef = (Quket.projection.np_weight[iphi]
               *np.exp(1j*Quket.projection.np_angle[iphi]
                       *(Quket.projection.n_active_electrons
                         - Quket.projection.n_active_orbitals)))

        state_g= Q.copy()
        circuit = QuantumCircuit(n_qubits)
        set_circuit_ExpNa(circuit, n_qubits, Quket.projection.np_angle[iphi])
        set_circuit_ExpNb(circuit, n_qubits, Quket.projection.np_angle[iphi])
        circuit.update_quantum_state(state_g)
        state_g.multiply_coef(coef)
        state_P.add_state(state_g)

    norm2 = state_P.get_squared_norm()
    if norm2 < threshold:
        error(f"Norm of number-projected state is too small!"
                f"(norm={norm2:4.4E} < {threshold})\n",
              f"This usually means the broken-symmetry state has NO component "
              f"of the target number.")
    state_P.normalize(norm2)
    return state_P

