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

expope.py

Functions to prepare several types of rotations, including singles and doubles rotations.

"""
import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import PauliRotation

from quket import config as cf
from quket.fileio import prints 
from quket.lib import QuantumState

def single_ope_Pauli(a, i, circuit, theta, approx=cf.approx_exp):
    """Function:
    Construct exp[theta(a!i - i!a)] as a whole unitary and add to circuit

    Author(s): Takashi Tsuchimochi
    """
    if abs(theta) < cf.theta_threshold:
        return
    ndim1 = max(a,i) - (min(a,i)+1)
    ndim = ndim1 + 2
    if a == i:
        error('a cannot be i in single_ope_Pauli')
    if i > a:
        # Parity
        theta *= - 1
    a, i = max(a,i), min(a,i)


    ### Purpose:
    ### (1)   Exp[ i theta/2  Prod_{k=i+1}^{a-1} Z_k Y_i X_a]
    ### (2)   Exp[-i theta/2  Prod_{k=i+1}^{a-1} Z_k Y_a X_i]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(i+1, a))
    target_list[ndim1:] = i, a
    pauli_index = [3]*ndim

    # (1)                Yi,Xa
    pauli_index[ndim1:] = 2, 1
    gate = PauliRotation(target_list, pauli_index, theta)
    circuit.add_gate(gate)

    # (2)                Xi,Ya
    pauli_index[ndim1:] = 1, 2
    gate = PauliRotation(target_list, pauli_index, -theta)
    circuit.add_gate(gate)
    
    cf.ncnot += (a-i) * 2 * 2


def double_ope_Pauli(b, a, j, i, circuit, theta):
    """Function:
    Construct exp[theta(a!b!ji - i!j!ba)]
    as a whole unitary and add to circuit

    Author(s): Takashi Tsuchimochi
    """
    ndim1 = min(a, j) - (i+1)
    ndim2 = ndim1 + b - (max(a, j)+1)
    ndim = ndim2 + 4

    ### Purpose:
    ### (1)   Exp[ i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (X_i X_j Y_a X_b)]
    ### (2)   Exp[ i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (Y_i X_j Y_a Y_b)]
    ### (3)   Exp[ i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (X_i Y_j Y_a Y_b)]
    ### (4)   Exp[ i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (X_i X_j X_a Y_b)]
    ### (5)   Exp[-i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (Y_i X_j X_a X_b)]
    ### (6)   Exp[-i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (X_i Y_j X_a X_b)]
    ### (7)   Exp[-i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (Y_i Y_j Y_a X_b)]
    ### (8)   Exp[-i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (Y_i Y_j X_a Y_b)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(i+1, min(a, j)))
    target_list[ndim1:ndim2] = list(range(max(a, j)+1, b))
    target_list[ndim2:] = i, j, a, b
    pauli_index = [3]*ndim

    ### (1)              Xi,Xj,Ya,Xb
    pauli_index[ndim2:] = 1, 1, 2, 1
    gate = PauliRotation(target_list, pauli_index, theta)
    circuit.add_gate(gate)

    ### (2)              Yi,Xj,Ya,Yb
    pauli_index[ndim2:] = 2, 1, 2, 2
    gate = PauliRotation(target_list, pauli_index, theta)
    circuit.add_gate(gate)

    ### (3)              Xi,Yj,Ya,Yb
    pauli_index[ndim2:] = 1, 2, 2, 2
    gate = PauliRotation(target_list, pauli_index, theta)
    circuit.add_gate(gate)

    ### (4)              Xi,Xj,Xa,Yb
    pauli_index[ndim2:] = 1, 1, 1, 2
    gate = PauliRotation(target_list, pauli_index, theta)
    circuit.add_gate(gate)

    ### (5)              Yi,Xj,Xa,Xb
    pauli_index[ndim2:] = 2, 1, 1, 1
    gate = PauliRotation(target_list, pauli_index, -theta)
    circuit.add_gate(gate)

    ### (6)              Xi,Yj,Xa,Xb
    pauli_index[ndim2:] = 1, 2, 1, 1
    gate = PauliRotation(target_list, pauli_index, -theta)
    circuit.add_gate(gate)

    ### (7)              Yi,Yj,Ya,Xb
    pauli_index[ndim2:] = 2, 2, 2, 1
    gate = PauliRotation(target_list, pauli_index, -theta)
    circuit.add_gate(gate)

    ### (8)              Yi,Yj,Xa,Yb
    pauli_index[ndim2:] = 2, 2, 1, 2
    gate = PauliRotation(target_list, pauli_index, -theta)
    circuit.add_gate(gate)



def single_ope(a, i, circuit, theta):
    """Function:
    Construct exp[theta(a!i - i!a)] by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """
    circuit.add_H_gate(a)
    circuit.add_RX_gate(i, -np.pi/2)
    for k in range(a, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, theta)
    for k in range(i, a):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(a)
    circuit.add_RX_gate(i, np.pi/2)

    circuit.add_H_gate(i)
    circuit.add_RX_gate(a, -np.pi/2)
    for k in range(a, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, -theta)
    for k in range(i, a):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(i)
    circuit.add_RX_gate(a, np.pi/2)


def double_ope_1(b, a, j, i, circuit, theta):
    """Function:
    Construct the first part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """
    circuit.add_H_gate(b)
    circuit.add_H_gate(a)
    circuit.add_RX_gate(j, -np.pi/2)
    circuit.add_H_gate(i)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(b)
    circuit.add_H_gate(a)
    circuit.add_RX_gate(j, np.pi/2)
    circuit.add_H_gate(i)


def double_ope_2(b, a, j, i, circuit, theta):
    """Function:
    Construct the second part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """
    circuit.add_RX_gate(b, -np.pi/2)
    circuit.add_H_gate(a)
    circuit.add_RX_gate(j, -np.pi/2)
    circuit.add_RX_gate(i, -np.pi/2)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_RX_gate(b, np.pi/2)
    circuit.add_H_gate(a)
    circuit.add_RX_gate(j, np.pi/2)
    circuit.add_RX_gate(i, np.pi/2)


def double_ope_3(b, a, j, i, circuit, theta):
    """Function:
    Construct the third part of exp[theta ( a!b!ji - i!j!ba ) ] by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """

    circuit.add_H_gate(b)
    circuit.add_RX_gate(a, -np.pi/2)
    circuit.add_RX_gate(j, -np.pi/2)
    circuit.add_RX_gate(i, -np.pi/2)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(b)
    circuit.add_RX_gate(a, np.pi/2)
    circuit.add_RX_gate(j, np.pi/2)
    circuit.add_RX_gate(i, np.pi/2)


def double_ope_4(b, a, j, i, circuit, theta):
    """Function:
    Construct the fourth part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """
    circuit.add_H_gate(b)
    circuit.add_H_gate(a)
    circuit.add_H_gate(j)
    circuit.add_RX_gate(i, -np.pi/2)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(b)
    circuit.add_H_gate(a)
    circuit.add_H_gate(j)
    circuit.add_RX_gate(i, np.pi/2)


def double_ope_5(b, a, j, i, circuit, theta):
    """Function:
    Construct the fifth part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """
    circuit.add_RX_gate(b, -np.pi/2)
    circuit.add_H_gate(a)
    circuit.add_H_gate(j)
    circuit.add_H_gate(i)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, -theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_RX_gate(b, np.pi/2)
    circuit.add_H_gate(a)
    circuit.add_H_gate(j)
    circuit.add_H_gate(i)


def double_ope_6(b, a, j, i, circuit, theta):
    """Function:
    Construct the sixth part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """
    circuit.add_H_gate(b)
    circuit.add_RX_gate(a, -np.pi/2)
    circuit.add_H_gate(j)
    circuit.add_H_gate(i)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, -theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(b)
    circuit.add_RX_gate(a, np.pi/2)
    circuit.add_H_gate(j)
    circuit.add_H_gate(i)


def double_ope_7(b, a, j, i, circuit, theta):
    """Function:
    Construct the seventh part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """
    circuit.add_RX_gate(b, -np.pi/2)
    circuit.add_RX_gate(a, -np.pi/2)
    circuit.add_RX_gate(j, -np.pi/2)
    circuit.add_H_gate(i)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, -theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_RX_gate(b, np.pi/2)
    circuit.add_RX_gate(a, np.pi/2)
    circuit.add_RX_gate(j, np.pi/2)
    circuit.add_H_gate(i)


def double_ope_8(b, a, j, i, circuit, theta):
    """Function:
    Construct the eighth part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """
    circuit.add_RX_gate(b, -np.pi/2)
    circuit.add_RX_gate(a, -np.pi/2)
    circuit.add_H_gate(j)
    circuit.add_RX_gate(i, -np.pi/2)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, -theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_RX_gate(b, np.pi/2)
    circuit.add_RX_gate(a, np.pi/2)
    circuit.add_H_gate(j)
    circuit.add_RX_gate(i, np.pi/2)

def double_ope(b, a, j, i, circuit, theta):
    """Function:
    Wrapper for exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ to be added to circuit

    Author(s): Yuto Mori
    """
    double_ope_1(b, a, j, i, circuit, theta)
    double_ope_2(b, a, j, i, circuit, theta)
    double_ope_3(b, a, j, i, circuit, theta)
    double_ope_4(b, a, j, i, circuit, theta)
    double_ope_5(b, a, j, i, circuit, theta)
    double_ope_6(b, a, j, i, circuit, theta)
    double_ope_7(b, a, j, i, circuit, theta)
    double_ope_8(b, a, j, i, circuit, theta)



def Gdouble_ope(p, q, r, s, circuit, theta):
    """ Function:
    Construct exp[theta ( p!q!rs - s!r!qp ) ] as a whole unitary
    and add to circuit.
    Here, max(p, q, r, s) = p is NOT assumed.
    (p, q, r, s) is re-ordered to (p', q', r', s') such that
    p' = max(p', q', r', s').
    There are 5 cases:
        (1) p > q > r > s (includes p > q > s > r)
        (2) p > r > q > s (includes p > r > s > q)
        (3) p = r > q > s (includes p = r > s > q)
        (4) p > q = r > s (includes p > q = s > r)
        (5) p > r > q = s (includes p > s > q = r)
    Note that in Qulacs, the function "PauliRotation" rotates as
            Exp [i theta/2 ...]
    so theta is divided by two.
    Accordingly, we need to multiply theta by two.

    Args:
        p (int): excitation index.
        q (int): excitation index.
        r (int): excitation index.
        s (int): excitation index.
        circuit (QuantumCircuit): circuit to be updated.
        theta (float): real rotation parameter.

    Returns:
        circuit (QuantumCircuit): circuit to be updated.

    Author(s): Takashi Tsuchimochi
    """
    if abs(theta) < cf.theta_threshold:
        ### No need for circuit
        return
    if p == q or r == s:
        prints(f"Caution:  p={p} == q={q}  or  r={r} == s={s}")
        return

    max_pqrs = max(p, q, r, s)
    max_pq = max(p, q)
    max_rs = max(r, s)

    if p == max_pqrs:
        # p > q. Test r and s
        if r == max_rs:
            parity = 1
            p, q, r, s = p, q, r, s
        else:
            parity = -1
            p, q, r, s = p, q, s, r
    elif q == max_pqrs:
        # q > p. Test r and s
        if r == max_rs:
            parity = -1
            p, q, r, s = q, p, r, s
        else:
            parity = 1
            p, q, r, s = q, p, s, r
    elif r == max_pqrs:
        # r > p, q, s. Test p and q
        if p == max_pq:
            parity = -1
            p, q, r, s = r, s, p, q
            #Note: This swapping means r!s!pq - p!q!rs = -(p!q!rs - s!r!qp)
            # So parity is -1 
        else:
            parity = 1
            p, q, r, s = r, s, q, p
            #Note: This swapping means r!s!qp - q!p!rs = p!q!rs - s!r!qp
            # So parity is 1 
    elif s == max_pqrs:
        # s > p, q, r. Test p and q
        if p == max_pq:
            parity = 1
            p, q, r, s = s, r, p, q
            #Note: This swapping means s!r!pq - q!p!rs = p!q!rs - s!r!qp
            # So parity is 1 
        else:
            parity = -1
            p, q, r, s = s, r, q, p
            #Note: This swapping means s!r!qp - p!q!rs = -(p!q!rs - s!r!qp)
            # So parity is -1 
    theta *= parity

    if p == r:
        if q > s:
            # p^ q^ p s  (p > q, p > s)
            Gdoubles_pqps(p, q, s, circuit, theta)
        elif q < s:
            # p^ q^ p s =  p^ s^ p q  (p > s, p > q)
            Gdoubles_pqps(p, s, q, circuit, theta)
        else:
            prints("Error!  p^r^ pr - h.c. = zero")
    elif p == s:  #(necessarily  r < s)
        Gdoubles_pqps(p, q, r, circuit, -theta)
    elif q == r:
        if q > s:
            # p^ q^ q s  (p > q, q > s)
            Gdoubles_pqqs(p, q, s, circuit, theta)
        elif q < s:
            # p^ q^ q s = - p^ q^ s q  (p > q, s > q)
            Gdoubles_pqrq(p, q, s, circuit, -theta)
    elif q == s:
        if q < r:
            # p^ q^ r q  (p > q, r > q)
            Gdoubles_pqrq(p, q, r, circuit, theta)
        elif q > r:
            # p^ q^ r q  = - p^ q^ q r  (p > q, q > r)
            Gdoubles_pqqs(p, q, r, circuit, -theta)
    else:
        if r > s:
            Gdoubles_pqrs(p, q, r, s, circuit, theta)
        elif r < s:
            Gdoubles_pqrs(p, q, s, r, circuit, -theta)


def Gdoubles_pqrs(p, q, r, s, circuit, theta, approx=cf.approx_exp):
    """ Function
    Given
            Epqrs = p^ q^ r s
    with p > q, r > s and max(p, q, r, s) = p,
    compute Exp[i theta (Epqrs - Epqrs!)]

    Author(s): Takashi Tsuchimochi
    """
    i4 = p  # (Fixed)
    if q > r > s:
        i1, i2, i3 = s, r, q
    elif r > q > s:
        i1, i2, i3 = s, q, r
    elif r > s > q:
        i1, i2, i3 = q, s, r
    ndim1 = i2 - (i1+1)
    ndim2 = ndim1 + i4 - (i3+1)
    ndim = ndim2 + 4
    cf.ncnot += (i4 - i3 + i2 - i1 + 1)*8*2
    ### Type (a):
    ### (1)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p X_q X_r X_s)]
    ### (2)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p Y_q X_r X_s)]
    ### (3)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p Y_q X_r Y_s)]
    ### (4)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p Y_q Y_r X_s)]
    ### (5)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p X_q X_r Y_s)]
    ### (6)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p X_q Y_r X_s)]
    ### (7)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p X_q Y_r Y_s)]
    ### (8)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p Y_q Y_r Y_s)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(i1+1, i2))
    target_list[ndim1:ndim2] = list(range(i3+1, i4))
    target_list[ndim2:] = p, q, r, s
    pauli_index = [3]*ndim

    ### (1)              Yp,Xq,Xr,Xs
    pauli_index[ndim2:] = 2, 1, 1, 1
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (2)              Xp,Yq,Xr,Xs
    pauli_index[ndim2:] = 1, 2, 1, 1
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (3)              Yp,Yq,Xr,Ys
    pauli_index[ndim2:] = 2, 2, 1, 2
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (4)              Yp,Yq,Yr,Xs
    pauli_index[ndim2:] = 2, 2, 2, 1
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (5)              Xp,Xq,Xr,Ys
    pauli_index[ndim2:] = 1, 1, 1, 2
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)

    ### (6)              Xp,Xq,Yr,Xs
    pauli_index[ndim2:] = 1, 1, 2, 1
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)

    ### (7)              Yp,Xq,Yr,Ys
    pauli_index[ndim2:] = 2, 1, 2, 2
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)

    ### (8)              Xp,Yq,Yr,Ys
    pauli_index[ndim2:] = 1, 2, 2, 2
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)


def Gdoubles_pqps(p, q, s, circuit, theta):
    """ Function
    Given
            Epqps = p^ q^ p s
    with p > q > s, compute Exp[i theta (Epqps - Epqps!)]
    """
    ndim1 = q - (s+1)
    ndim = ndim1 + 3

    cf.ncnot += (q - s + 1)*2*2 ## (1) and (3) require CNOT(p,q)
    cf.ncnot += (q - s)*2*2     ## (2) and (4)
    ### Type (b)
    ### (1)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k (Z_p X_q Y_s)]
    ### (2)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k (    Y_q X_s)]
    ### (3)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k (Z_p Y_q X_s)]
    ### (4)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k (    X_q Y_s)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(s+1, q))
    target_list[ndim1:] = p, q, s
    pauli_index = [3]*ndim

    ### (1)              Zp,Xq,Ys
    pauli_index[ndim1:] = 3, 1, 2
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (2)              Ip,Yq,Xs
    pauli_index[ndim1:] = 0, 2, 1
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (3)              Zp,Yq,Xs
    pauli_index[ndim1:] = 3, 2, 1
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    ### (4)              Ip,Xq,Ys
    pauli_index[ndim1:] = 0, 1, 2
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)


def Gdoubles_pqqs(p, q, s, circuit, theta):
    """ Function
    Given
            Epqps = p^ q^ q s
    with p > q > s, compute Exp[i theta (Epqqs - Epqqs!)]
    """
    ndim1 = q - (s+1)
    ndim2 = ndim1 + p - (q+1)
    ndim = ndim2 + 3

    cf.ncnot += (p - s)*4*2
    ### Type (c)
    ### (1)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (X_p Z_q Y_s)]
    ### (2)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (Y_p     X_s)]
    ### (3)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (Y_p Z_q X_s)]
    ### (4)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (X_p     Y_s)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(s+1, q))
    target_list[ndim1:ndim2] = list(range(q+1, p))
    target_list[ndim2:] = p, q, s
    pauli_index = [3]*ndim

    ### (1)              Xp,Zq,Ys
    pauli_index[ndim2:] = 1, 3, 2
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (2)              Yp,Iq,Xs
    pauli_index[ndim2:] = 2, 0, 1
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (3)              Yp,Zq,Xs
    pauli_index[ndim2:] = 2, 3, 1
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    ### (4)              Xp,Iq,Ys
    pauli_index[ndim2:] = 1, 0, 2
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)


def Gdoubles_pqrq(p, q, r, circuit, theta):
    """ Function
    Given
            Epqps = p^ q^ r q
    with p > r > q, compute Exp[i theta (Epqrq - Epqrq!)]
    """
    ndim1 = p - (r+1)
    ndim = ndim1 + 3

    cf.ncnot += (p - r + 1)*2*2  ## (1) and (3) require CNOT(r,q) 
    cf.ncnot += (p - r)*2*2  ## (2) and (4) 
    ### Type (d)
    ### (1)   Exp[ i theta/4  Prod_{k=r+1}^{p-1} Z_k  (X_p Z_q Y_r)]
    ### (2)   Exp[ i theta/4  Prod_{k=r+1}^{p-1} Z_k  (Y_p     X_r)]
    ### (3)   Exp[-i theta/4  Prod_{k=r+1}^{p-1} Z_k  (Y_p Z_q X_r)]
    ### (4)   Exp[-i theta/4  Prod_{k=r+1}^{p-1} Z_k  (X_p     Y_r)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(r+1, p))
    target_list[ndim1:] = p, q, r
    pauli_index = [3]*ndim

    ### (1)              Xp,Zq,Yr
    pauli_index[ndim1:] = 1, 3, 2
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (2)              Yp,Iq,Xr
    pauli_index[ndim1:] = 2, 0, 1
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (3)              Yp,Zq,Xr
    pauli_index[ndim1:] = 2, 3, 1
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    ### (4)              Xp,Iq,Yr
    pauli_index[ndim1:] = 1, 0, 2
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)


def bcs_single_ope(p, circuit, lam, theta):
    """Function
    Construct circuit for
           exp[ -i lam Z/2 ] exp[ -i theta Y/2) ]
    acting on 2p and 2p+1 qubits,
    required for BCS wave function.

    Args:
        p (int): orbital index
        circuit (QuantumCircuit): circuit to be updated
        lam (float): phase parameter
        theta (float): rotation parameter

    Returns:
        circuit (QuantumCircuit): circuit to be updated
    """
    pass


def fswap(p,circuit):
    ### Swap p and p+1 fermions
    circuit.add_H_gate(p)
    circuit.add_H_gate(p+1)
    circuit.add_CNOT_gate(p,p+1)
    circuit.add_RZ_gate(p+1, np.pi/2)
    circuit.add_CNOT_gate(p,p+1)
    circuit.add_H_gate(p)
    circuit.add_H_gate(p+1)
    circuit.add_RX_gate(p, -np.pi/2)
    circuit.add_RX_gate(p+1, -np.pi/2)
    circuit.add_CNOT_gate(p,p+1)
    circuit.add_RZ_gate(p+1, np.pi/2)
    circuit.add_CNOT_gate(p,p+1)
    circuit.add_RX_gate(p, np.pi/2)
    circuit.add_RX_gate(p+1, np.pi/2)

    circuit.add_RZ_gate(p,np.pi/2)
    circuit.add_RZ_gate(p+1,np.pi/2)
    cf.ncnot += 4
    return 

def fswap_pq(p,q,circuit):
    ### Swap p and q fermions
    if p==q:
        return
    ### Fswap gate we use is 
    ###  Exp[i pi/2]
    ###  Exp[-i pi/4 Prod_{k=min_pq+1}^{max_pq-1} Z_k (X_p X_q)]
    ###  Exp[-i pi/4 Prod_{k=min_pq+1}^{max_pq-1} Z_k (Y_p Y_q)]
    ###  Exp[-i pi/4 Prod_{k=min_pq+1}^{max_pq-1} Z_k (X_p X_q)]
    ###  Exp[-i pi/4 Prod_{k=min_pq+1}^{max_pq-1} Z_k (X_p X_q)]
    ###

    ### Global Phase i: This may not be needed in most actual applications,
    ###                 but _is_ required for state-vector emulation, which
    ###                 on many occasions uses transition amplitudes instead
    ###                 of expectation values.
    circuit.add_gate(PauliRotation([],[],np.pi))

    ### 
    circuit.add_RZ_gate(p,-np.pi/2)
    circuit.add_RZ_gate(q,-np.pi/2)

    max_pq = max(p,q)
    min_pq = min(p,q)
    ndim = max_pq - min_pq + 1
    target_list = [0]*ndim
    target_list[:ndim-3] = list(range(min_pq+1, max_pq))
    target_list[ndim-2:] = p, q

    ### (1)   Exp[-i theta/4  Prod_{k=min_pq+1}^{max_pq-1} Z_k (Y_p Y_q)]
    pauli_index = [3]*ndim
    pauli_index[ndim-2:] = 2, 2
    gate = PauliRotation(target_list, pauli_index, -np.pi/2)
    circuit.add_gate(gate)
    ### (2)   Exp[-i theta/4  Prod_{k=min_pq+1}^{max_pq-1} Z_k (X_p X_q)]
    pauli_index[ndim-2:] = 1, 1
    gate = PauliRotation(target_list, pauli_index, -np.pi/2)
    circuit.add_gate(gate)

    cf.ncnot += (max_pq - min_pq) * 2 * 2

    return


def Gdouble_fswap(p, q, r, s, circuit, theta):
    """ Function:
    Construct exp[theta ( p!q!rs - s!r!qp ) ] as a whole unitary
    and add to circuit.
    Here, max(p, q, r, s) = p is assumed.
    There are 5 cases:
        (1) p > q > r > s (includes p > q > s > r)
        (2) p > r > q > s (includes p > r > s > q)
        (3) p = r > q > s (includes p = r > s > q)
        (4) p > q = r > s (includes p > q = s > r)
        (5) p > r > q = s (includes p > s > q = r)
    Note that in Qulacs, the function "PauliRotation" rotates as
            Exp [i theta/2 ...]
    so theta is divided by two.
    Accordingly, we need to multiply theta by two.

    Args:
        p (int): excitation index.
        q (int): excitation index.
        r (int): excitation index.
        s (int): excitation index.
        circuit (QuantumCircuit): circuit to be updated.
        theta (float): real rotation parameter.

    Returns:
        circuit (QuantumCircuit): circuit to be updated.

    Author(s): Takashi Tsuchimochi
    """
    if abs(theta) < cf.theta_threshold:
        return
    if p == q or r == s:
        prints(f"Caution:  p={p} == q={q}  or  r={r} == s={s}")
        return

    max_pqrs = max(p, q, r, s)
    max_pq = max(p, q)
    max_rs = max(r, s)

    if p == max_pqrs:
        # p > q. Test r and s
        if r == max_rs:
            parity = 1
            p, q, r, s = p, q, r, s
        else:
            parity = -1
            p, q, r, s = p, q, s, r
    elif q == max_pqrs:
        # q > p. Test r and s
        if r == max_rs:
            parity = -1
            p, q, r, s = q, p, r, s
        else:
            parity = 1
            p, q, r, s = q, p, s, r
    elif r == max_pqrs:
        # r > p, q, s. Test p and q
        if p == max_pq:
            parity = -1
            p, q, r, s = r, s, p, q
            #Note: This swapping means r!s!pq - p!q!rs = -(p!q!rs - s!r!qp)
            # So parity is -1 
        else:
            parity = 1
            p, q, r, s = r, s, q, p
            #Note: This swapping means r!s!qp - q!p!rs = p!q!rs - s!r!qp
            # So parity is 1 
    elif s == max_pqrs:
        # s > p, q, r. Test p and q
        if p == max_pq:
            parity = 1
            p, q, r, s = s, r, p, q
            #Note: This swapping means s!r!pq - q!p!rs = p!q!rs - s!r!qp
            # So parity is 1 
        else:
            parity = -1
            p, q, r, s = s, r, q, p
            #Note: This swapping means s!r!qp - p!q!rs = -(p!q!rs - s!r!qp)
            # So parity is -1 
    theta *= parity

    if p == r:
        if q > s:
            # p^ q^ p s  (p > q, p > s)
            #Gdoubles_pqps_fswap(p, q, s, circuit, theta, do_fswap=cf.fswap)
            Gdoubles_pqps(p, q, s, circuit, theta)
        elif q < s:
            # p^ q^ p s =  p^ s^ p q  (p > s, p > q)
            #Gdoubles_pqps_fswap(p, s, q, circuit, theta, do_fswap=cf.fswap)
            Gdoubles_pqps(p, s, q, circuit, theta)
        else:
            prints("Error!  p^r^ pr - h.c. = zero")
    elif p == s:  #(necessarily  r < s)
        #Gdoubles_pqps_fswap(p, q, r, circuit, -theta, do_fswap=cf.fswap)
        Gdoubles_pqps(p, q, r, circuit, -theta)
    elif q == r:
        if q > s:
            # p^ q^ q s  (p > q, q > s)
            #Gdoubles_pqqs_fswap(p, q, s, circuit, theta, do_fswap=cf.fswap)
            Gdoubles_pqqs(p, q, s, circuit, theta)
        elif q < s:
            # p^ q^ q s = - p^ q^ s q  (p > q, s > q)
            #Gdoubles_pqrq_fswap(p, q, s, circuit, -theta, do_fswap=cf.fswap)
            Gdoubles_pqrq(p, q, s, circuit, -theta)
    elif q == s:
        if q < r:
            # p^ q^ r q  (p > q, r > q)
            #Gdoubles_pqrq_fswap(p, q, r, circuit, theta, do_fswap=cf.fswap)
            Gdoubles_pqrq(p, q, r, circuit, theta)
        elif q > r:
            # p^ q^ r q  = - p^ q^ q r  (p > q, q > r)
            #Gdoubles_pqqs_fswap(p, q, r, circuit, -theta, do_fswap=cf.fswap)
            Gdoubles_pqqs(p, q, r, circuit, -theta)
    else:
        if r > s:
            Gdoubles_pqrs_fswap(p, q, r, s, circuit, theta, do_fswap=cf.fswap)
        elif r < s:
            Gdoubles_pqrs_fswap(p, q, s, r, circuit, -theta, do_fswap=cf.fswap)


def Gdoubles_pqrs_fswap(p, q, r, s, circuit, theta, approx=cf.approx_exp, do_fswap=False):
    """ Function
    Given
            Epqrs = p^ q^ r s
    with p > q, r > s and max(p, q, r, s) = p,
    compute Exp[i theta (Epqrs - Epqrs!)]

    Use FSWAP sequentially such that
         q <--> p-1
         s <--> r-1
    for p > q > r > s, and perform
    Exp[i theta Ep.p-1.r.r-1 - h.c.]

    This can reduce the number of CNOTs

    Author(s): Takashi Tsuchimochi
    """
    j4 = p  # (Fixed)
    if do_fswap:
        if q > r > s:
            j1, j2, j3 = s, r, q
        elif r > q > s:
            j1, j2, j3 = s, q, r
        elif r > s > q:
            j1, j2, j3 = q, s, r

        #for k in range(j1,j2-1):
        #    fswap(k, circuit)

        #for k in range(j3,i4-1):
        #    fswap(k, circuit)
        fswap_pq(j1, j2-1, circuit)
        fswap_pq(j4, j3+1, circuit)

        i4 = j3+1
        i3 = j3
        i2 = j2
        i1 = i2-1

        p = i4
        if q > r > s:
            q, r, s = i3, i2, i1
        elif r > q > s:
            r, q, s = i3, i2, i1
        elif r > s > q:
            r, s, q = i3, i2, i1
    else:
        i4 = p
        if q > r > s:
            i1, i2, i3 = s, r, q
        elif r > q > s:
            i1, i2, i3 = s, q, r
        elif r > s > q:
            i1, i2, i3 = q, s, r


    ndim1 = i2 - (i1+1)
    ndim2 = ndim1 + i4 - (i3+1)
    ndim = ndim2 + 4
    ncnot = (i4 - i3 + i2 - i1 + 1)*8*2
    ### This value is always 48 because p, q, r, s are now all adjacent (pair excitation). 
    ### However, this number can be reduced to 40 because some of CNOTS cancel out as gates like
    ####   p ... ---------.--H-H--.--------- ...
    ####   q ... -------.-*--H-H--*-.------- ...
    ####   r ... -----.-*----H-Y!---*-.----- ...
    ####   s ... -Rz--*------Y-H------*--Rz- ...
    ####     ( . and * imply control and target of CNOT)
    #### in which the CNOT-H-H-CNOT gates in p and q wires is identity (this appers between exp(Xp Xq Xr Ys) and exp(Xp Xq Ys Xr) )
    #### So, for a pair excitation, it is advisirable to perform rotations in the following order
    ####
    ####   Xp Xq Xr Ys -> Xp Xq Ys Xr -> Yp Yq Ys Xr -> Yp Yq Xs Yr -> Xp Yq Xr Xs -> Yp Xq Xr Xs -> Yp Xq Yr Ys -> Xq Yp Yr Ys
    ####              (1)            (2)            (3)            (4)            (5)            (6)            (7)
    ####
    #### At (1), the exact same cancelation happens as above. 
    #### At (3), a similar cancelation happens but with CNOT-Y-Y!-CNOT for p and q
    #### At (5), the same cancelation as (1) happens for r and s if the CNOT ladder direction is up-side-down (see below) 
    #### At (7), the same as (3)
    ####
    ####   p ... -Rz--*------H-Y!-----*--Rz--*- ...
    ####   q ... -----.-*----Y-H----*-.------.- ...
    ####   r ... -------.-*--H-H--*-.---------- ...
    ####   s ... ---------.--H-H--.------------ ...
    ####
    #### Here we assume the gate is prepared in this way, so the number of CNOTs is set to 40
    ###
    #### Actually, it can be simply 13

    cf.ncnot += 13 

    ### Type (a):
    ### (1)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p X_q X_r X_s)]
    ### (2)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p Y_q X_r X_s)]
    ### (3)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p Y_q X_r Y_s)]
    ### (4)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p Y_q Y_r X_s)]
    ### (5)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p X_q X_r Y_s)]
    ### (6)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p X_q Y_r X_s)]
    ### (7)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p X_q Y_r Y_s)]
    ### (8)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p Y_q Y_r Y_s)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(i1+1, i2))
    target_list[ndim1:ndim2] = list(range(i3+1, i4))
    target_list[ndim2:] = p, q, r, s
    pauli_index = [3]*ndim
    ### (1)              Yp,Xq,Xr,Xs
    pauli_index[ndim2:] = 2, 1, 1, 1
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (2)              Xp,Yq,Xr,Xs
    pauli_index[ndim2:] = 1, 2, 1, 1
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (3)              Yp,Yq,Xr,Ys
    pauli_index[ndim2:] = 2, 2, 1, 2
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (4)              Yp,Yq,Yr,Xs
    pauli_index[ndim2:] = 2, 2, 2, 1
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (5)              Xp,Xq,Xr,Ys
    pauli_index[ndim2:] = 1, 1, 1, 2
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)

    ### (6)              Xp,Xq,Yr,Xs
    pauli_index[ndim2:] = 1, 1, 2, 1
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)

    ### (7)              Yp,Xq,Yr,Ys
    pauli_index[ndim2:] = 2, 1, 2, 2
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)

    ### (8)              Xp,Yq,Yr,Ys
    pauli_index[ndim2:] = 1, 2, 2, 2
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)

    if do_fswap:
        # Back again
        fswap_pq(j1, j2-1, circuit)
        fswap_pq(j4, j3+1, circuit)


def Gdoubles_pqps_fswap(p, q, s, circuit, theta, do_fswap=False):
    """ Function
    Given
            Epqps = p^ q^ p s
    with p > q > s, compute Exp[i theta (Epqps - Epqps!)]

    Set s --> q-1
    """
    if do_fswap:
        fswap_pq(s,q-1,circuit)
        s1 = q-1
        #for k in range(s, q-1):
        #    fswap(k, circuit)
    else:
        s1 = s

    ndim1 = q - (s1+1)
    ndim = ndim1 + 3

    cf.ncnot += (q - s1 + 1)*2*2 ## (1) and (3) require CNOT(p,q)
    cf.ncnot += (q - s1)*2*2  ## (2) and (4)
    ### Type (b)
    ### (1)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k (Z_p X_q Y_s)]
    ### (2)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k (    Y_q X_s)]
    ### (3)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k (Z_p Y_q X_s)]
    ### (4)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k (    X_q Y_s)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(s1+1, q))
    target_list[ndim1:] = p, q, s1
    pauli_index = [3]*ndim

    ### (1)              Zp,Xq,Ys
    pauli_index[ndim1:] = 3, 1, 2
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (2)              Ip,Yq,Xs
    pauli_index[ndim1:] = 0, 2, 1
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (3)              Zp,Yq,Xs
    pauli_index[ndim1:] = 3, 2, 1
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    ### (4)              Ip,Xq,Ys
    pauli_index[ndim1:] = 0, 1, 2
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    if do_fswap:
        for k in range(q-2, s-1, -1):
            fswap(k, circuit)

def Gdoubles_pqqs_fswap(p, q, s, circuit, theta, do_fswap=False):
    """ Function
    Given
            Epqps = p^ q^ q s
    with p > q > s, compute Exp[i theta (Epqqs - Epqqs!)]

    Set s --> q-1
        p --> q+1
    No gain with FSWAP
    FSWAP: 8(q-1 - s)  + 8(p - (q+1)) = 8(p - s - 2)
    Excit: ((q+1) - (q-1))* 4 + ((q+1) - (q-1) - 1)*4 = 8 + 4 = 12
    Total: 8(p - s) - 4
    """
    if do_fswap:
        q1 = p-1
        for k in range(q, p-1):
            fswap(k, circuit)
        s1 = p-2
        for k in range(s, p-2):
            fswap(k, circuit)
        p1 = q+1 
        q1 = q 
        s1 = q-1
    else:
        p1 = p
        q1 = q
        s1 = s

    ndim1 = q1 - (s1+1)
    ndim2 = ndim1 + p1 - (q1+1)
    ndim = ndim2 + 3

    cf.ncnot += (p1 - s1)*2*2  ## (1) and (3)
    cf.ncnot += (p1 - s1 - 1)*2*2 ## (2) and (4) do not require CNOT(q+1,q) and CNOT(q,q-1) but CNOT(q+1,q-1)
    ### Type (c)
    ### (1)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (X_p Z_q Y_s)]  (p-s)*2
    ### (2)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (Y_p     X_s)]  (p-s-1)*2
    ### (3)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (Y_p Z_q X_s)]  (p-s)*2
    ### (4)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (X_p     Y_s)]  (p-s-1)*2
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(s1+1, q1))
    target_list[ndim1:ndim2] = list(range(q1+1, p1))
    target_list[ndim2:] = p1, q1, s1
    pauli_index = [3]*ndim

    ### (1)              Xp,Zq,Ys
    pauli_index[ndim2:] = 1, 3, 2
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (2)              Yp,Iq,Xs
    pauli_index[ndim2:] = 2, 0, 1
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (3)              Yp,Zq,Xs
    pauli_index[ndim2:] = 2, 3, 1
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    ### (4)              Xp,Iq,Ys
    pauli_index[ndim2:] = 1, 0, 2
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    if do_fswap:
        for k in range(p-3, s-1, -1):
            fswap(k, circuit)
        for k in range(p-2, q-1, -1):
            fswap(k, circuit)

def Gdoubles_pqrq_fswap(p, q, r, circuit, theta, do_fswap=False):
    """ Function
    Given
            Epqps = p^ q^ r q
    with p > r > q, compute Exp[i theta (Epqrq - Epqrq!)]
    
    Set r --> p-1
    No gain with FSWAP 
    FSWAP:  8(p-1 - r) 
    Excit:  8(p - p-1) + 4 
    Total:  8(p - r) + 4
    """
    if do_fswap:
        r1 = p-1
        for k in range(r, p-1):
            fswap(k, circuit)
    else:
        r1 = r


    ndim1 = p - (r1+1)
    ndim = ndim1 + 3

    cf.ncnot += (p - r1 + 1)*2*2  ## (1) and (3) require CNOT(r,q)
    cf.ncnot += (p - r1)*2*2  ## (2) and (4) 
    ### Type (d)
    ### (1)   Exp[ i theta/4  Prod_{k=r+1}^{p-1} Z_k  (X_p Z_q Y_r)]
    ### (2)   Exp[ i theta/4  Prod_{k=r+1}^{p-1} Z_k  (Y_p     X_r)]
    ### (3)   Exp[-i theta/4  Prod_{k=r+1}^{p-1} Z_k  (Y_p Z_q X_r)]
    ### (4)   Exp[-i theta/4  Prod_{k=r+1}^{p-1} Z_k  (X_p     Y_r)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(r1+1, p))
    target_list[ndim1:] = p, q, r1
    pauli_index = [3]*ndim

    ### (1)              Xp,Zq,Yr
    pauli_index[ndim1:] = 1, 3, 2
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (2)              Yp,Iq,Xr
    pauli_index[ndim1:] = 2, 0, 1
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (3)              Yp,Zq,Xr
    pauli_index[ndim1:] = 2, 3, 1
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    ### (4)              Xp,Iq,Yr
    pauli_index[ndim1:] = 1, 0, 2
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    if do_fswap:
        for k in range(p-2, r-1, -1):
            fswap(k, circuit)

def Gdoubles_pqrs_Ry(n_qubits,theta_, p,q,r,s):
    """
        p > q > r > s
        Small CNOT (see PHYSICAL REVIEW A 102, 062612 (2020).)
    """
    circuit = QuantumCircuit(n_qubits)
    circuit.add_CNOT_gate(p,q)
    circuit.add_CNOT_gate(r,s)
    circuit.add_CNOT_gate(p,r)
    
    if p-q > 1:
        lowest = q+1
    elif r-s > 1:
        lowest = s+1
    else:
        lowest = 0
        
    for k in range(p-1,q+1,-1):
        circuit.add_CNOT_gate(k,k-1)
        lowest = k-1
#        print(f"CNOT1 [{k, k-1}]")
    if lowest>s+1 and r-s>1:
        circuit.add_CNOT_gate(lowest,r-1)
        lowest = r-1
#        print(f"CNOT2 [{lowest, r-1}]")  
    for k in range(r-1,s+1,-1):
        circuit.add_CNOT_gate(k,k-1)
#        print(f"CNOT3 [{k, k-1}]")  
        lowest = k-1
    if p > lowest and lowest != 0:
        circuit.add_CZ_gate(p,lowest)
#        print(f"CZ [{p, lowest}]")    
    
    circuit.add_X_gate(q)
    circuit.add_X_gate(s)
    
    
    ### C-Ry
    circuit.add_RY_gate(p,theta_/8)
    circuit.add_H_gate(q)
    circuit.add_CNOT_gate(p,q)
    #
    circuit.add_RY_gate(p,-theta_/8)
    circuit.add_H_gate(s)
    circuit.add_CNOT_gate(p,s)
    #
    circuit.add_RY_gate(p,theta_/8)
    circuit.add_CNOT_gate(p,q)
    #
    circuit.add_RY_gate(p,-theta_/8)
    circuit.add_H_gate(r)
    circuit.add_CNOT_gate(p,r)
    #
    circuit.add_RY_gate(p,theta_/8)
    circuit.add_CNOT_gate(p,q)
    #
    circuit.add_RY_gate(p,-theta_/8)
    circuit.add_CNOT_gate(p,s)
    #
    circuit.add_RY_gate(p,theta_/8)
    circuit.add_CNOT_gate(p,q)
    circuit.add_H_gate(s)
    #
# original    
#    circuit.add_RY_gate(p,-theta_/8)
#    circuit.add_H_gate(q)
#    circuit.add_CNOT_gate(p,r)
#    circuit.add_H_gate(r)
# reduced
    circuit.add_RY_gate(p,-theta_/8)
    circuit.add_H_gate(q)
    circuit.add_RZ_gate(r,-np.pi/2)
    circuit.add_CNOT_gate(p,r)
    circuit.add_RZ_gate(p,np.pi/2)
    circuit.add_RZ_gate(r,-np.pi/2)    
    circuit.add_RY_gate(r,-np.pi/2)    

    ### 
    
    circuit.add_X_gate(q)
    circuit.add_X_gate(s)
    
    
    if p > lowest and lowest != 0:
        circuit.add_CZ_gate(p,lowest)
#        print(f"CZ [{p, lowest}]")        

    circuit.add_CNOT_gate(p,q)
    circuit.add_CNOT_gate(r,s)
    
    highest = s+1
    for k in range(s+1,r-1):
        circuit.add_CNOT_gate(k+1,k)
        highest = k+1
#        print(f"CNOT3 [{k+1, k}]")   
    if p-q >1 and r-s>1:
        circuit.add_CNOT_gate(q+1,highest)
#        print(f"CNOT2 [{q+1, highest}]")    
        highest = q+1    
    for k in range(q+1,p-1):
        circuit.add_CNOT_gate(k+1,k)
#        print(f"CNOT1 [{k+1, k}]")    
    return circuit

def count_CNOT_Gdoubles_pqrs_Ry(p,q,r,s):
    """
        Test subrouitine to count # of CNOTs for Gdoubles_pqrs_Ry.
        p > q > r > s
        Always
        2 * (p-q+r-s) + 9
    """
#    circuit.add_CNOT_gate(p,q)
#    circuit.add_CNOT_gate(r,s)
#    circuit.add_CNOT_gate(p,r)
    ncnot = 3
    if p-q > 1:
        lowest = q+1
    elif r-s > 1:
        lowest = s+1
    else:
        lowest = 0
        
    for k in range(p-1,q+1,-1):
#        circuit.add_CNOT_gate(k,k-1)
        ncnot += 1
        lowest = k-1
#        print(f"CNOT1 [{k, k-1}]")
    if lowest>s+1 and r-s>1:
#        circuit.add_CNOT_gate(lowest,r-1)
        ncnot += 1        
        lowest = r-1
#        print(f"CNOT2 [{lowest, r-1}]")  
    for k in range(r-1,s+1,-1):
#        circuit.add_CNOT_gate(k,k-1)
        ncnot += 1        
#        print(f"CNOT3 [{k, k-1}]")  
        lowest = k-1
    if p > lowest and lowest != 0:
#        circuit.add_CZ_gate(p,lowest)
        ncnot += 1        
#        print(f"CZ [{p, lowest}]")    
    
#    circuit.add_X_gate(q)
#    circuit.add_X_gate(s)
    
    
    ### C-Ry
#    circuit.add_RY_gate(p,theta_/8)
#    circuit.add_H_gate(q)
#    circuit.add_CNOT_gate(p,q)
#    #
#    circuit.add_RY_gate(p,-theta_/8)
#    circuit.add_H_gate(s)
#    circuit.add_CNOT_gate(p,s)
#    #
#    circuit.add_RY_gate(p,theta_/8)
#    circuit.add_CNOT_gate(p,q)
#    #
#    circuit.add_RY_gate(p,-theta_/8)
#    circuit.add_H_gate(r)
#    circuit.add_CNOT_gate(p,r)
#    #
#    circuit.add_RY_gate(p,theta_/8)
#    circuit.add_CNOT_gate(p,q)
#    #
#    circuit.add_RY_gate(p,-theta_/8)
#    circuit.add_CNOT_gate(p,s)
#    #
#    circuit.add_RY_gate(p,theta_/8)
#    circuit.add_CNOT_gate(p,q)
#    circuit.add_H_gate(s)
#    #
## original    
##    circuit.add_RY_gate(p,-theta_/8)
##    circuit.add_H_gate(q)
##    circuit.add_CNOT_gate(p,r)
##    circuit.add_H_gate(r)
## reduced
#    circuit.add_RY_gate(p,-theta_/8)
#    circuit.add_H_gate(q)
#    circuit.add_RZ_gate(r,-np.pi/2)
#    circuit.add_CNOT_gate(p,r)
#    circuit.add_RZ_gate(p,np.pi/2)
#    circuit.add_RZ_gate(r,-np.pi/2)    
#    circuit.add_RY_gate(r,-np.pi/2)    
#
#    ### 
#    
#    circuit.add_X_gate(q)
#    circuit.add_X_gate(s)
    ncnot += 8
    
    if p > lowest and lowest != 0:
#        circuit.add_CZ_gate(p,lowest)
        ncnot += 1
#        print(f"CZ [{p, lowest}]")        

#    circuit.add_CNOT_gate(p,q)
#    circuit.add_CNOT_gate(r,s)
    ncnot += 2
    highest = s+1
    for k in range(s+1,r-1):
#        circuit.add_CNOT_gate(k+1,k)
        ncnot += 1
        highest = k+1
#        print(f"CNOT3 [{k+1, k}]")   
    if p-q >1 and r-s>1:
#        circuit.add_CNOT_gate(q+1,highest)
        ncnot += 1
#        print(f"CNOT2 [{q+1, highest}]")    
        highest = q+1    
    for k in range(q+1,p-1):
#        circuit.add_CNOT_gate(k+1,k)
        ncnot += 1
#        print(f"CNOT1 [{k+1, k}]")     
    return ncnot

def make_gate(n, index, pauli_id):
    """Function
    Make a gate for Pauli string like X0Y1Z2...

    Args:
        n (int): Number of qubits
        index (list): Index of Qubits to be operated.
        pauli_id (list): Index of 0,1,2,3 or I,X,Y,Z.

    Returns:
        circuit (QuantumCircuit)

    Example:
        If the target gate is X0 Y1 Z3 Y5 Z6 and the
        total number of qubits is 8, then set

        n = 8
        index = [0, 1, 3, 5, 6]
        pauli_id = ['X', 'Y', 'Z', 'Y', 'Z']
        or
        pauli_id = [1, 2, 3, 2, 3]
    """
    circuit = QuantumCircuit(n)
    for i in range(len(index)):
        gate_number = index[i]
        if pauli_id[i] == 1 or pauli_id[i] == 'X':
            circuit.add_X_gate(gate_number)
        elif pauli_id[i] == 2 or pauli_id[i] == 'Y':
            circuit.add_Y_gate(gate_number)
        elif pauli_id[i] == 3 or pauli_id[i] == 'Z':
            circuit.add_Z_gate(gate_number)
    return circuit

def Pauli2Circuit(n, Pauli, circuit=None):
    """Function
    Given a list of Pauli operators in the format of OpenFermion,
    for example, ((0, 'X'), (1, 'X'), (2, 'X'), (3, 'X'), (6, 'X'), (8, 'X')),
    create a gate circuit using qulacs.
    """
    if circuit is None:
        circuit = QuantumCircuit(n)
    for ibit, xyz in Pauli:
        if xyz == 'X':
            circuit.add_X_gate(ibit)
        elif xyz == 'Y':
            circuit.add_Y_gate(ibit)
        elif xyz == 'Z':
            circuit.add_Z_gate(ibit)
    return circuit

def set_exp_circuit(n_qubits, pauli_list, theta_list, rho=1):
    """Function
    Set the circuit
         prod exp[i theta pauli] 
    using qulacs' PauliRotation
    
    Args:
        n_qubits (int): Number of qubits.
        pauli_list (list): List of pauli strings
        theta_list (ndarray): List of theta (should match the dimensions!)
        rho (int): Trotter number
    Returns:
        circuit (QuantumCircuit):
    """
    def pauli_rotation_gate(pauli, theta, rho, circuit):
        for op, coef in pauli.terms.items():
            target_list = []
            pauli_index = []
            for op_ in op:
                m = op_[0]
                if op_[1] == 'X':
                    target_list.append(m)
                    pauli_index.append(1)
                elif op_[1] == 'Y':
                    target_list.append(m)
                    pauli_index.append(2)
                elif op_[1] == 'Z':
                    target_list.append(m)
                    pauli_index.append(3)        
            # PauliRotation does exp[i theta/2 P]
            if coef.real:
                gate = PauliRotation(target_list, pauli_index, 2*coef.real*theta/rho)
            elif coef.imag:
                gate = PauliRotation(target_list, pauli_index, 2*coef.imag*theta/rho)
            else:
                continue
            circuit.add_gate(gate)
            cf.ncnot += 2*(len(op) - 1)

    circuit = QuantumCircuit(n_qubits)
    for pauli, theta in zip(pauli_list, theta_list):
        if abs(theta) > cf.theta_threshold:
            if type(pauli) == list:
                ### pauli contains a series of pauli operators
                ### [[pauli_1], [pauli_2], ...]
                ### where pauli_1 and pauli_2 ... are a linear combination of
                ### pauli strings and they may not commute.
                ### These operators need to have the same phase (real/imag)
                for pauli_ in pauli:
                    pauli_rotation_gate(pauli_, theta, rho, circuit) 
            else:
                pauli_rotation_gate(pauli, theta, rho, circuit) 
    return circuit

def create_exp_state(Quket, init_state=None, pauli_list=None, theta_list=None, rho=None, overwrite=False):
    """
    Given pauli_list and theta_list, create an exponentially parameterized quantum state
     
         prod_i Exp[ (theta[i] * pauli[i]) ] |0>

    where |0> is the initial state. 

    Args:
        Quket: QuketData class instance
        init_state (optional): initial quantum state
        pauli_list (optional): List of paulis as openfermion.QubitOperator
        theta_list (optional): List of parameters theta
        rho (optional): Trotter number
        overwrite (optional): if init_state is provided, overwrite it without copy.
    Returns:
        
    Author(s): Takashi Tsuchimochi
    """
    n_qubits = Quket.n_qubits
    if init_state is None:
        #### Form a product state from integer 'current_det'
        #state = QuantumState(n_qubits)
        #state.set_computational_basis(Quket.current_det)
        state = Quket.init_state.copy()
    elif overwrite:
        state = init_state
    else:
        ### Copy the initial state
        state = init_state.copy()

    if pauli_list is None:
        pauli_list = Quket.pauli_list

    if theta_list is None:
        theta_list = Quket.theta_list

    if rho is None:
        rho = Quket.rho


    if len(pauli_list) != len(theta_list):
        raise Exception(f'Dimensions of pauli_list ({len(pauli_list)}) and theta_list ({len(theta_list)}) not consistent.')

    ### 
    circuit = set_exp_circuit(n_qubits, pauli_list, theta_list, rho)
    for i in range(rho):
        circuit.update_quantum_state(state)

    return state

