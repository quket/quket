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

utils.py

Utilities.

"""
import time
import numpy as np

def is_1bit(num, n):
    if num & (1 << n):
        return True
    return False    

def jw2bk(intjw, n_qubits):
    """Function
    Convert a bit string from jordan-wigner to bravyi-kitaev.
    The bit string may be given by base-10 integer.
    """
    intbk = 0
    parity = 1
    parity_sub = 1
    for i in range(n_qubits):
        k = intjw//2**(i) % 2
        parity *= (-1)**k
        parity_sub *= (-1)**k        
        if i % 2 == 0:
            #Where i is even, qubit i stores the occupation number of orbital i, as in the Jordan−Wigner mapping.
            intbk += k * 2**i    
        elif  (np.log2(i+1)).is_integer():
            #When log2 (i+1) is an integer, the qubit stores the parity of the occupation numbers of all orbitals with indices less than or equal to i. 
            intbk += (1-parity)/2 * 2**i
            parity_sub = 1 ### Reset            
        else:
            # For other cases, the qubit stores the parity of the occupation numbers of orbitals in subdividing binary sets.
            intbk += (1-parity_sub)/2 * 2**i
    return int(intbk)

def bk2jw(intbk, n_qubits):
    """Function
    Convert a bit string from bravyi-kitaev to jordan-wigner.
    The bit string may be given by base-10 integer.
    """
    intjw = 0
    occ = 0
    sub_occ = 0
    for i in range(n_qubits):
        k = intbk//2**(i) % 2      
        if i % 2 == 0:
            #Where i is even, qubit i stores the occupation number of orbital i, as in the Jordan−Wigner mapping.
            intjw += k * 2**i    
            occ += k
            sub_occ += k
        elif  (np.log2(i+1)).is_integer():
            kk = (k + occ) % 2
            #When log2 (i+1) is an integer, the qubit stores the parity of the occupation numbers of all orbitals with indices less than or equal to i. 
            intjw +=  kk * 2**i         
            occ += kk
            sub_occ = 0  ### Reset                 
        else:
            # For other cases, the qubit stores the parity of the occupation numbers of orbitals in subdividing binary sets.
            kk = (k + sub_occ) % 2
            intjw += kk * 2**i   
            sub_occ += kk
            occ += kk
    return int(intjw)


def pauli_bit_multi(pauli, bit_string):
    """
    Apply `pauli` in the form of OpenFermion, e.g., 
       pauli = ((0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'X'))
    to a bit_string `bit_string`.

    Args:
        pauli :  A tuple of (qubit, 'XYZ')
        bit_string : A base-10 integer to represent a bit string 
    Returns:
        bit_string_ : The transformed bit_string obtained by pauli * bit_string = phase * bit_string_
        phase : The phase
    """
    bit_string_ = bit_string
    phase = 1
    for k, XYZ in pauli:
        if XYZ == 'X':
            bit_string_ = bit_string_ ^ (1 << k)  # Flipping the bit_string k
        elif  XYZ == 'Y':
            if is_1bit(bit_string_, k):
                phase *= -1j
            else:
                phase *= 1j
            bit_string_ = bit_string_ ^ (1 << k) # Flipping the bit_string k
        elif XYZ == 'Z':
            if is_1bit(bit_string_, k): 
                phase *= -1
    return bit_string_, phase 

def append_01qubits(state, nc, ns):
    """Function
    Append |0> and/or |1> qubits to a quantum state. 

    Args:
        state (QuantumState): Quantum state with n qubits, 
                              and will turn into one with n + n_append qubits.
        nc (int): Number of 1 qubits appended to the beginning of state.
        ns (int): Number of 0 qubits appended to the end of state.

    Returns:
        state_appended (QuantumState): Quantum state with n + nc + ns qubits.
    """
    n_qubits = state.get_qubit_count()
    state_vec = state.get_vector()
    state_appended = QuantumState(n_qubits + nc + ns)

    indices = np.arange(2**n_qubits)
    indices = (indices << nc*2) + (2**(nc*2) - 1)
    vec = np.zeros(2**(n_qubits + nc + ns), complex)
    vec[indices] = state_vec
    state_appended.load(vec)
    return state_appended

