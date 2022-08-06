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
import re
import time
import copy

import os
import sys

import numpy as np
import scipy as sp
from numpy import linalg as LA
from scipy.linalg import expm, logm
from qulacs import QuantumCircuit
from qulacs.state import inner_product
from openfermion.ops import InteractionOperator

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import prints, printmat, error, print_state
from quket.lib import (
    QuantumState, 
    QubitOperator, 
    FermionOperator,
    hermitian_conjugated, 
    commutator,
    get_fermion_operator, 
    jordan_wigner,
    bravyi_kitaev,
    normal_ordered
    )
from .bit import jw2bk, bk2jw



def chkbool(string):
    """Function:
    Check string and return True or False as bool

    Author(s): Takashi Tsuchimochi
    """
    if type(string) is bool: 
        return string
    elif type(string) is str:
        if string.lower() in ("true", "1"):
            return True
        elif string.lower() in ("false", "0"):
            return False
        else:
            error(f"Unrecognized argument '{string}'")
    else:
        error(f"Argument '{string}' needs to be either bool or string.")


def chkmethod(method, ansatz):
    """Function:
    Check method is available in method_list

    Author(s): Takashi Tsuchimochi
    """
    if ansatz is None:
        return True

    if method in ("vqe", "mbe", "variational_imaginary", "variational_real"):
        if (ansatz in cf.vqe_ansatz_list) \
        or ("pccgsd" in ansatz) or ("bcs" in ansatz):
            return True
        else:
            return False
    elif method in ("qite", "qlanczos"):
        return True if ansatz in cf.qite_ansatz_list else False
    else:
        return False

def chkpostmethod(method):
    """Function:
    Check post_method is available in post_method_list

    Author(s): Takashi Tsuchimochi
    """
    if method == "none":
        return True

    if method in cf.post_method_list:
        return True
    else:
        return False

def chkint(string):
    if type(string) is int:
        return True
    elif type(string) is str:
        if string.isdecimal():
            return True
    else:
        return False

def chknaturalnum(string):
    if type(string) is int:
        return string >= 0
    elif type(string) is str:
        if string.isdecimal() and int(string) >= 0:
            return True
    else:
        return False

def isint(s): 
    try:
        int(s)  
    except ValueError:
        return False
    else:
        return True

def isfloat(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

def iscomplex(s):
    try:
        complex(s)
    except ValueError:
        return False
    else:
        return True

def chkdet(det, noa, nob):
    """
    Check if integer det has the correct numbers of alpha and beta electrons.
    """
    bindet = bin(det)[2:]
    my_noa = 0
    my_nob = 0
    # count alpha and beta electrons in det.
    for k, i in enumerate(bindet):
        if k%2:
            my_noa += int(i)
        else:
            my_nob += int(i)
    return my_noa == noa and my_nob == nob

def chk_energy(QuketData):
    """Function
        Calculate energy from 1RDM and 2RDM and check if this is the same as QuketData.energy.

    Author(s): Takashi Tsuchimochi
    """
    from quket.post import calc_energy
    Eaa = Ebb = 0
    Eaaaa = Ebbbb = Ebaab = 0
    Ecore = 0
    E_nuc = QuketData.nuclear_repulsion
    ncore = QuketData.n_frozen_orbitals
    if (QuketData.DA is not None
            and QuketData.DB is not None
            and QuketData.Daaaa is not None
            and QuketData.Dbbbb is not None
            and QuketData.Dbaab is not None
            and QuketData.energy is not None):
        if abs(calc_energy(QuketData) - QuketData.energy) > 1e-8:
            return False
        else:
            return True
    else:
        return False

def is_commute(op1, op2):
    if isinstance(op1, QubitOperator): 
        return commutator(op1, op2) == QubitOperator('',0)
    if isinstance(op1, FermionOperator): 
        return normal_ordered(commutator(op1, op2)) == FermionOperator('',0)
    
    raise TypeError(f'commutator is not applicable to {type(op1)} in is_commute().')

def fermi_to_str(fermion_operator):
    """
    Convert FermionOperator to string list.
    """
    if isinstance(fermion_operator, (openfermion.FermionOperator, FermionOperator)):
        string = str(normal_ordered(fermion_operator))
    else:
        string = str(normal_ordered(get_fermion_operator(fermion_operator)))
    #print(string)
    string_list = string.split("\n")

    hamiltonian_list = []
    for s in string_list:
        if "(" in s:
            coef = re.findall(r"(?<=\().+?(?=\))", s)[0]
            coef = re.findall(r"[-+]?[0-9]*.?[0-9]+", coef)[0].split("+")[0]
        else:
            coef = re.findall(r"[-+]?[0-9]*.?[0-9]+", s)[0]
        operator = re.findall(r"(?<=\[).+?(?=\])", s)
        operator = "" if len(operator) == 0 else operator[0]
        hamiltonian_list.append([coef, operator])
    return hamiltonian_list

def qubit_to_str(qubit_operator):
    string = str(qubit_operator).replace("]", "")
    hamiltonian_list = [x.strip().split("[")
                        for x in string.split("+")
                        if not x.strip() == ""]
    return hamiltonian_list

def FlipBitOrder(i, n_qubits):
    """Function
    Flip the ordering of bits as
        |01234 ... N>    -->   |N ... 43210>

    Note; format(23, '08b') = '00010111'
          format(23, '08b')[::-1] = '11101000'
          int(format(23, '08b')[::-1], 2) = 232
    """
    return int(format(i, f"0{n_qubits}b")[::-1], 2)

def orthogonal_constraint(Quket, state):
    """Function
    Compute the penalty term for excited states based on 'orthogonally-constrained VQE' scheme.
    """
    if not hasattr(Quket, 'lower_states'):
        Quket.lower_states = []
    nstates = len(Quket.lower_states)
    extra_cost = 0
    for i in range(nstates):
        Ei = Quket.lower_states[i]['energy']
        overlap = inner_product(Quket.lower_states[i]['state'], state)
        extra_cost += -Ei * abs(overlap)**2
    return extra_cost

def set_initial_det(noa, nob):
    """ Function
    Set the initial wave function to RHF/ROHF determinant.

    Author(s): Takashi Tsuchimochi
    """
    # Note: r'~~' means that it is a regular expression.
    # a: Number of Alpha spin electrons
    # b: Number of Beta spin electrons
    if noa >= nob:
        # Here calculate 'a_ab' as follow;
        # |r'(01){a-b}(11){b}'> wrote by regular expression.
        # e.g.)
        #   a=1, b=1: |11> = |3>
        #   a=3, b=1: |0101 11> = |23> = |3 + 5/2*2^3>
        # r'(01){a-b}' = (1 + 4 + 16 + ... + 4^(a-b-1))/2
        #              = (4^(a-b) - 1)/3
        # That is, it is the sum of the first term '1'
        # and the geometric progression of the common ratio '4'
        # up to the 'a-b' term.
        base = nob*2
        a_ab = (4**(noa-nob) - 1)//3
    elif noa < nob:
        # Here calculate 'a_ab' as follow;
        # |r'(10){b-a}(11){a}'> wrote by regular expression.
        # e.g.)
        #   a=1, b=1: |11> = |3>
        #   a=1, b=3: |1010 11> = |43> = |3 + 5*2^3>
        # r'(10){b-a}' = 2 + 8 + 32 + ... + 2*4^(b-a-1)
        #              = 2 * (4^(b-a) - 1)/3
        # That is, it is the sum of the first term '2'
        # and the geometric progression of the common ratio '4'
        # up to the 'a-b' term.
        base = noa*2
        a_ab = 2*(4**(nob-noa) - 1)//3
    return 2**base-1 + (a_ab<<base)

def set_multi_det_state(det_info, n_qubits):               
    """
    Create n_qubits QuantumState with det_info.

    Args:
        det_info (list): contains the weights (float) and determinants (int) in the following format
                        [[weight, det], [weight, det], ...]
        n_qubits (int): Number of qubits.

    Returns:
        state (QuantumState): Resulting (entangled) quantum state.
    """
    state = QuantumState(n_qubits)
    state.multiply_coef(0)
    for state_ in det_info:
        x = QuantumState(n_qubits)
        x.set_computational_basis(state_[1])
        x.multiply_coef(state_[0])
        state.add_state(x)
    norm2 = state.get_squared_norm()
    state.normalize(norm2)
    return state

def int2occ(state_int):
    """ Function
    Given an (base-10) integer, find the index for 1 in base-2 (occ_list)

    Author(s): Takashi Tsuchimochi
    """
    # Note: bin(23) = '0b010111'
    #       bin(23)[-1:1:-1] = '111010'
    # Same as; bin(23)[2:][::-1]
    # Occupied orbitals denotes '1'.
    occ_list = [i for i, k in enumerate(bin(state_int)[-1:1:-1]) if k == "1"]
    return occ_list

def get_occvir_lists(n_qubits, det):
    """Function
    Generate occlist and virlist for det (base-10 integer).

    Author(s): Takashi Tsuchimochi
    """
    occ_list = int2occ(det)
    vir_list = [i for i in range(n_qubits) if i not in occ_list]
    return occ_list, vir_list

def get_func_kwds(func, kwds):
    import inspect

    sig = inspect.signature(func).parameters
    init_dict = {s: kwds[s] for s in sig if s in kwds}
    return init_dict

def Gdoubles_list(norbs, singles=True, spinfree=False):
    """Function
    Compute the non-redundant list for general doubles.

    Args:
        norbs (int): number of spatial orbitals
        singles (bool): effective singles are treated (like p^ q^ q s) 
        spinfree (bool): whether or not pA^ qB^ pB qA like excitations are included (spin-breaking)
    Returns:
        r_list (list): Spin-free combinations, [p,q,r,s] with p > q, r > s
        u_list (list): Set of alpha and beta excitations for each spin-free pqrs
        parity_list (list): Set of parities for the excitations in u_list
    """
    # r_list contains the non-redundant spin-free general excitations
    # in spatial-orbital basis
    r_list = []
    u_list = []
    pq = 0
    for p in range(norbs):
        for q in range(p+1):
            rs = 0
            for r in range(norbs):
                for s in range(r+1):
                    if pq < rs or (p == q == r == s):
                        rs += 1
                        continue
                    if not singles and (p == r or q == s):
                        rs += 1
                        continue
                    r_list.append([p, q, r, s])
                    rs += 1
            pq += 1

    # u_list contains the non-redundant spin-free general excitations
    # in spin-orbital basis
    u_list = []
    parity_list = []
    for ilist in range(len(r_list)):
        p, q, r, s = r_list[ilist]
        tmp_ = []
        parity_ = []
        pA = p*2
        qA = q*2
        rA = r*2
        sA = s*2
        pB = pA + 1
        qB = qA + 1
        rB = rA + 1
        sB = sA + 1
        if p != q and r !=s:
            ### AAAA and BBBB
            if not (p == r and q == s):
                i1 = max(pA, qA)
                i2 = min(pA, qA)
                i3 = max(rA, sA)
                i4 = min(rA, sA)
                tmp = [i1, i2, i3, i4]
                tmp_.append(tmp)
                parity_.append(1)
                i1 = max(pB, qB)
                i2 = min(pB, qB)
                i3 = max(rB, sB)
                i4 = min(rB, sB)
                tmp = [i1, i2, i3, i4]
                tmp_.append(tmp)
                parity_.append(1)
            if p == r and q == s:
            ### BABA and ABAB
                if not spinfree:
                    # pA^ qB^ pB qA
                    i1 = max(pA, qB)
                    i2 = min(pA, qB)
                    i3 = max(rB, sA)
                    i4 = min(rB, sA)
                    if i1 > i3:
                        tmp = [i1, i2, i3, i4]
                    else:
                        tmp = [i3, i4, i1, i2]
                    tmp_.append(tmp)
                    parity_.append(int((-1)**(i1%2 + i4%2)))
            else:
            ### BABA and ABAB
                i1 = max(pB, qA)
                i2 = min(pB, qA)
                i3 = max(rB, sA)
                i4 = min(rB, sA)
                tmp = [i1, i2, i3, i4]
                tmp_.append(tmp)
                parity_.append(int((-1)**(i1%2 + i4%2)))

                i1 = max(pA, qB)
                i2 = min(pA, qB)
                i3 = max(rA, sB)
                i4 = min(rA, sB)
                tmp = [i1, i2, i3, i4]
                tmp_.append(tmp)
                parity_.append(int((-1)**(i1%2 + i4%2)))

            ### BAAB and ABBA
                i1 = max(pA, qB)
                i2 = min(pA, qB)
                i3 = max(rB, sA)
                i4 = min(rB, sA)
                #if i1 > i3:
                #    tmp = [i1, i2, i3, i4]
                #else:
                #    tmp = [i3, i4, i1, i2]
                tmp = [i1, i2, i3, i4]
                tmp_.append(tmp)
                parity_.append(int((-1)**(i1%2 + i4%2)))
                i1 = max(pB, qA)
                i2 = min(pB, qA)
                i3 = max(rA, sB)
                i4 = min(rA, sB)
                #if i1 > i3:
                #    tmp = [i1, i2, i3, i4]
                #else:
                #    tmp = [i3, i4, i1, i2]
                tmp = [i1, i2, i3, i4]
                tmp_.append(tmp)
                parity_.append(int((-1)**(i1%2 + i4%2)))
        elif p == q and r != s:
            tmp = [pB, qA, sA, rB]
            tmp_.append(tmp)
            parity_.append(1)
            tmp = [pB, qA, rA, sB]
            tmp_.append(tmp)
            parity_.append(1)
        elif p != q and r == s:
            tmp = [pB, qA, sA, rB]
            tmp_.append(tmp)
            parity_.append(1)
            tmp = [pA, qB, rB, sA]
            tmp_.append(tmp)
            parity_.append(1)
        elif p == q and r == s and p != r and q != s:
            tmp = [pB, qA, rB, sA]
            tmp_.append(tmp)
            parity_.append(1)

        if len(tmp_) > 0:
            u_list.append(tmp_)
            parity_list.append(parity_)
    return r_list, u_list, parity_list


def order_pqrs(p,q,r,s):
    """
    Reorder p,q,r,s in descending order
    """
    pqrs = sorted([p,q,r,s], reverse=True)
    return pqrs[0], pqrs[1], pqrs[2], pqrs[3]

def get_OpenFermion_integrals(Hamiltonian, n_orbitals):
    """
    get zero, one, and two-body integrals from Hamiltonian (openfermion class)
    'n_orbitals' is supposed to be n_active_orbitals used in the qubit simulation.
    """
    const = 0
    one_body_integrals = np.zeros((n_orbitals, n_orbitals), dtype=float)
    # This is so that two_body_integrals[p,q,r,s] contains h_pqrs = (pq|rs)
    # instead of the weird ordering of OpenFermion (ps|qr)
    two_body_integrals = np.zeros((n_orbitals, n_orbitals,
                                   n_orbitals, n_orbitals), dtype=float)
    #sys.stdout.flush()
    Hamiltonian = normal_ordered(Hamiltonian)
    if type(Hamiltonian) is InteractionOperator:
        from quket.lib import get_fermion_operator
        Hamiltonian = get_fermion_operator(Hamiltonian)
    for op, c in Hamiltonian.terms.items():
        if len(op) == 0:
            const = c
        elif len(op) == 2:
            p = op[0][0]//2
            q = op[1][0]//2
            one_body_integrals[p,q] = c
        elif len(op) == 4:
            p = op[0][0]//2
            p_spin = op[0][0]%2
            q = op[1][0]//2
            q_spin = op[1][0]%2
            r = op[2][0]//2
            r_spin = op[2][0]%2
            s = op[3][0]//2
            s_spin = op[3][0]%2
            if p_spin != q_spin and r_spin != s_spin:
                if p_spin == s_spin:
                    r_ = r
                    r  = s
                    s  = r_
                    c  = -c
                two_body_integrals[p,r,q,s] = -c
                two_body_integrals[r,p,q,s] = -c
                two_body_integrals[p,r,s,q] = -c
                two_body_integrals[r,p,s,q] = -c
                two_body_integrals[q,s,p,r] = -c
                two_body_integrals[q,s,r,p] = -c
                two_body_integrals[s,q,p,r] = -c
                two_body_integrals[s,q,r,p] = -c
    return const, one_body_integrals, two_body_integrals

def generate_general_openfermion_operator(h0, h1, h2):
    norbs = h1.shape[0] 
    if norbs != h1.shape[1] or norbs != h2.shape[0] or \
       norbs != h2.shape[1] or norbs != h2.shape[2] or \
       norbs != h2.shape[1]:
       raise ValueError("Wrong dimensions of h1 and h2.")
   
    h0_op = FermionOperator('', h0)
    h1_op = FermionOperator('', 0)
    for p in range(norbs):
        for q in range(norbs):
            coef = h1[p,q]
            h1_op += FermionOperator(((2*p, 1), (2*q, 0)), coef)
            h1_op += FermionOperator(((2*p+1, 1), (2*q+1, 0)), coef)

    h2_op = FermionOperator('', 0)
    for p in range(norbs):
        for q in range(norbs):
            for r in range(norbs):
                for s in range(norbs):
                    j = 0.5 * h2[p,q,r,s] # (pq|rs) = <pr|qs> p! r! s q
                    h2_op += FermionOperator(((2*p, 1), (2*r, 1), (2*s, 0), (2*q, 0)), j)
                    h2_op += FermionOperator(((2*p+1, 1), (2*r+1, 1), (2*s+1, 0), (2*q+1, 0)), j)
                    h2_op += FermionOperator(((2*p+1, 1), (2*r, 1), (2*s, 0), (2*q+1, 0)), j)
                    h2_op += FermionOperator(((2*p, 1), (2*r+1, 1), (2*s+1, 0), (2*q, 0)), j)
    normal_ordered(h2_op)
    return normal_ordered(h0_op + h1_op + h2_op)

def to_pyscf_geom(geometry, pyscf_geom):
    geom = [] 
    for iatom in range(len(pyscf_geom)):
        geom.append((pyscf_geom[iatom][0],
                    (geometry[iatom,0],
                     geometry[iatom,1],
                     geometry[iatom,2])))
    return geom

def remove_unpicklable(instance):
    """Function
    Remove unpicklable objects from instance.
    """
    def is_picklable(v):
        import pickle
        try:
            x=pickle.dumps(v)
            return True
        except:
            return False
    state = {}
    if type(instance) is dict:
        for k, v in instance.items():
            if is_picklable(v):
                state[k] = v
    else:
        for k, v in instance.__dict__.items():
            if (hasattr(v, '__dict__')):
                # May contain QuantumStates
                state_ = {}
                for k_, v_ in v.__dict__.items():
                    if is_picklable(v_):
                        state_[k_] = v_
                state[k] = state_
                    
            elif is_picklable(v):
                state[k] = v
    return state

def pauli_index(op_tuple):
    """Function
    Label pauli with a unique integer number.
    pauli has to be a single pauli string.

    Args:
        op_tuple (tuple): Pauli string to be labeled in QubitOperator format. e.g.,
                       ((0, 'Z'), (1, 'Y'), (2, 'Z'), (4, 'Z'), (6, 'Z'), (8, 'Z'), (10, 'Z'), (12, 'Z'))
    Returns:
        ind (int): identification number for pauli
    """
    ind = 0
    for op_ in op_tuple:
        if op_[1] == 'X':
            ind += (4**op_[0])
        elif op_[1] == 'Y':
            ind += 2*(4**op_[0])
        elif op_[1] == 'Z':
            ind += 3*(4**op_[0])
    return ind


def get_pauli_index_and_coef(pauli):
    """Function
    Given QubitOperator, check the identification numbers of each pauli string and its coefficient,
    make a list.
    """
    ind_list = []
    coef_list = []    
    for op, coef in pauli.terms.items():
        ind = pauli_index(op)
        ind_list.append(ind)
        coef_list.append(coef)
    return ind_list, coef_list

def get_unique_pauli_list(pauli_list):
    """Function
    Given redundant pauli_list, remove redundant ones.
    pauli_list has to be a list with each element being a single pauli,
    but NOT a linear combination of paulis.
    """
    pauli_dict = {}
    for pauli in pauli_list:
        ### Get index
        ind, coef = get_pauli_index_and_coef(pauli)
        ### Add or overwrite pauli in pauli_dict
        pauli_dict[str(ind[0])] = pauli

    ### Now we have a non-redundant pauli_dict. 
    ### Reconstruct the non-redundant pauli_list with a coefficient of 1.
    nonredundant_pauli_list = []
    i = 0
    for key, pauli in pauli_dict.items():
        coef = pauli.terms.values()
        pauli /= abs(list(coef)[0])
        nonredundant_pauli_list.append(pauli)
    return nonredundant_pauli_list
    

def get_unique_list(old, sign=False):
    """Function
    Remove redundant elements from old.
    If sign = True, absolute value is considered (if old contains [-1, 1, 1, 2], either -1 or 1 is taken, --> [-1, 2]).
    """
    import collections
    if isinstance(old, collections.Hashable):
        oldset = set(old)
    else:
        oldset = old
    new = []
    if sign:
        return [x for x in oldset if (x not in new) and (-x not in new) and not new.append(x)]
    else:
        return [x for x in oldset if x not in new and not new.append(x)]


def prepare_state(state_info, n_qubits):
    """
    Generate a quantum state using state_info.
    state_info includes determinants as integers and (real) weights. 
    Its format is
       state_info = [[det1, weight1], [det2, weight2], ...]
    and the generated state is
       state = weight1 * |det1> + weight2 * |det2> + ...
    which is then normalized.

    Args:
        state_info (list): Nested list including integers and weights
        n_qubits (int): Number of qubits
    Returns:
        state (QuantumState): Generated state

    Author(s): Takashi Tsuchimochi
    """
    state = QuantumState(n_qubits) 
    state.multiply_coef(0)
    for det, weight in state_info:
        state_ = QuantumState(n_qubits)
        state_.set_computational_basis(det)
        state_.multiply_coef(weight)
        state.add_state(state_)
    # Normalize
    norm = state.get_squared_norm()
    state.normalize(norm)
    return state
        
def transform_state_jw2bk(state):
    """Function
    Convert a QuantumState from jordan-wigner to bravyi-kitaev mapping.
    
    Args:
        state (QuantumState): QuantumState in Jordan-Wigner mapping to be transformed.
    Returns:
        state (QuantumState): QuantumState in Bravyi-Kitaev mapping.
    """
    n_qubits = state.get_qubit_count()
    v = state.get_vector()
    vec = np.zeros(2**n_qubits, dtype=complex)
    for i in range(2**n_qubits):
        vec[jw2bk(i, n_qubits)] = v[i]
    state_ = QuantumState(n_qubits)
    state_.load(vec)
    return state_

def transform_state_bk2jw(state):
    """Function
    Convert a QuantumState from bravyi-kitaev to jordan-wigner mapping.
    
    Args:
        state (QuantumState): QuantumState in Bravyi-Kitaev mapping to be transformed.
    Returns:
        state (QuantumState): QuantumState in Jordan-Wigner mapping.
    """
    n_qubits = state.get_qubit_count()
    v = state.get_vector()
    vec = np.zeros(2**n_qubits, dtype=complex)
    for i in range(2**n_qubits):
        vec[bk2jw(i, n_qubits)] = v[i]
    state_ = QuantumState(n_qubits)
    state_.load(vec)
    return state_


def get_tau(E, mapping='jordan_wigner', hc=True, n_qubits=None):
    """
    From orbital list [p,q], [p,q,r,s], ...,
    generate pauli operators of its anti-hermitiian form
    The first half of the list means creation operators,
    the last half of the list means annihilation operators,
    e.g., E=[p,q,r,s] gives tau=p^ q^ r s - s r q^ p^
    in QubitOperator basis.

    Args:
        E (list): integer list of p,q,...,r,s to represent the excitation p^ q^ ... r s
        hc (bool): Whether or not hermitian_conjugated is taken for E (E-E! or E).
        mapping : jordan_wigner or bravyi_kitaev.
                         For bravyi_kitaev, n_qubits is required.
    Returns:
        sigma (QubitOperator): Pauli string in QubitOperator class
    """
    rank = len(E)
    irank = 0
    excitation_string = ''
    if rank%2 != 0:
        raise ValueError('get_tau needs excitation string, i.e., even creation and annihilation')
    for p in E:
        if irank < rank//2: # creation
            excitation_string = excitation_string + str(p)+'^ '
        else: # annihilation
            excitation_string = excitation_string + str(p)+' '
        irank += 1
    #prints(excitation_string)
    fermi_op = FermionOperator(excitation_string)
    if hc:
        tau = fermi_op - hermitian_conjugated(fermi_op)
    else:
        tau = fermi_op
    if mapping == 'jordan_wigner':
        sigma = jordan_wigner(tau)
    elif mapping == 'bravyi_kitaev':
        if n_qubits is None:
            raise ValueError('n_qubits is necessary for bravyi_kitaev')
        sigma = bravyi_kitaev(tau, n_qubits)
    elif mapping is None:
        sigma = tau
    return sigma

