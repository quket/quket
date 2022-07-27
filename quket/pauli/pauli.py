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
from itertools import product, combinations
import numpy as np
import itertools
import openfermion
from copy import deepcopy
from qulacs import QuantumCircuit, Observable
from qulacs.state import inner_product
from qulacs.gate import Identity, X, Y, Z, RX, RY, RZ, H, CNOT, CZ, SWAP, merge, DenseMatrix, to_matrix_gate, PauliRotation

from quket.utils import int2occ, order_pqrs, get_unique_list, get_unique_pauli_list, Gdoubles_list, is_commute
from quket.fileio import prints, error
from quket.mpilib import mpilib as mpi
from quket.lib import (
    normal_ordered,
    jordan_wigner,
    bravyi_kitaev,
    hermitian_conjugated,
    get_fermion_operator,
    commutator,
    s_squared_operator,
    QubitOperator,
    FermionOperator
)
from quket.opelib import get_excite_dict, get_excite_dict_sf
from quket.tapering import tapering_off_operator, transform_pauli_list, transform_pauli
from quket.opelib import excitation
import quket.config as cf


def pure_imag(pauli):
    pauli_ = QubitOperator('',0)
    for op, coef in pauli.terms.items():
        pauli_ += QubitOperator(op, coef.imag * 1j)
    return pauli_

def get_pauli_list_hamiltonian(Quket, threshold=0):
    """
    Pauli list from local Hamiltonians 
    """
    pauli_list = []
    if isinstance(Quket.operators.qubit_Hamiltonian, (openfermion.QubitOperator, QubitOperator)):
        for key in Quket.operators.qubit_Hamiltonian.terms.keys():
            pauli_list.append(QubitOperator(key))
    return pauli_list
        

    
def get_pauli_list_fermionic_hamiltonian(Quket, anti=False, threshold=0):
    """
    Looking at local Hamiltonians p^ q^ s r + r^ s^ q p,
    make pauli string by jordan-wigner.
    If anti = True, anti-symmetrize p^ q^ s r - r^ s^ q p, and return as pauli_list.
    """
    pauli_list = []
    if not isinstance(Quket.operators.Hamiltonian, (openfermion.FermionOperator, FermionOperator)):
        H = get_fermion_operator(Quket.operators.Hamiltonian)
    else:
        H = Quket.operators.Hamiltonian
    H = normal_ordered(H)
    for key, val in H.terms.items():
        if abs(val) > threshold:
            if anti:
                # Anti-Hermitian
                if Quket.cf.mapping == "jordan_wigner":
                    op = jordan_wigner(FermionOperator(key)) - jordan_wigner(hermitian_conjugated(FermionOperator(key)))
                elif Quket.cf.mapping == "bravyi_kitaev":
                    op = bravyi_kitaev(FermionOperator(key)) - jordan_wigner(hermitian_conjugated(FermionOperator(key)), Quket.n_qubits)
            else:
                # Hermitian
                if Quket.cf.mapping == "jordan_wigner":
                    op = jordan_wigner(FermionOperator(key)) + jordan_wigner(hermitian_conjugated(FermionOperator(key)))
                elif Quket.cf.mapping == "bravyi_kitaev":
                    op = bravyi_kitaev(FermionOperator(key)) + jordan_wigner(hermitian_conjugated(FermionOperator(key)), Quket.n_qubits)
            if len(op.terms) == 0:
                pass
            else:
                if Quket.cf.disassemble_pauli:
                    for pauli, coef in op.terms.items():
                        pauli_list.append(QubitOperator(pauli, coef/abs(coef) * np.sign(coef)))
                else:
                    pauli_list.append(op)
    # Remove the redundant operators and get the unique list
    if Quket.cf.disassemble_pauli:
        pauli_list = get_unique_pauli_list(pauli_list)
    else:
        pauli_list = get_unique_list(pauli_list, sign=True)
    return pauli_list

def get_pauli_list_uccsd(Quket, singles=True):

    """
    Create pauli list for uccsd
    """
    if type(Quket.det) is not int:
        error(f'Quket.det with type {type(Quket.det)} is not compatible with uccsd.')
    excite_dict = get_excite_dict(Quket)
    Quket.excite_dict = excite_dict

    pauli_list = []
    ncnot_list = []
    if singles:
        if Quket.DS:
            order = ['singles','doubles']
        else:
            order = ['doubles','singles']
    else:
        order = ['doubles']
    for rank in order:
        if rank == 'singles':
            for i, a in excitation.singles(excite_dict):
                fermi = FermionOperator(((a,1),(i,0)),1)
                if Quket.cf.mapping == "jordan_wigner":
                    pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                elif Quket.cf.mapping == "bravyi_kitaev":
                    pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), Quket.n_qubits)
                if Quket.cf.disassemble_pauli:
                    for op, coef in pauli.terms.items():
                        pauli_list.append(QubitOperator(op, coef))
                        ncnot = 2*(a-i)
                        ncnot_list.append(ncnot)
                else:
                    pauli_list.append(pauli)
                    ncnot = 2*(a-i) + 1
                    ncnot_list.append(ncnot)
        if rank == 'doubles':
            for i, j, a, b in excitation.doubles(excite_dict):
                fermi = FermionOperator(((b,1),(a,1),(j,0),(i,0)),1)
                if Quket.cf.mapping == "jordan_wigner":
                    pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                elif Quket.cf.mapping == "bravyi_kitaev":
                    pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), Quket.n_qubits)

                if Quket.cf.disassemble_pauli:
                    for op, coef in pauli.terms.items():
                        pauli_list.append(QubitOperator(op, coef))
                        ncnot = 2*(max(a,b)-min(a,b)+max(i,j)-min(i,j)) + 2
                        ncnot_list.append(ncnot)
                else:
                    pauli_list.append(pauli)
                    ncnot = 2*(max(a,b)-min(a,b)+max(i,j)-min(i,j)) + 9
                    ncnot_list.append(ncnot)
    if Quket.cf.disassemble_pauli:
        pauli_list = get_unique_pauli_list(pauli_list)
    return pauli_list, ncnot_list


def get_pauli_list_uccsd_sf(Quket, no, nv, DS=0, x=0, singles=True, doubles=True):
    """Function:
    Create Pauli list for spin-adapted UCCSD (SA-UCCSD).
    We use spin-free excitation for each b,a,j,i combination,
    E^ba_ji =   b(A)^ a(A)^ j(A) i(A)
              + b(B)^ a(A)^ j(B) i(A)
              + b(A)^ a(B)^ j(A) i(B)
              + b(B)^ a(B)^ j(B) i(B)
    with the same theta: theta_list[baji].
    baji is a label number for the 'baji' spin-free excitation and is given by
    get_baji(b,a,j,i,noa).
    In quantum-computing, the exponential of these excitations cannot be done at once,
    but they have to be decomposed and performed separately.

    Essentially, there are two ways to do this.
    The first approach is slightly inefficient but simpler.
    We just prepare pauli_list for spin-free E^ba_ji and the corresponding theta_list.
    This is less efficient than the second approach, because we need twice gates for
    b(A)^ a(A)^ j(A) i(A) and b(B)^ a(B)^ j(B) i(B), generated by E^ba_ji and E^ab_ji.

    The second way is to decompose E^ba_ji to the same-spin excitation and the opposite-spin one.
    We gather and parameterize the latter,
    G^ba_ji =  b(B)^ a(A)^ j(B) i(A)
              +

    For  b(A)^ a(A)^ j(A) i(A),  we perform the exponential operation in
    the 'aa -> aa' loop, but with the amplitude,  theta_list[baji] - theta_list[abji].
    This is because the excitation b(A)^ a(A)^ j(A) i(A) appears also in the 'abji'
    spin-free excitation,
    E^ab_ji =   a(A)^ b(A)^ j(A) i(A)
              + a(B)^ b(A)^ j(A) i(B)
              + a(A)^ b(B)^ j(B) i(A)
              + a(B)^ b(B)^ j(B) i(B)
            = - b(A)^ a(A)^ j(A) i(A)
              + a(B)^ b(A)^ j(A) i(B)
              + a(A)^ b(B)^ j(B) i(A)
              - b(B)^ a(B)^ j(B) i(B)
    which has the amplitude  theta_list[abji].  In total, the amplitude for
    b(A)^ a(A)^ j(A) i(A)   is combined as   theta_list[baji] - theta_list[abji],
    as is exactly performed in the 'aa -> aa' loop. We use exactly the same amplitude
    for the 'bb -> bb' loop.
    Note that E^ba_ij = E^ab_ji  and E^ab_ij = E^ba_ji,  so we only need to consider
    bj >= ai.

    For  b(B)^ a(A)^ j(B) i(A),  we perform the exponential operation in
    the 'ab -> ab' loop, with  theta = theta_list[baji]
    However, since double_ope_Pauli(p,q,r,s,...) assumes  p>q, r>s, we may need to
    flip the sign of theta (by the anti-symmetry of fermion operators). That is,
    if b(B) > a(A), j(B) > i(A),  sign is +theta
    if a(A) > b(B), j(B) > i(A),  sign is -theta
    if b(B) > a(A), i(A) > j(B),  sign is -theta
    if a(A) > b(B), i(A) > j(B),  sign is +theta

    Note that the operation  b(A)^ a(B)^ j(A) i(B) is treated in this loop
    but with a <-> b and i <-> j, because
    b(A)^ a(B)^ j(A) i(B)  ==  a(B)^ b(A) i(B) j(A)
    Since  get_baji(b, a, j, i, noa) = get_baji(a, b, i, j, noa),
    we use the same theta.

    Author(s): Takashi Tsuchimochi
    """
    if type(Quket.det) is not int:
        error(f'Quket.det with type {type(Quket.det)} is not compatible with uccsd.')
    pauli_list = []
    ncnot_list = []
    if singles:
        if doubles:
            if DS:
                order = ['singles','doubles']
            else:
                order = ['doubles','singles']
        else:
            order = ['singles']
    else:
        order = ['doubles']
    for rank in order:
        if rank == 'singles':
            ####################
            ##    Singles     ##
            ####################
            for a in range(nv):
                aA = 2*(a+no)   # a(A)
                aB = aA + 1     # a(B)
                for i in range(no):
                    iA = 2*i      # i(A)
                    iB = iA + 1   # i(B)
                    fermi_1 = FermionOperator(((aA,1),(iA,0)),1)
                    fermi_2 = FermionOperator(((aB,1),(iB,0)),1)
                    if Quket.cf.mapping == "jordan_wigner":
                        pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                        pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                    elif Quket.cf.mapping == "bravyi_kitaev":
                        pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), Quket.n_qubits)
                        pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), Quket.n_qubits)
                    pauli_list.append([pauli_1, pauli_2])
                    ncnot_list.append(2*(aA-iA+aB-iB)+2)
        if rank == 'doubles':
            ####################
            ##    Doubles     ##
            ####################
            key = 0
            if key == 0:
                for a in range(nv):
                    aA = 2*(a+no) # a(A)
                    aB = aA + 1   # a(B)
                    for b in range(a+1):
                        bA = 2*(b+no)   # b(A)
                        bB = bA + 1     # b(B)
                        for i in range(no):
                            iA = 2*i      # i(A)
                            iB = iA + 1   # i(B)
                            for j in range(i+1):
                                jA = 2*j     # j(A)
                                jB = jA + 1  # j(B)
                                fermi_1 = FermionOperator(((aA,1),(bA,1),(jA,0),(iA,0)),1)
                                fermi_2 = FermionOperator(((aA,1),(bB,1),(jB,0),(iA,0)),1)
                                fermi_3 = FermionOperator(((aB,1),(bA,1),(jA,0),(iB,0)),1)
                                fermi_4 = FermionOperator(((aB,1),(bB,1),(jB,0),(iB,0)),1)
                                if Quket.cf.mapping == "jordan_wigner":
                                    pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                                    pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                                    pauli_3 = jordan_wigner(fermi_3 - hermitian_conjugated(fermi_3))
                                    pauli_4 = jordan_wigner(fermi_4 - hermitian_conjugated(fermi_4))
                                elif Quket.cf.mapping == "bravyi_kitaev":
                                    pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), Quket.n_qubits)
                                    pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), Quket.n_qubits)
                                    pauli_3 = bravyi_kitaev(fermi_3 - hermitian_conjugated(fermi_3), Quket.n_qubits)
                                    pauli_4 = bravyi_kitaev(fermi_4 - hermitian_conjugated(fermi_4), Quket.n_qubits)
                                pauli_list.append([pauli_1, pauli_2, pauli_3, pauli_4])
                                if aA != bA and iA != jA:
                                    ncnot = 2*(max(aA,bA)-min(aA,bA)+max(iA,jA)-min(iA,jA))+9\
                                          + 2*(max(aB,bA)-min(aB,bA)+max(iB,jA)-min(iB,jA))+9\
                                          + 2*(max(aA,bB)-min(aA,bB)+max(iA,jB)-min(iA,jB))+9\
                                          + 2*(max(aB,bB)-min(aB,bB)+max(iB,jB)-min(iB,jB))+9
                                elif aA != bA or iA != jA:
                                    ncnot = 2*(max(aB,bA)-min(aB,bA)+max(iB,jA)-min(iB,jA))+9\
                                          + 2*(max(aA,bB)-min(aA,bB)+max(iA,jB)-min(iA,jB))+9
                                else:
                                    ncnot = 2*(max(aA,bA)-min(aA,bA)+max(iA,jA)-min(iA,jA))+9\
                                          + 2*(max(aB,bB)-min(aB,bB)+max(iB,jB)-min(iB,jB))+9
                                ncnot_list.append(ncnot)

                                if a!=b and i!=j:
                                    fermi_1 = FermionOperator(((bA,1),(aA,1),(jA,0),(iA,0)),1)
                                    fermi_2 = FermionOperator(((bA,1),(aB,1),(jB,0),(iA,0)),1)
                                    fermi_3 = FermionOperator(((bB,1),(aA,1),(jA,0),(iB,0)),1)
                                    fermi_4 = FermionOperator(((bB,1),(aB,1),(jB,0),(iB,0)),1)
                                    if Quket.cf.mapping == "jordan_wigner":
                                        pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                                        pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                                        pauli_3 = jordan_wigner(fermi_3 - hermitian_conjugated(fermi_3))
                                        pauli_4 = jordan_wigner(fermi_4 - hermitian_conjugated(fermi_4))
                                    elif Quket.cf.mapping == "bravyi_kitaev":
                                        pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), Quket.n_qubits)
                                        pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), Quket.n_qubits)
                                        pauli_3 = bravyi_kitaev(fermi_3 - hermitian_conjugated(fermi_3), Quket.n_qubits)
                                        pauli_4 = bravyi_kitaev(fermi_4 - hermitian_conjugated(fermi_4), Quket.n_qubits)
                                    pauli_list.append([pauli_1, pauli_2, pauli_3, pauli_4])
                                    ncnot = 2*(max(aA,bA)-min(aA,bA)+max(iA,jA)-min(iA,jA))+9\
                                          + 2*(max(aB,bA)-min(aB,bA)+max(iB,jA)-min(iB,jA))+9\
                                          + 2*(max(aA,bB)-min(aA,bB)+max(iA,jB)-min(iA,jB))+9\
                                          + 2*(max(aB,bB)-min(aB,bB)+max(iB,jB)-min(iB,jB))+9
                                    ncnot_list.append(ncnot)
    return pauli_list, ncnot_list

def get_pauli_list_uccsd_sf(Quket, no, nv, DS=0, x=0, singles=True, doubles=True):
    """
    Create pauli list for spin-free uccsd
    """
    if type(Quket.det) is not int:
        error(f'Quket.det with type {type(Quket.det)} is not compatible with uccsd.')
    excite_dict = get_excite_dict_sf(Quket)
    Quket.excite_dict = excite_dict

    pauli_list = []
    ncnot_list = []
    if singles:
        if Quket.DS:
            order = ['singles','doubles']
        else:
            order = ['doubles','singles']
    else:
        order = ['doubles']
    for rank in order:
        if rank == 'singles':
            for i, a in excitation.singles_sf(excite_dict):
#                prints(f'{a} <- {i}')
                fermi_1 = FermionOperator(((2*a,1),(2*i,0)),1)
                fermi_2 = FermionOperator(((2*a+1,1),(2*i+1,0)),1)
                if Quket.cf.mapping == "jordan_wigner":
                    pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                    pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                elif Quket.cf.mapping == "bravyi_kitaev":
                    pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), Quket.n_qubits)
                    pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), Quket.n_qubits)
                if Quket.cf.disassemble_pauli:
                    pauli = pauli_1 + pauli_2
                    for op, coef in pauli.terms.items():
                        pauli_list.append(QubitOperator(op, coef))
                        ncnot = 4*(2*a-2*i)
                        ncnot_list.append(ncnot)
                else:
                    pauli_list.append([pauli_1, pauli_2])
                    ncnot = 4*(2*a-2*i) + 2
                    ncnot_list.append(ncnot)
        if rank == 'doubles':
            #fermi_list = []
            for i, j, a, b in excitation.doubles_sf(excite_dict):
#                prints(f'{b} {a} <- {j} {i}')
                fermi_1 = FermionOperator(((2*b,1),(2*a,1),(2*j,0),(2*i,0)),1)
                fermi_1 += FermionOperator(((2*b+1,1),(2*a+1,1),(2*j+1,0),(2*i+1,0)),1)
                fermi_2 = FermionOperator(((2*b,1),(2*a+1,1),(2*j+1,0),(2*i,0)),1)
                fermi_2 += FermionOperator(((2*b+1,1),(2*a,1),(2*j,0),(2*i+1,0)),1)
                #prints(normal_ordered(commutator(fermi,S2))==FermionOperator('',0))
                #fermi_list.append(fermi)
                if Quket.cf.mapping == "jordan_wigner":
                    pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                    pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                elif Quket.cf.mapping == "bravyi_kitaev":
                    pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), Quket.n_qubits)
                    pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), Quket.n_qubits)
                if Quket.cf.disassemble_pauli:
                    pauli = pauli_1 + pauli_2
                    for op, coef in pauli.terms.items():
                        pauli_list.append(QubitOperator(op, coef))
                        ncnot = 2*(max(2*a,2*b)-min(2*a,2*b)+max(2*i,2*j)-min(2*i,2*j)) + 2 \
                              + 2*(max(2*a+1,2*b)-min(2*a+1,2*b)+max(2*i,2*j+1)-min(2*i,2*j+1)) + 2 \
                              + 2*(max(2*a,2*b+1)-min(2*a,2*b+1)+max(2*i+1,2*j)-min(2*i+1,2*j)) + 2 \
                              + 2*(max(2*a+1,2*b+1)-min(2*a+1,2*b+1)+max(2*i+1,2*j+1)-min(2*i+1,2*j+1)) + 2
                        ncnot_list.append(ncnot)
                else:
                    if is_commute(pauli_1, pauli_2):
                        pauli_list.append(pauli_1+pauli_2)
                    else:
                        pauli_list.append([pauli_1, pauli_2])
                    ncnot = 2*(max(2*a,2*b)-min(2*a,2*b)+max(2*i,2*j)-min(2*i,2*j)) + 9 \
                          + 2*(max(2*a+1,2*b)-min(2*a+1,2*b)+max(2*i,2*j+1)-min(2*i,2*j+1)) + 9 \
                          + 2*(max(2*a,2*b+1)-min(2*a,2*b+1)+max(2*i+1,2*j)-min(2*i+1,2*j)) + 9 \
                          + 2*(max(2*a+1,2*b+1)-min(2*a+1,2*b+1)+max(2*i+1,2*j+1)-min(2*i+1,2*j+1)) + 9
                    ncnot_list.append(ncnot)
            #prints(f'{ndim2=}')
            #for i in range(len( pauli_list)):
            #    pauli_i = hermitian_conjugated(pauli_list[i])
            #    for j in range(i):
            #        pauli_j = pauli_list[j]
            #        #prints(normal_ordered(pauli_i * pauli_j))
            #        prints(pauli_i*pauli_j)

    if Quket.cf.disassemble_pauli:
        pauli_list = get_unique_pauli_list(pauli_list)
    return pauli_list, ncnot_list


def create_pauli_list_gs(norbs, pauli_list, ncnot_list=None, disassemble_pauli=False, mapping='jordan_wigner'):
    ####################
    ##    Singles     ##
    ####################
    for p in range(norbs):
        pA = 2*p         # p(A)
        pB = pA + 1      # p(B)
        for q in range(p):
            qA = 2*q      # q(A)
            qB = qA + 1   # q(B)
            fermi = FermionOperator(((pA,1),(qA,0)),1)
            pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
            if mapping in ("jw", "jordan_wigner"):
                pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
            elif mapping in ("bk", "bravyi_kitaev"):
                pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
            if disassemble_pauli:
                for op, coef in pauli.terms.items():
                    pauli_list.append(QubitOperator(op, coef))
                    if ncnot_list is not None:
                        ncnot = 2*(pA-qA)
                        ncnot_list.append(ncnot)
            else:
                pauli_list.append(pauli)
                if ncnot_list is not None:
                    ncnot = 2*(pA-qA) + 1
                    ncnot_list.append(ncnot)

            fermi = FermionOperator(((pB,1),(qB,0)),1)
            if mapping == "jordan_wigner":
                pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
            elif mapping == "bravyi_kitaev":
                pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
            if disassemble_pauli:
                for op, coef in pauli.terms.items():
                    pauli_list.append(QubitOperator(op, coef))
                    if ncnot_list is not None:
                        ncnot = 2*(pB-qB)
                        ncnot_list.append(ncnot)
            else:
                pauli_list.append(pauli)
                if ncnot_list is not None:
                    ncnot = 2*(pB-qB) + 1
                    ncnot_list.append(ncnot)
    if disassemble_pauli:
        pauli_list = get_unique_pauli_list(pauli_list)

    return pauli_list, ncnot_list

def create_pauli_list_gd(norbs, pauli_list, ncnot_list=None, disassemble_pauli=False, mapping='jordan_wigner'):
    ####################
    ##    Doubles     ##
    ####################
    r_list, u_list, parity_list = Gdoubles_list(norbs)
    for ilist in range(len(u_list)):
        for b, a, j, i in u_list[ilist]:
            fermi = FermionOperator(((b,1),(a,1),(j,0),(i,0)),1)
            if mapping in ("jw", "jordan_wigner"):
                pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
            elif mapping in ("bk", "bravyi_kitaev"):
                pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
            i4, i3, i2, i1 = order_pqrs(b,a,j,i)
            if disassemble_pauli:
                for op, coef in pauli.terms.items():
                    pauli_list.append(QubitOperator(op, coef))
                    if ncnot_list is not None:
                        ncnot = 2*(i4-i3+i2-i1) + 2
                        ncnot_list.append(ncnot)
            else:
                pauli_list.append(pauli)
                if ncnot_list is not None:
                    ncnot = 2*(i4-i3+i2-i1) + 9
                    ncnot_list.append(ncnot)
    if disassemble_pauli:
        pauli_list = get_unique_pauli_list(pauli_list)
    return pauli_list, ncnot_list

def create_pauli_list_gt(norbs, mapping='jordan_wigner'):
    pauli_list = []
    ### AAA
    for a, b, c in itertools.combinations(range(0,2*norbs,2), 3):
        for i, j, k in itertools.combinations(range(0,2*norbs,2), 3):
            fermi = FermionOperator(((a,1),(b,1),(c,1),(k,0),(j,0),(i,0)),1)
            if mapping in ("jw", "jordan_wigner"):
                pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
            elif mapping in ("bk", "bravyi_kitaev"):
                pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
            pauli_list.append(pauli)
    ### BAA
    for a in range(1, 2*norbs, 2):
        for b, c in itertools.combinations(range(0,2*norbs,2), 2):
            for i in range(1, 2*norbs, 2):
                for j, k in itertools.combinations(range(0,2*norbs,2), 2):
                    fermi = FermionOperator(((a,1),(b,1),(c,1),(k,0),(j,0),(i,0)),1)
                    if mapping in ("jw", "jordan_wigner"):
                        pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                    elif mapping in ("bk", "bravyi_kitaev"):
                        pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
                    pauli_list.append(pauli)
    ### BBA
    for a, b in itertools.combinations(range(1,2*norbs,2), 2):
        for c in range(0, 2*norbs, 2):
            for i, j in itertools.combinations(range(1,2*norbs,2), 2):
                for k in range(0, 2*norbs, 2):
                    fermi = FermionOperator(((a,1),(b,1),(c,1),(k,0),(j,0),(i,0)),1)
                    if mapping in ("jw", "jordan_wigner"):
                        pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                    elif mapping in ("bk", "bravyi_kitaev"):
                        pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
                    pauli_list.append(pauli)
    ### BBB
    for a, b, c in itertools.combinations(range(1,2*norbs,2), 3):
        for i, j, k in itertools.combinations(range(1,2*norbs,2), 3):
            fermi = FermionOperator(((a,1),(b,1),(c,1),(k,0),(j,0),(i,0)),1)
            if mapping in ("jw", "jordan_wigner"):
                pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
            elif mapping in ("bk", "bravyi_kitaev"):
                pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
            pauli_list.append(pauli)
    pauli_list = get_unique_list(pauli_list, sign=True)
    return pauli_list

def create_pauli_list_gq(norbs,  mapping='jordan_wigner'):
    pauli_list = []
    ### AAAA
    for a, b, c, d in itertools.combinations(range(0,2*norbs,2), 4):
        for i, j, k, l in itertools.combinations(range(0,2*norbs,2), 4):
            fermi = FermionOperator(((a,1),(b,1),(c,1),(d,1),(l,0),(k,0),(j,0),(i,0)),1)
            if mapping in ("jw", "jordan_wigner"):
                pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
            elif mapping in ("bk", "bravyi_kitaev"):
                pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
            pauli_list.append(pauli)
    ### BAAA
    for a in range(1, 2*norbs, 2):
        for b, c, d in itertools.combinations(range(0,2*norbs,2), 3):
            for i in range(1, 2*norbs, 2):
                for j, k, l in itertools.combinations(range(0,2*norbs,2), 3):
                    fermi = FermionOperator(((a,1),(b,1),(c,1),(d,1),(l,0),(k,0),(j,0),(i,0)),1)
                    if mapping in ("jw", "jordan_wigner"):
                        pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                    elif mapping in ("bk", "bravyi_kitaev"):
                        pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
                    pauli_list.append(pauli)
    ### BBAA
    for a, b in itertools.combinations(range(1,2*norbs,2), 2):
        for c, d in itertools.combinations(range(0,2*norbs,2), 2):
            for i, j in itertools.combinations(range(1,2*norbs,2), 2):
                for k, l in itertools.combinations(range(0,2*norbs,2), 2):
                    fermi = FermionOperator(((a,1),(b,1),(c,1),(d,1),(l,0),(k,0),(j,0),(i,0)),1)
                    if mapping in ("jw", "jordan_wigner"):
                        pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                    elif mapping in ("bk", "bravyi_kitaev"):
                        pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
                    pauli_list.append(pauli)
    ### BBBA
    for a, b, c in itertools.combinations(range(1,2*norbs,2), 3):
        for d in range(0, 2*norbs, 2):
            for i, j, k in itertools.combinations(range(1,2*norbs,2), 3):
                for l in range(0, 2*norbs, 2):
                    fermi = FermionOperator(((a,1),(b,1),(c,1),(d,1),(l,0),(k,0),(j,0),(i,0)),1)
                    if mapping in ("jw", "jordan_wigner"):
                        pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                    elif mapping in ("bk", "bravyi_kitaev"):
                        pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
                    pauli_list.append(pauli)
    ### BBBB
    for a, b, c, d in itertools.combinations(range(1,2*norbs,2), 4):
        for i, j, k, l in itertools.combinations(range(1,2*norbs,2), 4):
            fermi = FermionOperator(((a,1),(b,1),(c,1),(d,1),(l,0),(k,0),(j,0),(i,0)),1)
            if mapping in ("jw", "jordan_wigner"):
                pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
            elif mapping in ("bk", "bravyi_kitaev"):
                pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
            pauli_list.append(pauli)
    pauli_list = get_unique_list(pauli_list, sign=True)
    return pauli_list

def get_pauli_list_uccgsd(norbs, DS=0, singles=True, disassemble_pauli=False, mapping='jordan_wigner'):
    """Function:
    Create Pauli list for UCCGSD.

    Author(s): Takashi Tsuchimochi
    """
    pauli_list = []
    ncnot_list = []
    if singles:
        if DS:
            order = ['singles','doubles']
        else:
            order = ['doubles','singles']
    else:
        order = ['doubles']
    for rank in order:
        if rank == 'singles':
            pauli_list, ncnot_list = create_pauli_list_gs(norbs, pauli_list, ncnot_list=ncnot_list, disassemble_pauli=disassemble_pauli, mapping=mapping)
        if rank == 'doubles':
            pauli_list, ncnot_list = create_pauli_list_gd(norbs, pauli_list, ncnot_list=ncnot_list, disassemble_pauli=disassemble_pauli, mapping=mapping)
    return pauli_list, ncnot_list

def get_pauli_list_uccgsdt(norbs, mapping='jordan_wigner'):
    """Function:
    Create Pauli list for UCCGSDT.

    Author(s): Takashi Tsuchimochi
    """
    pauli_list = []
    ncnot_list = []
    order = ['triples', 'doubles', 'singles']
    for rank in order:
        if rank == 'singles':
            pauli_list, ncnot_list = create_pauli_list_gs(norbs, pauli_list, mapping=mapping)
        if rank == 'doubles':
            pauli_list, ncnot_list = create_pauli_list_gd(norbs, pauli_list, mapping=mapping)
        if rank == 'triples':
            pauli_list += create_pauli_list_gt(norbs, mapping=mapping)
    pauli_list = get_unique_list(pauli_list, sign=True)
    return pauli_list

def get_pauli_list_uccgsdtq(norbs, mapping='jordan_wigner'):
    """Function:
    Create Pauli list for UCCGSDT.

    Author(s): Takashi Tsuchimochi
    """
    pauli_list = []
    ncnot_list = []
    order = ['quadruples', 'triples', 'doubles', 'singles']
    for rank in order:
        if rank == 'singles':
            pauli_list, ncnot_list = create_pauli_list_gs(norbs, pauli_list, mapping=mapping)
        if rank == 'doubles':
            pauli_list, ncnot_list = create_pauli_list_gd(norbs, pauli_list, mapping=mapping)
        if rank == 'triples':
            pauli_list += create_pauli_list_gt(norbs, mapping=mapping)
        if rank == 'quadruples':
            pauli_list += create_pauli_list_gq(norbs, mapping=mapping)
    pauli_list = get_unique_list(pauli_list, sign=True)
    return pauli_list

def get_pauli_list_uccgsd_sf(norbs, DS=0, singles=True, mapping='jordan_wigner'):
    """Function:
    Create Pauli list for Spin-Free UCCGSD.

    Author(s): Takashi Tsuchimochi
    """
    ###

    pauli_list = []
    ncnot_list = []
    if singles:
        if DS:
            order = ['singles','doubles']
        else:
            order = ['doubles','singles']
    else:
        order = ['doubles']
    for rank in order:
        if rank == 'singles':
            ndim1 = 0
            ####################
            ##    Singles     ##
            ####################
            for p in range(norbs):
                pA = 2*p         # p(A)
                pB = pA + 1      # p(B)
                for q in range(p):
                    qA = 2*q      # q(A)
                    qB = qA + 1   # q(B)
                    fermi = FermionOperator(((pA,1),(qA,0)),1)\
                          + FermionOperator(((pB,1),(qB,0)),1)
                    if mapping in ("jw", "jordan_wigner"):
                        pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))\
                              + jordan_wigner(fermi - hermitian_conjugated(fermi))
                    elif mapping in ("bk", "bravyi_kitaev"):
                        pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)\
                              + bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
                    pauli_list.append(pauli)
                    ncnot = 2*(pA - qA + pB - qB) + 2
                    ncnot_list.append(ncnot)
                    ndim1 += 1

        if rank == 'doubles':
            S2 = s_squared_operator(norbs)
            ####################
            ##    Doubles     ##
            ####################
            r_list, u_list, parity_list = Gdoubles_list(norbs)
            for k, u in enumerate(u_list):
                if len(u) == 6:
                    ### two possibilities
                    #     aaaa   bbbb  baba  abab
                    #u1 = [u[0], u[1], u[2], u[3]]
                    #     aaaa   bbbb  baab  abba
                    #u2 = [u[0], u[1], u[4], u[5]]
                    fermi_1 = FermionOperator(((u[0][0],1),(u[0][1],1),(u[0][2],0),(u[0][3],0)),1)
                    fermi_1+= FermionOperator(((u[1][0],1),(u[1][1],1),(u[1][2],0),(u[1][3],0)),1)
                    fermi_2 = FermionOperator(((u[2][0],1),(u[2][1],1),(u[2][2],0),(u[2][3],0)),1)
                    fermi_2+= FermionOperator(((u[3][0],1),(u[3][1],1),(u[3][2],0),(u[3][3],0)),1)
                    if mapping in ("jw", "jordan_wigner"):
                        pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                        pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                    elif mapping in ("bk", "bravyi_kitaev"):
                        pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), norbs*2)
                        pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), norbs*2)
                    if is_commute(pauli_1, pauli_2):
                        pauli_list.append(pauli_1+pauli_2)
                    else:
                        pauli_list.append([pauli_1, pauli_2])
                    i4, i3, i2, i1 = order_pqrs(u[0][0],u[0][1],u[0][2],u[0][3])
                    ncnot = 2*(i4-i3+i2-i1) + 9
                    i4, i3, i2, i1 = order_pqrs(u[1][0],u[1][1],u[1][2],u[1][3])
                    ncnot += 2*(i4-i3+i2-i1) + 9
                    i4, i3, i2, i1 = order_pqrs(u[2][0],u[2][1],u[2][2],u[2][3])
                    ncnot += 2*(i4-i3+i2-i1) + 9
                    i4, i3, i2, i1 = order_pqrs(u[3][0],u[3][1],u[3][2],u[3][3])
                    ncnot += 2*(i4-i3+i2-i1) + 9
                    ncnot_list.append(ncnot)

                    fermi_1 = FermionOperator(((u[0][0],1),(u[0][1],1),(u[0][2],0),(u[0][3],0)),1)
                    fermi_1+= FermionOperator(((u[1][0],1),(u[1][1],1),(u[1][2],0),(u[1][3],0)),1)
                    fermi_2 = FermionOperator(((u[4][0],1),(u[4][1],1),(u[4][2],0),(u[4][3],0)),1)
                    fermi_2+= FermionOperator(((u[5][0],1),(u[5][1],1),(u[5][2],0),(u[5][3],0)),1)
                    if mapping in ("jw", "jordan_wigner"):
                        pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                        pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                    elif mapping in ("bk", "bravyi_kitaev"):
                        pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), norbs*2)
                        pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), norbs*2)
                    if is_commute(pauli_1, pauli_2):
                        pauli_list.append(pauli_1+pauli_2)
                    else:
                        pauli_list.append([pauli_1, pauli_2])
                    i4, i3, i2, i1 = order_pqrs(u[0][0],u[0][1],u[0][2],u[0][3])
                    ncnot = 2*(i4-i3+i2-i1) + 9
                    i4, i3, i2, i1 = order_pqrs(u[1][0],u[1][1],u[1][2],u[1][3])
                    ncnot += 2*(i4-i3+i2-i1) + 9
                    i4, i3, i2, i1 = order_pqrs(u[4][0],u[4][1],u[4][2],u[4][3])
                    ncnot += 2*(i4-i3+i2-i1) + 9
                    i4, i3, i2, i1 = order_pqrs(u[5][0],u[5][1],u[5][2],u[5][3])
                    ncnot += 2*(i4-i3+i2-i1) + 9
                    ncnot_list.append(ncnot)

                elif len(u) == 2:
                    fermi_1 = FermionOperator(((u[0][0],1),(u[0][1],1),(u[0][2],0),(u[0][3],0)),1)
                    fermi_2 = FermionOperator(((u[1][0],1),(u[1][1],1),(u[1][2],0),(u[1][3],0)),1)
                    if mapping in ("jw", "jordan_wigner"):
                        pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                        pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                    elif mapping in ("bk", "bravyi_kitaev"):
                        pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), norbs*2)
                        pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), norbs*2)
                    if is_commute(pauli_1, pauli_2):
                        pauli_list.append(pauli_1+pauli_2)
                    else:
                        pauli_list.append([pauli_1, pauli_2])
                    i4, i3, i2, i1 = order_pqrs(u[0][0],u[0][1],u[0][2],u[0][3])
                    ncnot = 2*(i4-i3+i2-i1) + 9
                    i4, i3, i2, i1 = order_pqrs(u[1][0],u[1][1],u[1][2],u[1][3])
                    ncnot += 2*(i4-i3+i2-i1) + 9
                    ncnot_list.append(ncnot)

                elif len(u) == 1:
                    fermi = FermionOperator(((u[0][0],1),(u[0][1],1),(u[0][2],0),(u[0][3],0)),1)
                    if (normal_ordered(commutator(fermi,S2))==FermionOperator('',0)):
                        if mapping in ("jw", "jordan_wigner"):
                            pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                        elif mapping in ("bk", "bravyi_kitaev"):
                            pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), norbs*2)
                        pauli_list.append(pauli)
                        i4, i3, i2, i1 = order_pqrs(u[0][0],u[0][1],u[0][2],u[0][3])
                        ncnot = 2*(i4-i3+i2-i1) + 9
                        ncnot_list.append(ncnot)
    return pauli_list, ncnot_list

def get_pauli_list_adapt(Quket):
    """Function
    Prepare the list of pauli strings to be used in ADAPT-VQE.
    """
    discard = False
    mapping = Quket.cf.mapping
    zero = QubitOperator('',0)
    if Quket.adapt.mode == "original":
        pauli_list = []
        ### Need to tweak
        ####################
        ##    Singles     ##
        ####################
        for p in range(Quket.n_active_orbitals):
            pA = 2*p         # p(A)
            pB = pA + 1      # p(B)
            for q in range(p):
                qA = 2*q      # q(A)
                qB = qA + 1   # q(B)
                fermi = FermionOperator(((pA,1),(qA,0)),1)
                fermi += FermionOperator(((pB,1),(qB,0)),1)
                if mapping in ("jw", "jordan_wigner"):
                    pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                elif mapping in ("bk", "bravyi_kitaev"):
                    pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), Quket.n_qubits)
                pauli_list.append(pauli)
        ####################
        ##    Doubles     ##
        ####################
        r_list, u_list, parity_list = Gdoubles_list(Quket.n_active_orbitals)
        for k, u in enumerate(u_list):
            if len(u) == 6:
                ### three possibilities
                #u1 = [u[0], u[1]]
                #u2 = [u[2], u[3]]
                #u3 = [u[4], u[5]]
                fermi_1 = FermionOperator(((u[0][0],1),(u[0][1],1),(u[0][2],0),(u[0][3],0)),1)
                fermi_2 = FermionOperator(((u[1][0],1),(u[1][1],1),(u[1][2],0),(u[1][3],0)),1)
                if mapping in ("jw", "jordan_wigner"):
                    pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                    pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                elif mapping in ("bk", "bravyi_kitaev"):
                    pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), Quket.n_qubits)
                    pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), Quket.n_qubits)
                pauli_list.append([pauli_1, pauli_2])

                fermi_1 = FermionOperator(((u[2][0],1),(u[2][1],1),(u[2][2],0),(u[2][3],0)),1)
                fermi_2 = FermionOperator(((u[3][0],1),(u[3][1],1),(u[3][2],0),(u[3][3],0)),1)
                if mapping in ("jw", "jordan_wigner"):
                    pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                    pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                elif mapping in ("bk", "bravyi_kitaev"):
                    pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), Quket.n_qubits)
                    pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), Quket.n_qubits)
                pauli_list.append([pauli_1, pauli_2])

                fermi_1 = FermionOperator(((u[4][0],1),(u[4][1],1),(u[4][2],0),(u[4][3],0)),1)
                fermi_2 = FermionOperator(((u[5][0],1),(u[5][1],1),(u[5][2],0),(u[5][3],0)),1)
                if mapping in ("jw", "jordan_wigner"):
                    pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                    pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                elif mapping in ("bk", "bravyi_kitaev"):
                    pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), Quket.n_qubits)
                    pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), Quket.n_qubits)
                pauli_list.append([pauli_1, pauli_2])

            elif len(u) == 2:
                fermi_1 = FermionOperator(((u[0][0],1),(u[0][1],1),(u[0][2],0),(u[0][3],0)),1)
                fermi_2 = FermionOperator(((u[1][0],1),(u[1][1],1),(u[1][2],0),(u[1][3],0)),1)
                if mapping in ("jw", "jordan_wigner"):
                    pauli_1 = jordan_wigner(fermi_1 - hermitian_conjugated(fermi_1))
                    pauli_2 = jordan_wigner(fermi_2 - hermitian_conjugated(fermi_2))
                elif mapping in ("bk", "bravyi_kitaev"):
                    pauli_1 = bravyi_kitaev(fermi_1 - hermitian_conjugated(fermi_1), Quket.n_qubits)
                    pauli_2 = bravyi_kitaev(fermi_2 - hermitian_conjugated(fermi_2), Quket.n_qubits)
                pauli_list.append([pauli_1, pauli_2])

            elif len(u) == 1:
                fermi = FermionOperator(((u[0][0],1),(u[0][1],1),(u[0][2],0),(u[0][3],0)),2)
                if mapping in ("jw", "jordan_wigner"):
                    pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                elif mapping in ("bk", "bravyi_kitaev"):
                    pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), Quket.n_qubits)
                pauli_list.append(pauli)

    elif Quket.adapt.mode == 'pauli_sz':
        pauli_list = get_pauli_list_uccgsd(Quket.n_active_orbitals, DS=0, singles=True)[0]
        ### Group paulis such that Sz and N are preserved.
        ### Remove Z
        pauli_list_new = []
        for pauli in pauli_list:
            sign = np.zeros(4, int)
            pauli_new = QubitOperator('',0)
            pauli_new1 = QubitOperator('',0)
            for op, coef in pauli.terms.items():
                target_list = []
                pauli_index = []
                string = ""
                for op_ in op:
                    m = op_[0]
                    if op_[1] == 'X':
                        target_list.append(m)
                        pauli_index.append(1)
                        string += f"X{m} "
                    elif op_[1] == 'Y':
                        target_list.append(m)
                        pauli_index.append(2)
                        string += f"Y{m} "

                if len(target_list) == 2:
                    p,q = target_list
                    P0 = QubitOperator(f'X{p} Y{q}', 0.5j)\
                       + QubitOperator(f'Y{p} X{q}', -0.5j)
                elif len(target_list) == 4:
                    p,q,r,s = target_list
                    if pauli_index == [1,1,1,2]:
                        sign[0] = np.sign(coef.imag)
                        P0 = QubitOperator(f'X{p} X{q} X{r} Y{s}',0.125j)\
                           + QubitOperator(f'Y{p} Y{q} Y{r} X{s}',-0.125j)
                    elif pauli_index == [1,1,2,1]:
                        sign[1] = np.sign(coef.imag)
                        P1 = QubitOperator(f'X{p} X{q} Y{r} X{s}',0.125j)\
                           + QubitOperator(f'Y{p} Y{q} X{r} Y{s}',-0.125j)
                    elif pauli_index == [1,2,1,1]:
                        sign[2] = np.sign(coef.imag)
                        P2 = QubitOperator(f'X{p} Y{q} X{r} X{s}',0.125j)\
                           + QubitOperator(f'Y{p} X{q} Y{r} Y{s}',-0.125j)
                    elif pauli_index == [2,1,1,1]:
                        sign[3] = np.sign(coef.imag)
                        P3 = QubitOperator(f'Y{p} X{q} X{r} X{s}',0.125j)\
                           + QubitOperator(f'X{p} Y{q} Y{r} Y{s}',-0.125j)
            if len(target_list) == 2:
                pauli_new = P0
                pauli_new1 = zero

            elif len(target_list) == 4:
                if p%2 == q%2 == r%2 == s%2:
                    if sign[0] != sign[1]:
                        pauli_new  = P0 - P1
                        pauli_new1 = P2 - P3
                    elif sign[0] != sign[2]:
                        pauli_new  = P0 - P2
                        pauli_new1 = P1 - P3
                    elif sign[0] != sign[3]:
                        pauli_new  = P0 - P3
                        pauli_new1 = P1 - P2
                elif p%2 == q%2 and p%2 != r%2:
                    if sign[0] != sign[1]:
                        pauli_new  = P0 - P1
                        pauli_new1 = P2 - P3
                    else:
                        prints('Strange. Report.')
                        error()
                elif p%2 != q%2 and p%2 == r%2:
                    if sign[0] != sign[2]:
                        pauli_new  = P0 - P2
                        pauli_new1 = P1 - P3
                    else:
                        prints('Strange. Report.')
                        error()
                elif p%2 != q%2 and p%2 == s%2:
                    if sign[0] != sign[3]:
                        pauli_new  = P0 - P3
                        pauli_new1 = P1 - P2
                    else:
                        prints('Strange. Report.')
                        error()

            if pauli_new != QubitOperator('',0):
                ### Check if pauli commutes with Sz and Number
                if commutator(Quket.operators.qubit_Sz,pauli_new) != zero:
                    prints(commutator(Quket.operators.qubit_Sz,pauli_new))
                    prints(type(commutator(Quket.operators.qubit_Sz,pauli_new)))
                    prints(commutator(Quket.operators.qubit_Sz,pauli_new)==zero)
                    error("Error!")
                if commutator(Quket.operators.qubit_Number,pauli_new) != zero:
                    prints(f'original pauli = {pauli}')
                    prints('\n new pauli = ',pauli_new, ' \nN non-commutative?', commutator(Quket.operators.qubit_Number,pauli_new))
                pauli_list_new.append(pauli_new)
            if not discard:
                if pauli_new1 != QubitOperator('',0):
                    if commutator(Quket.operators.qubit_Sz,pauli_new1) != zero:
                        prints(commutator(Quket.operators.qubit_Sz,pauli_new1))
                        error("Error!")
                    pauli_list_new.append(pauli_new1)
        pauli_list = pauli_list_new

    elif Quket.adapt.mode == 'pauli':
        pauli_list = get_pauli_list_uccgsd(Quket.n_active_orbitals, DS=0, singles=True)[0]

        ### Remove Z
        pauli_list_new = []
        for pauli in pauli_list:
            pauli_new = QubitOperator('',0)
            for op, coef in pauli.terms.items():
                string = ""
                for op_ in op:
                    m = op_[0]
                    if op_[1] == 'X':
                        string += f"X{m} "
                    elif op_[1] == 'Y':
                        string += f"Y{m} "
                pauli_new = QubitOperator(string, 1j)
                pauli_list_new.append(pauli_new)
                if discard:
                    # only one of paulis is taken.
                    # skip to the next paulis
                    break
        pauli_list = pauli_list_new
    elif Quket.adapt.mode in ('pauli_spin', 'spin'):
        pauli_list = get_pauli_list_uccgsd(Quket.n_active_orbitals, DS=0, singles=True)[0]
    elif Quket.adapt.mode in ('pauli_spin_xy', 'qeb'):
        pauli_list = get_pauli_list_uccgsd(Quket.n_active_orbitals, DS=0, singles=True)[0]
        ### Remove Z (QEB-ADAPT)
        pauli_list_new = []
        for pauli in pauli_list:
            pauli_new = QubitOperator('',0)
            for op, coef in pauli.terms.items():
                target_list = []
                pauli_index = []
                string = ""
                for op_ in op:
                    m = op_[0]
                    if op_[1] == 'X':
                        target_list.append(m)
                        pauli_index.append(1)
                        string += f"X{m} "
                    elif op_[1] == 'Y':
                        target_list.append(m)
                        pauli_index.append(2)
                        string += f"Y{m} "
                pauli_new += QubitOperator(string, coef)
            pauli_new.terms.pop((), None)
            if len(pauli_new.terms) == 0:
                pass
            else:
                pauli_list_new.append(pauli_new)
        pauli_list = pauli_list_new

    elif Quket.adapt.mode in ('qeb1', 'qeb2', 'qeb3'):
        ### qeb with only alpha singles and alpha-beta doubles
        norbs = Quket.n_active_orbitals
        pauli_list = []
        for a in range(norbs):
            for i in range(norbs):
                fermi = FermionOperator(((2*a,1),(2*i,0)),1)
                if mapping == "jordan_wigner":
                    pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                elif mapping == "bravyi_kitaev":
                    pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), Quket.n_qubits)
                pauli_list.append(pauli)
        for b in range(norbs):
            if Quket.adapt.mode == 'qeb1':
                x = b+1
            elif Quket.adapt.mode in ('qeb2', 'qeb3'):
                x = norbs
            for a in range(x):
                if Quket.adapt.mode in ('qeb1', 'qeb2'):
                    y = b+1
                elif Quket.adapt.mode == 'qeb3':
                    y = norbs
                for j in range(y):
                    for i in range(j+1):
                        if b == j and a <= i:
                            continue
                        fermi = FermionOperator(((2*b+1,1),(2*a,1),(2*j+1,0),(2*i,0)),1)
#                        prints('fermi:',fermi)
                        if mapping == "jordan_wigner":
                            pauli = jordan_wigner(fermi - hermitian_conjugated(fermi))
                        elif mapping == "bravyi_kitaev":
                            pauli = bravyi_kitaev(fermi - hermitian_conjugated(fermi), Quket.n_qubits)
                        pauli_list.append(pauli)
        ### Remove Z (QEB-ADAPT)
        pauli_list_new = []
        for pauli in pauli_list:
            pauli_new = QubitOperator('',0)
            for op, coef in pauli.terms.items():
                target_list = []
                pauli_index = []
                string = ""
                for op_ in op:
                    m = op_[0]
                    if op_[1] == 'X':
                        target_list.append(m)
                        pauli_index.append(1)
                        string += f"X{m} "
                    elif op_[1] == 'Y':
                        target_list.append(m)
                        pauli_index.append(2)
                        string += f"Y{m} "
                pauli_new += QubitOperator(string, coef)
            pauli_new.terms.pop((), None)
            if len(pauli_new.terms) == 0:
                pass
            else:
                pauli_list_new.append(pauli_new)
        pauli_list = pauli_list_new

    elif Quket.adapt.mode == 'pauli_yz':
        # TEST : Yi and Zi Yi+1 as in the original qubit-ADAPT paper
        #        But this seems to ionize the state so it doesn't seem to work if HF is the initial state...
        #if QubitAdapt:
        #    Quket.init_state.set_Haar_random_state()
        #    Quket.init_state.load(mpi.bcast(Quket.init_state.get_vector()))
        pauli_list = []
        for i in range(Quket.n_qubits-1):
            pauli_list.append(QubitOperator(f'Y{i}', 1j))
            pauli_list.append(QubitOperator(f'Z{i} Y{i+1}', 1j))

    ncnot_list = None
    if Quket.tapered["operators"]:
        if Quket.adapt.mode in ('original', 'pauli_spin', 'spin'):
            ### Make NCNOT list for Fermionic ADAPT

            # TODO: This is a temporary workaround to count CNOT gates in the tapering-off scheme and the correct CNOT gate counts have to be adopted.
            # We do not consider the actual CNOT gate counts after tapering-off, which can change original paulis and can make it complicated to count CNOT gates.
            # Rather, we count the CNOT gates that are originally required for paulis WITHOUT TAPERING-OFF (just aiming for the speed-up delivered by tapering-off qubits).
            # I assume that using a tapering-off algorithm can reduce CNOT gate counts, because some of the so this is just temporary.

            ### First reduce the pauli set by symmetry (those that do not hold the symmetry will be removed)
            pauli_list, allowed_pauli_list =  get_allowed_pauli_list(Quket.tapering, pauli_list)
            ### Then make a list for Ncnot for each pauli
            ncnot_list = NCNOT_list_adapt(pauli_list, Quket.adapt.mode)
        pauli_list, allowed_pauli_list = transform_pauli_list(Quket.tapering, pauli_list, reduce=True)
        Quket.tapered["pauli_list"] = True

    elif Quket.symmetry_pauli:
        pauli_list, allowed_pauli_list =  get_allowed_pauli_list(Quket.tapering, pauli_list)

    ### Remove the same operators
    if Quket.adapt.mode != 'original':
        pauli_list = get_unique_list(pauli_list, sign=True)
    if ncnot_list is None:
        ### Make CNOT list
        ncnot_list = NCNOT_list_adapt(pauli_list, Quket.adapt.mode)

    pauli_dict = {}
    for pauli, ncnot in zip(pauli_list, ncnot_list):
        pauli_dict[str(pauli)] = ncnot

    return pauli_list, pauli_dict

def remove_z_from_pauli(pauli_list):
    """
    Remove Z matrices from pauli_list.
    Also create the Ncnot list
    """
    pauli_list_new = []
    ncnot_list = []
    for pauli in pauli_list:
        pauli_new = QubitOperator('',0)
        for op, coef in pauli.terms.items():
            target_list = []
            pauli_index = []
            string = ""
            for op_ in op:
                m = op_[0]
                if op_[1] == 'X':
                    target_list.append(m)
                    pauli_index.append(1)
                    string += f"X{m} "
                elif op_[1] == 'Y':
                    target_list.append(m)
                    pauli_index.append(2)
                    string += f"Y{m} "
            pauli_new += QubitOperator(string, coef)
        pauli_new.terms.pop((), None)
        if len(pauli_new.terms) == 0:
            pass
        else:
            pauli_list_new.append(pauli_new)
            if len(pauli_new.terms) == 2:
                ncnot_list.append(2)
            elif len(pauli_new.terms) == 8:
                ncnot_list.append(13)
    return pauli_list_new, ncnot_list

def get_allowed_pauli_list(Tapering, pauli_list):
    """Function
    Rerturn symmetry-allowed pauli_list
    """
    # List of transformed pauli operators
    new_pauli_list = []
    # List of surviving/discarded operators because of symmetry
    allowed_pauli_list = []
    i = 0
    for pauli in pauli_list:
        i+=1
        if type(pauli) is list:
            for pauli_ in pauli:
                new_pauli_, allowed = transform_pauli(Tapering, pauli_, reduce=False)
                #prints(i, allowed)
                if not allowed:
                    allowed_pauli_list.append(False)
                    break
                #prints(i, 'passed', allowed)
            else:
                new_pauli_list.append(pauli)
                allowed_pauli_list.append(True)
                continue

        else:
            new_pauli, allowed = transform_pauli(Tapering, pauli, reduce=False)
            if allowed:
                new_pauli_list.append(pauli)
                allowed_pauli_list.append(True)
            else:
                allowed_pauli_list.append(False)
#    for i, pauli in enumerate(new_pauli_list):
#        prints(i,  '    ',  pauli)
#    prints(allowed_pauli_list)
    return new_pauli_list, allowed_pauli_list

def NCNOT_list_adapt(pauli_list, mode):
    """
    Count Ncnot for each pauli and make a list.
    """
    NCNOT_list = []
    if mode in ('original', 'pauli_spin', 'spin'):
        for pauli in pauli_list:
            target_list = []
            if type(pauli) is list:
                for pauli_ in pauli:
                    for op, coef in pauli_.terms.items():
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

            else:
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

            ncnot = 0
            for target in target_list:
                if len(target) == 2:
                    # Singles
                    nsf = target[1] - target[0]
                    ncnot += 2*nsf + 1
                elif len(target) == 4:
                    # Doubles
                    nsf = target[3] - target[2] + target[1] - target[0]
                    ncnot += 2*nsf + 9
            NCNOT_list.append(ncnot)

    elif mode == 'pauli':
        for pauli in pauli_list:
            target_list = []
            for op, coef in pauli.terms.items():
                for op_ in op:
                    m = op_[0]
                    if op_[1] == 'X' or op_[1] == 'Y':
                        target_list.append(m)
            if len(target_list) == 2:
            ### single-excitation-like gate can be implemented with 2 CNOTs
                NCNOT_list.append(2)
            ###  double-excitation-like gate includes 4 terms  Xp Xq Xr Ys + ...
            ###  can be implemented with 20 CNOts.
            elif len(target_list) == 4:
                NCNOT_list.append(6)
            else:
                prints('Weird pauli in pauli:',pauli)
    elif mode == 'pauli_sz':
        for pauli in pauli_list:
            if len(pauli.terms) == 2:
            ### single-excitation-like gate can be implemented with 2 CNOTs
                NCNOT_list.append(2)
            ###  double-excitation-like gate includes 4 terms  Xp Xq Xr Ys + ...
            ###  can be implemented with 20 CNOts.
            elif len(pauli.terms) == 4:
                NCNOT_list.append(20)
            else:
                prints('Weird pauli in pauli_sz:',pauli)

    elif mode in ('pauli_spin_xy', 'qeb', 'qeb_reduced', 'qeb1','qeb2', 'qeb3'):
        ### For singles, 4 CNOTs --> 2 CNOTs, so 2 can be canceled out
        ### Similarly, for doubles, 48 CNOTs --> 13 CNOTs, so 35 can be canceled out
        for pauli in pauli_list:
            if len(pauli.terms) == 2:
                NCNOT_list.append(2)
            elif len(pauli.terms) == 8:
                NCNOT_list.append(13)
            else:
                prints('Weird pauli in qeb:',pauli)
    return NCNOT_list
