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
import copy
import itertools
from pprint import pprint
import time

import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import PauliRotation
from qulacs.state import inner_product

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.opelib import OpenFermionOperator2QulacsObservable, OpenFermionOperator2QulacsGeneralOperator
from quket.fileio import prints, printmat
from quket.utils.utils import get_occvir_lists, fermi_to_str, get_pauli_index_and_coef
from quket.opelib.circuit import make_gate
from quket.linalg import root_inv,lstsq, Lowdin_orthonormalization, tikhonov
from quket.lib import QuantumState, QubitOperator, FermionOperator, get_fermion_operator, jordan_wigner, hermitian_conjugated

def multiply_Hpauli(chi_i, n, pauli, db, j):
    """
    |chi_i> とnとpauliとdbとjを与えられたら|chi_i>  = - db/j  h[i]|chi_dash>を計算
    """
    coef = pauli.get_coef()
    circuit = make_gate(n, pauli.get_index_list(), pauli.get_pauli_id_list())
    circuit.update_quantum_state(chi_i)
    chi_i.multiply_coef(coef)
    if j!=0:
        chi_i.multiply_coef(-db/j)
    return chi_i

def calc_Hpsi(H, psi, shift=0, db=1, j=1):
    """Function
    Perform |chi> = -db/j (H - shift) |psi> 

    **If H|psi> is desired, simply set db = -j = 1 and shift = 0.

    Args:
        H (Observable): Hamiltonian
        psi (QuantumState): |psi>
        shift (float): Shift energy
        db (float): imaginary time step
        i (int): Order of Taylor expansion

    Returns:
        chi (QuantumState): -db/j (H - shift) |psi>
    """
    n = psi.get_qubit_count()
    nterms = H.get_term_count()
    chi = QuantumState(n)
    chi.multiply_coef(0)
    ### MPI ###
    my_chi = chi.copy()
    ipos, my_nterms = mpi.myrange(nterms)
    for i in range(ipos, ipos+my_nterms):
        chi_i = psi.copy()
        pauli = H.get_term(i)
        chi_i = multiply_Hpauli(chi_i, n, pauli, db, j)
        my_chi.add_state(chi_i)
    my_vec = my_chi.get_vector()
    vec = np.zeros(2**n, dtype=complex)
    vec = mpi.allreduce(my_vec, mpi.MPI.SUM)
    chi.load(vec)

    # Add (shift * db/j) |psi>
    chi_shift = psi.copy()
    chi_shift.multiply_coef(shift*db/j)
    chi.add_state(chi_shift)
    return chi

def make_pauli_id(num, n, active):
    """
    4ビットの場合
    [0 0 0 0]
    [0 0 0 3]
    [0 0 1 0]
    [0 0 1 1]
    .....
    となるように作る
    """
    id_ = []
    quo = num
    for i in range(len(active)):
        rem = quo % 4
        quo = (quo-rem)//4
        id_.append(rem)
    id_.reverse()
    full_id = np.zeros(n, dtype=int)
    j = 0
    for i in range(len(active)):
        full_id[active[i]] = id_[j]
        j += 1
    return full_id

def calc_expHpsi(psi, observable, n, db, shift=0, order=0, ref_shift=0):
    """Function
    Form delta
            delta = exp[-db (H-shift)] psi
    where exp[-db (H-shift)] is Taylor-expanded.
    The Taylor-expansion is truncated at `order`
    If order = 0 (default), the expansion is exactly performed. 
    For the "real" QITE, the proposed order is 1.
    """
    chi = psi.copy()
    chi_dash = psi.copy()
    expHpsi = psi.copy()  # Will hold exp[-db H] psi after while statement
    nterms = observable.get_term_count()
    from ..fileio import print_state

    d = 10.0
    j = 1
    phase = -1
    while d > 1e-16:
        chi.multiply_coef(0)
        ### MPI ###
        my_chi = chi.copy()
        ipos, my_nterms = mpi.myrange(nterms)
        for i in range(ipos, ipos+my_nterms):
            chi_i = chi_dash.copy()
            pauli = observable.get_term(i)
            chi_i = multiply_Hpauli(chi_i, n, pauli, db, j)
            my_chi.add_state(chi_i)
        my_vec = my_chi.get_vector()
        vec = mpi.allreduce(my_vec, mpi.MPI.SUM)
        chi.load(vec)
        phase *= -1
        # Add (shift * db/j) |chi_dash>
        shift_chi_dash = chi_dash.copy()
        shift_chi_dash.multiply_coef(shift*db/j)
        chi.add_state(shift_chi_dash)

        chi_dash = chi.copy()
        # chi  = H.psi  ->  1/2 H.(H.psi) -> 1/3 H.(1/2 H.(H.psi)) ...
        expHpsi.add_state(chi)
        #print_state(expHpsi)
        # check chi =  (1/j!) H^j psi  is small enough
        d = np.sqrt(chi.get_squared_norm())
        # Truncation order
        if j == order:
            break
        j += 1

    norm = expHpsi.get_squared_norm()
    #expHpsi.normalize(norm)
    return expHpsi, norm


def calc_delta(psi, observable, n, db, shift=0, order=0, ref_shift=0):
    """Function
    Form delta
            delta = (exp[-db (H-shift)] - 1) psi
    where exp[-db (H-shift)] is Taylor-expanded.
    The Taylor-expansion is truncated at `order`
    If order = 0 (default), the expansion is exactly performed. 
    For the "real" QITE, the proposed order is 1.
    """
    chi = psi.copy()
    chi_dash = psi.copy()
    expHpsi = psi.copy()  # Will hold exp[-db H] psi after while statement
    nterms = observable.get_term_count()
    from ..fileio import print_state

    d = 10.0
    j = 1
    while d > 1e-8:
        chi.multiply_coef(0)
        ### MPI ###
        my_chi = chi.copy()
        ipos, my_nterms = mpi.myrange(nterms)
        for i in range(ipos, ipos+my_nterms):
            chi_i = chi_dash.copy()
            pauli = observable.get_term(i)
            chi_i = multiply_Hpauli(chi_i, n, pauli, db, j)
            my_chi.add_state(chi_i)
        my_vec = my_chi.get_vector()
        vec = mpi.allreduce(my_vec, mpi.MPI.SUM)
        chi.load(vec)
        # Add (shift * db/j) |chi_dash>
        shift_chi_dash = chi_dash.copy()
        shift_chi_dash.multiply_coef(shift*db/j)
        chi.add_state(shift_chi_dash)

        chi_dash = chi.copy()
        # chi  = H.psi  ->  1/2 H.(H.psi) -> 1/3 H.(1/2 H.(H.psi)) ...
        expHpsi.add_state(chi)
        # check chi =  (1/j!) H^j psi  is small enough
        d = np.sqrt(chi.get_squared_norm())
        # Truncation order
        if j == order:
            break
        j += 1

    norm = expHpsi.get_squared_norm()
    expHpsi.normalize(norm)
    psi0 = psi.copy()
    psi0.multiply_coef(-1)
# コピーは不要と思われる
    #delta = expHpsi.copy()
    delta = expHpsi
    delta.add_state(psi0)

    #cm = 1/np.sqrt(norm)
    #cm = 1/np.sqrt(norm) * np.exp(shift*db)
    # With shift present, `norm` is the following value: 
    #  norm = <psi| Exp[-2db * (H-shift)] |psi> 
    #       = <psi| Exp[-2db * H] |psi> * Exp[2*shift*db]
    # Since `shift` may change at each iteration (if `shift = step` is invoked), 
    # we may define `cm` as a `shift` independent quantity, e.g.,
    #  cm = 1/sqrt[<psi| Exp(-2db * H) |psi>] 
    #     = 1/sqrt[norm] * Exp[shift*db]
    # which may be used for the subsequent QLanczos program.
    # This is basically the proposal used original article 
    # (where `shift` was not even introduced).
    # 
    # However, this deinition of `cm` becomes exponentially small
    # because Exp[shift*db] too is exponentially small (`shift` ~ total energy).
    # Therefore, we intend to cancel out this effect by defining
    # `cm` as
    #  (new)   cm = 1/sqrt[<psi| Exp[-2db *(H - E0)] |psi>]
    # with some reference energy E0, such as HF energy. 
    # Note that Exp[db * E0] becomes a constant and will cancel out when
    # evaluating the QLanczos matrix elements.
    # Therefore, we can set
    #   DE = shift - E0
    # such that
    #   E0 = shift - DE
    # and substitute this E0 to the above equation for `cm`,
    #   cm = 1/sqrt[<psi| Exp[-2db *(H - (shift - DE))] |psi>]
    #      = 1/sqrt[<psi| Exp[-2db *(H - shift)] |psi>] * Exp[db * DE]
    #      = 1/sqrt[norm]  * Exp[db * DE]
    #                       
    # Now both norm and Exp[db * DE] are both large enough (DE is considered small enough).
    DE = shift - ref_shift
    return delta, 1/np.sqrt(norm) * np.exp(DE*db)

def calc_psi(psi_dash, n, index, a, active):
    circuit = QuantumCircuit(n)
    for i, a_i in enumerate(a):
        if abs(a_i) > 0:
            pauli_id = make_pauli_id(i, n, active)
            gate = PauliRotation(index, pauli_id, a_i*(-2))
            circuit.add_gate(gate)
    circuit.update_quantum_state(psi_dash)
    #norm = psi_dash.get_squared_norm()
    #psi_dash.normalize(norm)
    return psi_dash


def calc_psi_lessH(psi_dash, n, index, a, id_set):
    circuit = QuantumCircuit(n)
    psi_next = psi_dash.copy()
    for i, a_i in enumerate(a):
        if abs(a_i) > 0:
            pauli_id = id_set[i]
            gate = PauliRotation(index, pauli_id, a_i*(-2))
            circuit.add_gate(gate)
    circuit.update_quantum_state(psi_next)
    #norm = psi_dash.get_squared_norm()
    #psi_dash.normalize(norm)
    return psi_next


def make_state1(i, n, active_qubit, index, psi_dash):
    pauli_id = make_pauli_id(i, n, active_qubit)
    circuit = make_gate(n, index, pauli_id)
    state = psi_dash.copy()
    circuit.update_quantum_state(state)
    return state


def calc_inner1(i, j, n, active_qubit, index, psi_dash):
    s_i = make_state1(i, n, active_qubit, index, psi_dash)
    s_j = make_state1(j, n, active_qubit, index, psi_dash)
    s = inner_product(s_j, s_i)
    return s

def Nonredundant_Overlap_list(pauli_list, n_qubits):
    """Function
    Gets the unique list of single pauli strings appearing in the overlap (metric) operators defined by
    qubit operators in 'pauli_list.' 
    They are often a linear combination of pauli strings that consist of redundant pauli strings.
    That means, all pairs pauli_i and pauli_j in pauli_list are multiplied (with the former being hermitian_conjugated)
    to yield a set of pauli strings, which are then decomposed to make a comprehensive nonredundant list
    by removing redundant strings.

    Each single pauli string in such a list is then converted to Qulacs.Observable class. 
    The original qubit operators (linear combinations of pauli strings) computed by pauli_i(dag) *pauli_j 
    are stored as pauli_ij_index and pauli_ij_coef, 
    which are the list of indices to refer to nonredundant list, and the original coefficient. 

    Returns:
        nonredundant_pauli_list ([Observable]): list of unique pauli strings required for overlap qulacs observables 
        pauli_ij_index ([int][int]): tells which unique paulis in nonredundant_pauli_list
                                should be used for pauli_i * pauli_j
        pauli_ij_coef ([int][complex]): complex Coefficients
    """
    size = len(pauli_list)

    my_pauli_ij_list = []
    sizeT = size*(size+1)//2
    ij = 0

    ### First do pauli*pauli
    # Decompose the pauli_list and remove the redundant ones, and set coefficients to one
    pauli_list_decomp = []
    for pauli in pauli_list:
        pauli_list_decomp.append(pauli) 
    time0 = time.time()
    size_ = len(pauli_list_decomp)
    ipos, my_ndim = mpi.myrange(size_*(size_+1)//2)
    my_pauli_dict = {}
    my_pauli_ij_index = []
    my_pauli_ij_coef = []
    from quket.utils import pauli_index
    for i in range(size):
        pauli_i = pauli_list[i]
        for j in range(i+1):
            if ij in range(ipos, ipos+my_ndim):
                pauli_j = pauli_list[j] 
                pauli_ij = hermitian_conjugated(pauli_i)*pauli_j

                index_list = []
                coef_list = []
                for op, coef in pauli_ij.terms.items():
                    ind = pauli_index(op)
                    if coef.real:
                        ### Hermitian
                        my_pauli_dict[str(ind)] = QubitOperator(op)
                    #else: 
                    #    ### anti-Hermitian
                    #    my_pauli_dict[str(ind)] = QubitOperator(op)
                    if abs(coef) > 1e-10: 
                        if coef.real:
                            index_list.append(ind)
                            coef_list.append(coef)
                        #else:
                        #    coef_list.append(coef )
                    
                my_pauli_ij_index.append(index_list)
                my_pauli_ij_coef.append(coef_list)
            ij += 1
    ####
    time1 = time.time()
    #prints('Timing0:', time1-time0, len(my_pauli_dict))
    #prints(f'rank = {mpi.rank}   {len(my_pauli_dict)}', root=mpi.rank)
    #pauli_dicts = mpi.allgather([my_pauli_dict])
    #prints(pauli_dicts)
    pauli_dict = {}



    ### Gather pauli_dict to root
    ###########
    ### BUG?  with "-genv I_MPI_FABRICS tcp" option, communication gets deadlocked?
    ###########
    k = 0 
    while 2**(k) < mpi.nprocs:
        if not mpi.rank % (2**k):
            if mpi.rank/(2**k) % 2:
            #    prints(f'1) {mpi.rank = }  dest = {mpi.rank - 2**k}  tag = {mpi.rank - 2**k}',root=mpi.rank)
                mpi.comm.send(my_pauli_dict, dest = mpi.rank - (2**k), tag=mpi.rank-2**k)
            #    prints(f'(1) OK : {mpi.rank = }', root=mpi.rank)
            else:
                if mpi.rank + 2**k < mpi.nprocs:
            #        prints(f'1) {mpi.rank = }  source = {mpi.rank + 2**k}  tag = {mpi.rank}',root=mpi.rank)
                    buf = mpi.comm.recv(source=mpi.rank + 2**k, tag=mpi.rank)
            #        prints(f'(1) OK : {mpi.rank = }', root=mpi.rank)
                    my_pauli_dict.update(buf)
        mpi.barrier()
        k += 1
    ### bcast pauli_dict
    pauli_dict = mpi.bcast(my_pauli_dict)

    ### allgather pauli_ij_index and pauli_ij_coef
    #pauli_ij_index = mpi.allgather(my_pauli_ij_index)
    #pauli_ij_coef = mpi.allgather(my_pauli_ij_coef)

    #prints(f'my_list = {len(my_pauli_ij_index)}  {len(my_pauli_ij_coef)}    {mpi.rank}', root=mpi.rank)
    timex = time.time()
    k = 0 
    while 2**(k) < mpi.nprocs:
        if not mpi.rank % (2**k):
            if mpi.rank/(2**k) % 2:
            #    prints(f'2) {mpi.rank = }  dest = {mpi.rank - 2**k}  tag = {mpi.rank - 2**k}', root=mpi.rank)
                mpi.comm.send(my_pauli_ij_index, dest = mpi.rank - (2**k), tag=mpi.rank-2**k)
            #    prints(f'2) OK : {mpi.rank = }', root=mpi.rank)
            else:
                if mpi.rank + 2**k < mpi.nprocs:
            #        prints(f'2) {mpi.rank = }  source = {mpi.rank + 2**k}  tag = {mpi.rank}',root=mpi.rank)
                    buf = mpi.comm.recv(source=mpi.rank + 2**k, tag=mpi.rank)
            #        prints(f'(2) OK : {mpi.rank = }', root=mpi.rank)
                    my_pauli_ij_index += buf
        mpi.barrier()
        k += 1
    time0 = time.time()
    pauli_ij_index = mpi.bcast(my_pauli_ij_index)
                    
    k = 0 
    while 2**(k) < mpi.nprocs:
        if not mpi.rank % (2**k):
            if mpi.rank/(2**k) % 2:
            #    prints(f'3) {mpi.rank = }  dest = {mpi.rank - 2**k}  tag = {mpi.rank - 2**k}', root=mpi.rank)
                mpi.comm.send(my_pauli_ij_coef, dest = mpi.rank - (2**k), tag=mpi.rank-2**k)
            #    prints(f'(3) OK : {mpi.rank = }', root=mpi.rank)
            else:
                if mpi.rank + 2**k < mpi.nprocs:
            #        prints(f'3) {mpi.rank = }  source = {mpi.rank + 2**k}  tag = {mpi.rank}',root=mpi.rank)
                    buf = mpi.comm.recv(source=mpi.rank + 2**k, tag=mpi.rank)
            #        prints(f'(3) OK : {mpi.rank = }', root=mpi.rank)
                    my_pauli_ij_coef += buf
        mpi.barrier()
        k += 1
    pauli_ij_coef = mpi.bcast(my_pauli_ij_coef)
    time2 = time.time()
    len_list = len(pauli_dict)
    ### 
    # Redefine pauli_ij_dict as qulacs.Observable
    sigma_ij_dict = {} 
    for key, value in pauli_dict.items():
        sigma_ij_dict[key] = OpenFermionOperator2QulacsObservable(value, n_qubits)

    prints(f'Timing for preparing nonredundant sigma list: {time.time() - time0:.1f} sec')

    return sigma_ij_dict, pauli_ij_index, pauli_ij_coef

def Overlap_by_pauli_list(pauli_list, psi, correction=False):
    """
    Test function for evaluating Sij = <Ei! Ej> wrt psi, 
    where Ei is each qubit operator in pauli_list.
    """
    from quket.opelib import evolve
    size = len(pauli_list)
    S = np.zeros((size, size), dtype=complex) 
    sizeT = size*(size+1)//2
    ipos, my_ndim = mpi.myrange(sizeT)
    S = np.zeros((size, size), dtype=complex) 
    X = np.zeros((size, 1), dtype=complex) 
    ij = 0
    for i in range(size):
        psi_i = evolve(pauli_list[i], psi)
        if correction:
            X[i, 0] = inner_product(psi_i, psi)
        for j in range(i+1):
            if ij in range(ipos, ipos+my_ndim):
                psi_j = evolve(pauli_list[j], psi)
                S[i,j] = inner_product(psi_i,psi_j)
                S[j,i] = S[i,j].conjugate()
            ij += 1
    S = mpi.allreduce(S)
    S = S - X @ X.T
    return S
            
def Overlap_by_pauli_list_IJ(pauli_list, psi_I, psi_J):
    """
    Test function for evaluating Sij = <I|Ei! Ej|J> wrt psi_I and psi_J, 
    where Ei is each qubit operator in pauli_list.
    Also compute 1/2 i(<I|Ei|J> - <J|Ei|I>) = Im <I|Ei|J>
    """
    from quket.opelib import evolve
    size = len(pauli_list)
    ipos, my_ndim = mpi.myrange(size**2)
    SIJ = np.zeros((size, size), dtype=float) 
    ij = 0
    for i in range(size):
        psi_i = evolve(pauli_list[i], psi_I)
        for j in range(size):
            if ij in range(ipos, ipos+my_ndim):
                psi_j = evolve(pauli_list[j], psi_J)
                SIJ[i,j] = inner_product(psi_i,psi_j).real 
    #for i in range(size):
    #    psi_i = evolve(pauli_list[i], psi_I)
    #    for j in range(size):
    #        if ij in range(ipos, ipos+my_ndim):
    #            psi_j = evolve(pauli_list[j], psi_J)
    #            SIJ[i,j] = inner_product(psi_i,psi_j).real 
    #            #prints((inner_product(psi_i, psi_J) * inner_product(psi_I, psi_j)).real)
                
    #for i in range(size):
    #    psi_Ii = evolve(pauli_list[i], psi_I)
    #    psi_Ji = evolve(pauli_list[i], psi_J)
    #    for j in range(size):
    #        if ij in range(ipos, ipos+my_ndim):
    #            psi_Ij = evolve(pauli_list[j], psi_I)
    #            psi_Jj = evolve(pauli_list[j], psi_J)
    #            SIJ[i,j] = 0.5*(inner_product(psi_Ii,psi_Jj) + inner_product(psi_Ji, psi_Ij)).real 
    #        ij += 1
    SIJ = mpi.allreduce(SIJ)
    SIJ = (SIJ + SIJ.T) / 2

    return SIJ
            
def Overlap_by_nonredundant_pauli_list(nonredundant_sigma, pauli_ij_index, pauli_ij_coef, psi):
    """Function
    Compute Sij as expectation values
    Sij = <psi | pauli_i(dag) pauli_j | psi>
        = sum_k c_k <psi | sigma_k |psi>
    where pauli_i and pauli_j are qubit operators defined as a linear combination of pauli
    strings, sigma_k are the nonredundant pauli strings, and c_k are the expansion coefficients.


    First evaluate expectation values for the non-redundant set of sigma_i(dag) sigma_j,
    prepared in "sigma_list", and then distribute them in the correct order to
    make S matrix.
    
    Args:
        size (int): Dimension of S
        nonredundant_sigma (list): List of non-redundant sigma_k as Qulacs.Observable
        pauli_ij_index (list): Nested list of positions for pauli_i(dag) pauli_j in nonredundant_sigma
        pauli_ij_coef (list): Nested list of coefficients for pauli_i(dag) pauli_j in nonredundant_sigma
        psi (QuantumState): |psi> 
    
    Returns:
        S (2darray): S matrix
    """
    len_list = len(nonredundant_sigma)
    Sij_list = np.zeros(len_list) 
    Sij_dict = {}
    ipos, my_ndim = mpi.myrange(len_list)
    ij = 0
    t0 = time.time()
    key_list = []
    for  key, obs in nonredundant_sigma.items():
        key_list.append(key)
        if ipos <= ij < ipos + my_ndim:
            Sij_list[ij] = obs.get_expectation_value(psi)
        ij += 1
    t1 = time.time()

    ### Gather pauli_dict to root
    #k = 0 
    #while 2**(k) < mpi.nprocs:
    #    if not mpi.rank % (2**k):
    #        if mpi.rank/(2**k) % 2:
    #            mpi.comm.send(Sij_dict, dest = mpi.rank - (2**k), tag=mpi.rank-2**k)
    #        else:
    #            if mpi.rank + 2**k < mpi.nprocs:
    #                buf = mpi.comm.recv(source=mpi.rank + 2**k, tag=mpi.rank)
    #                Sij_dict.update(buf)
    #    k += 1
    Sij_list = mpi.allreduce(Sij_list)

    for ij, k in enumerate(key_list):
        Sij_dict[k] = Sij_list[ij]
    ### bcast pauli_dict
    #Sij_dict = mpi.bcast(Sij_dict)

    t2 = time.time()
    ### allgather pauli_ij_index and pauli_ij_coef
    #Sij_list = mpi.allreduce(Sij_list, mpi.MPI.SUM)
    # Distribute Sij
    size_ = len(pauli_ij_index)
    if size_ != len(pauli_ij_coef):
        error(f'Inconsistent len(pauli_ij_index) = {len(pauli_ij_index)} and len(pauli_ij_coef)  = {len(pauli_ij_coef)}')
    size = (-1+np.sqrt(1+8*size_))//2
    if not size.is_integer():
        raise ValueError(f'Vector is not [N*(N+1)] in symm (size = {size_})')
    size = int(size)
    ij = 0
    sizeT = size*(size+1)//2
    ipos, my_ndim = mpi.myrange(sizeT)
    S = np.zeros((size, size), dtype=complex) 
    for i in range(size):
        for j in range(i+1):
            if ij in range(ipos, ipos+my_ndim):
                for idx, coef in zip(pauli_ij_index[ij], pauli_ij_coef[ij]):
                    #prints('idx = ', idx, ' coef = ',coef, ' Sij ',Sij_dict[str(idx)])
                    S[i, j] += coef * Sij_dict[str(idx)]
                if i != j:
                    S[j, i] = S[i, j].conjugate()
            ij += 1
    S = mpi.allreduce(S, mpi.MPI.SUM)
    t3 = time.time()
    #prints(f'T0 = {t1 - t0}')
    #prints(f'T1 = {t2 - t1}')
    #prints(f'T2 = {t3 - t2}')
    return S


def qlanczos(cm_list, energy, s2, t, H_q=None, S_q=None, S2_q=None):
    """
    Single-state QLancozs.

    Args:
        cm_list (list): Normalization constant c at each m-th step.
        energy (list): Energy expectation value at each m-th step.
        s2 (list): S2 expectation value at each m-th step.
        t (int): Time t
        H_q (list): Previous Hamiltonian matrix (m < t).
        S_q (list): Previous Overlap matrix (m < t).
        S2_q (list): Previous S2 matrix (m < t).
        
    Returns:
        q_en (list): Energies as eigenvalues.
        S2_en (list): S2 as expectation values of QLanczos.
        H_q (list): Updated Hamiltonian matrix.
        S_q (list): Updated Overlap matrix.
        S2_q (list): Updated S2 matrix.

    Author(s): Yoohee Ryo, Takashi Tsuchimochi
    """
    # Construct Heff and Seff
    q_en = 0
    if t%2 == 0:
        ndim = (t+1)//2+1
    else:
        ndim = (t+1)//2

    if H_q is None or S_q is None:
        H_q = np.zeros((ndim, ndim), dtype=float)
        S_q = np.zeros((ndim, ndim), dtype=float)
        S2_q = np.zeros((ndim, ndim), dtype=float)
        istart = 0
    else:
        zeros = np.zeros((1,ndim-1), dtype=float)
        H_q = np.block([[H_q, zeros.T],
                       [zeros, 0]])
        S_q = np.block([[S_q, zeros.T],
                       [zeros, 0]])
        S2_q = np.block([[S2_q, zeros.T],
                       [zeros, 0]])
        istart = ndim-1
    for ti in range(istart, ndim):
        if t%2 == 0:
            ti2 = 2*ti
        else:
            ti2 = 2*ti + 1
        for tj in range(ti+1):
            if t%2 == 0:
                tj2 = 2*tj
            else:
                tj2 = 2*tj + 1
            tr = (ti2 + tj2)//2
            # Compute S element (based on Eq.56 in SI)
            #S_q[ti, tj] = nm_list[ti2] * nm_list[tj2] / nm_list[tr]**2

            # Use c[m] list instead of n[m] list
            # 1/(nm_list[ti2])**2 = c[0] * c[1] * ... * c[tj2] * ... * c[tr] * ... * c[ti2]
            # 1/(nm_list[tj2])**2 = c[0] * c[1] * ... * c[tj2]
            # 1/(nm_list[tr])**2  = c[0] * c[1] * ... * c[tj2] * ... * c[tr]
            # and therefore
            # S_q[ti,tj] =  sqrt(c[tj2+1] * ... * c[tr] / c[tr+1] * ... * c[ti2] )

            # Numerator
            Numerator = 1
            for tk in range(tj2+1, tr+1):
                Numerator *= cm_list[tk]
            # Denominator
            Denominator = 1
            for tk in range(tr+1, ti2+1):
                Denominator *= cm_list[tk]
            S_q[ti, tj] = np.sqrt(Numerator/Denominator)
            #S_q[ti, tj] = np.random.rand()
            #prints(Numerator, Denominator)

            # Compute H element (based on Eq.57 in SI)
            H_q[ti, tj] = S_q[ti, tj] * energy[tr]
            S2_q[ti, tj] = S_q[ti, tj] * s2[tr]

            # Symmetrize
            S_q[tj, ti] = S_q[ti, tj]
            H_q[tj, ti] = H_q[ti, tj]
            S2_q[tj, ti] = S2_q[ti, tj]
    s = 0.999
    scheme = 0
    MaxLen = 5000
    
    # Determine the imaginary time step used for QLanczos based on the regularization
    if scheme == 0:
        IList = [i for i in range(ndim)]
    elif scheme == 1:
    #########################################
    ### Original regularization algorithm ###
    #########################################
        IList = [0]
        jLast = 0
        j = 0
        while jLast < t//2 and j < t//2:
            j = jLast
            for k in range(j, t//2):
                j += 1
                #if mpi.main_rank:
                #    print('jLast =', jLast, '  S_q = ', S_q[jLast, j])
                if abs(S_q[jLast, j]) < s:
                    IList.append(j)
                    jLast = j
                    break

    elif scheme == 2:
    #########################################
    ### Backward regularization algorithm ###
    #########################################
        # Start from the current time instead of HF (t=0)
        IList = [ndim-1]
        jLast = ndim-1
        j = (t+1)//2
        while jLast >= 0 and j >= 0:
            j = jLast
            for k in range(jLast, -1, -1):
                j -= 1
                if abs(S_q[jLast,k]) < s:
                    IList.append(k)
                    jLast = k
                    break
            if len(IList) == MaxLen:
                break
        
    elif scheme == -2:
    ####################################
    ### Backward selection algorithm ###
    ####################################
        # Start from the current time instead of HF (t=0)
        last = max(-1, ndim-1-MaxLen)
        IList = [i for i in range(ndim-1, last, -1)]

    elif scheme == 3:
    #########################################
    ### Test regularization algorithm ###
    #########################################
        IList = [(t+1)//2]
        jLast = (t+1)//2
        jLast = (t+1)//2
        for j in range((t+1)//2-1, 0, -1):
            if j >= IList[-1]:
                continue
            NDimS = len(IList) + 1
            H_test = np.zeros((NDimS,NDimS), dtype=float)
            S_test = np.zeros((NDimS,NDimS), dtype=float)
            for k1 in range(NDimS - 1):
                for k2 in range(k1+1):
                    H_test[k1,k2] = H_q[IList[k1], IList[k2]]
                    H_test[k2,k1] = H_test[k1,k2]
                    S_test[k1,k2] = S_q[IList[k1], IList[k2]]
                    S_test[k2,k1] = S_test[k1,k2]

            for k in range(IList[-1]-1, 0, -1): 
                #j -= 1
                for k1 in range(NDimS - 1):
                    #prints('k1,j', IList[k1],k)
                    H_test[k1, NDimS-1] = H_q[IList[k1], k]
                    H_test[NDimS-1, k1] = H_test[k1, NDimS-1]
                    S_test[k1, NDimS-1] = S_q[IList[k1], k]
                    S_test[NDimS-1, k1] = S_test[k1, NDimS-1]
                H_test[NDimS-1, NDimS-1] = H_q[k, k]
                S_test[NDimS-1, NDimS-1] = S_q[k, k]
                #q_en = eig(H_test, S_test)
                #prints('t = {}   k = {}   {}'.format(t, k, q_en))
                s, v = np.linalg.eig(S_test) 
                judge = True
                for si in s:
                    if 1e-7 < si < 1e-4 or si < 0: 
                        judge = False
                        break
                if judge:
                    IList.append(k)
                    break
            if len(IList) == 2:
                break
    IList.sort() 
    Nused = len(IList)
    Sused = np.zeros((Nused,Nused), dtype=float)
    Hused = np.zeros((Nused,Nused), dtype=float)
    S2used = np.zeros((Nused,Nused), dtype=float)
    for k1 in range(Nused):
        for k2 in range(k1+1):
            Sused[k2,k1] = S_q[IList[k2], IList[k1]]
            Sused[k1,k2] = Sused[k2,k1]
            Hused[k2,k1] = H_q[IList[k2], IList[k1]]
            Hused[k1,k2] = Hused[k2,k1]
            S2used[k2,k1] = S2_q[IList[k2], IList[k1]]
            S2used[k1,k2] = S2used[k2,k1]
    # Overwrite
    #H_q = Hused
    #S_q = Sused

    #printmat(Hused, name='Hused')
    #printmat(Sused, name='Sused')
    if cf.debug:
        printmat(H_q.real, name='H_q')
        printmat(S_q.real, name='S_q')
    #    prints('IList = ',IList)
    #    printmat(Hused, name='Hused')
    #    printmat(Sused, name='Sused')
    #    #printmat(S2_q, name='S2used')

    #scheme = 4
    #if scheme == 4 and Sused.shape[0] > 6:
    #    ### Sequential Orthonormalization
    #    M = Sused.shape[0]
    #    block_size = 2
    #    nrange = M 
    #    left = M-2
    #    Sred = copy.deepcopy(Sused)
    #    Sred = (Sred + Sred.T)/2
    #    U = np.identity(M,dtype=float) 
    #    while left > 0:
    #        S_block = Sred[:block_size, :block_size] 
    #        root_invS = root_inv(S_block, eps=1e-8)
    #        dim = root_invS.shape
    #        prints(dim[1], M)
    #        zeroT = np.zeros((dim[0], left), dtype=float)
    #        zero = np.zeros((dim[1], left), dtype=float)
    #        one = np.identity(left, dtype=float)
    #        U_ = np.block([[root_invS, zeroT], [zero.T, one]])
    #        Sred = U_.T@Sred@U_
    #        Sred = (Sred + Sred.T)/2
    #        U = U@U_

    #        left -= 1

    #        flag = True
    #        Sred_ = Sred[:block_size, :block_size] - np.identity(block_size, dtype=float)
    #        if np.trace(Sred_@Sred_) > 1e-8:
    #            flag = False
    #        if flag:
    #            block_size += 1
    #    # Final orthonormalization
    #    S_block = Sred 
    #    root_invS = root_inv(S_block, eps=1e-8)
    #    Sred = root_invS.T@Sred@root_invS
    #    Sred = (Sred + Sred.T)/2
    #    root_invS = U@root_invS
    #else:
    # Check Singularity of S
    root_invS = root_inv(Sused, eps=1e-8)
    s = np.linalg.eigh(Sused)[0]
    s.sort()
    if s[0] < -1e-6 and mpi.main_rank:
        print(f'S is not positive semi-definite: smallest eigenvalue = ', s[0])
    s = np.sort(s)[::-1]
    importance_index = []
    for k, s_ in enumerate(s):
        if s_ > 0.01:
            importance_index.append(k)
    root_invS = root_invS[:,::-1]
    H_ortho = root_invS.T@Hused@root_invS
    if np.isnan(H_ortho).any() or np.isinf(H_ortho).any():
        print("contain nan or inf")
        printmat(Sused, name="Sused")
        error()
    else:
        en, dvec = np.linalg.eigh(H_ortho)
    ### The ground state energy should be the one 
    ### with the eigenvector that has the largest
    ### overlaps with the used QITE states, e.g.,
    ### that with the largest 0th component.

    index = np.argsort(en)
    en = en[index]
    dvec  = dvec[:, index]
    cvec  = root_invS@dvec
    #printmat(dvec,name='dvec', eig=s)
    #printmat(cvec,name='cvec')

    # Check each eigenvector in dvec to see if it has an important component. 
    physical_index = []
    for ind in importance_index:
        for k in range(dvec.shape[1]): 
            if (dvec[ind, k])**2 > 0.05 and k not in physical_index:
                physical_index.append(k)

    physical_index = np.sort(physical_index)
    #index_gs = np.argmax(abs(dvec[-1]))
    #ground_state_energy = en[index_gs] 
    #index = []
    #for i in range(len(en)):
    #    prints(dvec[-1,i])
    #    if abs(dvec[-1, i])  > 0.1:
    #        index.append(i)
    ### For excited states: Due to the variational
    ### principle in QLanczos, those energies that 
    ### are lower than ground state energy is 
    ### unphysical and should be removed.

    q_en = [e for e in en[physical_index]]
    #q_en = en[index]
    #prints(en)

    ### S**2
    S2dig = cvec.T@S2used@cvec
    q_s2 = [S2dig[i, i] for i in physical_index]
    #q_s2 = [S2dig[i, i] for i in index]

    #### Debug:
    #if cf.debug:
    #    Hdig = cvec.T@Hused@cvec
    #    q_en_debug = [Hdig[i, i].real for i in range(index_gs, len(en))]
    #    for i in range(len(q_en)):
    #        if abs(q_en[i] - q_en_debug[i]) > 1e-6:
    #            prints('q_en = ',q_en)
    #            prints('Hdig = ',q_en_debug)
    #            error(" Something is wrong ")

    #prints('q_en = ',q_en[:2])
    # Tikhonov regularization of Sused
    #Sinv = tikhonov(S_q, eps=1e-7)
    #prints('q_tik = ',np.sort(np.linalg.eig(Sinv @ H_q)[0])[:2])

    return q_en, q_s2, H_q, S_q, S2_q


def Form_Smat(size, sigma_list, sigma_ij_index, sigma_ij_coef, psi):
    """Function
    Compute Sij as expectation values of sigma_list.
    Sij = <psi | sigma_i(dag) sigma_j | psi>

    First evaluate expectation values for the non-redundant set of sigma_i(dag) sigma_j,
    prepared in "sigma_list", and then distribute them in the correct order to
    make S matrix.
    
    Args:
        size (int): Dimension of S
        sigma_list (list): List of non-redundant sigma_i(dag) sigma_j
        sigma_ij_index (list): List of positions for sigma_i(dag) sigma_j in sigma_list
        sigma_ij_coef (list): List of coefficients for sigma_i(dag) sigma_j in sigma_list
        psi (QuantumState): |psi> 

    """
    len_list = len(sigma_list)
    Sij_list = np.zeros(len_list) 
    Sij_my_list = np.zeros(len_list)
    ipos, my_ndim = mpi.myrange(len_list)
    for iope in range(ipos, ipos+my_ndim):
        Sij_my_list[iope] = sigma_list[iope].get_expectation_value(psi)
    Sij_list = mpi.allreduce(Sij_my_list, mpi.MPI.SUM)

    # Distribute Sij
    ij = 0
    sizeT = size*(size-1)//2
    ipos, my_ndim = mpi.myrange(sizeT)
    S = np.zeros((size, size), dtype=complex) 
    my_S = np.zeros((size, size), dtype=complex) 
    for i in range(size):
        for j in range(i):
            if ij in range(ipos, ipos+my_ndim):
                idx = sigma_ij_index[ij]
                coef = sigma_ij_coef[ij]
                my_S[i, j] = coef*Sij_list[idx]
                my_S[j, i] = my_S[i, j].conjugate()

            ij += 1
    S = mpi.allreduce(my_S, mpi.MPI.SUM)
    for i in range(size):
        S[i, i] = 1
    return S


def msqlanczos(d_list, nstates, t, S_list, H_list, S2_list, S_q, H_q, S2_q):
    """
    Single-state QLancozs.

    Args:
        cm_list (list): Normalization constant c at each m-th step.
        energy (list): Energy expectation value at each m-th step.
        s2 (list): S2 expectation value at each m-th step.
        t (int): Time t
        H_q (list): Previous Hamiltonian matrix (m < t).
        S_q (list): Previous Overlap matrix (m < t).
        S2_q (list): Previous S2 matrix (m < t).
        
    Returns:
        q_en (list): Energies as eigenvalues.
        S2_en (list): S2 as expectation values of QLanczos.
        H_q (list): Updated Hamiltonian matrix.
        S_q (list): Updated Overlap matrix.
        S2_q (list): Updated S2 matrix.

    Author(s): Yoohee Ryo, Takashi Tsuchimochi
    """
    t += 1
    ndim = t//2+1

    if t == 2:
        ### Initialize
        S_q = S_list[0]
        H_q = H_list[0]
        S2_q = S2_list[0]

    l = t
    for l_dash in range(0, t+2, 2):
        l_half = int((l+l_dash)/2)
        if l == l_dash:
            mini_s_q = S_list[l_dash]
            mini_h_q = H_list[l_dash]
            mini_s2_q = S2_list[l_dash]
        else:
            d_1 = d_list[l_half]
            for it in range(l_half+1, l):
                d_1 = d_1 @ d_list[it]

            d_2 = d_list[l_dash]
            for it in range(l_dash+1, l_half):
                d_2 = d_2 @ d_list[it]
#            printmat(d_1,f'd_1({l_dash})')
#            printmat(d_2,f'd_2({l_dash})')
#            printmat(S_list[l_half],f'S_list({l_half})')

            mini_s_q = d_1.T@S_list[l_half]@(np.linalg.inv(d_2))
            mini_h_q = d_1.T@H_list[l_half]@(np.linalg.inv(d_2))
            mini_s2_q = d_1.T@S2_list[l_half]@(np.linalg.inv(d_2))

        if l_dash == 0:
            s_row = mini_s_q.transpose()
            s_line = mini_s_q
            h_row = mini_h_q.transpose()
            h_line = mini_h_q
            s2_row = mini_s2_q.transpose()
            s2_line = mini_s2_q
        elif l_dash == l:
            s_line = np.block([s_line, mini_s_q])
            h_line = np.block([h_line, mini_h_q])
            s2_line = np.block([s2_line, mini_s2_q])
        else:
            s_row = np.block([[s_row], [mini_s_q.transpose()]])
            s_line = np.block([s_line, mini_s_q])
            h_row = np.block([[h_row], [mini_h_q.transpose()]])
            h_line = np.block([h_line, mini_h_q])
            s2_row = np.block([[s2_row], [mini_s2_q.transpose()]])
            s2_line = np.block([s2_line, mini_s2_q])

    S_q = np.block([S_q, s_row])
    S_q = np.block([[S_q], [s_line]])
    H_q = np.block([H_q, h_row])
    H_q = np.block([[H_q], [h_line]])
    S2_q = np.block([S2_q, s2_row])
    S2_q = np.block([[S2_q], [s2_line]])
#    printmat(H_q.real, name="H_q")
#    printmat(S_q.real, name="S_q")

    if cf.debug:
        #    prints(f"S_q:{S_q}")
        #    prints(f"H_q:{H_q}")
        printmat(H_q.real, name="H_q")
        printmat(S_q.real, name="S_q")
        #printmat(S2_q.real, name="S2_q")

    MaxLen = 5
    s = 0.99
    last = max(-1, ndim-1-MaxLen)
    IList = [i for i in range(ndim-1, last, -1)]
    IList.sort()
    scheme = 2
    if scheme == 0:
        IList = [i for i in range(ndim)]
    elif scheme == 2:
    #########################################
    ### Backward regularization algorithm ###
    #########################################
        # Start from the current time instead of HF (t=0)
        IList = [ndim-1]
        jLast = ndim-1
        j = (t+1)//2
        while jLast >= 0 and j >= 0:
            j = jLast
            for k in range(jLast, -1, -1):
                j -= 1
                if abs(S_q[jLast,k]) < s:
                    IList.append(k)
                    jLast = k
                    break
            if len(IList) == MaxLen:
                break
        
    elif scheme == -2:
    ####################################
    ### Backward selection algorithm ###
    ####################################
        # Start from the current time instead of HF (t=0)
        last = max(-1, ndim-1-MaxLen)
        IList = [i for i in range(ndim-1, last, -1)]
    Nused = len(IList)
    # Sused = np.zeros((Nused, Nused), dtype=float)
    # Hused = np.zeros((Nused, Nused), dtype=float)
    for k1 in range(Nused):
        for k2 in range(k1+1):
            index_k1 = IList[k1]*nstates
            index_k2 = IList[k2]*nstates
            index_k1_fin = index_k1+nstates
            index_k2_fin = index_k2+nstates
            # mini_s_u = S_q[index_k2:index_k2_fin, index_k1:index_k1_fin]
            mini_s_u = S_q[index_k2:index_k2_fin]
            mini_s_u = mini_s_u[:, index_k1:index_k1_fin]
            mini_h_u = H_q[index_k2:index_k2_fin]
            mini_h_u = mini_h_u[:, index_k1:index_k1_fin]
            mini_s2_u = S2_q[index_k2:index_k2_fin]
            mini_s2_u = mini_s2_u[:, index_k1:index_k1_fin]

            if k1 == 0 and k2 == 0:
                Sused = mini_s_u
                Hused = mini_h_u
                S2used = mini_s2_u
            elif k1 == k2:
                s_line = np.block([s_line, mini_s_u])
                h_line = np.block([h_line, mini_h_u])
                s2_line = np.block([s2_line, mini_s2_u])
                Sused = np.block([Sused, s_row])
                Sused = np.block([[Sused], [s_line]])
                Hused = np.block([Hused, h_row])
                Hused = np.block([[Hused], [h_line]])
                S2used = np.block([S2used, s2_row])
                S2used = np.block([[S2used], [s2_line]])
            elif k2 == 0:
                s_row = mini_s_u.transpose()
                s_line = mini_s_u
                h_row = mini_h_u.transpose()
                h_line = mini_h_u
                s2_row = mini_s2_u.transpose()
                s2_line = mini_s2_u
            else:
                s_row = np.block([[s_row], [mini_s_u.transpose()]])
                s_line = np.block([s_line, mini_s_u])
                h_row = np.block([[h_row], [mini_h_u.transpose()]])
                h_line = np.block([h_line, mini_h_u])
                s2_row = np.block([[s2_row], [mini_s2_u.transpose()]])
                s2_line = np.block([s2_line, mini_s2_u])
    if cf.debug:
        printmat(Hused.real, name="Hused")
        printmat(Sused.real, name="Sused")
    # 方程式とく
    Hused = Hused.real
    Sused = Sused.real
    root_invS = root_inv(Sused, eps=1e-8)
    s = np.linalg.eigh(Sused)[0]
    s.sort()
    if s[0] < -1e-6 and mpi.main_rank:
        print(f'S is not positive semi-definite: smallest eigenvalue = ', s[0])

    #H_ortho = root_invS.T@Hused@root_invS
    #if np.isnan(H_ortho).any() or np.isinf(H_ortho).any():
    #    print("contain nan or inf")
    #    printmat(Sused, name="Sused")
    #    error()
    #else:
    #    en, dvec = np.linalg.eigh(H_ortho)

    #index = np.argsort(en)
    #en = en[index]
    #dvec = dvec[:, index]

    #sq_list = []
    #last = len(en)-1
    #for i in range(len(en)):
    #    vec = dvec[:, i]
    #    sq = 0
    #    for j in range(nstates):
    #        sq = sq+vec[last-j]**2
    #    sq_list.append(sq)
    #sq_index = np.argsort(sq_list)[::-1]
    #prints(en)
    ##en = en[sq_index[0]:sq_index[-1]]
    #en = en[sq_index]

    s = np.sort(s)[::-1]
    importance_index = []
    for k, s_ in enumerate(s):
        if s_ > 0.01:
            importance_index.append(k)
    root_invS = root_invS[:,::-1]
    H_ortho = root_invS.T@Hused@root_invS
    if np.isnan(H_ortho).any() or np.isinf(H_ortho).any():
        print("contain nan or inf")
        printmat(Sused, name="Sused")
        error()
    else:
        en, dvec = np.linalg.eigh(H_ortho)
    ### The ground state energy should be the one 
    ### with the eigenvector that has the largest
    ### overlaps with the used QITE states, e.g.,
    ### that with the largest 0th component.

    index = np.argsort(en)
    en = en[index]
    dvec  = dvec[:, index]
    cvec  = root_invS@dvec
    printmat(dvec,name='dvec', eig=s)
    #printmat(cvec,name='cvec')

    # Check each eigenvector in dvec to see if it has an important component. 
    physical_index = []
    for ind in importance_index:
        for k in range(dvec.shape[1]): 
            if (dvec[ind, k])**2 > 0.1 and k not in physical_index:
                physical_index.append(k)

    physical_index = np.sort(physical_index)
    #index_gs = np.argmax(abs(dvec[-1]))
    #ground_state_energy = en[index_gs] 
    #index = []
    #for i in range(len(en)):
    #    prints(dvec[-1,i])
    #    if abs(dvec[-1, i])  > 0.1:
    #        index.append(i)
    ### For excited states: Due to the variational
    ### principle in QLanczos, those energies that 
    ### are lower than ground state energy is 
    ### unphysical and should be removed.

    q_en = [e for e in en[physical_index]]
    #q_en = en[index]
    #prints(en)

    ### S**2
    S2dig = cvec.T@S2used@cvec
    q_s2 = [S2dig[i, i] for i in physical_index]
    #q_s2 = [S2dig[i, i] for i in index]

    #### Debug:
    #if cf.debug:
    #    Hdig = cvec.T@Hused@cvec
    #    q_en_debug = [Hdig[i, i].real for i in range(index_gs, len(en))]
    #    for i in range(len(q_en)):
    #        if abs(q_en[i] - q_en_debug[i]) > 1e-6:
    #            prints('q_en = ',q_en)
    #            prints('Hdig = ',q_en_debug)
    #            error(" Something is wrong ")

    #prints('q_en = ',q_en[:2])
    # Tikhonov regularization of Sused
    #Sinv = tikhonov(S_q, eps=1e-7)
    #prints('q_tik = ',np.sort(np.linalg.eig(Sinv @ H_q)[0])[:2])

#    index_gs = np.argmax(abs(dvec[-1]))
#    """
#    ground_state_energy = en[index_gs]
#    q_en = [e for e in en[index_gs:]]
#    """
#    q_en = [en[i] for i in range(nstates)]
#    q_index = np.argsort(q_en)
#    q_en = np.sort(q_en)
#    cvec = root_invS@dvec
#    S2dig = cvec.T@S2used@cvec
#    q_s2 = [S2dig[i, i].real for i in sq_index]
#    q_s2=[q_s2[i] for i in range(nstates)]
#    q_s2 = [q_s2[i] for i in q_index]
#    # Debug:
#    if cf.debug:
#        Hdig = cvec.T@Hused@cvec
#        q_en_debug = [Hdig[i, i].real for i in sq_index]
#        q_en_debug = [q_en_debug[i] for i in range(nstates)]
#        q_en_debug = [q_en_debug[i] for i in q_index]
#        printmat(q_s2,name='q_s2')
#        printmat(q_en_debug,name='q_en_debug')
#        for i in range(len(q_en)):
#            if abs(q_en[i] - q_en_debug[i]) > 1e-6:
#                print('q_en = ', q_en)
#                print('Hdig = ', q_en_debug)
#                print('S2dig = ', q_s2)
#                print(" Something is wrong ")

    return S_q, H_q, q_en, S2_q, q_s2

