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

fci.py

FCI vqe solver using pyscf fci coefficients.
(This is an inefficient but straightfoward way of mapping fci coefficients onto the qubit representation)

"""

from operator import itemgetter
from itertools import combinations

import numpy as np
import scipy as sp
from qulacs import QuantumCircuit
from qulacs.state import inner_product

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import prints
from quket.linalg import Binomial
from quket.lib import QuantumState, QubitOperator
from .utils import cost_mpi, jac_mpi_deriv, jw2bk, bk2jw, transform_state_jw2bk

def cost_fci(norbs, nalpha, nbeta, coeff, NonzeroList, Hamiltonian, lower_states=None, mapping='jordan_wigner'):
    """Function
    From nonzero fci coefficients, form quantum state and evaluate FCI energy
    as an expectation value of Hamiltonian using Qulacs.observable.

    Args:
        norbs (int): Number of orbitals
        nalpha (int): Number of alpha electrons
        nbeta (int): Number of beta electrons
        coeff ([float]): FCI trial coefficients that are nonzero,
                         specified by NonzeroList.
        NonzeroList ([bool]): A bool array of [NDetA, NDetB] where NDetA and NDetB
                              are numbers of alpha and beta determinants. If [ia, ib]
                              element is True, FCI coefficient is nonzero. Otherwise,
                              FCI coefficient is supposed to be zero for perhaps symmetry
                              reasons, and we always fix this coefficient to be zero.
        Hamiltonian (OpenFermion.QubitOperator): Hamiltonian in Qubit basis.

    Returns:
        Efci (float): FCI Energy
    """
    from quket.opelib import evolve
    fci_state = create_fci(norbs, nalpha, nbeta, coeff, NonzeroList, init_state=None, mapping=mapping)


    H_state = evolve(Hamiltonian,  fci_state, parallel=True)
    Efci = inner_product(fci_state, H_state).real

    #Efci =  Hamiltonian.get_expectation_value(fci_state)
    for istate in range(len(lower_states)):
        overlap = inner_product(lower_states[istate]['state'], fci_state)
        Efci += -Efci * abs(overlap)**2
        
    return Efci

def fci2qubit(Quket, nroots=None, threshold=1e-8, shift=1, verbose=False):
    """Function
    Perform mapping from fci coefficients to qubit representation (Jordan-Wigner).
    This is simply VQE with FCI ansatz.

    Args:
        Quket (QuketData): QuketData instance
        nroots (int): Number of FCI states 
        threshold (float): Threshold for FCI coefficients to be included
        shift (float): shift for penalty function, shift * (S^2 - s(s+1))
    """
    if not hasattr(Quket, "fci_coeff"):
        if nroots is None:
            nroots = Quket.cf.nroots
        prints("Running OpenFemrion's QubitDavidson.")
        return exact_diagonalization_openfermion(Quket, nroots)
        
    if Quket.fci_coeff is None:
        if nroots is None:
            nroots = Quket.cf.nroots
        prints('No fci_coeff done by PySCF.')
        prints("Running OpenFemrion's QubitDavidson.")
        return exact_diagonalization_openfermion(Quket, nroots)
    if Quket.spin != Quket.multiplicity:
        prints("\nWARNING: PySCF uses high-spin FCI but low-spin FCI is not possible.\n",
               "         You may be able to get such states by changing in mod.py,\n",
               "            def run_pyscf_mod(guess, n_active_orbitals, n_active_electrons, molecule,...\n",
               "                ...\n",
               "                pyscf_casci.fix_spin(shift=1.0, ss=(spin-1)*(spin+1)/4)\n",
               "         to an appropriate shift and ss values.\n",
               "         Setting  shift <= 0  and  nroots > 1  may catch some low-spin states if you are lucky.\n\n") 
        return None
    nalpha = Quket.noa
    nbeta = Quket.nob
    norbs = Quket.n_active_orbitals
    n_qubits = Quket._n_qubits
    lower_states = Quket.lower_states.copy()
    ### Turn off projector
    SpinProj = Quket.projection.SpinProj
    NumberProj = Quket.projection.NumberProj
    Quket.projection.SpinProj = False
    Quket.projection.NumberProj = False
    #n_qubits_sym = Quket.n_qubits_sym
    spin = Quket.spin - 1
    from copy import deepcopy
    shift = 1
    Hamiltonian = deepcopy(Quket.operators.qubit_Hamiltonian)
    Hamiltonian += shift * (Quket.operators.qubit_S2 - QubitOperator('',spin*(spin+1)) )
    NDetA = Binomial(norbs, nalpha)
    NDetB = Binomial(norbs, nbeta)
    listA =  list(combinations(range(norbs), nalpha))
    listB =  list(combinations(range(norbs), nbeta))
    for isort in range(nalpha):
        listA = sorted(listA, key=itemgetter(isort))
    for isort in range(nbeta):
        listB = sorted(listB, key=itemgetter(isort))

    if type(Quket.fci_coeff) is list:
        nstates = len(Quket.fci_coeff)
    else:
        nstates = 1
    if nroots is not None and type(nroots) is int:
        nstates = min(nstates,nroots)
        

    istate = 0
    fci_states = []
    while istate < nstates: 

        NonzeroList = np.zeros((NDetA, NDetB), dtype=bool)

        if type(Quket.fci_coeff) is list:
            ### includes excited states
            fci_coeff = Quket.fci_coeff[istate]
        else: 
            fci_coeff = Quket.fci_coeff

        num = 0
        coeff = []
        for ib in range(NDetB):
            for ia in range(NDetA):
                if abs(fci_coeff[ia, ib]) > 1e-8:
                    NonzeroList[ia, ib] = True
                    num += 1
                    coeff.append(fci_coeff[ia, ib])

        opt_options = {"disp": verbose,
                       "maxiter": 1000,
                       "maxfun": Quket.cf.maxfun,
                       "gtol":1e-6,
                       "ftol":1e-13}
        cost_wrap = lambda coeff: \
                cost_fci(norbs,
                         nalpha,
                         nbeta,
                         coeff,
                         NonzeroList,
                         Hamiltonian,
                         lower_states=fci_states,
                         mapping=Quket.cf.mapping
                         )
        cost_wrap_mpi = lambda coeff: cost_mpi(cost_wrap, coeff)
        create_state = lambda coeff, init_state: create_fci(norbs, nalpha, nbeta, coeff, NonzeroList, init_state=None, mapping=Quket.cf.mapping)
        jac_wrap_mpi = lambda coeff: jac_mpi_deriv(create_state, Quket, coeff, Hamiltonian=Hamiltonian)

        opt = sp.optimize.minimize(cost_wrap_mpi, coeff,
                       jac=jac_wrap_mpi, method="L-BFGS-B", options=opt_options)
        coeff = opt.x
        vec = np.zeros(2**n_qubits)

        opt = f"0{n_qubits}b"
        i = 0
        for ib in range(NDetB):
            occB = np.array([n*2 + 1 for n in listB[ib]])
            for ia in range(NDetA):
                occA = np.array([n*2 for n in listA[ia]])
                if NonzeroList[ia, ib]:
                    k = np.sum(2**occA) + np.sum(2**occB)
                    vec[k] = coeff[i]
                    if abs(coeff[i]) > threshold and cf.debug:
                        prints(f"    Det# {i}: ".format(i), end="");
                        prints(f"| {format(k, opt)} >: {coeff[i]}  ")
                    i += 1
        fci_state = QuantumState(n_qubits)
        fci_state.load(vec)
        if Quket.cf.mapping == "bravyi_kitaev":
            fci_state = transform_state_jw2bk(fci_state)
        norm2 = fci_state.get_squared_norm()
        fci_state.normalize(norm2)
        E = Quket.get_E(fci_state)
        fci_dict = {'energy':E, 'state':fci_state, 'theta_list':[], 'det':0} 
        fci_states.append(fci_dict) 
        Quket.lower_states.append(fci_dict)
        istate += 1
    
    # Retrieve projection flags and lower_states
    Quket.projection.SpinProj = SpinProj
    Quket.projection.NumberProj = NumberProj
    Quket.lower_states = lower_states
    return fci_states

def create_fci(norbs, nalpha, nbeta, coeff, NonzeroList, init_state=None, mapping='jordan_wigner'):
    """Function
    From nonzero fci coefficients, form FCI quantum state.

    Args:
        norbs (int): Number of orbitals
        nalpha (int): Number of alpha electrons
        nbeta (int): Number of beta electrons
        coeff ([float]): FCI trial coefficients that are nonzero,
                         specified by NonzeroList.
        NonzeroList ([bool]): A bool array of [NDetA, NDetB] where NDetA and NDetB
                              are numbers of alpha and beta determinants. If [ia, ib]
                              element is True, FCI coefficient is nonzero. Otherwise,
                              FCI coefficient is supposed to be zero for perhaps symmetry
                              reasons, and we always fix this coefficient to be zero.
        init_state : Dummy argument, as FCI is always generated by a state-vector but not a quantum circuit.

    Returns:
        fci_state (QuantumState): FCI state
    """
    NDetA = Binomial(norbs, nalpha)
    NDetB = Binomial(norbs, nbeta)
    listA =  list(combinations(range(norbs), nalpha))
    listB =  list(combinations(range(norbs), nbeta))
    for isort in range(nalpha):
        listA = sorted(listA, key=itemgetter(isort))
    for isort in range(nbeta):
        listB = sorted(listB, key=itemgetter(isort))

    n_qubits = norbs*2
    vec = np.zeros(2**n_qubits)
    i = 0
    for ib in range(NDetB):
        occB = np.array([n*2 + 1 for n in listB[ib]])
        for ia in range(NDetA):
            occA = np.array([n*2 for n in listA[ia]])
            if NonzeroList[ia, ib]:
                k = np.sum(2**occA) + np.sum(2**occB)
                if mapping == "bravyi_kitaev":
                    k = jw2bk(k, n_qubits)
                vec[k] = coeff[i]
                i += 1

    fci_state = QuantumState(n_qubits)
    fci_state.load(vec)
    norm2 = fci_state.get_squared_norm()
    fci_state.normalize(norm2)
        
    return fci_state


def exact_diagonalization_openfermion(Quket, nroots):
    from openfermion.linalg import eigenspectrum, QubitDavidson
    # Setting up QubitDavidson
    Quket.operators.qubit_Hamiltonian.compress()
    qubit_eigen = QubitDavidson(Quket.operators.qubit_Hamiltonian)
    n_qubits = Quket.n_qubits
    results = qubit_eigen.get_lowest_n(nroots)

    # Tranlating the exact eigenstates of OpenFermion to qulacs QunatumState class
    fci_states = []
    for istate in range(nroots):
        ind_max = np.argmax([abs(results[2][k][istate]) for k in range(2**n_qubits)])
        vmax = results[2][ind_max][istate]
        phase = np.exp(-1j * np.arctan(vmax.imag/vmax.real))
        fci_wfn  = QuantumState(n_qubits)
        fci_wfn.multiply_coef(0)
        det  = QuantumState(n_qubits)
        for i_openfermion in range(2**n_qubits):
            coef = results[2][i_openfermion][istate] * phase
        
            ###    OpenFermion: i_openfermion =  k0 k1 k2 k3 ... k_{nqubit-1} 
            ###    Qulacs     : i_qulacs      =  k_{nqubit-1} ... k3 k2 k1 k0
            i_qulacs = 0
            for k in range (n_qubits):
                kk = 1 << k
                if kk & i_openfermion > 0:
                    i_qulacs = i_qulacs | 1 << n_qubits-k-1        
            det.set_computational_basis(i_qulacs)
                    
            det.multiply_coef(coef)
            fci_wfn.add_state(det)
        fci_dict = {'energy':results[1][istate], 'state':fci_wfn, 'theta_list':[], 'det':0} 
        fci_states.append(fci_dict)
    return fci_states
