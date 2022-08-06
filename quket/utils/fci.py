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


"""

from operator import itemgetter
from itertools import combinations

import numpy as np
import scipy as sp
from qulacs import QuantumCircuit
from qulacs.state import inner_product

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import prints, tstamp
from quket.fileio import print_state, printmat
from quket.linalg import Binomial
from quket.lib import QuantumState, QubitOperator
from .utils import jw2bk, bk2jw, transform_state_jw2bk
from .deriv import cost_mpi, jac_mpi_ana

### DEPRECATED ###
'''
FCI vqe solver using pyscf fci coefficients.
(This is an inefficient but straightfoward way of mapping fci coefficients onto the qubit representation)
'''
def cost_fci(norbs, nalpha, nbeta, coeff, NonzeroList, Hamiltonian, lower_states=None, mapping='jordan_wigner'):
    ### DEPRECATED ###
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

def _fci2qubit(Quket, nroots=None, threshold=1e-8, shift=1, verbose=False):
    ### DEPRECATED ###
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
        jac_wrap_mpi = lambda coeff: jac_mpi_ana(create_state, Quket, coeff, Hamiltonian=Hamiltonian)

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
    ### DEPRECATED ###
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



'''
###
###  For Davidson by Quket
###
'''
##def fci2qubit(Quket, nroots=None, threshold=1e-5, shift=1, verbose=False):
##    """Function
##    Perform Davidson diagonalization of qubit Hamiltonian.
##    
##    Args:
##        Quket (QuketData): QuketData instance, which includes qubit_Hamiltonian
##        nroots (int): Number of FCI states to be computed. If None, use Quket.nroots. 
##        threshold (float): Threshold for convergence (norm of residual)
##        shift (float): shift for spin using a penalty function, + shift * (S^2 - s(s+1))
##        verbose (bool): Detailed print if True
##
##    Returns:
##        fci_states (list): A list of FCI dictionary of each state
##
##    Author(s): Takashi Tsuchimochi
##    """
##    from quket.opelib import evolve
##    from quket.linalg.linalg import symm
##    if nroots is None:
##        nroots = Quket.nroots
##
##    n_qubits = Quket.n_qubits
##    maxiter = 100
##    from copy import deepcopy
##    qubit_Hamiltonian = deepcopy(Quket.operators.qubit_Hamiltonian)
##
##    # Shift Hamiltonian by spin penalty
##    qubit_Hamiltonian  += shift * (Quket.operators.qubit_S2 - (Quket.spin-1)*(Quket.spin+1)/4 )
##    if Quket.model in ('chemical', 'hubbard'):
##        det_list = get_chemical_det_list(Quket)
##    else:
##        ### What is the best initial guess for heisenberg?
##        det_list = [x for x in range(nroots)]
##    Hdiag_list = get_Hdiag_list(qubit_Hamiltonian, det_list)
##
##    if verbose or cf.debug:
##        tstamp('Entered fci2qubit')
##        prints(f'Target spin s = {Quket.spin}')
##        prints('Cycle  State       Energy      Norm')
##
##    ### Get lowest nroots states according to the diagonals
##    from heapq import nsmallest
##    result = [(value, k) for k, value in enumerate(Hdiag_list)]
##    min_result = nsmallest(nroots, result)
##    
##    initial_dets = []
##    for k in range(nroots):
##        initial_dets.append(det_list[min_result[k][1]])
##    
##    states = []
##    for k in range(nroots):
##        states.append(QuantumState(n_qubits, initial_dets[k]))
##    
##    Hsub = np.zeros(0)
##    norms = np.zeros(nroots)
##    converge = [False for x in range(nroots)]
##    Hstates = []
##    icyc = 0
##    fci_vec = np.zeros((nroots, 2**n_qubits), complex)
##    Hfci_vec = np.zeros((nroots, 2**n_qubits), complex)
##    new_state = np.zeros(2**n_qubits, complex)
##    ioff = 0
##    ntargets = nroots
##    while icyc < maxiter:
##        for k in range(ioff, ioff+ntargets):
##            Hstates.append(evolve(qubit_Hamiltonian, states[k], parallel=True))
##        
##        ### Subspace
##        k = 0
##        for i in range(ioff, ioff+ntargets):
##            k += 1
##            for j in range(0, ioff + k):
##                Hij = inner_product(states[j], Hstates[i]).real
##                Hsub = np.append(Hsub, Hij)
##                
##        Hsub_symm = symm(Hsub)
##        E, V = np.linalg.eigh(Hsub_symm)
##
##        norm = 0
##        reset = False 
##        for i in range(nroots):
##            if converge[i]:
##                continue
##            fci_vec[i] *= 0
##            Hfci_vec[i] *= 0
##            for j in range(V.shape[0]):
##                fci_vec[i] += states[j].get_vector() * V[j, i]
##                Hfci_vec[i] += Hstates[j].get_vector() * V[j, i] 
##            residual = Hfci_vec[i] - E[i] * fci_vec[i]
##             
##            norms[i] = np.linalg.norm(residual)
##            if norms[i] < threshold:
##                converge[i] = True
##            else:
##                new_state *= 0 
##                for k, det in enumerate(det_list):
##                    if abs(Hdiag_list[k] - E[i]) > 1e-12:
##                        new_state[det] = - residual[det] / (Hdiag_list[k] - E[i])
##                    else:
##                        new_state[det] = - residual[det] / 1e5
##                        
##                # Gram-Schmidt orthogonalization
##                state = QuantumState(n_qubits)
##                state.load(new_state)
##                norm2 = state.get_squared_norm()
##                state.normalize(norm2)
##                if np.sqrt(norm2) < threshold:
##                    reset = True
##                for old_state in states:
##                    state -= old_state * inner_product(state, old_state)
##                    norm2 = state.get_squared_norm()
##                    state.normalize(norm2)
##                    if np.sqrt(norm2) < threshold:
##                        reset = True
##                norm2 = state.get_squared_norm()
##                if norm2 < threshold:
##                    reset = True
##                state.normalize(norm2)
##                states.append(state)
##
##        if verbose or cf.debug:
##            prints(f'[{icyc:2d}]      0:  {E[0]:.10f}   {norms[0]:.2e}  ', end='')
##            if converge[0]:
##                prints('converged')
##            else:
##                prints('')
##            for k in range(1, nroots):
##                prints(f'          {k}:  {E[k]:.10f}   {norms[k]:.2e}  ', end='')
##                if converge[k]:
##                    prints('converged')
##                else:
##                    prints('')
##        prints('')
##        if all (converge):
##            break
##
##        ntargets = converge.count(False) 
##        ioff += ntargets
##        if reset:
##            # Round-off error : reset the cycle
##            if verbose or cf.debug:
##                prints('*** Small norm detected in residuals.')
##                prints('*** Reset subspace.')
##            ioff = nroots
##            Hsub = np.zeros(nroots*(nroots+1)//2)
##            states = []
##            Hstates = []
##            for i in range(nroots):
##                state = QuantumState(n_qubits)
##                state.load(fci_vec[i])
##                Hstate = QuantumState(n_qubits)
##                Hstate.load(Hfci_vec[i])
##                states.append(state)
##                Hstates.append (Hstate)
##                Hsub[(i+1)*(i+2)//2 - 1] = E[i]
##            
##            for i in range(nroots):
##                if converge[i]:
##                    continue
##                residual = Hfci_vec[i] - E[i] * fci_vec[i]
##                new_state *= 0 
##                for k, det in enumerate(det_list):
##                    if abs(Hdiag_list[k] - E[i]) > 1e-12:
##                        new_state[det] = - residual[det] / (Hdiag_list[k] - E[i])
##                # Gram-Schmidt orthogonalization
##                state = QuantumState(n_qubits)
##                state.load(new_state)
##                norm2 = state.get_squared_norm()
##                state.normalize(norm2)
##                for old_state in states:
##                    state -= old_state * inner_product(state, old_state)
##                    norm2 = state.get_squared_norm()
##                    state.normalize(norm2)
##                states.append(state)
##        icyc += 1
##    
##    
##    ### Create Quantum States
##    fci_states = []
##    for k in range(nroots):
##        fci_state = QuantumState(n_qubits)
##        from quket import printmat
##        fci_state.load(fci_vec[k])
##        fci_dict = {'energy':E[k], 'state':fci_state, 'theta_list':[], 'det':0} 
##        fci_states.append(fci_dict)
##    if verbose or cf.debug:
##        prints('Davidson done.')
##        tstamp('Leaving fci2qubit')
##    return fci_states


def get_chemical_det_list(Quket):
    """
    From the number of electrons (spins), compute the list of full CI determinants, 
    encoded to bit strings. 
    If symmetry information is available, the list contains only symmetry-allowed determinants.

    Args:
        Quket : QuketData instance.
    Returns:
        det_list : A list of FCI determinants as bit integers

    Author(s): Takashi Tsuchimochi
    """

    ###  Form the list of number- and Sz-preserving determinants as bit-strings
    
    from itertools import combinations
    from quket.linalg import Binomial
    from operator import itemgetter
    
    mapping = Quket.cf.mapping
    n_qubits = Quket._n_qubits
    tapered = Quket.tapered["operators"]
    
    norbs = Quket.n_active_orbitals
    nalpha = Quket.noa
    nbeta = Quket.nob
    NDetA = Binomial(norbs, nalpha)
    NDetB = Binomial(norbs, nbeta)
    listA =  list(combinations(range(norbs), nalpha))
    listB =  list(combinations(range(norbs), nbeta))
    for isort in range(nalpha):
        listA = sorted(listA, key=itemgetter(isort))
    for isort in range(nbeta):
        listB = sorted(listB, key=itemgetter(isort))
    
    det_list = []
    for ib in range(NDetB):
        occB = np.array([n*2 + 1 for n in listB[ib]])
        for ia in range(NDetA):
            occA = np.array([n*2 for n in listA[ia]])
            k = np.sum(2**occA) + np.sum(2**occB)
            if mapping == "bravyi_kitaev":
                k = jw2bk(k, n_qubits)
            if tapered:
                if Quket.tapering.check_symmetry(k):
                    k = Quket.tapering.transform_bit(k)[0]
                    det_list.append(k)
            else:
                det_list.append(k)
    return det_list


def get_Hdiag_list(qubit_Hamiltonian, det_list):
    """
    Given a qubit Hamiltonian and determiant (computational basis) lists,
    compute the expectation values.
    
    Args:
        qubit_Hamiltonian : Hamiltonian as QubitOperator
        det_list : A list of determinants for which the expectation values are computed
        
    Author(s): Takashi Tsuchimochi
    """
    from .bit import pauli_bit_multi
    qubit_Hamiltonian.compress()
    diag_list = []
    for det in det_list:
        diag = 0
        for pauli, coef in qubit_Hamiltonian.terms.items():
            det_, phase = pauli_bit_multi(pauli, det)
            if det_ == det:
                diag += coef * phase.real
        diag_list.append(diag)
    return diag_list 

def fci2qubit(Quket, nroots=None, threshold=1e-5, shift=1, verbose=False, maxiter=100):
    """Function
    Obtain the exact eigen states of the system given in Quket, by Davidson-diagonalizing qubit_Hamiltonian.
    
    Args:
        Quket (QuketData): QuketData instance, which includes qubit_Hamiltonian
        nroots (int): Number of FCI states to be computed. If None, use Quket.nroots. 
        threshold (float): Threshold for convergence (norm of residual)
        shift (float): shift for spin using a penalty function, + shift * (S^2 - s(s+1))
        verbose (bool): Detailed print if True

    Returns:
        fci_states (list): A list of FCI dictionary of each state

    Author(s): Takashi Tsuchimochi
    """
    from copy import deepcopy
    if nroots is None:
        nroots = Quket.nroots

    qubit_Hamiltonian = deepcopy(Quket.operators.qubit_Hamiltonian)
    if Quket.model in ('chemical', 'hubbard'):
        det_list = get_chemical_det_list(Quket)
        # Shift Hamiltonian by spin penalty
        qubit_Hamiltonian  += shift * (Quket.operators.qubit_S2 - (Quket.spin-1)*(Quket.spin+1)/4 )
    else:
        det_list = None

    if verbose or cf.debug:
        tstamp('Entered fci2qubit')
        prints(f'Target spin s = {Quket.spin}')

    return davidson(qubit_Hamiltonian, 
                    nroots=nroots, 
                    initial_states=None, 
                    det_list=det_list, 
                    threshold=threshold,
                    n_qubits=Quket.n_qubits,
                    maxiter=maxiter,
                    verbose=verbose)



def davidson(qubit_Hamiltonian, nroots=1, initial_states=None, det_list=None, threshold=1e-5, n_qubits=None, maxiter=100, verbose=False):
    """
    Perform Davidson diagonalization of qubit_Hamiltonian.

    Args:
        qubit_Hamiltonian (QubitOperator): Hamiltonian as a linear combination of Pauli operators
        nroots (int): Number of FCI states to be computed. If None, use Quket.nroots. 
        initial_states (list, optional): Initial guess states as a list of orhonormal QuantumStates 
        det_list (list, optional): A list that has determinants (bit strings) as integers that are considered in the update of Davidson procedure. That is, these integers correspond to a certain symmetry (number, Sz). Note this is crucial to speed up the calculation, but missing integers that are required to span the symmetry space will cause a failure of calculation. 
        threshold (float): Threshold for convergence (norm of residual)
        n_qubits (int, optional): Number of qubits
        maxiter (int, optional): Maximum number of iterations
        verbose (bool): Detailed print if True

    Returns:

    Author(s): Takashi Tsuchimochi
    """
    from quket.opelib import evolve
    from quket.linalg.linalg import symm

    if n_qubits is None:
        ### Estimate from Hamiltonian
        from quket.tapering.tapering import Hamiltonian2n_qubits
        n_qubits = Hamiltonian2n_qubits(qubit_Hamiltonian)

    if det_list is None:
        ### det_list includes all bit strings
        det_list = [x for x in range(2**n_qubits)]

    Hdiag_list = get_Hdiag_list(qubit_Hamiltonian, det_list)
    if cf.debug or verbose:
        prints('Diagonal matrix elements prepared.')

    if initial_states is None:
        ### Find the nroots lowest diagonals
        ### Get lowest nroots states according to the diagonals
        from heapq import nsmallest
        result = [(value, k) for k, value in enumerate(Hdiag_list)]
        min_result = nsmallest(nroots, result)
        initial_dets = []
        for k in range(nroots):
            initial_dets.append(det_list[min_result[k][1]])
     
        states = []
        for k in range(nroots):
            states.append(QuantumState(n_qubits, initial_dets[k]))
    else:
        states = initial_states

    if verbose or cf.debug:
        prints('Cycle  State       Energy      Norm')

    Hsub = np.zeros(0)
    norms = np.zeros(nroots)
    converge = [False for x in range(nroots)]
    Hstates = []
    icyc = 0
    fci_vec = np.zeros((nroots, 2**n_qubits), complex)
    Hfci_vec = np.zeros((nroots, 2**n_qubits), complex)
    new_state = np.zeros(2**n_qubits, complex)
    ioff = 0
    ntargets = nroots
    while icyc < maxiter:
        ### Subspace Hamiltonian
        ntargets = len(states) - len(Hstates) 
        for i in range(ioff, ioff+ntargets):
            Hstates.append(evolve(qubit_Hamiltonian, states[i], parallel=True))
            for j in range(i+1):
                Hij = inner_product(states[j], Hstates[i]).real
                Hsub = np.append(Hsub, Hij)
                
        Hsub_symm = symm(Hsub)
        E, V = np.linalg.eigh(Hsub_symm)

        reset = False 
        for i in range(nroots):
            fci_vec[i] *= 0
            Hfci_vec[i] *= 0
            for j in range(V.shape[0]):
                fci_vec[i] += states[j].get_vector() * V[j, i]
                Hfci_vec[i] += Hstates[j].get_vector() * V[j, i] 
            residual = Hfci_vec[i] - E[i] * fci_vec[i]
             
            norms[i] = np.linalg.norm(residual)
            if norms[i] < threshold:
                converge[i] = True
            else:
                converge[i] = False
                new_state *= 0 
                for k, det in enumerate(det_list):
                    if abs(Hdiag_list[k] - E[i]) > 1e-14:
                        new_state[det] = - residual[det] / (Hdiag_list[k] - E[i])
                    else:
                        new_state[det] = - residual[det] / 1e5
                        
                # Gram-Schmidt orthogonalization
                state = QuantumState(n_qubits)
                state.load(new_state)
                norm2 = state.get_squared_norm()
                state.normalize(norm2)
                if np.sqrt(norm2) < 1e-6:
                    reset = True
                for old_state in states:
                    state -= old_state * inner_product(state, old_state)
                    norm2 = state.get_squared_norm()
                    state.normalize(norm2)
                    if np.sqrt(norm2) < 1e-6:
                        ### This means the new vector is spanned by other vectors in the subspace. 
                        ### Skip this state.
                        break
                else:
                    states.append(state)

        if verbose or cf.debug:
            prints(f'[{icyc:2d}]      0:  {E[0]:+.10f}   {norms[0]:.2e}  ', end='')
            if converge[0]:
                prints('converged')
            else:
                prints('')
            for k in range(1, nroots):
                prints(f'          {k}:  {E[k]:+.10f}   {norms[k]:.2e}  ', end='')
                if converge[k]:
                    prints('converged')
                else:
                    prints('')
        if all (converge):
            break

        ioff += ntargets
        if reset:
            # Round-off error : reset the cycle
            if verbose or cf.debug:
                prints('*** Small norm detected in residuals.')
                prints('*** Reset subspace.')
            ioff = nroots
            Hsub = np.zeros(nroots*(nroots+1)//2)
            del states, state, Hstates
            states = []
            Hstates = []
            for i in range(nroots):
                state = QuantumState(n_qubits)
                state.load(fci_vec[i])
                Hstate = QuantumState(n_qubits)
                Hstate.load(Hfci_vec[i])
                states.append(state)
                Hstates.append (Hstate)
                Hsub[(i+1)*(i+2)//2 - 1] = E[i]
            
            for i in range(nroots):
                if converge[i]:
                    continue
                residual = Hfci_vec[i] - E[i] * fci_vec[i]
                new_state *= 0 
                for k, det in enumerate(det_list):
                    if abs(Hdiag_list[k] - E[i]) > 1e-12:
                        new_state[det] = - residual[det] / (Hdiag_list[k] - E[i])
                    else:
                        new_state[det] = - residual[det] / 1e5
                # Gram-Schmidt orthogonalization
                state = QuantumState(n_qubits)
                state.load(new_state)
                norm2 = state.get_squared_norm()
                state.normalize(norm2)
                for old_state in states:
                    state -= old_state * inner_product(state, old_state)
                    norm2 = state.get_squared_norm()
                    if np.sqrt(norm2) < 1e-6:
                        ### This means the new vector is spanned by other vectors in the subspace. 
                        ### Skip this state.
                        break
                    state.normalize(norm2)
                else:
                    states.append(state)
        icyc += 1
    
    
    ### Create Quantum States
    fci_states = []
    for k in range(nroots):
        fci_state = QuantumState(n_qubits)
        from quket import printmat
        fci_state.load(fci_vec[k])
        fci_dict = {'energy':E[k], 'state':fci_state, 'theta_list':[], 'det':0} 
        fci_states.append(fci_dict)
    if verbose or cf.debug:
        prints('Davidson done.')
        tstamp('Leaving fci2qubit')
    return fci_states


def get_chemical_det_list(Quket):
    """
    From the number of electrons (spins), compute the list of full CI determinants, 
    encoded to bit strings. 
    If symmetry information is available, the list contains only symmetry-allowed determinants.

    Args:
        Quket : QuketData instance.
    Returns:
        det_list : A list of FCI determinants as bit integers

    Author(s): Takashi Tsuchimochi
    """

    ###  Form the list of number- and Sz-preserving determinants as bit-strings
    
    from itertools import combinations
    from quket.linalg import Binomial
    from operator import itemgetter
    
    mapping = Quket.cf.mapping
    n_qubits = Quket._n_qubits
    tapered = Quket.tapered["operators"]
    
    norbs = Quket.n_active_orbitals
    nalpha = Quket.noa
    nbeta = Quket.nob
    NDetA = Binomial(norbs, nalpha)
    NDetB = Binomial(norbs, nbeta)
    listA =  list(combinations(range(norbs), nalpha))
    listB =  list(combinations(range(norbs), nbeta))
    for isort in range(nalpha):
        listA = sorted(listA, key=itemgetter(isort))
    for isort in range(nbeta):
        listB = sorted(listB, key=itemgetter(isort))
    
    det_list = []
    for ib in range(NDetB):
        occB = np.array([n*2 + 1 for n in listB[ib]])
        for ia in range(NDetA):
            occA = np.array([n*2 for n in listA[ia]])
            k = np.sum(2**occA) + np.sum(2**occB)
            if mapping == "bravyi_kitaev":
                k = jw2bk(k, n_qubits)
            if tapered:
                if Quket.tapering.check_symmetry(k):
                    k = Quket.tapering.transform_bit(k)[0]
                    det_list.append(k)
            else:
                det_list.append(k)
    return det_list


def get_Hdiag_list(qubit_Hamiltonian, det_list):
    """
    Given a qubit Hamiltonian and determiant (computational basis) lists,
    compute the expectation values.
    
    Args:
        qubit_Hamiltonian : Hamiltonian as QubitOperator
        det_list : A list of determinants for which the expectation values are computed
        
    Author(s): Takashi Tsuchimochi
    """
    from .bit import pauli_bit_multi
    qubit_Hamiltonian.compress()
    diag_list = []
    for det in det_list:
        diag = 0
        for pauli, coef in qubit_Hamiltonian.terms.items():
            det_, phase = pauli_bit_multi(pauli, det)
            if det_ == det:
                diag += coef * phase.real
        diag_list.append(diag)
    return diag_list 
