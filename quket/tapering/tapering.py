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

tapering.py

Functions transforming Hamiltonian such that
certain qubits are acted trivially (I or X).

"""

from operator import itemgetter
from itertools import compress, zip_longest, filterfalse, starmap
from copy import deepcopy
from math import log2, ceil, floor
from statistics import stdev, mean, median
from collections.abc import Iterable
from collections import deque
import time


import numpy as np
np.set_printoptions(precision=15, suppress=True)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=2000)


from quket.mpilib import mpilib as mpi
from quket import config as cf
from quket.utils import jw2bk
from quket.opelib.circuit import Pauli2Circuit
from quket.fileio import prints, tstamp
from quket.lib import QubitOperator, commutator

class Z2tapering():
    '''Class
    Transform Hamiltonian based on the tapering-off technique 
    (arXiv:1701.08213; arXiv:1910.14644).

    Author(s): Takashi Tsuchimochi, TsangSiuChung
    '''
    
    def __init__(self, H_old, n_qubits=None, det=None, PGS=None, Sz_symmetry=True):
        '''
        Args:
            Hamiltonian (QubitOperator): Hamiltonian.
            n_qubits (int): Number of qubits required for the Hamiltonian. Optional.
            det (int): Initial state (slater determinant) of the quantum object.
            PGS (tuple): Information that used to enhance tapering off by Point Group Symmetry.

        Returns:
            None
        '''
        self.initialized = 0
        self.stage = 0
        self.H_old = None
        self.n_qubits = None
        self.det = None
        self.symm_op_list = None
        self.irrep_list = None
        self.character_list = None
        self.E_matrix = None
        self.E_kernel = None
        self.g_list = None
        self.tau_list = None
        self.commutative_taus = None
        self.redundant_bits = None
        self.commutative_sigmas = None
        self.clifford_operators = None
        self.H_new = None
        self.validity = None
        self.X_eigvals = None
        self.H_renew = None

        self.H_old = H_old
        if n_qubits is None:
            self.n_qubits = Hamiltonian2n_qubits(self.H_old)
        else:
            self.n_qubits = n_qubits
        if det is not None:
            if type(det) is list:
                self.det = det[0][1]
            elif type(det) is int:
                ### For QUKET ###
                self.det = det
                #################
        else:
            raise ValueError("Please input reference determinant. It is required.")

        if PGS is not None:
            # Check for integrity
            for x in PGS:
                if x is None or len(x)==0:
                    ErorrMessage = 'Empty list is contained in PGS. \
                                    This would introduce bug. Please go check for it.'
                    raise ValueError(ErorrMessage)
            # Pass value to class variables if integrity is ensurred        
            self.symm_op_list, self.irrep_list, self.character_list = PGS

        self.Sz_symmetry = Sz_symmetry

    def __str__(self):
        
        if self.initialized == 0:
            prints(f"You have to do .run() first!")
            
        elif self.initialized == 1:
            if cf.debug:
                # Print PGS information
                if self.character_list is not None:
                    self.print_pgs()      
                    
                if self.is_available:
                        if cf.debug:
                            prints(f"Tapering success! New Hamiltoninan verified.")
                            if len(self.H_old.terms) != len(self.H_new.terms):
                                prints(f"Abnormal terms removed from the Hamiltonian.")
                                prints(
                                    f"Terms in Hamiltonian: {len(self.H_old.terms)} --> {len(self.H_new.terms)}", end='')
                                prints(
                                    f"  ({len(self.H_new.terms)-len(self.H_old.terms)})\n")
                            prints(f"Replaced redundent qubits by their eigenvalues.")
                            prints(
                                f"Terms in Hamiltonian: {len(self.H_new.terms)} --> {len(self.H_renew.terms)}", end='')
                            prints(
                                f"  ({len(self.H_renew.terms)-len(self.H_new.terms)})\n")

                else:
                    prints(f"Tapering off failure.\n")
                    prints(f"Guessing purify_hamiltonian() is malfunctioning.")
                    if not cf.debug:
                        prints(f"Toggle debug to check the details of error terms if needed")
                    else:
                        prints(f"Error terms in Hamiltonian:")
                        for x in self.validity:
                            prints(x)
                        prints("\n")
            
            # Print Tapering-off result
            self.print_result(debug=cf.debug)

            if self.redundant_bits is not None:
                for bit,tau in zip(self.redundant_bits, self.commutative_taus):
                    prints(f"Qubit: {bit}    Tau: {tau}")

        return "" 

    @property    
    def is_available(self):
        a = self.stage == 9
        b = len(self.validity) == 0
        c = self.H_renew is not None
        d = self.initialized == 1
        return a&b&c&d
        
        
    def print_pgs(self):
        title = "Point Group Symmetry"
        tablemaker(title, '', 'SymmOP', 'IRREP',
                   self.symm_op_list, self.irrep_list, self.character_list,
                   spacing=4, vectype='int', tolence=2)

        
    def print_result(self, debug=False):
        prints(f"Tapering-Off Results:")
        #prints(f"{len(self.redundant_bits)} Qubits ", end='')
        #prints(f"{'is' if len(self.redundant_bits) <2 else 'are'}", end='')
        #prints(f" eligible to be tapered off.")
        
        if debug:
            prints(f"qubit    coeff    commutativity    tau")
            for q, w, e, r in zip(self.redundant_bits,
                                  self.X_eigvals,
                                  filter(lambda x: x[0], self.tau_info),
                                  self.commutative_taus):
                prints(f"{q:>5} {w:>8}            {e[1]}    {r}")
            prints(f"\nTapering-off finished in {self.timelog} s\n")

        else:
            prints("List of Tapered-off Qubits: ",self.redundant_bits)

            
    def print_error(self):
        error_at = ['initialized', "You haven't done the .run()",
                    ('H_old', 'Check if the input value Hamiltonian is inserted properly.'),
                    ('character_list', 'Either rref(), get_parity_matrix_E(), binary_kernel() or include_pgss() is bugged,\n\or the Hamiltonin is not eligible to tapering off.\n'),
                    ('E_kernel', 'Either rref(), get_parity_matrix_E(), binary_kernel() or include_pgss() is bugged,\nor the Hamiltonin is not eligible to tapering off.\n'),
                    ('commutative_taus', 'Either get_commutative_tau() is bugged,\nor there is no available option for tapering off in this case.\n'),
                    ('redundant_bits', 'Either get_commutative_sigma() is bugged,\nor commutative_taus() is bugged.\n'),
                    ('clifford_operators',
                     'Either get_clifford_operator() is bugged,\nor get_commutative_sigma() is bugged.\n'),
                    ('H_new', 'Bug occured in purify_hamiltonian().\n'),
                    ('validity', 'Bug occured in judge_reduced_hamiltonian().\nThe new Hamiltonian is still contains Y or Z terms in redundant bits.\n')]
        error_part = error_at[self.stage][0]
        error_msg = error_at[self.stage][1]
        prints(f"tapering.{error_part} is empty.\n{error_msg}")

        
    def print_debug(self):
        
        def print_list(L):
            [prints(x) for x in L] 
            
        prints(f"\n#######################\n   TAPERING.PY DEBUG\nvvvvvvvvvvvvvvvvvvvvvvv\n")
        prints(f"is_available = {self.is_available}")
        prints(f"initialized = {self.initialized}\nstage = {self.stage}\n")
        prints(f"n_qubits = {self.n_qubits}\ndet = {self.det}")
        prints(f"\nH_old =")
        print_list(list(self.H_old.terms.items())[:10])
        prints(f"\nE_matrix =\n{np.array(self.E_matrix[:10])}\n")
        prints(f"\ncharacter_list =\n{np.array(self.character_list)}\n")
        prints(f"\nE_kernel =\n{np.array(self.E_kernel)}\n")
        prints(f"\ntau_list =")
        print_list(self.tau_list)
        prints(f"\ncommutative_sigmas =")
        print_list(self.commutative_sigmas)
        prints(f"\nredundant_bits =\n{self.redundant_bits}\n")
        prints(f"\nclifford_operators =")
        print_list(self.clifford_operators)
        prints(f"\n^^^^^^^^^^^^^^^^^^^^^^^\n   TAPERING.PY  DEBUG\n#######################\n")
        
        
    def run(self, mode='np', mapping='jordan_wigner', XZ='Z'):
        """Function
        Transform Hamiltonian based on the tapering-off techniques
        (arXiv:1701.08213; arXiv:1910.14644).

        Author(s): Takashi Tsuchimochi, TsangSiuChung
        """
        self.initialized = 1
        self.stage = 1
        t0 = time.time()

        if self.H_old is not None and self.Sz_symmetry:
            # Get get parity matrix E
            self.E_matrix = get_parity_matrix_E(self.H_old, 
                                                self.n_qubits, 
                                                mode)

            # Get the null space of E (subspace). 
            # This corresponds to a set of (gx|gz)
            self.E_kernel = binary_kernel(self.E_matrix)
            self.stage = 2
        # Initial PGS tapering if it is available
        if self.character_list is not None and len(self.character_list) != 0:
            if self.E_kernel is None:
                self.E_kernel = []
            self.E_kernel = include_pgss(self.E_kernel, 
                                         self.character_list, 
                                         mapping)
            self.stage = 3

        if self.E_kernel is not None:
            # Get the list of commutative taus (Pauli Z string)
            # e.g. IZIZZIZZI
            if not self.Sz_symmetry:
                ### No Sz symmetry assumed, but the number symmetry (N = NA + NB) itself is assumed.
                self.E_kernel.append(2**self.n_qubits - 1)
            self.g_list = rref(self.E_kernel)
            self.tau_list, self.tau_info = get_commutative_tau(self.H_old, 
                                                            self.g_list, 
                                                            self.n_qubits)
            tau_iscommute = list(map(itemgetter(0), self.tau_info))
            if all(tau_iscommute):
                self.commutative_taus = self.tau_list
            elif any(tau_iscommute):
                self.commutative_taus = list(compress(self.tau_list, tau_iscommute))
            self.stage = 4


        if self.commutative_taus is not None:
            # Get the list of commutative sigmas (Pauli X string)　e.g. IXIIIIIII
            redundant_bits_Z, commutative_sigmas_Z, taus_Z = get_commutative_sigma(self.commutative_taus, 'Z')
            redundant_bits_X, commutative_sigmas_X, taus_X = get_commutative_sigma(self.commutative_taus, 'X')
            if (taus_Z != [] and taus_Z is not None) and XZ == 'Z':
                self.redundant_bits = redundant_bits_Z
                self.commutative_sigmas = commutative_sigmas_Z
                self.commutative_taus = taus_Z
                redundant_bits = redundant_bits_Z
                commutative_sigmas = commutative_sigmas_Z
                commutative_taus = taus_Z
                #prints(self.redundant_bits)
                #prints(self.commutative_sigmas)
                #prints(taus_Z)
                if taus_X != [] and taus_X is not None:
                    prints('WARNING: X redudant tau exists. Only Z redundancy is considered.')
            elif taus_X != [] and taus_X is not None and XZ == 'X':
                if self.redundant_bits is not None:
                    self.redundant_bits.extend(redundant_bits_X)
                    self.commutative_sigmas.extend(commutative_sigmas_X)
                    self.commutative_taus.extend(taus_X)
                else:
                    self.redundant_bits = redundant_bits_X
                    self.commutative_sigmas = commutative_sigmas_X
                    self.commutative_taus =taus_X
                redundant_bits = redundant_bits_X
                commutative_sigmas = commutative_sigmas_X
                commutative_taus = taus_X
            elif taus_Z == taus_X == [] or taus_Z == taus_X == None:
                redundant_bits = None
                self.redundant_bits = None
                self.commutative_sigmas = None 
                self.commutative_taus = None
                #return
                 


            # Combine tau and sigma to obtan Clifford Operators
            # iff those component found are valid for tapering.
            # Split the commutative_sigma_result into components
            self.stage = 5

        if redundant_bits is not None and commutative_sigmas is not None:
            # Get all Clifford Operators Ui
            clifford_operators = get_clifford_operator(commutative_taus, 
                                                       commutative_sigmas)
            if self.clifford_operators is not None and XZ == 'X':
                self.clifford_operators.extend(clifford_operators)
            else:
                self.clifford_operators = clifford_operators
            self.stage = 6
        else:
            clifford_operators = None
            

        #if self.clifford_operators is not None:
        if clifford_operators is not None:
            # Finally, transform the Hamiltonian by U = U0 * U1 * ... * Uk
            H_new = self.H_old
            for clifford_operator in clifford_operators:
                H_new  =  (clifford_operator * H_new * clifford_operator)
                H_new.compress()
            # Clean up some 10e-7 terms especially in the case of NH3
            self.H_new = purify_hamiltonian(H_new, self.redundant_bits)
            self.stage = 7

        if self.H_new is not None:
            # Verify the result
            self.validity = judge_reduced_hamiltonian(self.H_new, self.redundant_bits)
            self.stage = 8

        if self.validity == []:
            # Obtain coefficients that replace the Pauli X or I on redundent qubits
            if mapping in ('jw', 'jordan_wigner'):
                det = self.det
            elif mapping in ('bk', 'bravyi_kitaev'):
                det = jw2bk(self.det, self.n_qubits)
            else:
                raise ValueError(f"Unrecognized mapping {mapping}")
            
            self.X_eigvals = get_new_coefficient(self.commutative_taus, 
                                                            det)
            # Replace redundent qubits by those coefficients 
            # and reconstruct Hamiltoian to remove redundent qubits
            #self.H_renew = tapering_off_operator(self.H_new, self.redundant_bits, 
            #                                                self.X_eigvals, 1)
            self.stage = 9

        if self.redundant_bits is not None:
            self.n_qubits_sym = self.n_qubits - len(self.redundant_bits)
        else:
            self.n_qubits_sym = self.n_qubits 
        t1 = time.time()
        self.timelog = t1-t0
        prints(self)
        return None
    
    
### Subroutines ###
def get_parity_matrix_E(H, n_qubits=None, mode='py'):   
    '''Function
    Acknowledgement: D. Gottesman(1997), Stabilizer Codes and Quantum Error Correction, p.8

    Return the parity matrix which use in tapering sequence.
    Generator matrix G is a matrx that encodes a Hamiltonian represented by 
    Pauli Operators to GF2(binary) space.
    Pauli Operators generated by {I,X,Y,Z} will map to {00,10,01,11}.
    First of all, all Pauli strings are expressed in the formalism of above mapping.
    E.g. IXZIY = func sigma(01001,00101)
    Although XZ = -iY, -i is just the phase and generally we ignore it.
    Hence there is an isomorphism between Pauli operators by generator <X,Z>
    and by generator <10,01>.
    G_matrix will be a matrix with dimension (2M=2*Mqubits,  r=# terms in Hamiltonian)
        Gx (int 2darray): A binary matrix of Pauli X partition of G.
        Gz (int 2darray): A binary matrix of Pauli Z partition of G.
        G_matrix (int 2darray): A binary matrix (Gx|Gz)^T.
    
    Parity matrix of a generator marix which is just the dual matrix of G.
    And Gx Gz swapped.

    Args:
        H (QubitOperator): Hamiltonian.

    Returns:
        E_matrix (int 2darray): A binary matrix = (Ex|Ez).

    Author(s): Takashi Tsuchimochi, TsangSiuChung
    '''
    def pyHamiltonian2Parity():
        # In pure python standard library
        # Using true binary arithmatic to make it numpy speed
        paulis = HamiltonianWOConstant(H).terms.keys()  # Get rid of constant term
        E_matrix = [ Pauli2RealBin(pauli, n_qubits, 1) for pauli in paulis ]
        return E_matrix

    def npHamiltonian2Parity():
        # Since that python standard library is the fastest in terms of appending,
        # first append the list and then transfer to np.ndarray.
        # This two-stage approach is 4 times faster than pure numpy solution.
        paulis = HamiltonianWOConstant(H).terms.keys()  # Get rid of constant term
        G_matrix = [ Pauli2Bin(pauli, n_qubits) for pauli in paulis ]
        G_matrix = np.array(G_matrix).T

        G_split = np.vsplit(G_matrix, 2)
        Gx = G_split[0]
        Gz = G_split[1]
        Ex = Gz.T  # Caution!! XZ inverted!!
        Ez = Gx.T  # Caution!! XZ inverted!!
        E_matrix = np.hstack([Ex,Ez])
        return E_matrix

    # Map every terms in the Hamiltonian to binary code s
    # then stack them up to construct the G_matrix.
    if n_qubits is None:
        n_qubits = Hamiltonian2n_qubits(H)
    func = {'py':pyHamiltonian2Parity , 'np':npHamiltonian2Parity}
    return func[mode]()  ## This is marginally faster than if-elif statement


def get_commutative_tau(H, g_list, n_qubits, cutoff=6):
    '''Function
    Returns a list of commutative Pauli Z strings for construction of Clifford
    Operators from a list of binary represented Pauli strings.

    Args:
        H (QubitOperator): Hamiltonian.
        g_list (int 2darray): List of binary Pauli strings represented in GF2.
        cutoff (int): Parameter to set in which level it is considered as 'commutative'.
                      For NH3 molecule to work, it has to set to 6.
    Returns:
        tau_list (QubitOperator 1darray): List of Pauli Z strings.
        tau_info (bool,float,dict tuple): Information regarding commutativity.

    Author(s): Takashi Tsuchimochi, TsangSiuChung
    '''
    if isinstance(g_list, np.ndarray):
        tau_list = [ Bin2QubitOperator(x, n_qubits) for x in g_list ]
    elif isinstance(g_list, list):
        tau_list = [ RealBin2QubitOperator(x, n_qubits) for x in g_list ]    
    tau_info = []
    for tau in tau_list:
        # Check if each tau commutes with Hamiltonian
        # Return commitativity information
        commutativity, remainder, stats = informative_openfermion_commutator(H, tau)
        sum_comm = sum(commutativity)
        # Append commutative tau in to the list
        if sum_comm == 0:
            comm = False
            comm_state = -1  # -1 means not commutative at all
        elif sum_comm < cutoff:
            comm = False
            comm_state = 10**-(sum_comm-1)
        elif sum_comm >= cutoff:
            comm = True
            comm_state = 10**-(sum_comm-1)
        tau_info.append((comm, comm_state, remainder, stats))
    return tau_list, tau_info


def get_commutative_sigma(commutative_taus, XZ='Z'):
    '''Function
    Returns a list of commutative Pauli X (Z) strings for construction of Clifford
    Operators and the index of them. 
    Those indices indicate the redundent qubits that eligible to tapering-off.
    This function is in fact redundent if rref() is functioning well because
    indices of redundent qubits is required to be independent. 
    So if rref() is not malfunctioning, the results must be theose pivots. 

    Args:
        commutative_taus (QubitOperator 1darray): List of Pauli Z strings
                                                    that commute with Hamiltonian.

    Returns:
        redundant_bits (int 1darray): List of index of Pauli X.
        commutative_sigmas (QubitOperator 1darray): List of Pauli X strings
                                                    that commute with all other
                                                    Pauli Z strings.

    Author(s): Takashi Tsuchimochi, TsangSiuChung
    '''
    # Reject if no commutative tau available
    nrows = len(commutative_taus) 
    if nrows == 0:
        return None, None, None

    # Get position of all paulis
    z_positions = QubitOperatorInfoExtracter(commutative_taus, 0, tuple)
    pauli_index = QubitOperatorInfoExtracter(commutative_taus, 1, tuple)

    # Searching for appropriate Pauli X string(Sigma) for each row(tau)
    redundant_bits = []
    commutative_sigmas = []
    taus = []
    total_num_checked = 0
    for i, (tau1, Zs) in enumerate(zip(commutative_taus, z_positions)):
        ncols = len(Zs) # number of entries in ith row
        # Scan through all z_positions in a tau for the commutative Sigma
        for k, Z in enumerate(Zs):
            # Skip if the current bit is already confirmed as redundent bit
            # Otherwise do the commutativity check
            if Z in redundant_bits:
                continue   
            # Sigma is a Pauli X string where X is right on redundent bit
            if pauli_index[i][k] == XZ:
                if XZ == 'Z':
                    sigma = QubitOperator((Z,'X'))
                elif XZ == 'X':
                    sigma = QubitOperator((Z,'Z'))
            else:
                continue
            # Compare Sigma to all other taus
            # If they all commutes, it is the redundent bit we are looking for
            # Keep track of how many rows Sigma commutes with
            count = 0
            # Count for total number of rows for second time
            for tau2 in commutative_taus:              
                # Skip if comoparing to the same tau
                # Otherwise do the commutativity check
                if tau1 == tau2:
                    continue  
                comm = commutator(sigma, tau2)
                total_num_checked += 1
                if len(comm.terms) == 0:
                    count +=1
                else:
                    prints(f"sigma of {tau1} not commutative with tau {tau2}")
                    # Can break tau1 since there is at least a chance not commute
                    break 
            if count == nrows-1:
                redundant_bits.append(Z)
                commutative_sigmas.append(sigma)
                taus.append(tau1)
            # Since the Sigma is found for the current row(tau),
            # We can break the second layer of loop to save some resources
            # Else it will search for another available sigma within same tau
            break

#    if total_num_checked != nrows*(nrows-1):
#        prints(f'''Error occured in get_commutative_sigma().
#Should be in total {nrows*(nrows-1)} times checking sequences initiated
#but only {total_num_checked} times is recorded.
#rref() may be malfunctioning. Go check it.
#Usually it is due to the last pivot not subtracted from the matrix.\n''')
    return redundant_bits, commutative_sigmas, taus

        
def get_clifford_operator(commutative_taus, commutative_sigmas):
    '''Function
    Returns a list of Clifford Operator Unitary which is simply the sum of each 
    commutative tau and its commutative sigma counterpart. 

    Args:
        commutative_taus (QubitOperator 1darray): List of Pauli Z strings
                                                    that commute with Hamiltonian.
        commutative_sigmas (QubitOperator 1darray): List of Pauli X strings
                                                    that commute with all other
                                                    Pauli Z strings.

    Returns:
        clifford_operators (int 1darray): List of Clifford Unitary operators.


    Author(s): Takashi Tsuchimochi, TsangSiuChung


    '''
    clifford_operators = []
    for sigma, tau in zip(commutative_sigmas, commutative_taus):
        index = QubitOperatorInfoExtracter(sigma, 0, tuple)
        pauli_index = QubitOperatorInfoExtracter(sigma, 1, tuple)
        if pauli_index[0][0] == 'Z':
            clifford_operators.append((1/2) * (sigma + tau) * (QubitOperator((index[0][0],'X')) + QubitOperator((index[0][0],'Z'))))
        elif pauli_index[0][0] == 'X':
            clifford_operators.append((1/np.sqrt(2)) * (sigma + tau))
    return clifford_operators


def purify_hamiltonian(H, redundant_bits, keep='X'):
    '''Function
    Clean up abnormal 10e-7 terms in Hamiltonian. Especially for NH3 molecule.

    Args:
        H (QubitOperator): Hamiltonian.
        redundant_bits (int 1darray): A list of removed bits.
        keep (str): The Pauli XYZ intendended to keep in place.

    Returns:
        H_purified (QubitOperator): Hamiltonian without abnormal 10e-7 terms.

    Author(s): Takashi Tsuchimochi, TsangSiuChung
    '''
    if redundant_bits is None:
        return H
    # Prepare for fast set relation check
    redbits = frozenset(redundant_bits)
    forbidden = {'X', 'Y', 'Z'}
    forbidden.discard(keep)
    H_purified = H
    # Convert all pauli tuples to dictionary
    # hashed object to enhance speed
    paulis = list(dict(x) for x in HamiltonianWOConstant(H).terms.keys())  # Pop constant term
    # Extract terms that intersect with {redundent bits}
    mask = (redbits.intersection(x) for x in paulis)
    xyz = (tuple(p.items()) for p, m in zip(paulis, mask)
           if m and forbidden.intersection(itemgetter(*m)(p)))
    # Eliminate those polluted terms from the Hamiltonian
    for k in xyz:
        H_purified -= QubitOperator(k, H.terms[k])
    return H_purified


def judge_reduced_hamiltonian(H, redundant_bits, keep='X'):
    """Function
    Check if the transformed Hamiltonian only contains I or X at 
    the redundant qubits.
    Introduce extra loopings only if tapering has failed.

    Args:
        H (QubitOperator): Transformed Hamiltonian by Tapering-Off qubits.    
        redundant_bits (int 1darray): A list of removed bits

    Returns:
        errors (2darray): List of terms carry things other than 
                          I&X redundant bits. Length == 0 if no error occured. 

    Author(s): Takashi Tsuchimochi, TsangSiuChung
    """
    if redundant_bits is None:
        return True
    # Prepare for fast set relation check
    redbits = frozenset(redundant_bits)
    forbidden = {'X', 'Y', 'Z'}
    forbidden.discard(keep)
    # Convert all pauli tuples to dictionary
    # hashed object to enhance speed
    paulis = list(dict(x) for x in HamiltonianWOConstant(
        H).terms.keys())  # Pop constant term
    # Extract terms that intersect with {redundent bits}
    mask = (redbits.intersection(x) for x in paulis)
    # See is there still polluted rows
    is_polluted = [tuple(p.items()) for p, m in zip(paulis, mask)
                   if m and forbidden.intersection(itemgetter(*m)(p))]
    return is_polluted

    
def get_new_coefficient(commutative_taus, det):
    '''Function
    Returns the list of new coefficients for post-tapering-off Hamiltonian at 
    redundent qubits.
    As those redundent qubits will became Pauli I or X after tapering-off,
    eigenvalues for certain Pauli operator will be the expectation value when
    taking measurements.
    For Pauli X operator, its eigenvalues is {+1, -1}.
    By taking the measurement with respect to Pauli X at each redundent qubit, 
    the coefficient is taken from the expectation value in base state, 
    which is Hartree-Fock state. 
    Such as |HF> = |00001111111111> for 10 electrons in 14 orbitals
    The expectation value is taken in the follow fashion.
    <HF|Ui Xi Ui|HF>  such that U is the Cliiford Operator for 
    i th redundent qubit.
    Which Ui = 1/sqrt(2) * [ (Xi) + (Za Zb ... Zi ... Zk) ]
    Hence <HF|Ui Xi Ui|HF> = 1/2 <HF|   Xi    Xi    Xi   |HF>
                           + 1/2 <HF| Zabik   Xi    Xi   |HF>
                           + 1/2 <HF|   Xi    Xi   Zabik |HF>
                           + 1/2 <HF| Zabik   Xi    Zi   |HF>

                           = 1/2 <HF|    Xi   |HF>
                           + 1/2 <HF|  Zabik  |HF>
                           + 1/2 <HF|  Zabik  |HF>
                           + 1/2 <HF|   -Xi   |HF>

    Since Xi is bit-flip and Zi is phase-flip, 
    Xi alter |HF> and kronecker delta = 0 for Xi's .

    Thus, <HF|Ui Xi Ui|HF> = 1/2 <HF|  Zabik |HF>
                           + 1/2 <HF|  Zabik |HF>  = <HF| Zabik |HF>
    
    Therefore we only have to count how many phase-flip occured in total,
    that is, the total number of Pauli Z operator within occupied orbitals.

    For example for H2O @ D2h symmetry,
        redundent qubits =    [  0,  1,  4,  8]
    new_coefficient_list =    [ -1, -1,  1,  1]

    Args:
        commutative_taus (QubitOperator 1darray): The Z partition of Clliford.
        det (int): Determinant.

    Returns:
        X_eigvals (int 1darray): List of new coefficients for 
                                            post-tapering-off Hamiltonian 
                                            at redundent qubits.

    Author(s): TsangSiuChung, Takashi Tsuchimochi
    '''
    index_list = QubitOperatorInfoExtracter(commutative_taus, 0, tuple)
    X_eigvals = []
    from quket.utils import is_1bit
    for index in index_list:
        exponential = sum( is_1bit(det, x) for x in index )
        X_eigvals.append((-1)**exponential)
    return X_eigvals
        
        
    
def tapering_off_operator(operator, redundant_bits, eigvals_list, eliminate=True): 
    '''Function
    Modify operator after Unitary Transformation by
    replacing the Pauli X acting of redundent qubit with its eigenvalue.
    Args:
        operator (QubitOperator): Operator after Unitart Transformation. 
                               Openfermion QubitOperator instance.
        redundant_bits (int 1darray): List of redundent qubits that are available 
                               to be tapering-off.
        evals_list (int 1darray): List of eigvalues corresponding to each redbit.
        eliminate (bool): Toggle to eliminate redundent qubits.

    Return:
        new_operator (QubitOperator): Operator with redundent qubit replaced 
                               by eigenvalues.
                               Openfermion QubitOperator instance.

    Author(s): Takashi Tsuchimochi, Kazuki Sasasako, TsangSiuChung
    '''
    # Prepare an empty Operator for containing stuffs
    new_operator = QubitOperator('',0)
    # Prepare for fast set relation check
    alternative = dict(zip(redundant_bits, eigvals_list))
    # Pickout terms from the post-tapering-off Operator
    for key,value in operator.terms.items():
        # Batch transformation all entries into list 
        # Because modifying things in list directly instead of appending
        # is much faster
        key = [ list(item) for item in key ]
        # Then call out each entry in Pauli string
        for j,(bit,XYZ) in enumerate(key):
            # Target = Pauli X act on the redundent qubit.
            # Multiply the coefficient of current Pauli string by 
            # redundent Pauli X's corresponding eigenvalue 
            if (bit in redundant_bits) and (XYZ =='X'):
                value *= alternative[bit]
                key[j] = (None, None)
            # Error handling if something else is in redundent qubit
            elif (bit in redundant_bits) and (XYZ !='X'):
                #prints("Something wrong. not X is in redbit.")
                #prints("So no count for this row!")
                key[j] = (None, None)
                break
            # Otherwise just reduce the index of where XYZ act on normal qubit
            # (because we have already tapered-off those redundent qubits!)
            else:
                if eliminate:
                    subtrahend = sum( bit > r_bit for r_bit in redundant_bits )
                    key[j][0] -= subtrahend
        # Batch construction of QubitOperator from Pauli string and coefficient
        new_operator_ = Pauli2QubitOperator((key, value))
        # Add to the renewed Operator
        new_operator += new_operator_
    # Optional action. Not necessary.
    new_operator.compress()
    return new_operator
  
    
### Uilities ###
def pyShape(L, shp=()):
    '''Function
    Return shape of Python Iterable.
    Error occur when len(Iterable)==0.

    Args:
        L (Iterable): Python Iterable.
        shp (list): Parameter. Must not mess with this.
    
    Reurn:
        shp (int tuple): Shape of Iterable.

    Author(s): TsangSiuChung
    '''
    if isinstance(L, Iterable):
        pass
    if isinstance(L, str):
        return None        
    shp += (len(L),)
    check = [ isinstance(row, Iterable) for row in L ]
    if all(check):
        temp = pyShape(L[0], shp)
        if temp is not None:
            shp = temp
    return tuple(shp)


def rref(A, debug=False):
    """Function
    Perform Gaussian elimination in GF2 to get
    reduced row echelon form of a binary matrix.

    Args:
        A (int 2darray): A binary matrix

    Returns:
        A_ (int 2darray): The reduced row echelon form of A

    Author(s): Takashi Tsuchimochi, TsangSiuChung
    """
    if isinstance(A, np.ndarray):
        if A.ndim != 2:  # Raise alert if input value is not a matrix
            prints('Error @ rref because input value is not 2d Binary Matrix.')
            return A
        elif A.ndim == 1:
            return A
        A_ = np.copy(A)
        # Empty list as the containment of pivovt_row's index using at the end of loop
        indices_of_pivot_rows = []
        m, n = A_.shape  # m x n matrix
        i = 0
        j = 0
        # Scanning the whole matrix
        while i < m and j < n:
            # Scan each column to find the pivot
            # Each column is partition by i th row
            # When there is nothing more then zeros in the column partition,
            # jump to next column
            while np.sum(A_[i:, j]) == 0:
                j += 1
                # Stop when counting exceeds the width of the matrix,
                # Because all pivots will be found when we reach the last column
                # So performs twice loop breaker to excape from loop
                if j == n:
                    break
            if j == n:
                break
            # Here we obtain the index of one pivot
            # Extract row index of the largest number in the column partition
            # Since this row index is from the partition,
            # we have to add the remainings to obtain the full row index
            pivot_i = np.argmax(A_[i:, j]) + i
            pivot = A_[pivot_i, j]
            if pivot == 1:  # if the pivot==1, but what if it is bigger than 1?
                # If pivot is not the current row, swap them
                if pivot_i != i:
                    temp = np.copy(A_[pivot_i])
                    A_[pivot_i] = A_[i]
                    A_[i] = temp
                # pivot row is now the current row
                # Extract current row i.e. pivot row after the pivot
                pivot_row_partition = A_[i, j:]
                # current column's component taken from all rows,
                # literally the pivot column
                pivot_col = np.copy(A_[:, j])
                # Remove pivot row from pivot column,
                # since we are not performing elimination onto itself
                pivot_col[i] = 0
                # Subtract pivot row from all other row to eliminate the dependent components
                # This is in fact subtracting the outer prduct from the original partiton matrix
                # Just Google for some papers
                subtrachend = np.outer(pivot_col, pivot_row_partition)
                # Since it is a binary matrix, take the XOR
                A_[:, j:] = A_[:, j:] ^ subtrachend
                # We are sure enough the current row is pocessing a pivot
                # Hence store the current row number in the prepared list
                indices_of_pivot_rows.append(i)
            else:  # Raise alert if pivot!=1 since input value is wrong
                prints(f"""*******\nError alert.
                        pivot of {pivot} appears during binary Gaussian Eliminiation.
                        While pivot of 1 is the preferred value.
                        *******""")
            # Cleared one pivot, so move on to find next pivot
            i += 1
            j += 1
        # Return the Gaussian Eliminated matrix
        return A_[indices_of_pivot_rows]

    elif isinstance(A, list):
        shape = pyShape(A)
        if len(shape) != 1:
            prints('Error @ rref because input value is not 1d Binary Matrix.')
            return A
        elif len(shape) == 0:
            return A
        A_ = deepcopy(A)
        # nbits := width of the matrix
        nbits = len(bin(max(A_))[2:])
        # P_ideal := ideal Pivot Row which log2(x)=integer
        P_ideal = 1 << (nbits-1)
        # P_len := length of Pivot Rrow currently in search
        P_len = nbits
        #--- For debug ---#
        if debug:
            prints(f"{np.array(A_)}\n{nbits}\n{P_ideal} {bin(P_ideal)}\n")
        for n in range(nbits):
            # Clean up A_ to shrink the matrix
            A_ = set(A_)
            A_.discard(0)
            A_ = list(A_)
            # Sort to speedup
            # Sort twice is necessary and faster (I don't know why)
            # Basically put sort all to-be-eliminated rows on top
            # to speed up the cherry picking process right after it
            A_.sort(reverse=True)
            # break if can not reduce any more
            if P_len < len(bin(A_[-1])[2:]):
                break
            A_.sort(reverse=True, key=lambda x: bool(x & P_ideal))
            # --- For debug ---# Print A_ pre Gaussian Elimination
            if debug:
                pyE2 = [bin(x)[2:] for x in A_[:30]]
                for row in pyE2:
                    prints(f"{row:>40}")
                prints("\n")
            # Picking up those rows eligible to be eliminated
            end_pt = 0
            for i, x in enumerate(A_):
                # Since A_ is sorted by x & P_ideal, this is safe
                if x & P_ideal == 0:
                    break
                end_pt += 1
            # Take the last one as Pivot Row for good
            # It might be independent since the A_ is sorted
            # independent -> reduce bit-flips to increase performance
            P_row = A_[end_pt-1]
            #--- For debug ---#
            if debug:
                prints(f"end_pt = {end_pt}")
                prints(
                    f"P_ideal= {P_ideal} {bin(P_ideal)[2:]}\nP_len= {P_len}")
                prints(
                    f"P_row= {P_row} {bin(P_row)[2:]} @{len(bin(P_row)[2:])}")
                prints(f"A_[:end_pt] before = {A_[:end_pt]}")
            # Skip if current column is already independent
            if len(bin(P_row)[2:]) > P_len:
                P_len -= 1
                P_ideal >>= 1
                continue
            # Eliminate all Pivots from eligible rows
            # Directly work on the original list, also preserving P_row
            for k, x in enumerate(A_[:end_pt-1]):
                A_[k] = x ^ P_row
            # --- For debug ---# Print if the Gaussian Elimination successfuly performed
            if debug:
                prints(f"A_[:end_pt] after  = {A_[:end_pt]}")
            P_ideal >>= 1
            P_len -= 1
            # --- For debug ---# Print A_ post Gaussian Elimination
            if debug:
                pyE2 = [bin(x)[2:] for x in A_[:30]]
                for row in pyE2:
                    prints(f"{row:>40}")
                prints("\n")
        return A_


def binary_kernel(A, debug=False):
    """Function
    Given a matrix A, return its kernel (null space) in GF2.

    Args:
        A (int 2darray): Binary matrix for which the null space is searched for.

    Returns:
        nullspace (int 2darray): A set of null vectors (kernel) of matrix A.

    Author(s): Takashi Tsuchimochi, TsangSiuChung
    """
    if isinstance(A, np.ndarray):
        # First perform Gaussian Elimination in GF2 to get
        # the reduced row echelon form.
        rrefA = rref(A)
        m, n = rrefA.shape
        #--- DEBUG ---#
        if debug:
            prints(m, n)
        # Check pivots
        # Check the linearly dependent basis
        i = 0
        j = 0
        independency = []
        null_list = []
        while i < m:  # Counting Columns
            # If diagonal == 1, it must be the pivot, else see next column
            if rrefA[i, j] == 1:
                independency.append(True)  # True for independent component
                i += 1  # Next Row
                j += 1  # Next Column
            else:
                independency.append(False)  # False for dependent component
                null_list.append(j)
                j += 1  # Next Column
        # Complete the null_list for the rest of it
        if n > j:
            delta = n-j
            #--- DEBUG ---#
            if debug:
                prints(delta)
            independency += [False for w in range(delta)]
            null_list += [k for k in range(j, n, 1)]
        #--- DEBUG ---#
        if debug:
            prints(null_list)
            prints(independency, 'len=', len(independency))
            prints(rrefA)
        rrefA = rrefA[:, null_list]
        #--- DEBUG ---#
        if debug:
            prints(rrefA)
            prints(f"kernel size= {independency.count(False)}")
        n2 = len(null_list)
        for i, j in enumerate(null_list):
            temp = np.zeros(n2, dtype=int)
            temp[i] = 1
            rrefA = np.insert(rrefA, j, temp, axis=0)
            if debug:
                prints(f"rrefA >>\n{rrefA}\n")
        nullspace = rrefA.T
        if debug:
            prints(f"nullspace >>\n{nullspace}\n")
        return rref(nullspace)

    elif isinstance(A, list):
        # First perform Gaussian Elimination in GF2 to get
        # the reduced row echelon form.
        rrefA = rref(A)
        m = pyShape(rrefA)[0]
        n = len(bin(rrefA[0])[2:])
        nullity = n-m
        rrefA = [BinNum2List(x, n) for x in rrefA]
        #--- DEBUG ---#
        if debug:
            prints(m, n)
            prints(np.array(rrefA))
        # Check pivots
        # Check the linearly dependent basis
        dependency = []
        null_list = []
        i = 0
        j = 0
        while i < m:  # Counting Columns
            if rrefA[i][j] == 1:
                dependency.append(False)  # True for independent component
                j += 1  # Next Column
                i += 1  # Next Row
            else:
                dependency.append(True)  # False for dependent component
                null_list.append(j)
                j += 1  # Next Column
        # Complete the null_list for the rest of it
        delta = n-j
        if delta > 0:
            dependency += [True for w in range(delta)]
            null_list += [k for k in range(j, n, 1)]
        if nullity != len(null_list):
            prints("nullity != len(null_list)")
        rank = null_list[0]
        dependency = dependency[rank:]
        excave = dependency.count(False)
        #--- DEBUG ---#
        if debug:
            prints(f"delta = {delta}")
            prints(f"j = {j}")
            prints(f"dependency = {dependency}")
            prints(f"null_list = {null_list}")
            prints(f"rank = {rank}")
            prints(f"nullity = {nullity}")
            prints(f"excave = {excave}")
        nullspace = list(map(itemgetter(*null_list), rrefA))
        #--- DEBUG ---#
        if debug:
            prints(f"nullspace =\n{nullspace}\n")
        tail = []
        for k in range(excave):
            tail.append(nullspace.pop())
        if debug:
            prints(f"tail =\n{tail}\n")
        null_count = 0
        for dep in dependency:
            if dep:
                temp = [0 for k in range(nullity)]
                temp[null_count] = 1
                null_count += 1
                nullspace.append(temp)
            else:
                nullspace.append(tail.pop())
            if debug:
                prints(f"nullspace >>\n{np.array(nullspace)}\n")

        nullspace = list(map(BinList2Num, zip(*nullspace)))
        #--- DEBUG ---#
        if debug:
            prints(np.array(nullspace))
        if len(nullspace) == 0:
            return None
        else:
            return nullspace


def Pauli2Bin(pauli, n_qubits, swap=False):
    """Function
    Convert a tensor of Pauli operators to a binary vector  a = (ax|az).
    Opposite to Bin2Pauli.

    Args:
        pauli (tuple): A tuple representing Pauli operator in
                       the OpenFermion format.
        n_qubits (int): Number of qubits.

    Returns:
        a (int 1darray): A binary array (ax|az).

    Author(s): Takashi Tsuchimochi, TsangSiuChung
    """
    ax = [0 for n in range(n_qubits)]
    az = [0 for n in range(n_qubits)]
    for qubit, XYZ in pauli:
        if XYZ == "X":
            ax[qubit] = 1
        elif XYZ == "Y":
            ax[qubit] = 1
            az[qubit] = 1
        elif XYZ == "Z":
            az[qubit] = 1
    if swap:
        return az+ax
    else:
        return ax+az


def Bin2Pauli(a, n_qubits):
    """Function
    Convert a binary vector  a = (ax|az) to a tensor of Pauli operator.
    Opposite to Pauli2Bin.

    Args:
        a (int 1darray): A binary array (ax|az).
        n_qubits (int): Number of qubits.

    Returns:
        pauli (tuple): A tuple representing Pauli operator in
                       the OpenFermion format.

    Author(s): Takashi Tsuchimochi
    """
    ax = a[0:n_qubits]
    az = a[n_qubits : 2*n_qubits]
    pauli = []
    for i in range(n_qubits):
        if ax[i] == az[i] == 1:
            pauli.append((i, "Y"))
        elif ax[i] == 1:
            pauli.append((i, "X"))
        elif az[i] == 1:
            pauli.append((i, "Z"))
    return pauli


def Bin2QubitOperator(a, n_qubits):
    """Function
    Directly convert a = (ax|az) to QubitOperator (with coefficient 1).

    Args:
        a (int 1darray): A binary array (ax|az).
        n_qubits (int): Number of qubits.

    Returns:
        QubitOperator (QubitOperator): QubitOperator in the OpenFermion format.

    Author(s): Takashi Tsuchimochi
    """
    return QubitOperator(Bin2Pauli(a, n_qubits))


def Pauli2RealBin(pauli, n_qubits, swap=False):
    k = n_qubits-1
    ax = 0
    az = 0  
    for qubit, XYZ in pauli:
        n = k - qubit
        if XYZ == "X":
            ax += 2**n
        elif XYZ == "Y":
            ax += 2**n
            az += 2**n
        elif XYZ == "Z":
            az += 2**n
    if swap:
        az <<= n_qubits
        return az|ax
    else:
        ax <<= n_qubits
        return ax|az


def RealBin2Pauli(a, n_qubits, swap=False):
    if swap:
        ax = a & 2**n_qubits-1  #  0011'1001'
        az = a >> n_qubits      # '0011'1001
    else:
        az = a & 2**n_qubits-1  #  0011'1001'
        ax = a >> n_qubits      # '0011'1001
    ay = ax & az
    ax = (~ay) & ax 
    az = (~ay) & az
    a = {'X':bin(ax)[2:], 'Y':bin(ay)[2:], 'Z':bin(az)[2:]}
    pauli = []
    for key,val in a.items():
        head_len = n_qubits - len(val)
        start_pt = 0
        while True:
            try:
                index = val.index('1',start_pt)
                bit = index + head_len
                pauli.append((bit, key))
                start_pt = index+1
            except ValueError:
                break
    return pauli


def RealBin2QubitOperator(a, n_qubits, swap=False):
    if swap:
        ax = a & 2**n_qubits-1  #  0011'1001'
        az = a >> n_qubits      # '0011'1001
    else:
        az = a & 2**n_qubits-1  #  0011'1001'
        ax = a >> n_qubits      # '0011'1001
    ay = ax & az
    ax = (~ay) & ax 
    az = (~ay) & az
    a = {'X':bin(ax)[2:], 'Y':bin(ay)[2:], 'Z':bin(az)[2:]}
    pauli = []
    for key,val in a.items():
        head_len = n_qubits - len(val)
        start_pt = 0
        while True:
            try:
                index = val.index('1',start_pt)
                bit = index + head_len
                pauli.append(f"{key}{str(bit)} ")
                start_pt = index+1
            except ValueError:
                break
    pauli = ''.join(pauli)
    return QubitOperator(pauli, 1)


def Pauli2QubitOperator(pauli_tuple):
    '''Function
    [Pauli Strings, coefficient] -> QubitOperator

    Args:
        pauli_tuple (tuple): Information about a single QubitOperator:
                           (Pauli string, Coefficient)

    Returns:
        QubitOperator (QubitOperator): An OpenFermion QubitOperator instance.

    Author(s): TsangSiuChung
    '''
    pauli = pauli_tuple[0]
    coeff = pauli_tuple[1]
    temp = [ f"{XYZ}{bit} " for bit,XYZ in pauli \
            if (bit,XYZ) != (None,None) ]
    final_pauli_str = ''.join(temp)
    return QubitOperator(final_pauli_str, coeff)
    

def BinNum2List(S, width=None):
        S = bin(S)[2:]
        q = len(S)
        if width is None:
            delta = 0
        else:
            delta = width-q
        return [0 for ii in range(delta)] + [int(x) for x in S]


def List2Str(L):
    return ''.join(str(L)[1:-1].split(', '))


def BinList2Num(l):
    N = eval(f"0b{List2Str(l)}")
    return N


def HamiltonianWOConstant(H):
    '''Function
    Get rid of constant from the Hamiltonian.

    '''
    H_ = deepcopy(H)
    H_ -= H_.constant
    return H_


def Hamiltonian2n_qubits(H):
    '''Function
    Return number of qubits of a Hamiltonian.
    Author(s): TsangSiuChung
    '''
    # Get the number of qubits used in this Hamiltonian
    paulis = deque(H.terms.keys())
    paulis.popleft()
    index_list = set( x[-1][0] if x!=() else 0 for x in paulis )
    n_qubits = max(index_list) + 1 # plus 1 since total number = index+1
    return n_qubits


def QubitOperatorInfoExtracter(qubitoperators, info_index=0, mode=list):
    '''Function
    Turn QubitOperators into tuple of Pauli Strings.
    Then extract the target informations out of those Pauli Strings.

    Args:
        qubitoperators (QubitOperator 2darray): A list of OpenFermion 
                                                    QubitOperator instances.
        info_index=0 for extracting index
        info_index=1 for extracting XYZ
        mode='list' for storing each row as list
        mode='set' for storing each row as set
    Returns:
        results (obj 2darray): List contains infomation you need for each QubitOperator.

    Author(s): TsangSiuChung
    '''
    keys = [ tuple(quop.terms.keys())[0] for quop in qubitoperators ]
    for i,key in enumerate(keys):
        keys[i] = mode(map( itemgetter(info_index), key  ))
    return keys


def HamiltonianInfoExtracter(Hamiltonian, info_index=0, mode=list):
    '''Function
    Get information of Pauli strings from a Hamiltonian except the constant term.

    Author(s): TsangSiuChung
    '''
    keys = deque(Hamiltonian.terms.keys())
    keys.popleft()
    for i,key in enumerate(keys):
        keys[i] = mode(map(itemgetter(info_index), key))
    return keys


def PauliInfoExtracter(paulis, info_index=0, mode=list):
    '''Function
    Turn QubitOperators into tuple of Pauli Strings.
    Then extract the target informations out of those Pauli Strings.

    Author(s): TsangSiuChung
    '''
    keys = [ pauli for pauli in paulis if len(pauli)!=0 ]
    for i,key in enumerate(keys):
        key[i] = mode(map( itemgetter(info_index), key ))
    return result


def gz_jw2bk(pgss):
    '''Function
    Performs transformation of a set of tensor of Pauli Z operator 
    from Jordan-Wigner to Bravyi-Kitaev encoding.

    Args:
        pgss (int 2darray): A set of tensor of Pauli Z operator by Jordan-Wigner 
                            encoding that is expressed in GF2. 
                            One tensor for each row.

    Returns:
        pgss_bk (int 2darray): A set of tensor of Pauli Z operator by Bravyi-Kitaev 
                                encoding that is expressed in GF2. 
                                One tensor for each row.

    Author(s): TsangSiuChung, Takashi Tsuchimochi

    Example:
    pgss = np.array([[1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]])
    pgss_bk = gz_jw2bk(pgss)
    prints("pgss=")
    prints(pgss)
    prints("pgss_bk")
    prints(pgss_bk)
    >>> pgss=
        [[1 0 1 0 0 1 1 0 0 1 1 0 0 1]
         [0 1 0 1 0 1 0 1 0 1 0 1 0 1]
         [0 0 0 0 1 1 0 0 0 0 0 0 1 1]
         [0 0 0 0 0 0 0 0 1 1 0 0 0 0]]
        pgss_bk=
        [[1 0 1 0 1 1 1 0 1 1 1 0 1 1]
         [1 0 1 0 1 0 1 1 1 0 1 1 1 1]
         [0 0 0 0 0 1 0 0 0 0 0 0 0 1]
         [0 0 0 0 0 0 0 0 0 1 0 0 0 0]]
    '''
    from openfermion.transforms import bravyi_kitaev
    from quket.lib import FermionOperator

    width = pgss.shape[1]
    pgss_bk = []

    for each_row in pgss:
        jw_gZ = np.where(each_row == 1)[0]
        count = 0
        bk_gZ_indices = []

        for each_index in jw_gZ:
            fermion_operator_string = str(each_index)+"^ "+str(each_index)
            bk_gZ = bravyi_kitaev(FermionOperator((fermion_operator_string)))

            for terms in bk_gZ.terms:
                if len(terms) != 0:
                    temp = [0 for k in range(width)]
                    for eachZ in terms:
                        index = eachZ[0]
                        temp[index] = 1
                    bk_gZ_indices.append(temp)

        bk_gZ_indices = np.array(bk_gZ_indices, dtype=int)
        bk_gZ_indices = np.sum(bk_gZ_indices, axis=0)
        bk_gZ_indices = bk_gZ_indices % 2
        pgss_bk.append(bk_gZ_indices)
    pgss_bk = np.array(pgss_bk)
    return pgss_bk


def tablemaker(title=None, subtitle=None, y_title=None, x_title=None,
               row_vectors_name=('',), column_vectors_name=('',), row_vectors=[[0]],
               spacing=4, vectype='str', tolence=2):
    '''Function
    Print structured tabula from the input values.
    Automatically performs spacing which calculated from elements' string length. 

    Args:
        title (str): Title of the table, default as empty str.
        subtitle (str): Subtitle shown on the corner of the table, default as empty str.
        y_title (str): Title for Y axis, default as empty str.
        x_title (str): Title for X axis, default as empty str.
        row_vectors_name (1darray of str): Names for each row vector, default as empty tuple.
        column_vectors_name (1darray of str): Names for each column vector, default as empty tuple.
        row_vector (2darray): The 2darray=matrix you want to print, default as empty 2darray.
        spacing (int): Spacing of the table, default as 4.
        vectype (str): Data type of the matrix, default as 'str'.
        tolence (int): Maxium print out decimal places if the matrix is composed by numbers, default as 2.

    Returns:
        None

    Example:
        title = "AAAAAAA"
        subtitle = 'D2h'
        y_title = 'SymmOP'
        x_title = 'IRREP'
        row_vectors_name = ['C2^z', 'C2^y', 'C2^x']
        column_vectors_name = ['Ag', 'Ag', 'B1u', 'B1u']
        row_vectors = [[1, 1, 1, 1],[1, 1,-1,-1],[1, 1,-1,-1]]
        vectype = 'int'
        tablemaker(title, subtitle, y_title, x_title,
                row_vectors_name, column_vectors_name, row_vectors,
                vectype=vectype)
        >>>AAAAAAA:
        D2h | SymmOP \ IRREP  Ag  Ag B1u B1u
                C2^z           1   1   1   1
                C2^y           1   1  -1  -1
                C2^x           1   1  -1  -1
    
    Author(s): TsangSiuChung
    '''
    # Restricton on input data type
    spacing = int(spacing)
    if vectype == 'float':
        row_vectors = [[round(y,toleration) for y in x] for x in row_vectors]
    elif vectype == 'int':
        row_vectors = [[str(y) for y in x] for x in row_vectors]

    # Print table title
    if title:
        prints(f"\n{str(title)}:")

    # Prepare the information appears in the upper left corner of table
    if subtitle:
        subtitle = str(subtitle)
        subtitle = subtitle + " | "
    else:
        subtitle = ""
    if y_title:
        y_title = str(y_title)
        y_title = y_title + " \ "
    else:
        y_title = ""
    if x_title:
        x_title = str(x_title)
        x_title = x_title
    else:
        x_title = ""

    table_info = subtitle + y_title + x_title
    infol = len(table_info)
    
    # Restricton on input data type
    shape = np.array(row_vectors_name).shape
    if len(shape) == 1:
        shape = np.array(column_vectors_name).shape
        if len(shape) == 1:
            # Calculate width of the first column
            row_vectors_name = [str(x) for x in row_vectors_name]
            rowl = [len(x) for x in row_vectors_name] # row lenth
            rowl = np.append(rowl, infol)
            rowl_max = np.amax(rowl)
            row_width = ceil(rowl_max/spacing)*spacing

            # Calculate width of the other columns
            column_vectors_name = [str(x) for x in column_vectors_name]
            coll = [len(x) for x in column_vectors_name] # column length
            try:
                elel = np.amax(rowl) # element length
            except: #TypeError("""Check for the argment vectype. For numeric matrix, you should add vectype='int' or vectype='float'.""")
                pass
            else:
                coll = np.append(coll,elel)
                coll_max = np.amax(coll)
                column_width = ceil(coll_max/spacing)*spacing

                # Print the first row
                forespacel = row_width - infol
                string = " "*forespacel + str(table_info)   
                prints(string, end='')

                for name in column_vectors_name:
                    string = " "*(column_width-len(name)) + name
                    prints(string, end='')
                prints("")

                # Print the other rows
                for i in range(len(row_vectors_name)):
                    namel = len(str(row_vectors_name[i]))
                    forespacel = row_width - namel
                    string = " "*forespacel + str(row_vectors_name[i])
                    prints(string, end='')
                    for j in row_vectors[i]:
                        string = " "*(column_width-len(str(j))) + str(j)
                        prints(string, end='')
                    prints("")
                prints("")

        else:
            prints("row_vectors_name must be a 1darray.")
    else:
        prints("column_vectors_name must be a 1darray.")

        
def informative_openfermion_commutator(A,B,f=16):
    '''Function
    Informative Openfermion Commutator.
    From tolerance=0 to tolerance=15,
    check for all possibilities that two FermionOperator are commutative to each other.

    Args:
        A (QubitOperator): Operator that is created by OpenFermion.QubitOperator
        B (QubitOperator): Operator that is created by OpenFermion.QubitOperator

    Returns:
        commutativity (list): Commutativity result under different tolerance cases
        statistics (dict): Statistics of the commutativity checking result.
                            Including std, max, min, mean, median.

    Author(s): TsangSiuChung
    '''
    result = commutator(A, B)
    remainder = len(result.terms)
    # Dictionary for statistics of commutativity checking result
    stat = dict()
    # If (len(terms) != 0) := not total commutative,
    # we have to look into details
    if remainder != 0:
        # prints(result.terms.values())
        coeffs = list(result.terms.values())
        # prints(coeffs)
        coeffs_norm = list(map(abs, coeffs))
        npnorm = np.linalg.norm(coeffs_norm)

        # Populate dictionary 'stat'
        stat['norm'] = npnorm
        stat['stdeviation'] = stdev(coeffs_norm)
        stat['maxium'] = max(coeffs_norm)
        stat['minimum'] = min(coeffs_norm)
        stat['meanval'] = mean(coeffs_norm)
        stat['medianval'] = median(coeffs_norm)
        # Check commutativity represented by 1 if commute, 1 if non commute
        # to the 15th place after decimal
        # Information stored in a list
        commutativity = [int(npnorm <= eval(f"1e-{i}")) for i in range(f)]

    # Return all variables as 0 due to (len(terms) != 0) := total commutative
    else:
        # Populate dictionary 'stat
        stat['norm'] = 0
        stat['stdeviation'] = 0
        stat['maxium'] = 0
        stat['minimum'] = 0
        stat['meanval'] = 0
        stat['medianval'] = 0
        commutativity = [1 for n in range(f)]

    return commutativity, remainder, stat


def include_pgss(E, pgss, mapping='jordan_wigner'):
    '''Function
    Merge result from binary kernel and point group symmetry.
    Create a full set of symmetries for tapering off.

    Args:
        E (int 2darray): Binary kernel of Hamiltonian.
        pgss (int 2darray): Eigenvalues of point group symmetry.
        mapping (str): Mapping Method. 
    Returns:
        Epgs (int 2darray): New binary kernel.

    Author(s): TsangSiuChung
    '''
    # Escape if input is not correct
    if pgss is None or len(pgss) == 0:
        return E

    Epgs = deepcopy(E)

    if isinstance(Epgs, np.ndarray):
        pgss = np.where(np.array(pgss, dtype=int) == -1, 1, 0)
        if mapping in ('bk', 'bravyi_kitaev'):
            pgss = gz_jw2bk(pgss)
        width = Epgs.shape[1] - pgss.shape[1]
        complement = np.zeros((pgss.shape[0], width), dtype=int)
        pgss = np.hstack([complement, pgss])
        Epgs = np.vstack([pgss, Epgs])

    elif isinstance(Epgs, list):
        pgssZ = [[1 if x == -1 else 0 for x in pgs] for pgs in pgss]
        if mapping in ('bk', 'bravyi_kitaev'):
            pgssZ = gz_jw2bk(pgssZ)
        pgss = list(map(BinList2Num, pgssZ))
        Epgs += pgss

    return Epgs


def transform_state(state_old, U_list, redbits, eigvals_list, backtransform=False, reduce=True):
    """Function
    This subroutine transforms QuantumState to different represnetation,
    ruled by the unitary U. In other words, simply apply U to |state>.
    If qubits have been or are to be tapered-off, redbits and eigvals_list are needed.
    
    Forward transformation:
    U * state_old (n_qubits)  
    ->  [Tapering-off qubits based on redbits]  
    ->  state_new (n_qubits_reduced)
    
    Back transformation:
    state_old (n_qubits_reduced) 
    -> [Recovering qubits based on redbits] 
    -> U * state_old (n_qubits) 
    -> state_new (n_qubits)
    
    Args:
        state_old (QuantumState): QuantumState instance to be transformed.
        U_list (list of QubitOperator): List of QubitOperator instance to apply (has to be unitary!)
                                        in the original representation (before tapering-off).
        backtransform (bool): if true, the tapered-off qubits are recovered.
        redbits (int list): List of tapered-off qubits. If present, taper-off or recover qubits.
        eigvals_list (int list): Eigenvalues -1 or 1 used to replace X of redbits in the original representation.
    
    Returns:
        transformed_state (QuantumState): QuantumState instance transformed (normalized).
    
    """
    from quket.lib import QuantumState
    nredbits = len(U_list)
    n_qubits_old = state_old.get_qubit_count()
    reduction = redbits is not None
    if reduce:
        if backtransform:
            n_qubits_total = n_qubits_old + nredbits
            n_qubits_new = n_qubits_total
        else:
            n_qubits_total = n_qubits_old
            n_qubits_new = n_qubits_total - nredbits
    else:
        n_qubits_total = n_qubits_old        
        n_qubits_new = n_qubits_old

    state_vec_old = state_old.get_vector()
    state_vec_new = np.zeros(2**n_qubits_new, dtype=complex)

    if backtransform and reduction:
        ########################################
        #  Insert tapered-off qubits           #
        # (eigvals -> 1/sqrt(2) (|0> +- |1>))  #
        ########################################
        for ibit in range(2**n_qubits_old):
            ibit_base = ibit
            for k in redbits:
                jbit_quo = ibit_base // (2**k)
                jbit_mod = ibit_base % (2**k)
                ibit_base = jbit_quo<<k+1 | jbit_mod
            for redbit_int in range(2**(nredbits)):
                ibit_ = ibit_base
                par = 1
                for k in range(nredbits):     
                    if redbit_int & 2**k > 0:
                        par *= eigvals_list[k]
                        ibit_ += 2**redbits[k] 
                state_vec_new[ibit_] = state_vec_old[ibit] * par / (np.sqrt(2)**nredbits)
        state_tmp = QuantumState(n_qubits_new)
        state_tmp.load(state_vec_new)
    else:
        state_tmp = state_old.copy()

    ########################  
    # Transform U * state  #
    ########################    
    state_new = QuantumState(n_qubits_total)

    ### MPI ###
    U = 1
    for U_ in U_list: 
        U *= U_
    state_new.multiply_coef(0)
    k = 0
    for Pauli, coef in U.terms.items():
        if k % mpi.nprocs == mpi.rank:
            circuit = Pauli2Circuit(n_qubits_total, Pauli)
            state_ = state_tmp.copy()
            state_.multiply_coef(coef)        
            circuit.update_quantum_state(state_)
            state_new.add_state(state_)
        k += 1
    new_vector = state_new.get_vector()
    vector = mpi.allreduce(new_vector)
    del(new_vector)
    del(state_tmp)
    state_new.load(vector)


    if not backtransform:
        from quket.opelib import OpenFermionOperator2QulacsObservable
        ### Check if this state respects symmetry.
        for redbit, eigval in zip(redbits, eigvals_list):
            X = OpenFermionOperator2QulacsObservable(QubitOperator(f'X{redbit}'), n_qubits_total)
            X_expectation_value = X.get_expectation_value(state_new)
            if abs(X_expectation_value - eigval) > 1e-6:
                prints(f'<X{redbit}> = {X_expectation_value}   should be {eigval}')

            
    if not reduce:
        return state_new
    ######################
    #  Taper off qubits  #
    ######################    
    if not backtransform and (redbits is not None and eigvals_list is not None):
        state_vec = state_new.get_vector()
        red_vec = np.zeros(2**n_qubits_new, dtype=complex)
        # Reduced bits as integer 
        redbit_int = 0
        for redbit in redbits:
            redbit_int += 2**redbit
        
        for ibit in range(2**n_qubits_old):
            if abs(state_vec[ibit]) < 1e-8:
                continue
            ## Check the parity of this bit for redbits
            parity = ibit & redbit_int
            npar = bin(parity).count("1")            
            #prints(f'{npar=}')
            #
            
            # We want to remove bits from this integer 'ibit'
            # It is expected that the transformed state has the following structure: 
            #     a  |jn jn-1 ... jk+1  0  jk-1 ... j0>  +/-  a  |jn jn-1 ... jk+1  1  jk-1 ... j0> 
            # where a is the coefficient and the sign +/- is already determined by eigvals_list.
            # We then remove this bit to obtain
            #     sqrt(2) a  |jn jn-1 ... jk+1  jk-1 ... j0>
            
            # (1) Let us first determine if the bit to be reduced is 0 or 1.
            # (2) Separate the integer to two sets of integers
            #        int1 = |jn jn-1 ... jk+1>   and   int2 = |jk-1 ... j0>
            # (3) Shift int1 by one bit to right, and add to int2

             
            ibit_ = ibit
            #prints('Reducing ',bin(ibit)[2:])
            for redbit in redbits[::-1]:
                jbit_quo = ibit_ // 2**(redbit)
                redbit_val = jbit_quo % 2    # Reduced bit contained in ibit jk 
                jbit_quo = jbit_quo // 2     # Left side |jn jn-1 ... jk+1>
                jbit_mod = ibit_ % 2**redbit # Right side |jk-1 ... j0>
                #prints('int1 = ', bin(jbit_quo)[2:])
                #prints('int2 = ', bin(jbit_mod)[2:])
                # Produce a new integer eliminating the bit jk
                ibit_red = (jbit_quo)<<redbit | jbit_mod
                #prints(redbit,'-th bit reduced ->', bin(ibit_red)[2:])
                ibit_ = ibit_red
            #prints(bin(ibit_red)[2:], '  ', state_vec[ibit])
            #red_vec[ibit_red] += state_vec[ibit]  * np.sqrt(2) * (-1)**npar

            if red_vec[ibit_red] == 0:
                red_vec[ibit_red] = (state_vec[ibit] *  (np.sqrt(2))**len(redbits))
        state_new = QuantumState(n_qubits_new)
        state_new.load(red_vec)

    ### Transformation done. Just in case, normalize...
    norm=state_new.get_squared_norm()
    state_new.normalize(norm)  
    return state_new

def transform_operator(operator, clifford_operators, redundant_bits, X_eigvals, reduce=True):
    new_operator = operator
    for clifford_operator in clifford_operators:
        new_operator  =  (clifford_operator * new_operator * clifford_operator)
        new_operator.compress()
    # Clean up some 10e-7 terms especially in the case of NH3
    new_operator = purify_hamiltonian(new_operator, redundant_bits)
    # Replace redundent qubits by those coefficients 
    # and reconstruct operator to remove redundent qubits
    if reduce:
        new_operator = tapering_off_operator(new_operator, redundant_bits, 
                                                        X_eigvals, 1)
    return new_operator

def transform_pauli_list(Tapering, pauli_list, reduce=True):
    """Function
        Transform pauli_list to the tapered-off basis by using Tapering.
    """
    ndim1 = 0
    ndim2 = 0
    # List of transformed pauli operators
    new_pauli_list = []
    # List of surviving/discarded operators because of symmetry
    allowed_pauli_list = []
    ndim = len(pauli_list)
    ipos, my_ndim = mpi.myrange(ndim)
    for pauli in pauli_list[ipos:ipos+my_ndim]:
        if type(pauli) is list:
            new_pauli_list_ = []
            allowed_pauli_list_ = []
            for pauli_ in pauli:
                new_pauli_, allowed_ = transform_pauli(Tapering, pauli_, reduce)
                if new_pauli_ is not None and new_pauli_ != QubitOperator('',0):
                    new_pauli_list_.append(new_pauli_)
                    allowed_pauli_list_.append(allowed_)
            # Bug fixed. 
            allowed = len(allowed_pauli_list_) > 0 and all(allowed_pauli_list_)
            if allowed: 
                ### Quick check for commutativity
                new_pauli_list_commute = [new_pauli_list_[0]]
                k_ = 0
                for k in range(1, len(new_pauli_list_)):
                    if commutator(new_pauli_list_commute[k_], new_pauli_list_[k]) == QubitOperator('',0):
                        new_pauli_list_commute[k_] += new_pauli_list_[k]
                    else:
                        new_pauli_list_commute.append(new_pauli_list_[k])
                        k_ += 1
                new_pauli_list.append(new_pauli_list_commute)
            allowed_pauli_list.append(allowed)
        else:
            new_pauli, allowed = transform_pauli(Tapering, pauli, reduce)
            if new_pauli != QubitOperator('',0) and new_pauli is not None:
                new_pauli_list.append(new_pauli)
            allowed_pauli_list.append(allowed)
    new_pauli_list = mpi.allgather(new_pauli_list)
    allowed_pauli_list = mpi.allgather(allowed_pauli_list)
    return new_pauli_list, allowed_pauli_list


def transform_pauli(Tapering, pauli, reduce):
    """Function
        Transform pauli to the tapered-off basis by using Tapering.
    """
    # List of transformed pauli operators
    # List of surviving/discarded operators because of symmetry
    new_pauli = pauli
    for clifford_operator in Tapering.clifford_operators:
        new_pauli  =  (clifford_operator * new_pauli * clifford_operator)
        new_pauli.compress()

    # Check if the transformed operator contains invalid qubit operations
    allowed_pauli = True
    for bit in Tapering.redundant_bits:
        Ybit = "Y"+str(bit)+" "
        Zbit = "Z"+str(bit)+" "
        if Ybit in str(new_pauli) or Zbit in str(new_pauli):
            # invalid operation
            allowed_pauli = False
            return None, False
    if allowed_pauli:
        if reduce: 
            new_pauli = tapering_off_operator(new_pauli, Tapering.redundant_bits, 
                                                            Tapering.X_eigvals, 1)
    return new_pauli, allowed_pauli
