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

opelib.py

Library for operators.

"""
import time
import numpy as np
import openfermion
from qulacs import QuantumCircuit
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_text
from qulacs.state import inner_product
from openfermion.ops.representations.interaction_operator import InteractionOperator

from quket.fileio import error, prints, print_state
from .circuit import Gdouble_ope, create_exp_state, single_ope_Pauli
from .excitation import evolve
import quket.config as cf
from quket.lib import (
    QuantumState,
    FermionOperator,
    QubitOperator,
    jordan_wigner,
    bravyi_kitaev,
    normal_ordered,
    commutator,
    hermitian_conjugated
    )

def create_1body_operator(XA,
                          XB=None, const=0, mo_coeff=None, ao=False, n_active_orbitals=None):
    """Function
    Given XA (=XB) as a (n_orbitals x n_orbitals) matrix,
    return FermionOperator in OpenFermion Format.
    For active-space calculations, zero electron part (const) may be added.

    If ao is True, XA is ao basis.

    Author(s): Takashi Tsuchimochi
    """
    moA = XA.copy()
    if XB is not None:
        moB = XB.copy()
    core = const
    if n_active_orbitals is None:
        n_active_orbitals = moA.shape[0]
    if ao:
        ### XA is in AO basis. Transform to MO.
        n_core_orbitals = moA.shape[0] - n_active_orbitals
        moA = mo_coeff.T@moA@mo_coeff
        core = np.sum([moA[i, i] for i in range(n_core_orbitals)])
        core = 2* np.sum([moA[i, i] for i in range(n_core_orbitals)])
        moA = moA[n_core_orbitals:, n_core_orbitals:]
        if XB is not None:
            ### XB is in AO basis. Transform to MO.
            moB = mo_coeff.T@moB@mo_coeff
            moB = moB[n_core_orbitals:, n_core_orbitals:]

    ### XA (and XB) is in MO basis.
    Operator = FermionOperator("", core)
    for i in range(2*n_active_orbitals):
        for j in range(2*n_active_orbitals):
            string = f"{j}^ {i}"
            ii = i//2
            jj = j//2
            if i%2 == 0 and j%2 == 0:  # Alpha-Alpha
                Operator += FermionOperator(string, moA[jj, ii])
            elif i%2 == 1 and j%2 == 1:  # Beta-Beta
                if XB is None:
                    Operator += FermionOperator(string, moA[jj, ii])
                else:
                    Operator += FermionOperator(string, moB[jj, ii])
                    #error("Currently, UHF basis is not supported "
                    #      "in create_1body_operator.")
    return Operator


def OpenFermionOperator2QulacsObservable(operator, n_qubits, mapping=None):
    """
    Create qulacs observable from Openfermion Fermion/Qubit Operator.

    Author(s): Masaki Taii, Takashi Tsuchimochi, Yuma Shimomoto
    """
    if isinstance(operator, (openfermion.FermionOperator, FermionOperator)):
        if mapping in ("jw", "jordan_wigner"):
            str_qubit = str(jordan_wigner(operator))
        elif mapping in ("bk", "bravyi_kitaev"):
            str_qubit = str(bravyi_kitaev(operator, n_qubits))
        else:
            raise ValueError(f'Incorrect mapping = {mapping}')
    else:
        str_qubit = str(operator)
    string = f"(0.0000000000000000+0j) [Z{n_qubits-1}]"
    if str_qubit == "0":
        str_qubit = string
    else:
        str_qubit += f" + \n{string}"
    return create_observable_from_openfermion_text(str_qubit)


def OpenFermionOperator2QulacsGeneralOperator(operator, n_qubits, mapping=None):
    """Function
    Create qulacs general operator from OpenFermion FermionOperator `operator`.

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    if isinstance(operator, (openfermion.FermionOperator, FermionOperator)):
        if mapping in ("jw", "jordan_wigner"):
            str_qubit = str(jordan_wigner(operator))
        elif mapping in ("bk", "bravyi_kitaev"):
            str_qubit = str(bravyi_kitaev(operator, n_qubits))
        else:
            raise ValueError(f'Incorrect mapping = {mapping}')
    else:
        str_qubit = str(operator)
    string = f"(0.0000000000000000+0j) [Z{n_qubits-1}]"
    if str_qubit == "0":
        str_qubit = string
    else:
        str_qubit += f" + \n{string}"
    return create_quantum_operator_from_openfermion_text(str_qubit)

def Separate_Fermionic_Hamiltonian(fermionic_hamiltonian, state):
    """Function
    Separate two-body fermionic hamiltonian into zero-, one-, and two-body components
    (with zero-body meaning just a constant).

    Args:
        fermionic_hamiltonian (InteractionOperator): full Hamiltonian  H0 + H1 + H2
    Returns:
        zero_body (InteractionOperator): H0
        one_body (InteractionOperator): H1
        two_body (InteractionOperator): H2
    """
    n_qubits = state.get_qubit_count()
    zero_body = InteractionOperator
    zero_body = zero_body.zero(n_qubits)
    zero_body.n_body_tensors[()] = fermionic_hamiltonian.n_body_tensors[()]
    
    one_body = InteractionOperator
    one_body = one_body.zero(n_qubits)
    one_body.n_body_tensors[(1, 0)] = fermionic_hamiltonian.n_body_tensors[(1,0)]
    
    two_body = InteractionOperator
    two_body = two_body.zero(n_qubits)
    two_body.n_body_tensors[(1, 1, 0, 0)] = fermionic_hamiltonian.n_body_tensors[(1,1,0,0)]

    return zero_body, one_body, two_body
    

def Orthonormalize(state0, state1, normalize=True):
    """Function
    Project out state 0 from state 1
    |1>  <= (1 - |0><0| ) |1>

    |1> is renormalized.

    Author(s): Takashi Tsuchimochi
    """
    S01 = inner_product(state0, state1)

    tmp = state0.copy()
    tmp.multiply_coef(-S01)
    state1.add_state(tmp)
    if normalize:
        # Normalize
        norm2 = state1.get_squared_norm()
        state1.normalize(norm2)



def single_operator_gradient(a, i, state, qubit_Hamiltonian, numerical=True, debug=False, mapping='jordan_wigner'):
    """Function
    Compute gradient <state| [H,(a^ i - i^ a)] |state>

    Args:
        a (int): Qubit label
        i (int): Qubit label
        state (QuantumState): Quantum state to be used
        qubit_Hamiltonian (QubitOperator): Jordan_Wigner transformed Hamiltonian
    """
    # Analytical
    n_qubits = state.get_qubit_count()
    fermi = FermionOperator(f"{a}^ {i}", 1.) + FermionOperator(f"{i}^ {a}", -1.)
    if mapping in ("jw", "jordan_wigner"):
        qubit_fermi = jordan_wigner(fermi)
    elif mapping in ("bk", "bravyi_kitaev"):
        qubit_fermi = bravyi_kitaev(fermi, n_qubits)
    else:
        raise ValueError(f'Incorrect mapping = {mapping}')
    Hfermi = commutator(qubit_Hamiltonian, qubit_fermi)
    observable_gradient = OpenFermionOperator2QulacsObservable(Hfermi, n_qubits)
    # Calculate gradient of energy using observable
    gradient = observable_gradient.get_expectation_value(state)

    if debug:
        ham = create_observable_from_openfermion_text(str(qubit_Hamiltonian))
        E0 = ham.get_expectation_value(state)
        test_state = state.copy()
        test_circuit = QuantumCircuit(n_qubits)
        single_ope_Pauli(a, i, test_circuit, 2e-6)
        test_circuit.update_quantum_state(test_state)
        Ep = ham.get_expectation_value(test_state)
        num_grad = (Ep-E0)/2e-6
        prints(f"analytical(a={a}, i={i}): {gradient:+4.4E}")
        prints(f"numerical (a={a}, i={i}): {num_grad:+4.4E}")
        if gradient - num_grad > 1e-6:
            prints( "--------------------there is something wrong!----------------------")
            exit()
    return gradient


def double_operator_gradient(b, a, j, i, state, qubit_Hamiltonian, debug=False, mapping='jordan_wigner'):
    """Function
    Compute gradient <state| [H,(b^ a^ j i - i^ j^ a b)] |state>

    Args:
        b (int): Qubit label
        a (int): Qubit label
        j (int): Qubit label
        i (int): Qubit label
        state (QuantumState): Quantum state to be used
        qubit_Hamiltonian (QubitOperator): Jordan_Wigner transformed Hamiltonian
    """

    n_qubits = state.get_qubit_count()
    fermi = (FermionOperator(f"{b}^ {a}^ {j} {i}", 1.)
             + FermionOperator(f"{i}^ {j}^ {a} {b}", -1.))
    if mapping in ("jw", "jordan_wigner"):
        qubit_fermi = jordan_wigner(fermi)
    elif mapping in ("bk", "bravyi_kitaev"):
        qubit_fermi = bravyi_kitaev(fermi, n_qubits)
    else:
        raise ValueError(f'Incorrect mapping = {mapping}')
    Hfermi = commutator(qubit_Hamiltonian, qubit_fermi)
    observable_gradient = OpenFermionOperator2QulacsObservable(Hfermi, n_qubits, mapping=mapping)
    t0 = time.time()
    gradient = observable_gradient.get_expectation_value(state)
    #prints(time.time() - t0)
    if debug:  # for debag
        ham = create_observable_from_openfermion_text(str(qubit_Hamiltonian))
        E0 = ham.get_expectation_value(state)
        test_state = state.copy()
        test_circuit = QuantumCircuit(n_qubits)
        Gdouble_ope(b, a, j, i, test_circuit, 1e-8)
        test_circuit.update_quantum_state(test_state)
        Ep = ham.get_expectation_value(test_state)
        num_grad = (Ep-E0)/1e-8
        prints(f"analytical(b={b}, a={a}, j={j}, i={i}): {gradient:+4.4E}")
        prints(f"numerical (b={b}, a={a}, j={j}, i={i}): {num_grad:+4.4E}")
        #if gradient - num_grad > 1e-8:
        #    prints("--------------------there is something wrong!----------------------")
        #    exit()
    return gradient

def spin_double_grad(b, a, j, i, state, qubit_Hamiltonian , Quket, bs_orbitals=None, numerical=True, H_state=None):
    """Function
    Compute derivative d < e(-theta A) HP exp(theta A) > / d theta
    where A is a double excitation operator. For spin-projection, P is taken into account.
    For non-spin-projection, P is set to 1.
    Numerical or analytical (<[HP,A]>) but the former is faster.
    """
    from quket.projection import S2Proj
    
    t0 = time.time()
    n_qubits = Quket.n_qubits
    mapping = Quket.cf.mapping

    if numerical:
        # Numerical (faster and accurate)
        state_p = state.copy()
        state_m = state.copy()
        #
        double_excitation_circuit = QuantumCircuit(n_qubits)
        Gdouble_ope( b , a , j , i , double_excitation_circuit , 1e-6 )
        double_excitation_circuit.update_quantum_state( state_p )
        double_excitation_circuit = QuantumCircuit(n_qubits)
        Gdouble_ope( b , a , j , i , double_excitation_circuit ,-1e-6 )
        double_excitation_circuit.update_quantum_state( state_m )
        #
        if Quket.projection.SpinProj:
            stateP_p     = S2Proj( Quket, state_p )
            stateP_m     = S2Proj( Quket, state_m )
        else:
            stateP_p = state_p
            stateP_m = state_m
        #
        Ep = Quket.qulacs.Hamiltonian.get_expectation_value(stateP_p)
        Em = Quket.qulacs.Hamiltonian.get_expectation_value(stateP_m)

        gradient = (Ep-Em)/2e-6
        #prints('Numerical ',  time.time() - t0,  ' grad ',num_grad)

        ## Numerical (old)
        #non_operated_state = state.copy()
        #operated_state     = state.copy()
        #
        #double_excitation_circuit = QuantumCircuit(n_qubits)
        #Gdouble_ope( b , a , j , i , double_excitation_circuit , 2e-6 )
        #double_excitation_circuit.update_quantum_state( operated_state )
        #
        #if Quket.projection.SpinProj:
        #    non_operated_state = S2Proj( Quket, non_operated_state )
        #    operated_state     = S2Proj( Quket,     operated_state )
        #
        #E0 = Quket.qulacs.Hamiltonian.get_expectation_value( non_operated_state )
        #E1 = Quket.qulacs.Hamiltonian.get_expectation_value(     operated_state )
        #num_grad=(E1-E0)/2e-6
        #prints('Numerical ',  time.time() - t0,  ' grad ',num_grad)

    else:
        if H_state is not None:
            # Analytical (accurate and fast)
            # H_state = H|state>   or   (H-E0)P|state>    
            # is at hand  
            # Create Ak |state> by evolve(Ak, state)
            # grad = <state| [H, Ak] |state> =  2 <H_state|Ak_state>
            # grad = <state| [(H-E0)P, Ak] |state> = 2 <H_state|Ak_state>  
            #  
            Ak_fermi = FermionOperator( str( b ) +'^ '+ str( a ) +'^ '+ str( j ) +' '+ str( i ) , 1.0)\
                      +FermionOperator( str( i ) +'^ '+ str( j ) +'^ '+ str( a ) +' '+ str( b ) ,-1.0)
            Ak_state = evolve(Ak_fermi, state)
            gradient = 2 * inner_product(H_state, Ak_state).real
#            if abs(inner_product(H_state, state))  > 1e-6:
#                print(Ak_fermi)
#                raise ValueError('<(H-E)P> has to be zero')
        else:
            # Analytical (accurate but slow)
            E0 = Quket.energy
            # -2 <Phi| Ak HP |Phi> / <Phi | P | Phi>   where P|Phi> is normalized
            Ak_fermi = FermionOperator( str( b ) +'^ '+ str( a ) +'^ '+ str( j ) +' '+ str( i ) , 1.0)\
                      +FermionOperator( str( i ) +'^ '+ str( j ) +'^ '+ str( a ) +' '+ str( b ) ,-1.0)
            if mapping in ("jw", "jordan_wigner"):
                Ak_qubit = jordan_wigner( Ak_fermi ) 
            elif mapping in ("bk", "bravyi_kitaev"):
                Ak_qubit = bravyi_kitaev( Ak_fermi, n_qubits ) 
            else:
                raise ValueError(f'Incorrect mapping = {mapping}')
            AkH_qubit = Ak_qubit * (qubit_Hamiltonian -  E0) 
            AkH_observable = OpenFermionOperator2QulacsGeneralOperator(AkH_qubit, n_qubits, mapping=mapping)

            if Quket.projection.SpinProj:
                state_P = S2Proj( Quket , state, bs_orbitals=bs_orbitals )
            else:
                state_P = state

            t0 = time.time()
            gradient = -2.0*AkH_observable.get_transition_amplitude(state,state_P) / inner_product(state,state_P)
            gradient=gradient.real

        #prints('Analytical ',  time.time() - t0,  ' grad ',gradient)
    
    #if cf.debug and abs( gradient ) - abs( num_grad ) > 1e-6:
    #    prints("|numerical|  = ", abs( num_grad ) )
    #    prints("|analytical| = ", abs( gradient ) )
    #    prints("diff of grad = ", abs( gradient ) - abs(num_grad) )
    #    prints("--------------------there is someting wrong! in double----------------------")

    return gradient

def spin_single_grad( a , i , state , qubit_Hamiltonian, Quket, bs_orbitals=None, numerical=True, H_state=None):
    """Function
    Compute derivative d < e(-theta A) HP exp(theta A) > / d theta
    where A is a single excitation operator. For spin-projection, P is taken into account.
    For non-spin-projection, P is set to 1.
    Numerical or analytical (<[HP,A]>) but the former is faster.
    """
    from quket.projection import S2Proj
    
    n_qubits = Quket.n_qubits
    mapping = Quket.cf.mapping
    #ham=create_observable_from_openfermion_text(str(jordan_wigner_hamiltonian))
    
    if numerical:
        # Numerical (faster and accurate)
        state_p = state.copy()
        state_m = state.copy()
        #
        single_excitation_circuit = QuantumCircuit(n_qubits)
        single_ope_Pauli( a , i , single_excitation_circuit , 1e-6 )
        single_excitation_circuit.update_quantum_state( state_p )
        single_excitation_circuit = QuantumCircuit(n_qubits)
        single_ope_Pauli( a , i , single_excitation_circuit ,-1e-6 )
        single_excitation_circuit.update_quantum_state( state_m )
        #
        if Quket.projection.SpinProj:
            stateP_p     = S2Proj( Quket, state_p )
            stateP_m     = S2Proj( Quket, state_m )
        else:
            stateP_p = state_p
            stateP_m = state_m
        #
        Ep = Quket.qulacs.Hamiltonian.get_expectation_value(stateP_p)
        Em = Quket.qulacs.Hamiltonian.get_expectation_value(stateP_m)

        gradient = (Ep-Em)/2e-6
        #prints('Numerical ',  time.time() - t0,  ' grad ',num_grad)

        # Numerical (old)
        #non_operated_state = state.copy()
        #operated_state     = state.copy()
        #
        #single_excitation_circuit = QuantumCircuit(n_qubit)
        #single_ope_Pauli( a , i , single_excitation_circuit , 2e-6 )
        #single_excitation_circuit.update_quantum_state( operated_state )
        #
        #non_operated_state = S2Proj( Quket, non_operated_state )
        #operated_state     = S2Proj( Quket,     operated_state )

        #E0 = ham.get_expectation_value( non_operated_state )
        #E1 = ham.get_expectation_value(     operated_state )

        #num_grad=( E1 - E0 )/2e-6
        
    else:
        if H_state is not None:
            # Analytical (accurate and fast)
            # H_state = H|state>   or   (H-E0)P|state>    
            # is at hand  
            # Create Ak |state> by evolve(Ak, state)
            # grad = <state| [H, Ak] |state> =  2 <H_state|Ak_state>
            # grad = <state| [(H-E0)P, Ak] |state> = 2 <H_state|Ak_state>  
            #  
            Ak_fermi = FermionOperator( str( a ) +'^ ' + str( i ) ,  1.0 )\
                      +FermionOperator( str( i ) +'^ ' + str( a ) , -1.0 )
            Ak_state = evolve(Ak_fermi, state)
            gradient = 2 * inner_product(H_state, Ak_state).real
        else:
            # Analytical (accurate but slow)
            E0 = Quket.energy
            Ak_fermi = FermionOperator( str( a ) +'^ ' + str( i ) ,  1.0 )\
                      +FermionOperator( str( i ) +'^ ' + str( a ) , -1.0 )
            Ak_qubit = jordan_wigner( Ak_fermi )
            if mapping in ("jw", "jordan_wigner"):
                Ak_qubit = jordan_wigner( Ak_fermi ) 
            elif mapping in ("bk", "bravyi_kitaev"):
                Ak_qubit = bravyi_kitaev( Ak_fermi, n_qubits ) 
            else:
                raise ValueError(f'Incorrect mapping = {mapping}')
            AkH_qubit = Ak_qubit * (qubit_Hamiltonian -  E0) 
            AkH_observable = OpenFermionOperator2QulacsGeneralOperator(AkH_qubit, n_qubits, mapping=mapping)

            if Quket.projection.SpinProj:
                state_P = S2Proj( Quket , state, bs_orbitals=bs_orbitals )
            else:
                state_P = state

            gradient = -2.0*AkH_observable.get_transition_amplitude(state,state_P) / inner_product(state,state_P)
    
            gradient=gradient.real
    
    #if cf.debug and abs( gradient ) - abs( num_grad ) > 1e-6:
    #    prints("|numerical|  = ", abs( num_grad ) )
    #    prints("|analytical| = ", abs( gradient ) )
    #    prints("diff of grad = ", abs( gradient ) - abs(num_grad) )
    #    prints("--------------------there is someting wrong! in single----------------------")

    return gradient

def pauli_grad(pauli, state , qubit_Hamiltonian, Quket, bs_orbitals=None, numerical=True, H_state=None):
    """Function
    Compute derivative d < e(-theta A) HP exp(theta A) > / d theta
    where A is a single excitation operator. For spin-projection, P is taken into account.
    For non-spin-projection, P is set to 1.
    Numerical or analytical (<[HP,A]>) but the former is faster.
    """
    from quket.projection import S2Proj
    
    n_qubits = Quket.n_qubits
    #ham=create_observable_from_openfermion_text(str(jordan_wigner_hamiltonian))
    
    if numerical:
        # Numerical (slower)
        #
        state_p = create_exp_state(Quket, init_state=state, pauli_list=[pauli], theta_list=[1e-6], rho=1)
        state_m = create_exp_state(Quket, init_state=state, pauli_list=[pauli], theta_list=[-1e-6], rho=1)
        #
        if Quket.projection.SpinProj:
            stateP_p     = S2Proj( Quket, state_p, normalize=False)
            stateP_m     = S2Proj( Quket, state_m, normalize=False )
            HstateP_p    = evolve(qubit_Hamiltonian, state_p)
            HstateP_m    = evolve(qubit_Hamiltonian, state_m)

            Ep = inner_product(state_p, HstateP_p).real / inner_product(state_p, stateP_p).real
            Em = inner_product(state_m, HstateP_m).real / inner_product(state_m, stateP_m).real
        else:
            stateP_p = state_p
            stateP_m = state_m
        #
            Ep = Quket.qulacs.Hamiltonian.get_expectation_value(stateP_p)
            Em = Quket.qulacs.Hamiltonian.get_expectation_value(stateP_m)

        if Quket.constraint_lambda_Sz > 0:
            Sz2state = evolve(Quket.operators.Sz, state_p, parallel=False) ## Sz|Phi>
            Sz2state = evolve(Quket.operators.Sz, Sz2state, parallel=False) ## Sz|Phi>
            penalty = Quket.constraint_lambda_Sz * (inner_product(state_p, Sz2state)).real ## <phi|Sz|phi> as penalty 
            Ep += penalty 
            Sz2state = evolve(Quket.operators.Sz, state_m, parallel=False) ## Sz|Phi>
            Sz2state = evolve(Quket.operators.Sz, Sz2state, parallel=False) ## Sz|Phi>
            penalty = Quket.constraint_lambda_Sz * (inner_product(state_m, Sz2state)).real ## <phi|Sz|phi> as penalty 
            Em += penalty 

        gradient = (Ep-Em)/2e-6

    else:
        if H_state is not None:
            # Analytical (accurate and fast)
            # H_state = H|state>   or   (H-E0)P|state>    
            # is at hand  
            # Create Ak |state> by evolve(Ak, state)
            # grad = <state| [H, Ak] |state> =  2 <H_state|Ak_state>
            # grad = <state| [(H-E0)P, Ak] |state> = 2 <H_state|Ak_state>  
            #  
            Ak_state = evolve(pauli, state)
            #print_state(Ak_state, name=f'{pauli}', threshold=1e-6)
            gradient = 2 * inner_product(H_state, Ak_state).real

#            if Quket.constraint_lambda_Sz > 0:
#                Sz2state = evolve(Quket.operators.Sz, state, parallel=False) ## Sz|Phi>
#                Sz2state = evolve(Quket.operators.Sz, Sz2state, parallel=False) ## Sz|Phi>
#                ## Ms2 = <Phi|Sz**2|Phi>
#                Ms2 = inner_product(state, Sz2state).real
#                ## lambda (Sz**2 - Ms2)|Phi>
#                state_ = state.copy()
#                state_.multiply_coef(-Ms2)
#                Sz2state.add_state(state_)
#                Sz2state.multiply_coef(Quket.constraint_lambda_Sz)
#                penalty = 2* Quket.constraint_lambda_Sz * (inner_product(Ak_state, Sz2state)).real ## <phi|Sz|phi> as penalty 
#                prints(pauli, penalty)
#                gradient += penalty
        else:
            error()
    
    #if cf.debug and abs( gradient ) - abs( num_grad ) > 1e-6:
    #    prints("|numerical|  = ", abs( num_grad ) )
    #    prints("|analytical| = ", abs( gradient ) )
    #    prints("diff of grad = ", abs( gradient ) - abs(num_grad) )
    #    prints("--------------------there is someting wrong! in single----------------------")

    return gradient

def Decompose_expT1_to_Givens_rotations(T1, eps=1e-12):
    """
    Given a skew matrix T1 for orbital rotation, which defines
       \hat T1 = \sum_{p>q}  Kpq (a_p^\dag a_q  - a_q^\dag a_p)
    we would like to decompose its exponential and obtain 
       exp(\hat T1) = \hat u1(theta1) \hat u2(theta2) ... \hat uM(thetaM)
    where each \hat u is a Givens rotation,
       u(theta) = exp( theta (a_i^\dag a_j  - a_j^\dag a_i) )

    Arg(s):
        T1 (np.ndarray): squared, skew matrix that corresponds to the T1 amplitude (within the active-space)
        eps (float): threshold to terminate 

    Return(s):
        pauli_list (list): A list of qubit operators
        theta_list (list): A list of theta

    Author(s): Takashi Tsuchimochi
    """
    from quket.linalg.linalg import _Decompose_expK_to_Givens_rotations
    pauli_list = []
    theta_list = []
    u_list = _Decompose_expK_to_Givens_rotations(T1, eps=eps)
    for i, j, theta in u_list:
        iA = i*2
        jA = j*2
        iB = iA + 1
        jB = jA + 1
        fop = FermionOperator(f'{iA}^ {jA}') + FermionOperator(f'{iB}^ {jB}')
        fop = fop - hermitian_conjugated(fop)
        qubit_op = jordan_wigner(fop)
        pauli_list.append(qubit_op)
        theta_list.append(theta)

    return pauli_list, theta_list


if __name__ == '__main__':
    vec = [0,0,0,0.5,0,0,0.5,0,0,0.5,0,0,0.5,0,0,0]
    state = QuantumState(4)
    state.load(vec)
    print_state(state, name='test state')

    op00 = FermionOperator('0^ 0')
    op20 = FermionOperator('2^ 0')
    op22 = FermionOperator('2^ 2')
    
    op00_obs = OpenFermionOperator2QulacsObservable(op00,4)
    op20_gen = OpenFermionOperator2QulacsGeneralOperator(op20,4)
    op22_obs = OpenFermionOperator2QulacsObservable(op22,4)

    D00 = op00_obs.get_expectation_value(state)
    D20 = op20_gen.get_expectation_value(state)
    D22 = op22_obs.get_expectation_value(state)
    D = [[D00, D20], [D20, D22]]
    print('Density matrix ', D)


if __name__ == '__main__':
    vec = [0,0,0,0.5,0,0,0.5,0,0,0.5,0,0,0.5,0,0,0]
    state = QuantumState(4)
    state.load(vec)
    print_state(state, name='test state')

    op00 = FermionOperator('0^ 0')
    op20 = FermionOperator('2^ 0')
    op22 = FermionOperator('2^ 2')
    
    op00_obs = OpenFermionOperator2QulacsObservable(op00,4)
    op20_gen = OpenFermionOperator2QulacsGeneralOperator(op20,4)
    op22_obs = OpenFermionOperator2QulacsObservable(op22,4)

    D00 = op00_obs.get_expectation_value(state)
    D20 = op20_gen.get_expectation_value(state)
    D22 = op22_obs.get_expectation_value(state)
    D = [[D00, D20], [D20, D22]]
    print('Density matrix ', D)
