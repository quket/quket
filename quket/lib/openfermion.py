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
# See the License for the specific language governing permissions and
# limitations under the License.
import openfermion
import qulacs
import copy
COEFFICIENT_TYPES = (int, float, complex)
from .qulacs import QuantumState
"""
OpenFermion's QubitOperator and FermionOperator are extended so that 
they can be directly multiplied to (extended) QuantumState class
"""

###                        ###
###   Override the method  ###
###                        ###
class QubitOperator(openfermion.QubitOperator):
    def __init__(self, term=None, coefficient=1.):
        super().__init__(term, coefficient)
    def __mul__(self, multiplier):
        """Return self * multiplier for a scalar, or a SymbolicOperator.

        Args:
            multiplier: A scalar, or a SymbolicOperator, or QuantumState.

        Returns:
            product (SymbolicOperator) or QuantumState

        Raises:
            TypeError: Invalid type cannot be multiply with SymbolicOperator.
        """
        if isinstance(multiplier, COEFFICIENT_TYPES + (type(self),)):
            product = copy.deepcopy(self)
            product *= multiplier
            return product
        elif isinstance(multiplier, (qulacs.QuantumState, QuantumState)):
            from quket.opelib.excitation import evolve
            state = evolve(self, multiplier)
            return state
        else:
            raise TypeError('Object of invalid type cannot multiply with ' +
                            type(self) + '.')

class FermionOperator(openfermion.FermionOperator):
    def __init__(self, term=None, coefficient=1.):
        super().__init__(term, coefficient)
    def __mul__(self, multiplier):
        """Return self * multiplier for a scalar, or a SymbolicOperator.

        Args:
            multiplier: A scalar, or a SymbolicOperator, or QuantumState (ONLY JORDAN-WIGNER PICTURE).

        Returns:
            product (SymbolicOperator) or QuantumState

        Raises:
            TypeError: Invalid type cannot be multiply with SymbolicOperator.
        """
        if isinstance(multiplier, COEFFICIENT_TYPES + (type(self),)):
            product = copy.deepcopy(self)
            product *= multiplier
            return product
        elif isinstance(multiplier, (qulacs.QuantumState, QuantumState)):
            from quket.opelib.excitation import evolve
            state = evolve(self, multiplier, mapping='jw')
            return state
        else:
            raise TypeError('Object of invalid type cannot multiply with ' +
                            type(self) + '.')


###                        ###
###   Override the class   ###
###                        ###
def jordan_wigner(operator):
    qubit_operator = openfermion.jordan_wigner(operator)
    qubit_operator.__class__ = QubitOperator
    return qubit_operator

def reverse_jordan_wigner(operator, n_qubits=None):
    fermion_operator = openfermion.reverse_jordan_wigner(operator, n_qubits=n_qubits)
    fermion_operator.__class__ = FermionOperator
    return fermion_operator

def bravyi_kitaev(operator, n_qubits=None):
    qubit_operator = openfermion.bravyi_kitaev(operator, n_qubits=n_qubits)
    qubit_operator.__class__ = QubitOperator
    return qubit_operator

def get_fermion_operator(operator):
    fermion_operator = openfermion.get_fermion_operator(operator)
    fermion_operator.__class__ = FermionOperator
    return fermion_operator

def commutator(operator_a, operator_b):
    operator_c = openfermion.commutator(operator_a, operator_b)
    operator_c.__class__ = operator_a.__class__
    return operator_c

def s_squared_operator(n_spatial_orbitals):
    operator = openfermion.s_squared_operator(n_spatial_orbitals)
    operator.__class__ = FermionOperator
    return operator

def number_operator(n_modes, mode = None, coefficient=1.0, parity = -1):
    operator = openfermion.number_operator(n_modes, mode=mode, coefficient=coefficient, parity=parity)
    if isinstance(operator, openfermion.FermionOperator): 
        operator.__class__ = FermionOperator
    return operator

def normal_ordered(operator, hbar=1.0):
    operator_ = openfermion.normal_ordered(operator)
    operator_.__class__ = operator.__class__
    return operator_

def hermitian_conjugated(operator):
    operator_ = openfermion.hermitian_conjugated(operator)
    operator_.__class__ = operator.__class__ 
    return operator_
    
