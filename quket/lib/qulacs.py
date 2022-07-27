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

"""
Qulacs' QuantumState is extended for further usability
"""
import qulacs
import copy
COEFFICIENT_TYPES = (int, float, complex)
import time

class QuantumState(qulacs.QuantumState):
    def __init__(self, n_qubits):
        super().__init__(n_qubits)
    """
    Override
    """
    def copy(self):
        state_ = QuantumState(self.get_qubit_count())
        state_.load(self)
        return state_

    def allocate_buffer(self):
        return QuantumState(self.get_qubit_count())

    """
    Useful functions
    """

    def __add__(self, state):
        if isinstance(state, (qulacs.QuantumState, QuantumState)):
            vec = self.get_vector() + state.get_vector()
            state_ = QuantumState(self.get_qubit_count())
            state_.load(vec)
            return state_
        else:
            raise TypeError('Only qulacs.QuantumState or quket.QuantumState is allowed for `+`.')

    def __iadd__(self, state):
        if isinstance(state, (qulacs.QuantumState, QuantumState)):
            self.add_state(state)
            return self
        else:
            raise TypeError('Only qulacs.QuantumState or quket.QuantumState is allowed for `+`.')

    def __sub__(self, state):
        if isinstance(state, (qulacs.QuantumState, QuantumState)):
            vec = self.get_vector() - state.get_vector()
            state_ = QuantumState(self.get_qubit_count())
            state_.load(vec)
            return state_
        else:
            raise TypeError('Only qulacs.QuantumState or quket.QuantumState is allowed for `-`.')

    def __isub__(self, state):
        if isinstance(state, (qulacs.QuantumState, QuantumState)):
            vec = self.get_vector() - state.get_vector()
            self.load(vec)
            return self
        else:
            raise TypeError('Only qulacs.QuantumState or quket.QuantumState is allowed for `-`.')

    def __mul__(self, a):
        if isinstance(a, COEFFICIENT_TYPES):
            state_ = self.copy()
            state_.multiply_coef(a)
            return state_
        else:
            raise TypeError('Only scalar values are allowed for `*`.')
            

    def __rmul__(self, a):
        if isinstance(a, COEFFICIENT_TYPES):     
            state_ = self.copy()
            state_.multiply_coef(a)
            return state_
        else:
            raise TypeError('Only scalar values are allowed for `*`.')

    def __imul__(self, a):
        if isinstance(a, COEFFICIENT_TYPES):
            self.multiply_coef(a)
            return self
        else:
            raise TypeError('Only scalar values are allowed for `*`.')

    def __truediv__(self, a):
        if isinstance(a, COEFFICIENT_TYPES):
            state_ = self.copy()
            state_.multiply_coef(1/a)
            return state_
        else:
            raise TypeError('Only scalar values are allowed for `/`.')

    def __itruediv__(self, a):
        if isinstance(a, COEFFICIENT_TYPES):
            self.multiply_coef(1/a)
            return self
        else:
            raise TypeError('Only scalar values are allowed for `/`.')
