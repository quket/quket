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
from dataclasses import dataclass, field, InitVar


from quket import config as cf
from quket.fileio import error, prints
from quket.lib import QubitOperator


@dataclass
class Heisenberg():
    """Heisenberg model class.

    Attributes:
        nspin (int): Number of spin.
        n_qubits (int): Number of qubits.

    Author(s): Yuma Shimomoto
    """
    # Note; rename 'n_orbitals' to 'n_active_orbitals' when read input file.
    n_active_orbitals: InitVar[int] = None

    basis: str = "lr-heisenberg"
    n_orbitals: int = None

    nspin: int = field(init=False)
    n_qubits: int = field(init=False)

    def __post_init__(self, n_active_orbitals, *args, **kwds):
        if n_active_orbitals is None:
            error("'n_orbitals' is None.")
        if n_active_orbitals <= 0:
            error("# orbitals <= 0!")

        self.nspin = self.n_qubits = self.n_orbitals = n_active_orbitals

    @property
    def n_frozen_orbitals(self):
        return 0

    @property
    def n_core_orbitals(self):
        return 0

    @property
    def n_active_orbitals(self):
        return self.n_orbitals

    @n_active_orbitals.setter
    def n_active_orbitals(self, value):
        if cf.debug:
            prints(f"Claim that 'n_active_orbitals' is changed "
                   f"from {self.n_orbitals} to {value}.")
        self.n_orbitals = value

    @property
    def n_secondary_orbitals(self):
        return 0

    @property
    def nf(self):
        return self.n_frozen_orbitals

    @property
    def nc(self):
        return self.n_core_orbitals

    @property
    def na(self):
        return self.n_active_orbitals

    @na.setter
    def na(self, value):
        self.n_active_orbitals = value

    @property
    def ns(self):
        return self.n_secondary_orbitals

    def get_operators(self):
        sx = []
        sy = []
        sz = []
        for i in range(self.nspin):
            sx.append(QubitOperator(f"X{i}"))
            sy.append(QubitOperator(f"Y{i}"))
            sz.append(QubitOperator(f"Z{i}"))

        qubit_Hamiltonian = 0*QubitOperator("")
        if "lr" in self.basis:
            for i in range(self.nspin):
                j = (i+1)%self.nspin
                qubit_Hamiltonian += 0.5*(sx[i]*sx[j]
                                       + sy[i]*sy[j]
                                       + sz[i]*sz[j])
            for i in range(2):
                j = i+2
                qubit_Hamiltonian += 1./3.*(sx[i]*sx[j]
                                         + sy[i]*sy[j]
                                         + sz[i]*sz[j])
        else:
            for i in range(self.nspin):
                j = (i+1)%self.nspin
                qubit_Hamiltonian += sx[i]*sx[j] + sy[i]*sy[j] + sz[i]*sz[j]
        return qubit_Hamiltonian
