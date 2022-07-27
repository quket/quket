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
from typing import List, Dict
from dataclasses import dataclass, field, InitVar

import numpy as np
from numpy import ndarray
try:
    from openfermion.hamiltonians import MolecularData
except:
    from openfermion.chem import MolecularData
from openfermion.hamiltonians import fermi_hubbard


from quket.mpilib import mpilib as mpi
from quket import config as cf
from quket.fileio import prints, error
from quket.lib import number_operator, s_squared_operator


@dataclass
class Hubbard():
    """Hubbard model class.

    Attributes:
        hubbard_u (float): Hubbard U
        hubbard_nx (int): Number of hubbard sites for x-axis.
        hubbard_ny (int): Number of hubbard sites for y-axis.
        natom (int): Number of atoms.
        n_orbitals (int): Number of spatial orbitals.

    Author(s): Yuma Shhimomoto
    """
    # Note; rename 'n_orbitals' to 'n_active_orbitals' when read input file.
    n_active_electrons: InitVar[int] = None

    basis: str = "hubbard"
    multiplicity: int = 1
    Ms: int = None
    hubbard_u: float = None   
    hubbard_nx: int = None
    hubbard_ny: int = 1
    n_electrons: int = None
    hubbard_ao: bool = True
    from_vir: bool = False
    n_frozen_orbitals: int = 0
    n_secondary_orbitals: int = 0
    n_core_orbitals: int = 0
    n_frozenv_orbitals: int = 0

    natom: int = field(init=False)
    n_orbitals: int = field(init=False)

    n_qubits: int = field(init=False, default=None)
    include: Dict = field(default_factory=dict)

    def __post_init__(self, n_active_electrons, *args, **kwds):
        if self.hubbard_u is None or self.hubbard_nx is None:
            error("For hubbard, hubbard_u and hubbard_nx have to be given")
        if n_active_electrons is None:
            error("No electron number")
        if self.hubbard_nx <= 0:
            error("Hubbard model but hubbard_nx is not defined!")
        if n_active_electrons <= 0:
            error("# electrons <= 0 !")
        self.n_electrons = n_active_electrons
        if self.Ms is not None:
            self.multiplicity = self.Ms + 1
        else:
            self.Ms = self.multiplicity - 1

        self.natom = self.hubbard_nx*self.hubbard_ny
        self.n_orbitals = self.hubbard_nx*self.hubbard_ny
        self.n_qubits = self.n_orbitals*2

        # Initializing parameters

        default_include = {"c": "a/s", "a": "a/s",
                           "cc": "aa/as/ss", "ca": "aa/as/ss", "aa": "aa/as/ss"}
        default_include.update(self.include)
        self.include = default_include


    @property
    def n_active_electrons(self):
        return self.n_electrons

    @n_active_electrons.setter
    def n_active_electrons(self, value):
        if cf.debug:
            prints(f"Claim that 'n_active_electrons' is changed "
                   f"from {self.n_electrons} to {value}.")
        self.n_electrons = value

    #@property
    #def n_frozen_orbitals(self):
    #    return 0

    #@property
    #def n_frozenv_orbitals(self):
    #    return 0

    #@property
    #def n_core_orbitals(self):
    #    return 0

    @property
    def n_active_orbitals(self):
        return self.n_orbitals

    @n_active_orbitals.setter
    def n_active_orbitals(self, value):
        if cf.debug:
            prints(f"Claim that 'n_active_orbitals' is changed "
                   f"from {self.n_orbitals} to {value}.")
        self.n_orbitals = value

    #@property
    #def n_secondary_orbitals(self):
    #    return 0

    @property
    def noa(self):
        # NOA; Number of Occupied orbitals of Alpha.
        return (self.n_active_electrons+self.multiplicity-1)//2

    @noa.setter
    def noa(self, value):
        self.n_active_electrons = 2*value - self.multiplicity + 1

    @property
    def nob(self):
        # NOB; Number of Occupied orbitals of Beta.
        return self.n_active_electrons - self.noa

    @nob.setter
    def nob(self, value):
        self.n_active_electrons = value + self.noa

    @property
    def nva(self):
        # NVA; Number of Virtual orbitals of Alpha.
        return self.n_active_orbitals - self.noa

    @nva.setter
    def nva(self, value):
        self.n_active_orbitals = value + self.noa

    @property
    def nvb(self):
        # NVB; Number of Virtual orbitals of Beta.
        return self.n_active_orbitals - self.nob

    @nvb.setter
    def nvb(self, value):
        self.n_active_orbitals = value + self.nob

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

    def get_operators(self, guess="minao", run_fci=True, run_hf=True):
        if self.hubbard_ao:
            ### Local (AO) basis 
            Hamiltonian = fermi_hubbard(self.hubbard_nx, self.hubbard_ny,
                                    1, self.hubbard_u)
            fci_coeff = None
        else:
            if mpi.main_rank:
                ### Run HF and transform Hamiltonian (1D)
                from pyscf import gto
                from quket.utils import run_pyscf_mod
                molecule = gto.Mole()
                molecule.spin = self.multiplicity - 1
                molecule.nelectron = self.n_electrons
                molecule.build()
                molecule.hubbard_nx = self.hubbard_nx
                molecule.hubbard_u = self.hubbard_u
                molecule, pyscf_molecule = run_pyscf_mod(guess, self.n_active_orbitals, self.n_active_electrons, molecule,
                      spin=None, run_fci=run_fci, nroots=1, system='hubbard')
                if run_fci:
                    fci_coeff = pyscf_molecule.fci_coeff
                else:
                    fci_coeff = None
                from quket.utils import generate_general_openfermion_operator
                Hamiltonian = generate_general_openfermion_operator(0, molecule.one_body_integrals, molecule.two_body_integrals)
            else:
                Hamiltonian = None
                fci_coeff = None
        
        Hamiltonian = mpi.bcast(Hamiltonian, root=0)
        S2 = s_squared_operator(self.n_orbitals)
        Number = number_operator(self.n_orbitals*2)
        fci_coeff = mpi.bcast(fci_coeff, root=0)

        self.fci_coeff = fci_coeff
#        if run_hf:
#            from .utils import get_OpenFermion_integrals
#            from .fileio import printmat
#            const, one_body_integrals, two_body_integrals = get_OpenFermion_integrals(Hamiltonian, self.n_orbitals)
#            run_custom_pyscf(const, one_body_integrals, two_body_integrals, self.n_electrons, self.n_orbitals)
#            error()
        return Hamiltonian, S2, Number

#def run_custom_pyscf(const, one_body_integrals, two_body_integrals,  n_electrons, n_orbitals):
#    from pyscf import gto, scf, ao2mo, fci
#    from openfermionpyscf._run_pyscf import (compute_scf, compute_integrals,
#                                             PyscfMolecularData)
#    molecule = gto.Mole()
#    molecule.nelectron = n_electrons
#    #molecule.spin = molecule.multiplicity - 1
#    molecule.build()
#    pyscf_scf = scf.RHF(molecule)
#    h1 = np.zeros((n_orbitals,n_orbitals))
#    pyscf_scf.get_hcore = lambda *args: one_body_integrals
#    pyscf_scf.get_ovlp = lambda *args: np.eye(n_orbitals)
#    pyscf_scf._eri = two_body_integrals
#    # 2e Hamiltonian in 4-fold symmetry
#    pyscf_scf._eri = ao2mo.restore(4, pyscf_scf._eri, n_orbitals)
#    pyscf_scf.kernel()
#    print(pyscf_scf.hf_energy)
#
#    # Populate fields.
#    molecule.canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
#    molecule.orbital_energies = pyscf_scf.mo_energy.astype(float)
#    print(pyscf_scf.mo_coeff)
#
#    two_electron_compressed = ao2mo.kernel(molecule,
#                                           pyscf_scf.mo_coeff)
#    # Get two electron integrals in compressed format.
#        
#    #molecule.two_body_integrals = two_body_integrals
#    molecule.overlap_integrals = pyscf_scf.get_ovlp()
#
#    ## Run FCI.
#    #pyscf_fci = fci.FCI(molecule, pyscf_scf.mo_coeff)
#    #pyscf_fci.verbose = 0
#    #molecule.fci_energy = pyscf_fci.kernel()[0]
#    #pyscf_data["fci"] = pyscf_fci
#    #if verbose:
#    #    print(f"FCI energy for {molecule.name} "
#    #          f"({molecule.n_electrons} electrons) is "
#    #          f"{molecule.fci_energy}.")
#    #rdm1 = pyscf_fci.make_rdm1(pyscf_fci.ci,
#    #                           n_active_orbitals, n_active_electrons)
#    #occ, no = eigh(rdm1)
#    #occ = occ[::-1]
#    #no = no[:, ::-1]
#    #pyscf_scf.mo_coeff = pyscf_scf.mo_coeff.dot(no)
#    #molecule.canonical_orbitals = pyscf_scf.mo_coeff
#    #molecule.orbital_energies = occ
#    #one_body_integrals, two_body_integrals \
#    #        = compute_integrals(pyscf_molecule, pyscf_scf)
#    #molecule.one_body_integrals = one_body_integrals
#    #molecule.two_body_integrals = two_body_integrals
#    #pyscf_molecule_buf = prepare_pyscf_molecule_mod(molecule)
#    #pyscf_scf_buf = compute_scf(pyscf_molecule_buf)
#    #molecule.overlap_integrals = pyscf_scf_buf.get_ovlp()
#    return molecule, pyscf_scf
