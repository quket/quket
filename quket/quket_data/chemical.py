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
from dataclasses import dataclass, field

import sys
import numpy as np
try:
    from openfermion.hamiltonians import MolecularData
except:
    from openfermion.chem import MolecularData


from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.utils import run_pyscf_mod, prepare_pyscf_molecule_mod
from quket.opelib import create_1body_operator
from quket.fileio import prints, print_geom, error
from quket.tapering import get_pointgroup_character_table
from quket.lib import number_operator, s_squared_operator


@dataclass
class Chemical(MolecularData):
    geometry: List = field(default_factory=list)
    basis: str = None
    multiplicity: int = 1
    Ms: int = None
    charge: int = 0
    description: str = ""
    filename: str = ""
    data_directory: str = None
    n_active_electrons: int = None
    n_active_orbitals: int = None
    n_core_orbitals: int = 0
    n_frozen_orbitals: int = 0
    symmetry: bool = True
    symmetry_subgroup: str = None
    spin: int = None
    weights: int = None
    pyscf_guess: str = "minao"
    nroots: int = 1
    ##----- For MBE -----
    min_use_core: int = 0
    max_use_core: int = 0
    min_use_secondary: int = 0
    max_use_secondary: int = 0
    n_secondary_orbitals: int = -1
    include: Dict = field(default_factory=dict)
    #color: int = None
    #later: bool = False
    mo_basis: str = "hf"
    #mbe_exact: bool = False
    #mbe_correlator: str = "sauccsd"
    #mbe_oo: bool = False
    #mbe_oo_ansatz: str = None
    #from_vir: bool = False


    n_qubits: int = field(init=False, default=None)
    DA: np.ndarray = field(init=False, default=None)
    DB: np.ndarray = field(init=False, default=None)
    RelDA: np.ndarray = field(init=False, default=None)
    RelDB: np.ndarray = field(init=False, default=None)
    Daaaa: np.ndarray = field(init=False, default=None)   # Unrelaxed 2 density matrix for alpha
    Dbbbb: np.ndarray = field(init=False, default=None)   # Unrelaxed 2 density matrix for beta
    Dbaab: np.ndarray = field(init=False, default=None)   # Unrelaxed 2 density matrix for beta alpha alpha beta
    Daaaaaa: np.ndarray = field(init=False, default=None) # Unrelaxed 3 density matrix for alpha
    Dbbbbbb: np.ndarray = field(init=False, default=None) # Unrelaxed 3 density matrix for alpha
    Dbaaaab: np.ndarray = field(init=False, default=None) # Unrelaxed 3 density matrix for alpha beta
    Dbbaabb: np.ndarray = field(init=False, default=None) # Unrelaxed 3 density matrix for alpha beta
    Daaaaaaaa: np.ndarray = field(init=False, default=None) # Unrelaxed 4 density matrix for alpha
    Dbbbbbbbb: np.ndarray = field(init=False, default=None) # Unrelaxed 4 density matrix for beta
    Dbaaaaaab: np.ndarray = field(init=False, default=None) # Unrelaxed 4 density matrix for alpha beta
    Dbbaaaabb: np.ndarray = field(init=False, default=None) # Unrelaxed 4 density matrix for alpha beta
    Dbbbaabbb: np.ndarray = field(init=False, default=None) # Unrelaxed 4 density matrix for alpha beta

    one_body_integrals: np.ndarray = field(init=False, default=None)
    two_body_integrals: np.ndarray = field(init=False, default=None)
    overlap_integrals: np.ndarray = field(init=False, default=None)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __post_init__(self, *args, **kwds):
        if self.n_active_orbitals is not None:
            if self.n_active_orbitals <= 0:
                error(f"# orbitals = {self.n_active_orbitals}!")
        if self.n_active_electrons is not None:
            if self.n_active_electrons <= 0:
                error(f"# electrons = {self.n_active_electrons}!")
        if self.n_core_orbitals < 0:
            error(f"# core orbitals = {self.n_core_orbitals}!")
        if self.min_use_core < 0:
            error(f"Min # using core orbitals = {self.min_use_core}!")
        if self.max_use_core < 0:
            error(f"Max # using core orbitals = {self.max_use_core}!")
        if self.min_use_core > self.max_use_core:
            error(f"min_use_core={self.min_use_core} > max_use_core={self.max_use_core}")
        if self.min_use_secondary < 0:
            error(f"Min # using secondary orbitals = {self.min_use_secondary}!")
        if self.max_use_secondary < 0:
            error(f"Max # using secondary orbitals = {self.max_use_secondary}!")
        if self.min_use_secondary > self.max_use_secondary:
            error(f"min_use_secondary={self.min_use_secondary} > max_use_secondary={self.max_use_secondary}")
        if self.Ms is not None:
            self.multiplicity = self.Ms + 1
        else:
            self.Ms = self.multiplicity - 1

        if cf._units in ('au', 'bohr'):
            geom = []
            for iatom in range(len(self.geometry)):
                geom.append((self.geometry[iatom][0],
                            (self.geometry[iatom][1][0] * 0.529177249,
                             self.geometry[iatom][1][1] * 0.529177249,
                             self.geometry[iatom][1][2] * 0.529177249)))
            cf._units = 'angstrom'
            self.geometry = geom

        super().__init__(geometry=self.geometry, basis=self.basis,
                         multiplicity=self.multiplicity, charge=self.charge,
                         description=self.description, filename=self.filename,
                         data_directory=self.data_directory)

        pyscf_mol = prepare_pyscf_molecule_mod(self)

        self.n_orbitals = int(pyscf_mol.nao_nr())

        noa = (self.n_electrons+self.multiplicity-1)//2
        nob = self.n_electrons - noa
        nva = self.n_orbitals - noa
        nvb = self.n_orbitals - nob
        # Note; if n_secondary_orbitals < 0 then secondary-space is dynamic,
        #       else static.

        if self.n_active_electrons is None:
            if self.n_active_orbitals is None:
                ### Assuming all orbitals and electrons are active
                ### except for secondary space, which may have been
                ### specified.
                self.n_frozen_orbitals = 0
                if self.n_secondary_orbitals < 0:
                    self.n_secondary_orbitals = 0
                self.n_active_orbitals \
                        = self.n_orbitals \
                        - (self.n_secondary_orbitals
                           + self.n_core_orbitals
                           + self.n_frozen_orbitals)
            else:
                if self.n_secondary_orbitals < 0:
                    self.n_frozen_orbitals = 0
                    self.n_secondary_orbitals \
                            = self.n_orbitals \
                            - (self.n_active_orbitals
                               + self.n_core_orbitals
                               + self.n_frozen_orbitals)
                else:
                    self.n_frozen_orbitals \
                            = self.n_orbitals \
                            - (self.n_secondary_orbitals
                               + self.n_active_orbitals
                               + self.n_core_orbitals)
            self.n_active_electrons \
                    = self.n_electrons \
                    - (self.n_core_orbitals+self.n_frozen_orbitals)*2
        else:
            if self.n_active_orbitals is None:
                ### active Ne known
                ###   --> can derive n_frozen_orbitals
                self.n_frozen_orbitals \
                        = (self.n_electrons
                           - (self.n_active_electrons
                              + self.n_core_orbitals*2))//2
                ### active orbitals are not set, so
                ### use the rest orbitals (no secondary)
                if self.n_secondary_orbitals < 0:
                    self.n_secondary_orbitals = 0
                self.n_active_orbitals \
                        = self.n_orbitals \
                        - (self.n_core_orbitals
                           + self.n_frozen_orbitals)
            else:
                if self.n_secondary_orbitals < 0:
                    self.n_frozen_orbitals \
                            = (self.n_electrons
                               - (self.n_active_electrons
                                  + self.n_core_orbitals*2))//2
                    self.n_secondary_orbitals \
                            = self.n_orbitals \
                            - (self.n_active_orbitals
                               + self.n_core_orbitals
                               + self.n_frozen_orbitals)
                else:
                    #self.n_frozen_orbitals \
                    #        = self.n_orbitals \
                    #        - (self.n_secondary_orbitals
                    #           + self.n_active_orbitals
                    #           + self.n_core_orbitals)
                    self.n_frozen_orbitals \
                            = (self.n_electrons -  self.n_active_electrons) // 2 \
                              - self.n_core_orbitals
                    #self.n_secondary_orbitals \
                    #        = self.n_orbitals \
                    #        - (self.n_active_orbitals
                    #           + self.n_core_orbitals
                    #           + self.n_frozen_orbitals)

        # n_secondary_orbitals are only meant to be used for post calculations such as MBE and LUCC.
        # So, let it be inert until then...
        self._n_secondary_orbitals = self.n_secondary_orbitals
        self.n_secondary_orbitals = 0
        self.core_offset = self.n_core_orbitals
        self.n_frozenv_orbitals = self.n_orbitals - (self.ns + self.na + self.nc + self.nf)
        # Note; ns, na, nc, nf is abbreviation. Show property.
        #assert self.n_orbitals == self.ns + self.na + self.nc + self.nf
        #assert self.n_electrons == self.n_active_electrons + (self.nc + self.nf)*2
        if self.n_electrons == self.n_active_electrons + (self.nc + self.nf)*2:
            pass
        else:
            prints(f"Check and use the following space sizes at your own risk.\n"
                   f"# total orbitals = {self.n_orbitals}\n"
                   f"# frozen core orbitals = {self.nf}\n"
                   f"# core orbitals = {self.nc}\n"
                   f"# active orbitals = {self.na}\n"
                   f"# secondary orbitals = {self.ns}\n")
        if  (self.n_active_orbitals < 0) \
         or (self.n_frozen_orbitals < 0) \
         or (self.n_core_orbitals < 0) \
         or (self.n_secondary_orbitals < 0):
            error(f"Error detected in the space size.\n"
                   f"# total orbitals = {self.n_orbitals}\n"
                   f"# frozen core orbitals = {self.nf}\n"
                   f"# core orbitals = {self.nc}\n"
                   f"# active orbitals = {self.na}\n"
                   f"# secondary orbitals = {self.ns}\n")

        self._guess = self.pyscf_guess

        #if self.max_use_core == 0:
        #    self.max_use_core = self.n_core_orbitals
        #if self.max_use_secondary == 0:
        #    self.max_use_secondary = (self.n_orbitals
        #                              - self.n_active_orbitals
        #                              - self.n_core_orbitals
        #                              - self.n_frozen_orbitals)
        self.n_qubits = self.n_active_orbitals*2

        default_include = {"c": "a/s", "a": "a/s",
                           "cc": "aa/as/ss", "ca": "aa/as/ss", "aa": "aa/as/ss"}
        default_include.update(self.include)
        self.include = default_include

        if cf.debug:
            prints("+----------------+")
            prints("| CHEMICAL STATE |")
            prints("+----------------+")
            prints(f"\tbasis={self.basis}")
            prints(f"\tmo_basis={self.mo_basis}")
            prints(f"\tn_orbitals={self.n_orbitals}")
            prints(f"\tn_frozen_orbitals={self.n_frozen_orbitals}")
            prints(f"\tn_core_orbitals={self.n_core_orbitals}")
            prints(f"\tn_active_orbitals={self.n_active_orbitals}")
            prints(f"\tn_frozenv_orbitals={self.n_frozenv_orbitals}")
            prints(f"\tn_secondary_orbitals={self.n_secondary_orbitals}")
            prints(f"\tn_electrons={self.n_electrons}")
            prints(f"\tn_active_electrons={self.n_active_electrons}")
            prints(f"\texcitation")
            prints(f"\tinclude={self.include}")

    #@property
    #def n_frozen_orbitals(self):
    #    return self.n_frozen_orbitals

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
        # NF; Number of Frozen orbitals.
        # Note; Frozen orbitals must have same alpha/beta orbitals.
        return self.n_frozen_orbitals

    @property
    def nc(self):
        # NC; Number of Core orbitals.
        # Note; Core orbitals must have same alpha/beta orbitals.
        return self.n_core_orbitals

    @nc.setter
    def nc(self, value):
        self.n_core_orbitals = value

    @property
    def na(self):
        return self.n_active_orbitals

    @na.setter
    def na(self, value):
        self.n_active_orbitals = value

    @property
    def ns(self):
        return self.n_secondary_orbitals

    @ns.setter
    def ns(self, value):
        self.n_secondary_orbitals = value


    def get_operators(self, guess="minao",
                      run_fci=True, run_ccsd=True,
                      run_mp2=False, run_casscf=False,
                      bcast_Hamiltonian=True):
        """Function
        Run PySCF and get operators.
        For large basis, ERI can be large: epsecially, Hamiltonian contains full-spin ERI.
        So, in some cases it is better to use bcast_Hamiltonian = False not to bcast over all processes.
        """
        #sys.stdout.flush()
        #print(f"I am {mpi.rank=}  and {mpi.main_rank=} ", flush=True)
        if mpi.main_rank:
            # Run electronic structure calculations
            #if guess != self._guess:
            frozen_indices = list(range(self.nf + self.nc))
            frozen_indices.extend(list(range(self.nf + self.nc + self.na,
                                             self.n_orbitals)))
            self, pyscf_mol = run_pyscf_mod(guess,
                                            self.n_active_orbitals,
                                            self.n_active_electrons,
                                            self,
                                            spin=self.spin,
                                            run_casci=run_fci,
                                            run_ccsd=run_ccsd,
                                            run_mp2=run_mp2,
                                            run_casscf=run_casscf,
                                            mo_basis=self.mo_basis,
                                            nroots=self.nroots,
                                            frozen_indices=frozen_indices,
                                            ncore=self.nf+self.nc)

            n_core = self.nf + self.nc
            occupied_indices = list(range(n_core))
            active_indices = list(range(n_core, n_core+self.na))
            Hamiltonian = self.get_molecular_hamiltonian(
                    occupied_indices=occupied_indices,
                    active_indices=active_indices)
            S2 = s_squared_operator(self.na)
            Number = number_operator(self.na*2)

            hf_energy = self.hf_energy
            fci_energy = self.fci_energy
            ccsd_energy = self.ccsd_energy
            mp2_energy = self.mp2_energy
            casscf_energy = self.casscf_energy
            nuclear_repulsion = self.nuclear_repulsion

            mo_coeff = np.array(self.canonical_orbitals)
            mo_energy = self.orbital_energies
            one_body_integrals = self.one_body_integrals
            # This is so that two_body_integrals[p,q,r,s] contains h_pqrs = (pq|rs)
            # instead of the weird ordering of OpenFermion (ps|qr)
            two_body_integrals = self.two_body_integrals.transpose(0, 3, 1, 2)
            #two_body_integrals = self.two_body_integrals.copy()
            overlap_integrals = self.overlap_integrals
            natom = pyscf_mol.natm
            atom_charges = pyscf_mol.atom_charges()
            atom_coords = pyscf_mol.atom_coords()
            rint = pyscf_mol.intor("int1e_r")
            if run_fci:
                fci_coeff = pyscf_mol.fci_coeff
            else:
                fci_coeff = None

            # Dipole operators from dipole integrals (AO)
            rx = create_1body_operator(rint[0], mo_coeff=mo_coeff, ao=True,
                                       n_active_orbitals=self.na)
            ry = create_1body_operator(rint[1], mo_coeff=mo_coeff, ao=True,
                                       n_active_orbitals=self.na)
            rz = create_1body_operator(rint[2], mo_coeff=mo_coeff, ao=True,
                                       n_active_orbitals=self.na)
            Dipole = [rx, ry, rz]

            # For point-group symmetry
            irrep_name = pyscf_mol.irrep_name
            symm_orb = pyscf_mol.symm_orb
            topgroup = pyscf_mol.topgroup
            groupname = pyscf_mol.groupname
            if self.symmetry:
                symm_operations, irrep_list, character_list = get_pointgroup_character_table(
                                                    pyscf_mol, groupname,
                                                    irrep_name, symm_orb, mo_coeff)
            else:
                symm_operations, irrep_list, character_list = None, ['A' for _ in range(2*self.n_orbitals)], None


        else:
            Hamiltonian = None
            S2 = None
            Number = None
            Dipole = None
            hf_energy = None
            fci_energy = None
            ccsd_energy = None
            mp2_energy = None
            casscf_energy = None
            nuclear_repulsion = None
            rint = None
            mo_coeff = None
            mo_energy = None
            one_body_integrals = None
            #two_body_integrals = None
            two_body_integrals = np.zeros((self.n_orbitals, self.n_orbitals,
                                           self.n_orbitals, self.n_orbitals), dtype=float)
            #sys.stdout.flush()
            #two_body_integrals_dummy = np.empty((self.n_orbitals, self.n_orbitals, self.n_orbitals), dtype=float)
            overlap_integrals = None
            natom = None
            atom_charges = None
            atom_coords = None
            fci_coeff = None

            # For point-group symmetry
            irrep_name = None
            symm_orb = None
            topgroup = None
            groupname = None
            symm_operations = None
            irrep_list = None
            character_list = None


        # MPI broadcasting
        if bcast_Hamiltonian:
            Hamiltonian = mpi.bcast(Hamiltonian, root=0)
        S2 = mpi.bcast(S2, root=0)
        Number = mpi.bcast(Number, root=0)
        Dipole = mpi.bcast(Dipole, root=0)
        hf_energy = mpi.bcast(hf_energy, root=0)
        fci_energy = mpi.bcast(fci_energy, root=0)
        ccsd_energy = mpi.bcast(ccsd_energy, root=0)
        nuclear_repulsion = mpi.bcast(nuclear_repulsion, root=0)
        rint = mpi.bcast(rint, root=0)
        mo_coeff = mpi.bcast(mo_coeff, root=0)
        canonical_orbitals = mpi.bcast(self.canonical_orbitals, root=0)
        mo_energy = mpi.bcast(mo_energy, root=0)
        one_body_integrals = mpi.bcast(one_body_integrals, root=0)
        ### Need to do some dirty-laundry...mpi4py is not working for node-parallel bcast with deep ndarray (>3)?
        for i in range(self.n_orbitals):
            if mpi.main_rank:
                two_body_integrals_dummy = two_body_integrals[i]
            else:
                two_body_integrals_dummy = None
#
            two_body_integrals_dummy = mpi.bcast(two_body_integrals_dummy, root=0)
            two_body_integrals[i] = two_body_integrals_dummy.copy()
        overlap_integrals = mpi.bcast(overlap_integrals, root=0)
        natom = mpi.bcast(natom, root=0)
        atom_charges = mpi.bcast(atom_charges, root=0)
        atom_coords = mpi.bcast(atom_coords, root=0)
        fci_coeff = mpi.bcast(fci_coeff, root=0)

        # For point-group symmetry
        irrep_name = mpi.bcast(irrep_name, root=0)
        symm_orb = mpi.bcast(symm_orb, root=0)
        topgroup = mpi.bcast(topgroup, root=0)
        groupname = mpi.bcast(groupname, root=0)
        symm_operations = mpi.bcast(symm_operations, root=0)
        irrep_list = mpi.bcast(irrep_list, root=0)
        character_list = mpi.bcast(character_list, root=0)

        # Put values in self
        self.hf_energy = hf_energy
        self.fci_energy = fci_energy
        self.ccsd_energy = ccsd_energy
        self.mo_coeff = mo_coeff
        self.canonical_orbitals = canonical_orbitals
        self.mo_coeff0 = mo_coeff
        self.mo_energy = mo_energy
        self.rint = rint
        self.nuclear_repulsion = nuclear_repulsion
        self.one_body_integrals = np.array(one_body_integrals)
        self.two_body_integrals = two_body_integrals
        if bcast_Hamiltonian:
            self.zero_body_integrals_active = Hamiltonian.constant
            self.one_body_integrals_active = Hamiltonian.one_body_tensor
            self.two_body_integrals_active = Hamiltonian.two_body_tensor
        #############################
        ### Notation of integrals ###
        #############################
        #  nuclear_repulsion  : Nuclear repulsion
        #  one_body_integrals : spin-integrated, spatial-orbital basis, all molecular orbitals, (norbs, norbs)
        #  two_body_integrals : spin-integrated, spatial-orbital basis, all molecular orbitals, (norbs, norbs, norbs, norbs)
        #                       two_body_integrals[p,q,r,s] = (pq|rs) = int dr1 dr2  p*(r1) q(r1) r*(r2) s(r2) / |r1-r2|
        #                       where p(r) are spatial orbitals.
        #  So the combination of these represents the full Hamiltonian
        #     H_full =  nuclear_repulsion
        #             + one_body_integrals[p,q] (pA^ qA + pB^ qB)
        #             + 1/2 two_body_integrals[p,r,q,s]  (pA^ qA^ sA rA  +  pA^ qB^ sB rA  + pB^ qA^ sA rB  +  pB^ qB^ sB rB)
        #  where pA, qB, etc. are in the spin orbital representaion.
        #
        #  zero_body_integrals_active : nuclear_repulsion + Ecore, where Ecore is the core electron energy
        #  one_body_integrals_active  : spin-orbital basis (qubit order), active orbitals, includes core contribution, (n_qubits, n_qubits)
        #  two_body_integrals_active  : spin-orbital basis (qubit order), active orbitals, (n_qubits, n_qubits, n_qubits, n_qubits)
        #                               two_body_integrals_active[p,q,r,s] = <pq|sr>
        #  So the combination of these represents the active space Hamiltonian
        #     H_act  =  zero_body_integrals_active
        #             + one_body_integrals_active[p,q] (p^ q)
        #             + 1/2 two_body_integrals_active[p,q,r,s]  p^ q^ r s
        #
        self.overlap_integrals = overlap_integrals
        self.natom = natom
        self.atom_charges = atom_charges
        self.atom_coords = atom_coords
        self.fci_coeff = fci_coeff

        # For point-group symmetry
        self.irrep_name = irrep_name
        self.symm_orb = symm_orb
        self.topgroup = topgroup
        self.groupname = groupname
        self.symm_operations = symm_operations
        self.irrep_list = irrep_list
        self.character_list = character_list

        # Print out some results
        if self.spin is None:
            spin = self.multiplicity
            self.spin = spin
        else:
            spin = self.spin
        if run_fci:
            if isinstance(fci_energy, float):
                prints(f"E[FCI]    = {fci_energy:0.12f}     (Spin = {spin}   Ms = {self.Ms})")
            else:
                prints(f"E[FCI]    = {fci_energy[0]:0.12f}     (Spin = {spin}   Ms = {self.Ms})")
                for i in range(1, len(fci_energy)):
                    prints(f"            {fci_energy[i]:0.12f}     (Spin = {spin}   Ms = {self.Ms})")
        if run_mp2:
            prints(f"E[MP2]    = {mp2_energy:0.12f}     (Spin = {spin}   Ms = {self.Ms})")
        if run_ccsd:
            prints(f"E[CCSD]   = {ccsd_energy:0.12f}     (Spin = {spin}   Ms = {self.Ms})")
        if run_casscf:
            if isinstance(fci_energy, float):
                prints(f"E[CASSCF] = {casscf_energy:0.12f}     (Spin = {spin}   Ms = {self.Ms})")
            else:
                prints(f"E[CASSCF] = {casscf_energy[0]:0.12f}     (Spin = {spin}   Ms = {self.Ms})")
                for i in range(1, len(fci_energy)):
                    prints(f"            {casscf_energy[i]:0.12f}     (Spin = {spin}   Ms = {self.Ms})")
        prints(f"E[HF]     = {hf_energy:0.12f}     (Spin = {self.Ms+1}   Ms = {self.Ms})")

        return Hamiltonian, S2, Number, Dipole

