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

mod.py

Modified versions of OpenFermionPySCF routines
to enable active space calculations.

"""
import numpy as np
from functools import reduce
from scipy.linalg import eigh
from pyscf import scf, ci, cc, fci, mp, mcscf, gto, ao2mo
try:
    from openfermion.hamiltonians import MolecularData
except:
    from openfermion.chem import MolecularData
from openfermionpyscf._run_pyscf import (compute_scf, compute_integrals,
                                         PyscfMolecularData)

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import error, prints, printmat


def prepare_pyscf_molecule_mod(molecule):
    """Function
    This function creates and saves a pyscf input file.
    Args:
        molecule: An instance of the MolecularData class.
    Returns:
        pyscf_molecule: A pyscf molecule instance.

    Author(s): Takashi Tsuchimochi
    """
    pyscf_molecule = gto.Mole()
    pyscf_molecule.atom = molecule.geometry
    pyscf_molecule.basis = molecule.basis
    pyscf_molecule.spin = molecule.multiplicity - 1
    pyscf_molecule.charge = molecule.charge
    pyscf_molecule.symmetry = molecule.symmetry
    if molecule.symmetry_subgroup is not None:
        pyscf_molecule.symmetry_subgroup = molecule.symmetry_subgroup
    pyscf_molecule.build()


    # Check for infinitestmal rotation group Coov and Dooh
    # and change them to C2v and D2h respectively
    infinite_group = {'Coov':'C2v', 'Dooh':'D2h', 'SO3':'D2h'}
    if pyscf_molecule.topgroup in infinite_group:
        pyscf_molecule.symmetry_subgroup = pyscf_molecule.groupname = infinite_group[pyscf_molecule.topgroup]
        # After assigning new subgroup name,
        # it is necessary to re-build the molecule according to the new subgroup
        pyscf_molecule.build()

    return pyscf_molecule


#### modify generate_molecular_hamiltonian to be able to use chkfile
#def generate_molecular_hamiltonian_mod(guess, geometry, basis, multiplicity,
#                                       charge=0, n_active_electrons=None,
#                                       n_active_orbitals=None):
#    """Function
#    Old subroutine to get molecular hamiltonian by using pyscf.
#
#    Author(s): Takashi Tsuchimochi
#    """
#
#    # Run electronic structure calculations
#    molecule, pyscf_mol \
#            = run_pyscf_mod(guess, n_active_orbitals, n_active_electrons,
#                            MolecularData(geometry, basis, multiplicity, charge,
#                                          data_directory=cf.input_dir))
#    # Freeze core orbitals and truncate to active space
#    if n_active_electrons is None:
#        n_core_orbitals = 0
#        occupied_indices = None
#    else:
#        n_core_orbitals = (molecule.n_electrons-n_active_electrons)//2
#        occupied_indices = list(range(n_core_orbitals))
#
#    if n_active_orbitals is None:
#        active_indices = None
#    else:
#        active_indices = list(range(n_core_orbitals,
#                                    n_core_orbitals+n_active_orbitals))
#
#    return molecule.get_molecular_hamiltonian(occupied_indices=occupied_indices,
#                                              active_indices=active_indices)
#

def run_pyscf_mod(guess, n_active_orbitals, n_active_electrons, molecule,
                  spin=None, run_mp2=False, run_cisd=False,
                  run_ccsd=False, run_fci=False,
                  run_casscf=False, run_casci=True, verbose=False,
                  mo_basis="hf", nroots=1, system='chemical',
                  frozen_indices=None, ncore=None):
    """Function
    This function runs a pyscf calculation.

    Args:
        molecule: An instance of the MolecularData or PyscfMolecularData class.
        run_mp2: Optional boolean to run MP2 calculation.
        run_cisd: Optional boolean to run CISD calculation.
        run_ccsd: Optional boolean to run CCSD calculation.
        run_fci: Optional boolean to FCI calculation.
        run_casscf: Optional boolean to CASSCF calculation.
        run_casci: optional boolean to CASCI calculation.
        verbose: Boolean whether to print calculation results to screen.
    Returns:
        molecule: The updated PyscfMolecularData object. Note the attributes
                  of the input molecule are also updated in this function.

    Author(s): Takashi Tsuchimochi
    """
    if system == 'chemical':
        # Prepare pyscf molecule.
        pyscf_molecule = prepare_pyscf_molecule_mod(molecule)
        # Store pyscf data 
        molecule.pyscf = pyscf_molecule
        prints(f"Symmetry {pyscf_molecule.topgroup} : {pyscf_molecule.groupname}",end='')
        if pyscf_molecule.groupname in cf.abelian_groups:
            prints("(Abelian)")
        else:
            prints("(non Abelian)")
        molecule.n_orbitals = int(pyscf_molecule.nao_nr())
        molecule.nuclear_repulsion = float(pyscf_molecule.energy_nuc())
        # Run SCF.
        pyscf_scf = compute_scf(pyscf_molecule)
        #pyscf_scf.scf()
        #pyscf_scf.verbose = 1
        _guess = "chkfile" if guess == "read" else guess
        #pyscf_scf.run(chkfile=cf.chk, init_guess=_guess,
        #              conv_tol=1e-12, conv_tol_grad=1e-12)
        pyscf_scf.chkfile = cf.chk
        pyscf_scf.init_guess = _guess
        pyscf_scf.conv_tol = 1e-9
        pyscf_scf.conv_tol_grad = 1e-5
        pyscf_scf.verbose = 0
        pyscf_scf.kernel()
        if not pyscf_scf.converged:
            prints(f'WARNING: pyscf did not converge.')
            prints(f'         This may be an issue with initial guess {_guess}.')
            prints(f'         You may use different guess (minao, huckel, read)')
        #pyscf_scf.kernel(IntPQ=cf.IntPQ)

    elif system == 'hubbard':
        n = molecule.hubbard_nx
        u = molecule.hubbard_u
        molecule.n_orbitals = n
        molecule.nuclear_repulsion = float(0)
        molecule.incore_anyway = True
        pyscf_molecule = molecule
        h1 = np.zeros([n] * 2, dtype=np.float64)
        for i in range(n-1):
            h1[i, i+1] = h1[i+1, i] = -1.
        h1[n-1, 0] = h1[0, n-1] = -1.
        eri = np.zeros([n] * 4, dtype=np.float64)
        for i in range(n):
            eri[i, i, i, i] = u
        pyscf_scf = scf.RHF(molecule)
        pyscf_scf.get_hcore = lambda *args: h1
        pyscf_scf.get_ovlp = lambda *args: np.eye(n)
        pyscf_scf._eri = ao2mo.restore(8, eri, n) # 8-fold symmetry
        pyscf_scf.init_guess = '1e'
        pyscf_scf.kernel()
        bas = []
        for i in range(n):
            bas.append( [ 0,  0,  3,  1,  0, 28, 31,  0] )
        pyscf_molecule._bas = np.array(bas, dtype=np.int32)

    molecule.hf_energy = float(pyscf_scf.e_tot)

    # Set number of active electrons/orbitals.
    if n_active_electrons is None:
        n_active_electrons = molecule.n_electrons
    if n_active_orbitals is None:
        n_active_orbitals = molecule.n_orbitals

    # Check
    nvir = pyscf_scf.mo_coeff.shape[1] - ncore - n_active_orbitals
    if nvir < 0 or ncore < 0:
        error(f'Inconsistent n_orbitals and n_electrons.\n'
              f'  Number of MOs = {pyscf_scf.mo_coeff.shape[1]}\n'
              f'  Number of electrons = {molecule.n_electrons}\n'
              f'  Number of active electrons = {n_active_electrons}\n'
              f'  Number of active orbitals = {n_active_orbitals}\n'
              f'  Number of frozen-core orbitals = {ncore}\n'
              f'  Number of frozen-vir orbitals = {nvir}\n')


    # Hold pyscf data in molecule. They are required to compute density
    # matrices and other quantities.
    molecule._pyscf_data = {}
    pyscf_data = {}
    pyscf_data["mol"] = pyscf_molecule
    pyscf_data["scf"] = pyscf_scf

    # Populate fields.
    molecule.canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
    molecule.orbital_energies = pyscf_scf.mo_energy.astype(float)
    molecule.casscf_energy = None

    # Get integrals.
    one_body_integrals, two_body_integrals \
            = compute_integrals_mod(pyscf_molecule, pyscf_scf, system)
    molecule.one_body_integrals = one_body_integrals
    molecule.two_body_integrals = two_body_integrals
    molecule.overlap_integrals = pyscf_scf.get_ovlp()

    if cf.IntPQ is not None:
        nbasis = pyscf_scf.mo_coeff.shape[0]
        IntPQ_AO = np.zeros((nbasis, nbasis), dtype=float)
        IntPQ_AO[cf.IntPQ[0], cf.IntPQ[1]] += cf.IntPQ[2]
        IntPQ_AO[cf.IntPQ[1], cf.IntPQ[0]] += cf.IntPQ[2]
        #one_electron_compressed = reduce(
        #    np.dot, (pyscf_scf.mo_coeff.T, IntPQ_AO, pyscf_scf.mo_coeff)
        #)
        #IntPQ_MO = one_electron_compressed.reshape(
        #    n_orbitals, n_orbitals
        #).astype(float)
        IntPQ_MO = pyscf_scf.mo_coeff.T @ IntPQ_AO @ pyscf_scf.mo_coeff
        molecule.one_body_integrals = one_body_integrals + IntPQ_MO
        

    if run_mp2 or "mp2" in mo_basis:
        if molecule.multiplicity != 1:
            error("WARNING: RO-MP2 is not available in PySCF.")
        else:
            pyscf_mp2 = mp.MP2(pyscf_scf)
            pyscf_mp2.verbose = 0
            #pyscf_mp2.run(chkfile=cf.chk, init_guess=guess,
            #              conv_tol=1e-12, conv_tol_grad=1e-12)
            #pyscf_mp2.run(init_guess=guess, conv_tol=1e-12, conv_tol_grad=1e-12)
            pyscf_mp2.kernel()
            molecule.mp2_energy = pyscf_mp2.e_tot  # pyscf-1.4.4 or higher
            #molecule.mp2_energy = pyscf_scf.e_tot + pyscf_mp2.e_corr # else
            pyscf_data["mp2"] = pyscf_mp2
            if verbose:
                print(f"MP2 energy for {molecule.name} "
                      f"({molecule.n_electrons} electrons) is "
                      f"{molecule.mp2_energy}.")
            if "mp2" in mo_basis:
                rdm1 = pyscf_mp2.make_rdm1()
                occ, no = eigh(rdm1)
# ソート順を絶対値の大きい順に変更する？
                occ = occ[::-1]
                no = no[:, ::-1]
                printmat(occ, name="Natural Occupations of MP2: \n", n=1, m=len(occ))
                pyscf_scf.mo_coeff = pyscf_scf.mo_coeff.dot(no)
                molecule.canonical_orbitals = pyscf_scf.mo_coeff
                molecule.orbital_energies = occ
                one_body_integrals, two_body_integrals \
                        = compute_integrals_mod(pyscf_molecule, pyscf_scf, system)
                molecule.one_body_integrals = one_body_integrals
                molecule.two_body_integrals = two_body_integrals
                pyscf_molecule_buf = prepare_pyscf_molecule_mod(molecule)
                pyscf_scf_buf = compute_scf(pyscf_molecule_buf)
                molecule.overlap_integrals = pyscf_scf_buf.get_ovlp()

    # Run CISD.
    if run_cisd or mo_basis == "cisd":
        pyscf_cisd = ci.CISD(pyscf_scf)
        pyscf_cisd.verbose = 0
        #pyscf_cisd.run(chkfile=cf.chk, init_guess=guess,
        #               conv_tol=1e-12, conv_tol_grad=1e-12)
        #pyscf_cisd.run(init_guess=guess, conv_tol=1e-12, conv_tol_grad=1e-12)
        pyscf_cisd.kernel()
        molecule.cisd_energy = pyscf_cisd.e_tot
        pyscf_data["cisd"] = pyscf_cisd
        if verbose:
            print(f"CISD energy for {molecule.name} "
                  f"({molecule.n_electrons} electrons) is "
                  f"{molecule.cisd_energy}.")
        if mo_basis == "cisd":
            rdm1 = pyscf_cisd.make_rdm1()
            occ, no = eigh(rdm1)
# ソート順を絶対値の大きい順に変更する？
            occ = occ[::-1]
            no = no[:, ::-1]
            pyscf_scf.mo_coeff = pyscf_scf.mo_coeff.dot(no)
            molecule.canonical_orbitals = pyscf_scf.mo_coeff
            molecule.orbital_energies = occ
            one_body_integrals, two_body_integrals \
                    = compute_integrals_mod(pyscf_molecule, pyscf_scf, system)
            molecule.one_body_integrals = one_body_integrals
            molecule.two_body_integrals = two_body_integrals
            pyscf_molecule_buf = prepare_pyscf_molecule_mod(molecule)
            pyscf_scf_buf = compute_scf(pyscf_molecule_buf)
            molecule.overlap_integrals = pyscf_scf_buf.get_ovlp()

    # Run CCSD.
    if run_ccsd or mo_basis == "ccsd":
        if mo_basis != "ccsd":
            pyscf_ccsd = cc.CCSD(pyscf_scf, frozen=frozen_indices)
        else:
            pyscf_ccsd = cc.CCSD(pyscf_scf)
        pyscf_ccsd.verbose = 0
        #pyscf_ccsd.run(chkfile=cf.chk, init_guess=guess,
        #               conv_tol=1e-12, conv_tol_grad=1e-12)
        #pyscf_ccsd.run(init_guess=guess, conv_tol=1e-12, conv_tol_grad=1e-12)
        pyscf_ccsd.kernel()
        molecule.ccsd_energy = pyscf_ccsd.e_tot
        pyscf_data["ccsd"] = pyscf_ccsd
        if verbose:
            print(f"CCSD energy for {molecule.name} "
                  f"({molecule.n_electrons} electrons) is "
                  f"{molecule.ccsd_energy}.")
        if mo_basis == "ccsd":
            rdm1_ = pyscf_ccsd.make_rdm1()
            if molecule.multiplicity != 1:
                rdm1 = rdm1_[0] + rdm1_[1]
            else:
                rdm1 = rdm1_
            occ, no = eigh(rdm1)
            occ = occ[::-1]
            no = no[:, ::-1]
            pyscf_scf.mo_coeff = pyscf_scf.mo_coeff.dot(no)
            molecule.canonical_orbitals = pyscf_scf.mo_coeff
            molecule.orbital_energies = occ
            one_body_integrals, two_body_integrals \
                    = compute_integrals_mod(pyscf_molecule, pyscf_scf, system)
            molecule.one_body_integrals = one_body_integrals
            molecule.two_body_integrals = two_body_integrals
            pyscf_molecule_buf = prepare_pyscf_molecule_mod(molecule)
            pyscf_scf_buf = compute_scf(pyscf_molecule_buf)
            molecule.overlap_integrals = pyscf_scf_buf.get_ovlp()

    # Run FCI.
    if run_fci or mo_basis == "fci":
        pyscf_fci = fci.FCI(pyscf_scf, pyscf_scf.mo_coeff)
        pyscf_fci.verbose = 0
        molecule.fci_energy = pyscf_fci.kernel()[0]
        pyscf_data["fci"] = pyscf_fci
        molecule.fci_coeff = pyscf_fci.ci
        if verbose:
            print(f"FCI energy for {molecule.name} "
                  f"({molecule.n_electrons} electrons) is "
                  f"{molecule.fci_energy}.")
        if mo_basis == "FCI":
            rdm1 = pyscf_fci.make_rdm1(pyscf_fci.ci,
                                       n_active_orbitals, n_active_electrons)
            occ, no = eigh(rdm1)
            occ = occ[::-1]
            no = no[:, ::-1]
            pyscf_scf.mo_coeff = pyscf_scf.mo_coeff.dot(no)
            molecule.canonical_orbitals = pyscf_scf.mo_coeff
            molecule.orbital_energies = occ
            one_body_integrals, two_body_integrals \
                    = compute_integrals_mod(pyscf_molecule, pyscf_scf, system)
            molecule.one_body_integrals = one_body_integrals
            molecule.two_body_integrals = two_body_integrals
            pyscf_molecule_buf = prepare_pyscf_molecule_mod(molecule)
            pyscf_scf_buf = compute_scf(pyscf_molecule_buf)
            molecule.overlap_integrals = pyscf_scf_buf.get_ovlp()

    # Run CASSCF.
    if run_casscf or "casscf" in mo_basis:
        if spin is None:
            spin = molecule.multiplicity
        # Let's check 
        try:
            ind = mo_basis.index('casscf')
            try:
                n_CAS_electrons = int(mo_basis[ind+1])
                n_CAS_orbitals = int(mo_basis[ind+2])
                try:
                    nroots_cas = int(mo_basis[ind+3])
                except:
                    #nroots_cas = 1
                    nroots_cas = nroots
            except:
                error('Incorrect specification for CASSCF/SA-CASSCF orbital setting?\n',
                      'Use the following format:\n',
                      '      mo_basis = CASSCF(ne, no)\n',
                      '  or  mo_basis = CASSCF(ne, no, nroots)')
        except:
            n_CAS_orbitals = n_active_orbitals
            n_CAS_electrons = n_active_electrons
            nroots_cas = nroots
        pyscf_casscf = mcscf.CASSCF(pyscf_scf,
                                    n_CAS_orbitals, n_CAS_electrons) 
        pyscf_casscf.verbose = 1
        if nroots_cas > 1:
            nbuf = 4
            pyscf_casscf.fcisolver.nroots = nroots_cas + nbuf
            weights = [1/nroots_cas for i in range(nroots_cas)]
            for ibuf in range(nbuf):
                weights.append(1e-14)
            
            weights = tuple(weights)
            mcscf.state_average_(pyscf_casscf, weights=weights)
            # Level shift to remove spin states
            pyscf_casscf.fix_spin(shift=1, ss=(spin-1)*(spin+1)/4)
            results = pyscf_casscf.kernel()
            pyscf_casci = mcscf.CASCI(pyscf_scf,
                                    n_CAS_orbitals, n_CAS_electrons) 
            pyscf_casci.fcisolver.nroots = nroots_cas
            pyscf_casci.fix_spin(shift=1, ss=(spin-1)*(spin+1)/4)
            results_casci = pyscf_casci.kernel(mo_coeff=results[3])
            prints('SA-CASSCF energies: ', results_casci[0])
        else:
            results = pyscf_casscf.kernel()
            pyscf_molecule.casscf_energy = results[0]

        # Convert to natural orbitals.
        # mo_coeff, ci, mo_occ will be converted.
        pyscf_casscf.cas_natorb_()
        molecule.casscf_energy = pyscf_casscf.e_tot
        pyscf_molecule.casscf_coeff = pyscf_casscf.ci
        rdm1 = pyscf_casscf.make_rdm1()

        if mo_basis[0] == "casscf" or mo_basis[0] == "sa-casscf":
            molecule.canonical_orbitals = pyscf_casscf.mo_coeff
            molecule.orbital_energies = pyscf_casscf.mo_energy
            one_body_integrals, two_body_integrals \
                    = compute_integrals_mod(pyscf_molecule, pyscf_scf, system)
            molecule.one_body_integrals = one_body_integrals
            molecule.two_body_integrals = two_body_integrals
            pyscf_molecule_buf = prepare_pyscf_molecule_mod(molecule)
            pyscf_scf_buf = compute_scf(pyscf_molecule_buf)
            molecule.overlap_integrals = pyscf_scf_buf.get_ovlp()
            S = molecule.overlap_integrals
            c = pyscf_casscf.mo_coeff
            rdm1_no = c.T @ S @ rdm1 @ S @ c
            occ = [rdm1_no[i,i] for i in range(c.shape[1])]
            printmat(occ, name="Natural Occupations of CASSCF: \n", n=1, m=len(occ))

    # Run CASCI (FCI).
    ### Change the spin ... (S,Ms) = (spin, multiplicity)
    if run_casci or mo_basis == "casci":
        if spin is None:
            spin = molecule.multiplicity
        pyscf_molecule.spin = spin - 1
        pyscf_casci = mcscf.CASCI(pyscf_scf,
                                  n_active_orbitals, n_active_electrons)
        pyscf_casci.fix_spin(shift=1.0, ss=(spin-1)*(spin+1)/4)
        pyscf_casci.fcisolver.nroots = nroots
        pyscf_casci.verbose = 0
        # comment: Necessary to set mo_coeff if the pre-computed MP2/CCSD/... orbitals are desired. 
        #          Otherwise HF canonical orbitals are used!
        pyscf_casci.kernel(mo_coeff=molecule.canonical_orbitals)
        # Convert to natural orbitals.
        # mo_coeff, ci, mo_occ will be converted.
        molecule.fci_energy = pyscf_casci.e_tot
        pyscf_molecule.fci_coeff = pyscf_casci.ci

        if mo_basis == "casci":
            if nroots > 1:
                error(f'nroots cannot be larger than 1 for mo_basis = casci')
            pyscf_casci.cas_natorb_()
            molecule.canonical_orbitals = pyscf_casci.mo_coeff
            molecule.orbital_energies = pyscf_casci.mo_energy
            one_body_integrals, two_body_integrals \
                    = compute_integrals_mod(pyscf_molecule, pyscf_scf, system)
            molecule.one_body_integrals = one_body_integrals
            molecule.two_body_integrals = two_body_integrals
            pyscf_molecule_buf = prepare_pyscf_molecule_mod(molecule)
            pyscf_scf_buf = compute_scf(pyscf_molecule_buf)
            molecule.overlap_integrals = pyscf_scf_buf.get_ovlp()

    #molecule._pyscf_data = pyscf_data
    return molecule, pyscf_molecule



def compute_integrals_mod(pyscf_molecule, pyscf_scf, system):
    """
    Compute the 1-electron and 2-electron integrals.

    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = reduce(np.dot, (pyscf_scf.mo_coeff.T,
                                                 pyscf_scf.get_hcore(),
                                                 pyscf_scf.mo_coeff))
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    if system == 'chemical':
        # two-electron integrals are directly computed with atomic basis functions
        two_electron_compressed = ao2mo.kernel(pyscf_molecule,
                                               pyscf_scf.mo_coeff)
    else:
        # two-electron integrals are explicitly stored in _eri
        two_electron_compressed = ao2mo.kernel(pyscf_scf._eri,
                                               pyscf_scf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed, n_orbitals)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals

