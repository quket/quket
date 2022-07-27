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

prop.py

Properties.

"""
import numpy as np

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import prints, printmat, SaveTheta, print_state

def prop(Quket):
    if Quket.state is not None:
        name = Quket.ansatz + " (" + Quket.method + ") "
        print_state(Quket.state, name=name)
        Quket.get_1RDM()
        if Quket.Do2RDM:
            if Quket.Daaaa is None:
                Quket.get_2RDM()
            E = calc_energy(Quket, verbose=True)
        
        
        if Quket.model == "chemical":
            dipole(Quket)

def dipole(Quket):
    """Function
    Prepare the dipole operator and get expectation value.

    Author(s): Takashi Tsuchimochi
    """
    n_qubits = Quket.n_qubits
    Dipole = Quket.operators.Dipole

    if Quket.DA is None and Quket.DB is None:
        Quket.get_1RDM()

    # Compute dipole and get expectation value from 1RDM
    n_orbitals = Quket.n_orbitals
    D_1dim = Quket.DA + Quket.DB

    # AO to MO
    rx = Quket.mo_coeff.T@Quket.rint[0]@Quket.mo_coeff
    ry = Quket.mo_coeff.T@Quket.rint[1]@Quket.mo_coeff
    rz = Quket.mo_coeff.T@Quket.rint[2]@Quket.mo_coeff

    dx = dy = dz = 0
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            dx -= rx[p][q]*D_1dim[q][p]
            dy -= ry[p][q]*D_1dim[q][p]
            dz -= rz[p][q]*D_1dim[q][p]

    d = np.array([dx, dy, dz])
    d += (Quket.atom_charges.reshape(1, -1)@Quket.atom_coords).reshape(-1)
    d /= 0.393456

    prints("\nDipole moment from 1RDM (in Debye):")
    prints(f"x = {d[0]:.5f}  y = {d[1]:.5f}  z = {d[2]:.5f}")
    prints(f"| mu | = {np.linalg.norm(d):.5f}")

    if Quket.RelDA is not None and Quket.RelDB is not None:
        # Compute dipole and get expectation value from 1RDM
        n_orbitals = Quket.n_orbitals
        D_1dim = Quket.RelDA + Quket.RelDB

       # AO to MO
        rx = Quket.mo_coeff.T@Quket.rint[0]@Quket.mo_coeff
        ry = Quket.mo_coeff.T@Quket.rint[1]@Quket.mo_coeff
        rz = Quket.mo_coeff.T@Quket.rint[2]@Quket.mo_coeff

        dx = dy = dz = 0
        for p in range(n_orbitals):
            for q in range(n_orbitals):
                dx += -rx[p][q]*D_1dim[q][p]
                dy += -ry[p][q]*D_1dim[q][p]
                dz += -rz[p][q]*D_1dim[q][p]

        d = np.array([dx, dy, dz])
        d += (Quket.atom_charges.reshape(1, -1)@Quket.atom_coords).reshape(-1)
        d /= 0.393456

        prints("\nDipole moment from relaxed 1RDM (in Debye):")
        prints(f"x = {d[0]:.5f}     y = {d[1]:.5f}    z = {d[2]:.5f}")
        prints(f"| mu | = {np.linalg.norm(d):.5f}")


def calc_energy(QuketData, verbose=False):
    """Function
        Calculate energy from 1RDM and 2RDM.

    Author(s): Taisei Nishimaki
    """
    Eaa = Ebb = 0
    Eaaaa = Ebbbb = Ebaab = 0
    Ecore = 0
    E_nuc = QuketData.nuclear_repulsion
    ncore = QuketData.n_frozen_orbitals
    if (QuketData.DA is not None
            and QuketData.DB is not None
            and QuketData.Daaaa is not None
            and QuketData.Dbbbb is not None
            and QuketData.Dbaab is not None):
        h1 = QuketData.one_body_integrals
        h2 = QuketData.two_body_integrals
        n_qubit = QuketData.n_qubits 
        nact = QuketData.n_active_orbitals
        norbs = QuketData.n_orbitals

        #calculate 1dim energy(minus)
        for p in range(norbs):
            for q in range (norbs):
                Eaa += h1[p, q]*QuketData.DA[p, q]
                Ebb += h1[p, q]*QuketData.DB[p, q]
        E1 = Eaa + Ebb

        #calculate 2dim energy(plus)
        for p in range(nact):
            for q in range(nact):
                for r in range(nact):
                    for s in range(nact):
                        val = h2[p+ncore,s+ncore,q+ncore,r+ncore]
                        Eaaaa += val  * QuketData.Daaaa[p, q, r, s]
                        Ebbbb += val  * QuketData.Dbbbb[p, q, r, s]
                        Ebaab += val  * QuketData.Dbaab[p, q, r, s]
                        ## (ps|qr)
        Eaaaa = np.einsum('psqr,pqrs->',h2[ncore:nact+ncore,ncore:nact+ncore,ncore:nact+ncore,ncore:nact+ncore], QuketData.Daaaa)
        Ebbbb = 2*np.einsum('psqr,pqrs->',h2[ncore:nact+ncore,ncore:nact+ncore,ncore:nact+ncore,ncore:nact+ncore], QuketData.Dbaab)
        Ebaab = np.einsum('psqr,pqrs->',h2[ncore:nact+ncore,ncore:nact+ncore,ncore:nact+ncore,ncore:nact+ncore], QuketData.Dbbbb)

        Ecore = 2*np.einsum('psii,ps->',h2[ncore:nact+ncore,ncore:ncore+nact, :ncore,:ncore], QuketData.DA[ncore:nact+ncore,ncore:nact+ncore])
        Ecore += 2*np.einsum('psii,ps->',h2[ncore:nact+ncore,ncore:ncore+nact, :ncore,:ncore], QuketData.DB[ncore:nact+ncore,ncore:nact+ncore])
        Ecore -= np.einsum('piis,ps->',h2[ncore:nact+ncore,:ncore,:ncore,ncore:nact+ncore], QuketData.DA[ncore:nact+ncore,ncore:nact+ncore])
        Ecore -= np.einsum('piis,ps->',h2[ncore:nact+ncore,:ncore,:ncore,ncore:nact+ncore], QuketData.DB[ncore:nact+ncore,ncore:nact+ncore])
        for i in range(ncore):
            for j in range(ncore):
                Ecore += 2*h2[i,i,j,j] - h2[i,j,j,i]


        E2 = 1/2*Eaaaa + 1/2*Ebbbb + 1/2*Ebaab + Ecore
    
        if verbose:
            prints(f"\n Nuclear-replusion energy: {E_nuc}")
            prints(f"1RDM energy: {E1}")
            prints(f"2RDM energy: {E2}")
            prints(f"Total energy: {E_nuc+E1+E2}")
    return E1 + E2 + E_nuc
    
