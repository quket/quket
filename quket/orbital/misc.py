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

orbital/misc.py

Functions related to orbital gradient and hessian

"""
import numpy as np
import time

from pyscf.lo import Boys
from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import error, prints, printmat, print_state
from quket.linalg import symm, vectorize_symm

def get_JK(h2, DA, DB):
    """
    JK_A[p,q] = (pq|rs) DA[s,r] + DB[s,r] - (ps|rq) DA[s,r]
    JK_B[p,q] = (pq|rs) DA[s,r] + DB[s,r] - (ps|rq) DB[s,r]
    """
    n = DA.shape[0]
    J_A = np.einsum('pqrs, rs -> pq', h2, DA)
    K_A = np.einsum('psrq, rs -> pq', h2, DA)
    J_B = np.einsum('pqrs, rs -> pq', h2, DB)
    K_B = np.einsum('psrq, rs -> pq', h2, DB)
    return J_A, J_B, K_A, K_B

def get_Fock(h1, h2, DA, DB):
    """
    get one-body hamiltonian
    """
    J_A, J_B, K_A, K_B = get_JK(h2, DA, DB)
    FA = h1 + J_A + J_B - K_A
    FB = h1 + J_A + J_B - K_B
    return FA, FB

def get_htilde(Quket, h1=None, h2=None):
    """
    Ecore = sum 2h[p,p] + 2(pp|qq) - (pq|qp)
    htilde = h[p,q] + 2(pq|KK) - (pK|Kq)
    """
    ncore = Quket.n_frozen_orbitals# + Quket.n_core_orbitals　＃changed since ver8.0, dfferent from ver7.0
    norbs = Quket.n_orbitals
    Ecore = 0
    if h1 is None:
        htilde = Quket.one_body_integrals.copy()
    else:
        htilde = h1.copy()
    if h2 is None:
        h_pqIJ = Quket.two_body_integrals[:,:,:ncore,:ncore]
        h_pIJq = Quket.two_body_integrals[:,:ncore,:ncore,:]
    else:
        h_pqIJ = h2[:,:,:ncore,:ncore]
        h_pIJq = h2[:,:ncore,:ncore,:]

    for p in range(ncore):
        Ecore +=  htilde[p,p]
    for p in range(norbs):
        for q in range(norbs):
            for K in range(ncore):
                htilde[p,q] += 2*h_pqIJ[p,q,K,K] - h_pIJq[p,K,K,q]
    for p in range(ncore):
        Ecore +=  htilde[p,p]
    Ecore += Quket.nuclear_repulsion
    return Ecore, htilde


def boys(Quket, *args):
    from .oo import orbital_rotation
    Quket.mo_coeff[:, args]=Boys(Quket.pyscf).kernel(Quket.mo_coeff[:, args], verbose=4)
    Quket.mo_coeff0 = Quket.mo_coeff.copy()
    Quket.orbital_rotation(mo_coeff=Quket.mo_coeff)

def decompose_CAV(x, ncore, nact, nsec):
    """
    Decompose a vector x[pq] based on orbital space, Core, Active, and Virtual.
    """
    norbs = ncore + nact + nsec
    nott = norbs*(norbs-1)//2
    natt = nact*(nact-1)//2
    if x.shape[0] != nott:
        raise ValueError(f"x has dimensions of {x.shape}, which is not consistent with norbs={norbs}.")
    xAC = []
    xAA = []
    xVC = []
    xVA = []
    pq = 0
    for p in range(norbs):
        for q in range(p):
            if ncore <= p < ncore+nact:
                # A
                if q < ncore:
                    # AC
                    xAC.append(x[pq])
                elif ncore <= q < ncore+nact:
                    # AA
                    xAA.append(x[pq])

            elif ncore+nact <= p:
                # V
                if q < ncore:
                    # VC
                    xVC.append(x[pq])
                elif ncore <= q < ncore+nact:
                    # VA
                    xVA.append(x[pq])
            pq += 1
    xAC = np.array(xAC)
    xAA = np.array(xAA)
    xVC = np.array(xVC)
    xVA = np.array(xVA)
    return xAC, xAA, xVC, xVA

def compose_CAV(xAC, xAA, xVC, xVA, ncore, nact, nsec):
    """
    Combine a vector x[pq] based on orbital space, Core, Active, and Virtual.
    """
    norbs = ncore + nact + nsec
    nott = norbs*(norbs-1)//2
    natt = nact*(nact-1)//2
    x = np.zeros(nott)
    pq_AC = 0
    pq_AA = 0
    pq_VC = 0
    pq_VA = 0
    pq = 0
    for p in range(norbs):
        for q in range(p):
            if ncore <= p < ncore+nact:
                # A
                if q < ncore:
                    # AC
                    x[pq] = xAC[pq_AC]
                    pq_AC += 1
                elif ncore <= q < ncore+nact:
                    # AA
                    x[pq] = xAA[pq_AA]
                    pq_AA += 1

            elif ncore+nact <= p:
                # V
                if q < ncore:
                    # VC
                    x[pq] = xVC[pq_VC]
                    pq_VC += 1
                elif ncore <= q < ncore+nact:
                    # VA
                    x[pq] = xVA[pq_VA]
                    pq_VA += 1
            pq += 1
    return x

def ao2mo(Eri_AO, C, compact=True):
    """
    MPI version of ao2mo (incore)
    """
    nbasis = C.shape[0]
    norbs = C.shape[1]
    ntt = nbasis*(nbasis+1)//2
    nott = norbs*(norbs+1)//2
    Vij_pq = np.zeros((ntt, nott), dtype=float)

    ipos, my_ndim = mpi.myrange(ntt)
    for ij in range(ipos, ipos+my_ndim):
        # (ij|pq) <- (ij|kl) Ckp Clq
        V = symm(Eri_AO[ij,:])
        A = C.T@V@C
        Vij_pq[ij, :] = vectorize_symm(A)
    Vij_pq = mpi.allreduce(Vij_pq)
    Vpq_ij = Vij_pq.transpose(1,0).copy()
    del(Vij_pq)
    Vpq_rs = np.zeros((nott, nott), dtype=float)

    ipos, my_ndim = mpi.myrange(nott)
    for pq in range(ipos, ipos+my_ndim):
        # (pq|rs) <- (pq|ij) Cir Cjs
        V = symm(Vpq_ij[pq,:])
        A = C.T@V@C
        Vpq_rs[pq,:] = vectorize_symm(A)

    Vpq_rs = mpi.allreduce(Vpq_rs)

    if compact:
        return Vpq_rs
    else:
        del(Vpq_ij)
        Vpq_rs_full = np.zeros((norbs,norbs,norbs,norbs), dtype=float)
        pq = 0
        for p in range(norbs):
            for q in range(p+1):
                Vpq_rs_full[p,q,:,:] = Vpq_rs_full[q,p,:,:] = symm(Vpq_rs[pq,:])
                pq += 1
        return Vpq_rs_full
