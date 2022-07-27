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
import time

import scipy as sp
import numpy as np
from numpy import linalg as LA
from qulacs.state import inner_product
from qulacs.observable import create_observable_from_openfermion_text

from .qite_function import calc_delta, calc_psi, make_state1, calc_inner1
from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import prints, print_state
from quket.linalg import lstsq
from quket.utils import transform_state_jw2bk
from quket.lib import QuantumState, QubitOperator, FermionOperator


def make_hamiltonian(model, nspin, nterm):
    if model == "heisenberg":
        sx = []
        sy = []
        sz = []
        for i in range(nspin):
            sx.append(QubitOperator(f"X{i}"))
            sy.append(QubitOperator(f"Y{i}"))
            sz.append(QubitOperator(f"Z{i}"))

        H = []
        active = []
        for term in range(nterm):
            qubit_hamiltonian = 0*QubitOperator("")
            if term == nspin-1:
                i = term
                j = 0
                active.append([i, j])
                qubit_hamiltonian += sx[i]*sx[j] + sy[i]*sy[j] + sz[i]*sz[j]
                str_qubit_hamiltonian = str(qubit_hamiltonian)
                observable = create_observable_from_openfermion_text(
                        str_qubit_hamiltonian)
                H.append(observable)
            else:
                i = term
                j = i + 1
                active.append([i, j])
                qubit_hamiltonian += sx[i]*sx[j] + sy[i]*sy[j] + sz[i]*sz[j]
                str_qubit_hamiltonian = str(qubit_hamiltonian)
                observable = create_observable_from_openfermion_text(
                        str_qubit_hamiltonian)
                H.append(observable)

        qubit_hamiltonian_full = 0*QubitOperator("")
        for i in range(nspin):
            if i < nspin-1:
                j = i + 1
            else:
                j = 0
            qubit_hamiltonian_full += sx[i]*sx[j] + sy[i]*sy[j] + sz[i]*sz[j]
        str_qubit_hamiltonian_full = str(qubit_hamiltonian_full)
        observable_full = create_observable_from_openfermion_text(
                str_qubit_hamiltonian_full)
    return H, active, observable_full


def qite_inexact(Quket, nterm, D):
    ### Parameter setting
    n = Quket.n_qubits
    db = Quket.dt
    ntime = Quket.maxiter
    qbit = Quket.current_det
    model = Quket.model
    threshold = Quket.ftol

    H, active, observable_full = make_hamiltonian(model, n, nterm)
    size = 4**D
    index = np.arange(n)
    delta = QuantumState(n)
    first_state = QuantumState(n)
    first_state.set_computational_basis(qbit)
    if Quket.cf.mapping == "bravyi_kitaev":
        first_state = transform_state_jw2bk(first_state)

    prints(f"Inexact QITE: Pauli operator group size = {size}")

    energy = []
    psi_dash = first_state.copy()
    value = observable_full.get_expectation_value(psi_dash)
    energy.append(value)

    t1 = time.time()
    cf.t_old = t1

    dE = 100
    for t in range(ntime):
        t2 = time.time()
        cput = t2 - cf.t_old
        cf.t_old = t2
        if cf.debug:
            print_state(psi_dash)
        prints(f"{t*db:6.2f}: E = {value:.12f}  CPU Time = {cput:5.2f}")

        if abs(dE) < threshold:
            break

        for term in range(nterm):
            active_qubit = active[term]
            observable = H[term]
            psi_dash_copy = psi_dash.copy()

            S_part = np.zeros((size, size), dtype=complex)
            S = np.zeros((size, size), dtype=complex)
            sizeT = size*(size-1)//2
            ipos, my_ndim = mpi.myrange(sizeT)
            ij = 0
            for i in range(size):
                for j in range(i):
                    if ij in range(ipos, ipos+my_ndim):
                        S_part[i, j] = calc_inner1(i, j, n, active_qubit,
                                                   index, psi_dash)
                        S_part[j, i] = S_part[i, j].conjugate()
                    ij += 1
            mpi.comm.Allreduce(S_part, S, mpi.MPI.SUM)
            for i in range(size):
                S[i, i] = 1

            sigma = [None]*size
            for i in range(size):
                state_i = make_state1(i, n, active_qubit, index, psi_dash)
                sigma[i] = state_i

            delta = calc_delta(psi_dash, observable, n, db)[0]
            b_l = []
            b_l = np.zeros(size)
            for i in range(size):
                b_i = inner_product(sigma[i], delta)
                b_i = -2*b_i.imag
                b_l[i] = b_i
            Amat = 2*np.real(S)
            zct = b_l@Amat

            x, res, rnk, s = lstsq(Amat, b_l, cond=1e-8)
            ### Just in case, broadcast a...
            a = x.copy()
            mpi.comm.Bcast(a, root=0)

            psi_dash = calc_psi(psi_dash_copy, n, index, a, active_qubit)

        value = observable_full.get_expectation_value(psi_dash)
        energy.append(value)
        dE = energy[t+1] - energy[t]
