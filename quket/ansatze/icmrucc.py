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
import copy
import itertools

import numpy as np
from scipy.special import comb
from qulacs import QuantumCircuit

from quket import config as cf
from quket.linalg import root_inv
from quket.opelib import Gdouble_ope
from quket.fileio import prints, print_state, SaveTheta, printmat
from quket.utils import int2occ, transform_state_jw2bk
from quket.opelib import single_ope_Pauli
from quket.projection import S2Proj
from quket.lib import QuantumState
from .jmucc import create_HS2S



def ucc_Gsingles_listver(circuit, a_list, i_list, theta_list, theta_index,
                         sf=False):
    """ Function
    generalized singles. i_list -> a_list
    Author(s): Yuto Mori
    """
    if np.all(a_list == i_list):
        ai_list = itertools.combinations(a_list, 2)
    else:
        ai_list = itertools.product(a_list, i_list)

    for a, i in ai_list:
        cf.ncnot += (a-i) * 2 * 2
        single_ope_Pauli(a, i, circuit, theta_list[theta_index])
        if not sf:
            theta_index += 1
        single_ope_Pauli(a+1, i+1, circuit, theta_list[theta_index])
        theta_index += 1
    return circuit, theta_index


def ucc_Gdoubles_listver(circuit, i1_list, i2_list, a1_list, a2_list,
                         theta_list, theta_index,
                         nc=None, na=None, nv=None, ndim1=None,
                         a2a=None, sf=False):
    """ Function
    Author(s): Yuto Mori
    """
    if np.all(i1_list == i2_list):
        i1i2_aa = list(itertools.combinations(i1_list, 2))
        i1i2_ab = list(itertools.product(i1_list, i2_list))
    else:
        i1i2_aa = list(itertools.product(i1_list, i2_list))
        i1i2_ab = list(itertools.product(i1_list, i2_list))
        i1i2_ab.extend(list(itertools.product(i2_list, i1_list)))
    if np.all(a1_list == a2_list):
        a1a2_aa = list(itertools.combinations(a1_list, 2))
        a1a2_ab = list(itertools.product(a1_list, a2_list))
    else:
        a1a2_aa = list(itertools.product(a1_list, a2_list))
        a1a2_ab = list(itertools.product(a1_list, a2_list))
        a1a2_ab.extend(list(itertools.product(a2_list, a1_list)))

    def operation(a, b, i, j, theta_index):
        max_id = max(a, b, i, j)
        if a != i or b != j:
            if not sf:
                theta = theta_list[theta_index]
            else:
                if (a+b)%2 == 1:
                    baji = (get_baji_for_icmrucc_spinfree(
                        a2, a1, i2, i1, nc, na, nv, a2a) + ndim1)
                    theta = theta_list[baji]
                else:
                    baji = (get_baji_for_icmrucc_spinfree(
                            a2, a1, i2, i1, nc, na, nv, a2a) + ndim1)
                    baij = (get_baji_for_icmrucc_spinfree(
                            a2, a1, i1, i2, nc, na, nv, a2a) + ndim1)
                    theta = theta_list[baji] - theta_list[baij]

            if b == max_id:
                Gdouble_ope(b, a, j, i, circuit, theta)
            elif a == max_id:
                Gdouble_ope(a, b, i, j, circuit, theta)
            elif j == max_id:
                Gdouble_ope(j, i, b, a, circuit, -theta)
            elif i == max_id:
                Gdouble_ope(i, j, a, b, circuit, -theta)
            theta_index += 1
        return theta_index

    # aa -> aa
    for i1, i2 in i1i2_aa:
        for a1, a2 in a1a2_aa:
            theta_index = operation(a1*2, a2*2, i1*2, i2*2, theta_index)
    # bb -> bb
    for i1, i2 in i1i2_aa:
        for a1, a2 in a1a2_aa:
            theta_index = operation(a1*2 + 1, a2*2 + 1, i1*2 + 1, i2*2 + 1,
                                    theta_index)
    # ab -> ab
    for i1, i2 in i1i2_ab:
        for a1, a2 in a1a2_ab:
            theta_index = operation(a1*2, a2*2 + 1, i1*2, i2*2 + 1, theta_index)
    return circuit, theta_index


def icmr_ucc_singles(circuit, n_qubits, nc, na, nv, theta_list,
                     ndim2=0, sf=False):
    """ Function
    Author(s): Yuto Mori
    """
    theta_index = ndim2
    core_list_a = np.arange(nc*2)[::2]
    #core_list_b = np.arange(nc)[1::2]
    act_list_a  = np.arange(nc*2, (nc+na)*2)[::2]
    #act_list_b  = np.arange(nc, nc+na)[1::2]
    vir_list_a  = np.arange((nc+na)*2, n_qubits)[::2]
    #vir_list_b  = np.arange(nc+na, n_qubits)[1::2]

    circuit, theta_index \
            = ucc_Gsingles_listver(circuit, act_list_a, core_list_a,
                                   theta_list, theta_index,
                                   sf=sf)
    circuit, theta_index \
            = ucc_Gsingles_listver(circuit, vir_list_a, core_list_a,
                                   theta_list, theta_index,
                                   sf=sf)
    circuit, theta_index \
            = ucc_Gsingles_listver(circuit, act_list_a, act_list_a,
                                   theta_list, theta_index,
                                   sf=sf)
    circuit, theta_index \
            = ucc_Gsingles_listver(circuit, vir_list_a, act_list_a,
                                   theta_list, theta_index,
                                   sf=sf)


def icmr_ucc_doubles(circuit, n_qubits, nc, na, nv, theta_list, a2a,
                     ndim1=0, sf=False):
    """ Function
    Author(s): Yuto Mori
    """
    theta_index = ndim1
    core_list = np.arange(nc)
    act_list = np.arange(nc, nc+na)
    vir_list = np.arange(nc+na, nc+na+nv)

    def wrap(circuit, theta_index, i1_list, i2_list, a1_list, a2_list):
        if sf:
            return ucc_Gdoubles_listver(circuit,
                                        i1_list, i2_list, a1_list, a2_list,
                                        theta_list, theta_index,
                                        nc=nc, na=na, nv=nv, ndim1=ndim1,
                                        a2a=a2a, sf=sf)
        else:
            return ucc_Gdoubles_listver(circuit,
                                        i1_list, i2_list, a1_list, a2_list,
                                        theta_list, theta_index)

    circuit, theta_index = wrap(circuit, theta_index,
                                core_list, core_list, vir_list, vir_list)
    circuit, theta_index = wrap(circuit, theta_index,
                                core_list, core_list, act_list, vir_list)
    circuit, theta_index = wrap(circuit, theta_index,
                                core_list, core_list, act_list, act_list)

    circuit, theta_index = wrap(circuit, theta_index,
                                core_list, act_list, vir_list, vir_list)
    circuit, theta_index = wrap(circuit, theta_index,
                                core_list, act_list, act_list, vir_list)
    circuit, theta_index = wrap(circuit, theta_index,
                                core_list, act_list, act_list, act_list)

    circuit, theta_index = wrap(circuit, theta_index,
                                act_list, act_list, vir_list, vir_list)
    circuit, theta_index = wrap(circuit, theta_index,
                                act_list, act_list, act_list, vir_list)
    if a2a:
        circuit, theta_index = wrap(circuit, theta_index,
                                    act_list, act_list, act_list, act_list)
    return circuit


def set_circuit_ic_mrucc(n_qubits, nv, na, nc, DS, theta_list,
                         a2a, ndim1, sf=False):
    """ Function
    Author(s): Yuto Mori
    """
    circuit = QuantumCircuit(n_qubits)

    if DS:
        icmr_ucc_singles(circuit, n_qubits, nc, na, nv, theta_list, 0, sf)
        icmr_ucc_doubles(circuit, n_qubits, nc, na, nv, theta_list, a2a,
                         ndim1, sf)
    else:
        icmr_ucc_doubles(circuit, n_qubits, nc, na, nv, theta_list, a2a,
                         ndim1, sf)
        icmr_ucc_singles(circuit, n_qubits, nc, na, nv, theta_list, 0, sf)
    return circuit


def create_icmr_uccsd_state(n_qubits, nv, na, nc, rho, DS, theta_list,
                            det, a2a, ndim1,
                            init_state=None, sf=False, SpinProj=False,
                            mapping="jordan_wigner"):
    """ Function
    Author(s): Yuto Mori
    """
    if init_state is None:
        state = QuantumState(n_qubits)
        state.set_computational_basis(det)
        if mapping == "bravyi_kitaev":
            state = transform_state_jw2bk(state)
    else:
        state = init_state.copy()

    theta_list_rho = theta_list/rho
    circuit = set_circuit_ic_mrucc(n_qubits, nv, na, nc, DS,
                                   theta_list_rho, a2a, ndim1, sf)
    for i in range(rho):
        circuit.update_quantum_state(state)

    if SpinProj:
        state = S2Proj(Quket,state)
    return state


def cost_ic_mrucc(Quket, print_level, qulacs_hamiltonian, qulacs_s2,
                  theta_list, sf=False):
    """ Function
    Author(s): Yuto Mori
    """

    t1 = time.time()

    nca = ncb = Quket.nc
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    rho = Quket.rho
    DS = Quket.DS
    det = Quket.current_det
    n_qubits = Quket.n_qubits
    ndim1 = Quket.ndim1
    ndim2 = Quket.ndim2
    ndim = Quket.ndim
    nstates = Quket.multi.nstates
    a2a = Quket.multi.act2act_opt

    # assume that nca = ncb, noa = nob and nva = nvb
    nc = nca
    no = noa
    nv = nva

    states = []
    for istate in range(nstates):
        det = Quket.multi.init_states_info[istate]
        if not isinstance(Quket.multi.init_states_info[istate], (int, np.integer)):
            raise TypeError('Model space given by `multi` section has to have only single determinants')
        cf.ncnot = 0
        state = create_icmr_uccsd_state(n_qubits, nv, no, nc, rho, DS,
                                        theta_list, det, a2a, ndim1,
                                        sf=sf,
                                        SpinProj=Quket.projection.SpinProj,
                                        mapping=Quket.cf.mapping)
        #prints('# of CNOTS? = ',cf.ncnot)
        #error()
        states.append(state)
    H, S2, S = create_HS2S(Quket, states)
    #printmat(H)

    root_invS = root_inv(S.real)
    H_ortho = root_invS.T@H@root_invS
    nstates0 = root_invS.shape[1]

    en, dvec = np.linalg.eigh(H_ortho)
    idx   = np.argsort(en.real,-1)
    en    = en.real[idx]
    dvec  = dvec[:, idx]
    cvec  = root_invS@dvec
    S2dig = cvec.T@S2@cvec
    s2 = [S2dig[i, i].real for i in range(nstates0)]

    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        string = f"{cf.icyc:6d}: "
        for istate in range(nstates0):
            prints(f"{string} E[{istate}-IC-MRUCC] = {en[istate]:.8f}  "
                   f"(<S**2> = {s2[istate]:+7.5f})  ",
                   end="")
            string = f"\n        "

        prints(f"  Grad = {cf.grad:4.2e}  "
               f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        SaveTheta(ndim, theta_list, cf.tmp)
        Quket.theta_list = theta_list.copy()
        # cf.iter_threshold = 0
    if print_level > 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        string = " Final: "
        for istate in range(nstates0):
            prints(f"{string} E[{istate}-IC-MRUCC] = {en[istate]:.8f}  "
                   f"(<S**2> = {s2[istate]:+7.5f})  ", end="")
            string = f"\n        "
        prints(f"  CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)\n")
        prints("------------------------------------")
        for istate in range(nstates):
            print_state(states[istate], name=f"ic Basis   {istate}")
        printmat(cvec.real, name="Coefficients:")
        prints("------------------------------------\n\n")
        prints("###############################################")
        prints("#                  ic states                  #")
        prints("###############################################", end="")

        for istate in range(nstates0):
            prints()
            prints(f"State         : {istate}")
            prints(f"E             : {en[istate]:.8f}")
            prints(f"<S**2>        : {s2[istate]:+.5f}")
            spstate = QuantumState(n_qubits)
            spstate.multiply_coef(0)
            for jstate in range(nstates0):
                state = states[jstate].copy()
                coef  = cvec[jstate, istate]
                state.multiply_coef(coef)
                spstate.add_state(state)
            print_state(spstate, name="Superposition:")
        prints("###############################################")

    cost = norm = 0
    norm = np.sum(Quket.multi.weights)
    cost = np.sum(Quket.multi.weights*en)
    cost /= norm
    return cost, s2


###################
#### Spin-Free ####
###################
def get_baji_for_icmrucc_spinfree(b, a, j, i, nc, na, nv, a2a):
    """
    Author(s): Yuhto Mori
    """
    bj = (b-nc)*(na+nc) + j
    ai = (a-nc)*(na+nc) + i
    if a2a:
        if bj > ai:
            if b > j:
                if b >= na + nc:
                    redu = na*(na+1)//2
                else:
                    redu = (b-nc)*(b-nc+1)//2
            elif b < j:
                redu = (b-nc+1)*(b-nc+2)//2
            else:
                redu = (b-nc)*(b-nc+1)//2 + a - nc
                if a < i:
                    redu += 1
            baji = bj*(bj+1)//2 + ai - redu
        else:
            if a > i:
                if a >= na + nc:
                    redu = na*(na+1)//2
                else:
                    redu = (a-nc)*(a-nc+1)//2
            elif a < i:
                redu = (a-nc+1)*(a-nc+2)//2
            else:
                redu = (a-nc)*(a-nc+1)//2 + b - nc
                if b < j:
                    redu += 1
            baji = ai*(ai+1)//2 + bj - redu
    else:
        if bj > ai:
            if b >= na + nc:
                redu = (na*na)*((na*na) + 1)//2
            else:
                tmp = na*(b-nc)
                if j >= nc:
                    tmp += j - nc
                redu = tmp*(tmp+1)//2 + na*(a-nc)
            baji = bj*(bj+1)//2 + ai - redu
        else:
            if a >= na + nc:
                redu = (na*na)*((na*na) + 1) // 2
            else:
                tmp = na*(a-nc)
                if i >= nc:
                    tmp += i - nc
                redu = tmp*(tmp+1)//2 + na*(b-nc)
            baji = ai*(ai+1)//2 + bj - redu
    return baji
