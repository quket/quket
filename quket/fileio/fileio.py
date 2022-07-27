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

fileio.py

File reading/writing utilities.

"""
import os
import re, sys
import ast
import copy
import datetime
import itertools

import numpy as np

from quket import config as cf
from quket.mpilib import mpilib as mpi


def prints(*args, filepath=None, opentype="a", end=None, root=None):
    """Function:
    Print wrapper.

    Args:
        args     : arguments to be printed
        filepath : file to be printed 
                   if None (default), print to cf.log. 
                   if filepath = '' or cf.log is None, print on command-line
        opentype : 'a' = add (default)
                 : 'w' = create the file, overwrite
        end      : if end=None, break at the end of priting
                   if end="",   don't break
        root     : mpi rank (default is main_rank)

    Author(s): Takashi Tsuchimochi
    """
    if mpi.rank == root or mpi.main_rank:
        if filepath == '' or (filepath is None and cf.log is None):
            print(*args, end=end)
            return
        if filepath is None:
            filepath_ = cf.log
        else:
            filepath_ = filepath
        with open(filepath_, opentype) as f:
            print(*args, file=f, end=end)

def tstamp(*args, filepath=None, end=None, root=0):
    """
    Time stamp.
    """

    prints(f"{datetime.datetime.now()}: ", *args, filepath=filepath, end=end, root=root)

def print_geom(geometry, filepath=None):
    """Function:
    Print geometry in the cartesian coordinates.

    Author(s): Takashi Tsuchimochi
    """
    prints("\n*** Geometry ******************************",filepath=filepath)
    for iatom in range(len(geometry)):
        prints(f"  {geometry[iatom][0]:2s}"
               f"  {geometry[iatom][1][0]:11.7f}"
               f"  {geometry[iatom][1][1]:11.7f}"
               f"  {geometry[iatom][1][2]:11.7f}", filepath=filepath)
    prints("*******************************************\n", filepath=filepath)


def print_grad(geometry, grad):
    """Function:
    Print gradient in the cartesian coordinates.

    Author(s): Takashi Tsuchimochi
    """
    prints("\n*** Nuclear Gradients *********************")
    prints(f"            x            y            z")
    for iatom in range(len(geometry)):
        prints(f"  {geometry[iatom][0]:2s}"
               f"  {grad[iatom,0]:11.7f}"
               f"  {grad[iatom,1]:11.7f}"
               f"  {grad[iatom,2]:11.7f}")
    prints("*******************************************\n")

def openfermion_print_state(state, n_qubits, j_state,
                            threshold=1e-2, digit=4, filepath=cf.log):
    """Function
    print out jth wave function in state

    Author(s): Takashi Tsuchimochi
    """
    opt = f"0{n_qubits}b"
    qubit_len = n_qubits + 4 - len("Basis")
    coef_len = 2*digit + 8
    prints(" "*(qubit_len//2), "Basis",
           " "*(qubit_len//2 + coef_len//2 - 1), "Coef")
    for i in range(2**n_qubits):
        v = state[i][j_state]
        if abs(v)**2 > threshold:
            formstr = (f"{{a.real:+.{digit}f}} "
                       f"{{a.imag:+.{digit}f}}i")
            prints(f"| {format(i, opt)} > : {formstr.format(a=v)}",
                   filepath=filepath)


def SaveTheta(ndim, save, filepath, opentype="w", offset=0):
    if not cf.SaveTheta:
        return
    if save.size != ndim:
        error(f"save.size={save.size} but ndim={ndim}")

    ### Handle multiple columns in filepath (corresponding to
    ### different theta_lists for different states)
    success = True
    if mpi.main_rank:
        if os.path.isfile(filepath):
            load = np.loadtxt(filepath).reshape(-1)
            if load.size % ndim != 0:
                if offset == 0:
                    #prints(f"Warning: load.shape[0]={load.shape[0]} but ndim={ndim} for {filepath}\n"
                    #       f"May overwrite")
                    nstates = 1
                    load = save
                else:
                    prints(f"WARNING!")
                    prints(f"Length of {filepath} needs to be divisible by ndim={ndim}")
                    prints(f"Your theta file is screwed up.")
                    prints(f"Perhaps you switched on/off tapering-off during VQD calculation.")
                    prints(f"Theta file is not stored, but k-th theta_list can be found in ")
                    prints(f"   QuketData.lower_states[k]['theta_list']")
                    return
            else:
                nstates = load.size / ndim
        else:
            load = save
            nstates = 1
        if offset == nstates:
            save_ = np.hstack([load, save])
        elif offset < nstates:
            save_ = load
            save_[offset*ndim:(offset+1)*ndim] = save[0:ndim]
        else:
            prints(f"offset={offset} but nstates={nstates} for {filepath}")
            success = False
        if opentype == "w":
            np.savetxt(filepath, save_)
        elif opentype == "a":
            with open(filepath, opentype) as f:
                np.savetxt(f, save_)

    success = mpi.bcast(success, root=0)
    if not success:
        error()


def LoadTheta(ndim, filepath, offset=0):
    if mpi.main_rank:
        load = np.loadtxt(filepath).reshape(-1)
        if load.size % ndim != 0:
            prints(f"load.size={load.size} but ndim={ndim} for {filepath}")
        load_ = load[offset*ndim:(offset+1)*ndim]
    else:
        load_ = None
    load_ = mpi.bcast(load_, root=0)
    return load_

def SaveAdapt(Quket, filepath):
    if mpi.main_rank:
        #if Quket.adapt.mode in ('pauli', 'pauli_sz', 'pauli_yz', 'spin', 'pauli_spin', 'pauli_spin_xy', 'qeb', 'qeb1', 'qeb2', 'qeb3'): 
        prints(f'', end="", filepath=filepath, opentype="w")
        ndim = len(Quket.pauli_list)
        for icyc in range(ndim):
            prints(f'{icyc};   ', filepath=filepath, end='')
            pauli = Quket.pauli_list[icyc]
            if type(pauli) == list: 
                for pauli_ in pauli:
                    prints(f'{pauli_.terms} | ', filepath=filepath, end='')
                prints(f'{Quket.adapt.grad_list[icyc]}', filepath=filepath)
            else:
                prints(f'{pauli.terms};   {Quket.adapt.grad_list[icyc]}', filepath=filepath)
        #else:
        #    prints(f'', end="", filepath=filepath, opentype="w")
        #    ndim = len(Quket.adapt.b_list)
        #    for icyc in range(ndim):
        #        b = Quket.adapt.b_list[icyc]
        #        a = Quket.adapt.a_list[icyc]
        #        j = Quket.adapt.j_list[icyc]
        #        i = Quket.adapt.i_list[icyc]
        #        spin = Quket.adapt.spin_list[icyc]
        #        grad = Quket.adapt.grad_list[icyc]
        #        prints(f'{icyc},   {b},  {a},  {j},  {i},  {spin},  {grad}', filepath=filepath)

def LoadAdapt(Quket, filepath):
    from quket.lib import QubitOperator
    #if Quket.adapt.mode in ('pauli', 'pauli_sz', 'pauli_yz', 'pauli_spin', 'pauli_spin_xy', 'qeb'):
    pauli_list = []
    grad_list = []
    if mpi.main_rank:
        with open(filepath, "r") as f:
            lines = f.readlines()
        k = 0 
        while k in range(len(lines)):
            icyc, string, grad = lines[k].replace("\n", "").split(";") 
            print(string)
            string = string.strip(' ')
            pauli_dict = ast.literal_eval(string)
            pauli = QubitOperator('',0)
            for op, coef in pauli_dict.items():
                pauli += QubitOperator(op, coef.imag * 1j)
            pauli_list.append(pauli)
            grad_list.append(float(grad))
            k += 1
    pauli_list = mpi.bcast(pauli_list)
    grad_list = mpi.bcast(grad_list)
    Quket.pauli_list = pauli_list
    Quket.adapt.grad_list = grad_list

    #else:
    #    # Initialize
    #    b_list = []
    #    a_list = []
    #    j_list = []
    #    i_list = []
    #    spin_list = []
    #    grad_list = []
    #    if mpi.main_rank:
    #        with open(filepath, "r") as f:
    #            lines = f.readlines()
    #        k = 0 
    #        while k in range(len(lines)):
    #            icyc, b, a, j, i, spin, grad = lines[k].replace("\n", "").split(",") 
    #            b_list.append(int(b))
    #            a_list.append(int(a))
    #            j_list.append(int(j))
    #            i_list.append(int(i))
    #            spin_list.append(int(spin))
    #            grad_list.append(float(grad))
    #            if int(b) != int(a) or int(j) != int(i):
    #                if int(a) not in Quket.adapt.bs_orbitals:
    #                    Quket.adapt.bs_orbitals.append(int(a))
    #                if int(b) not in Quket.adapt.bs_orbitals:
    #                    Quket.adapt.bs_orbitals.append(int(b))
    #                if int(i) not in Quket.adapt.bs_orbitals:
    #                    Quket.adapt.bs_orbitals.append(int(i))
    #                if int(j) not in Quket.adapt.bs_orbitals:
    #                    Quket.adapt.bs_orbitals.append(int(j))
    #            k += 1
    #    b_list = mpi.bcast(b_list, root=0)
    #    a_list = mpi.bcast(a_list, root=0)
    #    j_list = mpi.bcast(j_list, root=0)
    #    i_list = mpi.bcast(i_list, root=0)
    #    spin_list = mpi.bcast(spin_list, root=0)
    #    grad_list = mpi.bcast(grad_list, root=0)
    #    Quket.adapt.b_list = copy.deepcopy(b_list)
    #    Quket.adapt.a_list = copy.deepcopy(a_list)
    #    Quket.adapt.j_list = copy.deepcopy(j_list)
    #    Quket.adapt.i_list = copy.deepcopy(i_list)
    #    Quket.adapt.spin_list = copy.deepcopy(spin_list)
    #    Quket.adapt.grad_list = copy.deepcopy(grad_list)
    

def error(*message):
    prints("\n", *message, "\n")
    if cf.__interactive__:
        raise Exception("Error termination of quket.")
    else:
        prints("Error termination of quket.")
        prints(datetime.datetime.now())
        mpi.MPI.Finalize()
        sys.exit()


def print_state(state, name=None, n_qubits=None, filepath=None,
                threshold=0.01, digit=4, root=None):
    """Function
    print out quantum state as qubits

    Author(s): Takashi Tsuchimochi
    """
    if isinstance(name, str):
        prints(name, filepath=filepath, root=root)
    if n_qubits is None:
        n_qubits = state.get_qubit_count()

    opt = f"0{n_qubits}b"
    qubit_len = n_qubits + 4 - len("Basis")
    coef_len = 2*digit + 8
    prints(" "*(qubit_len//2), "Basis",
           " "*(qubit_len//2 + coef_len//2 - 1), "Coef", root=root)
    vec = state.get_vector()
    for i in range(2**n_qubits):
        v = vec[i]
        if abs(v)**2 > threshold:
            formstr = (f"{{a.real:+.{digit}f}} "
                       f"{{a.imag:+.{digit}f}}i")
            prints(f"| {format(i, opt)} > : {formstr.format(a=v)}",
                   filepath=filepath, root=root)
    prints(filepath=filepath, root=root)


def print_amplitudes(theta_list, noa, nob, nva, nvb,
                     threshold=1e-2, filepath=cf.log):
    """Function
    Print out amplitudes of CCSD

    Author(s): Takashi Tsuchimochi
    """
    prints("\n----Amplitudes----")
    ### print singles amplitudes ###
    ia = 0
    for a in range(nva):
        aa = a + 1 + noa
        for i in range(noa):
            ii = i + 1
            if abs(theta_list[ia]) > threshold:
                prints(f"{ii}a -> {aa}a : {theta_list[ia]:2.10f}",
                       filepath=filepath)
            ia += 1
    for a in range(nvb):
        aa = a + 1 + nob
        for i in range(nob):
            ii = i + 1
            if abs(theta_list[ia]) > threshold:
                prints(f"{ii}b -> {aa}b : {theta_list[ia]:2.10f}",
                       filepath=filepath)
            ia += 1
    ### print doubles amplitudes ###
    ijab = ia
    for b in range(nva):
        bb = b + 1 + noa
        for a in range(b):
            aa = a + 1 + noa
            for j in range(noa):
                jj = j + 1
                for i in range(j):
                    ii = i + 1
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}a {jj}a -> {aa}a {bb}a : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1

    ### ab -> ab ###
    for b in range(nvb):
        bb = b + 1 + nob
        for a in range(min(b+1, nva)):
            aa = a + 1 + noa
            for j in range(nob):
                jj = j + 1
                for i in range(j+1):
                    ii = i + 1
                    # b > a, j > i
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1
                for i in range(j+1, noa):
                    ii = i + 1
                    # b > a, i > j
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1
        for a in range(min(b+1, nva), nva):
            aa = a + 1 + noa
            for j in range(nob):
                jj = j + 1
                for i in range(j+1):
                    ii = i + 1
                    # a > b, j > i
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1
                for i in range(j+1, noa):
                    ii = i + 1
                    # a > b, i > j
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1

    ### bb -> bb ###
    for b in range(nvb):
        bb = b + 1 + nob
        for a in range(b):
            aa = a + 1 + nob
            for j in range(nob):
                jj = j + 1
                for i in range(j):
                    ii = i + 1
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{ii}b {jj}b -> {aa}b {bb}b :"
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1
    prints("------------------")


def print_amplitudes_listver(theta_list, noa, nob, nva, nvb, occ_list,
                     threshold=1e-2, filepath=cf.log, name="Amplitudes"):
    """Function
    Print out amplitudes of CCSD

    Author(s): Takashi Tsuchimochi
    """
    norbs = noa + nva
    assert norbs == nob + nvb 

    vir_list = [i for i in range(2*norbs) if i not in occ_list]
    occ_list_a = [i//2 for i in occ_list if i%2 == 0]
    occ_list_b = [i//2 for i in occ_list if i%2 == 1]
    vir_list_a = [i//2 for i in vir_list if i%2 == 0]
    vir_list_b = [i//2 for i in vir_list if i%2 == 1]
    prints(f"\n----{name}----")
    ### print singles amplitudes ###
    ia = 0
    ### alpha ###
    for a in vir_list_a:
        for i in occ_list_a:
            if abs(theta_list[ia]) > threshold:
                prints(f"{i}a -> {a}a : {theta_list[ia]:2.10f}",
                       filepath=filepath)
            ia += 1

    ### beta ###
    for a in vir_list_b:
        for i in occ_list_b:
            if abs(theta_list[ia]) > threshold:
                prints(f"{i}b -> {a}b : {theta_list[ia]:2.10f}",
                       filepath=filepath)
            ia += 1

    ### print doubles amplitudes ###
    ijab = ia
    ### aa -> aa ###
    for a, b in itertools.combinations(vir_list_a, 2):
        for i, j in itertools.combinations(occ_list_a, 2):
            if abs(theta_list[ijab]) > threshold:
                prints(f"{i}a {j}a -> {a}a {b}a : "
                       f"{theta_list[ijab]:2.10f}",
                       filepath=filepath)
            ijab += 1

    ### ab -> ab ###
    for b in vir_list_b:
        for a in vir_list_a:
            for j in occ_list_b:
                for i in occ_list_a:
                    if abs(theta_list[ijab]) > threshold:
                        prints(f"{i}a {j}b -> {a}a {b}b : "
                               f"{theta_list[ijab]:2.10f}",
                               filepath=filepath)
                    ijab += 1

    ### bb -> bb ###
    for a, b in itertools.combinations(vir_list_b, 2):
        for i, j in itertools.combinations(occ_list_b, 2):
            if abs(theta_list[ijab]) > threshold:
                prints(f"{i}b {j}b -> {a}b {b}b : "
                       f"{theta_list[ijab]:2.10f}",
                       filepath=filepath)
            ijab += 1
    prints("------------------")


def print_amplitudes_adapt(theta_list, Quket, filepath=cf.log, name="Amplitudes"):
    """Function
    Print out amplitudes of ADAPT

    Author(s): Takashi Tsuchimochi
    """
    AA_BB = 0
    AAAA_BBBB = 1
    BABA_ABAB = 2
    AA = 3
    BB = 4
    AAAA = 5
    BBBB = 6
    ABAB = 7
    BABA = 8
    ABBA = 9
    BAAB = 10


    ndim = len(theta_list)
    prints(f"\n----{name}----")
    for k in range(0, ndim):
        
        b = Quket.adapt.b_list[k]
        a = Quket.adapt.a_list[k]
        j = Quket.adapt.j_list[k]
        i = Quket.adapt.i_list[k]
        
        if Quket.adapt.spin_list[k] == AA_BB:  ##single spin
            prints(f"{i}a -> {a}a : {theta_list[k]:2.10f}",
                   filepath=filepath)
            prints(f"{i}b -> {a}b : {theta_list[k]:2.10f}",
                   filepath=filepath)
        elif Quket.adapt.spin_list[k] == AAAA_BBBB:  ##same spin
            prints(f"{i}a {j}a -> {a}a {b}a : "
                   f"{theta_list[k]:2.10f}",
                   filepath=filepath)
            prints(f"{i}b {j}b -> {a}b {b}b : "
                   f"{theta_list[k]:2.10f}",
                   filepath=filepath)
        elif Quket.adapt.spin_list[k] == BABA_ABAB:  # different spin
            prints(f"{i}a {j}b -> {a}a {b}b : "
                   f"{theta_list[k]:2.10f}",
                   filepath=filepath)
            prints(f"{i}b {j}a -> {a}b {b}a : "
                   f"{theta_list[k]:2.10f}",
                   filepath=filepath)
        elif Quket.adapt.spin_list[k] == BABA:
            prints(f"{i}a {j}b -> {a}a {b}b : "
                   f"{theta_list[k]:2.10f}",
                   filepath=filepath)
        elif Quket.adapt.spin_list[k] == ABAB:
            prints(f"{i}b {j}a -> {a}b {b}a : "
                   f"{theta_list[k]:2.10f}",
                   filepath=filepath)
        elif Quket.adapt.spin_list[k] == AA:
            prints(f"{i}a -> {a}a : "
                   f"{theta_list[k]:2.10f}",
                   filepath=filepath)
        elif Quket.adapt.spin_list[k] == BB:
            prints(f"{i}b -> {a}b : "
                   f"{theta_list[k]:2.10f}",
                   filepath=filepath)
    prints("------------------")

def print_amplitudes_spinfree(theta_list, no, nv,
                              threshold=0.01, filepath=cf.log):
    """Function:
    Print out amplitudes of spin-free CCSD

    Author(s): Takashi Tsuchimochi
    """
    from quket.ansatze import get_baji

    prints("\n----Amplitudes----")
    ### print singles amplitudes ###
    ia = 0
    for a in range(nv):
        aa = a + 1 + no
        for i in range(no):
            ii = i + 1
            if abs(theta_list[ia]) > threshold:
                prints(f"{ii} -> {aa} : {theta_list[ia]}", filepath=filepath)
            ia += 1

    ### print doubles amplitudes ###
    for b in range(nv):
        bb = b + 1 + no
        for a in range(b):
            aa = a + 1 + no
            for j in range(no):
                jj = j + 1
                for i in range(j):
                    ii = i + 1
                    baji = get_baji(b, a, j, i, no) + ia
                    abji = get_baji(a, b, j, i, no) + ia
                    theta = theta_list[baji] + theta_list[abji]
                    if abs(theta) > threshold:
                        prints(f"{ii}a {jj}a -> {aa}a {bb}a : {theta:2.10f}",
                               filepath=filepath)
                        prints(f"{ii}b {jj}b -> {aa}b {bb}b : {theta:2.10f}",
                               filepath=filepath)

    ### ab -> ab ###
    for b in range(nv):
        bb = b + 1 + no
        for a in range(min(b+1, nv)):
            aa = a + 1 + no
            for j in range(no):
                jj = j + 1
                for i in range(j+1):
                    ii = i + 1
                    # b > a, j > i
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[baji]:2.10f}",
                               filepath=filepath)
                for i in range(j+1, no):
                    ii = i + 1
                    # b > a, i > j
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[baji]:2.10f}",
                               filepath=filepath)
        for a in range(b+1, nv):
            aa = a + 1 + no
            for j in range(no):
                jj = j + 1
                for i in range(j+1):
                    ii = i + 1
                    # a > b, j > i
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[baji]:2.10f}",
                               filepath=filepath)
                for i in range(j+1, no):
                    ii = i + 1
                    # a > b, i > j
                    baji = get_baji(b, a, j, i, no) + ia
                    if abs(theta_list[baji]) > threshold:
                        prints(f"{ii}a {jj}b -> {aa}a {bb}b : "
                               f"{theta_list[baji]:2.10f}",
                               filepath=filepath)
    prints("------------------")


def printmat(A, name=None, eig=None, mmax=10, filepath=cf.log, n=None, m=None, format="11.7f", root=0):
    """Function:
    Print out A in a readable format.

        A         :  1D or 2D numpy array of dimension
        eig       :  Given eigenvectros A[:,i], eig[i] are corresponding eigenvalues (ndarray or list)
        filepath  :  file to be printed
        mmax      :  maxixmum number of columns to print for each block
        name      :  Name to be printed
        n,m       :  Need to be specified if A is a matrix,
                     but loaded as a 1D array
        format    :  Printing format
        root     : mpi rank (default is main rank, 0)

    Author(s): Takashi Tsuchimochi
    """
    if isinstance(A, list):
        dimension = 1
    elif isinstance(A, np.ndarray):
        dimension = A.ndim
    if dimension == 0 or dimension > 2:
        error("Neither scalar nor tensor is printable with printmat.")

    if mpi.rank != root:
        return

    old_stdout = sys.stdout
    if filepath is not None:
        sys.stdout = open(filepath, 'a')
    if True:
    #with open(filepath, 'a') as f:
        print()#file=f)
        if name is not None:
            print(name, )#file=f)
        #if cf.debug:
        ### set precisions

        if format.find('f') != -1:
            ### Float
            style = "f"
            idx_f = format.find('f') 
            idx = format.find('.')
            digits = int(format[:idx]) 
            decimal = int(format[idx+1:idx_f])
        elif format.find('d') != -1:
            style = "d"
            idx = format.find('d')
            digits = int(format[:idx])
            decimal = ''
        else:
            error(f'format={format} not supported in printmat.')

        if dimension == 2:
            n, m = A.shape
            imax = 0
            while imax < m:
                imin = imax + 1
                imax = imax + mmax
                if imax > m:
                    imax = m
                print()#file=f)
                if eig is None:
                    print("           ", end="", )#file=f)
                else:
                    print("  eig  |", end="", )#file=f)
                    
                for i in range(imin-1, imax):
                    if eig is None:
                        print(f"{i:{digits-6}d}          ", end="", )#file=f)
                    else:
                        print(f" {eig[i]:{format}}  |", end="",)
                print()#file=f)
                if eig is not None:
                    print("       ",end="",)
                    for i in range(imin-1, imax):
                        print(f"---------------",end="",)
                print()#file=f)
                for j in range(n):
                    print(f" {j:4d}  ", end="", )#file=f)
                    for i in range(imin-1, imax):
                        print(f"  {A[j][i]:{format}}  ", end="", )#file=f)
                    print()#file=f)
        elif dimension == 1:
            if n is None or m is None:
                if isinstance(A, list):
                    n = len(A)
                    m = 1
                elif isinstance(A, np.ndarray):
                    n = A.size
                    m = 1
            imax = 0
            while imax < m:
                imin = imax + 1
                imax = imax + mmax
                if imax > m:
                    imax = m
                if eig is None:
                    print("           ", end="", )#file=f)
                else:
                    print(" eig:  ", end="", )#file=f)
                    
                for i in range(imin-1, imax):
                    if eig is None:
                        print(f"{i:{digits-6}d}          ", end="", )#file=f)
                    else:
                        print(f"  {eig[i]:{format}}  ", end="",)
                print()#file=f)
                for j in range(n):
                    if n > 1:
                        print(f" {j:4d}  ", end="", )#file=f)
                    else:
                        print(f"       ", end="", )#file=f)
                    for i in range(imin-1, imax):
                        print(f"  {A[j + i*n]:{format}}  ", end="", )#file=f)
                    print()#file=f)
        print()#file=f)
    sys.stdout = old_stdout


def printmath(A, mmax=10, filepath=cf.log, name=None, n=None, m=None, format="16.12f"):
    """Function:
    Print out A in a readable format.

        A         :  1D or 2D numpy array of dimension
        filepath  :  file to be printed
        mmax      :  maxixmum number of columns to print for each block
        name      :  Name to be printed
        n,m       :  Need to be specified if A is a matrix,
                     but loaded as a 1D array

    Author(s): Takashi Tsuchimochi
    """
    if isinstance(A, list):
        dimension = 1
    elif isinstance(A, np.ndarray):
        dimension = A.ndim
    if dimension == 0 or dimension > 2:
        error("Neither scalar nor tensor is printable with printmat.")

    if not mpi.main_rank:
        return

    old_stdout = sys.stdout
    if filepath is not None:
        sys.stdout = open(filepath, 'a')
    if True:
    #with open(filepath, 'a') as f:
        print()
        if name is not None:
            print(name, '=')
        #if cf.debug:
        

        if format.find('f') != -1:
            ### Float
            style = "f"
            idx_f = format.find('f') 
            idx = format.find('.')
            digits = int(format[:idx]) 
            decimal = int(format[idx+1:idx_f])
        elif format.find('d') != -1:
            style = "d"
            idx = format.find('d')
            digits = int(format[:idx])
            decimal = ''
        else:
            error(f'format={format} not supported in printmat.')

        if dimension == 2:
            n, m = A.shape
            prints(n,m)
            if m%8 == 0:
                k=m//8 - 1 
            else:
                k=m//8
            print('{', end='')
            for i in range(n-1):
                print('{', end='')
                for j in range(k):
                    for jk in range(8):
                        print(f"{A[i, j*8+jk]:{format}}, ", end='')
                    print()
                j = k
                for jk in range((m-1)%8):
                    print(f"{A[i, j*8+jk]:{format}}, ", end='')
                print(f"{A[i, j*8+(m-1)%8]:{format}}}}, ")
            i = n-1
            print('{', end='')
            for j in range(k):
                for jk in range(8):
                    print(f"{A[i, j*8+jk]:{format}}, ", end='')
                print()
            j = k
            for jk in range((m-1)%8):
                print(f"{A[i, j*8+jk]:{format}}, ", end='')
            print(f"{A[i, j*8+(m-1)%8]:{format}} }}}}; ")

        elif dimension == 1:
            if n is None or m is None:
                if isinstance(A, list):
                    n = len(A)
                    m = 1
                elif isinstance(A, np.ndarray):
                    n = A.size
                    m = 1
            
            if m%8 == 0:
                k=m//8 - 1 
            else:
                k=m//8
            print('{', end='')
            for i in range(n-1):
                print('{', end='')
                for j in range(k):
                    for jk in range(8):
                        print(f"{A[i*m + j*8+jk]:{format}}, ", end='')
                    print()
                j = k
                for jk in range((m-1)%8):
                    print(f"{A[i*m + j*8+jk]:{format}}, ", end='')
                print(f"{A[i*m + j*8+(m-1)%8]:{format}}}}, ")
            i = n-1
            print('{', end='')
            for j in range(k):
                for jk in range(8):
                    print(f"{A[i*m + j*8+jk]:{format}}, ", end='')
                print()
            j = k
            for jk in range((m-1)%8):
                print(f"{A[i*m + j*8+jk]:{format}}, ", end='')
            print(f"{A[i*m + j*8+(m-1)%8]:{format}} }}}}; ")
    sys.stdout = old_stdout
            
                
