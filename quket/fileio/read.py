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
import os
import re
import string

from quket import config as cf
from quket.fileio import prints, error
from quket.utils import isint, isfloat, iscomplex
import numpy as np


#############################
###    Default options    ###
#############################
# How to write; 'option's name': 'attribute's name'
integers = {
        #----- For Config -----
        "print_level": "print_level",
        "mix_level": "mix_level",
        "kappa_to_t1": "Kappa_to_T1",
        "nterm": "nterm",
        "dimension": "dimension",
        "nroots": "nroots",
        #----- For General -----
        "n_electrons": "n_active_electrons",
        "n_orbitals": "n_active_orbitals",
        "n_core_orbitals": "n_core_orbitals",
        "multiplicity": "multiplicity",
        "ms": "Ms",
        "charge": "charge",
        "rho": "rho",
        "mix_level": "mix_level",
        "maxiter": "maxiter",
        "oo_maxiter": "oo_maxiter",
        "hubbard_nx": "hubbard_nx",
        "hubbard_ny": "hubbard_ny",
        "layers": "pauli_list_nlayers",
        #----- For VQE -----
        "ds": "DS",
        #----- For Symmetry-Projection -----
        "spin": "spin",
        #----- For ADAPT -----
        "adapt_max": "max",
        "max_ncnot": "max_ncnot",
        "adapt_svqe": "svqe",
        #----- For MBE -----
        "min_use_core": "min_use_core",
        "max_use_core": "max_use_core",
        "min_use_secondary": "min_use_secondary",
        "max_use_secondary": "max_use_secondary",
        "n_secondary_orbitals": "n_secondary_orbitals",
        "mbe_comm_color": "color",
        #----- For OO -----
        "oo_maxiter": "oo_maxiter",
        }
floats = {
        #----- For Config -----
        "eps": "eps",
        "theta_threshold": "theta_threshold",
        #----- For General -----
        "lambda": "constraint_lambda",
        "lambda_s2": "constraint_lambda_S2_expectation",
        "lambda_sz": "constraint_lambda_Sz",
        "hubbard_u": "hubbard_u",
        "gtol": "gtol",
        "ftol": "ftol",
        "print_amp_thres": "print_amp_thres",
        "s2shift": "s2shift",
        "regularization": "regularization",
        #----- For QITE -----
        "timestep": "dt", "db": "dt", "dt": "dt",
        "hamiltonian_threshold": "hamiltonian_threshold",
        "truncate": "hamiltonian_threshold",
        #----- For Adapt-VQE -----
        "adapt_eps": "eps",
        #----- For OO -----
        "oo_ftol": "oo_ftol",
        "oo_gtol": "oo_gtol",
        }
bools = {
        #----- For Config -----
        "print_fci": "print_fci",
        "approx_exp": "approx_exp",
        "debug": "debug",
        "taper_off": "do_taper_off",
        "fswap": "fswap",
        "fast_evaluation": "fast_evaluation",
        "numerical_derivative": "finite_difference",
        "finite_difference": "finite_difference",
        #----- For General -----
        "run_fci": "run_fci",
        "run_mp2": "run_mp2",
        "run_ccsd": "run_ccsd",
        "run_casscf": "run_casscf",
        "symmetry": "symmetry",
        "symmetry_pauli": "symmetry_pauli",
        "qubit_excitation": "qubit_excitation", "qe": "qubit_excitation",
        "disassemble_pauli":"disassemble_pauli",
        "run_qubitfci": "run_qubitfci",
        "qubitfci": "run_qubitfci",
        "fci2qubit": "run_qubitfci",
        "hubbard_ao": "hubbard_ao",
        "post_general": "post_general",
        "post_hermitian_conjugated": "post_hermitian_conjugated",
        "spinfree":"spinfree",
        "opt":"geom_opt",
        "grad":"do_grad",
        "oo":"oo",
        # ----- For QITE -----
        "qlanczos": "qlanczos",
        "folded_spectrum": "folded_spectrum",
        "fs": "folded_spectrum",
        #----- For VQE -----
        "1rdm": "Do1RDM",
        "2rdm": "Do2RDM",
        "qse" : "DoQSE" ,
        #----- Symmetry-Projection -----
        "spinproj": "SpinProj",
        "post_spinproj": "post_SpinProj",
        #----- For Multi/Excited-state -----
        "act2act": "act2act_opt",
        #----- For MBE -----
        "calc_delta_later": "later",
        "mbe_exact": "mbe_exact",
        "mbe_oo": "mbe_oo",
        #----- For Adapt-VQE------#
        "adapt_lucc_nvqe": "lucc_nvqe",
        }
strings = {
        #----- For Config -----
        "units": "units",
        "opt_method": "opt_method",
        "pyscf_guess": "pyscf_guess",
        "kappa_guess": "kappa_guess",
        "theta_guess": "theta_guess",
        "npar": "nthreads",
        "intpq": "intpq",
        #----- For General -----
        "read": "read",
        "method": "method",
        "post_method": "post_method",
        "ansatz": "ansatz",
        "load": "load_file",
        "symmetry_subgroup": "symmetry_subgroup",
        "operator": "operator_basis",
        "mapping": "mapping",
        #----- For MBE -----
        "mo_basis": "mo_basis",
        "mbe_correlator": "mbe_correlator",
        "mbe_oo_ansatz": "mbe_oo_ansatz",
        #----- For QITE -----
        "shift": "shift",
        #----- For Adapt-VQE -----
        "adapt_mode": "mode",
        "adapt_guess": "adapt_guess",
        "adapt_prop": "adapt_prop"
        }
strings_bools ={
        "from_vir": "from_vir",
        "msqite": "msqite",
        }
comment = {
        "comment": "comment"
        }


def read_multi(line):
    """
    Investigate line, which should have the following string format
        coef*|bit> + coef*|bit> + ...
    or
        coef*|bit> + coef*|bit> + ...  weight
    and extract coef and bit as lists (and weight as float).
    """
    if type(line) is list:
        tmp = [list(x) for x in line]
        list_line = [x for tmp_ in tmp for x in tmp_]
    else:
        list_line = list(line)
    # Deal with +
    indexes = [i for i, x in enumerate(list_line) if x == '+']
    for i in range(len(indexes)):
        ind = indexes[i]
        if ind > 0:
            while list_line[ind+1] == ' ':
                list_line.pop(ind+1)
                indexes = [x-1 for x in indexes]
            if list_line[ind-1] != ' ':
                list_line.insert(ind, ' ')
                indexes = [x+1 for x in indexes]
    # Deal with -
    indexes = [i for i, x in enumerate(list_line) if x == '-']
    for i in range(len(indexes)):
        ind = indexes[i]
        if ind > 0:
            while list_line[ind+1] == ' ':
                list_line.pop(ind+1)
                indexes = [x-1 for x in indexes]
            if list_line[ind-1] != ' ':
                list_line.insert(ind, ' ')
                indexes = [x+1 for x in indexes]            
    # Deal with *
    indexes = [i for i, x in enumerate(list_line) if x == '*']
    for i in range(len(indexes)):
        ind = indexes[i]
        if ind > 0:
            while list_line[ind+1] != ' ':
                list_line.insert(ind+1,' ')
                indexes = [x+1 for x in indexes]
            if list_line[ind-1] != ' ':
                list_line.insert(ind, ' ')
                indexes = [x+1 for x in indexes]                        
    strlist= "".join(list_line)
    value = strlist.strip().split(' ') 
    '''
    value = 
    ['+0.5',
     '*',
     '00001111',
     '+0.5',
     '*',
     '11110000',
     '+0.25',
     '*',
     '00011111',
     '0.25']
    '''
    prints(value)
    if len(value) == 1:
        ### Should have the regular integer form
        ### e.g.) 00001111 
        try:
            state = [[1, int(f"0b{value[0].replace('|','').replace('>','')}",2)]]
            weight = 1
        except:
            ### Failed: this line is likely to be a different option
            return False, False

    elif len(value) == 2:
        ### Should have the regular integer form
        ### e.g.) 00001111 0.5 
        try:
            state = [[1, int(f"0b{value[0].replace('|','').replace('>','')}",2)]]
            weight = float(value[1])
        except:
            ### Failed: this line is likely to be a different option
            return False, False
    elif len(value) % 3 ==0:
        ### Should have the entangled form
        ### e.g.) 0.5*00001111+0.3*00110011
        state = []
        for i in range((len(value)) // 3):
            try:        
                state.append([complex(value[3*i]), int(f"0b{value[3*i+2].replace('|','').replace('>','')}",2)])
                weight = float(1)
            except:
                ### Failed: this line is likely to be a different option
                return False, False
    elif (len(value) - 1) % 3 ==0:
        ### Should have the entangled form
        ### e.g.) 0.5*00001111+0.3*00110011 0.5 
        state = []
        try:
            weight = float(value[-1])
        except:
            ### Failed: this line is likely to be a different option
            return False, False
            
        for i in range((len(value) - 1) // 3):
            try:        
                state.append([complex(value[3*i]), int(f"0b{value[3*i+2].replace('|','').replace('>','')}",2)])
            except:
                ### Failed: this line is likely to be a different option
                return False, False
    else:
        return False, False
    # Check if state is single determinant
    if len(state) == 1:
        state = state[0][1]
    return state, weight

def read_pauli_list(lines):
    str_pauli_list = []
    for i, line in enumerate(lines):
        line = re.sub(r" +", r" ", line).rstrip("\n").strip()
        if line[-1] == "\\":
            line = line.replace("\\","")

        if i == 0:
            key, line = line.split("=",1)
            key = key.strip().lower()
            if key != "pauli_list":
                error("Something is wrong.")
        line = line.split(",")
        my_list = [line[i].strip() for i in range(len(line))]

        str_pauli_list.extend(my_list)
    return str_pauli_list

def get_line(line, subsequent=False):
    # Removal of certain symbols and convert multi-space to single-space.
    remstr = ":,'"
    line = line.translate(str.maketrans(remstr, " "*len(remstr), ""))
    line = re.sub(r" +", r" ", line).rstrip("\n").strip()
    if len(line) == 0:
        # Ignore blank line.
        return
    if line[0] in ["!", "#"]:
        # Ignore comment line.
        return
    if line[-1] == "\\":
        line = line.replace("\\","")
        cont = True
    else:
        cont = False

    if "=" not in line:
        if subsequent:
            ### This line is supposed to be continued from the previous line
            key = None
            value = line
        else:
            return line
#        key = None
#        value = line
#        if value == "@@@" or value == "clear":
#            return value
    else:
        key, value = line.split("=",1)
        key = key.strip()
    value = value.strip()
    value = value.split(" ")
    if len(value) == 1:
        return key, value[0], cont
    else:
        return key, value, cont


def replace_params(value, params):
    if type(value) == list:
        value_ = []
        for v in value:
            v_ = v
            for key, param in params.items():
                v_ = v_.replace(key, param)
            value_.append(v_)
        return value_
    else:
        value_ = value
        for key, param in params.items():
            value_ = value_.replace(key, param)
        return value_

def read_input():
    """Function:
    Open ***.inp and read options.
    The read options are stored as global variables in config.py.

    Return:
        List: Whether input is read to EOF

    Notes:
        Here is the list of allowed options:

        method (str):           Name of method.
        multiplicity (int):     Spin-multiplicity for determinant, i.e. Nalpha - Nbeta + 1
        ms (int):               Nalpha - Nbeta. If set, overwrites multiplicity  (Default: None)
        spin (int):             Spin-multiplicity for post-HF in PYSCF (Default: ms + 1 or multiplicity)
        ms (int):               Nalpha - Nbeta. If set, overwrites multiplicity  (Default: None)
        charge (int):           Charge
        rho (int):              Trotter-steps
        run_fci (bool):         Whether to run fci with pyscf
        print_level (int):      Pritinging level
        opt_method (str):       Optimization method for VQE
        mix_level (int):        Number of orbitals to break spin-symmetry
        eps (float):            Step-size for numerical gradients
        gtol (float):           Convergence criterion of VQE (grad)
        ftol (float):           Convergence criterion of VQE (energy)
        print_amp_thres (int):  Printing level of amplitudes (theta_list, etc)
        maxiter (int):          Maximum iteration number of VQE
        pyscf_guess (str):      Guess for pyscf calculation (miao, read)
        kappa_guess (str):      Guess for initial kappa (zero, mix, random, read)
        theta_guess (str):      Guess for initial theta (zero, random, read)
        1rdm (bool):            Whether to compute 1RDM
        2rdm (bool):            Whether to compute 2RDM
        qse (bool):             Whether to compute QSE
        kappa_to_t1 (bool):     Whether to use kappa_list for T1 amplitudes.
        n_electrons (int):      Number of electrons
        n_orbitals (int):       Number of orbitals (x2 = number of qubits)
        spinproj (bool):        Whether to perform spin-projection
        spin (int):             Target spin state of spin-projection
        euler (ints):           Euler angles for spin-projection
        ds (bool):              Ordering of Singles/Doubles. If True,
                                Exp[T2][T1], if False, Exp[T1][T2]
        lambda (float):         Lagrange multiplier for spin-constrained calculation
        geometry (dict):        Standard cartesian geometry
        det, determinant (int): Initial determinant (like 000111)
        multi (ints):           Determinants for multi-determinantal calculation
        excited (ints):         Initial determinants for excited calculations
        npar, nthreads (int):   Number of threads
        hubbard_u (float):      Hubbard U
        hubbard_nx (int):       Hubbard lattice size for x-direction
        hubbard_ny (int):       Hubbard lattice size for y-direction
        hamiltonian:            Either fermionic or qubit Hamiltonian (FermionOperator, QubitOperator) or their string format

        Putting '@@@' in lines separates multiple jobs in one input file.
        (the options from previous jobs will be used unless redefined)
    """
    from quket.utils.utils import chkbool
    # Read input lines.
    with open(cf.input_file, "r") as f:
        lines = f.readlines()

    prints('\n##################### INPUT ########################')
    for line in lines:
        prints(line, end='')
    prints('####################################################\n')


    ######################################
    ###    Start reading input file    ###
    ######################################
    kwds_list = []
    kwds = {}
    params = {}
    # Values forced to be initialized
    # Add 'data_directory' for 'chemical'.
    kwds["data_directory"] = cf.input_dir
    # Add 'read' and initialize 
    kwds["read"] = False
    i = 0
    while i < len(lines):
        line = get_line(lines[i])
        if isinstance(line, str):
            key_ = line.strip()
            key = key_.lower()
            value = ""
        elif isinstance(line, tuple):
            key_, value, cont = line
            if key_ is not None:
                key = key_.lower()
            else:
                key = None
            i_org = i
            while cont:
                #value[-1].replace("\\",'')
                i += 1
                line = get_line(lines[i], True)
                _, value_, cont = line
                value.extend(value_)

        else:
            i += 1
            continue
        ### Check any parameters in value
        value = replace_params(value, params)
        ################
        # Easy setting #
        ################
        if key in integers:
            kwds[integers[key]] = int(value)
        elif key in floats:
            kwds[floats[key]] = float(value)
        elif key in bools:
            kwds[bools[key]] = chkbool(value)
        elif key in strings:
            if type(value) is list:
                kwds[strings[key]] = list(map(str.lower, value))
            else:
                kwds[strings[key]] = value.lower()
        elif key in strings_bools:
            if type(value) is str:
                kwds[strings_bools[key]] = value.lower()
            elif type(value) is bool:
                kwds[strings_bools[key]] = str(value).lower()
            else:
                error(f"Type({key}={value}) is {type(value)} but it must be either str or bool.")
            
        elif key in comment:
            kwds[comment[key]] = " ".join(value)
        ###########
        # General #
        ###########
        elif key == "basis":
            if len(value.split(" ")) == 1:
                # e.g.) basis = sto-3g
                kwds[key] = value
            else:
                # e.g.) basis = H sto-3g, O 6-31g
                atoms = value[::2]
                atom_basis = value[1::2]
                kwds[key] = dict(zip(atoms, atom_basis))
        elif key == "geometry":
            # See if this is Z matrix or Cartesian
            test_line = get_line(lines[i+1])
            atom_info = test_line.split(" ")
            if len(atom_info) == 4:
                # Cartesian
                # e.g.) geometry:
                #       H 0 0 0
                #       H 0 0 0.74

                geom = []
                for j in range(i+1, len(lines)):
                    next_line = get_line(lines[j])
                    if not isinstance(next_line, str):
                        break
                    atom_info = next_line.split(" ")
                    if len(atom_info) != 4\
                       or atom_info[0] not in cf.PeriodicTable:
                        break
                    atom = atom_info[0]
                    xyz = tuple(map(float, atom_info[1:4]))
                    geom.append((atom, xyz))
            elif len(atom_info) == 1:
                # Z-matrix
                # e.g.) geometry:
                #       N 
                #       X 1 1.01393
                #       H 1 1.01393 2 106.429
                #       H 1 1.01393 2 106.429 3  120.0
                #       H 1 1.01393 2 106.429 3 -120.0
                natom = 0
                zmat = ""
                for j in range(i+1, len(lines)):
                    natom += 1
                    next_line = get_line(lines[j])
                    if not isinstance(next_line, str):
                        break
                    atom_info = next_line.split(" ")
                    if len(atom_info) != min(2*natom-1, 7)\
                       or atom_info[0] not in cf.PeriodicTable:
                        break
                    ### string in pyscf format
                    zmat += f"{next_line}\n" 
                from pyscf.gto.mole import from_zmatrix
                geom = from_zmatrix(zmat)

            ### Check symbol and remove ghost atoms 
            for x in geom[:]:
                if x[0] == 'X':
                    # Remove
                    geom.remove(x)

            # Set geometry and skip molecule's information lines.
            kwds[key] = geom
            i = j - 1
        elif key in ["det", "determinant"]:
            # e.g.) 000011
            #if type(value) is int:
            #    if value.isdecimal() and int(value) >= 0:
            #        kwds["det"] = int(f"0b{value}", 2)
            #    else:
            #        error(f"Invalid determinant description '{value}'")
            #elif type(value) is list:
            # e.g.) 0.5 * 000011  + 0.5 * 001100
            try:
                if isinstance(value, str):
                    prints(value)
                    if value.lower() == 'random':
                        kwds["det"] = 'random'
                        state = True
                    else:
                        state, weight = read_multi(value)
                        kwds["det"] = state
                elif isinstance(value, list):
                    state, weight = read_multi(''.join(value))
                    kwds["det"] = state
            except:
                try:
                    if value.lower() == 'random':
                        kwds["det"] = 'random'
                        state = True
                    else:
                        prints("EHRER")
                        error(f"Invalid determinant description '{value}'")
                except:
                    prints("OR EHRER", ''.join(value))
                    error(f"Invalid determinant description '{value}'")
            if not state:
                error(f"The format of det = {value} is not correct.\n Use the following format:\n   det = coef * bit + coef * bit + ...")
            #else:
            #    error(f"Invalid determinant description '{value}'")
        #######################
        # Symmetry-Projection #
        #######################
        elif key == "euler":
            # e.g.) euler = -1
            # e.g.) euler = 1 1
            # e.g.) euler = 1 -2 3
            x = 0
            y = -1
            z = 0
            if len(value) == 1:
                y = int(value)
            elif len(value) == 2:
                x, y = map(int, value)
            elif len(value) == 3:
                x, y, z = map(int, value)
            else:
                error("Format for Euler angle is wrong")
            kwds["euler_ngrids"] = [x, y, z]
        elif key == "nproj":
            kwds["number_ngrids"] = int(value)
            kwds["NumberProj"] = cf.number_ngrids > 1
        ######################
        # Multi/Excite-state #
        ######################
        #elif key == "multi":
        #    # e.g.) multi:
        #    #       000011 0.25
        #    #       000110 0.25
        #    #       001001 0.25
        #    #       001100 0.25
        #    states = []
        #    weights = []
        #    for j in range(i+1, len(lines)):
        #        next_line = get_line(lines[j])
        #        if not isinstance(next_line, str):
        #            break
        #        if len(next_line.split(" ")) != 2:
        #            break
        #        state, weight = next_line.split(" ")
        #        try:
        #            states.append(int(f"0b{state}", 2))
        #            weights.append(float(weight))
        #        except:
        #            ### Failed: this line is likely to be a different option
        #            j -= 1
        #            break
        #    # Set states/weights and skip multi's information lines.
        #    kwds["init_states"] = states
        #    kwds["weights"] = weights
        #    i = j - 1
        elif key == "multi":
            # e.g.) multi_:
            #       000011 0.25
            #       0.5 * 000110 - 0.8* 001001 0.25
            #       1 * 101100 +0.2*000000-0.14*1000000 0.35
            #  should be put as 
            #  states = [
            #            [["000011", 1]], 
            #            [["000110",0.5], ["001001",-0.8]]
            #            [["101100", 1], ["000000",+0.2], ["100000",-0.14]]
            #           ]
            #  weights = [0.25, 0.25, 0.35]
            states = []
            weights = []
            for j in range(i+1, len(lines)):
                next_line = get_line(lines[j])
                if not isinstance(next_line, str):
                    break
                state, weight = read_multi(next_line)
                if state is not None:
                    states.append(state)
                    weights.append(weight)
                else:
                    j -= 1 
                    break
            kwds["init_states_info"] = states
            kwds["weights"] = weights
        elif key == "excited":
            # e.g.) excited:
            #       000110
            #       001001
            #       001100
            excited_states = []
            for j in range(i+1, len(lines)):
                next_line = get_line(lines[j])
                if not isinstance(next_line, str):
                    break
                state = next_line.strip()
                try:
                    excited_states.append(int(f"0b{state.replace('|','').replace('>','')}", 2))
                except:
                    ### Failed: this line is likely to be a different option
                    j -= 1
                    break
            # Set excited_states and skip excited_states' information lines.
            kwds["excited_states"] = excited_states
            i = j - 1
        elif key == "local":
            value = [int(x) for x in value]
            kwds["local"] = value
        elif key == "alter":
            value = [int(x.replace('[','').replace(']','')) for x in value]
            if len(value)%2 == 1: # Not pair! 
                raise Exception(f'Switching orbitals in alter have to be paired: {value}\n'
                                f'Use either one of the following formats:\n'
                                f'  alter = [[2,5], [6,8], ...]\n'
                                f'  alter = [2, 5, 6, 8, ...]')
            kwds["alter_pairs"] = value
        #############
        # Otherwise #
        #############
        elif key == "include":
            # Including excitation variety.
            # e.g.) include:
            #       c->a/s
            #       cc->aa/as/ss
            #       aa->as/ss
            # Note; c = core space, a = active space, s = secondary space
            # c->a/s means from core to active/secondary excitations.
            # cc->aa/as/ss means rom core
            # to active-active/active-secondary/secondary-secondary excitations.
            includes = {}
            for j in range(i+1, len(lines)):
                next_line = get_line(lines[j])
                if not isinstance(next_line, str):
                    break
                from_, to = next_line.split("->")
                includes[from_] = to
            kwds["include"] = includes
            i = j - 1
        elif key == "read":
            # Read quket data
            kwds["read"] = value
        elif key == "@@@":
            # Go next job input.
            # Set current 'kwds' to 'kwds_list'.
            # Note; 'kwds' may be overwritten at next job
            #       for the sake of simplicity in the input file.
            kwds_list.append(kwds.copy())
        elif key == "clear":
            # Clear all the keywords
            kwds = {}
            kwds["data_directory"] = cf.input_dir
        elif key == "hamiltonian": 
            if type(value) is not list:
                value = [value]
            kwds["user_defined_hamiltonian"] = ' '.join(value) 
        elif key == "pauli_list": 
            ### Need to deal with input with a more sophisticated manner
            value = read_pauli_list(lines[i_org:i+1])
            if type(value) is not list:
                error(f"pauli_list must be a list\nyour entry: {value}")
            kwds["user_defined_pauli_list"] = value
            ### Since pauli_list is provided, this is user-defined ansatz.
            kwds["ansatz"] = "user-defined"
        elif value != "":
            try:
                params[key_] = str(eval("".join(value)))
            except:
                error(f"No option '{key_}'")

        # Go next line
        i += 1

    kwds_list.append(kwds)
    return kwds_list



def set_config(kwds, Quket):
    for key, value in kwds.items():
        #----- For General -----
        if key == "fswap":
            cf.fswap = value
        elif key == "fast_evaluation":
            cf.fast_evaluation = value
        #----- For VQE -----
        elif key == "approx_exp":
            cf.approx_exp = value
        #----- For QITE -----
        elif key == "nterm":
            cf.nterm = value
        elif key == "dimension":
            cf.dimension = value
        #----- For System -----
        elif key == "debug":
            cf.debug = value
        elif key == "units":
            cf._units = value
        elif key == "npar":
            prints("npar option is obsolete! Use -nt option instead")
            error(f" Example) mpirun -np 12 python3.8 main.py {input} -nt {value}")
        elif key == "intpq":
            if len(value) != 3:
                error("Format for IntPQ is wrong")
            cf.IntPQ = []
            cf.IntPQ.append(int(value[0]))
            cf.IntPQ.append(int(value[1]))
            cf.IntPQ.append(float(value[2]))
        elif key == "theta_threshold":
            cf.theta_threshold = value
        elif key == "read":
            if type(value) == bool: 
                # Default setting reading input.qkt. Has to have input file to give the name
                pass
            elif type(value) == str:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    cf.qkt = cf.input_dir + value
                    value = True


    if Quket.method == "mbe":
        cf.SaveTheta = False

def read_input_command_line(kwds_):
    """Function
    Screen keywords input by command lines.
    Some keywords need special treatments for re-formatting.
    (including euler, multi, excited, include)
    """
    from quket.utils.utils import chkbool
    # Values forced to be initialized
    # Add 'data_directory' for 'chemical'.
    kwds = {}
    kwds["data_directory"] = cf.input_dir
    i = 0
    for key_, value in kwds_.items():
        key = key_.lower()
        ################
        # Easy setting #
        ################
        if value is None:
            kwds[key] = None
            continue
        if key in integers:
            kwds[integers[key]] = int(value)
        elif key in floats:
            kwds[floats[key]] = float(value)
        elif key in bools:
            kwds[bools[key]] = chkbool(value)
        elif key in strings:
            if type(value) is list:
                kwds[strings[key]] = list(map(str.lower, value))
            else:
                kwds[strings[key]] = value.lower()
        elif key in strings_bools:
            if type(value) is str:
                kwds[strings_bools[key]] = value.lower()
            elif type(value) is bool:
                kwds[strings_bools[key]] = str(value).lower()
            else:
                error(f"Type({key}={value}) is {type(value)} but it must be either str or bool.")
        elif key in comment:
            kwds[comment[key]] = " ".join(value)
        ###########
        # General #
        ###########
        elif key == "basis":
            if len(value.split(" ")) == 1:
                # e.g.) basis = sto-3g
                kwds[key] = value
            else:
                # e.g.) basis = H sto-3g, O 6-31g
                atoms = value[::2]
                atom_basis = value[1::2]
                kwds[key] = dict(zip(atoms, atom_basis))
        elif key == "geometry":
            if value is None:
                kwds["geometry"] = None
                continue
            # See if this is list or str 
            if type(value) is list:
                kwds[key] = value
            elif type(value) is str:
                ### Need to convert str format to list 
                # check if this is zmat 
                value = value.replace(';','\n')
                x = value.split('\n')
                length_first_list =  len(x[0].strip().split())
                if length_first_list == 1:
                    geom_format = 'zmat'
                elif length_first_list == 4:
                    geom_format = 'xyz'
                else:
                    prints(f'Warning: Unrecognized geometry format:\n {value}')
                if length_first_list == 4:
                    # Cartesian
                    # e.g.) geometry:
                    #       H 0 0 0
                    #       H 0 0 0.74

                    geom = []
                    for j in range(len(x)):
                        next_line = get_line(x[j])
                        if not isinstance(next_line, str):
                            break
                        atom_info = next_line.split(" ")
                        if len(atom_info) != 4\
                           or atom_info[0] not in cf.PeriodicTable:
                            break
                        atom = atom_info[0]
                        xyz = tuple(map(float, atom_info[1:4]))
                        geom.append((atom, xyz))
                elif length_first_list == 1:
                    # Z-matrix
                    # e.g.) geometry:
                    #       N 
                    #       X 1 1.01393
                    #       H 1 1.01393 2 106.429
                    #       H 1 1.01393 2 106.429 3  120.0
                    #       H 1 1.01393 2 106.429 3 -120.0
                    natom = 0
                    zmat = ""
                    for j in range(len(x)):
                        natom += 1
                        next_line = get_line(x[j])
                        if not isinstance(next_line, str):
                            break
                        atom_info = next_line.split(" ")
                        if len(atom_info) != min(2*natom-1, 7)\
                           or atom_info[0] not in cf.PeriodicTable:
                            break
                        ### string in pyscf format
                        zmat += f"{next_line}\n"
                    from pyscf.gto.mole import from_zmatrix
                    geom = from_zmatrix(zmat)

                ### Check symbol and remove ghost atoms 
                for x in geom[:]:
                    if x[0] == 'X':
                        # Remove
                        geom.remove(x)

                # Set geometry and skip molecule's information lines.
                kwds[key] = geom
            else:
                prints(f'Warning: Unrecognized geometry format {type(value)}\n {value}')
        elif key in ["det", "determinant"]:
            if value is None:
                kwds["det"] = None
                continue
            #if value.isdecimal() and int(value) >= 0:
            #    # e.g.) 000011
            #    kwds["det"] = int(f"0b{value}", 2)
            #else:
            # e.g.) 0.5 * 000011  + 0.5 * 001100
            try:
                if value.lower() == 'random':
                    kwds["det"] = 'random'
                    state = True
                else:
                    state, weight = read_multi(value)
                    kwds["det"] = state
            except:
                try:
                    if value.lower() == 'random':
                        kwds["det"] = 'random'
                        state = True
                    else:
                        error(f"Invalid determinant description '{value}'")
                except:
                    error(f"Invalid determinant description '{value}'")
            if not state:
                error(f"The format of det = {value} is not correct.\n Use the following format:\n   det = coef * bit + coef * bit + ...")
        #######################
        # Symmetry-Projection #
        #######################
        elif key == "euler":
            # e.g.) euler = [-1]
            # e.g.) euler = [1, 1]
            # e.g.) euler = [1, -2, 3]
            x = 0
            y = -1
            z = 0
            if len(value) == 1:
                y = int(value)
            elif len(value) == 2:
                x, y = map(int, value)
            elif len(value) == 3:
                x, y, z = map(int, value)
            else:
                error("Format for Euler angle is wrong")
            kwds["euler_ngrids"] = [x, y, z]
        elif key == "nproj":
            kwds["number_ngrids"] = int(value)
            kwds["NumberProj"] = cf.number_ngrids > 1
        ######################
        # Multi/Excite-state #
        ######################
        #elif key == "multi":
        #    # e.g.) multi:
        #    #       000011 0.25
        #    #       000110 0.25
        #    #       001001 0.25
        #    #       001100 0.25
        #    #  should be passed as a nested-list 
        #    #  [["000011", "000110", "001001", "001100"], [0.25, 0.25, 0.25, 0.25]]
        #    states = []
        #    weights = value[1]
        #    if len(value) != 2:
        #        print('Need two lists: a list of strings for configurations (like "000011") and a list of weights.')
        #        print(f'Your input = {value}')
        #        print('Skip multi section...')
        #        continue
        #    if len(value[0]) != len(value[1]):
        #        print('Numbers of configurations and weights have to be equal.')
        #        print('Skip multi section...')
        #        continue
        #    for state in value[0]:
        #        states.append(int(f"0b{state}", 2))
        #    # Set states/weights and skip multi's information lines.
        #    kwds["init_states"] = states
        #    kwds["weights"] = weights
        elif key == "multi":
            # e.g.) multi_:
            #       000011 0.25
            #       0.5 * 000110 - 0.8* 001001 0.25
            #       1 * 101100 +0.2*000000-0.14*1000000 0.35
            #  should be put as 
            #  states = [
            #            [["000011", 1]], 
            #            [["000110",0.5], ["001001",-0.8]]
            #            [["101100", 1], ["000000",+0.2], ["100000",-0.14]]
            #           ]
            #  weights = [0.25, 0.25, 0.35]
            states = []
            weights = []
            for j in range(len(value)):
                next_line = value[j]
                if isinstance(next_line, str):
                    state, weight = read_multi(next_line)
                elif isinstance(next_line, tuple) or isinstance(next_line, list):
                    if len(next_line) == 2:
                        next_line = str(next_line[0])  + ' ' +  str(next_line[1]) 
                        state, weight = read_multi(next_line)
                    else:
                        prints(f"incorrect specification for multi: {value}")
                else:
                    prints(f"incorrect specification for multi: {value}")
                if state is not None:
                    states.append(state)
                    weights.append(weight)
            kwds["init_states_info"] = states
            kwds["weights"] = weights
            
        elif key == "excited":
            # e.g.) excited:
            #       000110
            #       001001
            #       001100
            #  should be passed as a list of bits,
            #  ["000110", "001001", "001100"]
            # or if only one excited state, string of bit can be allowed.
            # excited="000110" is same as 
            #       excited:
            #       000110
            excited_states = []
            if type(value) is str:
                excited_states.append(int(f"0b{state.replace('|','').replace('>','')}", 2))
            elif type(value) is list:
                for state in value:
                    excited_states.append(int(f"0b{state.replace('|','').replace('>','')}", 2))
            else:
                print(f'Unrecognized type for excited {value} : {type(value)}')
                continue
            # Set excited_states and skip excited_states' information lines.
            kwds["excited_states"] = excited_states
        elif key == "local":
            value = [int(x) for x in value]
            kwds["local"] = value
        elif key == "alter":
            ### see if the list is nested (i.e., multiple pairs)
            if type(value[0]) is list:
                value_ = []
                for x_ in value:
                    try:                    
                        if len(x_) == 2: 
                            for x in x_: 
                                value_.append(int(x))
                    except:
                        raise Exception(f'Switching orbitals in alter have to be paired: {value}\n'
                                        f'Use either one of the following formats:\n'
                                        f'  alter = [[2,5], [6,8], ...]\n'
                                        f'  alter = [2, 5, 6, 8, ...]')
                value = value_
            else: 
                value = [int(x) for x in value]
                if len(value)%2 == 1: # Not pair! 
                    raise Exception(f'Switching orbitals in alter have to be paired: {value}\n'
                                    f'Use either one of the following formats:\n'
                                    f'  alter = [[2,5], [6,8], ...]\n'
                                    f'  alter = [2, 5, 6, 8, ...]')
            kwds["alter_pairs"] = value
        elif key == "include":
            # Including excitation variety.
            # e.g.) include:
            #       c->a/s
            #       cc->aa/as/ss
            #       aa->as/ss
            # Note; c = core space, a = active space, s = secondary space
            # c->a/s means from core to active/secondary excitations.
            # cc->aa/as/ss means rom core
            # to active-active/active-secondary/secondary-secondary excitations.
            includes = {}
            if isinstance(value, str):
                from_, to = value.split("->")
                includes[from_] = to
            elif isinstance(value, list):
                for next_line in value:
                    if not isinstance(next_line, str):
                        break
                    from_, to = next_line.split("->")
                    includes[from_] = to
            kwds["include"] = includes
        elif key == "read":
            # Read quket data
            kwds["read"] = value
        #else:
        #    kwds[key] = value
        elif key == "hamiltonian": 
            if type(value) != str: 
                value = str(value)
            kwds["user_defined_hamiltonian"] = ''.join(value) 
        elif key == "pauli_list": 
            if type(value) is not list:
                error(f"pauli_list must be a list\nyour entry: {value}")
            kwds["user_defined_pauli_list"] = value
            ### Since pauli_list is provided, this is user-defined ansatz.
            kwds["ansatz"] = "user-defined"
        else:
            if value != "":
                error(f"No option '{key}'")
    return kwds

