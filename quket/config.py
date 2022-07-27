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

config.py

Global arguments are stored in this code.
Default values are set up.

"""

import os
import re
import sys
import time
from argparse import ArgumentParser

# Debugging option.
debug = False
"""
>>> import quket.config as cf
>>> cf.debug = True
"""
# or set debug = True in the input.

def set_filenames(input):
    """Function
    Set file names and paths based on input.
    These names/paths will be used for printing/storing.
    They are determined automatically in the initial step of calculation, 
    but can be modified anytime by directly changing them:

    >>> from src import config as cf
    >>> cf.log = "./modify.log" 
    """

    input_dir, base_name = os.path.split(input)
    input_dir = "." if input_dir == "" else input_dir
    input_name, ext = os.path.splitext(base_name)
    if ext == "":
        ext = ".inp"
    elif ext not in ".inp":
        input_name += ext
        ext = ".inp"
    input_file = f"{input_dir}/{input_name+ext}"
    theta_list_file = f"{input_dir}/{input_name}.theta"
    tmp = f"{input_dir}/{input_name}.tmp"
    kappa_list_file = f"{input_dir}/{input_name}.kappa"
    chk = f"{input_dir}/{input_name}.chk"
    # Define the names of other useful files
    rdm1 = f"{input_dir}/{input_name}.1rdm"
    adapt = f"{input_dir}/{input_name}.adapt"
    return input_name, input_dir, input_file, theta_list_file, kappa_list_file, tmp, chk, rdm1, adapt 

### Check if quket is called in the interactive mode or by main.py
ext_args = sys.argv
#if len(ext_args) == 1:
if 'main.py' in ext_args[0]:
    if len(ext_args) == 1:
        ###  main.py was invoked without an input
        print("No input file!")
        sys.exit()

    else:
        ### Have input. Read it.
        def get_option(nthreads,log):
            parser = ArgumentParser()
            parser.add_argument("input",  type=str,
                            help='No input file!')
    #    parser.add_argument('-i', '--input', type=str,
    #                        default="",
    #                        help="No input file!")
            parser.add_argument('-nt', '--nthreads', type=str,
                               default=nthreads,
                               help='Specify number of threads')
            parser.add_argument('-l', '--log', type=str,
                               default=log,
                               help='Log file name')
            return parser.parse_args()
        args = get_option("1",None)
    
        ################################################################
        #                   Setting for input and output               #
        ################################################################
        # First argument = Input file,  either "***.inp" or "***" is allowed.
        input_name, input_dir, input_file, theta_list_file, kappa_list_file, tmp, chk, rdm1, adapt\
        = set_filenames(args.input) 
        qkt = f"{input_dir}/{input_name}.qkt"

    ## OMP_NUM_THREADS ##############################################
    nthreads = args.nthreads
    os.environ["OMP_NUM_THREADS"] = nthreads
    #################################################################

else:
    ### interactive mode
    ### set dummy args 
    args = ArgumentParser().parse_args(args=[])
    args.input = ''
    args.log = None
    input_name = ''
    input_dir = os.getcwd()


log_file = args.log
if log_file is None:
    if input_name in ('',None):
        log = None
        __interactive__ = True
    else:
        log_file = input_name
        log = f"{input_dir}/{log_file}.log"
        __interactive__ = False
else:
    log = log_file
    __interactive__ = False

#################################################################


# Method and Ansatz list
vqe_ansatz_list = [
        "hf",
        "uhf",
        "phf",
        "suhf",
        "sghf",
        "uccd",
        "uccsd",
        "uccgd",
        "uccgsd",
        "uccgsdt",
        "uccgsdtq",
        "sauccd",
        "sauccsd",
        "sauccgd",
        "sauccgsd",
        "jmucc",
        "opt_puccd",
        "opt_puccsd",
        "puccsd",
        "opt_psauccd",
        "adapt",
        "ic_mrucc",
        "ic_mrucc_spinfree",
        "hamiltonian",
        "anti-hamiltonian",
        "hva",
        "ahva",
        "user-defined"
        ]
qite_ansatz_list = [
        "exact",
        "inexact",
        "hamiltonian",
        "anti-hamiltonian",
        "hva",
        "ahva",
        "hamiltonian2",
        "cite",
        "uccsd",
        "uccgsd",
        "upccgsd",
        "uccgsdt",
        "uccgsdtq",
        "qeb",
        "pauli",
        "pauli_yz",
        "user-defined"
        ]
post_method_list = [
        "lucc",
        "luccsd",
        "luccd",
        "lucc2",
        "luccsd2",
        "cepa0",
        "cisd",
        "ucisd",
        "pt2",
        "ct",
        ]


# PeriodicTable to check whether the input atoms are supported.
PeriodicTable = ["H",                 "X",                 "He",
                 "Li", "Be",  "B",  "C",  "N",  "O",  "F", "Ne",
                 "Na", "Mg", "Al", "Si",  "P",  "S", "Cl", "Ar"]

abelian_groups =['C1', 'C2', 'Ci', 'Cs', 'C2v', 'C2h', 'D2', 'D2h']

##################################
#  Define only useful variables  #
##################################
_mapping = "jordan_wigner"
_units = "angstrom"
_bohr_to_angstrom = 0.529177249
timing = False
fast_evaluation = True
fswap = False
ncnot = 0
approx_exp = False
SaveTheta = True
IntPQ = None
# cycle
icyc = 0
grad = 0
gradv = None
# Time
t_old = 0
theta_threshold = 1e-9 # If theta is smaller than this value, do not construct the circuit

