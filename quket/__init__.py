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

import os, sys
from quket import config as cf

from ._version import __version__


from quket.fileio import(
    printmat,
    printmath,
    print_state,
    )
from quket.utils import(
    transform_state_jw2bk,
    transform_state_bk2jw,
    )
from quket.opelib import(
    evolve,
    devolve,
    create_exp_state,
    )
from quket.opelib import(
    OpenFermionOperator2QulacsObservable,
    OpenFermionOperator2QulacsGeneralOperator,
    )
from quket.orbital import(
    orbital_rotation
    )
from quket.projection import(   
    S2Proj,
    NProj,
    )
from quket.utils import(
    Gdoubles_list,
    )
from quket.vqe import(
    vqe
    )
from quket.lib import(
    QuantumState,
    QubitOperator, 
    FermionOperator, 
    jordan_wigner, 
    reverse_jordan_wigner,
    bravyi_kitaev, 
    get_fermion_operator, 
    commutator,
    s_squared_operator,
    number_operator,
    normal_ordered,
    hermitian_conjugated
    )

from qulacs.state import inner_product

def create(read=None, log=None, job=0, **kwds):
    """Function
    Create QuketData instance. 

    Args:
        read (str): Path to the Quket input file.
        log (str): Path to the log file.
        job (int): Job number to perform in the read input file (separated by '@@@').
        **kwds: Options directly passed (may overwrite those given in the read input).

    Returns:
        Quket (QuketData): QuketData instance.

    Author(s): Yuma Shimomoto, Takashi Tsuchimochi
    """
    cf.debug = False
    if read is not None:
        # Setting input name
        read = read.strip()
        cf.input_name, cf.input_dir, cf.input_file, cf.theta_list_file, cf.kappa_list_file, cf.tmp, cf.chk, cf.rdm1, cf.adapt\
        = cf.set_filenames(read) 
    else:
        cf.input_name, cf.input_dir, cf.input_file, cf.theta_list_file, cf.kappa_list_file, cf.tmp, cf.chk, cf.rdm1, cf.adapt\
        = cf.set_filenames('') 


    # Changing log-file name
    if log in (None, ''):
        cf.log = cf.log_file = None
    elif type(log) is str:
        cf.log = cf.log_file = log
    else:
        raise ValueError (f"Incorrect log option {log}")

    from quket.fileio.read import read_input, read_input_command_line, set_config
    from quket.utils.utils import get_func_kwds
    from quket.quket_data.quket_data import QuketData
    from quket.fileio.fileio import prints, printmat, tstamp

    if read is not None and read != '':
        # Loading options from input file
        read_kwds = read_input()
        if len(read_kwds) < job+1:
            raise ValueError(f'job number {job} is specified but {cf.input_file} has only {len(read_kwds)} job(s).')
        # Options directly specified in kwds will overwrite.
        kwds = read_input_command_line(kwds)
        read_kwds[job].update(kwds)
        kwds = read_kwds[job]
    else:
        kwds = read_input_command_line(kwds)
    init_dict = get_func_kwds(QuketData.__init__, kwds)
    Quket = QuketData(**init_dict)
    set_config(kwds, Quket)
    Quket.initialize(**kwds)
    #Quket.jw_to_qulacs()
    #### Tweaking orbitals...
    #if Quket.alter_pairs != []:
    #    ## Switch orbitals
    #    Quket.alter(Quket.alter_pairs)
    #if Quket.local != []:
    #    ## Localize orbitals
    #    Quket.boys(*Quket.local)
    #Quket.set_projection()
    #Quket.get_pauli_list()
    
    # Saving input 
    Quket._init_dict = init_dict
    Quket._kwds = kwds

    if Quket.model == "chemical":
        prints(f"NBasis = {Quket.mo_coeff.shape[0]}")
        if Quket.cf.print_level > 1 or cf.debug:
            if cf.debug:
                format = '18.14f'
            else:
                format = '10.7f'
            printmat(Quket.mo_coeff, name="MO coeff", format=format)
        if cf.debug:
            printmat(Quket.overlap_integrals, name="Overlap", format=format)

#    if Quket.cf.do_taper_off or Quket.symmetry_pauli:
#        Quket.tapering.run(mapping=Quket.cf.mapping)
#        if Quket.cf.do_taper_off and Quket.method != 'mbe':
#            ### Create excitation-pauli list, and transform relevant stuff by unitary
#            Quket.transform_all(reduce=True)
#        elif Quket.get_allowed_pauli_list:
#            Quket.get_allowed_pauli_list()

    if Quket.run_qubitfci:
        Quket.fci2qubit()

    return Quket


