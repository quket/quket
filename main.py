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
### To avoid conflict
os.environ["OMP_NUM_THREADS"] = "1"  ### Initial setting
os.environ["MKL_NUM_THREADS"] = "1"  ### Initial setting
os.environ["NUMEXPR_NUM_THREADS"] = "1"  ### Initial setting

import inspect
import datetime

from quket import _version
from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.vqe import VQE_driver
from quket.qite import QITE_driver
from quket.fileio import error, prints, printmat, print_geom, print_grad, tstamp
from quket.fileio import read_input, set_config
from quket.quket_data import QuketData
from quket.utils import get_func_kwds


prints(f"///////////////////////////////////////////////////////////////////////////////////", opentype="w")
prints(f"///                                                                             ///")
prints(f"///                                                                             ///")
prints(f"///              QQQ       UUU  UUU    KKK   KK    EEEEEEE    TTTTTTT           ///")
prints(f"///             Q   Q       u    U      K   K       E    E    T  T  T           ///")
prints(f"///            Q     Q      U    U      K  K        E  E         T              ///")
prints(f"///            Q     Q      U    U      KKK         EEEE         T              ///")
prints(f"///            Q QQQ Q      U    U      K  K        E  E         T              ///")
prints(f"///             Q   Q       U    U      K   K       E    E       T              ///")
prints(f"///              QQQ QQ      UUUU      KKK   KK    EEEEEEE      TTT             ///")
prints(f"///                                                                             ///")
prints(f"///                      Quantum Computing Simulator Ver {_version.__version__:10}             ///")
prints(f"///                                                                             ///")
prints(f"///        Copyright 2019-2022                                                  ///")
prints(f"///        The Quket Developers                                                 ///")
prints(f"///        All rights Reserved.                                                 ///")
prints(f"///                                                                             ///")
prints(f"///////////////////////////////////////////////////////////////////////////////////")
tstamp() 

prints(f"{mpi.nprocs} processes x {cf.nthreads} = "
       f"Total {mpi.nprocs*int(cf.nthreads)} cores")
######################################
###    Start reading input file    ###
######################################
kwds_list = read_input()

for job_no, kwds in enumerate(kwds_list, 1):
    prints("+-------------+")
    prints("|  Job # %3d  |" % job_no)
    prints("+-------------+")
    try:
        prints(kwds["comment"])
    except:
        pass

    # Get kwds for initialize QuketData
    init_dict = get_func_kwds(QuketData.__init__, kwds)
    Quket = QuketData(**init_dict)

    ##############
    # Set config #
    ##############
    set_config(kwds, Quket)

    #######################
    # Construct QuketData #
    #######################
    if kwds["read"]:
        Quket.load(cf.qkt)
    else:
        Quket.initialize(**kwds)
        if cf.debug:
            tstamp('QuketData initialized')
        if Quket.method != 'mbe':
            # Transform Jordan-Wigner Operators to Qulacs Format
            Quket.openfermion_to_qulacs()
            ### Tweaking orbitals...
            if Quket.alter_pairs != []:
                ## Switch orbitals
                Quket.alter(Quket.alter_pairs)
            if Quket.local != []:
                ## Localize orbitals
                Quket.boys(*Quket.local)
            # 
            Quket.get_pauli_list()
        # Set projection parameters
        Quket.set_projection()

        # Saving input 
        Quket._init_dict = init_dict
        Quket._kwds = kwds

    if Quket.model == "chemical":
        prints(f"NBasis = {Quket.mo_coeff.shape[0]}")
        Quket.print_mo_energy()
        if Quket.cf.print_level > 1 or cf.debug:
            if cf.debug:
                format = '18.14f'
            else:
                format = '11.7f'
            printmat(Quket.mo_coeff, eig=Quket.orbital_energies, name="MO coeff", format=format)
        if cf.debug:
            printmat(Quket.overlap_integrals, name="Overlap", format=format)

    if Quket.run_qubitfci:
        Quket.fci2qubit()

    if Quket.cf.do_taper_off or Quket.symmetry_pauli:
        Quket.tapering.run(mapping=Quket.cf.mapping)
        if Quket.cf.do_taper_off and Quket.method != 'mbe':
            ### Create excitation-pauli list, and transform relevant stuff by unitary
            Quket.transform_all(reduce=True)
        elif Quket.get_allowed_pauli_list:
            Quket.get_allowed_pauli_list()

        if Quket.fci_states is not None:
            prints("FCI in tapered-off qubits")
            for fci_state in Quket.fci_states:
                tmp = Quket.get_E(fci_state)
                Quket.print_state(fci_state, name=f"(FCI state : E = {tmp})")

    if Quket.ansatz is None or Quket.maxiter == 0:
        prints(f"\n   Skipping job since maxiter = {Quket.maxiter}.\n")
        Quket.energy = Quket.qulacs.Hamiltonian.get_expectation_value(Quket.state)
    else:
         
        ############
        # VQE part #
        ############
        if Quket.method in ("vqe", "mbe"):
            VQE_driver(Quket,
                       Quket.cf.kappa_guess,
                       Quket.cf.theta_guess,
                       Quket.cf.mix_level,
                       Quket.cf.opt_method,
                       Quket.cf.opt_options,
                       Quket.cf.print_level,
                       Quket.cf.maxiter,
                       Quket.cf.Kappa_to_T1)

            if Quket.cf.oo:
                if Quket.cf.do_taper_off:
                    Quket.transform_all(backtransform=True) 
                from quket.orbital.oo import oo
                oo(Quket, Quket.oo_maxiter, Quket.oo_gtol, Quket.oo_ftol)

            # post VQE for excited states
            for istate in range(Quket.nexcited):
                prints(f"Performing VQE for excited states: "
                       f"{istate+1}/{Quket.nexcited} states")
                Quket.vqd(det=Quket.excited_states[istate])
                VQE_driver(Quket,
                           "zero",
                           Quket.cf.theta_guess,
                           Quket.cf.mix_level,
                           Quket.cf.opt_method,
                           Quket.cf.opt_options,
                           Quket.cf.print_level,
                           Quket.cf.maxiter,
                           False)
        #############
        # QITE part #
        #############
        elif Quket.method == "qite":
            QITE_driver(Quket)

        if Quket.method != 'mbe':
            if Quket.cf.do_taper_off:
                # Back to the original space
                Quket.transform_all(backtransform=True)
            Quket.prop()
            
    
    #################
    # Post-VQE part #
    #################
    # Nuclear gradient and/or Geometry optimization 
    if Quket.do_grad or Quket.geom_opt:
        from quket.post import grad 
        Quket.nuclear_grad = grad.nuclear_grad(Quket)
        if Quket.geom_opt:
            grad.geomopt(Quket,init_dict,kwds)

    # Everything is fine.
    if mpi.main_rank and os.path.exists(cf.tmp):
        os.remove(cf.tmp)
    # This job is done. Go to the next job.
    Quket.save(cf.qkt)

prints(f"Normal termination of quket at {datetime.datetime.now()}")
