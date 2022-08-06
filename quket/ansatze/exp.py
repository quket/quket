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

exp.py

Functions preparing exponential pauli gates and circuits.
Cost and derivative functions are also defined here.
"""
import time
import numpy as np
from qulacs.state import inner_product

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import SaveTheta, print_state, prints, printmat, error
from quket.opelib import set_exp_circuit
from quket.utils import orthogonal_constraint
from quket.projection import S2Proj
from quket.lib import QuantumState
from qulacs.state import inner_product
from openfermion.ops import InteractionOperator
from quket.lib import (
    QuantumState, 
    hermitian_conjugated, 
    )

def create_exp_state(Quket, init_state=None, pauli_list=None, theta_list=None, rho=None, overwrite=False):
    """
    Given pauli_list and theta_list, create an exponentially parameterized quantum state
     
         prod_i Exp[ (theta[i] * pauli[i]) ] |0>

    where |0> is the initial state. 

    Args:
        Quket: QuketData class instance
        init_state (optional): initial quantum state
        pauli_list (optional): List of paulis as openfermion.QubitOperator
        theta_list (optional): List of parameters theta
        rho (optional): Trotter number
        overwrite (optional): if init_state is provided, overwrite it without copy.
    Returns:
        
    Author(s): Takashi Tsuchimochi
    """
    n_qubits = Quket.n_qubits
    if init_state is None:
        #### Form a product state from integer 'current_det'
        #state = QuantumState(n_qubits)
        #state.set_computational_basis(Quket.current_det)
        state = Quket.init_state.copy()
    elif overwrite:
        state = init_state
    else:
        ### Copy the initial state
        state = init_state.copy()

    if pauli_list is None:
        pauli_list = Quket.pauli_list

    if theta_list is None:
        theta_list = Quket.theta_list

    if rho is None:
        rho = Quket.rho


    if len(pauli_list) != len(theta_list):
        raise Exception(f'Dimensions of pauli_list ({len(pauli_list)}) and theta_list ({len(theta_list)}) not consistent.')

    ### 
    circuit = set_exp_circuit(n_qubits, pauli_list, theta_list, rho)
    for i in range(rho):
        circuit.update_quantum_state(state)

    return state


def cost_exp(Quket, print_level, theta_list, parallel=True):
    """Function:
    Energy functional of general exponential ansatz. 
    Generalized to sequential excited state calculations,
    by projecting out lower_states.

    Author(s): Takashi Tsuchimochi
    """
    t1 = time.time()

    ansatz = Quket.ansatz
    rho = Quket.rho
    DS = Quket.DS
    n_qubits = Quket.n_qubits
    ndim = Quket.ndim
    init_state = Quket.init_state
    #"""
    if Quket.cf.create_w_1process \
       and parallel \
       and cf._user_api is not None\
       and cf.nthreads > 1 \
       and Quket.rho == 1:
        ### If threadpoolctl is available and this is an MPI-OpenMP hybrid calculation,
        ### only one MPI process will perform these state creations by using most threads (other MPI processes are dormant) 
        debug_time = True
        state, _intermediate_state =  create_exp_intermediate_states(Quket, theta_list=theta_list, init_state=init_state, debug_time=debug_time)
    else:
        if parallel \
           and cf._user_api is not None\
           and cf.nthreads > 1:
            n_cores_allowed = (cf.nthreads-1)*cf.nprocs_my_node + 1
            state = init_state.copy()
            from threadpoolctl import threadpool_limits
            if mpi.main_rank:
                with threadpool_limits(limits=n_cores_allowed, user_api=f'{cf._user_api}'):
                    state = create_exp_state(Quket, init_state =init_state, theta_list=theta_list, rho=Quket.rho)
            state = mpi.bcast(state) 
        else:
            state = create_exp_state(Quket, init_state =init_state, theta_list=theta_list, rho=Quket.rho)
        _intermediate_state = None
    #cf.ncnot = count_CNOT_ucc(Quket, theta_list)
    if Quket.projection.SpinProj:
        Quket.state_unproj = state.copy()
        state = S2Proj(Quket, state)

    t2 = time.time()
    if cf.debug:
        prints(f'Time for state preparation in cost {t2-t1:0.5f}')
    # Store the current wave function
    Quket.state = state
    Quket._intermediate_state = _intermediate_state
    t2 = time.time()
    t1 = time.time()
    Energy = Quket.get_E(parallel=parallel)
    if Quket.operators.S2 is not None:
        S2 = Quket.get_S2(parallel=parallel)
    else:
        S2 = 0
    t2 = time.time()
    if cf.debug:
        prints(f'Time for expectation value in cost {t2-t1:0.5f}')
    
    #prints(f'time for inner_prodcut: {t_inner - t_get}')

    cost = Energy
    t3 = time.time()

    ### Project out the states contained in 'lower_states'
    cost += orthogonal_constraint(Quket, state)

    if Quket.constraint_lambda > 0:
        s = (Quket.spin - 1)/2
        S4 = Quket.qulacs.S4.get_expectation_value(state)
        penalty = Quket.constraint_lambda*(S4 - S2*(s*(s+1) + (s*(s+1))**2))
        cost += penalty

    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == -1:
        prints(f"Initial: E[{ansatz}] = {Energy:+.12f}  "
               f"<S**2> = {S2:+8.6f}  ", end='')
        if Quket.fci_states is not None:
            prints(f"Fidelity = {Quket.fidelity():.6f}  ", end='')
        prints(f"rho = {rho}  ")
               #f"CNOT = {cf.ncnot}")
    if print_level == 1:
        ## cf.constraint_lambda *= 1.1
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(f"{cf.icyc:7d}: E[{ansatz}] = {Energy:+.12f}  "
               f"<S**2> = {S2:+8.6f}  ", end='')
        if Quket.fci_states is not None:
            prints(f"Fidelity = {Quket.fidelity():.6f}  ", end='')
        prints(f"Grad = {cf.grad:4.2e}  "
               #f"CNOT = {cf.ncnot}  "
               f"CPU Time = {cput:10.5f}  ({cpu1:2.2f} / step)")
        if Quket.constraint_lambda != 0:
            prints(f"lambda = {Quket.constraint_lambda}  "
                   f"<S**4> = {S4:17.15f}  "
                   f"Penalty = {penalty:2.15f}")
        SaveTheta(ndim, theta_list, cf.tmp)
        Quket.theta_list = theta_list.copy()
    if print_level > 1:
        istate = len(Quket.lower_states)
        if istate > 0:
            state_str = str(istate) + '-'
        else:
            state_str = ''
        prints(f"  Final: E[{state_str}{ansatz}] = {Energy:+.12f}  "
               f"<S**2> = {S2:+8.6f}  ", end='')
        if Quket.fci_states is not None:
            prints(f"Fidelity = {Quket.fidelity():.6f}  ", end='')
        prints(f"rho = {rho} ")
               #f"CNOT = {cf.ncnot}  ")
        prints(f"\n({ansatz} state)")
        print_state(state)

    Quket.energy = Energy
    Quket.s2 = S2
    return cost, S2


def create_exp_intermediate_states(Quket, pauli_list=None, theta_list=None, init_state=None, debug_time=False):
    """
    Given pauli_list and theta_list, create an exponentially parameterized quantum state
     
         FinalState = prod_i Exp[ (theta[i] * pauli[i]) ] |0>

    where |0> is the initial state. 
    In contrast to `create_exp_state()`, this function also generates "intermediate" states
    that are useful for derivative later.

         IntermediateState = prod_i^n Exp[ (theta[i] * pauli[i]) ] |0>

    where n is MPI-process-dependent. 
    In this function, the unitary evolution is done by using available cores as threads as many as possible,
    where only one MPI process per node is used. Then, for other MPI processes in the same local node will
    receive the intermediate states.

    Args:
        Quket: QuketData class instance
        init_state (optional): initial quantum state
        pauli_list (optional): List of paulis as openfermion.QubitOperator
        theta_list (optional): List of parameters theta
    Returns:
        FinalState (QuantumState): The complete state 
        IntermediateState (QunatumState): The MPI-process-dependent intermediate state
        
    Author(s): Takashi Tsuchimochi
    """
    if init_state is None:
        IntermediateState = Quket.init_state.copy()
    else:
        IntermediateState = init_state.copy()

    if pauli_list is None:
        pauli_list = Quket.pauli_list

    if theta_list is None:
        theta_list = Quket.theta_list

    ndim = theta_list.size
    ipos, my_ndim = mpi.myrange(ndim)
    # Preparation
    ipos_list = mpi.gather(ipos)
    ipos_list = mpi.bcast(ipos_list)
    skip_state = False
    if mpi.rank > 0: 
        skip_state = ipos_list[mpi.rank-1] == ipos_list[mpi.rank] 
    skip_states = mpi.gather(skip_state)
    skip_states = mpi.bcast(skip_states)
    # IntermediateState preparation
    #    We are to evaluate  
    #        <H| U[n-1] U[n-2] ... U[i+1] sigma[i] U[i] ... U[1] U[0] |0>
    #        ----------------------------          ----------------------
    #           handled in jac_mpi_deriv             partially dealt here
    #
    #     IntermediateState =   U[n-ipos-2] ... U[0] |0>
    #
    ipos_list = [ndim - ipos for ipos in ipos_list] 
    if mpi.top_rank: 
        my_ipos_range = ipos_list[mpi.rank:mpi.rank+cf.cpu[mpi.name]] 
        my_ipos_len = len(my_ipos_range)
        my_ipos_range = {k + mpi.rank: my_ipos_range[k] for k in range(my_ipos_len)}

    FinalState = IntermediateState.copy()

    t0 = time.time()
    if mpi.top_rank:
        # |State> = U[n-1] U[n-2] ... U[2] U[1] U[0] |0>
        # 
        n_cores_allowed = (cf.nthreads-1)*cf.nprocs_my_node + 1
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=n_cores_allowed, user_api=f'{cf._user_api}'):
            recv_rank = mpi.rank + my_ipos_len - 1
            # Now start creating the intermediate state for each MPI process by utilizing available threads, and send/recv
            time_send2 = time.time()
            for k in my_ipos_range.keys():
                if recv_rank == mpi.rank + my_ipos_len - 1:
                    ### Initially, create state up
                    my_start = 0
                else:
                    my_start = my_end
                my_end = my_ipos_range[recv_rank] 
                if my_end <= 0:
                    # Skip 
                    recv_rank -= 1
                    continue
                if recv_rank - 1 not in my_ipos_range.keys():
                    # Myself
                    IntermediateState = create_exp_state(Quket, init_state=IntermediateState,\
                                                pauli_list=pauli_list[my_start:my_end],\
                                                theta_list=theta_list[my_start:my_end],
                                                overwrite=True)
                    
                    if mpi.main_rank:
                        # This corresponds to the end of mpi processes of my node
                        # So, do the rest part to finish the job by creating FinalState
                        my_start = my_end
                        FinalState = create_exp_state(Quket, init_state=IntermediateState,\
                                                 pauli_list=pauli_list[my_start:],\
                                                 theta_list=theta_list[my_start:],
                                                 overwrite=False)
                    break

                IntermediateState = create_exp_state(Quket, init_state=IntermediateState,\
                                         pauli_list=pauli_list[my_start:my_end],\
                                         theta_list=theta_list[my_start:my_end],
                                         overwrite=True)
                time_send = time.time()
                if debug_time:
                    prints(f'cost_exp: (recv_rank = {recv_rank}) Time for bra/ket states preparation:  {time_send - time_send2:.5f}', root=mpi.rank, end='')
                mpi.send(IntermediateState.get_vector(), dest=recv_rank, tag=recv_rank)
                time_send2 = time.time()
                if debug_time:
                    prints(f'      Time for sending:  {time_send2 - time_send:.5f}', root=mpi.rank)
                recv_rank -= 1

    else:
        # TODO: Sending QuantumState would be a simpler code but don't know how to do it 
        if ipos_list[mpi.rank] > 0: 
            vec = mpi.comm.recv(source=mpi.my_top_rank, tag=mpi.rank)
            time1 = time.time() 
            IntermediateState.load(vec)
            time2 = time.time()
            if debug_time:
                prints(f'cost_exp: (recv_rank = {mpi.rank}) Time for loading  {time2 - time1:.5f}', root=mpi.rank)
        elif ipos_list[mpi.rank] <= 0: 
            ### Skipping as this process will not be used (too many processes and too less parameters...)
            IntermediateState = None
            if debug_time:
                prints(f'cost_exp: (recv_rank = {mpi.rank}) Skipped', root=mpi.rank)
    FinalState = mpi.bcast(FinalState) 
    return FinalState, IntermediateState



