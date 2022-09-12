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

deriv.py

MPI-wrapper for cost and derivative functions.
"""
import time
import copy
import numpy as np
from qulacs.state import inner_product

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import SaveTheta, print_state, prints, printmat, error
from openfermion.ops import InteractionOperator
from quket.lib import (
    QuantumState, 
    hermitian_conjugated, 
    )

def cost_mpi(cost, theta):
    """Function
    Simply run the given cost function with varaibles theta,
    but ensure that all MPI processes contain the same cost.
    This should help eliminate the possible deadlock caused by numerical round errors.

    Author(s): Takashi Tsuchimochi
    """
    cost_bcast = cost(theta) #if mpi.main_rank else 0
    cost_bcast = mpi.bcast(cost_bcast, root=0)
    return cost_bcast


def jac_mpi_num(cost, theta, stepsize=1e-8):
    """Function
    Given a cost function of varaibles theta,
    return the first derivatives (jacobian)
    computed with MPI (Numerical difference).

    Author(s): Takashi Tsuchimochi
    """
    ### Just in case, broadcast theta...
    t0 = time.time()
    theta = mpi.bcast(theta, root=0)

    ndim = theta.size
    theta_d = copy.copy(theta)

    E0 = cost(theta)
    grad = np.zeros(ndim)
    grad_r = np.zeros(ndim)
    ipos, my_ndim = mpi.myrange(ndim)
    for iloop in range(ipos, ipos+my_ndim):
        theta_d[iloop] += stepsize
        Ep = cost(theta_d)
        theta_d[iloop] -= stepsize
        grad[iloop] = (Ep-E0)/stepsize
    grad_r = mpi.allreduce(grad, mpi.MPI.SUM)
    cf.grad = np.linalg.norm(grad_r)
    cf.gradv = np.copy(grad_r)
    cf.grad_max = max(grad_r)
    t1 = time.time()
    if cf.debug:
        prints(f' cost = {E0:22.16f}    ||g|| = {cf.grad:4.2e}    g_max = {cf.grad_max:4.2e}')
        prints(f'Time for gradient:  {t1-t0:0.3f}')
        printmat(grad_r)
    return grad_r

def jac_mpi_ana(create_state, Quket, theta, stepsize=1e-8, current_state=None, init_state=None, Hamiltonian=None):
    """Function
    Given a cost function of varaibles theta,
    return the first derivatives (jacobian)
    computed with MPI.

    This is a faster version of jac_mpi, where we perform analytical derivatives
     grad_r[i] = d/dtheta[i]   <psi(theta[:])| H |psi(theta[:])>
               =  <psi(theta[:])| H  d/dtheta[i] |psi(theta[:])>
               = 2 * Re <H| U[n-1] U[n-2] ... U[i+1] sigma[i] U[i] ... U[1] U[0] |0>
    
    Here,
        <H|  =  <psi(theta[:])| H
        U[i] = exp(theta[i] sigma[i]). 
    
    We do this in a step-by-step, sweep-like manner, because creating 
    the derivative state d/dtheta[i] |psi(theta[:])>  and is time-consuming.

    Set   |psi'>   <--  |psi>  =  U[n-1] U[n-2] ... U[1] U[0]  |0> 
    Set   |H'>  <--  |H> = H|psi> 
    (0-a)   |s_psi'> = sigma[n-1] |psi'> =  sigma[n-1] U[n-1] U[n-2] ... U[1] U[0]  |0> 
    (0-b)  Evaluate <H'|s_psi'> = <H| sigma[n-1] U[n-1] U[n-2] ... U[i] ... U[1] U[0] |0>
    (0-c)   |psi'>   <--  U[n-1]! |psi'> 
    (0-d)   |H'>  <--  U[n-1]! |H'>  
    (1-a)   |s_psi'> = sigma[n-2] |psi'>  
    (1-b)  Evaluate <H'|s_psi'> = <H| U[n-1] sigma[n-2] U[n-2] ... U[i] ... U[1] U[0] |0>
    (1-c)   |psi'>   <--  U[n-2]! |psi'> 
    (1-d)   |H'>  <--  U[n-2]! |H'>  
    ...
    (i-a)   |s_psi'> = sigma[n-i-1] |0'>
    (i-b)  Evaluate <UH'|s0'> = <H| U[n-1] U[n-2] ... U[n-i] sigma[n-i-1] U[n-i-1] ... U[2] U[1] U[0] |0>
    (i-c)   |psi'>   <--  U[n-i-1]! |psi'>  
    (i-d)   |H'>  <--  U[n-i-1]! |H'>  
    ...

    Args:
        create_state (func): function to prepare a VQE state by theta
        Quket (QuketData): QuketData instance
        theta (1darray): theta list
        stepsize (float): Step-size for numerical derivative of wave function
        init_state (QuantumState): Initial state to be used in create_state()
        Hamiltonian (QubitOperator): Hamiltonian H for which we take the derivative of expectation, <VQE|H|VQE>

    Author(s): Takashi Tsuchimochi
    """
    from quket.opelib import evolve  
    from quket.ansatze import create_exp_state
    from quket.utils import orthogonal_constraint

    t_initial = time.time()
    ### Just in case, broadcast theta...
    t0 = time.time()
    theta = mpi.bcast(theta, root=0)

    ndim = theta.size
    theta_d = copy.copy(theta)

    if Hamiltonian is None:
        Hamiltonian = Quket.operators.qubit_Hamiltonian
    
    grad = np.zeros(ndim)
    grad_r = np.zeros(ndim)
    grad_test = np.zeros(ndim)
    if init_state is None:
        init_state = Quket.init_state
    if current_state is None:
        state = create_state(theta, init_state=init_state)
    else:
        state = current_state.copy()
    if Quket.projection.SpinProj:
        from quket.projection import S2Proj
        Pstate = S2Proj( Quket , state, normalize=False)
        norm = inner_product(Pstate, state)  ### <phi|P|phi>
        Hstate = evolve(Hamiltonian,  Pstate, parallel=True)  ## HP|phi>
        E0 = (inner_product(state, Hstate)/norm).real   ## <phi|HP|phi>/<phi|P|phi>
        Pstate.multiply_coef(-E0) ## -E0 P|phi>
        Hstate.add_state(Pstate)  ## (H-E0) P|phi>
        Hstate.multiply_coef(1/norm)  ## (H-E0) P|phi> / <phi|P|phi>
        #Hstate.multiply_coef(1/norm)  ## H P|phi> / <phi|P|phi>
        #Pstate.multiply_coef(1/norm) ##  P|phi> / <phi|P|phi>
        ### Required to set E0 to zero 
        #E0 = 0
        del(Pstate)
    else:
        Hstate = evolve(Hamiltonian, state, parallel=True)
        E0 = inner_product(state, Hstate).real
        Pstate = None

    ### S4 penalty
    if Quket.constraint_lambda > 0:
        s = (Quket.spin - 1)/2
        S4state = evolve(Quket.operators.qubit_S4, state, parallel=True)
        S4state.multiply_coef(Quket.constraint_lambda)
        S2state = evolve(Quket.operators.qubit_S2, state, parallel=True)
        S2state.multiply_coef(- Quket.constraint_lambda * s * (s+1) )
        state_ = state.copy() 
        state_.multiply_coef(Quket.constraint_lambda * ( s * (s+1) )**2 )
        Hstate.add_state(S4state)
        Hstate.add_state(S2state)
        Hstate.add_state(state_)
        E0 = inner_product(state, Hstate).real
        del(S4state)
     
    if Quket.adapt.mode == 'pauli':
    ### Sz penalty
        if Quket.constraint_lambda_Sz > 0:
            #  Sz ** 2
            Sz2 = Quket.operators.qubit_Sz * Quket.operators.qubit_Sz
            Sz2state = evolve(Sz2, state, parallel=True) ## Sz|Phi>
            ## Ms2 = <Phi|Sz**2|Phi>
            Ms2 = inner_product(state, Sz2state).real
            ### lambda (Sz**2 - Ms2)|Phi>
            state_ = state.copy()
            state_.multiply_coef(-Ms2)
            Sz2state.add_state(state_)
            Sz2state.multiply_coef(Quket.constraint_lambda_Sz)
            Hstate.add_state(Sz2state)
            del(Sz2state)
        if Quket.constraint_lambda_S2_expectation > 0 :
            # penalty (<S**2> - s(s+1))
            S2state = evolve(Quket.operators.qubit_S2, state, parallel=True) ## S2|Phi>
            ## S2 = <Phi|S**2|Phi>
            S2 = inner_product(state, S2state).real

            state_ = state.copy()
            state_.multiply_coef(-S2)
            S2state.add_state(state_)

            ## lambda (S2-<S**2>)|Phi>
            S2state.multiply_coef(Quket.constraint_lambda_S2_expectation)
            Hstate.add_state(S2state)
            del(S2state)

    ### orthogonal constraints
    if not hasattr(Quket, 'lower_states'):
        Quket.lower_states = []
    nstates = len(Quket.lower_states)
    istate = None
    for i in range(nstates):
        Ei = Quket.lower_states[i]['energy']
        overlap = inner_product(Quket.lower_states[i]['state'], state).real
        istate = Quket.lower_states[i]['state'].copy()
        istate.multiply_coef(-Ei*overlap)
        Hstate.add_state(istate)
        E0 += -Ei * abs(overlap)**2
    if istate is not None:
        del(istate)
    t1 = time.time()
    t_hstate = t1 - t0
    t_init = 0
    if Quket.rho == 1 and "fci2qubit" not in str(create_state):
        ### Exponential ansatz with one Trotter-slice.
        ### Exact state-vector treatment with very efficient sweep algorithm
        t_cu = 0
        t_cuH = 0
        t_sigma = 0
        t_inner = 0
        t_init = 0
        ipos, my_ndim = mpi.myrange(ndim)

        ### Set the size of pauli_list to that of theta_list
        pauli_list = Quket.pauli_list[:ndim]
        cf.debug_time = False
        # Construct the initial bra state 
        # <UH'| = <H| U[n-1] U[n-2] ... U[n-i] U[n-i-1] ... U[2] U[1] U[0] 
        # and/or the ket state 
        #  U[n-1] U[n-2] ... U[n-i] U[n-i-1] ... U[2] U[1] U[0] |0>
        if Quket.cf.create_w_1process and cf._user_api is not None:
            ### If threadpoolctl is available and this is an MPI-OpenMP hybrid calculation,
            ### only one MPI process per node will perform these state creations by using most of rest threads (other MPI processes are dormant) 
            from threadpoolctl import threadpool_limits

            # Preparation
            ipos_list = mpi.gather(ipos)
            ipos_list = mpi.bcast(ipos_list)
           
            skip_states = [x == ndim for x in ipos_list]
            if mpi.top_rank: 
                my_ipos_range = ipos_list[mpi.rank:mpi.rank+cf.cpu[mpi.name]] 
                my_ipos_range = {k+mpi.rank : my_ipos_range[k] for k in range(len(my_ipos_range))}
                # Delete redundant ranks
                unique = []
                my_ipos_range_ = dict()
                for key, val in my_ipos_range.items():
                    if val not in unique:
                        unique.append(val)
                        my_ipos_range_[key] = val
                my_ipos_range = my_ipos_range_
            # HState preparation
            mpi.barrier()
            t0 = time.time()
            if mpi.top_rank:
                if mpi.main_rank:
                    state_ = state.copy()
                    Hstate_ = Hstate.copy()
                    recv_rank = 1
                else:
                    if Quket._intermediate_state is not None:
                        state_ = Quket._intermediate_state.copy()
                    else:
                        ### Dummy
                        state_ = state.copy()
                    Hstate_ = Hstate.copy()
                    recv_rank = mpi.rank + 1
                n_cores_allowed = (cf.nthreads-1)*cf.nprocs_my_node + 1
                with threadpool_limits(limits=n_cores_allowed, user_api=f'{cf._user_api}'):
                    time_send2 = time.time()
                    for irev in range(ndim-1, -1, -1):
                        if recv_rank not in my_ipos_range.keys():
                            # This corresponds to the end of mpi processes of my node
                            break
                        if type(pauli_list[irev]) is list: 
                            for k in reversed(range(len(pauli_list[irev]))):
                                phase = 1
                                if list(pauli_list[irev][k].terms.values())[0].real:
                                    # It is assumed the pauli is real & theta is imaginary
                                    phase = -1  ### pauli is real and theta is imaginary (treated as real), so flip the sign of pauli for h.c. below
                                if Quket._intermediate_state is None:
                                    # state <---  U[n-ipos-1]! ... U[n-1]! |state>  =  U[n-ipos-2] ... U[0] |state>
                                    state = create_exp_state(Quket, init_state=state,\
                                                             pauli_list=[phase * hermitian_conjugated(pauli_list[irev][k])],\
                                                             theta_list=[theta[irev]],
                                                             overwrite=True)
                                # Hstate <---  U[n-ipos-1]! ... U[n-1]! H|state>
                                Hstate = create_exp_state(Quket, init_state=Hstate,\
                                                     pauli_list=[phase * hermitian_conjugated(pauli_list[irev][k])],\
                                                     theta_list=[theta[irev]],
                                                     overwrite=True)
                        else:
                            phase = 1
                            if list(pauli_list[irev].terms.values())[0].real:
                                # It is assumed the pauli is real & theta is imaginary
                                phase = -1  ### pauli is real and theta is imaginary (treated as real), so flip the sign of pauli for h.c. below
                            if Quket._intermediate_state is None:
                                # state <---  U[n-ipos-1]! ... U[n-1]! |state>  =  U[n-ipos-2] ... U[0] |state>
                                state = create_exp_state(Quket, init_state=state,\
                                                         pauli_list=[phase * hermitian_conjugated(pauli_list[irev])],\
                                                         theta_list=[theta[irev]],
                                                         overwrite=True)
                            # Hstate <---  U[n-ipos-1]! ... U[n-1]! H|state>
                            Hstate = create_exp_state(Quket, init_state=Hstate,\
                                             pauli_list=[phase * hermitian_conjugated(pauli_list[irev])],\
                                             theta_list=[theta[irev]],
                                             overwrite=True)
                        if irev == ndim - my_ipos_range[recv_rank] and not skip_states[recv_rank]:
                            time_send = time.time()
                            if cf.debug_time:
                                prints(f'{irev:5d}  (recv_rank = {recv_rank}) Time for bra/ket states preparation:  {time_send - time_send2:.5f}', root=mpi.rank, end='')
                            mpi.send( Hstate.get_vector(), dest=recv_rank, tag=recv_rank)
                            if Quket._intermediate_state is None:
                                mpi.comm.send( state.get_vector(), dest=recv_rank, tag=recv_rank)
                            time_send2 = time.time()
                            if cf.debug_time:
                                prints(f'      Time for sending:  {time_send2 - time_send:.5f}', root=mpi.rank)
                            recv_rank += 1
                        elif irev == ndim - my_ipos_range[mpi.rank] and not skip_states[mpi.rank]:
                            ### This is for myself (top_rank but not main_rank)
                            if Quket._intermediate_state is None:
                                state_ = state.copy()
                            Hstate_ = Hstate.copy()
            else:
                # TODO: Sending QuantumState would be a simpler code but don't know how to do it 
                if not skip_states[mpi.rank]:
                    Hvec = mpi.comm.recv(source=mpi.my_top_rank, tag=mpi.rank)
                    time1 = time.time() 
                    Hstate.load(Hvec)
                    if Quket._intermediate_state is None:
                        vec = mpi.comm.recv(source=mpi.my_top_rank, tag=mpi.rank)
                        state.load(vec)
                    else:
                        state = Quket._intermediate_state.copy()
                    time2 = time.time()
                    if cf.debug_time:
                        prints(f'(recv_rank = {mpi.rank}) Time for loading  {time2 - time1:.5f}', root=mpi.rank)
                else:
                    ### Skipping as this process will not be used (too many processes and too less parameters...)
                    pass

            if mpi.top_rank:
                # Retrieve
                state = state_.copy()
                Hstate = Hstate_.copy()
            ### Test
            #state_test = state.copy()
            #Hstate_test = Hstate.copy()
            ###
            t1 = time.time()
            t_init = t1-t0

        else:
            ### Initial setting for each MPI rank
            ### Each MPI rank starts with the U[ndim-ipos-1]! ... U[n-1]! |state>
            ### Depending on the value 'ndim-ipos-1', the initial state should change to evolve each state faster. 
            ### if ndim - ipos - 1 < ndim//2
            ###   Better to start from init_state
            ### else
            ###   Better to start from current state
            t0 = time.time()
            backward = ndim-ipos-1 >= ndim//2
            if not backward:
                state = init_state.copy()
                for ifor in range(ndim-ipos):
                    # state <---  U[n-ipos] ... U[0] |init_state>
                    state = create_exp_state(Quket, init_state=state,\
                                             pauli_list=[pauli_list[ifor]],\
                                             theta_list=[theta[ifor]],
                                             overwrite=True)

            for irev in range(ndim-1, ndim-ipos-1, -1):
                if type(pauli_list[irev]) is list: 
                    for k in reversed(range(len(pauli_list[irev]))):
                        phase = 1
                        if list(pauli_list[irev][k].terms.values())[0].real:
                            # It is assumed the pauli is real & theta is imaginary
                            phase = -1  ### pauli is real and theta is imaginary (treated as real), so flip the sign of pauli for h.c. below

                        if backward: 
                            # state <---  U[n-ipos-1]! ... U[n-1]! |state>  =  U[n-ipos-2] ... U[0] |state>
                            state = create_exp_state(Quket, init_state=state,\
                                                     pauli_list=[phase * hermitian_conjugated(pauli_list[irev][k])],\
                                                     theta_list=[theta[irev]],
                                                     overwrite=True)
                        # Hstate <---  U[n-ipos-1]! ... U[n-1]! H|state>
                        Hstate = create_exp_state(Quket, init_state=Hstate,\
                                             pauli_list=[phase * hermitian_conjugated(pauli_list[irev][k])],\
                                             theta_list=[theta[irev]],
                                             overwrite=True)
                else:
                    phase = 1
                    if list(pauli_list[irev].terms.values())[0].real:
                        # It is assumed the pauli is real & theta is imaginary
                        phase = -1  ### pauli is real and theta is imaginary (treated as real), so flip the sign of pauli for h.c. below
                    if backward: 
                        # state <---  U[n-ipos-1]! ... U[n-1]! |state>
                        state = create_exp_state(Quket, init_state=state,\
                                                 pauli_list=[phase * hermitian_conjugated(pauli_list[irev])],\
                                                 theta_list=[theta[irev]],
                                                 overwrite=True)
                    # Hstate <---  U[n-ipos-1]! ... U[n-1]! H|state>
                    Hstate = create_exp_state(Quket, init_state=Hstate,\
                                     pauli_list=[phase * hermitian_conjugated(pauli_list[irev])],\
                                     theta_list=[theta[irev]],
                                     overwrite=True)
                #Hstate_old = Hstate.copy()
            t1 = time.time()
            t_init = t1-t0
            if cf.debug_time:
                prints(f'Using {cf.nthreads} threads ...   {t1-t0:.5f}')
            ### Test
            #state_test = state.copy()
            #Hstate_test = Hstate.copy()
            ###
        t0 = time.time()
        #### TEST 
        #for iproc in range(mpi.nprocs):
        #    if iproc == mpi.rank:
        #        print_state(state_test, f'state mpirank = {mpi.rank}', root=mpi.rank)
        #    mpi.barrier()
        #for iproc in range(mpi.nprocs):
        #    if iproc == mpi.rank:
        #        print_state(Hstate_test, f'Hstate mpirank = {mpi.rank}', root=mpi.rank)
        #    mpi.barrier()
        for iloop in range(ndim - ipos - 1, ndim - (ipos + my_ndim) -1, -1):
            ###  Now we take one step forward and compute the derivative of the intermediate state 
            ###
            if type(pauli_list[iloop]) is list: 
                ###
                ### Non-commutatitive because of 'spin-free' operator
                ### 
                ### We have to decompose pauli
                    
                for k in reversed(range(len(pauli_list[iloop]))):
                    #  (a)   sigma[iloop][k] |psi'>  
                    t0 = time.time()
                    sigma_state = evolve(pauli_list[iloop][k], state, parallel=False)
                    t1 = time.time()
                    t_sigma += t1-t0

                    #  (b)  Evaluate <H'|s|psi'> = <H| U[n-1] U[n-2] ... sigma[iloop] U[iloop] ... U[1] U[0] |0>
                    #grad[iloop] += 2 * inner_product(sigma_state, Hstate).real
                    if list(pauli_list[iloop][k].terms.values())[0].real:
                        if list(pauli_list[iloop][k].terms.values())[0].imag:
                            error(f"Incorrect pauli for pauli_list[{iloop}]: {pauli_list[iloop]}")
                        # It is assumed the pauli is real & theta is imaginary
                        grad[iloop] += 2 * inner_product(sigma_state, Hstate).imag
                        phase = -1  ### pauli is real and theta is imaginary (treated as real), so flip the sign of pauli for h.c. below
                    else:
                        # It is assumed the pauli is imaginary & theta is real
                        grad[iloop] += 2 * inner_product(sigma_state, Hstate).real
                        phase = 1
                    t2 = time.time()
                    t_inner += t2 - t1

                    # For next state, downgrade the states
                    #if iloop != ndim-(ipos+my_ndim):
                    #  (c)   |psi'>   <--  U[iloop][k]! |psi'> 
                    t0 = time.time()
                    state = create_exp_state(Quket, init_state=state,\
                                             pauli_list=[phase * hermitian_conjugated(pauli_list[iloop][k])],\
                                             theta_list=[theta[iloop]],
                                             overwrite=True)

                    t1 = time.time()
                    t_cu += t1-t0
                    t0 = time.time()
                    #  (d)   |H'>  <--  U[iloop][k]! |H'>  (= U[iloop]! ... U[n-2]! U[n-1]! |H> ) 
                    Hstate = create_exp_state(Quket, init_state=Hstate,\
                                         pauli_list=[phase * hermitian_conjugated(pauli_list[iloop][k])],\
                                         theta_list=[theta[iloop]],
                                         overwrite=True)
                    t1 = time.time()
                    t_cuH += t1-t0
                    

            else:  ### Commutative
                #  (a)   sigma[iloop] |psi'>  
                t0 = time.time()
                sigma_state = evolve(pauli_list[iloop], state, parallel=False)
                t1 = time.time()
                t_sigma += t1-t0
                #  (b)  Evaluate <UH'|s0'> = <H| U[n-1] U[n-2] ... sigma[iloop] U[iloop] ... U[1] U[0] |0>
                if list(pauli_list[iloop].terms.values())[0].real:
                    if list(pauli_list[iloop].terms.values())[0].imag:
                        error(f"Incorrect pauli for pauli_list[{iloop}]: {pauli_list[iloop]}")
                    # It is assumed the pauli is real & theta is imaginary
                    grad[iloop] = 2 * inner_product(sigma_state, Hstate).imag
                    phase = -1  ### pauli is real and theta is imaginary (treated as real), so flip the sign of pauli for h.c. below
                else:
                    # It is assumed the pauli is imaginary & theta is real
                    grad[iloop] = 2 * inner_product(sigma_state, Hstate).real
                    phase = 1
                t2 = time.time()
                t_inner += t2 - t1

                # For next state, downgrade the states
                #if iloop != ndim-(ipos+my_ndim):
                t0 = time.time()
                #  (c)   |psi'>   <--  U[iloop]! |psi'> 
                state = create_exp_state(Quket, init_state=state,\
                                         pauli_list=[phase*hermitian_conjugated(pauli_list[iloop])],\
                                         theta_list=[theta[iloop]],
                                         overwrite=True)
                t1 = time.time()
                t_cu += t1-t0
                t0 = time.time()
                #  (d)   |H'>  <--  U[iloop]! |H'>  (= U[iloop]! ... U[n-2]! U[n-1]! |H> ) 
                Hstate = create_exp_state(Quket, init_state=Hstate,\
                                     pauli_list=[phase*hermitian_conjugated(pauli_list[iloop])],\
                                     theta_list=[theta[iloop]],
                                     overwrite=True)
                t1 = time.time()
                t_cuH += t1-t0


    else:
        ### For rho > 1, currently only numerical derivatives are available
        t_create = 0
        t_inner = 0
        ipos, my_ndim = mpi.myrange(ndim)
        for iloop in range(ipos, ipos+my_ndim):
            theta_d[iloop] += stepsize
            t0 = time.time()
            state = create_state(theta_d, init_state=init_state)
            t1 = time.time()
            t_create += t1 - t0
            Ep = inner_product(state, Hstate).real
            t2 = time.time()
            t_inner += t2 - t1
            if Quket.projection.SpinProj:
                if abs(Ep/E0) > 1e-15:
                    grad[iloop] = 2*(Ep)/stepsize
            elif abs((Ep-E0)/E0) > 1e-15:
                grad[iloop] = 2*(Ep-E0)/stepsize
            else:
                # this may well be round-off error. 
                # w have to implement analytic gradient...
                pass
            theta_d[iloop] -= stepsize
    grad_r = mpi.allreduce(grad, mpi.MPI.SUM)
    cf.grad = np.linalg.norm(grad_r)
    cf.gradv = np.copy(grad_r)
    cf.grad_max = max(grad_r)
    t_final = time.time()
    if cf.debug:
        prints(f' cost = {E0:22.16f}    ||g|| = {cf.grad:4.2e}    g_max = {cf.grad_max:4.2e}   grad time = {t_final-t_initial:0.3f}')
        for irank in range(mpi.nprocs):
            if irank == mpi.rank:
                prints(f'mpi.rank={mpi.rank}    Initial |H>:  {t_hstate:0.3f}   Initial U!|H>: {t_init:0.3f}    U!|0>:  {t_cu:0.3f}    U!|H>: {t_cuH:0.3f}    sigma|0>:  {t_sigma:0.3f}   Inner_product:  {t_inner:0.3f}',root=mpi.rank)
            mpi.barrier()

    return grad_r



def jac_mpi_deriv_SA(create_state, Quket, theta, stepsize=1e-8):
    """Function
    Given a cost function of varaibles theta,
    return the first derivatives (jacobian) of the state-averaged energy.
    
    The states are DECOUPLED (i.e., no Hamiltonian coupling, and take the derivative of the weighted sum of energies.) 
    """
    nstates = Quket.multi.nstates
    init_states = Quket.multi.init_states
    weights_ = Quket.multi.weights
    weights = [x/sum(weights_) for x in weights_]
    ndim = theta.size
    grad = np.zeros((ndim, nstates), dtype=float)
    grad_r = np.zeros(ndim, dtype=float)
    theta = mpi.bcast(theta, root=0)
      
    from quket.opelib import evolve  
    for istate in range(nstates): 
        theta_d = copy.copy(theta)
        Quket.init_state = init_states[istate].copy()
        state = create_state(theta, init_state=Quket.init_state)
        if Quket.projection.SpinProj:
            from quket.projection import S2Proj
            Pstate = S2Proj( Quket , state, normalize=False)
            norm = inner_product(Pstate, state)  ### <phi|P|phi>
            Hstate = evolve(Quket.operators.qubit_Hamiltonian,  Pstate, parallel=True)  ## HP|phi>
            E0 = (inner_product(state, Hstate)/norm).real   ## <phi|HP|phi>/<phi|P|phi>
            Pstate.multiply_coef(-E0) ## -E0 P|phi>
            Hstate.add_state(Pstate)  ## (H-E0) P|phi>
            Hstate.multiply_coef(1/norm)  ## (H-E0) P|phi> / <phi|P|phi>
        else:
            Hstate = evolve(Quket.operators.qubit_Hamiltonian, state, parallel=True)
            E0 = inner_product(state, Hstate).real
            Pstate = None
            
        t1 = time.time()

        ### S4 penalty
        if Quket.constraint_lambda > 0:
            s = (Quket.spin - 1)/2
            S4state = evolve(Quket.operators.qubit_S4, state, parallel=True)
            S4state.multiply_coef(Quket.constraint_lambda)
            S2state = evolve(Quket.operators.qubit_S2, state, parallel=True)
            S2state.multiply_coef(- Quket.constraint_lambda * s * (s+1) )
            state_ = state.copy() 
            state_.multiply_coef(Quket.constraint_lambda * ( s * (s+1) )**2 )
            Hstate.add_state(S4state)
            Hstate.add_state(S2state)
            Hstate.add_state(state_)
            E0 = inner_product(state, Hstate).real

            

        ### orthogonal constraints
        nstates = len(Quket.lower_states)
        for i in range(nstates):
            Ei = Quket.lower_states[i]['energy']
            overlap = inner_product(Quket.lower_states[i]['state'], state).real
            istate = Quket.lower_states[i]['state'].copy()
            istate.multiply_coef(-Ei*overlap)
            Hstate.add_state(istate)
            E0 += -Ei * abs(overlap)**2
            
        
        ipos, my_ndim = mpi.myrange(ndim)
        for iloop in range(ipos, ipos+my_ndim):
            theta_d[iloop] += stepsize
            t0 = time.time()
            Quket.init_state = init_states[istate].copy()
            state = create_state(theta_d, init_state=Quket.init_state)
            t_create = time.time()
            Ep = inner_product(state, Hstate).real
            if Quket.projection.SpinProj:
                if abs(Ep/E0) > 1e-15:
                    grad[iloop, istate] = 2*(Ep)/stepsize *weights[istate]
                    grad_r[iloop] += grad[iloop, istate]
            elif abs((Ep-E0)/E0) > 1e-15:
                grad[iloop, istate] = 2*(Ep-E0)/stepsize * weights[istate]
                grad_r[iloop] += grad[iloop, istate]
            else:
                # this may well be round-off error. 
                # w have to implement analytic gradient...
                pass
            theta_d[iloop] -= stepsize
    grad = mpi.allreduce(grad, mpi.MPI.SUM)
    grad_r = mpi.allreduce(grad_r, mpi.MPI.SUM)
    if cf.debug:
        printmat(grad, name='gradients')
    cf.grad = np.linalg.norm(grad_r)
    cf.gradv = np.copy(grad_r)
    cf.grad_max = max(grad_r)
    #t2 = time.time()
    return grad_r
