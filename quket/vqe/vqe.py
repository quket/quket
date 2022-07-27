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

vqe.py

Main driver of VQE.

"""
import time

import numpy as np
import itertools
from scipy.optimize import minimize
from qulacs.observable import create_observable_from_openfermion_text

from quket.mpilib import mpilib as mpi
from quket import config as cf
from quket.utils import cost_mpi, jac_mpi_num, jac_mpi_deriv, jac_mpi_deriv_SA
from quket.linalg import T1mult
from quket.linalg import SR1, LSR1
from quket.fileio import LoadTheta, SaveTheta, error, prints, printmat, tstamp, print_state
from quket.opelib import create_exp_state, set_exp_circuit
from quket.utils import get_ndims
from quket.lib import QubitOperator, QuantumState

def cost_StateAverage(Quket, print_level, theta_list):
    """Function
    State averaging.
    Author(s): Yuto Mori
    """
    from quket.projection import S2proj
    
    t1 = time.time()

    DS = Quket.DS
    n_qubits = Quket.n_qubits
    nstates = Quket.multi.nstates
    ansatz = Quket.ansatz

    if len(Quket.multi.states) != nstates:
        Quket.multi.states = []
        for istate in range(nstates):
            Quket.multi.states.append(QuantumState(n_qubits))

    Es = []
    S2s = []
    circuit = set_exp_circuit(n_qubits, Quket.pauli_list, theta_list, Quket.rho)

    Heff = np.zeros((nstates, nstates), dtype=complex)
    S2eff = np.zeros((nstates, nstates), dtype=complex)
    for istate in range(nstates):
        cf.ncnot = 0
        state = Quket.multi.init_states[istate].copy()
        for i in range(Quket.rho):
            circuit.update_quantum_state(state)
            if Quket.projection.SpinProj:
                state = S2Proj(Quket,state)
        Energy = Quket.get_E(state)
        S2 = Quket.get_S2(state)
        Es.append(Energy)
        S2s.append(S2)
        Heff[istate, istate] = Energy
        S2eff[istate, istate] = S2
        Quket.multi.states[istate] = state


    ### Checking Heff
    for istate, jstate in itertools.combinations(range(nstates), 2):
        Heff[istate,jstate] = Quket.get_Heff(Quket.multi.states[istate], Quket.multi.states[jstate]).real
        S2eff[istate,jstate] = Quket.get_transition_amplitude(Quket.qulacs.S2, Quket.multi.states[istate], Quket.multi.states[jstate]).real
        Heff[jstate, istate] = Heff[istate, jstate]
        S2eff[jstate, istate] = S2eff[istate, jstate]
    if cf.debug:
        printmat(Heff, name='Heff')
    Es, c =np.linalg.eigh(Heff)
    S2s = np.diag(c.T @ S2eff @ c).real

    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        string = f"{cf.icyc:6d}: "
        for istate in range(nstates):
            prints(f"{string} E[SA-{ansatz}{istate}] = {Es[istate]:.8f}  "
                   f"(<S**2> = {S2s[istate]:+7.5f})  ", end="")
            string = f"\n        "
        prints(f"  Grad = {cf.grad:4.2e}  "
               f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        SaveTheta(Quket.ndim, theta_list, cf.tmp)
        Quket.theta_list = theta_list.copy()
    if print_level > 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        string = " Final: "
        for istate in range(nstates):
            prints(f"{string} E[SA-{ansatz}{istate}] = {Es[istate]:.8f}  "
                   f"(<S**2> = {S2s[istate]:+7.5f})  ")
            string = "        "
        prints(f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)\n")
        prints("------------------------------------")
        prints("###############################################")
        prints("#           State-averaged states             #")
        prints("###############################################", end="")

        for istate in range(nstates):
            prints()
            prints(f"State         : {istate}")
            prints(f"E             : {Es[istate]:.8f}")
            prints(f"<S**2>        : {S2s[istate]:+.5f}")
            print_state(Quket.multi.states[istate])
        prints("###############################################")

    cost = norm = 0
    norm = np.sum(Quket.multi.weights)
    for istate in range(nstates):
        cost = cost + Quket.multi.weights[istate] * Es[istate]
    cost /= norm
    Quket.energy = cost
    return cost, S2s


def VQE_driver(Quket, kappa_guess, theta_guess, mix_level, opt_method,
               opt_options, print_level, maxiter, Kappa_to_T1):
    """Function:
    Main driver for VQE

    Author(s): Takashi Tsuchimochi
    """
    from quket.ansatze import adapt_vqe_driver
    from quket.ansatze import cost_uccgd_forSAOO
    from quket.ansatze import cost_bcs
    from quket.ansatze import cost_uhf, mix_orbitals
    from quket.ansatze import cost_proj
    from quket.ansatze import cost_exp
    from quket.ansatze import cost_upccgsd
    if cf.debug:
        tstamp('Enter VQE_driver')
    if Quket.ansatz == "adapt":
        if Quket.adapt.mode in ('original', 'pauli', 'pauli_sz', 'pauli_yz', 'spin', 'pauli_spin', 'pauli_spin_xy', 'qeb', 'qeb_reduced', 'qeb1', 'qeb2', 'qeb3'):
            adapt_vqe_driver(Quket, opt_method, opt_options, print_level, maxiter)
        else:
            raise ValueError(f'No option for adapt_mode = {Quket.adapt.mode}')
        #    adapt_vqe_driver(Quket, opt_method, opt_options, print_level, maxiter)
        return

    prints('Entered VQE driver')
    qubit_hamiltonian = Quket.operators.qubit_Hamiltonian
    qubit_s2 = Quket.operators.qubit_S2
    ansatz = Quket.ansatz
    if Quket.ansatz is None and Quket.pauli_list is None:
        error('ansatz undefined in vqe.')
    try:
        noa = Quket.noa
        nob = Quket.nob
        nva = Quket.nva
        nvb = Quket.nvb
    except:
        noa = 0
        nob = 0
        nva = 0
        nvb = 0
    try:
        nca = ncb = Quket.nc
    except:
        nca = ncb = 0
    try:
        norbs = Quket.n_active_orbitals
    except:
        norbs = 0
    spin_gen = ansatz in ["sghf"]

    t1 = time.time()
    cf.t_old = t1
    print_control = 1

    optk = 0
    Gen = 0
    # cf.constraint_lambda = 100
    istate = len(Quket.lower_states)  

    ### set up the number of orbitals and such ###
    n_qubits = Quket.n_qubits
    n_Pqubits = Quket.n_qubits + 1
    anc = Quket.anc

    ### get ndim ###
    ndim1, ndim2, ndim = get_ndims(Quket)
    _ndim1, _ndim2, _ndim = ndim1, ndim2, ndim
    try:
        if "bcs" in ansatz or "pccgsd" in ansatz:
            k_param = ndim//(ndim1+ndim2)
    except:
        k_param = 0

    # set number of dimensions QuketData

    if Quket._ndim is None:
        _ndim1, _ndim2, _ndim = ndim1, ndim2, ndim
        Quket._ndim = _ndim
        try:
            if "bcs" in ansatz or "pccgsd" in ansatz:
                k_param = ndim//(ndim1+ndim2)
        except:
            k_param = 0

        # set number of dimensions QuketData
        prints(f'Number of parameters: Singles {ndim1}    Doubles {ndim2}    Total {ndim}')
        Quket.ndim1 = ndim1
        Quket.ndim2 = ndim2
        Quket.ndim = ndim
        Quket._ndim1 = ndim1
        Quket._ndim2 = ndim2
        Quket._ndim = ndim
    else:
        _ndim = Quket._ndim

    if Quket.pauli_list is not None:
        ndim = len(Quket.pauli_list)
        Quket.ndim = ndim   # Possibly Tapered

    ### HxZ and S**2xZ and IxZ  ###
    ### Trick! Remove the zeroth-order term, which is the largest
    term0_H = Quket.qulacs.Hamiltonian.get_term(0)
    coef0_H = term0_H.get_coef()
    coef0_H = coef0_H.real
    if Quket.qulacs.S2 is not None:
        term0_S2 = Quket.qulacs.S2.get_term(0)
        coef0_S2 = term0_S2.get_coef()
        coef0_S2 = coef0_S2.real

    coef0_H = 0
    coef0_S2 = 0

    qubit_ancZ = QubitOperator(f"Z{anc}")
    qubit_hamiltonianZ = (qubit_hamiltonian - coef0_H*QubitOperator(""))*qubit_ancZ
    if qubit_s2 is not None:
        qubit_s2Z = (qubit_s2 - coef0_S2*QubitOperator(""))*qubit_ancZ
    qulacs_hamiltonianZ \
            = create_observable_from_openfermion_text(str(qubit_hamiltonianZ))
    if qubit_s2 is not None:
        qulacs_s2Z \
                = create_observable_from_openfermion_text(str(qubit_s2Z))
    qulacs_ancZ \
            = create_observable_from_openfermion_text(str(qubit_ancZ))
    create_state = None
    #############################
    ### set up cost functions ###
    #############################
    if ansatz in ["phf", "suhf", "sghf"]:
        ### PHF ###
        cost_wrap = lambda kappa_list: cost_proj(
                Quket,
                0,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                )[0]
        cost_callback = lambda kappa_list, print_control: cost_proj(
                Quket,
                print_control,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                )
    elif ansatz == "uhf":
        ### UHF ###
        cost_wrap = lambda kappa_list: cost_uhf(
                Quket,
                0,
                kappa_list,
                )[0]
        cost_callback = lambda kappa_list, print_control: cost_uhf(
                Quket,
                print_control,
                kappa_list,
                )
    elif "bcs" in ansatz:
        ###BCS###
        cost_wrap = lambda theta_list: cost_bcs(
                Quket,
                0,
                theta_list,
                k_param,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_bcs(
                Quket,
                print_control,
                theta_list,
                k_param,
                )
    elif "pccgsd" in ansatz:
        ###UpCCGSD###
        cost_wrap = lambda theta_list: cost_upccgsd(
                Quket,
                0,
                kappa_list,
                theta_list,
                k_param,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_upccgsd(
                Quket,
                print_control,
                kappa_list,
                theta_list,
                k_param,
                )
    elif ansatz == "puccsd":
        ### UCCSD ###
        cost_wrap = lambda theta_list: cost_proj(
                Quket,
                0,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_proj(
                Quket,
                print_control,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )
    elif ansatz == "puccd":
        ### UCCSD ###
        cost_wrap = lambda theta_list: cost_proj(
                Quket,
                0,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_proj(
                Quket,
                print_control,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )
    elif ansatz in ["opt_puccd", "opt_psauccd"]:
        ### UCCSD ###
        cost_wrap = lambda theta_list: cost_proj(
                Quket,
                0,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_proj(
                Quket,
                print_control,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )
    else:  #ansatz in ("hf", "uccsd", "sauccsd", "uccgsd", "uccd", "sauccd", "uccgd", "sauccgsd", "sauccgd", "uccgsdt", "uccgsdtq"):
        if Quket.multi.nstates == 0:
            cost_wrap = lambda theta_list: cost_exp(
                    Quket,
                    0,
                    theta_list,
                    parallel=not Quket.cf.finite_difference
                    )[0]
            cost_callback = lambda theta_list, print_control: cost_exp(
                    Quket,
                    print_control,
                    theta_list,
                    parallel=not Quket.cf.finite_difference
                    )

            create_state = lambda theta_list, init_state: create_exp_state(Quket, init_state=Quket.init_state, pauli_list=Quket.pauli_list, theta_list=theta_list, rho=Quket.rho)
        else:
            cost_wrap = lambda theta_list : cost_StateAverage(
                    Quket,
                    0,
                    theta_list
                    )[0]
            cost_callback = lambda theta_list, print_control: cost_StateAverage(
                    Quket,
                    print_control,
                    theta_list
                    )
            create_state = lambda theta_list, init_state: create_exp_state(Quket, init_state=Quket.init_state, pauli_list=Quket.pauli_list, theta_list=theta_list, rho=Quket.rho)

    fstr = f"0{n_qubits}b"
    prints(f"Performing VQE for {ansatz}")
    prints(f"Number of VQE parameters: {ndim}")
    if type(Quket.current_det) is int:
        prints(f"Initial configuration: |{format(Quket.current_det, fstr)}>")
    else:
        prints(f"Initial configuration: ",end='')
        for i, state_ in enumerate(Quket.current_det):
            prints(f" {state_[0]:+.4f} * |{format(state_[1], fstr)}>", end='')
            if i > 10:
                prints(" ... ", end='')
                break
        prints('')
    prints(f"Convergence criteria: ftol = {Quket.ftol:1.0E}, "
                                 f"gtol = {Quket.gtol:1.0E}")

    if Quket.cf.finite_difference:
        prints(f"Derivatives: Numerical")
    else:
        prints(f"Derivatives: Analytical")

    #############################
    ### set up initial kappa  ###
    #############################
    kappa_list = np.zeros(ndim1)
    if kappa_guess == "mix":
        if mix_level > 0:
            mix = mix_orbitals(noa, nob, nva, nvb, mix_level, False, np.pi/4)
            kappa_list = mix[:ndim1]
            printmat(kappa_list)
        elif mix_level == 0:
            error("kappa_guess = mix but mix_level = 0!")
    elif kappa_guess == "random":
        if spin_gen:
            mix = mix_orbitals(noa+nob, 0, nva+nvb, 0, mix_level, True, np.pi/4)
        else:
            mix = mix_orbitals(noa, nob, nva, nvb, mix_level, True, np.pi/4)
        kappa_list = mix[:ndim1]
    elif kappa_guess == "read":
        kappa_list = LoadTheta(ndim1, cf.kappa_list_file, offset=istate)
        if mix_level > 0:
            temp = kappa_list[:ndim1].copy()
            mix = mix_orbitals(noa, nob, nva, nvb, mix_level, False, np.pi/4)
            temp = T1mult(noa, nob, nva, nvb, mix, temp)
            kappa_list = temp[:ndim1].copy()
        printmat(kappa_list)
    elif kappa_guess == "zero":
        kappa_list *= 0
    #############################
    ### set up initial theta  ###
    #############################
    if theta_guess == "prev":
        if Quket.theta_list is not None:
            ## Read Quket.theta_list as guess
            theta_list = Quket.theta_list[:len(Quket.pauli_list)]
        else:
            theta_list = np.zeros(ndim, float)
    else:
        theta_list = np.zeros(_ndim)
        if theta_guess == "zero":
            theta_list *= 0
        elif theta_guess == "read":
            theta_list = LoadTheta(_ndim, cf.theta_list_file, offset=istate)
        elif theta_guess == "random":
            theta_list = (0.5-np.random.rand(_ndim))*0.001
        if Kappa_to_T1 and theta_guess != "read":
            ### Use Kappa for T1  ###
            if Quket.DS:
                theta_list[:ndim1] = kappa_list[:ndim1]
            else:
                theta_list[ndim2:] = kappa_list[:ndim1]
            kappa_list *= 0
            prints("Initial T1 amplitudes will be read from kappa.")

        if optk:
            theta_list_fix = theta_list[ndim1:]
            theta_list = theta_list[:ndim1]
            if Gen:
                # Generalized Singles.
                temp = theta_list.copy()
                theta_list = np.zeros(_ndim)
                indices = [i*(i-1)//2 + j
                            for i in range(norbs)
                                for j in range(i)
                                    if i >= noa and j < noa]
                indices.extend([i*(i-1)//2 + j + ndim1
                                    for i in range(norbs)
                                        for j in range(i)
                                            if i >= nob and j < nob])
                theta_list[indices] = temp[:len(indices)]
        else:
            theta_list_fix = 0

        ### Broadcast lists
        kappa_list = mpi.bcast(kappa_list, root=0)
        theta_list = mpi.bcast(theta_list, root=0)
    #######
    # Based on pauli_list
    #######
    if Quket.pauli_list is not None and Quket.method != 'mbe':
        # overwrite ndim to the actual number of VQE parameters
        ndim = len(Quket.pauli_list)
        Quket.ndim = ndim
        if Quket.tapered['pauli_list'] and \
           hasattr(Quket, 'allowed_pauli_list') and \
           len(theta_list) == _ndim:
            ndim = len(Quket.pauli_list)
            new_theta_list = []
            for i in range(_ndim):
                if Quket.allowed_pauli_list[i]:
                    new_theta_list.append(theta_list[i])
            theta_list = np.array(new_theta_list)
        # set number of dimensions QuketData
            Quket.ndim = ndim
            if ndim != len(theta_list):
                prints(f'Dimensions of theta_list ({len(theta_list)}) and pauli_list ({len(Quket.pauli_list)}) are not consistent.')
                prints(f'Take over theta_list and force it to have {len(Quket.pauli_list)} zero elements (guess will be gone!).')
                theta_list = np.zeros(ndim, dtype=float)
                _ndim = ndim
                #raise ValueError(f'Something is wrong?\n'
                #       f'ndim = {ndim}, len(theta_list) = {len(theta_list)},  _ndim = {_ndim}')
    elif Quket.method == 'mbe':
        ndim = len(Quket.pauli_list)
        Quket.ndim = ndim

    # If everything is good, theta_list should have the same dimension as pauli_list (if the latter exists) 
    # Sanity check
    if Quket.pauli_list is not None:
        if len(theta_list) == len(Quket.pauli_list):# or Quket.method == 'mbe':
            pass
        else:
            prints(f'Dimensions of theta_list ({len(theta_list)}) and pauli_list ({len(Quket.pauli_list)}) are not consistent.')
            prints(f'Take over theta_list and force it to have {len(Quket.pauli_list)} zero elements (guess will be gone!).')
            theta_list = np.zeros(ndim, dtype=float)

    Quket.tapered['theta_list'] =  Quket.tapered['pauli_list']

    #################################
    ### print out initial results ###
    #################################
    print_control = -1
    if ansatz in ["uhf", "phf", "suhf", "sghf"]:
        cost_callback(kappa_list, print_control)
    else:
        if Quket.DS:
            prints("Circuit order: Exp[T2] Exp[T1] |0>")
        else:
            prints("Circuit order: Exp[T1] Exp[T2] |0>")
        # prints('Initial T1 amplitudes:')
        # prints('Intial T2 amplitudes:')
        cost_callback(theta_list, print_control)
    print_control = 1

    if maxiter > 0:
        ###################
        ### perform VQE ###
        ###################
        cf.icyc = 0
        # Use MPI for evaluating Gradients
        cost_wrap_mpi = lambda theta_list: cost_mpi(cost_wrap, theta_list)
        if create_state is None or Quket.cf.finite_difference: 
            ### Numerical finite difference of energy expectation values
            jac_wrap_mpi = lambda theta_list: jac_mpi_num(cost_wrap, theta_list)
        else:
            ### Numerical/Analytical derivatives of wave function (vector-state)
            if Quket.multi.nstates==0:
                # Currently analytical
                jac_wrap_mpi = lambda theta_list: jac_mpi_deriv(create_state, Quket, theta_list, init_state=None, current_state=Quket.state)
            else:
                # Currently numerical
                jac_wrap_mpi = lambda theta_list: jac_mpi_deriv_SA(create_state, Quket, theta_list)
        if Quket.cf.opt_method in ["l-bfgs-b", "bfgs"]:
            if ansatz in ["uhf", "phf", "suhf", "sghf"]:
                opt = minimize(
                        cost_wrap_mpi,
                        kappa_list,
                        jac=jac_wrap_mpi,
                        method=opt_method,
                        options=opt_options,
                        callback=lambda x: cost_callback(x, print_control),
                        )
            else:  ### correlated ansatz
                opt = minimize(
                        cost_wrap_mpi,
                        theta_list,
                        jac=jac_wrap_mpi,
                        method=opt_method,
                        options=opt_options,
                        callback=lambda x: cost_callback(x, print_control),
                        )
            Quket.converge = opt.success
            ### print out final results ###
            final_param_list = opt.x 
        elif Quket.cf.opt_method == "sr1":
            final_param_list = SR1(cost_wrap_mpi, jac_wrap_mpi, theta_list, ndim, cost_callback)
        elif Quket.cf.opt_method == "lsr1":
            final_param_list = LSR1(cost_wrap_mpi, jac_wrap_mpi, theta_list, ndim, cost_callback)

    else:
        # Skip VQE, and perform one-shot calculation
        prints("One-shot evaluation without parameter optimization")
        if ansatz in ["uhf", "phf", "suhf", "sghf"]:
            final_param_list = kappa_list
        else:
            final_param_list = theta_list

    # Calculate final parameters.
    Evqe, S2 = cost_callback(final_param_list, print_control+1)
    if ansatz in ["uhf", "phf", "suhf", "sghf"]:
        SaveTheta(ndim, final_param_list, cf.kappa_list_file, offset=istate)
        Quket.kappa_list = final_param_list.copy()
    else:
        SaveTheta(ndim, final_param_list, cf.theta_list_file, offset=istate)
        Quket.theta_list = final_param_list.copy()



    t2 = time.time()
    cput = t2 - t1
    prints(f"\nVQE Done: CPU Time = {cput:15.4f}\n")

    return Evqe, S2

def vqe(Quket):
    """
    Alternative subroutine for VQE.
    Run VQE with pauli_list and theta_list in Quket.
    The exponential ansats form is assumed, and is applied to Quket.init_state.
    Only single-state VQE is supported currently.
    """

    from quket.ansatze import cost_exp
    cf.icyc = 0
    kappa_list = None
    theta_list = Quket.theta_list.copy()
    cost_wrap = lambda theta_list: cost_exp(
            Quket,
            0,
            theta_list,
            parallel=not Quket.cf.finite_difference
            )[0]
    cost_callback = lambda theta_list, print_control: cost_exp(
            Quket,
            print_control,
            theta_list,
            parallel=not Quket.cf.finite_difference
            )

    create_state = lambda theta_list, init_state: create_exp_state(Quket, init_state=Quket.init_state, pauli_list=Quket.pauli_list, theta_list=theta_list, rho=Quket.rho)

    cost_wrap_mpi = lambda theta_list: cost_mpi(cost_wrap, theta_list)
    if Quket.cf.finite_difference: 
        ### Numerical finite difference of energy expectation values
        jac_wrap_mpi = lambda theta_list: jac_mpi_num(cost_wrap, theta_list)
    else:
        ### Numerical/Analytical derivatives of wave function (vector-state)
        if Quket.multi.nstates==0:
                # Currently analytical
            jac_wrap_mpi = lambda theta_list: jac_mpi_deriv(create_state, Quket, theta_list, init_state=None, current_state=Quket.state)
        else:
            prints('Not yet supported.')
            return 

    """
      Set guess
    """
    cf.t_old = time.time()
    opt = minimize(
           cost_wrap_mpi,
           theta_list,
           jac=jac_wrap_mpi,
           method=Quket.cf.opt_method,
           options=Quket.cf.opt_options,
           callback=lambda x: cost_callback(x, print_control=1),
          )
    cost_callback(opt.x, print_control=2),
    prints('VQE Done.')
    Quket.theta_list = opt.x
    Quket.energy = opt.fun
    Quket.state = create_state(opt.x, Quket.init_state)

