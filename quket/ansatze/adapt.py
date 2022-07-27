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
# See the License for the specific language governing permissions and
# limitations under the License.
"""
######################j
#        quket        #
#######################

adapt.py

Main driver of ADAPT-VQE.

"""

import numpy as np
import time
import copy

from scipy.optimize import minimize
from qulacs import QuantumCircuit
from qulacs.state import inner_product

from quket.mpilib import mpilib as mpi
from quket import config as cf

#from .pauli_algebra.fast_commutator import fast_commutator_general
from quket.lib import QubitOperator, QuantumState, commutator
from quket.post import prop 
from quket.fileio import prints, print_state, print_amplitudes_adapt, error, SaveAdapt, LoadAdapt, SaveTheta, LoadTheta
from quket.utils import Gdoubles_list
from quket.utils import cost_mpi, jac_mpi_num, jac_mpi_deriv
from quket.opelib import spin_single_grad, spin_double_grad,pauli_grad, OpenFermionOperator2QulacsObservable
from quket.opelib import evolve 
from quket.pauli import get_pauli_list_adapt, get_allowed_pauli_list, get_pauli_list_uccgsd
from quket.opelib import Gdouble_ope, Gdouble_fswap, create_exp_state
from quket.opelib import single_ope_Pauli
from quket.projection import S2Proj

ALPHA = 0
BETA  = 1

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

def adapt_vqe_driver(Quket, opt_method, opt_options, print_level, maxiter):
    """Function:
    Main driver for qubit-ADAPT-VQE

    Several adapt-mode can be set to construct an operator pool:
    pauli: decompose all the paulis in each excitation and remove all Z rotations
    pauli_spin: use spin excitations (2 for singles and 8 for doubles) i.e., do not decompose
    pauli_spin_xy, qeb: same as pauli_spin but all Z rotations are removed
    pauli_sz: decompose partially the paulis in each excitation such that each pauli 
              string can preserve Sz and Number. All Z rotations are removed.
    pauli_yz: Y0, Y1, ..., Yn-2, Z0Y1, Z1Y2, ..., Zn-2Yn-1. 
              Note that they all ionize, either attach or detach one electron 
              (that is, they do not contain number-preserving components) 
              so not a favorable choise for chemical Hamiltonian.
    
    Args:
        Quket (QuketData): Quket data
        opt_method (str): Minimization method (L-BFGS-B)
        opt_options (dict): Options for minimize
        print_level (int): Printing level
        maxiter (int): Maximum number of iterations


    Author(s): Masaki Taii, Takashi Tsuchimochi
    """

    t0 = time.time()
    cf.t_old = t0

    #############################
    ### set up cost functions ###
    #############################
    ##number of operator is going to dynamically increase
    ndim = 0
    ndim1 = 0
    ndim2 = 0
    theta_list = []  ## initialize
    discard = True  # whether the other complement paulis are discarded
    pauli_list, pauli_ncnot_dict = get_pauli_list_adapt(Quket)
    #for i, pauli in enumerate(pauli_list):
    #    prints('i=',i,'   ', pauli)
    state = Quket.init_state

    Quket.pauli_list = [] 
    ### Tight convergence 
    if Quket.adapt.eps < 1e-4:
        Quket.ftol = min(1e-13, Quket.ftol)
        Quket.gtol = min(1e-6, Quket.gtol)
        opt_options["ftol"] = Quket.ftol
        opt_options["gtol"] = Quket.gtol

    fstr = f"0{Quket.n_qubits}b"
    prints("Performing ADAPT-VQE ", end="")
    if Quket.projection.SpinProj:
        prints("(spin-projection) ", end="")
    prints(f"[{Quket.adapt.mode}]")
    prints(f"ADAPT Convergence criterion = {Quket.adapt.eps:1.0E}")
    prints(f"Initial configuration: | {format(Quket.current_det, fstr)} >")
    prints(f"VQE Convergence criteria: ftol = {Quket.ftol:1.0E}, "
                                 f"gtol = {Quket.gtol:1.0E}")
    # How many derivatives do we have to evaluate? 
    nderiv =len(pauli_list)
    prints(f"Number of operators in pool: {nderiv}") 
    if Quket.adapt.adapt_guess == "read" or Quket.adapt.adapt_guess == "continue":
        LoadAdapt(Quket, cf.adapt)
        ndim_ = len(Quket.pauli_list)
        previous_pauli_list = Quket.pauli_list.copy()
        #theta_list = LoadTheta(ndim_, cf.theta_list_file, offset=0)
        #ndim = len(theta_list)
        #if ndim < ndim_:
        #    for icyc in range(ndim_ - ndim): 
        #        Quket.pauli_list.pop(-1)
        if Quket.adapt.adapt_guess == "read":
            # initialize theta_list to start over
            theta_list = []
            ndim = 0
    # ADAPT excitation list may have been read.
    # For the first nprev cycles, we will skip calculating grad to decide the excitation,
    # but rather use the read excitation list.
    nprev = len(Quket.pauli_list)
    #Quket.energy = Quket.qulacs.Hamiltonian.get_expectation_value(Quket.state)
    Quket.energy = Quket.get_E()
    Energy_old = Quket.energy
    ncnot = 0
    vqe_cyc = 0
    while True:  ##for ADAPT-VQE
        t1 = time.time()
        vqe_cyc += 1
        Quket.ndim = ndim
        #############################
        ### set up initial theta  ###
        #############################
        ndim += 1
        theta_list = np.append(theta_list, 0)
        if nprev < ndim:
            grad = grad_driver_pauli(Quket, theta_list, pauli_list)
            #SaveAdapt(Quket, cf.adapt)
        else:
            grad = Quket.adapt.grad_list[ndim-1]
        ### Check if the same operator has been chosen consecutively.

        def chk_new_operator(Quket):
            if Quket.ndim < 2:
                return True
            current_pauli = Quket.pauli_list[-1]
            previous_pauli = Quket.pauli_list[-2]
            if (current_pauli == previous_pauli):
                return False
            else:
                return True
        if(not chk_new_operator(Quket) and nprev < ndim):
            prints(f"The same operator has been chosen as the last ADAPT cycle!")
            # Remove the last pauli and save
            Quket.pauli_list.pop(-1)
            SaveAdapt(Quket, cf.adapt)
            break

        if not Quket.adapt.lucc_nvqe or (vqe_cyc < Quket.adapt.svqe and Quket.adapt.lucc_nvqe):
            cost_wrap = lambda theta_list: cost_adapt_pauli(theta_list, Quket)[0]
            cost_wrap_serial = lambda theta_list: cost_adapt_pauli(theta_list,  Quket, parallel=False)[0]
            cost_callback = lambda theta_list: cost_adapt_pauli(theta_list, Quket, parallel=False)[0]
            cost_wrap_last = lambda theta_last: cost_adapt_pauli_last(theta_last,  Quket)[0]

            theta_list = np.array(theta_list)

            ### Broadcast lists
            theta_list = mpi.bcast(theta_list, root=0)

        #abs for lucc_nvqe
        if abs(grad) < Quket.adapt.eps:
            ndim -= 1
            prints("gradient norm = %e < %e" % (grad, Quket.adapt.eps))
            break

        if Quket.adapt.max <= ndim-1:
            ndim -= 1
            prints("adapt_max = ndim =", Quket.adapt.eps)
            break

        ### Number of CNOT gates is counted assuming no tapering-off of qubits 
        ncnot = count_CNOT(Quket.pauli_list, pauli_ncnot_dict)
        if Quket.projection.SpinProj:
            ncnot += len(Quket.adapt.bs_orbitals) * 8
        if ncnot >= Quket.adapt.max_ncnot:
            ndim -= 1
            prints(f"ncnot = {ncnot} > max_ncnot = {Quket.adapt.max_ncnot}")
            break

        #svqe = stop vqe / when quket stops vqe
        if (maxiter > 0 and not Quket.adapt.lucc_nvqe) or (vqe_cyc < Quket.adapt.svqe and Quket.adapt.lucc_nvqe):
            #prints('Do VQE')
            ###################
            ### perform VQE ###
            ###################
            cf.icyc = 0
            # Use MPI for evaluating Gradients
            cost_wrap_mpi = lambda theta_list: cost_mpi(cost_wrap, theta_list)
            create_state = lambda theta_list, init_state: create_adapt_state_pauli(Quket,theta_list, init_state=Quket.init_state)[0]
            if Quket.constraint_lambda_Sz > 0:
                jac_wrap_mpi = lambda theta_list: jac_mpi_num(cost_wrap_serial, theta_list)
                #jac_wrap_mpi = None
                #jac_wrap_mpi = lambda theta_list: jac_mpi_deriv(create_state, Quket, theta_list)
            else:
                jac_wrap_mpi = lambda theta_list: jac_mpi_deriv(create_state, Quket, theta_list, init_state=None, current_state=Quket.state)
            # For sanity check
            cost_wrap_mpi_last = lambda theta_last: cost_mpi(cost_wrap_last, theta_last)
            #create_state_last = lambda theta_list, init_state: create_adapt_state_pauli(Quket,theta_list, init_state=Quket.init_state)[0]
            #jac_wrap_mpi_last = lambda theta_last: jac_mpi_deriv(create_state_last, Quket, theta_last)

            # update ftol and gtol
            opt_options["ftol"] = Quket.ftol
            opt_options["gtol"] = Quket.gtol
            tx = time.time()
            # Do VQE
            opt = minimize(
                cost_wrap_mpi,
                theta_list,
                jac=jac_wrap_mpi,
                method=opt_method,
                options=opt_options,
                callback=lambda x: cost_callback(x),
            )

            theta_list = opt.x  ##use for next VQE

            #if Quket.adapt.grad_list[-1] < 1e-5:
            #    # Just in case the last parameter is minimized to make sure 0 gradient.
            #    opt = minimize(
            #        cost_wrap_mpi_last,
            #        [theta_list[-1]],
            #        method=opt_method,
            #        options=opt_options,
            #        callback=lambda x: cost_callback(x),
            #    )
            #    theta_list[-1] = opt.x[0]

            ty = time.time()
            if cf.timing:
                prints(f'VQE part: {ty-tx}')
            Quket.theta_list = theta_list

        elif Quket.adapt.lucc_nvqe: 
            Quket.theta_list = theta_list

        elif maxiter == -1:
            # Skip VQE, and perform one-shot calculation
            prints("One-shot evaluation without parameter optimization")

        # prints('theta_list =', theta_list)
        tm = time.time()
        cost, Energy, S2 = cost_adapt_pauli(Quket.theta_list, Quket)
        t2 = time.time()
        if cf.timing:
            prints('cost_adapt:', t2-tm)
        cput = t2 - t1
        state, circuit_depth = create_adapt_state_pauli(Quket,theta_list)
        prints(
            "{cyc:5}:".format(cyc=ndim),
            "  E[{}] = {: .12f}   <S**2> = {: 17.14f}".format(Quket.ansatz, Energy, S2),
            "  Grad = {:4.2e}   Fidelity = {:6f}   CPU Time = {:5.2f}".format(grad, Quket.fidelity(), cput),
            "  <N> = {: 17.14f}".format(Quket.get_N()),
            "  <Sz> = {: 17.14f}".format(Quket.get_Sz()),
            #"  Circuit Depth =",circuit_depth,
            "  CNOT =",ncnot#, "   CNOT_SWAP = ",cf.ncnot_swap 
        )
        if (Energy_old - Energy) == 0:
            prints("STOP: Energy did not change. Too small gradients that scipy's minimize function cannot nail down.")
            break

        Quket.energy = Energy
        Energy_old = Energy
        Quket.s2 = S2
        SaveTheta(ndim, theta_list, cf.theta_list_file)
        if Quket.adapt.adapt_prop:
            prop(Quket)

    prints("")
    prints("-----------ADAPT-VQE finished----------")
    prints("number of parameter is ", ndim)
    Quket.pauli_list = Quket.pauli_list[:ndim]
    #print_amplitudes_adapt(theta_list, Quket)
    cost, Energy, S2 = cost_adapt_pauli(Quket.theta_list, Quket)
    prints(
        "Final: E[{}] = {:.12f}   <S**2> = {:17.15f}   Fidelity = {:6f}".format(Quket.ansatz, Energy, S2, 
        Quket.fidelity()),
    )
    if Quket.projection.SpinProj:
        ### Ensure 'state' is projected over Sz
        Quket.state = S2Proj(Quket, state, nalpha=max(Quket.projection.euler_ngrids[0], Quket.projection.euler_ngrids[1]*2))

    Quket.print_state()
    t2 = time.time()
    cput = t2 - t0
    prints("\n Done: CPU Time =  ", "%15.4f" % cput)

def create_adapt_state_pauli(Quket, theta_list, depth=None, init_state=None):
    """Function
    Prepare the initial states/substitute the following values based on the molecule you will calculate

    Args:
        Quket (QuketData): Quket data
        theta_list ([float]): parameter list for theta (dimension of depth)

    Returns:
        state (QuantumState): Updatated quantum state
    """
    #n_qubits = Quket.n_qubits
    #state = QuantumState(n_qubits)
    #circuit = QuantumCircuit(n_qubits)
    #if init_state is None:
    #    state = Quket.init_state.copy()
    #else:
    #    state = init_state
    if depth == None and theta_list is not None:
            depth = len(theta_list)
    if theta_list is not None:
        state = create_exp_state(Quket, init_state=init_state, pauli_list=Quket.pauli_list[:depth], theta_list=theta_list[:depth], rho=1)
    else:
        state = Quket.init_state
    # Calculate ncnot

    return state, 0

def create_adapt_state_pauli_last(Quket, theta_list, theta_last, init_state=None):
    """Function
    Prepare the ADAPT state based on pauli_list, but with the last element of theta_list replaced by theta_last 

    Args:
        Quket (QuketData): Quket data
        theta_list ([float]): parameter list for theta (dimension of depth)
        theta_last (float): last parameter
        init_state : dummy

    Returns:
        state (QuantumState): Updatated quantum state
    """
    state, dummy = create_adapt_state_pauli(Quket, theta_list[:-1], init_state=init_state)
    state = create_exp_state(Quket, init_state=state, pauli_list=[Quket.pauli_list[-1]], theta_list=theta_last, rho=1)
    return state, 0


def cost_adapt_pauli(theta_list, Quket, parallel=True):
    """Function
    Compute energy and S2 expectation values of ADAPT-VQE.

    Args:
        theta_list ([float]): Parameter list
        Quket (QuketData): Quket data

    Returns:
        cost (float): Energy expectation value
        S2 (float): S2 expectation value
    """
    state, dummy = create_adapt_state_pauli(Quket, theta_list)
    Quket.state = state

    """the expectation value of hamiltonian"""
    if Quket.projection.SpinProj:
        Quket.state_unproj = state.copy()

        Pstate = S2Proj( Quket , state, normalize=False)
        norm = inner_product(Pstate, state)  ### <phi|P|phi>
        Hstate = evolve(Quket.operators.qubit_Hamiltonian,  Pstate,  parallel=parallel)  ## HP|phi>
        Energy = (inner_product(state, Hstate)/norm).real   ## <phi|HP|phi>/<phi|P|phi>
        cost = Energy

        S2state = evolve(Quket.operators.qubit_S2,  Pstate,  parallel=parallel)  ## S2P|phi>
        S2 = (inner_product(state, S2state)/norm).real   ## <phi|HP|phi>/<phi|P|phi>

    else:
        ### Parallel
        Quket.state = state
        Energy = Quket.get_E()
        cost = Energy
        S2 = Quket.get_S2()


    ### Penalty Sz
    #  Sz ** 2
    if Quket.constraint_lambda_Sz > 0:
        Sz2 = Quket.operators.qubit_Sz * Quket.operators.qubit_Sz
        Sz2state = evolve(Sz2, state, parallel=parallel) ## Sz|Phi>
        penalty = Quket.constraint_lambda_Sz * (inner_product(state, Sz2state)).real ## <phi|Sz|phi> as penalty 
        cost += penalty 
    if Quket.constraint_lambda_S2_expectation > 0 :
        # penalty (<S**2> - s(s+1))
        S2state = evolve(Quket.operators.qubit_S2, state, parallel=parallel) ## S2|Phi>
        ## S2 = <Phi|S**2|Phi>
        penalty = Quket.constraint_lambda * inner_product(state, S2state).real
        cost += penalty
    return cost, Energy, S2

def cost_adapt_pauli_last(theta_last,  Quket):
    """Function
    Compute energy of ADAPT-VQE (as a function of last VQE parameter), to nail down the exact parameter that gives 0 gradient.

    Args:
        theta_list_last ([float]): The last VQE parameter
        theta_list ([float]): Parameter list 
        Quket (QuketData): Quket data

    Returns:
        cost (float): Energy expectation value
    """
    state, dummy = create_adapt_state_pauli_last(Quket, theta_list, theta_last)
    Quket.state = state

    """the expectation value of hamiltonian"""
    if Quket.projection.SpinProj:
        Quket.state_unproj = state.copy()

        Pstate = S2Proj( Quket , state, normalize=False)
        norm = inner_product(Pstate, state)  ### <phi|P|phi>
        Hstate = evolve(Quket.operators.qubit_Hamiltonian,  Pstate,  parallel=True)  ## HP|phi>
        Energy = (inner_product(state, Hstate)/norm).real   ## <phi|HP|phi>/<phi|P|phi>
        cost = Energy
        

        S2state = evolve(Quket.operators.qubit_S2,  Pstate,  parallel=True)  ## S2P|phi>
        S2 = (inner_product(state, S2state)/norm).real   ## <phi|HP|phi>/<phi|P|phi>


    else:
        ### Parallel
        Quket.state = state
        Energy = Quket.get_E()
        cost = Energy
        S2 = Quket.get_S2()


    ### Penalty Sz
    #  Sz ** 2
    if Quket.constraint_lambda_Sz > 0:
        Sz2 = Quket.operators.qubit_Sz * Quket.operators.qubit_Sz
        Sz2state = evolve(Sz2, state, parallel=True) ## Sz|Phi>
        penalty = Quket.constraint_lambda_Sz * (inner_product(state, Sz2state)).real ## <phi|Sz|phi> as penalty 
        #prints("penalty = ", penalty)
        cost += penalty 
    if Quket.constraint_lambda_S2_expectation > 0 :
        # penalty (<S**2> - s(s+1))
        S2state = evolve(Quket.operators.qubit_S2, state, parallel=True) ## S2|Phi>
        ## S2 = <Phi|S**2|Phi>
        penalty = Quket.constraint_lambda * inner_product(state, S2state).real
        cost += penalty
    return cost, Energy, S2

def grad_driver_pauli(Quket, theta_list, pauli_list):

    if Quket.adapt.lucc_nvqe:
        min_grad, min_theta = calculate_lucc_non_vqe_gradient(Quket, pauli_list)
        Quket.adapt.grad_list.append(min_grad)
        theta_list[-1] = min_theta
        grad = abs(min_grad)
        return grad

    max_grad = calculate_spin_adapt_gradient_pauli(Quket, theta_list, pauli_list, fullspin=True)
    Quket.adapt.grad_list.append(max_grad)
    return max_grad

def calculate_lucc_non_vqe_gradient(Quket,  pauli_list):

    def get_TE(Quket, ope):

        #compute theta and energy with first order
        #H_ope = fast_commutator_general(Quket.operators.qubit_Hamiltonian, ope)
        #H_ope_ope = fast_commutator_general(H_ope, ope)
        H_ope = commutator(Quket.operators.qubit_Hamiltonian, ope)
        H_ope_ope = commutator(H_ope, ope)

        H_ope = OpenFermionOperator2QulacsObservable(H_ope, Quket.n_qubits)
        H_ope_ope = OpenFermionOperator2QulacsObservable(H_ope_ope, Quket.n_qubits)

        H_ope = H_ope.get_expectation_value(Quket.state)
        if H_ope == 0:
            return 0., 0.
        H_ope_ope = H_ope_ope.get_expectation_value(Quket.state)

        if H_ope_ope == 0:
            return 0., 0.

        theta = -H_ope/H_ope_ope
        energy = -(H_ope**2)/H_ope_ope
        #prints('theta=',theta)
        #prints('energy=',energy)
        return theta, energy

    min_energy, min_theta, min_pauli = 0, 0, pauli_list[0]
    for pauli in pauli_list:
        theta, energy = get_TE(Quket, pauli)
        if energy < min_energy:
            min_theta = theta
            min_energy = energy
            min_pauli = pauli

    Quket.pauli_list.append(min_pauli)
    return min_energy, min_theta

def calculate_spin_adapt_gradient_pauli(Quket, theta_list, pauli_list, fullspin=False):
    """Function
    Compute ADAPT gradients for a set of spin dependent excitations.
    """

    t0 = time.time()
    spatial_orbitals = Quket.n_active_orbitals
    qubit_Hamiltonian = Quket.operators.qubit_Hamiltonian
    ndim = len( theta_list )
    #state here is not spin projected
    state, dummy = create_adapt_state_pauli(Quket, theta_list, depth=ndim-1)
    #en=Quket.qulacs.Hamiltonian.get_expectation_value(state)

    if ndim > 1:
        last = Quket.pauli_list[-1]
    else:
        last = QubitOperator('')


    tH = time.time()
    ### Faster gradient evaluation
    if cf.fast_evaluation:
        if Quket.projection.SpinProj:
            P_state = S2Proj( Quket , state, normalize=False)
            norm = inner_product(P_state, state)
            if abs(norm) < 1e-6:
                prints(f"ADAPT stopped because the norm <P> is too small! norm = {norm}")
                prints(f"This is because the reference state has no component for the target spin.")
                error("")
            H_state = evolve(Quket.operators.qubit_Hamiltonian,  P_state, parallel=True)
            P_state.multiply_coef(-Quket.energy)
            H_state.add_state(P_state)
            H_state.multiply_coef(1/norm)

        else:
            H_state = evolve(Quket.operators.qubit_Hamiltonian, state, parallel=True)
        if Quket.constraint_lambda_Sz > 0 :
            # penalty (Sz - Ms)**2
            Sz2 = Quket.operators.qubit_Sz * Quket.operators.qubit_Sz
            Sz2state = evolve(Sz2, state, parallel=True) ## Sz|Phi>
            ## Ms2 = <Phi|Sz**2|Phi>
            Ms2 = inner_product(state, Sz2state).real

            ## lambda (Sz**2 - Ms2)|Phi>
            state_ = state.copy()
            state_.multiply_coef(-Ms2)
            Sz2state.add_state(state_)
            Sz2state.multiply_coef(Quket.constraint_lambda_Sz)
            H_state.add_state(Sz2state)

        if Quket.constraint_lambda_S2_expectation > 0 :
            # penalty (<S**2> - s(s+1))
            S2state = evolve(Quket.operators.qubit_S2, state, parallel=True) ## S2|Phi>
            ## S2 = <Phi|S**2|Phi>
            S2 = inner_product(state, S2state).real

            ## lambda (S2-<S**2>)|Phi>
            state_ = state.copy()
            state_.multiply_coef(-S2)
            S2state.add_state(state_)
            S2state.multiply_coef(Quket.constraint_lambda_S2_expectation)
            H_state.add_state(S2state)
        numerical = False
    else:
        H_state = None
        numerical = True
    #print_state(H_state, name=f'H_state', threshold = 1e-6)
    my_a = my_b = my_i = my_j = 0
    my_gradient_list=[]
    my_pauli_list=[]
    my_grad = 0 #initial grad
    num     = 0 #for parallel calc
    ty = time.time()
    for pauli in pauli_list:
        if num % mpi.nprocs == mpi.rank:
            if type(pauli) is list:
                grad = 0
                for pauli_ in pauli:
                    grad += abs(pauli_grad(pauli_, state , qubit_Hamiltonian , Quket, numerical=numerical, H_state=H_state) )
            else:
                grad = abs(pauli_grad(pauli, state , qubit_Hamiltonian , Quket, numerical=numerical, H_state=H_state) )
            my_gradient_list.append( grad )
            present = pauli
            my_pauli_list.append( present )
            #if present == last:
            #    if grad > 1e-4: 
            #        prints(f'{present} was chosen last time and now gradient is {grad}', root=mpi.rank)
 
            if my_grad < grad:
                my_grad = grad
                my_pauli = pauli
        num += 1
       
    tx = time.time()
    ### Gather my_gradient_list and my_orbital_list
    data1 = mpi.gather(my_gradient_list, root=0)
    data2 = mpi.gather(my_pauli_list, root=0)
    if cf.timing:
        prints(f'create_adapt_state   {tH-t0}')
        prints(f'Hstate   {ty-tH}')
        prints(f'Gradient {tx-ty}')
    if mpi.rank == 0:
        gradient_list = [x for l in data1 for x in l]
        pauli_list = [x for l in data2 for x in l]
    else:
        gradient_list = None
        pauli_list = None
    gradient_list = mpi.bcast(gradient_list, root=0)
    pauli_list = mpi.bcast(pauli_list, root=0)
    if cf.debug:
        prints(f' obirtal        grad ')
        i = 0
        for (grad, pauli) in zip(gradient_list, pauli_list):
            if last == pauli:
                prints(f'i={i}   {pauli}  -->  {grad} is supposed to be zero!')
            elif abs(grad) > 1e-10:
                prints(f'i={i}   {pauli}  -->  {grad}')
            i+=1
    
    # calc grad norm
    grad_norm = np.linalg.norm( gradient_list )


    maxind = np.argmax(np.array(gradient_list))
    if pauli_list[maxind] == last:
        prints(f'Warning! This operator {pauli_list[maxind]} with grad = {gradient_list[maxind]} has been chosen last time.') 
        prints(f'Change the operator to the one with the next largest gradient.')
        gradient_list[maxind] = 0
        maxind = np.argmax(np.array(gradient_list))
        prints(f'Also, tighten ftol and gtol')
        Quket.ftol *= 0.1 
        Quket.gtol *= 0.1
    max_grad = gradient_list[maxind]
    # Check and compare a^ b^ i j. 
    # If grad[aB^ bA^ iB^ jA] ~ grad[bB^ aA^ jB^ iA] with b > a 
    # then the former is better in terms of CNOT count
    our_pauli = pauli_list[maxind]
    Quket.pauli_list.append(our_pauli)
    if cf.debug and mpi.main_rank:
        prints(f'Chosen : pauli={our_pauli} max_grad={max_grad}')
#    if our_spin == 8:
#        # There should be ABAB
#        if our_a > our_i:
#            ind_reverse = orbital_list.index([our_a, our_b, our_i, our_j, our_spin]) 
#            if abs(gradient_list[ind_reverse] - gradient_list[maxind]) < 1e-5: 
#                if (our_b - our_a) > (our_i - our_j):
#                    our_b, our_a, our_j, our_i = our_a, our_b, our_i, our_j
#                    if cf.debug and mpi.main_rank:
#                        prints(f'Reset : b={our_b}  a={our_a}  j={our_j}  i={our_i}  spin={our_spin} max_grad={max_grad}')

    ### Only broken-symmetry orbitals are subject to spin-projection.
    bs_orbitals = copy.deepcopy(Quket.adapt.bs_orbitals)
    ### TODO: Check this
    target = []
    if type(our_pauli) == list: 
        for our_pauli_ in our_pauli:
            for op, coef in our_pauli_.terms.items():
                for op_ in op:
                    m = op_[0]//2
                    if op_[1] == 'X' or op_[1] == 'Y':
                        target.append(m)
                break
        
    else:
        for op, coef in our_pauli.terms.items():
            for op_ in op:
                m = op_[0]//2
                if op_[1] == 'X' or op_[1] == 'Y':
                    target.append(m)
            break
    if len(target) == 2:
        # Singles, or doubles with either creation/annihilation being same orbital (e.g., p_alpha^ p_beta^ q_beta r_alpha)
        for m in target:
            if m not in bs_orbitals:
                bs_orbitals.append(m)
    elif len(target) == 4:
        # Symmbetry-breaking doubles
        for m in target:
            if target.count(m) == 1 and m not in bs_orbitals:
                bs_orbitals.append(m)

#

    Quket.adapt.bs_orbitals = bs_orbitals
    if cf.debug:
        prints(f"bs_orbitals = {Quket.adapt.bs_orbitals}")
    return grad_norm

def count_CNOT(pauli_list, pauli_ncnot_dict):
    ncnot = 0
    for pauli in pauli_list:
        ncnot += pauli_ncnot_dict[str(pauli)]
    return ncnot

