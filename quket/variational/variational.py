import numpy as np
import itertools

#from .pauli import set_circuit_RTE
from quket.opelib import set_exp_circuit
#from openfermion.ops import FermionOperator
#from openfermion.transforms import jordan_wigner
from quket.quket_data import QuketData
from qulacs.gate import PauliRotation
from quket.opelib import evolve
from quket.mpilib import mpilib as mpi
from qulacs.state import inner_product
from qulacs import QuantumState, QuantumCircuit
from quket import config as cf
from quket.utils import get_ndims
from quket.ansatze import mix_orbitals
from quket.fileio import LoadTheta, error, prints, printmat, print_state
from quket.utils  import get_occvir_lists
from quket.utils  import jac_mpi_num
from quket.linalg import T1mult
from quket.qite.qite_function import calc_expHpsi
from quket.vqe import VQE_driver
#from quket.ucclib import set_circuit_uccsdX, cost_uccsdX, create_uccsd_state, create_sauccsd_state, create_mbe_uccsd_state 

def create_real_state(n_qubits, rho, theta_list, pauli_list, initial_state, Quket):
    state = initial_state.copy()
    #theta_list = [1]
    #pauli_list = [pauli_list[0]]

    #print_state(state,name="state0")
    #print_state(initial_state,name="initial_state0")
    #prints(f"theta_list = {theta_list}")
    #prints(f"pauli_listt = {pauli_list}")

    circuit = set_exp_circuit(n_qubits, pauli_list, theta_list)
    #circuit = set_circuit_RTE(Quket)

    for i in range(rho):
        circuit.update_quantum_state(state)

    #print_state(state, name="state1")
    #print_state(initial_state ,name="initial_state1")
    return state

def create_V(loop, theta_list, n_qubits, rho, DS, det, ndim1, ucc, stepsize, method, pauli_list, initial_state, ansatz, noa, nva, Quket):
    theta_list[loop] += stepsize
    #prints(f"{pauli_list}")
    #prints(f"{theta_list}")
    if method == "variational_imaginary":
        state = create_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1)
    elif method == "variational_real":
        state = create_real_state(n_qubits, rho, theta_list, pauli_list, initial_state, Quket)
    #if ansatz == "sauccsd":
    #    state = create_sauccsd_state(n_qubits, noa, nva, rho, DS, theta_list, det, ndim1)
    #elif ansatz == "mbe":
    #    state = create_mbe_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1)
    #prints(f"{state}")
    state.add_state(ucc)
    #print_state(ucc, name="ucc")
    #print_state(state, name="state", threshold = 1e-5)
    
    state.multiply_coef(1/stepsize)
    theta_list[loop] -= stepsize
    return state

def Variational(Quket, kappa_guess, theta_guess, mix_level, Kappa_to_T1):
    prints("Entered variational")
    qubit_hamiltonian = Quket.operators.qubit_Hamiltonian
    qubit_s2 = Quket.operators.qubit_S2
    ansatz = Quket.ansatz
    n_qubits = Quket.n_qubits
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    nca = ncb = Quket.nc
    norbs = Quket.n_active_orbitals
    rho = Quket.rho
    DS = Quket.DS
    det = Quket.current_det
    method = Quket.method
    dt = Quket.dt
    initial_state = Quket.state.copy()
    ansatz = Quket.ansatz
    maxiter = Quket.maxiter


    #print_state(initial_state, name = "initial")
    #Quket.get_pauli_list()
    pauli_list = Quket.pauli_list
    #prints(f"{pauli_list}")

    #H = Quket.operators.Hamiltonian
    #prints(f"{H}")
    #for key, value in H.terms.items():
    #    prints(f"{key}")

    #x = jordan_wigner(FermionOperator('6^ 0') + FermionOperator('0^ 6'))
    #prints(f"{x}")

    optk = 0
    Gen = 0

    #get ndim
    ndim1, ndim2, ndim = get_ndims(Quket)

    #set number of dimensions QuketData
    Quket.ndim1 = ndim1
    Quket.ndim2 = ndim2
    Quket.ndim = ndim

    #set up kappa_list
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
            temp = kappa_list[:ndim1]
            mix = mix_orbitals(noa, nob, nva, nvb, mix_level, False, np.pi/4)
            temp = T1mult(noa, nob, nva, nvb, mix, temp)
            kappa_list = temp[:ndim1]
        printmat(kappa_list)
    elif kappa_guess == "zero":
        kappa_list *= 0

    #set up theta_list
    ndim = len(pauli_list)
    theta_list = np.zeros(ndim)
    prints(f"Theta list = {theta_guess}")
    if theta_guess == "zero":
        theta_list *= 0
    elif theta_guess == "read":
        theta_list = LoadTheta(ndim, cf.theta_list_file, offset=istate)
    elif theta_guess == "random":
        theta_list = (0.5-np.random.rand(ndim))*0.1
    if Kappa_to_T1 and theta_guess != "read":
        ### Use Kappa for T1  ###
        theta_list[:ndim1] = kappa_list[:ndim1]
        kappa_list *= 0
        prints("Initial T1 amplitudes will be read from kappa.")
    
    if optk:
        theta_list_fix = theta_list[ndim1:]
        theta_list = theta_list[:ndim1]
        if Gen:
            # Generalized Singles.
            temp = theta_list.copy()
            theta_list = np.zeros(ndim)
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

    prints(f"Number of VQE parameters: {ndim}")

    fstr = f"0{n_qubits}b"
    prints(f"Initial configuration: | {format(Quket.current_det, fstr)} >")

    time = 0
    stepsize = 1e-08

    #prints(f"{Quket.cf.opt_method}")
    #opt_method = Quket.cf.opt_method
    #opt_options = Quket.cf.opt_options
    #print_level = Quket.cf.print_level
    #Quket.method = "vqe"
    #Quket.ansatz = "uccsd"
    #Quket.run()
    #prints("hoge")
    #prints(f"{Quket}")
    #prints(f"{kappa_guess}")
    #prints(f"{theta_guess}")
    #prints(f"{mix_level}")
    #prints(f"{opt_method}")
    #prints(f"{opt_options}")
    #prints(f"{print_level}")
    #prints(f"{maxiter}")
    #prints(f"{Kappa_to_T1}")
    #Quket.get_pauli_list()
    #prints(f"aaaa")
    #Eqe, S2 = VQE_driver(Quket, kappa_guess, theta_guess, mix_level, opt_method, opt_options, print_level, maxiter, Kappa_to_T1)
    #prints(f"bbbb")
    #true_state = Quket.state.copy()
    #print_state(true_state, name = "true_wave_function")
    #Quket.method = "variational_real"
    #Quket.ansatz = "hamiltonian" 
    #Quket.get_pauli_list()

    #Distance_list = []
    State_list = []

    for t in range(maxiter):

        time += dt

        #prints(f"aaaa")
        # set up cost_function
        #cost_wrap = lambda theta_list: cost_uccsdX(Quket, 0, kappa_list, theta_list)[0]

        #C の数値微分
        #grad = jac_mpi(cost_wrap, theta_list)
        #C = list(map(lambda x : x*(-0.5), grad))
        #prints("Cの数値微分")
        #prints(f"C (jac_mpi):{grad}")
        #prints(f"C:{C}")
        #prints(f"theta_list{theta_list}")
        #grad = lambda theta_list: jac_mpi(cost_wrap, theta_list)

        #C の回路(uccsd)
        #fstr = f"0{n_qubits}b"
        #prints(f"Initial configuration: | {format(Quket.current_det, fstr)} >")
        #state = create_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1)
        #prints("Cの回路")
        #stepsize = 1e-8
        #ipos, my_ndim = mpi.myrange(ndim)
        #for iloop in range(ipos, ipos+my_ndim):
        #    theta_list[iloop] += stepsize
        #    state = create_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1)
        #    E =  Quket.qulacs.Hamiltonian.get_expectation_value(state)
        #    theta_list[iloop] -= stepsize
        #    prints(f"E : {E}")

        ### Given that UCC is expressed as
        ###   |UCC>   = Un Un-1 ... Ui ... U1 U0 |HF>
        ### where each Ui is exp[p^ q - q^ p], etc., and contains parameter theta_i, 
        ### the idea is to create quantum states 
        ###   |Vi>  = d |UCC> / d theta_i
        ###         = Un Un-1 ... sigma_i Ui ... U1 U0 |HF>
        ### where 
        ###   sigma_i = jordan_wigner(p^ q - q^ p)
        ###
        ### create_uccsd_state() creates a quantum circuit for
        ###      Un Un-1 ... Ui ... U1 U0 
        ### and applies it to |HF> to determine |UCC>.
        ### However, we need a quantum ciruit for
        ###         Un Un-1 ... sigma_i Ui ... U1 U0
        ###                     ~~~~~~~
        ### to determine |Vi>, which is used to get the derivative C.
        ###
        ### We can cheat (for simulation) to create |Vi>  numerically
        ### |Vi> = (|UCC(theta_i + delta)> - |UCC(theta_i)>) / delta 
        ### as attempted in the above for-loop. 

        #my_C = np.zeros(ndim, dtype=complex)
        C = np.zeros(ndim, dtype=complex)
        #print_state(initial_state, name = "c_calc")
        #prints("create_ucc")
        #ucc = create_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1)
        if method == "variational_imaginary":
            ucc = create_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1)
        elif method == "variational_real":
            ucc = create_real_state(n_qubits, rho, theta_list, pauli_list, initial_state, Quket)
        #if ansatz == "sauccsd":
        #    ucc = create_sauccsd_state(n_qubits, noa, nva, rho, DS, theta_list, det, ndim1)
        #elif ansatz == "mbe":
        #    ucc = create_mbe_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1)
        #prints(f"{ucc}")
        #ucc.multiply_coef(-1)   ### -|UCC>
        #print_state(ucc, name = "ucc")
        #braを計算するのにuccに符号をかけていてこれにHをかけてHuccを作っているので、Huccは - H| ucc(θ)>が入っている
        HUCC = evolve(Quket.operators.qubit_Hamiltonian, ucc)
        ucc.multiply_coef(-1)   ### -|UCC>
        
        #ipos, my_ndim = mpi.myrange(ndim)
        #prints(f"ipos, my_ndim = {ipos}, {my_ndim}")
        #my_state_list = []
        state_list = []
        #for iloop in range(ipos, ipos+my_ndim):
            #state = create_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1)
            ### get |vi> = ( |UCC(theta_i + delta)> - |UCC> ) / delta
            #state.add_state(ucc)
            #state.multiply_coef(1/stepsize)
        
        #    state = create_V(iloop, theta_list, n_qubits, rho, DS, det, ndim1, ucc, stepsize)
        #    my_state_list.append(state) 

            ### Ci = <vi | H | UCC> 
            ###   (= <HF | U0^ U1^ ... Ui^ sigma_i^ ... Un-1^ Un^  H  Un Un-1 ... Ui ... U1 U0 |HF> )
            #Ci =  Quket.qulacs.Hamiltonian.get_transition_amplitude(state, ucc)
            #my_C[iloop] = Ci
        #    my_C[iloop] = inner_product(state, HUCC)
        #mpi.comm.Allreduce(my_C, C, mpi.MPI.SUM)
        #mpi.comm.Allgather(my_state_list, state_list)
        #prints("create_C")
        #print_state(initial_state)
        for iloop in range(ndim):
            #prints(f"iloop")
            #prints(f"i = {iloop}")
            state = create_V(iloop, theta_list, n_qubits, rho, DS, det, ndim1, ucc, stepsize, method, pauli_list, initial_state, ansatz, noa, nva, Quket)
            #print_state(state, name = "create_vi")
            state_list.append(state)

            C[iloop] = inner_product(state, HUCC)
        #prints(f"{C}")
        if method == "variational_real":
            #prints(f"method : {method}")
            #prints(#f"C : {C}")
            C = list(map(lambda x : x*(-1j), C))
            #prints(f"C : {C}")
        elif method == "variational_imaginary":
            #prints(f"method : {method}")
            C = list(map(lambda x : x*(-1), C))

        #my_A = np.zeros((ndim, ndim), dtype=complex)
        A = np.zeros((ndim, ndim), dtype=complex)    

        #for jloop in range(ipos, ipos+my_ndim):
        #    for iloop in range(ndim):
        #        my_A[iloop][jloop] = inner_product(state_list[iloop], state_list[jloop])
        #for iloop in range(ndim):
        #    mpi.comm.Allreduce(my_A[iloop], A[iloop], mpi.MPI.SUM)
        for i in range(ndim):
            for j in range(i):
                A[i][j] = inner_product(state_list[i], state_list[j])
                A[j][i] = A[i][j]
        for i in range(ndim):
            A[i][i] = inner_product(state_list[i], state_list[i]) 

        #theta_listの更新
        #prints(f"C {C}")
        #prints(f"A {A}")
        Ainv = np.linalg.pinv(A)
        theta_list += (np.dot(Ainv, C)).real*dt
        #prints(f"{theta_list}")

        if method == "variational_imaginary":
            state = create_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1)
        elif method == "variational_real":
            state = create_real_state(n_qubits, rho, theta_list, pauli_list, initial_state, Quket)
        #if ansatz == "sauccsd":
        #    state = create_sauccsd_state(n_qubits, noa, nva, rho, DS, theta_list, det, ndim1)
        #elif ansatz == "mbe":
        #    state = create_mbe_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1)
        E =  Quket.qulacs.Hamiltonian.get_expectation_value(state)
        State_list.append(state)
        #FCI(H2_1.25) = -1.0457831445498
        #dif = E + 1.0457831445498
        #自己相関関数
        autocorr = inner_product(Quket.state, state)
        prints(f"Time = {time:7.3f}  E = {E:12.8f}  <phi[t=0] | phi[t]> {autocorr}")
        #trial_state = state.copy()
        #print_state(trial_state, name = "trial_wave_function")
        
        #true_wave_function
        #E0 = Quket.get_E(true_state)
        #true_state = calc_expHpsi(true_state, Quket.qulacs.Hamiltonian, n_qubits, -dt*1j,  shift=E0)[0].copy()
        #true_state.multiply_coef(np.exp(-dt * 1j * E0))
        #print_state(true_state, name = "true_wave_function")

        #calc_distance
        #true_state.multiply_coef(-1)
        #trial_state.add_state(true_state)
        #print_state(trial_state)
        #true_state.multiply_coef(-1)
        #distance = np.sqrt(trial_state.get_squared_norm())/2 
        #distance = trial_state.get_squared_norm()/2
        #Distance_list.append(distance)

        #prints(f"Trace Distance = {distance}")
        #prints(f"{distance}")
        #prints(f"{time/2/np.pi}")

        #inverse iteration
        #H_eff = evolve(Quket.operators.qubit_Hamiltonian, state)
        #H_eff = inner_product(state, H_eff)
        #S_eff = inner_product(state, state)
        #prints(f"H_eff = {H_eff}")
        #prints(f"S_eff = {S_eff}")
        #eig = np.linalg.eig(np.linalg.pinv(S_eff) @ H_eff)[0].real
        #prints(f"icyc = {t}, eigenvalue = {eig}")

    #print_state(state, threshold = 0.000001)
    #Quket.distance = Distance_list
    Quket.State_list = State_list
