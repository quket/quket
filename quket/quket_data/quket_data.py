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

"""
import os
import pickle
from copy import deepcopy
from typing import List, Dict
from dataclasses import dataclass, field, InitVar, make_dataclass

import numpy as np
import openfermion
from qulacs import Observable
from qulacs.state import inner_product
from qulacs.observable import create_observable_from_openfermion_text
from openfermion.ops import InteractionOperator
try:
    from openfermion.utils import QubitDavidson
except:
    from openfermion.linalg import QubitDavidson

from quket.lib import FermionOperator, QubitOperator, QuantumState, jordan_wigner, bravyi_kitaev
from quket.tapering.tapering import Z2tapering
from quket.utils import chkdet, chkmethod, chkpostmethod, get_func_kwds, set_initial_det, set_multi_det_state, remove_unpicklable
from quket.utils import fci2qubit, transform_state_jw2bk, transform_state_bk2jw
from quket.fileio import error, prints, printmat, SaveTheta, print_state, print_geom, tstamp
from quket import config as cf
from quket.mpilib import mpilib as mpi


@dataclass(repr=False)
class Operators():
    """
    Operator sections.

    Attributes:
        Hamiltonian (InteractionOperator): Hamiltonian operator.
        S2 (InteractionOperator): S2 operator.
        Number (InteractionOperator): Number operator.
        qubit_Hamiltonian (QubitOperator): Qubit hamiltonian.
        qubit_S2 (QubitOperator): Qubit S2.
        qubit_Number (QubitOperator): Qubit number operator.
        Dipole (list): Dipole moment.
    """
    mapping: InitVar[str] 
    n_qubits: InitVar[int] 
    Hamiltonian: InteractionOperator = None
    S2: InteractionOperator = None
    Sz: InteractionOperator = None
    Number: InteractionOperator = None
    Dipole: List = None
    S4: InteractionOperator = None

    qubit_Hamiltonian: QubitOperator = field(init=False, default=None)
    qubit_S2: QubitOperator = field(init=False, default=None)
    qubit_Sz: QubitOperator = field(init=False, default=None)
    qubit_Number: QubitOperator = field(init=False, default=None)
    qubit_S4: QubitOperator = field(init=False, default=None)

    def __post_init__(self, mapping="jordan_wigner", n_qubits=None, *args, **kwds):
        if mapping == "jordan_wigner":
            self.jordan_wigner()
        elif mapping == "bravyi_kitaev":
            self.bravyi_kitaev(n_qubits)

    def jordan_wigner(self):
        if self.Hamiltonian is not None:
            self.qubit_Hamiltonian = jordan_wigner(self.Hamiltonian)
        else:
            self.qubit_Hamiltonian = None
        if self.S2 is not None:
            self.qubit_S2 = jordan_wigner(self.S2)
        else:
            self.qubit_S2 = None
        if self.Sz is not None:
            self.qubit_Sz = jordan_wigner(self.Sz)
        else:
            self.qubit_Sz = None
        if self.Number is not None:
            self.qubit_Number = jordan_wigner(self.Number)
        else:
            self.qubit_Number = None
        if self.S4 is not None:
            self.qubit_S4 = jordan_wigner(self.S4)
        else:
            self.qubit_S4 = None

    def bravyi_kitaev(self, n_qubits):
        if self.Hamiltonian is not None:
            if isinstance(self.Hamiltonian, (openfermion.FermionOperator, FermionOperator)):
                self.qubit_Hamiltonian = bravyi_kitaev(self.Hamiltonian, n_qubits)
            else:
                from quket.lib import get_fermion_operator
                self.qubit_Hamiltonian = bravyi_kitaev(get_fermion_operator(self.Hamiltonian), n_qubits)
        else:
            self.qubit_Hamiltonian = None
        if self.S2 is not None:
            self.qubit_S2 = bravyi_kitaev(self.S2, n_qubits)
        else:
            self.qubit_S2 = None
        if self.Sz is not None:
            self.qubit_Sz = bravyi_kitaev(self.Sz, n_qubits)
        else:
            self.qubit_Sz = None
        if self.Number is not None:
            self.qubit_Number = bravyi_kitaev(self.Number, n_qubits)
        else:
            self.qubit_Number = None
        if self.S4 is not None:
            self.qubit_S4 = bravyi_kitaev(self.S4, n_qubits)
        else:
            self.qubit_S4 = None

@dataclass(repr=False)
class Config():
    # PySCF guess
    pyscf_guess: str = "minao"  # Guess for pyscf: 'minao', 'chkfile'
    # Lagrange multiplier for Spin-Constrained Calculation
    # qulacs (VQE part)
    print_level: int = 1  # Printing level
    print_fci: int = 0  # Whether fci is printed initially
    mix_level: int = 0  # Number of pairs of orbitals to be mixed (to break symmetry)
    kappa_guess: str = "zero"  # Guess for kappa: 'zero', 'read', 'mix', 'random'
    theta_guess: str = "zero"  # Guess for T1 and T2 amplitudes: 'zero', 'read', 'mix', 'random'
    Kappa_to_T1: int = 0  # Flag to use ***.kappa file (T1-like) for initial guess of T1

    # scipy.optimize
    opt_method: int = "l-bfgs-b"  # Method for optimization
    eps: float = 1e-6  # Numerical step
    maxfun: int = 10000000000  # Maximum function evaluations. Virtual infinity.
    gtol: float = 1e-5  # Convergence criterion based on gradient
    ftol: float = 1e-9  # Convergence criterion based on energy (cost)
    maxiter: int = 100  # Maximum iterations: if 0, skip VQE and only qubit transformation is carried out.
    

    # pauli_list
    user_defined_hamiltonian: QubitOperator = None
    user_defined_pauli_list: List = None
    disassemble_pauli: bool = False
    operator_basis: str = "fermi"
    # CASCI nroots
    nroots: int = 1

    # Quket options
    mapping: str = 'jordan_wigner'
    finite_difference: bool = False
    do_taper_off: bool = False
    oo: bool = False

    load_file: str = None

    def __post_init__(self, *args, **kwds):
        if cf.__interactive__:
            pr = False
        else:
            pr = mpi.main_rank
        if self.opt_method == "l-bfgs-b":
            self.opt_options = {"disp": pr,
                              "maxiter": self.maxiter,
                              "gtol": self.gtol,
                              "ftol": self.ftol,
                              "eps": self.eps,
                              "maxfun": self.maxfun}
        elif self.opt_method == "bfgs":
            self.opt_options = {"disp": pr,
                              "maxiter": self.maxiter,
                              "gtol": self.gtol,
                              "eps": self.eps}
        else:
            self.opt_options = None

        if self.operator_basis in ("fermionic", "fermi"):
            self.disassemble_pauli = False
        elif self.operator_basis in ("qubit", "pauli"):
            self.disassemble_pauli = True


        if self.mapping in ("jw", "jordan_wigner"):
            self.mapping = "jordan_wigner"
        elif self.mapping in ("bk", "bravyi_kitaev"):
            self.mapping = "bravyi_kitaev"

@dataclass(repr=False)
class Qulacs():
    """
    Qulacs section.

    Attributes:
        Hamiltonian (Observable): Quantum hamiltonian.
        S2 (Observable): Quansum S2.
    """
    qubit_Hamiltonian: InitVar[QubitOperator]
    qubit_S2: InitVar[QubitOperator] 
    qubit_Sz: InitVar[QubitOperator] 
    qubit_Number: InitVar[QubitOperator] 
    qubit_S4: InitVar[QubitOperator] 

    n_qubits: InitVar[QubitOperator]

    Hamiltonian: Observable = None
    S2: Observable = None
    Sz: Observable = None
    Number: Observable = None
    S4: Observable = None

    def __post_init__(self, qubit_Hamiltonian, qubit_S2, qubit_Sz, qubit_Number, qubit_S4, n_qubits, *args, **kwds):
        from quket.opelib import OpenFermionOperator2QulacsObservable
        if qubit_Hamiltonian is not None:
            self.Hamiltonian = OpenFermionOperator2QulacsObservable(qubit_Hamiltonian, n_qubits)
        if qubit_S2 is not None:
            self.S2 = OpenFermionOperator2QulacsObservable(qubit_S2, n_qubits)
        if qubit_Sz is not None:
            self.Sz = create_observable_from_openfermion_text(str(qubit_Sz))
        if qubit_Number is not None:
            self.Number = OpenFermionOperator2QulacsObservable(qubit_Number, n_qubits)
        if qubit_S4 is not None:
            self.S4 = OpenFermionOperator2QulacsObservable(qubit_S4, n_qubits)

@dataclass(repr=False)
class Projection():
    """
    Symmetry-Projection section.

    Attributes:
        SpinProj (bool): Spin projection.
        NumberProj (bool): Number projection.
        spin (int): Target spin for spin projection.
        Ms (int): Nalpha - Nbeta
        euler_ngrids (list): Grid points for spin projection.
        number_ngrids (int): Grid points for number projection.
    """

    Ms: int = None
    spin: int = None
    SpinProj: bool = False
    post_SpinProj: bool = False
    NumberProj: bool = False
    number_ngrids: int = 0
    euler_ngrids: List[int] = field(default_factory=lambda :[0, -1, 0])

    def __post_init__(self,  *args, **kwds):
        pass
    def set_projection(self, trap=True):
        from quket.projection import weightspin, trapezoidal, simpson, S2Proj
        if self.SpinProj or self.post_SpinProj:
            prints(f"Projecting to spin space : "
                   f"s = {(self.spin-1)/2:.1f}    "
                   f"Ms = {self.Ms} ")
            prints(f"             Grid points : "
                   f"(alpha, beta, gamma) = ({self.euler_ngrids[0]}, "
                                           f"{self.euler_ngrids[1]}, "
                                           f"{self.euler_ngrids[2]})")

            self.sp_angle = []
            self.sp_weight = []
            # Alpha
            if self.euler_ngrids[0] > 1:
                if trap:
                    alpha, wg_alpha = trapezoidal(0, 2*np.pi,
                                                  self.euler_ngrids[0])
                else:
                    alpha, wg_alpha = simpson(0, 2*np.pi, self.euler_ngrids[0])
            else:
                alpha = [0]
                wg_alpha = [1]
            self.sp_angle.append(alpha)
            self.sp_weight.append(wg_alpha)

            # Beta
            if self.euler_ngrids[1] > 1:
                beta, wg_beta \
                        = np.polynomial.legendre.leggauss(self.euler_ngrids[1])
                beta = np.arccos(beta)
                beta = beta.tolist()
                self.dmm = weightspin(self.euler_ngrids[1], self.spin,
                                      self.Ms, self.Ms, beta)
            elif self.euler_ngrids[1] == 1:
                beta = [np.pi/2]
                wg_beta = [1]
                self.dmm = weightspin(self.euler_ngrids[1], self.spin,
                                      self.Ms, self.Ms, beta)
            else:
                beta = [0]
                wg_beta = [1]
                self.dmm = [1]
            self.sp_angle.append(beta)
            self.sp_weight.append(wg_beta)

            # Gamma
            if self.euler_ngrids[2] > 1:
                if trap:
                    gamma, wg_gamma = trapezoidal(0, 2*np.pi,
                                                  self.euler_ngrids[2])
                else:
                    gamma, wg_gamma = simpson(0, 2*np.pi, self.euler_ngrids[2])
            else:
                gamma = [0]
                wg_gamma = [1]
            self.sp_angle.append(gamma)
            self.sp_weight.append(wg_gamma)


        if self.NumberProj:
            prints(f"Projecting to number space :  "
                   f"N = {self.number_ngrids}")

            self.np_angle = []
            self.np_weight = []

            # phi
            if self.number_ngrids > 1:
                if trap:
                    phi, wg_phi = trapezoidal(0, 2*np.pi, self.number_ngrids)
                else:
                    gamma, wg_gamma = simpson(0, 2*np.pi, self.number_ngrids)
            else:
                phi = [0]
                wg_phi = [1]
            self.np_angle = phi
            self.np_weight = wg_phi

    def get_Rg_pauli_list(self, n_orbitals, mapping="jordan_wigner"):
        ### Rotation Operators as pauli_list
        # Exp[ -i angle Sz ]
        # prod exp[i theta pauli]
        # Sz = 1/4 (-Z0 +Z1 -Z2 +Z3 ...)
        # Sy = 1/4 (Y0 X1 - X0 Y1) + (Y2 X3 - X2 Y3) + ...
        Sz = 0
        Sy = 0
        for i in range(n_orbitals):
            Sz += FermionOperator(f"{2*i}^ {2*i}",0.5)
            Sz += FermionOperator(f"{2*i+1}^ {2*i+1}",-0.5)

            Sy += FermionOperator(f"{2*i+1}^ {2*i}",0.5)
            Sy += FermionOperator(f"{2*i}^ {2*i+1}",-0.5)

        if mapping == "jordan_wigner":
            qubit_Sz = jordan_wigner(Sz)
            qubit_Sy = jordan_wigner(Sy)
        elif mapping == "bravyi_kitaev":
            qubit_Sz = bravyi_kitaev(Sz, 2*n_orbitals)
            qubit_Sy = bravyi_kitaev(Sy, 2*n_orbitals)
            
        self.Rg_pauli_list = [-qubit_Sz, -qubit_Sy, -qubit_Sz]

@dataclass(repr=False)
class Multi():
    """
    Multi/Excited-State calculation section.

    Attributes:
        act2act_opt (bool): ??
        states (list): Initial determinants (bits)
                       for multi-state calculations; JM-UCC or ic-MRUCC.
        weights (list): Weight for state-average calculations;
                        usually 1 for all.
    """
    act2act_opt: bool = False

    init_states: List = field(default_factory=list)
    init_states_info: List = field(default_factory=list)
    weights: List = field(default_factory=list)

    nstates: int = field(init=False)
    states: List[QuantumState] = field(default_factory=list)

    def __post_init__(self, *args, **kwds):
        self.nstates = len(self.weights) if self.weights else 0


@dataclass(repr=False)
class Adapt():
    """
    Adapt section.

    Atrributes:
        eps (float): Convergence criterion
        max (int): Maximum number of VQE parameters
        a_list (List): Creation operator list.
        b_list (List): Creation operator list.
        i_list (List): Annihilation operator list.
        j_list (List): Annihilation operator list.
        spin_list (List): Spin list.
        init_theta_list (List): Initial wave function's list.
    """
    eps: float = 0.1
    max: int = 1000
    mode: str = "original"
    adapt_guess: str = None
    lucc_nvqe: bool = False
    svqe: int = 1 #in default, from first steps lucc_nvqe do not do vqe

    adapt_prop: bool = False
    max_ncnot: int = 999999999999

    a_list: List = field(default_factory=list)
    b_list: List = field(default_factory=list)
    i_list: List = field(default_factory=list)
    j_list: List = field(default_factory=list)
    spin_list: List = field(default_factory=list)
    init_theta_list: List = field(default_factory=list)
    #pauli_list: List = field(default_factory=list)
    #theta_list: List = field(default_factory=list)

    bs_orbitals: List = field(default_factory=list)
    grad_list: List = field(default_factory=list)

    def __post_init__(self, *args, **kwds):
        pass


def create_parent_class(self, kwds):
    """Function
    Prepare a sub-class of QuketData.
    Either hubbard, heisenberg, or chemical is allowed.
    kwds contains input keywords, which are then screened
    to the appropriate format for the chosen dynamic class as obj.

    Args:
        self (QuketData):
        kwds (**kwargs): input dictionary
    Returns:
        obj (Class): Dynamic class

    Author(s): Yuma Shimomoto, Takashi Tsuchimochi
    """
    #######################
    # Create parent class #
    #######################
    if self.model == "hubbard":
        from .hubbard import Hubbard

        init_dict = get_func_kwds(Hubbard.__init__, kwds)
        if "n_active_electrons" not in init_dict:
            init_dict["n_active_electrons"] = init_dict["n_electrons"]
        obj = Hubbard(**init_dict)

        prints(f"Hubbard model: nx = {obj.hubbard_nx}  "
               f"ny = {obj.hubbard_ny}  "
               f"U = {obj.hubbard_u:2.2f}"
               f"Ne = {obj.n_electrons}")
    elif self.model == "heisenberg":
        from .heisenberg import Heisenberg

        init_dict = get_func_kwds(Heisenberg.__init__, kwds)
        obj = Heisenberg(**init_dict)

        if self.det is None:
            self.det = 1
        self.current_det = self.det
    elif self.model == "chemical":
        from .chemical import Chemical
        init_dict = get_func_kwds(Chemical.__init__, kwds)
        obj = Chemical(**init_dict)

    else:
        obj = None

    return obj


def set_dynamic_class(self, kwds, obj):
    """Function
    Set dynamic class as a sub-class of QuketData.
    Either hubbard, heisenberg, or chemical is allowed.
    kwds contains input keywords, which are then screened
    to the appropriate format for the chosen dynamic class.

    Author(s): Yuma Shimomoto, Takashi Tsuchimochi
    """
    #######################################
    # Inherit parent class                #
    #   MAGIC: DYNAMIC CLASS INHERITANCE. #
    #######################################
    for k, v in obj.__dict__.items():
        if k not in self.__dict__:
            self.__dict__[k] = v
    # Keep them under the control of dataclass and rename my class name.
    my_fields = []
    eri = None
    #from pympler import asizeof
    for k, v in self.__dict__.items():
        ### ERI can be large and slow down make_dataclass
        ### Current workaround is to add this after make_dataclass
        if k == 'two_body_integrals':
            eri = v
        elif k == 'two_body_integrals_active':
            eri_active = v
        elif isinstance(v, (dict, list, set)):
            my_fields.append((k, type(v), field(default_factory=v)))
        else:
            my_fields.append((k, type(v), v))
    self.__class__ = make_dataclass(f"{obj.__class__.__name__}QuketData",
                                    my_fields,
                                    bases=(QuketData, obj.__class__),
                                    repr=False)

    if eri is not None:
        self.__dict__['two_body_integrals'] = eri
    if eri is not None:
        self.__dict__['two_body_integrals_active'] = eri_active
    # Force to update n_qubits
    self.n_qubits = obj.n_qubits
    self._n_qubits = self.n_qubits  # Original number of qubits
    self.H_n_qubits = self._n_qubits
    # Force to update n_active_electrons and n_active_orbitals
    if hasattr(obj, "n_active_electrons"):
        self.n_active_electrons = obj.n_active_electrons
    if hasattr(obj, "n_active_orbitals"):
        self.n_active_orbitals = obj.n_active_orbitals
    if hasattr(obj, "n_frozen_orbitals"):
        self.n_frozen_orbitals = obj.n_frozen_orbitals
    if hasattr(obj, "n_core_orbitals"):
        self.n_core_orbitals = obj.n_core_orbitals
    if hasattr(obj, "n_secondary_orbitals"):
        self.n_secondary_orbitals = obj.n_secondary_orbitals
    

    # Add functions and properties to myself.
    for k in dir(obj):
        if k not in dir(self):
            setattr(self, k, getattr(obj, k))

def _format_picklable_QuketData(Quket):
    """
    Format QuketData to picklable data structure. May lose some (unnecessary) information.
    """
    ## Search _pyscf_data
    #for k in dir(Quket):
    #    if k == "_pyscf_data":
    #        ### _pyscf_data contains unpicklable objects, so remove.
    #        for k1 in Quket._pyscf_data:
    #            Quket._pyscf_data[k1] = remove_unpicklable(Quket._pyscf_data[k1])

    if hasattr(Quket, "pyscf"):
        try:
            del(Quket.pyscf)
        except:
            pass
    # Now remove other unpicklable objects (necessary _pyscf_data should be held)
    data = remove_unpicklable(Quket)
    # Quantum State treatment
    for k, v in Quket.__dict__.items():
        if isinstance(v, QuantumState):
        ### Quantum State ###
            state = {}
            state["_vec"] = v.get_vector()
            state["_nqubit"] = v.get_qubit_count()
            state_name = k + "_QuantumState"
            data[state_name] = state
        elif type(v) is list:
            if len(v) > 0:
                if isinstance(v[0], QuantumState):
                # List of QuantumStates
                    states = []
                    state_name = k + "_QuantumState"
                    for i, q in enumerate(v):
                        state = {}
                        state["_vec"] = q.get_vector()
                        state["_nqubit"] = q.get_qubit_count()
                        states.append(state)
                    data[state_name] = states
                elif type(v[0]) == dict:
                    if 'state' in v[0]:
                        # Dict contains 'energy', 'state', and perhaps 'theta_list', 'det'
                        # Change 'state'=QuantumState to Vector
                        states = []
                        for i, q in enumerate(v):
                            state = {}
                            for k1, v1 in q.items():
                                state_name = k + "_QuantumState"
                                if k1 == 'state':
                                    state["_vec"] = v1.get_vector()
                                    state["_nqubit"] = v1.get_qubit_count()
                                else:
                                    state[k1] = v1
                            states.append(state)
                        data[state_name] = states
        elif hasattr(v, '__dict__'):
            for k_, v_ in list(v.__dict__.items()):
                if type(v_) is list:
                    if len(v_) > 0:
                        if isinstance(v_[0], QuantumState):
                        # List of QuantumStates
                            states = []
                            state_name_ = k_ + "_QuantumState"
                            for i, q in enumerate(v.states):
                                state = {}
                                state["_vec"] = q.get_vector()
                                state["_nqubit"] = q.get_qubit_count()
                                states.append(state)
                            v.__dict__.pop(k_)
                            v.__dict__[state_name_] = states
                            data[k] = remove_unpicklable(v)
    return data
    ###############################################

def _restore_quantumstate_in_Data(data):
    """Function
    Restore QuantumState from 2^n vector in the dictionary 'data'.

    Author(s): Takashi Tsuchimochi
    """
    # Recover Quantum State #
    for k, v in list(data.items()):
        if "_QuantumState" in k:
            ### Check if this is a list (dict's) or dict
            if type(data[k]) is list:
                if type(data[k][0]) == dict:
                    # Dict contains 'energy', 'state', and perhaps 'theta_list', 'det'
                    # Change 'state'=Vector to QuantumState
                    states = []
                    state_name = k.replace('_QuantumState', '')
                    for i, q in enumerate(data[k]):
                        state = {} 
                        if 'energy' in q:
                            state['energy'] = q['energy']
                        nqubit = q["_nqubit"]
                        vec = q["_vec"]
                        state['state'] = QuantumState(nqubit)
                        state['state'].load(vec)
                        if 'theta_list' in q:
                            state['theta_list'] = q['theta_list']
                        if 'det' in q:
                            state['det'] = q['det']
                        states.append(state)
                    data.pop(k)
                    data[state_name] = states
            else:
                nqubit = data[k]["_nqubit"]
                vec = data[k]["_vec"]
                state = QuantumState(nqubit)
                state.load(vec)
                state_name = k.replace('_QuantumState', '')
                data.pop(k)
                data[state_name] = state
        elif (type(v) is dict):
            for k_, v_ in list(v.items()):
                if "_QuantumState" in k_:
                    ### Check if this is a list (dict's) or dict
                    if type(v[k_]) is list:
                        states = []
                        state_name = k_.replace('_QuantumState', '')
                        for i, q in enumerate(v[k_]):
                            nqubit = q["_nqubit"]
                            vec = q["_vec"]
                            state = QuantumState(nqubit)
                            state.load(vec)
                            states.append(state)
                        v.pop(k_)
                        v[state_name] = states
                    else:
                        nqubit = v[k_]["_nqubit"]
                        vec = v[k_]["_vec"]
                        state = QuantumState(nqubit)
                        state.load(vec)
                        state_name = k_.replace('_QuantumState', '')
                        v.pop(k_)
                        v[state_name] = state
                    data[k] = v
    return data

def copy_quket_data(Q):
     data = deepcopy(_format_picklable_QuketData(Q))
     return _restore_quantumstate_in_Data(data)

@dataclass(repr=False)
class QuketData():
    """Data class for Quket.

    Attributes:
        method (str): Computation method; 'vqe' or 'qite'.
        model (str): Computation model; 'chemical', 'hubbard' or 'heisenberg'.
        ansatz (str): VQE or QITE ansatz; 'uccsd' and so on.
        det (int): A decimal value of the determinant of the quantum state.
        run_fci (bool): Whether run FCI or not.
        rho (int): Trotter number for related ansatz.
        DS (int): If 0/1, the operation order is;
                  Exp[T1] Exp[T2] / Exp[T2] Exp[T1]
        Do1RDM (bool): Whether do 1RDM or not.
        Do2RDM (bool): Whether do 2RDM or not.
        DoQSE (bool): Whether do QSE or not.
        print_amp_thres (float): Threshold for printing VQE parameters.
        ftol (float): Convergence criterion based on energy (cost).
        gtol (float): Convergence criterion based on gradients.
        dt (flaot): Time-step for time-evolution.
        truncate (float): Truncation threshold for anti-symmetric hamiltonian.
        excited_states (list): Initial determinants
                               for excited state calculations.

    Author(s): Takashi Tsuchimochi, Yuma Shimomoto
    """
    #----------For QuketData----------
    method: str = "vqe"
    post_method: str = "none"
    post_general: bool = True
    spinfree: bool = False
    post_hermitian_conjugated: bool = True
    regularization: float = 0
    model: str = None
    ansatz: str = None
    basis: str = None
    det: int = None
    run_fci: bool = True
    run_ccsd: bool = False
    run_casscf: bool = False
    run_qubitfci: bool = False
    rho: int = 1
    DS: int = 0
    Do1RDM: bool = False
    Do2RDM: bool = False
    DoQSE: bool = False
    print_amp_thres: float = 1e-2
    ftol: float = 1e-9
    gtol: float = 1e-5
    dt: float = 0.1
    truncate: float = 0.
    maxiter: int = 100
    state: QuantumState = None
    init_state: QuantumState = None
    state_unproj: QuantumState = None
    fci_states: QuantumState = None
    energy: float = 0.
    s2: float = 0.
    geom_opt: bool = False
    do_grad: bool = False
    n_qubits: int = None
    _n_qubits: int = None
    H_n_qubits: int = None
    _ndim: int = None
    converge: bool = False
    symmetry: bool = True
    symmetry_pauli: bool = False
    qubit_excitation: bool = False
    symmetry_subgroup: str = None
    local: List = field(default_factory=list)
    # List of orbital indices to be localized.
    alter_pairs: List = field(default_factory=list)
    # List of orbital indices to be swiched their orderings.
    # (a,b),(c,d),... indicates a <-> b and c <-> d, ...
    #----VQE---
    theta_list: np.ndarray = field(init=False, default=None)
    kappa_list: np.ndarray = field(init=False, default=None)
    constraint_lambda: float = 0
    constraint_lambda_Sz: float = 0
    constraint_lambda_S2_expectation: float = 0
    lower_states = []
    #---Pauli_list---
    pauli_list: List = field(init=False, default=None)
    pauli_list_nlayers: int = 1
    #----QITE----
    shift: str = "step"
    qlanczos: bool = False
    msqite: str = "false"
    s2shift: float = 0
    folded_spectrum: bool = False
    hamiltonian_threshold: float = 0.
    #----OO----
    oo_maxiter: int = 20
    oo_ftol: float = 1e-9
    oo_gtol: float = 1e-4
    #----- For MBE -----
    #min_use_core: int = 0
    #max_use_core: int = 0
    #min_use_secondary: int = 0
    #max_use_secondary: int = 0
    #n_secondary_orbitals: int = -1
    #include: Dict = field(default_factory=dict)
    color: int = None
    later: bool = False
    #mo_basis: str = "hf"
    mbe_exact: bool = False
    mbe_correlator: str = "sauccsd"
    mbe_oo: bool = False
    mbe_oo_ansatz: str = None
    from_vir: str = "none"

    #---Tapering flags---
    tapered: Dict = field(default_factory=dict)


    excited_states: List = field(default_factory=list)

    operators: Operators = field(init=False, default=None)
    qulacs: Qulacs = field(init=False, default=None)
    projection: Projection = field(init=False, default=None)
    multi: Multi = field(init=False, default=None)
    cf: Config = field(init=False, default=None)
    tapering: Z2tapering = field(init=False, default=None)

    def __post_init__(self, *args, **kwds):
        if self.ansatz is None and self.maxiter > 0 and not cf.__interactive__:
            error(f"Unspecified ansatz.")
        elif not chkmethod(self.method, self.ansatz) and self.maxiter > 0:
            error(f"No method option {self.method} "
                  f"with {self.ansatz} available.")
        elif not chkpostmethod(self.post_method):
            error(f"No post_method option {self.post_method}  available.")
        if self.method == "mbe":
            cf.SaveTheta = False
            if self.ansatz in ("uccgsd", "sauccgsd") and self.from_vir == "none":
                self.from_vir = True
        if self.from_vir in ("none", "false"):
            self.from_vir = False
        elif self.from_vir == "true":
            self.from_vir = True


    def __setstate__(self,state):
        self.__dict__.update

    def __repr__(self):
        return str(type(self))

    def __str__(self):
        if self.model == "hubbard":
            print(f"Hubbard model: "
                  f"nx           = {self.hubbard_nx}"
                  f"ny           = {self.hubbard_ny}"
                  f"U            = {self.hubbard_u:2.2f}"
                  f"Ne           = {self.n_electrons}")
        elif self.model == "heisenberg":
            print(f"{self.basis}  {self.n_qubits} spin chain.")
        elif self.model == "chemical":
            print_geom(self.geometry, filepath='')
            print(f"Basis        = {self.basis}")
            print(f"NBasis       = {self.mo_coeff.shape[0]}")
            print(f"Ne           = {self.n_active_electrons}")
            print(f"Norbs        = {self.n_active_orbitals}")
            print(f"Multiplicity = {self.multiplicity}")
        elif self.model == "user-defined":
            print(f"User-deinfed Hamiltonian:\n{self.operators.qubit_Hamiltonian}")
        else:
            print(f"System undefined.")
        print(f"Method       = {self.method}")
        print(f"Ansatz       = {self.ansatz}")
        print(f"Mapping      = {self.cf.mapping}")
        print(f"Nqubits      = {self.n_qubits}")
        print(f"Energy       = {self.energy}")
        print(f"Converged    = {self.converge}")
        if self.cf.finite_difference:
            print(f"Derivative   = Numerical (Forward)")
        else:
            print(f"Derivative   = Analytical (Exact)")
        return ""

    @property
    def anc(self):
        return self.n_qubits

    def initialize(self, *, pyscf_guess="minao", **kwds):
        """Function
        Run PySCF and initialize parameters.

        Args:
            pyscf_guess (str): PySCF guess.
        """
        if hasattr(self, '_kwds') and kwds == {}:
            kwds = self._kwds
        elif hasattr(self, '_kwds') and kwds != {}:
            for k, v in self._kwds.items():
                if not hasattr(kwds, k):
                    kwds[k] = v

        ##################
        # Set Subclasses #
        ##################
        # Projection
        init_dict = get_func_kwds(Projection.__init__, kwds)
        self.projection = Projection(**init_dict)
        # Multi
        init_dict = get_func_kwds(Multi.__init__, kwds)
        self.multi = Multi(**init_dict)
        # Adapt
        init_dict = get_func_kwds(Adapt.__init__, kwds)
        self.adapt = Adapt(**init_dict)
        # Config
        init_dict = get_func_kwds(Config.__init__, kwds)
        self.cf = Config(**init_dict)
        # tapered (stamps for whether transfomation has been done or not)
        self.tapered["operators"] = False
        self.tapered["states"] = False
        self.tapered["pauli_list"] = False
        self.tapered["theta_list"] = False

        #############
        # Set model #
        #############
        if kwds.get("basis"):
            if kwds["basis"] == "hubbard":
                self.model = "hubbard"
            elif "heisenberg" in kwds["basis"]:
                self.model = "heisenberg"
            elif kwds.get("user_defined_hamiltonian"):
                self.model = "user-defined"
            elif kwds.get("geometry"):
                self.model = "chemical"
        elif kwds.get("user_defined_hamiltonian"):
            self.model = "user-defined"
        #else:
        #    # Default
        #    self.model = None

        #######################
        # Create parent class #
        #######################
        obj = create_parent_class(self, kwds)

        #################
        # Get Operators #
        #################
        if self.model == "hubbard":
            Hamiltonian, S2, Number = obj.get_operators(guess=pyscf_guess)
            if self.constraint_lambda > 0:
                S4 = S2*S2
            else:
                S4 = None
            Sz = FermionOperator('',0)
            for i in range(obj.n_qubits//2):
                Sz += FermionOperator(f"{2*i}^ {2*i}",0.5)
                Sz += FermionOperator(f"{2*i+1}^ {2*i+1}",-0.5)
            self.operators = Operators(self.cf.mapping, obj.n_qubits, Hamiltonian=Hamiltonian, S2=S2, Sz=Sz,
                                       Number=Number, S4=S4)
            self.operators.pgs = None
            ### HF energy (not exactly HF, we have to do orbital optimization)
            #self.operators.
            if self.run_fci and obj.hubbard_ao:
                self.operators.qubit_Hamiltonian.compress()
                qubit_eigen = QubitDavidson(self.operators.qubit_Hamiltonian,
                                            obj.n_qubits)
                # Initial guess :  | 0000...00111111>
                #                             ~~~~~~ = n_electrons
                guess = np.zeros((2**obj.n_qubits, 1))
                guess[2**obj.n_electrons - 1][0] = 1.0
                n_state = 1
                results = qubit_eigen.get_lowest_n(n_state, guess)
                prints("Convergence?           : ", results[0])
                prints("Ground State Energy    : ", results[1][0])
                obj.fci_energy = results[1][0]
                #prints("Wave function          : ")
                #openfermion_print_state(results[2], n_qubits, 0)
        elif self.model == "heisenberg":
            qubit_Hamiltonian = obj.get_operators()
            self.operators = Operators(self.cf.mapping, self.n_qubits)
            self.operators.qubit_Hamiltonian = qubit_Hamiltonian
            self.operators.pgs = None
        elif self.model == "chemical":
            # New geometry found. Run PySCF and get operators.
            prints(f"Basis set = {kwds['basis']}")
            print_geom(obj.geometry)
            Hamiltonian, S2, Number, Dipole \
                    = obj.get_operators(guess=pyscf_guess,
                                        run_fci=self.run_fci,
                                        run_ccsd=self.run_ccsd,
                                        run_casscf=self.run_casscf)
            if self.constraint_lambda > 0:
                S4 = S2*S2
            else:
                S4 = None
            Sz = FermionOperator('',0)
            for i in range(obj.n_active_orbitals):
                Sz += FermionOperator(f"{2*i}^ {2*i}",0.5)
                Sz += FermionOperator(f"{2*i+1}^ {2*i+1}",-0.5)
            self.operators = Operators(self.cf.mapping, obj.n_qubits,
                                       Hamiltonian=Hamiltonian, S2=S2, Sz=Sz,
                                       Number=Number, Dipole=Dipole, S4=S4) 
            from quket.utils import prepare_pyscf_molecule_mod
            self.pyscf = prepare_pyscf_molecule_mod(obj)

            #TODO: PySCF does not support symmetry for open-shell? Need to check.
            if self.symmetry and obj.symm_operations != []:
                nfrozen = obj.n_frozen_orbitals * 2
                ncore = obj.n_core_orbitals * 2
                nact = obj.n_active_orbitals * 2
                pgs_head, pgs_tail = nfrozen, nfrozen+ncore+nact
                self.operators.pgs = (obj.symm_operations,
                      obj.irrep_list[pgs_head:pgs_tail],
                      [x[pgs_head:pgs_tail] for x in obj.character_list])
            else:
                self.operators.pgs = None
            #self.obj = obj

        elif self.model == None:
            ### Empty QuketData
            if self.operators is None:
                self.operators = Operators(self.cf.mapping, self.n_qubits, Hamiltonian=None, S2=None, Number=None, Dipole=None)
                self.operators.pgs = None
                return
            ### Check if hamiltonian and other things are given...
            if self.operators.Hamiltonian is not None:
                from openfermion import is_hermitian
                if not is_hermitian(self.operators.Hamiltonian):
                    prints("\n##########################################\n")
                    prints("  WARNING: Hamiltonian is not Hermitian   ")
                    prints("\n##########################################\n\n")
                self.operators = Operators(self.cf.mapping, self.n_qubits, Hamiltonian=self.operators.Hamiltonian, S2=self.operators.S2, Number=self.operators.Number, Dipole=self.operators.Dipole)
                self.model = "user-defined"
                ### Check if qubit Hamiltonian already exists
                if self.operators.qubit_Hamiltonian is None:
                    pass                    
            elif self.operators.qubit_Hamiltonian is not None:
                self.model = "user-defined"
            else:
                prints(f"Need to define Hamiltonian.\n"
                       f"  Q.operators.qubit_Hamiltonian = QubitOperator('0.5 [X0 Y1] + 0.2 [Z1 Z4] + ...')\n"
                       f"or\n"
                       f"  Q.operators.Hamiltonian = FermionOperator('0.5 [0^ 1] + 0.2 [1^ 4] + ...')")
                if not hasattr(self,'geometry'):
                    prints("You can also specify geometry along with basis-set to generate chemical Hamiltonian\n"
                           f"   Q.set(geometry = 'H 0 0 0; H 0 0 1',  basis = 'sto-6g',  ... )")
                return

        ##################################
        ###  User-defined Hamiltonian  ###
        ##################################
        if self.model == "user-defined":
            ### Check if user_defined_hamiltonian is available
            self.operators = Operators(self.cf.mapping, self.n_qubits)
            if kwds.get('user_defined_hamiltonian'):
                try:
                    self.operators = Operators(self.cf.mapping, self.n_qubits, Hamiltonian=FermionOperator(kwds['user_defined_hamiltonian']))
                except:
                    try:
                        self.operators.qubit_Hamiltonian = QubitOperator(kwds['user_defined_hamiltonian'])
                    except:
                        error(f"User-defined Hamiltonian is not in the form of FermionOperator or QubitOperator.\n {type(kwds['user_defined_hamiltonian'])}\n {kwds['user_defined_hamiltonian']}")
                    
            if self.operators.Hamiltonian is not None:
                from openfermion import is_hermitian
                if not is_hermitian(self.operators.Hamiltonian):
                    prints("\n##########################################")
                    prints("  WARNING: Hamiltonian is not Hermitian   ")
                    prints("\n##########################################\n")
                self.operators = Operators(self.cf.mapping, self.n_qubits, Hamiltonian=self.operators.Hamiltonian, S2=self.operators.S2, Number=self.operators.Number, Dipole=self.operators.Dipole)
            elif self.operators.qubit_Hamiltonian is None:
                prints(f"Need to define Hamiltonian.\n"
                       f"  Q.operators.qubit_Hamiltonian = QubitOperator('0.5 [X0 Y1] + 0.2 [Z1 Z4] + ...')\n"
                       f"or\n"
                       f"  Q.operators.Hamiltonian = FermionOperator('0.5 [0^ 1] + 0.2 [1^ 4] + ...')")
                if not hasattr(self,'geometry'):
                    prints("You can also specify geometry along with basis-set to generate chemical Hamiltonian\n"
                           f"   Q.set(geometry = 'H 0 0 0; H 0 0 1',  basis = 'sto-6g',  ... )")
                return
            if self.operators.Hamiltonian is None:
                # Assuming jordan-wigner mapping
                if self.cf.mapping != "jordan_wigner":
                    prints("\n##########################################")
                    prints("  WARNING: Fermionic Hamiltonian is generated based on Jordan-Wigner mapping   ")
                    prints("\n##########################################\n")
                from quket.lib import reverse_jordan_wigner
                self.operators.Hamiltonian = reverse_jordan_wigner(self.operators.qubit_Hamiltonian)
            self.operators.pgs = None
            ### Get n_qubits
            # Largest value in qubit_Hamiltonian.
            self.H_n_qubits = 1
            qubit_H = list(self.operators.qubit_Hamiltonian.terms.keys())
            for h in qubit_H:
                if h != ():
                    self.H_n_qubits = max(self.H_n_qubits, h[-1][0]+1)

            ### Get init_state from det.
            if self.det is not None:
                if type(self.det) is int:
                    if self.det > 0:
                        state_n_qubits = int(np.ceil(np.log2(self.det)))
                    else:
                        state_n_qubits = 0
                    self.n_qubits = max(self.H_n_qubits, state_n_qubits)
                    self.init_state = QuantumState(self.n_qubits)
                    self.init_state.set_computational_basis(self.det)
                    if self.cf.mapping == "bravyi_kitaev":
                        self.init_state = transform_state_jw2bk(self.init_state)
                    self.current_det = self.det
                    det = self.det
                elif type(self.det) is list:
                    state_n_qubits = 1
                    for det_ in self.det:
                        state_n_qubits = max(state_n_qubits, int(np.ceil(np.log2(max(det_[1],1)))))
                    self.n_qubits = max(self.H_n_qubits, state_n_qubits)
                    self.state = set_multi_det_state(self.det, self.n_qubits)
                    if self.cf.mapping == "bravyi_kitaev":
                        self.state = transform_state_jw2bk(self.state)
                    self.init_state = self.state.copy()
                    self.current_det = self.det.copy()
                    det = self.det[0][1]
                elif self.det == "random":
                    self.n_qubits = self.H_n_qubits
                    self.init_state = QuantumState(self.n_qubits)
                    self.init_state.set_Haar_random_state()
                    ### Broadcast
                    self.init_state = mpi.bcast(self.init_state)
                    vec = self.init_state.get_vector()
                    det = []
                    for i in range(len(vec)):
                        det.append([vec[i], i])
                    # overwrite
                    self.det = det.copy()
                    self.current_det = self.det.copy()
                    det = self.det[0][1]
                    
            if self.init_state is None:
                prints("Initial state is not supplied, so we use |0>.\n"
                       "You may specify the initial state by one of the following ways:\n"
                       "\n"
                       "  Q.set(det='|0011>')\n"
                       "for product state,\n"
                       "  Q.set(det='0.5 * |0011> + 0.5 * |0101>')\n"
                       "for entangled state, and\n"
                       "  Q.init_state = QuantumState(Q.n_qubits)\n"
                       "  ...\n"
                       "to directly subsititute user-defined QuantumState to init_state.\n")
                self.n_qubits = self.H_n_qubits
                self.init_state = QuantumState(self.H_n_qubits)
                self.det = 0
                self.current_det = 0
                det = 0
            elif self.init_state.get_qubit_count() > self.H_n_qubits:
                self.n_qubits = self.init_state.get_qubit_count() 
                #prints("Inconsistent qubit number:"
                #      f"init_state = {self.init_state.get_qubit_count()},  n_qubits = {self.n_qubits}")
                #return
            self._n_qubits = self.n_qubits
            # Create subclass Quket.tapering
            self.tapering = Z2tapering(self.operators.qubit_Hamiltonian,
                                       self.n_qubits,
                                       det,
                                       self.operators.pgs,
                                       not self.projection.SpinProj)
            self.lower_states = []
            self.nexcited = 0
            self.DA = self.DB = self.RelDA = self.RelDB = self.Daaaa = self.Dbbbb = self.Dbaab = None
            self.openfermion_to_qulacs()
            ### Tweaking orbitals...
            if self.alter_pairs != []:
                ## Switch orbitals
                self.alter(self.alter_pairs)
            if self.local != []:
                ## Localize orbitals
                self.boys(*self.local)
            self.set_projection()
            prints("Simulation detail\n"
                  f"Hamiltonian:\n"
                  f"{self.operators.qubit_Hamiltonian}\n"
                  f"Initial state:")
            print_state(self.init_state)

            self.state = self.init_state.copy()
            if self.ansatz is not None or self.cf.user_defined_pauli_list is not None:
                self.get_pauli_list()
            else:
                prints("Need to define pauli_list:\n"
                       "  Q.get_pauli_list(['X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1', 'X0 X1', 'X0 Y1', 'X0 Z1', 'Y0 X1', 'Y0 Y1', 'Y0 Z1', 'Z0 X1', 'Z0 Y1', 'Z0 Z1'])")
            if self.pauli_list is not None:
                self.theta_list = np.zeros(len(self.pauli_list), float)
                if self.cf.theta_guess == "read":
                    from quket.fileio import LoadTheta
                    self.theta_list = LoadTheta(self._ndim, cf.theta_list_file, offset=0)
                elif self.cf.theta_guess == "random":
                    self.theta_list = (0.5-np.random.rand(self._ndim))*0.001

                ### Broadcast
                self.theta_list = mpi.bcast(self.theta_list, root=0)

            init_dict = get_func_kwds(QuketData.__init__, kwds)
            self._init_dict = init_dict
            self._kwds = kwds

            ### Tapering
            if self.cf.do_taper_off or self.symmetry_pauli:
                self.tapering.run(mapping=self.cf.mapping)
                if self.cf.do_taper_off and self.method != 'mbe':
                    self.transform_all(reduce=True)
                elif self.get_allowed_pauli_list:
                    self.get_allowed_pauli_list()
            return
                
        #######################################
        # Inherit parent class                #
        #   MAGIC: DYNAMIC CLASS INHERITANCE. #
        #######################################
        set_dynamic_class(self, kwds, obj)
#
        # Check multiplicity.
        if self.model != 'heisenberg' and (self.n_active_electrons+self.multiplicity-1)%2 != 0:
            prints(f"Incorrect specification for "
                   f"n_active_electrons = {self.n_active_electrons} "
                   f"and multiplicity = {self.multiplicity}")

        # Check initial determinant
        if self.det is None:
            if self.model == 'heisenberg':
                # Default initial state is 0 (all spin up).
                self.det = 0
                det = self.det
            else:
                # Default initial determinant is RHF or ROHF
                self.det = set_initial_det(self.noa, self.nob)
                det = self.det
        elif self.model in ('chemical', 'hubbard'):
            if type(self.det) is int:
            # Determinant supplied. Does this have the correct N and Sz symmetries?
                if not chkdet(self.det, self.noa, self.nob):
                    opt = f"0{self.n_qubits}b"
                    print(f'\n\nWARNING: Discrepancy between initial determinant {format(self.det, opt)} \n'
                          f'         and alpha and beta electrons NA = {self.noa}, NB = {self.nob}.\n\n'
                          f'         I hope you know what you are doing.\n\n')
                det = self.det
            elif type(self.det) is list:
                for state in self.det:
                    if not chkdet(state[1], self.noa, self.nob):
                        opt = f"0{self.n_qubits}b"
                        print(f'\n\nWARNING: Discrepancy between initial determinant {format(state[1], opt)} \n'
                              f'         and alpha and beta electrons NA = {self.noa}, NB = {self.nob}.\n\n'
                              f'         I hope you know what you are doing.\n\n')
                det = self.det[0][1]
            elif self.det == 'random':
                det = set_initial_det(self.noa, self.nob) ## Just to avoid error

        self.current_det = self.det

        # Create subclass Quket.tapering
        self.tapering = Z2tapering(self.operators.qubit_Hamiltonian,
                                   self.n_qubits,
                                   det,
                                   self.operators.pgs,
                                   not self.projection.SpinProj)
        self.openfermion_to_qulacs()
        ### Tweaking orbitals...
        if self.alter_pairs != []:
            ## Switch orbitals
            self.alter(self.alter_pairs)
        if self.local != []:
            ## Localize orbitals
            self.boys(*self.local)
        self.set_projection()


        ###########################
        # Initialize QuantumState #
        ###########################
        if self.n_qubits > 24:
            prints(f"A rather large n_qubits {n_qubits} is detected.")
            prints(f"This simulation may not be feasible without tapering qubits off.")
            prints(f"Skipping preparing quantum states in initialize(),")
            prints(f"and initial quantum states with tapered-off representation will be handled later.")
        else:
            self.state = QuantumState(self.n_qubits)
            if type(self.det) is int:
                self.state.set_computational_basis(self.det)
                if self.ansatz in ("phf", "suhf", "sghf", "puccsd", "puccd",
                                   "opt_puccd", "opt_psauccd"):
                    self.init_state = QuantumState(self.n_qubits+1)
                else:
                    self.init_state = QuantumState(self.n_qubits)
                self.init_state.set_computational_basis(self.current_det)
                if self.cf.mapping == "bravyi_kitaev":
                    self.init_state = transform_state_jw2bk(self.init_state)
                    self.state = transform_state_jw2bk(self.state)
            elif type(self.det) is list:
                self.init_state = set_multi_det_state(self.det, self.n_qubits)
                if self.cf.mapping == "bravyi_kitaev":
                    self.init_state = transform_state_jw2bk(self.init_state)
                self.state = self.init_state.copy()
            elif self.det == "random":
                self.init_state = QuantumState(self.n_qubits)
                self.init_state.set_Haar_random_state()
                ### Broadcast
                self.init_state = mpi.bcast(self.init_state, root=0)
                vec = self.init_state.get_vector()
                det = []
                for i in range(len(vec)):
                    det.append([vec[i], i])
                # overwrite
                self.det = det.copy()
                self.current_det = self.det.copy()
            if len(self.multi.init_states_info) != 0:
                ### Set initial quantum states for multi
                for istate in range(len(self.multi.init_states_info)):
                    if type(self.multi.init_states_info[istate]) is list:
                        state = set_multi_det_state(self.multi.init_states_info[istate], self.n_qubits)
                    else:
                        state = QuantumState(self.n_qubits)
                        state.set_computational_basis(self.multi.init_states_info[istate])
                        if self.cf.mapping == "bravyi_kitaev":
                            state = transform_state_jw2bk(state)
                        
                    self.multi.states.append(state.copy())
                    self.multi.init_states.append(state.copy())

        # Excited states (orthogonally-constraint)
        self.nexcited = len(self.excited_states) if self.excited_states else 0
        self.lower_states = []

        ###################
        # Initialize RDMs #
        ###################
        self.DA = self.DB = self.RelDA = self.RelDB = self.Daaaa = self.Dbbbb = self.Dbaab = None
        #########################
        # Initialize fci_states #
        #########################
        self.fci_states = None

        if self.ansatz is not None or self.cf.user_defined_pauli_list is not None:
            self.get_pauli_list()
        if self.pauli_list is not None:
            self.theta_list = np.zeros(len(self.pauli_list))
            if self.cf.theta_guess == "read":
                from quket.fileio import LoadTheta
                self.theta_list = LoadTheta(self._ndim, cf.theta_list_file, offset=0)
            elif self.cf.theta_guess == "random":
                self.theta_list = (0.5-np.random.rand(self._ndim))*0.001
            ### Broadcast
            self.theta_list = mpi.bcast(self.theta_list, root=0)
        init_dict = get_func_kwds(QuketData.__init__, kwds)
        self._init_dict = init_dict
        self._kwds = kwds
        self._n_qubits = self.n_qubits

        ### Tapering
        if self.cf.do_taper_off or self.symmetry_pauli:
            self.tapering.run(mapping=self.cf.mapping)
            if self.cf.do_taper_off and self.method != 'mbe':
                ### Create excitation-pauli list, and transform relevant stuff by unitary
                self.transform_all(reduce=True)
            elif self.get_allowed_pauli_list:
                self.get_allowed_pauli_list()

    def openfermion_to_qulacs(self):
        self.qulacs = Qulacs(qubit_Hamiltonian=self.operators.qubit_Hamiltonian,
                             qubit_S2=self.operators.qubit_S2,
                             qubit_Sz=self.operators.qubit_Sz,
                             qubit_Number=self.operators.qubit_Number,
                             qubit_S4=self.operators.qubit_S4,
                             n_qubits=self.H_n_qubits)

    def set_projection(self, euler_ngrids=None, number_ngrids=None, trap=True):
        """Function
        Set the angles and weights for integration of
        spin-projection and number-projection.

        Spin-rotation operator;
                Exp[ -i alpha Sz ]   Exp[ -i beta Sy ]   Exp[ -i gamma Sz ]
        weight
                Exp[ i m' alpha]      d*[j,m',m]         Exp[ i m gamma]

        Angles alpha and gamma are determined by Trapezoidal quadrature,
        and beta is determined by Gauss-Legendre quadrature.

        Number-rotation operator
                Exp[ -i phi N ]
        weight
                Exp[  i phi Ne ]

        phi are determined by Trapezoidal quadrature.
        """
        if self.model in ["heisenberg", None]:
            # No spin or number symmetry in the model
            return

        if euler_ngrids is not None:
            self.projection.euler_ngrids = euler_ngrids
        if number_ngrids is not None:
            self.projection.number_ngrids = number_ngrids

        # Check spin, multiplicity, and Ms
        if not hasattr(self, 'multiplicity'):
            return
        if self.ansatz in ("phf", "suhf", "sghf", "opt_puccsd", "opt_puccd"):
            self.projection.SpinProj = True
        if self.projection.spin is None:
            self.projection.spin = self.multiplicity  # Default
        if self.projection.Ms is None:
            self.projection.Ms = self.multiplicity - 1
        if (self.projection.spin-self.multiplicity)%2 != 0 \
                or self.projection.spin < self.projection.Ms+1:
            prints(f"Spin = {self.projection.spin}    "
                   f"Ms = {self.projection.Ms}")
            error("Spin and Ms not cosistent.")

        self.projection.set_projection(trap=trap)
        self.projection.get_Rg_pauli_list(self.n_active_orbitals)

    def fci2qubit(self, nroots=None, threshold=1e-8, verbose=False):
        """Function
        Get FCI wave function in qubits.
        """
        backtransformed = False
        if hasattr(self, "fci_coeff"):
            if self.n_qubits != self._n_qubits or any(self.tapered.values()):
                ### Tapering-off already performed, which is currently not compatible with fci2qubit.
                ### Backtransform 
                self.taper_off(backtransform=True)
                backtransformed = True
        self.fci_states = fci2qubit(self, nroots=nroots, threshold=threshold, verbose=verbose)
        
        if backtransformed:
            ### Retrieve things before backtransformation 
            self.taper_off()
        if self.fci_states is None:
            return

        prints("FCI in Qubits",end='')
        if self.n_qubits != self._n_qubits or any(self.tapered.values()):
            prints(" (tapered-off mapping)")
        else:
            prints("")
        for istate in range(len(self.fci_states)):
            #tmp = self.get_E(self.fci_states[istate])
            print_state(self.fci_states[istate]['state'], name=f"(FCI state : E = {self.fci_states[istate]['energy']})")


    ### Defining other useful functions ###
    def fidelity(self, state=None, istate=0):
        """Function
        Compute the fidelity of 'state'
        """
        if self.fci_states is None:
            return 0
        else:
            if state is None:
                state = self.state
            if self.projection.SpinProj:
                Pstate = S2Proj(self, self.state_unproj, normalize=False)
                norm = inner_product(self.state_unproj, Pstate).real
                return abs(inner_product(self.fci_states[istate]['state'], Pstate))**2/norm
            else:
                return abs(inner_product(self.fci_states[istate]['state'], state))**2


    def print(self):
        if mpi.main_rank:
            formatstr = f"0{self.n_qubits}b"
            max_len = max(map(len, list(self.__dict__.keys())))
            for k, v in self.__dict__.items():
                if callable(v):
                    continue
                if k == "det":
                    print(f"{k.ljust(max_len)} : {format(v, formatstr)}")
                else:
                    print(f"{k.ljust(max_len)} : {v}")

    def taper_off(self, backtransform=False, reduce=True):
        if not self.tapering.initialized:
            self.tapering.run(mapping=self.cf.mapping)
            if backtransform:
                return
            self.transform_all(reduce=reduce)
        else:
            self.transform_all(backtransform=backtransform, reduce=reduce)

    def prop(self):
        from quket.post import prop
        if self.model in ("chemical", "hubbard"):
            prop(self)

    def get_1RDM(self, state=None, relax=True):
        from quket.post import get_1RDM, get_relax_delta_full
        ### For the current implementation, tapered qubits need to be retrieved.
        if state is not None and self.tapering.redundant_bits is not None:
            # Check if state has the correct number of qubits
            state_n_qubits = sate.get_qubit_count()
            if state_n_qubits() != self._n_qubits and \
               state_n_qubits() + len(self.tapering.redundant_bits) != self._n_qubits:
                raise ValueError('Confused: in get_1RDM, qubits of state = {state.get_qubit_count()} but n_active_orbitals = {self.n_active_orbitals}')
        else:
            if self.n_qubits != self._n_qubits:
                prints(f'Warning: tapered-off QuantumState is plugged in in get_1RDM,\n'
                       f'         which does assume untapered-off QuantumState.\n'
                       f'         Run'
                       f'              taper_off(backtransform=True)'
                       f'         or'
                       f'              transform_states(backtransform=True)'
                       f'         to backtransform.\n')
                return

        self.DA, self.DB = get_1RDM(self, state=state)

        if relax and self.model == "chemical":
            if self.method == 'vqe':
                DeltaA, DeltaB = get_relax_delta_full(self, print_level=0)
                self.RelDA = self.DA + DeltaA
                self.RelDB = self.DB + DeltaB
            else:
                prints(f'No relaxed density available for {self.method}')

    def get_2RDM(self, state=None):
        from quket.post import get_2RDM
        ### For the current implementation, tapered qubits need to be retrieved.
        if state is not None and self.tapering.redundant_bits is not None:
            # Check if state has the correct number of qubits
            state_n_qubits = sate.get_qubit_count()
            if state_n_qubits() != self._n_qubits and \
               state_n_qubits() + len(self.tapering.redundant_bits) != self._n_qubits:
                raise ValueError('Confused: in get_2RDM, qubits of state = {state.get_qubit_count()} but n_active_orbitals = {self.n_active_orbitals}')
        else:
            if self.n_qubits != self._n_qubits:
                prints(f'Warning: tapered-off QuantumState is plugged in in get_2RDM,\n'
                       f'         which does assume untapered-off QuantumState.\n'
                       f'         Run'
                       f'              taper_off(backtransform=True)'
                       f'         or'
                       f'              transform_states(backtransform=True)'
                       f'         to backtransform.\n')
                return

        self.Daaaa, self.Dbaab, self.Dbbbb = get_2RDM(self, state=state)

    def get_E(self, state=None, parallel=True):
        if state is None:
            state = self.state
        return self.get_expectation_value(self.qulacs.Hamiltonian, state, parallel=parallel)

    def get_S2(self, state=None, parallel=True):
        if state is None:
            state = self.state
        return self.get_expectation_value(self.qulacs.S2, state, parallel=parallel)

    def get_Sz(self, state=None, parallel=True):
        if state is None:
            state = self.state
        return self.get_expectation_value(self.qulacs.Sz, state, parallel=parallel)

    def get_N(self, state=None, parallel=True):
        if state is None:
            state = self.state
        return self.get_expectation_value(self.qulacs.Number, state, parallel=parallel)

    def get_Heff(self, state_i, state_j, parallel=True):
        return self.get_transition_amplitude(self.qulacs.Hamiltonian, state_i, state_j, parallel=parallel)

    def get_expectation_value(self, obs, state, parallel=True):
        """Function
        Get expectation value of obs wrt state.
        obs is decomposed to each process and measured.
        The measured values are collected by MPI.allreduce.
        So, this is simply an MPI implementation of Qulacs Observable's
        'get_expectation_value()' method.
        If parallel == False, then this is just serial get_expectation_value.

        Args:
            obs (qulacs.Observable): Has to be Hermitian
            state (qulacs.QuantumState):
        Return:
            :float: sampled expectation value of the observable

        Author(s): Takashi Tsuchimochi
        """

        # Check if operators and states are in the same representation (symmetry-reduced or standard mapping)
        if self.tapered["operators"] !=  self.tapered["states"]:
            prints(f'Warning!\n',
                  f'Operators are tapered-off [{self.tapered["operators"]}]\n'
                  f'   States are tapered-off [{self.tapered["states"]}]')
            prints('The result below may be nonsense.')

        n_qubits = obs.get_qubit_count()
        n_qubits_ = state.get_qubit_count()
        if n_qubits != n_qubits_:
            error(f'Mismatch of n_qubits between ops ({n_qubits}) and state ({n_qubits_})')
        if not parallel:
            return obs.get_expectation_value(state)
        n_term = obs.get_term_count()
        my_observable = Observable(n_qubits)
        ipos, my_ndim = mpi.myrange(n_term)
        for i in range(ipos, ipos+my_ndim):
            pauli_term = obs.get_term(i)
            coef = pauli_term.get_coef()
            my_observable.add_operator(pauli_term)

        my_val = my_observable.get_expectation_value(state)
        val = mpi.allreduce(my_val, op=mpi.MPI.SUM)
        return val

    def get_transition_amplitude(self, obs, state_i, state_j, parallel=True):
        """Function
        Get matrix element of obs wrt state_i and state_j.
        obs is decomposed to each process and measured.
        The measured values are collected by MPI.allreduce.
        So, this is simply an MPI implementation of Qulacs Observable's
        'get_transition_amplitude()' method.
        If parallel == False, then this is just serial get_expectation_value.

        Args:
            obs (qulacs.Observable): Has to be Hermitian
            state (qulacs.QuantumState):
        Return:
            :float: sampled expectation value of the observable

        Author(s): Takashi Tsuchimochi
        """

        # Check if operators and states are in the same representation (symmetry-reduced or standard mapping)
        if self.tapered["operators"] !=  self.tapered["states"]:
            prints(f'Warning!\n',
                  f'Operators are tapered-off [{self.tapered["operators"]}]\n'
                  f'   States are tapered-off [{self.tapered["states"]}]')
            prints('The result below may be nonsense.')

        n_qubits = obs.get_qubit_count()
        n_qubits_i = state_i.get_qubit_count()
        n_qubits_j = state_i.get_qubit_count()
        if n_qubits != n_qubits_i != n_qubits_j:
            error(f'Mismatch of n_qubits between ops ({n_qubits}) and state_i ({n_qubits_i}) and state_j ({n_qubits_j})')
        if not parallel:
            return obs.get_transition_amplitude(state_i, state_j)
        n_term = obs.get_term_count()
        my_observable = Observable(n_qubits)
        ipos, my_ndim = mpi.myrange(n_term)
        for i in range(ipos, ipos+my_ndim):
            pauli_term = obs.get_term(i)
            coef = pauli_term.get_coef()
            my_observable.add_operator(pauli_term)

        my_val = my_observable.get_transition_amplitude(state_i, state_j)
        val = mpi.allreduce(my_val, op=mpi.MPI.SUM)
        return val

    def print_state(self, state=None, n_qubits=None, filepath=None,
                threshold=0.01, name=None, digit=4, mapping=None):
        if state is None:
            state = self.state
        if mapping is None:
            mapping = self.cf.mapping
            state_ = state
        elif mapping in ("jw", "jordan_wigner"):
            if self.cf.mapping == "bravyi_kitaev":
                state_ = transform_state_bk2jw(state)
            else:
                state_ = state
        elif mapping in ("bk", "bravyi_kitaev"):
            if self.cf.mapping == "jordan_wigner":
                state_ = transform_state_jw2bk(state)
            else:
                state_ = state
        print_state(state_, n_qubits=self.n_qubits, filepath=filepath,
                threshold=threshold, name=name, digit=digit)

    def print_mo_energy(self):
        Norbs = self.mo_coeff.shape[1]
        Nalpha = (self.n_electrons + self.multiplicity-1)//2
        Nbeta = self.n_electrons - Nalpha
        prints('\n[Molecular Orbitals]')
        prints('---+-------+------------+------------------')
        prints(' # |  Sym  |   energy   |     category')
        prints('---+-------+------------+------------------')
        for i in range(Norbs):
            prints(f'{i:2d} | {self.irrep_list[2*i].center(5)} | {self.mo_energy[i]:10.4f} | ', end="")
            if i < self.nf:
                prints("Frozen Core")
            elif i < self.nf + self.nc:
                prints("Core")
            elif i < self.nf + self.nc + self.na:
                if i < Nbeta:
                    prints("Active (occupied)")
                elif i < Nalpha:
                    prints("Active (singly-occupied)")
                else:
                    prints("Active (virtual)")
            else:
                prints("Secondary")
        prints('---+-------+------------+------------------')
        prints('')

    ###########################
    # Excitation list related #
    ###########################
    def get_excite_dict(self):
        """Function
        Call get_excite_dict.
        """
        from quket.opelib import get_excite_dict, get_excite_dict_sf
        self.excite_dict = get_excite_dict(self)

    def get_pauli_list(self, qubit_operators=None):
        """Function
        Get pauli list created by fermionic excitations for VQE.

        Set ansatz as
          Q.set(ansatz='uccsd')
        or directly as
          Q.ansatz = 'uccsd'
        Pauli strings can be specified (this has higher priority) as
          Q.get_pauli_list(['X0 Y1', 'Y2 Z3', ...])

        Author(s): Kazuki Sasasako, Takashi Tsuchimochi
        """
        from quket.pauli import (get_pauli_list_uccsd_sf, get_pauli_list_uccsd, get_pauli_list_uccgsd, get_pauli_list_uccgsdt, get_pauli_list_uccgsdtq,
                            get_pauli_list_uccgsd_sf, get_pauli_list_fermionic_hamiltonian, get_pauli_list_hamiltonian, create_pauli_list_gs)

        if qubit_operators is not None:
            max_qubit = 0
            if self.n_qubits is None:
                error("You performed get_pauli_list() without specifying system Hamiltonian.\n First run initialize() to determine the number of qubits required.")
            ### qubit_operators = ['X0 Y1', 'Z0 Y3 Z5', ...]
            self.pauli_list = []
            for pauli_ in qubit_operators:
                if type(pauli_) == list:
                    pauli_list_ = []
                    for pauli__ in pauli_:
                        if type(pauli__) == str:
                            pauli__ = QubitOperator(pauli__)
                        #elif type(pauli__) == QubitOperator:
                        elif isinstance(pauli__, QubitOperator):
                            pass
                        if self.n_qubits <= max([list(pauli__.terms.keys())[0][i][0] for i in range(len(list(pauli__.terms.keys())[0]))]):
                            error("Inconsistent pauli string in terms of qubit number."
                                  "Current Hamiltonian or initial state contains at most {self.n_qubits} qubits"
                                  "but we have detected {pauli__} for pauli.")
                        pauli_list_.append(pauli__)
                    self.pauli_list.append(pauli_list_)    
                else:
                    if type(pauli_) == str:
                        pauli_ = QubitOperator(pauli_)
                    #elif type(pauli_) == QubitOperator:
                    elif isinstance(pauli_, QubitOperator):
                        pass
                    if self.n_qubits <= max([list(pauli_.terms.keys())[0][i][0] for i in range(len(list(pauli_.terms.keys())[0]))]):
                        error("Inconsistent pauli string in terms of qubit number."
                              "Current Hamiltonian or initial state contains at most {self.n_qubits} qubits"
                              "but we have detected {pauli_} for pauli.")
                    self.pauli_list.append(pauli_)
            self.ansatz = 'user-defined'
        elif self.cf.user_defined_pauli_list is not None:
            prints("user-defined pauli_list found.")
            self.get_pauli_list(self.cf.user_defined_pauli_list)
        elif self.ansatz is None:
            return
        elif self.ansatz in ("anti-hamiltonian", "ahva"):
            self.pauli_list = get_pauli_list_fermionic_hamiltonian(self, anti=True, threshold=self.hamiltonian_threshold)
        elif self.ansatz in ("hamiltonian", "hva"):
            self.pauli_list = get_pauli_list_hamiltonian(self, threshold=self.hamiltonian_threshold)
        else:
            if self.model == "user-defined" and self.ansatz != "user-defined": 
                error("Chemical ansatz only available for molecular/Hubbard systems.")
            elif self.ansatz == "sauccsd":
                self.pauli_list, self.ncnot_list = get_pauli_list_uccsd_sf(self, self.noa, self.nva, singles=True)
            elif self.ansatz == "sauccd":
                self.pauli_list, self.ncnot_list = get_pauli_list_uccsd_sf(self, self.noa, self.nva, singles=False)
            elif self.ansatz == "uccsd":
                self.pauli_list, self.ncnot_list = get_pauli_list_uccsd(self, singles=True)
            elif self.ansatz == "uccd":
                self.pauli_list, self.ncnot_list = get_pauli_list_uccsd(self, singles=False)
            elif self.ansatz == "uccgsd":
                self.pauli_list, self.ncnot_list = get_pauli_list_uccgsd(self.n_active_orbitals, singles=True, disassemble_pauli=self.cf.disassemble_pauli, mapping=self.cf.mapping)
            elif self.ansatz == "uccgd":
                self.pauli_list, self.ncnot_list = get_pauli_list_uccgsd(self.n_active_orbitals, singles=False, disassemble_pauli=self.cf.disassemble_pauli, mapping=self.cf.mapping)
            elif self.ansatz == "sauccgsd":
                self.pauli_list, self.ncnot_list = get_pauli_list_uccgsd_sf(self.n_active_orbitals, singles=True, mapping=self.cf.mapping)
            elif self.ansatz == "sauccgd":
                self.pauli_list, self.ncnot_list = get_pauli_list_uccgsd_sf(self.n_active_orbitals, singles=False, mapping=self.cf.mapping)
            elif self.ansatz == "uccgsdt":
                self.pauli_list = get_pauli_list_uccgsdt(self.n_active_orbitals, mapping=self.cf.mapping)
            elif self.ansatz == "uccgsdtq":
                self.pauli_list = get_pauli_list_uccgsdtq(self.n_active_orbitals, mapping=self.cf.mapping)
            elif self.ansatz == "hf":
                self.pauli_list, self.ncnot_list = create_pauli_list_gs(self.n_active_orbitals, [], ncnot_list=None, disassemble_pauli=False, mapping=self.cf.mapping)
            elif self.ansatz != "adapt" and self.ansatz is not None:
                prints(f'TODO: Pauli_list is not supported for ansatz = {self.ansatz}.')

        ### Layers?
        if self.pauli_list_nlayers > 1:
            pauli_layer = []
            for ilay in range(self.pauli_list_nlayers):
                pauli_layer.extend(self.pauli_list)
            self.pauli_list = pauli_layer

        if self.pauli_list is not None:
            if self.qubit_excitation:
                ### Remove all Z operations
                self.pauli_list, self.ncnot_list = remove_z_from_pauli(self.pauli_list)

            self._ndim = len(self.pauli_list)
        else:
            self._ndim = None

        self.tapered['pauli_list'] = False

        #if self.tapered['theta_list']:
        #    self.transform_theta_list(backtransform=True)
        if self.theta_list is not None:
            if not self.tapered['theta_list'] and len(self.pauli_list) != len(self.theta_list):
                prints(f'Existing theta_list is inconsistent :\n   size of pauli_list {len(self.pauli_list)} != size of theta_list {len(self.theta_list)}.\nOvewrite theta_list by zero.')
                self.theta_list = None
                self.tapered['theta_list'] = False

    ########################
    # Tapering-off related #
    ########################
    def get_allowed_pauli_list(self):
        """Function
        Update pauli_list to include only symmetry-allowed ones
        """
        if self.tapering.redundant_bits is None:
            prints('First perform tapering.run(). No transformation done')
            return
        if self.tapered['pauli_list']:
            prints('pauli_list already transformed and tapered-off.')
            return
        if self.pauli_list is None or self.pauli_list == []:
            prints('pauli_list found.')
            return

        from quket.pauli import get_allowed_pauli_list
        self.pauli_list, self.allowed_pauli_list = get_allowed_pauli_list(self.tapering, self.pauli_list)
        if self.ncnot_list is not None:
            if len(self.allowed_pauli_list) != len(self.ncnot_list):
                prints('len(self.allowed_pauli_list) != len(self.ncnot_list). Check.')
                return
            ncnot_list = []
            for x,y in zip(self.allowed_pauli_list, self.ncnot_list):
                if x:
                    ncnot_list.append(y)
            self.ncnot_list = ncnot_list

    def transform_pauli_list(self, backtransform=False, reduce=True):
        """Function
        Transform pauli list by unitary to symmetry-reduced mapping.

        Author(s): Kazuki Sasasako, Takashi Tsuchimochi
        """
        if self.tapering.redundant_bits is None:
            prints('First perform tapering.run(). No transformation done')
            return
        if self.pauli_list is None:
            prints('pauli_list not found. Perform get_pauli_list().')
        from quket.tapering import transform_pauli_list
        if backtransform:
            if self.tapered["pauli_list"]:
                self.get_pauli_list()
                self.tapered["pauli_list"] = False
                prints('pauli_list backtransformed.')
            else:
                prints('Current pauli_list is not tapered-off. No back-transformation done')
        else:
            if self.tapered["pauli_list"]:
                prints('Current pauli_list is already tapered-off. No transformation done')
            else:
                self.pauli_list, self.allowed_pauli_list = transform_pauli_list(self.tapering, self.pauli_list, reduce=reduce)
                self.tapered["pauli_list"] = True
                prints('pauli_list transformed.')

    def transform_state(self, state=None, backtransform=False, reduce=True):
        """Function
        Transform QuantumState from/to tapered-off representaiton (symmetry-reduced mapping).
        """
        from quket.tapering import transform_state
        if state is None:
            state = self.state
        state = transform_state(state, self.tapering.clifford_operators, \
                                     self.tapering.redundant_bits, \
                                     self.tapering.X_eigvals, \
                                     backtransform=backtransform, \
                                     reduce=reduce)

        return state

    def transform_states(self, backtransform=False, reduce=True):
        """Function
        Transform QuantumState from/to tapered-off representaiton (symmetry-reduced mapping).
        The target QuantumState instances include:
            self.state
            self.init_state
            self.fci_states[] (if exist)
        The target integers include:
            self.det
            self.multi.dets[] (if exist)
            self.excited_states[] (if exist)

        Args:
            backtransform (bool): True -> from symmetry-reduced mapping to original mapping.
                                  False -> the other way around.
            reduce (bool): Qubits are actually tapered-off or not.

        Author(s): Takashi Tsuchimochi
        """
        if self.tapering.redundant_bits is None:
            prints('First perform tapering.run(). No transformation done')
            return
        if not backtransform and self.tapered["states"]:
            prints('Current states are already tapered-off. No transformation done')
            return
        elif backtransform and not self.tapered["states"]:
            prints('Current states are not tapered-off. No back-transformation done')
            return
        elif backtransform:
            self.tapered["states"] = False
            prints('States     backtransformed.')

        else:

            self.tapered["states"] = True
            prints('States     transformed.')

        if self.state is not None:
            self.state = self.transform_state(self.state, backtransform=backtransform, reduce=reduce)
        if self.init_state is not None:
            self.init_state = self.transform_state(self.init_state, backtransform=backtransform, reduce=reduce)
        if self.fci_states is not None:
            for k in range(len(self.fci_states)):
                self.fci_states[k]['state'] = self.transform_state(self.fci_states[k]['state'], backtransform=backtransform, reduce=reduce)
        if len(self.lower_states) != 0:
            for k in range(len(self.lower_states)):
                self.lower_states[k]['state'] = self.transform_state(self.lower_states[k]['state'], backtransform=backtransform, reduce=reduce)
        if len(self.multi.states) != 0:
            for k in range(len(self.multi.states)):
                self.multi.states[k] = self.transform_state(self.multi.states[k], backtransform=backtransform, reduce=reduce)

        if len(self.multi.init_states) != 0:
            for k in range(len(self.multi.init_states)):
                self.multi.init_states[k] = self.transform_state(self.multi.init_states[k], backtransform=backtransform, reduce=reduce)

        if backtransform:
            self.n_qubits = self._n_qubits
        elif reduce:
            self.n_qubits -= len(self.tapering.redundant_bits)

    def transform_det(self, det=None, reduce=True, backtransform=False):
        from quket.tapering import transform_state
        # This will unlikely work...
        state = QuantumState(self.n_qubits)
        if det is None:
            det = self.det
        state.set_computational_basis(det)
        if self.cf.mapping == "bravyi_kitaev":
            state = transform_state_jw2bk(state)
        state = transform_state(state, self.tapering.clifford_operators, \
                                self.tapering.redundant_bits, \
                                self.tapering.X_eigvals, \
                                backtransform=backtransform, \
                                reduce=reduce)

        if reduce:
            #Check this is stil a determinant
            #Most likely it is entangled.
            v = state.get_vector()
            ind = np.argmax(abs(v))
            if abs(v[ind]**2 - 1) > 1e-6:
                prints('Warning! In transform_det, transformed_det does not seem to be a determinant.')
        return state


    def transform_operators(self, backtransform=False, reduce=True):
        """Function
        Transform Hamiltonian, S2, and Number operators to tapered-off representaiton (symmetry-reduced mapping).
        If backtransform is True, symmetry-reduced mapping to standard mapping.
        Replace the corresponding Qulacs observables (overwritten).

        Added 1/25: Transform rotation operators of spin-projection Rg.

        Author(s): TsangSiuChung, Takashi Tsuchimochi
        """
        from quket.tapering import transform_operator 
        if self.tapering.redundant_bits is None:
            prints('First perform tapering.run(). No transformation done')
            return
        if backtransform:
            if not self.tapered["operators"]:
                prints('Current operators are not tapered-off. No back-transformation done')
                return
            if self.cf.mapping == "jordan_wigner":
                self.operators.qubit_Hamiltonian = jordan_wigner(self.operators.Hamiltonian)
                if self.operators.S2 is not None:
                    self.operators.qubit_S2 = jordan_wigner(self.operators.S2)
                if self.operators.Number is not None:
                    self.operators.qubit_Number = jordan_wigner(self.operators.Number)
                if self.operators.Sz is not None:
                    self.operators.qubit_Sz = jordan_wigner(self.operators.Sz)
                if self.operators.S4 is not None:
                    self.operators.qubit_S4 = jordan_wigner(self.operators.S4)
                if hasattr(self.projection, 'Rg_pauli_list'):
                    self.projection.get_Rg_pauli_list(self.n_active_orbitals)
            elif self.cf.mapping == "bravyi_kitaev":
                if isinstance(self.operators.Hamiltonian, (openfermion.FermionOperator, FermionOperator)):
                    self.operators.qubit_Hamiltonian = bravyi_kitaev(self.operators.Hamiltonian, self.n_qubits)
                else:
                    from quket.lib import get_fermion_operator
                    self.operators.qubit_Hamiltonian = bravyi_kitaev(get_fermion_operator(self.operators.Hamiltonian), self.n_qubits)
                if self.operators.S2 is not None:
                    self.operators.qubit_S2 = bravyi_kitaev(self.operators.S2, self.n_qubits)
                if self.operators.Number is not None:
                    self.operators.qubit_Number = bravyi_kitaev(self.operators.Number, self.n_qubits)
                if self.operators.Sz is not None:
                    self.operators.qubit_Sz = bravyi_kitaev(self.operators.Sz, self.n_qubits)
                if self.operators.S4 is not None:
                    self.operators.qubit_S4 = bravyi_kitaev(self.operators.S4, self.n_qubits)
                if hasattr(self.projection, 'Rg_pauli_list'):
                    self.projection.get_Rg_pauli_list(self.n_active_orbitals, mapping="bravyi_kitaev")
            self.tapered["operators"] = False
            prints('Operators  backtransformed.')
        else:
            if self.tapered["operators"]:
                prints('Current operators are already tapered-off. No transformation done')
                return

            self.operators.qubit_Hamiltonian =  transform_operator(self.operators.qubit_Hamiltonian, \
                                                                    self.tapering.clifford_operators, \
                                                                    self.tapering.redundant_bits, \
                                                                    self.tapering.X_eigvals, \
                                                                    reduce=reduce)


            if self.operators.qubit_S2 is not None:
                self.operators.qubit_S2 =  transform_operator(self.operators.qubit_S2, \
                                                                self.tapering.clifford_operators, \
                                                                self.tapering.redundant_bits, \
                                                                self.tapering.X_eigvals, \
                                                                reduce=reduce)
            if self.operators.qubit_Number is not None:
                self.operators.qubit_Number =  transform_operator(self.operators.qubit_Number, \
                                                                self.tapering.clifford_operators, \
                                                                self.tapering.redundant_bits, \
                                                                self.tapering.X_eigvals, \
                                                                reduce=reduce)
            if self.operators.qubit_Sz is not None:
                self.operators.qubit_Sz =  transform_operator(self.operators.qubit_Sz,\
                                                                self.tapering.clifford_operators, \
                                                                self.tapering.redundant_bits, \
                                                                self.tapering.X_eigvals, \
                                                                reduce=reduce)
            if self.operators.qubit_S4 is not None:
                self.operators.qubit_S4 =  transform_operator(self.operators.qubit_S4,\
                                                                self.tapering.clifford_operators, \
                                                                self.tapering.redundant_bits, \
                                                                self.tapering.X_eigvals, \
                                                                reduce=reduce)
 
            if hasattr(self.projection, 'Rg_pauli_list'):
                from quket.tapering import transform_pauli_list
                self.projection.Rg_pauli_list, dummy = transform_pauli_list(self.tapering, self.projection.Rg_pauli_list, reduce=reduce)

            self.tapered["operators"] = True
            prints('Operators  transformed.')

        if backtransform:
            self.H_n_qubits = self._n_qubits
        elif reduce:
            self.H_n_qubits -= len(self.tapering.redundant_bits)
        self.openfermion_to_qulacs()

    def transform_theta_list(self, backtransform=False, reduce=True):
        """Function
        Reorder theta_list elements
        """
        if backtransform and not self.tapered["theta_list"]:
            prints('Current theta_list is not tapered-off. No back-transformation done')
            return
        elif not backtransform and self.tapered["theta_list"]:
            prints('Current theta_list is already tapered-off. No transformation done')
            return
        if self.theta_list is None:
            self.tapered["theta_list"] = not backtransform
            return
        if hasattr(self, 'theta_list') and hasattr(self, 'allowed_pauli_list'):
            _ndim = len(self.allowed_pauli_list)
            if backtransform:
                new_theta_list = np.zeros(_ndim, dtype=float)
                _i = 0
                for i in range(_ndim):
                    if self.allowed_pauli_list[i]:
                        new_theta_list[i] = self.theta_list[_i]
                        _i += 1
                self.tapered["theta_list"] = False
                prints('theta_list backtransformed.')
            else:
                ### May discard broken-symmetry non-zero elements
                new_theta_list = []
                for i in range(_ndim):
                    if self.allowed_pauli_list[i]:
                        new_theta_list.append(self.theta_list[i])
                new_theta_list = np.array(new_theta_list)
                self.tapered["theta_list"] = True
                prints('theta_list transformed.')

            self.theta_list = new_theta_list.copy()
        else:
            pass

    #def transform_theta_list(self, backtransform=True):
    #    """Function
    #    Reorder theta_list from/to symmetry-reduced mapping/standard mapping.
    #
    #    Author(s): Takashi Tsuchimochi
    #    """
    #    if not hasattr(self, 'allowed_pauli_list'):
    #        prints('First perform transform_pauli_list. No transformation done')
    #        return
    #    if not hasattr(self, 'theta_list'):
    #        prints('No theta_list found. First perform VQE by run(). No transformation done')
    #        return
    #    prints('Under construction.')


    def transform_all(self, backtransform=False, reduce=True):
        """Function
        Wrapper for transformation of states, operators, and theta_list.
        """
        if self.tapering.initialized == 0 :
            prints('First perform tapering.run(). No transformation done')
            return
        if self.tapering.redundant_bits is None:
            prints('No qubits to be tapered-off. No transformation done')
            return
        self.transform_states(backtransform=backtransform, reduce=reduce)
        self.transform_operators(backtransform=backtransform, reduce=reduce)
        if self.pauli_list is not None:
            self.transform_pauli_list(backtransform=backtransform, reduce=reduce)
        if self.theta_list is not None:
            self.transform_theta_list(backtransform=backtransform, reduce=reduce)

    ####################
    # Orbital related  #
    ####################
    def alter(self, alter_pairs):
        """
        Exchange the orbitals.
        alter_pairs is assumed to have n pairs of orbital indices (thus even number of elements),
        which are supposed to be switched their orderings.
        """
        alter_list = iter(alter_pairs)
        prints("Orbital switched:")
        for i, j in zip(alter_list, alter_list):
            prints(f'   {i} <-> {j} ')
            # Save original lists into temporary lists
            temp = self.mo_coeff.copy()
            temp_E = self.mo_energy.copy()
            temp_irrep = self.irrep_list.copy()
            temp_character_list = deepcopy(self.character_list)

            # Switch
            self.mo_coeff[:, j] =  temp[:, i].copy()
            self.mo_coeff[:, i] =  temp[:, j].copy()
            self.mo_energy[j] = temp_E[i]
            self.mo_energy[i] = temp_E[j]
            self.irrep_list[2*j] = temp_irrep[2*i]
            self.irrep_list[2*j+1] = temp_irrep[2*i+1]
            self.irrep_list[2*i] = temp_irrep[2*j]
            self.irrep_list[2*i+1] = temp_irrep[2*j+1]

            for k in range(len(temp_character_list)):
                self.character_list[k][2*j] = temp_character_list[k][2*i]
                self.character_list[k][2*j+1] = temp_character_list[k][2*i+1]
                self.character_list[k][2*i] = temp_character_list[k][2*j]
                self.character_list[k][2*i+1] = temp_character_list[k][2*j+1]

        self.mo_coeff0 = self.mo_coeff.copy()
        self.orbital_rotation(mo_coeff=self.mo_coeff)

        # Redefine symmetry section
        if self.operators.pgs is not None:
            nfrozen = self.n_frozen_orbitals * 2
            ncore = self.n_core_orbitals * 2
            nact = self.n_active_orbitals * 2
            pgs_head, pgs_tail = nfrozen, nfrozen+ncore+nact
            self.operators.pgs = (self.symm_operations,
                  self.irrep_list[pgs_head:pgs_tail],
                  [x[pgs_head:pgs_tail] for x in self.character_list])
        # Redefine tapering class
        self.tapering = Z2tapering(self.operators.qubit_Hamiltonian,
                                   self.n_qubits,
                                   self.det,
                                   self.operators.pgs)

    def boys(self, *args):
        """
        Perform Boys' localization for selected orbitals using PySCF module.
        New mo_coeff is loaded in Q.mo_coeff (Q.canonical_orbitals are reserved for HF orbitals).
        It also transforms Hamiltonian automatically.

        Example:
            Q.boys(5,6)
            gives localized orbitals using 5 and 6 th orbitals.
            Q.boys(*[x for x in range(5)])
            gives localized orbitals using 0, 1, 2, 3, 4 th orbitals.
        """
        from quket.orbital import boys
        return boys(self, *args)

    def get_htilde(self):
        """
        Ecore = sum 2h[p,p] + 2(pp|qq) - (pq|qp)
        htilde = h[p,q] + 2(pq|KK) - (pK|Kq)
        """
        from quket.orbital import get_htilde
        return get_htilde(self)

    def ao2mo(self, mo_coeff=None, compact=True):
        """
        Integral transformation from ao to mo with incore memory (MPI)
        EXTREMELY SLOW?
        """
        from quket.orbital import ao2mo
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        return ao2mo(self.pyscf.intor('int2e', aosym='s4'), mo_coeff, False)

    def orbital_rotation(self, kappa_list=None,  mo_coeff=None):
        """
        Orbital rotation of Hamiltonian.
        """
        from quket.orbital import orbital_rotation
        orbital_rotation(self, kappa_list=kappa_list, mo_coeff=mo_coeff)


    def run(self):
        """Function
        Perform simulation by mimicing main.py
        """
        from quket.vqe import VQE_driver
        from quket.qite import QITE_driver

        # Sanitary check: use the new parameters for config
        self.cf.maxiter = self.maxiter
        self.cf.gtol = self.gtol
        self.cf.ftol = self.ftol
        self.cf.__post_init__()
        #

        # Remove DA, DB
        self.DA = None
        self.DB = None
        self.RelDA = None
        self.RelDB = None
        self.Daaaa = None
        self.Dbbbb = None
        self.Dbaab = None

        #if self.ansatz is None:
        #    prints(f"\n   Skipping job since ansatz is not specified.\n")
        if self.maxiter == 0:
            prints(f"\n   Skipping job since maxiter = {self.maxiter}.\n")
            self.energy = self.qulacs.Hamiltonian.get_expectation_value(self.state)
        else:
            if self.pauli_list is not None:
                ndim = len(self.pauli_list)
                theta_list = np.zeros((ndim))

            ############
            # VQE part #
            ############
            if self.method in ("vqe", "mbe"):
                VQE_driver(self,
                           self.cf.kappa_guess,
                           self.cf.theta_guess,
                           self.cf.mix_level,
                           self.cf.opt_method,
                           self.cf.opt_options,
                           self.cf.print_level,
                           self.maxiter,
                           self.cf.Kappa_to_T1)

            #############
            # QITE part #
            #############
            elif self.method == "qite":
                QITE_driver(self)


    def vqd(self, det=None, converge=None, delete=False):
        """
        Prepare VQD calculation.
        
        Args:
            det (str): String of determinant(s) for VQD initial state
            converge (bool): Force to prepare VQD even if the previous VQE(VQD) has not been converged.
            delete (bool): Cancel the VQD setting and roll back to one VQD setting. 
        """
        if converge is None:
            converge = self.converge

        if delete:
            if len(self.lower_states) == 0:
                raise ValueError(f"No state to be deleted in VQD preparation.")
            self.converge = True
            self.energy = self.lower_states[-1]['energy']
            self.state = self.lower_states[-1]['state'].copy()
            self.theta_list = self.lower_states[-1]['theta_list'].copy()
            try:
                self.det = self.lower_states[-1]['det'].copy()
            except:
                self.det = self.lower_states[-1]['det']
            self.init_state = set_state(self.det, self._n_qubits, mapping=self.cf.mapping)
            self.current_det = self.det
            if self.tapered['states']:
                self.init_state = self.transform_state(self.init_state, backtransform=False)
                prints('Tapered-off initial state...')
            if self.tapered['pauli_list']:
                self.get_pauli_list()
                self.transform_pauli_list()
                prints('Tapered-off pauli list...')
            else:
                self.get_pauli_list()
            self.lower_states.pop()
            prints(f'VQD rolled back to {len(self.lower_states)}th state.')
            return 

        if not converge:
            prints('VQE is not converged.  No VQD is performed.')
            return
        else:
            # About to store this state in lower_states list.
            # Check if this state has been already obtained.
            flag = True
            if len(self.lower_states) > 0:
                for k, x in enumerate(self.lower_states):
                    inner = abs(inner_product(x['state'], self.state))**2
                    if inner > 0.99:
                        prints(f"VQD Failed! \n The converged state is the same as previously converged k={k}th lower-state: \n    |<current|k>|**2 = {inner:8.6f}")
                        prints(f"The current state will be discarded.")
                        self.lower_states.pop(k)
                        prints(f"Tips: To obtain a meaningful state, run vqd() again with a different det (currently |{format(self.det, f'0{self.n_qubits}b')}> is used).")
                        return
            self.converge = False

            ### Set previous VQE information
            lower_state = {'energy': self.energy, 'state': self.state, 'theta_list': self.theta_list, 'det': self.det}
            self.lower_states.append(lower_state)

            prints(f"Performing VQD for excited state {len(self.lower_states)}")
            self.theta_list = None
            if det is None:
                prints(f"Initial determinant is not set, so use previous det:", end='')
                det = self.det
                opt = f"0{self._n_qubits}b"
                if type(det) is int:
                    prints(f" | {format(det, opt)} >")
                else:
                    for i, state_ in enumerate(det):
                        prints(f" {state_[0]:+.4f} * | {format(state_[1], opt)} >", end='')
                        if i > 10:
                            prints(" ... ", end='')
                            break
                        prints('')
            elif type(det) == int:
                if det >= 0:
                    self.det = det
                else:
                    prints(f"Invalid determinant description '{det}'")
            elif type(det) == str:
                from quket.fileio.read import read_multi
                try:
                    self.det, weight = read_multi(det)
                    if not self.det:
                        prints(f'Unrecognized det specification {det}.') 
                        self.vqd(delete=True)
                        return
                except:
                    prints(f"Invalid determinant description '{det}'")
            else:
                prints(f"Invalid determinant description '{det}'")
            self.current_det = self.det
            self.init_state = set_state(self.det, self._n_qubits, mapping=self.cf.mapping)
            if self.tapered['states']:
                self.init_state = self.transform_state(self.init_state, backtransform=False)
                prints('Tapered-off initial state...')
            if self.tapered['pauli_list']:
                self.get_pauli_list()
                self.transform_pauli_list()
                prints('Tapered-off pauli list...')
            else:
                self.get_pauli_list()
            self.state = self.init_state.copy()
            prints('VQD ready. Perform run().')


    ###############################################


    def vqe(self):
        from quket.vqe.vqe import vqe
        if self.qulacs.Hamiltonian is None:
            error('Hamiltonian (Observable) undefined in vqe.')
            return
        if self.ansatz is None and self.pauli_list is None:
            error('ansatz undefined in vqe.')
            return
        self.ndim = len(self.pauli_list)
        if self.theta_list is None:
            self.theta_list = np.zeros(self.ndim)
        elif self.ndim != len(self.theta_list):
            prints(f'Length of pauli_list {len(self.pauli_list)} != Length of theta_list {len(self.theta_list)}')
            if self.ndim > len(self.theta_list):
                prints(f'Append theta_list by zeros.')
                self.theta_list = np.append(self.theta_list, np.zeros(self.ndim - len(self.theta_list)))
            else:
                prints(f'Shrink theta_list.')
                self.theta_list = self.theta_list[:self.ndim]
        self.cf.opt_options['maxiter'] = self.maxiter
        vqe(self)

    def post_run(self): 
        #################
        # Post-VQE part #
        #################
        return

    def grad(self):
        # Nuclear gradient
        from quket.post import grad
        if self.DA is None and self.Daaaa is None:
            self.get_1RDM()
            if self.DA is None and self.Daaaa is None:
                return
        self.nuclear_grad = grad.nuclear_grad(self)

    def opt(self):
        # Geometry optimization
        from quket.post import grad
        if not self.converge:
            self.run()
        if self._n_qubits != self.n_qubits:
            reduce = True
        else:
            reduce = False
        if self.tapered["states"]:
            self.transform_states(backtransform=True, reduce=reduce)
        if self.tapered["operators"]:
            self.transform_operators(backtransform=True, reduce=reduce)
        if self.tapered["pauli_list"]:
            self.transform_pauli_list(backtransform=True, reduce=reduce)
        if self.tapered["theta_list"]:
            self.transform_theta_list(backtransform=True, reduce=reduce)
        self.nuclear_grad = grad.nuclear_grad(self)
        grad.geomopt(self,self._init_dict,self._kwds)

    def oo(self):
        from quket.orbital import oo
        if self.method == "vqe":
            oo(self)
        else:
            error(f'Orbital-optimization is not supported for {self.method}.')

    def read(self, fname, _i=0):
        """Function
        Reconstruct QuketData with input file.
        Essentially the same as quket.create(read="fname").
        """
        from quket.fileio import read_input, set_config
        from quket.utils import get_func_kwds

        input_dir, base_name = os.path.split(fname)
        input_dir = "." if input_dir == "" else input_dir
        input_name, ext = os.path.splitext(base_name)
        if ext == "":
            ext = ".inp"
        elif ext not in ".inp":
            input_name += ext
            ext = ".inp"
        input_file = f"{input_dir}/{input_name+ext}"
        cf.input_file = input_file
        cf.theta_list_file = f"{input_dir}/{input_name}.theta"
        cf.tmp = f"{input_dir}/{input_name}.tmp"
        cf.kappa_list_file = f"{input_dir}/{input_name}.kappa"
        cf.chk = f"{input_dir}/{input_name}.chk"
        cf.rdm1 = f"{input_dir}/{input_name}.1rdm"
        cf.adapt = f"{input_dir}/{input_name}.adapt"

        read_kwds = read_input()[_i]
        kwds = read_kwds

        ###
        ### self may have unexpected variables and parameters from previous calculations.
        ### Initialize ALL attributes
        ###
        init_dict = get_func_kwds(QuketData.__init__, kwds)
        Quket = QuketData(**init_dict)
        self.__dict__ = Quket.__dict__

        set_config(kwds, self)
        self.initialize(**kwds)
        # Saving input
        Quket._init_dict = init_dict
        Quket._kwds = kwds

        if self.model == "chemical":
            prints(f"NBasis = {self.mo_coeff.shape[0]}")
            if self.cf.print_level > 1 or cf.debug:
                if cf.debug:
                    format = '18.14f'
                else:
                    format = '10.7f'
                printmat(self.mo_coeff, name="MO coeff", format=format)
            if cf.debug:
                printmat(self.overlap_integrals, name="Overlap", format=format)

        if self.cf.do_taper_off:
            self.tapering.run(mapping=self.cf.mapping)

        if self.run_qubitfci:
            self.fci2qubit()
    ###############################################


    def save(self, filepath, verbose=True):
        """Function
        Save (mostly) parameters of QuketData.
        For qulacs.QuantumState and _pyscf_data, we decompose the data to pickle.
        qulacs.Observable and other unpicklable objects are discarded.

        Author(s): Takashi Tsuchimochi
        """

        data = _format_picklable_QuketData(self)

        if filepath is None:
            if cf.pkl != './.pkl':
                filepath = cf.pkl
            else:
                error('filepath required.')
                return
        if mpi.main_rank:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        if verbose:
            prints(f'Saved in {filepath}.')
        return
    ###############################################

    def load(self, filepath, verbose=True):
        """Function
        Load parameters of QuketData.
        For qulacs.QuantumState, we reconstruct from vector (2**nqubit).
        qulacs.Observable and other unpicklable objects are also reconstructed.

        Author(s): Takashi Tsuchimochi
        """
        if mpi.main_rank:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            data = None
        data = mpi.bcast(data, root=0)
        # Recover Quantum State #
        data = _restore_quantumstate_in_Data(data)
        self.__dict__.update(data)

        kwds = self.__dict__

        # Config
        init_dict = get_func_kwds(Config.__init__, kwds)
        self.cf = Config(**init_dict)
        self.cf.__dict__.update(data['cf'])
        # Operators
        self.operators = Operators(self.cf.mapping, self.n_qubits)
        self.operators.__dict__.update(data['operators'])
        # Projection
        init_dict = get_func_kwds(Projection.__init__, kwds)
        self.projection = Projection(**init_dict)
        self.projection.__dict__.update(data['projection'])
        # Multi
        init_dict = get_func_kwds(Multi.__init__, kwds)
        self.multi = Multi(**init_dict)
        self.multi.__dict__.update(data['multi'])
        # Adapt
        init_dict = get_func_kwds(Adapt.__init__, kwds)
        self.adapt = Adapt(**init_dict)
        self.adapt.__dict__.update(data['adapt'])

        # Construct Dynamic class like QuketData.initiliaze()
        obj = create_parent_class(self, kwds)
        if self.model == "chemical":
            n_qubits = self.n_qubits
            # Generate PySCF M
            from quket.utils import prepare_pyscf_molecule_mod
            self.pyscf = prepare_pyscf_molecule_mod(obj)
            set_dynamic_class(self, kwds, obj)
            self.n_qubits = n_qubits
            self.H_n_qubits = n_qubits

        # Reconstruct qulacs.Observable objects
        self.openfermion_to_qulacs()

        # Z2Tapering
        tapering_dict = self.tapering
        self.tapering = Z2tapering(self.operators.qubit_Hamiltonian, det=0)
        self.tapering.__dict__ = tapering_dict

        if verbose:
            prints('Data loaded successfully.')
        return

    def copy(self):
        """
        Copy QuketData.

        """
        init_dict = get_func_kwds(QuketData.__init__, self.__dict__["_kwds"])
        Quket = QuketData(**init_dict)
        data = deepcopy(_format_picklable_QuketData(self))
        data = _restore_quantumstate_in_Data(data)
        Quket.__dict__.update(data)
        kwds = Quket.__dict__
        # Config
        init_dict = get_func_kwds(Config.__init__, kwds)
        Quket.cf = Config(**init_dict)
        Quket.cf.__dict__.update(data['cf'])
        # Operators
        Quket.operators = Operators(self.cf.mapping, self.n_qubits)
        Quket.operators.__dict__.update(data['operators'])
        # Projection
        init_dict = get_func_kwds(Projection.__init__, kwds)
        Quket.projection = Projection(**init_dict)
        Quket.projection.__dict__.update(data['projection'])
        # Multi
        init_dict = get_func_kwds(Multi.__init__, kwds)
        Quket.multi = Multi(**init_dict)
        Quket.multi.__dict__.update(data['multi'])
        # Adapt
        init_dict = get_func_kwds(Adapt.__init__, kwds)
        Quket.adapt = Adapt(**init_dict)
        Quket.adapt.__dict__.update(data['adapt'])

        # Construct Dynamic class like QuketData.initiliaze()
        n_qubits = self.n_qubits
        obj = create_parent_class(Quket, kwds)

        if Quket.model == "chemical":
            # Generate PySCF M
            from quket.utils import prepare_pyscf_molecule_mod
            Quket.pyscf = prepare_pyscf_molecule_mod(obj)

        set_dynamic_class(Quket, kwds, obj)
        Quket.n_qubits = n_qubits
        Quket.H_n_qubits = n_qubits
        # Reconstruct qulacs.Observable objects
        Quket.openfermion_to_qulacs()
        # Z2Tapering
        tapering_dict = self.tapering.__dict__
        Quket.tapering = Z2tapering(Quket.operators.qubit_Hamiltonian, det=0)
        Quket.tapering.__dict__ = tapering_dict
        return Quket

    def set(self, **kwds):
        """
        Set or change parameters.
        Essentially the same as create(), but different in the following ways:
          (1) set() does not read an input file, for which read() should be used.
          (2) set() updates and overwrites parameters (whereas create() initializes all)
        """
        from quket.fileio import read_input_command_line, set_config
        kwds_ = read_input_command_line(kwds)
        self._kwds.update(kwds_)
        kwds__ = get_func_kwds(QuketData.__init__, kwds_)
        self.__dict__.update(kwds__)
        set_config(kwds, self)

def set_state(det_info, n_qubits, mapping='jordan_wigner'):
    if type(det_info) is str:
        det_info, weight = read_multi(det_info)
    if type(det_info) is int:
        if det_info < 0:
            prints(f"Invalid determinant description '{det_info}'")
        state = QuantumState(n_qubits)
        state.set_computational_basis(det_info)
        if mapping == "bravyi_kitaev":
            state = transform_state_jw2bk(state)
    elif type(det_info) is list:
        try:
            state = set_multi_det_state(det_info, n_qubits)
        except:
            prints(f"Invalid determinant description '{det_info}'")
        if mapping == "bravyi_kitaev":
            state = transform_state_jw2bk(state)
    elif det_info == "random":
        state = QuantumState(n_qubits)
        state.set_Haar_random_state()
        ### Broadcast
        state = mpi.bcast(state, root=0)
    return state
