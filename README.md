# quket 0.9.4
```

 Quket: Quantum Unified Kernel for Emulator Toolbox 
 Official Version 0.9.4
 
     Copyright 2019-2022 Takashi Tsuchimochi, Yuto Mori, Yuma Shimomoto, Taisei Nishimaki, Yoohee Ryo,
                         Masaki Taii, Takahiro Yoshikura, TsangSiuChung, Kazuki Sasasako, Kengo Yoshimura. All rights Reserved.

 This suite of programs simulates quantum computing for electronic Hamiltonian.
```
* [Required libraries](#required-libraries)
* [How to install](#how-to-install)
* [Quick introduction](#quick-introduction)
* [How to use (more detailed document)](#how-to-use-more-detailed-document)
* [File specifications](#file-specifications)
* [How to write ***.inp](#how-to-write-inp)
   * [Necessary options](#necessary-options)
   * [Tips: multiple jobs](#tips-multiple-jobs)
   * [Other options](#other-options)
      * [For PySCF](#for-pyscf)
      * [For Quantum Circuit part](#for-quantum-circuit-part)
      * [For scipy.optimize](#for-scipyoptimize)
      * [Spin Symmetry Projection](#spin-symmetry-projection)
      * [Initial state (determinant)](#initial-state-determinant)
      * [Multi-state](#multi-state)
      * [VQD](#vqd)
# Required libraries
 - python             >= 3.6
 - pyscf              >= 1.7.5 
 - openfermion        >= 0.10.0 
 - openfermionpyscf   >= 0.4    
 - Qulacs             >= 0.1.9   
 - numpy           
 - scipy 
 - psutil
 - pyberny
 - (optional) mpi4py: MPI
 - (optional) threadpoolctl >= 3: Faster ansatz operation for hybrid parallel calculation. (For this to work, unbind processes and cores, e.g., I_MPI_PIN=0)

# How to install
```bash
git clone https://github.com/quket/quket
cd quket
pip install .
```

# Quick introduction
Simply create `QuketData` instance and run.
```python
import quket
Q = quket.create(basis="sto-3g", geometry=[('H', (0,0,0)), ('H', (0,0,1))], ansatz="uccsd")
Q.run()
```
`Quket` overrides `QuantumState` of `Qulacs` and `QubitOperator` and `FermionOperator` of `OpenFermion` as `quket.QuantumState`, etc. They are interfaced via `Quket`, allowing for easy-non-unitary operations.
```python
import quket
# Set an operator: A = X0 Z1 + Z0 X1
A = quket.QubitOperator('[X0 Z1] + [Z0 X1]')
# Set a two-qubit state:  psi = |00>
psi = quket.QuantumState(2)
# Apply A to psi to obtain a new state
Apsi = A@psi
# Print state
quket.print_state(Apsi)
```
```
[Sample Output]
 Basis         Coef
| 01 > : +1.0000 +0.0000i
| 10 > : +1.0000 +0.0000i
```
___
# How to use (more detailed document)
Please refer to documents in `docs` directory for further instructions.
   * [<strong>I. On command line</strong>](#i-on-command-line)
      * [0. (Optional) Open-MP thread setting](#0-optional-open-mp-thread-setting)
      * [1. Import quket and create an instance of QuketData class](#1-import-quket-and-create-an-instance-of-quketdata-class)
         * [(1) Put options in argumetns of create()](#1-put-options-in-argumetns-of-create)
         * [(2) (<strong>Recommended</strong>) Use read option of create() to read in an input](#2-recommended-use-read-option-of-create-to-read-in-an-input)
         * [(3) （<strong>Recommended</strong>）Use read() function of QuketData](#3-recommendeduse-read-function-of-quketdata)
         * [(4) Update parameters in QuketData (set() and initialize())](#4-update-parameters-in-quketdata-set-and-initialize)
      * [2. Simulate with methods in QuketData](#2-simulate-with-methods-in-quketdata)
         * [(1). run() and vqe()](#1-run-and-vqe)
         * [(2). print_state()](#2-print_state)
         * [(3). get_E(), get_S2(), get_N()](#3-get_e-get_s2-get_n)
         * [(4). fci2qubit(nroots=1)](#4-fci2qubitnroots1)
         * [(5). fidelity()](#5-fidelity)
         * [(6). get_1RDM(), get_2RDM()](#6-get_1rdm-get_2rdm)
         * [(7). taper_off(backtransform=False, reduce=True)](#7-taper_offbacktransformfalse-reducetrue)
            * [(7a). First perform tapering.run()](#7a-first-perform-taperingrun)
            * [(7b). transform_***](#7b-transform_)
            * [Example](#example)
         * [(8). read(filepath)](#8-readfilepath)
         * [(9). save(filepath), load(filepath)](#9-savefilepath-loadfilepath)
         * [(10) copy()](#10-copy)
         * [(11). grad(), opt()](#11-grad-opt)
         * [(12). vqd(det=None)](#12-vqddetnone)
         * [(13). oo()](#13-oo)
   * [<strong>II. On cluster (parallel execution)</strong>](#ii-on-cluster-parallel-execution)
      * [Parallel execution](#parallel-execution)
         * [Thread parallelization](#thread-parallelization)
         * [MPI parallelization](#mpi-parallelization)
         * [Hybrid parallelization](#hybrid-parallelization)
<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

## __I. On command line__
### 0. (Optional) Open-MP thread setting
To change the number of threads to `nthreads`, e.g. 4, do either of the followings _**before**_ importing `quket` (more specifically, `numpy`) 
```bash
export OMP_NUM_THREADS=4 
```
or
```python
import os
os.environ["OMP_NUM_THREADS"] = "4"
```

### 1. Import quket and create an instance of `QuketData` class
`create()` function generates a `QuketData` instance that holds most of the information needed for simulations. 
If no arguments are passed, it generates an empty `QuketData` instance.
In such a case, all the attributes are default values; for example, the number of qubits is undefined.
```python
import quket
Q = quket.create()
Q
```
```
System undefined.  
Method    = vqe  
Ansatz    = None  
Nqubits   = None  
Energy    = 0.0  
Converged = False  
```

To initialize a simulation, use either of the following options.
#### (1) Put options in argumetns of `create()`
This prepares a simulation based on options entered, automatically running pyscf and performing Jordan-Wigner transformation, etc.
```python
Q = quket.create(basis="sto-3g", geometry="H 0 0 0; H 0 0 1", ansatz="uccsd")
```

```
Basis set = sto-3g  
  
*** Geometry ******************************  
  H     0.0000000    0.0000000    0.0000000  
  H     0.0000000    0.0000000    1.0000000  
*******************************************  
  
Symmetry Dooh : D2h(Abelian)  
E[FCI]    = -1.1011503302326187     (Spin = 1   Ms = 0)  
E[HF]     = -1.0661086493179366    (Spin = 1   Ms = 0)  
NBasis = 2  
```

#### (2) (__Recommended__) Use read option of `create()` to read in an input
Write an input file `H2.inp` as follows:
```
basis = sto-6g
ansatz = uccsd
geometry
H 0 0 0
H 0 0 1
```
Pass the path of `H2.inp` as str in `create()`:
```python
Q = quket.create(read="./H2.inp")
```
Here, `read=` and `.inp` can be both omitted. 
```python
Q = quket.create("H2")
```
##### Tips (a): One input can hold multiple jobs by separating the lines by `@@@`. If this is the case, use `job` option to specify which job number you want to perform.
##### Tips (b): If other options are specified directly in `create()` in addition to `read` option, then the options are overwritten by the former. For instance, if one wants to use the same input as `H2.inp` but with the 6-31G basis, the following command suffices:
```python
Q = quket.create("H2", basis="6-31g")
```
```
*** Geometry ******************************
  H     0.0000000    0.0000000    0.0000000
  H     0.0000000    0.0000000    1.0000000
*******************************************

Symmetry Dooh : D2h(Abelian)
E[FCI] = -1.12677835261831
E[HF]  = -1.0948079628605112
NBasis = 4
```


#### (3) （__Recommended__）Use `read()` function of `QuketData`
If there exists a (empty) `QuketData` instance, one can use `read()` option to read an input.
```python
Q = quket.create()
Q.read("H2")
```
Also, as noted below, one may use `load()` function to load a binary file stored in the proceeding simulation.


#### (4) Update parameters in `QuketData` (`set()` and `initialize()`)
Use `set()` method to redefine and update parameters, and perform `initialize()` to reflect the changes:
```python
Q = quket.create(geometry="H 0 0 0; H 0 0 1", basis="6-31g")
Q.set(ansatz="uccsd", basis="sto-3g")
Q.initialize()
```
`initialize()` re-build the qubit Hamiltonian and re-define all the necessary parameters (number of electrons, etc.). Note that `create()` indeed performs `initialize()`.

___
___
___
### 2. Simulate with methods in `QuketData`
The following functions are essential ones but are not comprehensive.
#### (1). `run()` and `vqe()`
`run()` performs a calculation corresponding to `method` and `ansatz` attributes.
```python
Q.run()
```
```
Entered VQE driver
Performing VQE for uccsd
Number of VQE parameters: 3
Initial configuration: | 0011 >
Convergence criteria: ftol = 1E-09, gtol = 1E-05
Theta list = zero
Circuit order: Exp[T1] Exp[T2] |0>
Initial E[uccsd] = -1.066108649318  <S**2> = +0.000000000000000  rho = 1
     1: E[uccsd] = -1.100498918254  <S**2> = +0.000000000000000  Grad = 5.45e-02  CPU Time =  0.00  (0.00 / step)
     2: E[uccsd] = -1.101149957270  <S**2> = +0.000000000000000  Grad = 1.30e-03  CPU Time =  0.00  (0.00 / step)
...
```
Another way to perform VQE is `vqe()`. 

#### (2). `print_state()`
Print out the quantum state in `QuketData.state`. The same as `quket.fileio.print_state(QuketData.state)`.
```python
Q.print_state()
```
```
  Basis          Coef
| 0011 > : +0.9844 +0.0000i
| 1100 > : -0.1762 +0.0000i
```

#### (3). `get_E()`, `get_S2()`, `get_N()`
The expectation values of Energy, S^2, and electron number of `QuketData.state`.
```python
Q.get_E()
```
```
-1.1088730601683976
```
This is the same as the result of using `qulacs`'s `Observable`, which is stored in `QuketData.qulacs.Hamiltonian`:
```python
Q.qulacs.Hamiltonian.get_expectation_value(Q.state)
```
```
-1.1088730601683976
```
However, `get_E()` calls `QuketData.get_expectation_value()` that performs the same operation with MPI, if `mpi4py` is available. 
Use `get_E(state=***)` to compute the expectation value of an arbitrary `QuantumState` instance (the number of qubits has to be the same)。
```python
Q.get_E(Q.init_state)
```
```
-1.0735829307863616
```
Since `Q.init_state` contains the HF state, we get the HF energy.

#### (4). `fci2qubit(nroots=1)`
This performs FCI in the qubit representation (VQE) but is significantly slow for larger qubits. 
It uses `QuketData.fci_coeff` obtained from `PySCF`. If `nroots` option (int) is given, multiple states are computed, but this requres `nroots` option in `create()`, too, so that `QuketData.fci_coeff` contains `nroots` solutions. 
```python
Q.fci2qubit()
```
```
FCI Energy by Qubits: -1.10887306016844
(FCI state)
  Basis          Coef
| 0011 > : -0.9844 +0.0000i
| 1100 > : +0.1762 +0.0000i
```
The final FCI `QuantumState`s are stored in `QuketData.fci_states[:]`.

#### (5). `fidelity()`
If `QuketData.fci_states` is defined via `fci2qubit()`, then `QuketData.fidelity()` computes the fidelity of `QuketData.state`.
```python
Q.fidelity()
```
```
0.9999999999999625
```
Note that UCCSD is equivalent to FCI for 2e systems.

#### (6). `get_1RDM()`, `get_2RDM()`
These compute 1RDM (unrelaxed and relaxed ones) and 2RDM of `QuketData.state` in the spin-orbital basis. `get_1RDM()` may perform `get_2RDM()`, as needed for relaxed 1RDM. The results are stored in `QuketData.DA`, `QuketData.DB`, `QuketData.Daaaa`, `QuketData.Dbbbb`, `QuketData.Dbaab`. The relaxed 1RDM is stored in `QuketData.RelDA` and `QuketData.RelDB`.
```python
Q.get_1RDM()
from quket import printmat
printmat(Q.RelDA, 'Relaxed density matrix')
```
```
Relaxed density matrix

              0             1          

    0     0.9689452    -0.0000000  
    1    -0.0000000     0.0310548  

```

#### (7). `taper_off(backtransform=False, reduce=True)`
This function calls `QuketData.tapering.run()` to find the redundant qubits that can be tapered-off, and transform `QuketData.state`, `QuketData.qulacs.Hamiltonian`, `Quket.pauli_list`, `Quket.theta_list`, etc., to the new reduced mapping. If `backtransform=True`, backtransformation is done. The option `reduce=False` transform things to the new mapping but not reduce the qubits. 
```python
Q.taper_off()
```
```
Tapering-Off Results:
List of Tapered-off Qubits:  [0, 1, 2]
Qubit: 0    Tau: 1.0 [Z0 Z3]
Qubit: 1    Tau: 1.0 [Z1 Z3]
Qubit: 2    Tau: 1.0 [Z2 Z3]

States     transformed.
Operators  transformed.
pauli_list transformed.
theta_list transformed.
```
Usually, `taper_off()` method can be used blindly, but how it works is explained below for clarity:

##### (7a). First perform `tapering.run()`
Perform the tapering-off algorithm to find out the reduced qubits and unitary transformation.
(If `create()` has `taper_off = True` option, this is automatically done.)
```python
Q.read("H2")
Q.tapering.run()
```
```
Tapering-Off Results:
List of Tapered Qubits: 0 1 2   (total of 3 qubits)
Qubit: 0    Tau: 1.0 [Z0 Z3]
Qubit: 1    Tau: 1.0 [Z1 Z3]
Qubit: 2    Tau: 1.0 [Z2 Z3]
```
The unitary operator is stored in `QuketData.tapering.clifford_operators`、the reduced qubits in `QuketData.tapering.redundant_bits`、and the corresponding eigenvalues in `QuketData.tapering.X_eigvals`. 

##### (7b). `transform_***`
Perform the transformation for each quantity.
- `transform_state(backtransform=False, reduce=True)`  : Transform `QuketData.state`
- `transform_states(backtransform=False, reduce=False)` : Transform states in `QuketData` (`state`, `init_state`, `fci_states`, etc.)
- `transform_operators(backtransform=False, reduce=True)` : Transform operators in `QuketData` (Hamiltonian, S^2, Number, etc.)
- `transform_pauli_list(backtransform=False, reduce=True)` : Transform `QuketData.pauli_list`
- `transform_theta_list(backtransform=False, reduce=True)` : Transform `QuketData.theta_list`

##### Example
```python
Q.transform_states()
```
```
States     transformed.
```
```python
Q.print_state()
```
```
Basis         Coef
| 0 > : +1.0000 +0.0000i
```
0, 1, and 2 qubits from the total of 4 qubits are tapered-off. Note that `qulacs.Observable` is not transformed yet, so using `get_E()` results in an error:
```python
Q.n_qubits
```
```
1
```
```python
Q.get_E()
```
```
Warning!
 Operators are tapered-off [False]
   States are tapered-off [True]
The result below may be nonsense.

 Mismatch of n_qubits between ops (4) and state (1) 

Exception: Error termination of quket.
```
Use `transform_operators()` to reconcile this inconsistency:  
```python
Q.transform_operators()
```
```
Operators  transformed.
```
```python
Q.get_E()
```
```
-1.0735829307863611
```
For simulations, `QuketData.pauli_list` has to be also transformed.
Most of the methods implemented in Quket use Pauli rotations, for which Pauli strings are listed in `QuketData.pauli_list`. 
```python
Q.transform_pauli_list()
```
```
pauli_list transformed.
```python
Q.run()
```
```
Entered VQE driver
Performing VQE for uccsd
Number of VQE parameters: 1
Initial configuration: | 11 >
Convergence criteria: ftol = 1E-09, gtol = 1E-05
Circuit order: Exp[T1] Exp[T2] |0>
Initial E[uccsd] = -1.073582930786  <S**2> = +0.000000000000000  rho = 1  CNOT = 0
     1: E[uccsd] = -1.108219989158  <S**2> = +0.000000000000000  Grad = 5.45e-02  CNOT = 13  CPU Time =  0.00  (0.00 / step)
     2: E[uccsd] = -1.108872678203  <S**2> = +0.000000000000000  Grad = 1.32e-03  CNOT = 13  CPU Time =  0.00  (0.00 / step)
     3: E[uccsd] = -1.108873060168  <S**2> = +0.000000000000000  Grad = 5.33e-07  CNOT = 13  CPU Time =  0.00  (0.00 / step)
 Final: E[uccsd] = -1.108873060168  <S**2> = +0.000000000000000  CNOT = 13  rho = 1

(uccsd state)
 Basis         Coef
| 0 > : +0.9844 +0.0000i
| 1 > : -0.1762 +0.0000i


VQE Done: CPU Time =          0.0119
```

To backtransform to the original mapping, use `backtransform=True`:
```python
Q.transform_operators(backtransform=True)
Q.transform_states(backtransform=True)
```
```
Operators  backtransformed.
States     backtransformed.
>>> Q.print_state()
  Basis          Coef
| 0011 > : +0.9844 +0.0000i
| 1100 > : -0.1762 +0.0000i
```

As said above, these operations are done in one step by `QuketData.taper_off(backtransform=False, reduce=True)`
```python
Q.taper_off()
```
```
States     transformed.
Operators  transformed.
Current pauli_list is already tapered-off. No transformation done
Current theta_list is already tapered-off. No transformation done
```
Note `QuketData.tapered` manages the flags if states and operators (and pauli_list, etc.) are tapered-off or not. 
It is highly suggested to make sure that all entries are consistent:
```python
Q.tapered
```
```
{'operators': True, 'states': True, 'pauli_list': True, 'theta_list': True}
```
```python
Q.taper_off(backtransform=True)
```
```
States     backtransformed.
Operators  backtransformed.
pauli_list backtransformed.
theta_list backtransformed.
```
```python
Q.tapered
```
```
{'operators': False, 'states': False, 'pauli_list': False, 'theta_list': False}
```

If qubits are not to be reduced, set `reduce=False`:
```python
Q.taper_off(reduce=False)
Q.get_E()
```
```
States     transformed.
Operators  transformed.
pauli_list transformed.
theta_list transformed.
-1.108873060168396 
```
```python
Q.print_state()
```
```
  Basis          Coef
| 0000 > : +0.3480 +0.0000i
| 0001 > : -0.3480 +0.0000i
| 0010 > : -0.3480 +0.0000i
| 0011 > : +0.3480 +0.0000i
| 0100 > : +0.3480 +0.0000i
| 0101 > : -0.3480 +0.0000i
| 0110 > : -0.3480 +0.0000i
| 0111 > : +0.3480 +0.0000i
```
However, `reduce=False` is mainly for a debugging purpose.

#### (8). `read(filepath)`
Read an input file `filepath` and reconstruct `QuketData`.

#### (9). `save(filepath)`, `load(filepath)`
`save(filepath)` creates a binary file at `filepath` and store QuketData in it、and `load(filepath)` loads it.
In the example below, `QuketData` (`Q`) is stored in "Q.qkt" (in the same working directory) and we load this data in a different empty `QuketData` to copy. 
Note that loading the data overwrites the information (except for the information that are not stored in the data file). 
```python
Q.save('Q.qkt')
```
```
Saved in Q.qkt.
```
```python
new = quket.create()
new.load('Q.qkt')
```
```
Data loaded successfully.
```
```python
new
```
```
*** Geometry ******************************
  H     0.0000000    0.0000000    0.0000000
  H     0.0000000    0.0000000    1.0000000
*******************************************

Basis     = sto-6g
NBasis    = 2
Ne        = 2
Norbs     = 2
Ms        = 1
Method    = vqe
Ansatz    = uccsd
Nqubits   = 4
Energy    = -1.1088730601683976
Converged = True

```

#### (10) `copy()`
Copy a `QuketData` instance:
```python
X = Q.copy()
X.get_E()
```
```
-1.108873060168396 
```

#### (11). `grad()`, `opt()`
Compute the nuclear energy gradient (`grad()`) and perform geometry optimization (`opt()`).
```python
Q.grad()
```
```
 === Computing 1RDM === 
 === Computing 2RDM === 

*** Nuclear Gradients *********************
            x            y            z
  H     0.0000000    0.0000000   -0.1133090
  H     0.0000000    0.0000000    0.1133090
*******************************************
```

```python
Q.opt()
```
```
...
...
...
*** Nuclear Gradients *********************
            x            y            z
  H     0.0000000    0.0000000   -0.0000559
  H     0.0000000    0.0000000    0.0000559
*******************************************

Geometry Optimization Converged in 5 cycles

*** Geometry ******************************
  H     0.0000000    0.0000000   -0.3665310
  H     0.0000000    0.0000000    0.3665310
*******************************************
```

#### (12). `vqd(det=None)`
Initiate VQD (VQE is assumed to have finished). The initial state of the VQD is specified by, for example, `det = "1100"`.
If not specified, HF is used, and VQD possibly converges to the first VQE state.
```python
Q.vqd(det='1001')
```
```
Performing VQD for excited state 1
VQD ready. Perform run().
```
Then do `run()` to do VQE (VQD).
```python
Q.run()
```
```
Entered VQE driver
Performing VQE for uccsd
Number of VQE parameters: 3
Initial configuration: | 1001 >
Convergence criteria: ftol = 1E-09, gtol = 1E-05
Circuit order: Exp[T1] Exp[T2] |0>
Initial E[uccsd] = -0.558570381277  <S**2> = +1.000000000000000  rho = 1  CNOT = 0
     1: E[uccsd] = -0.737814227010  <S**2> = +1.909297426825680  Grad = 1.64e-01  CNOT = 17  CPU Time =  0.01  (0.00 / step)
     2: E[uccsd] = -0.753222511412  <S**2> = +1.987463086021676  Grad = 6.22e-02  CNOT = 17  CPU Time =  0.01  (0.00 / step)
     3: E[uccsd] = -0.755692877954  <S**2> = +1.999995164490748  Grad = 1.23e-03  CNOT = 17  CPU Time =  0.00  (0.00 / step)
     4: E[uccsd] = -0.755693831130  <S**2> = +1.999999999915676  Grad = 5.06e-06  CNOT = 17  CPU Time =  0.00  (0.00 / step)
 Final: E[1-uccsd] = -0.755693831130  <S**2> = +1.999999999915676  CNOT = 17  rho = 1

(uccsd state)
  Basis          Coef
| 0110 > : +0.7071 +0.0000i
| 1001 > : +0.7071 +0.0000i


VQE Done: CPU Time =          0.0210
```

#### (13). `oo()`
Orbital-optimization is performed. See sample inputs.

A sample `N2.ipynb` is in docs directory.


___
___
___


## __II. On cluster (parallel execution)__
`Quket` also provides a wrapper to automatically perform the simulation on a cluster based on an input file. 

(1) Prepare an input `***.inp`

(2) Use `main.py` to run
```bash
python main.py *** 
```
This will return `***.log` (among other files), in which the results are printed. 

### Parallel execution
`Quket` supports open-MP and MPI (via `mpi4py`).
#### Thread parallelization
Specify by `-nt` option to use $NTHREADS (defaulted to 1) threads.
```bash
python3.8 main.py *** -nt $NTHREADS  
```

#### MPI parallelization
MPI is available in many places in `quket`, such as VQE gradients, etc.
For an MPI parallel calculation using $NPROCS processes, do
```bash
mpirun -np $NPROCS python3.8 -m mpi4py main.py ***
```

#### Hybrid parallelization
```bash
mpirun -np $NPROCS python3.8 -m mpi4py main.py *** -nt $NTHREADS
```


___
___
___


# File specifications 

- `***.inp` ： input file
- `***.log` ： output file
- `***.out` ： stdout file 
- `***.chk` ： PySCF chkpoint file 
Depending on methods, the following files may be generated. 
- `***.theta` ： VQE parameters 
- `***.kappa` ： Orbital rotation parameters 
These files can be read as an initial guess for VQE.
___
# How to write `***.inp`
An input file should be prepared by writing options in a line-by-line style. Some sample inputs (and outputs) can be found in `samples` directory.

## Necessary options
- `method`        : vqe, qite, mbe
- `ansatz`        : uccsd, sauccsd (spin-adapted uccsd), uccgd, 2-UpCCGSD, etc.
- `geometry`      : Either in the xyz format or z-matrix format.
```
geometry
  A    Ax  Ay  Az
  B    Bx  By  Bz
  C    Cx  Cy  Cz 
  ...
```
or
```
geometry
  A
  B 1 RAB
  C 1 RAC 2 Ang
  ...
```
For the interative mode, one can pass `geometry` in terms of string with `;` being a break. E.g.,
```python
Q = create(geometry="O 0 0 0; H 0 0 0.9; H 0 0.9 0")
Q = create(geometry="O; H 1 0.9; H 1 0.9 2 90")
```

## Tips: multiple jobs
Multiple jobs can be prepared in one input file, by separating each job by `@@@`. Note that most options are taken over from the previous job as is. Set `clear` to remove/clear all the previous options.

## Other options
### For PySCF
- `basis`               : Basis function used (default: sto-3g)
- `multiplicity`        : __NOT__ spin multiplicity, but Nalpha - Nbeta + 1 (default: 1)
- `ms`                  : Number of unpaired electrons, Nalpha - Nbeta. This has a higher priority than `multiplicity` (default: 0)
- `charge`              : Charge (default: 0) 
- `pyscf_guess`         : Initial guess for PySCF. 'minao' or 'read' (same as 'chkfile').
- `n_electrons`         : Number of active electrons (default: all) 
- `n_orbitals`          : Number of active orbitals (default: all)
- `spin`                : Spin multiplicity for post-HF (e.g., FCI). Default is the same as `multiplicity`. If you wish to run HF in the singlet, and then use the so-obtained HF orbitals to run FCI in the triplet state, set `ms=0`, `spin=3` (or set `multiplicity=1`, `spin=3`)


### For Quantum Circuit part
- `rho`                 : Trotter number (default is 1).
- `layers`              : Number of layers (Trotter slices with different parameters)
- `kappa_guess`         : Initial guess for kappa. 'zero' (default), 'read', 'mix', 'random'.
- `theta_guess`         : Initial guess for VQE amplitudes. 'zero' (default), 'read', 'random'.
- `Kappa_to_T1`         : If true, use `***.kappa` to create the initial guess of T1 in UCCSD. Default is False.
- `mix_level`           : The number of orbitals around HOMO-LUMO to be mixed, in order to generate a spin mixed state (`kappa_guess = "mix"`). This makes the calculation intentially spin-unrestricted.
- `DS`                  : The ordering of application of T1 and T2. If 0 (default), Exp[T1]Exp[T2]. If 1, Exp[T2]Exp[T1].
- `constraint_lambda`   : The lambda value for S**2 penalty.
- `ansatz`              : Currently the following ansatze are available: hf, uhf, phf, uccd, uccsd, uccgd, uccgsd, uccgsdt, uccgsdtq, sauccd (= spin-adapted uccd), sauccsd, sauccgd, sauccgsd, adapt, k-upccgsd (with k being an integer, e.g., 2-upccgsd), hva/hamiltonian (Hamiltonian variational ansatze), ahva/anti-hamiltonian (anti-symmetrized HVA). 


### For scipy.optimize
- `opt_method`          : Scipy method for minimize (default is 'L-BFGS')
- `gtol`                : Same as scipy's gtol (gradient threshold for convergence).
- `ftol`                : Same as scipy's ftol (energy threshold for convergence).
- `eps`                 : Stepsize for gradient. 
- `maxiter`             : Maximum iteration number. If 0, just run PySCF and perform Jordan-Wigner transformation, and exit. If -1, just compute the energy.

### Spin Symmetry Projection
- `spinproj`            : If `True`, spin-projection is performed (for VQE).
- `spin`                : Spin multiplicity for spin-projection (same as post-HF, see above).
- `euler`               : Numbers of euler angles for projection, (alpha,beta,gamma). Default is (0,-1,0) (i.e., no projection). Since usually Sz is preserved and thus only beta is required, if `euler` has one integer input, it is regarded as beta. If it has two integer inputs, they are (alpha, beta), and if it has three inputs, they are (alpha,beta,gamma). 
```
euler =  4        ->  (1,4,1)
euler =  2, 4     ->  (2,4,1)
euler =  2, 4, 3  ->  (2,4,3)
```
__Note that in several cases, spin-projection is applied via `S2Proj(QuantumState)` in a state-vector format, which yields a superposition of spin-rotated states of `QuantumState` according to (`alpha`,`beta`,`gamma`) grids. For the complete spin-projection, it requires to set `alpha`, `beta` > 0 even for Sz-preserving ansatze such as UCCSD.__ 


### Initial state (determinant)
`det`, `determinant`   : Initial determinant (or configuration). Default is Hartree-Fock.
```
det = 00001111
det = |00001111>
det = 1 * 00001111
det = 1 * |00001111> - 1 * |00110011>
```
Any of the above works; however, for ansatze that assume a single determinant as a starting point, such as UCCSD, a multi-determinant specification like the last example will result in an error termination.

For the interative-mode, one can define `det` in terms of string,
```python
Q = create(det="1 * |00001111> - 1 * |00110011>")
```

### Multi-state
`multi` section is supported for multi-state calculations such as Model-Space Quantum Imaginary Time Evolution. After `multi`, set states as a bit string or multi-determinant specification. The subsequent value is `weight` and can be omitted (default is 1.0). 
```
multi:
    |00001111>   0.5 
    |00110011>   0.8
``` 
For the comman-line mode, pass a list containing the bit strings:
```python
Q = create(....., multi=["00001111   0.5", "|00100111>  0.8"], ...)
```

### VQD
To enable VQD, prepare the `excited` section. Currently, only orthogonally-Constrained VQE (OC-VQE) of Lee et al. is supported, where the penalty is the previous energy. Also, the allowed state specification is a single determinant, but multi-determinant states are not supported yet. 
If determinants (bits) are given, VQD is performed starting from these determinants. In the example below, we perform two excited state calculations starting from `|00001111>` and `|00100111>` in this order.
```
excited:
    |00001111>
    |00100111>
```
For the comman-line mode, pass a list containing the bit strings:
```python
Q = create(....., excited=["00001111","00100111"], ...)
```
This will perform VQE and then VQD. Instead, one can simply perform VQE first without specifying `excited` and then perform `vqd()`
```python
Q.vqe()
Q.vqd(det="00110011")
Q.vqe()
```

