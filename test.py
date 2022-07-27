#!/usr/local/bin/python3.8
"""test.py
This test code should be run to find possible bugs. 
This program relies on the Quket log format where
the final converged energy appears as

Final: E[uccsd] = -7.882391426828  ... 
                  ~~~~~~~~~~~~~~~
"""
import os
import sys

### CHANGE ACCORDINGLY ###
#nprocs=16
#nthreads=2
########

QDIR = os.getcwd()
SAMPLEDIR = QDIR+"/samples/"

UCC_test_list = {
"UCC/bsuccsd_h2o": -75.459750463545,  ##  Final:  E[uccsd] =  -75.459750463545 
"UCC/sauccsd_h2o": -75.455012061410 , 
"UCC/uccsd_h2": -1.137283834489,     ##  Final: E[uccsd] = -1.137283834489  
"UCC/uccsd_h4": -1.967526532457,     ##  Final:  E[uccsd] = -1.967526532457 
"UCC/uccsd_o_triplet":  -74.809265567978,  ##  Final: E[uccsd] = -74.809265567978 
"UCC/hubbard_6sites_U8_doped": 4.764182183781,   ##  Final:  E[uccsd] =  4.764198299062   
}

Excited_test_list = {
"UCC/uccsd_excited": [-7.882313805063, +0.000000000105260,  -7.766572162159, +1.999927671243692, -7.882313805063, +0.000000000105260,  -7.749271291916, 0.000125391641885]
}
# Final: E[uccsd] = -7.882313805063  <S**2> = +0.000000000105260  rho = 1
# Final: E[uccsd] = -7.766572162159  <S**2> = +1.999927671243692  rho = 1
# Final: E[uccsd] = -7.882313805063  <S**2> = +0.000000000105260  rho = 1
# Final: E[uccsd] = -7.749263372527  <S**2> = +0.001567252484652  rho = 1

UpCCGSD_test_list = {
"k-UpCCGSD/2-upccgsd_LiH":  -7.882395806145, ##Final: E[2-UpCCGSD] = -7.882395806145  <S**2> = -0.000000000000001
"k-UpCCGSD/3-upccgsd_N2": -107.588100237741,  ##  Final: E[3-UpCCGSD] = -107.588100237741
"k-UpCCGSD/S2Proj_2-upccgsd_o_triplet": -74.808873630481, ## Final: E[2-UpCCGSD] = -74.808873630481  <S**2> = +1.999999999999994
}

ADAPT_test_list = {
"ADAPT/adapt_h4": -1.969513701027,   ##  Final: E[adapt] = -1.969513701027 
"ADAPT/adapt_n2": -107.520149911103, ##  Final: E[adapt] = -107.520149911103   <S**2> = 0.000166740269650   Fidelity = 0.000000
"ADAPT/adapt_n2_spin": -107.519862465823, ## Final: E[adapt] = -107.519862465823   <S**2> = 0.000166896813282   Fidelity = 0.000000
"ADAPT/adapt_n2_spinproj": -107.520386372635, ## Final: E[adapt] = -107.520386372635   <S**2> = 0.000000000000000   Fidelity = 0.000000
"ADAPT/adapt_h6_read": -2.825001966646, ## Final: E[adapt] = -2.825001966646   <S**2> = 0.995559026524723   Fidelity = 0.000000 
}

PHF_test_list = {
"PHF/phf_h2o": -75.460201192275 ,  ##  Final:  E[phf] = -75.460201192275   <S**2> = -0.000000000000003  rho = 1 
"PHF/sghf_h2o": -75.460320773368, ##  Final:  E[sghf] = -75.460320773368   <S**2> = -0.000000000000002  rho = 1
"PUCCD/puccd_n2": -108.496335627576,  ##  Final:  E[opt_puccd] = -108.496335627576   <S**2> =  0.000000000000001  rho = 1
}

Tapering_test_list = {
"Tapering/h4_tapered": ["List of Tapered-off Qubits:  [0, 1, 2, 4]",
                        "Qubit: 0    Tau: 1.0 [Z0 Z3 Z5 Z6]",
                        "Qubit: 1    Tau: 1.0 [Z1 Z3 Z5 Z7]",
                        "Qubit: 2    Tau: 1.0 [Z2 Z3]",
                        "Qubit: 4    Tau: 1.0 [Z4 Z5]"],
"Tapering/n2_tapered": ["List of Tapered-off Qubits:  [0, 1, 2, 4, 6]",
                        "Qubit: 0    Tau: 1.0 [Z0 Z3 Z5 Z7 Z9 Z10]",
                        "Qubit: 1    Tau: 1.0 [Z1 Z3 Z5 Z7 Z9 Z11]",
                        "Qubit: 2    Tau: 1.0 [Z2 Z3 Z8 Z9]",
                        "Qubit: 4    Tau: 1.0 [Z4 Z5 Z8 Z9 Z10 Z11]",
                        "Qubit: 6    Tau: 1.0 [Z6 Z7 Z8 Z9 Z10 Z11]"]
}                        

QITE_test_list = {
"QITE/h4_cite": -2.033812415092,
"QITE/h4_uccsd": -2.033809570175,
"QITE/h4_ahva": -2.033812406994
}

MSQITE_test_list = {
"MSQITE/msqite_h4": [-1.915106549508, -1.900779501949, -1.764318305093, -1.708685492323],
"MSQITE/n2_tapered": [-107.51339479156, -107.294861989743, -107.236231484409],
#Final:  E[0-MSQITE] = -107.513393311705  (<S**2> = +0.00000)  E[1-MSQITE] = -107.294861068392  (<S**2> = +0.00000)  E[2-MSQITE] = -107.236227890797  (<S**2> = +0.00000)  
}

RDM_test_list = {
"RDM/uccd_LiH": [ -12.426206248579847, 3.5486270612648463, -7.881963096544056,
-3.87551, -1.80256, 2.16307, 4.79038,
-3.78583, -1.76085, 2.11302, 4.67952],
#1RDM energy: -12.426206248579847
#2RDM energy: 3.5486270612648463
#Total energy: -7.881963096544056
#Dipole moment from 1RDM (in Debye):
#x = -3.87551  y = -1.80256  z = 2.16307
#| mu | = 4.79038
#Dipole moment from relaxed 1RDM (in Debye):
#x = -3.78583     y = -1.76085    z = 2.11302
#| mu | = 4.67952
"RDM/uccsd_BH": [  -37.47189182033296, 10.514386916131325, -24.809870119623707,
0.61095, -0.00002, 0.00037, 0.61095,
0.61030, 0.00000, -0.00008, 0.61030],

#1RDM energy: -37.47189182033296
#2RDM energy: 10.514386916131325
#Total energy: -24.809870119623707
#Dipole moment from 1RDM (in Debye):
#x = 0.61095  y = -0.00002  z = 0.00037
#| mu | = 0.61095
#
#Dipole moment from relaxed 1RDM (in Debye):
#x = 0.61030     y = 0.00000    z = -0.00008
#| mu | = 0.61030

"RDM/adapt_LiH": [-12.426328435167845, 3.548875314817474, -7.881837029579426,
-3.87557, -1.80259, 2.16311, 4.79045,
-3.78723, -1.76146, 2.11375, 4.68122],
# Nuclear-replusion energy: 0.9956160907709449
#1RDM energy: -12.426328435167845
#2RDM energy: 3.548875314817474
#Total energy: -7.881837029579426
#
#Dipole moment from 1RDM (in Debye):
#x = -3.87557  y = -1.80259  z = 2.16311
#| mu | = 4.79045
#
#Dipole moment from relaxed 1RDM (in Debye):
#x = -3.78723     y = -1.76146    z = 2.11375
#| mu | = 4.68122

"RDM/adapt_BH_frozen":[-37.481663866759924, 10.524864668794804, -24.809164413387194,
0.76339, 0, 0, 0.76339, 
0.61564, 0, 0, 0.61564],
#Nuclear-replusion energy: 2.1476347845779227
#1RDM energy: -37.481663866759924
#2RDM energy: 10.524864668794804
#Total energy: -24.809164413387194

#
#Dipole moment from 1RDM (in Debye):
#x = 0.76339  y = 0.00000  z = 0.00000
#| mu | = 0.76339
#
#Dipole moment from relaxed 1RDM (in Debye):
#x = 0.61564     y = -0.00000    z = 0.00000
#| mu | = 0.61564
}

OPT_test_list={
        "OPT/h2o_ucc": -75.723436476109,
        "OPT/h2o_spadapt":-75.720952651959,
# Final: E[uccsd] = -75.719148265171  <S**2> = +0.000000001030918  rho = 1
# Final: E[uccsd] = -75.723381776564  <S**2> = +0.000000001267093  rho = 1
# Final: E[uccsd] = -75.723433897557  <S**2> = +0.000000000751822  rho = 1
# Final: E[uccsd] = -75.723436072380  <S**2> = +0.000000000135427  rho = 1
# Final: E[uccsd] = -75.723436476109  <S**2> = +0.000000000278127  rho = 1
# Final: E[adapt] = -75.713280689848   <S**2> = -0.000000000000001   Fidelity = 0.000000
# Final: E[adapt] = -75.709778058634   <S**2> = 0.000000000000000   Fidelity = 0.000000
# Final: E[adapt] = -75.720277278370   <S**2> = -0.000000000000001   Fidelity = 0.000000
# Final: E[adapt] = -75.720923353614   <S**2> = 0.000000000000000   Fidelity = 0.000000
# Final: E[adapt] = -75.720948207457   <S**2> = 0.000000000000000   Fidelity = 0.000000
# Final: E[adapt] = -75.720951651834   <S**2> = -0.000000000000002   Fidelity = 0.000000
# Final: E[adapt] = -75.720952651959   <S**2> = -0.000000000000000   Fidelity = 0.000000

        }
OO_test_list={
        "OO/h2o_oo-uccd":-76.07434929770854,
# Final: E[uccd] = -76.028435792671  <S**2> = +0.000000000707055  CNOT = 204  rho = 1
# Final: E[uccd] = -76.068512228342  <S**2> = +0.000000000137134  CNOT = 204  rho = 1
# Final: E[uccd] = -76.073209797554  <S**2> = +0.000000000064151  CNOT = 204  rho = 1
# Final: E[uccd] = -76.073554357816  <S**2> = +0.000000000029810  CNOT = 204  rho = 1
# Final: E[uccd] = -76.073591203711  <S**2> = +0.000000000160435  CNOT = 204  rho = 1
# Final: E[uccd] = -76.074007133981  <S**2> = +0.000000000095071  CNOT = 204  rho = 1
# Final: E[uccd] = -76.074325877000  <S**2> = +0.000000000027948  CNOT = 204  rho = 1
# Final: E[uccd] = -76.074347462424  <S**2> = +0.000000000022511  CNOT = 204  rho = 1
# Final: E[uccd] = -76.074349154614  <S**2> = +0.000000000085299  CNOT = 204  rho = 1
# Final: E[uccd] = -76.074349297709  <S**2> = +0.000000000080063  CNOT = 204  rho = 1
#  Final: E[oo-uccd] = -76.07434929770854         
        "OO/n2_oo-sauccgd": -109.119198740332,
# Final: E[sauccgd] = -109.041499611760  <S**2> = +0.000001521630632  CNOT = 1458  rho = 1
# Final: E[sauccgd] = -109.077746258204  <S**2> = +0.000002667379990  CNOT = 1458  rho = 1
# Final: E[sauccgd] = -109.087069446795  <S**2> = +0.000002560945739  CNOT = 1458  rho = 1
# Final: E[sauccgd] = -109.108842972358  <S**2> = +0.000001814886701  CNOT = 1458  rho = 1
# Final: E[sauccgd] = -109.118855898278  <S**2> = +0.000001651475217  CNOT = 1458  rho = 1
# Final: E[sauccgd] = -109.119155497479  <S**2> = +0.000001660834636  CNOT = 1458  rho = 1
# Final: E[sauccgd] = -109.119190441263  <S**2> = +0.000001664410005  CNOT = 1458  rho = 1
# Final: E[sauccgd] = -109.119194305167  <S**2> = +0.000001664856220  CNOT = 1458  rho = 1
# Final: E[sauccgd] = -109.119194768730  <S**2> = +0.000001666225513  CNOT = 1458  rho = 1
# Final: E[sauccgd] = -109.119194862831  <S**2> = +0.000001665673650  CNOT = 1458  rho = 1
# Final: E[sauccgd] = -109.119194931540  <S**2> = +0.000001669565327  CNOT = 1458  rho = 1
# Final: E[sauccgd] = -109.119195055070  <S**2> = +0.000001668147347  CNOT = 1458  rho = 1
#  Final: E[oo-sauccgd] = -109.11919505506967 
        }
User_defined_test_list = {
"User-defined/random_hamiltonian": -1.332963129119,
"User-defined/kitaev": -4.247573557412    #Final: E[hva] = -4.247573557412  <S**2> = +0.000000000000000  rho = 1 
}

import os
import re
import sys
import time
import sh
import glob
import numpy as np
from math import isclose
python=sys.executable
np.set_printoptions(precision=10, suppress=True)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=2000)
wildcard = "*"

print(f'Running test calculations.') 
print(f'Enter the number of MPI processes (Hit enter to skip)')
########
system=input()
while True:
    if isinstance(system, str):
        try:
            nprocs = int(system)
            break
        except:
            if system == '':
                nprocs = 1
                break
            else:
                print(f'Unrecognized nprocs = {system}')
                system=input()
    else:
        nprocs = 1
        break
print(f'Enter the number of threads (Hit enter to skip)')
########
system=input()
while True:
    if isinstance(system, str):
        try:
            nthreads = int(system)
            break
        except:
            if system == '':
                nthreads = 1
                break
            else:
                print(f'Unrecognized nthreads = {system}')
                system=input()
    else:
        nthreads = 1
        break
 

print(f'This may take several minutes.')

UCC     = False
PHF     = False
ADAPT   = False
TAPER   = False
RDM     = False
Excited = False
UpCCGSD = False
QITE    = False
MSQITE  = False
OPT     = False
OO      = False
USER    = False

len_argv = len(sys.argv)
if len_argv == 1:
    UCC     = True
    PHF     = True
    ADAPT   = True
    TAPER   = True
    RDM     = True
    Excited = True
    UpCCGSD = True
    QITE    = True
    MSQITE  = True
    OPT     = True
    OO      = True
    USER    = True
else:
    for opt in sys.argv:
        if opt.lower() == "ucc":
            UCC = True
        if opt.lower() == "phf":
            PHF = True
        if opt.lower() == "adapt":
            ADAPT = True
        if opt.lower() == "taper":
            TAPER = True
        if opt.lower() == "rdm":
            RDM = True
        if opt.lower() == "excited":
            Excited = True
        if opt.lower() == "upccgsd":
            UpCCGSD = True
        if opt.lower() == "qite":
            QITE = True
        if opt.lower() == "msqite":
            MSQITE = True
        if opt.lower() == "opt":
            OPT = True
        if opt.lower() == "oo":
            OO = True
        if opt.lower() == "user":
            USER = True
            

def Submit(jobname, nprocs=1, nthreads=1):
    """Function
    Submit job. Track the submitted job until its finished.
    """
    job = '.job'
    o = '.o'
    e = '.e'
    chk = '.chk'
    err = '.err'
    out = '.out'
    qkt = '.qkt'
    inp = '.inp'
    theta = '.theta'
    jobinp = f"{jobname+inp}"
    if not os.path.isfile(jobinp):
        print(f"{jobinp} not found!") 
        return

    job_script = f"{jobname+job}"
    ## Make job script
    with open(job_script, "w") as f:
        print("#!/bin/bash", file=f)
        print("#PBS -N ", jobname, file=f)
        print(f"#PBS -l select=1:ncpus={nthreads*nprocs}:mpiprocs={nprocs}", file=f)
        #if system=='2':
        #    print("source /etc/profile.d/modules.sh", file=f)
        #    print("module load compiler", file=f)
        #    print("module load mpi", file=f)
        print("", file=f)
        print("cd ${PBS_O_WORKDIR}", file=f)
        print("", file=f)
        print(f"export OMP_NUM_THREADS={nthreads}", file=f)
        print(f"mpirun -np {nprocs} {python} -m mpi4py {QDIR}/main.py {jobname} -nt {nthreads}", file=f)
        #print(f"mpirun -np {nprocs} -genv I_MPI_FABRICS tcp {python} -m mpi4py {QDIR}/main.py {jobname}", file=f)
    
      
    sh.chmod("755", job_script)
    job_id = sh.qsub(job_script)
    job_id = job_id.strip()
    
    qstat = True
    test = 0
    while qstat:
        log = sh.qstat().split()
        if job_id not in log:
            # Job done
            qstat = False
        time.sleep(1)
        print(".", end="", flush=True)
    sh.rm(sh.glob(f"{jobname+o+wildcard}"))
    sh.rm(sh.glob(f"{job_script}"))
    if os.path.isfile(f"{jobname+chk}"):
        sh.rm(sh.glob(f"{jobname+chk}"))
    if os.path.isfile(f"{jobname+qkt}"):
        sh.rm(sh.glob(f"{jobname+qkt}"))
    if os.path.isfile(f"{jobname+out}"):
        sh.rm(sh.glob(f"{jobname+out}"))

def run(sample):
    subdir = sample[:sample.find('/')]
    sample_name = sample[sample.find('/')+1:]
    log = ".log"
    sample_path = f"{SAMPLEDIR+subdir}"
    sample_log = f"{sample_name+log}"
    sh.cd(sample_path)
    Submit(sample_name, nprocs=nprocs, nthreads=nthreads)
    return sample_log

def delete_err(sample):
    subdir = sample[:sample.find('/')]
    sample_name = sample[sample.find('/')+1:]
    sample_path = f"{SAMPLEDIR+subdir}"
    sh.cd(sample_path)
    sh.rm(sh.glob(f"{sample_name}.e{wildcard}"))

############
#  UCC     #
############
if UCC:
    for sample, ref in UCC_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        flag = True
        results = None
        try:
            log = str(sh.grep("Final:", sample_log)).split("\n")
            results = float(log[0].split()[3])
            if isclose(ref, results, abs_tol=1e-6):
                pass
            else:
                flag = False
        except:
            flag = False
        if flag:    
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")


############
#  ADAPT   #
############
if ADAPT:
    for sample, ref in ADAPT_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        flag = True
        try:
            log = str(sh.grep("Final:", sample_log)).split("\n")
            results = float(log[0].split()[3])
            if isclose(ref, results, abs_tol=1e-6):
                pass
            else:
                flag = False
        except:
            flag = False
            results = None
        if flag:
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")


##########
#  PHF   #
##########
if PHF:
    for sample, ref in PHF_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        flag = True
        results = None
        try:
            log = str(sh.grep("Final:", sample_log)).split("\n")
            results = float(log[0].split()[3])
            if isclose(ref, results, abs_tol=1e-6):
                pass
            else:
                flag = False
        except:
            flag = False
        if flag:
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")

###############
#  TAPERING   #
###############
if TAPER:
    for sample, ref in Tapering_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        flag = True
        results = None
        try:
            results = str(sh.grep("-e", "List of Tapered-off Qubits:", "-e","Qubit:", sample_log)).split("\n")
            for i in range(len(ref)):
                if results[i] == ref[i]:
                    pass
                else:
                    flag = False
        except:
            flag = False
        if flag:    
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")


############
#  RDM     #
############
if RDM:
    for sample, ref in RDM_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        flag = True
        results = None
        try:
            log = str(sh.grep("-A", 2, "-e", "1RDM energy", 
                              "-A", 2, "-e", "Dipole moment from 1RDM",
                              "-A", 2, "-e", "Dipole moment from relaxed 1RDM",
                              sample_log)).split("\n")
            results = []                  
            # 1RDM, 2RDM, Total energies 
            for i in range(3):
                try:
                    results.append(float(log[i].split()[2]))
                    if i < 2 and isclose(results[i], ref[i], abs_tol=1e-3):
                        # 1-body energy and 2-body energy are loose
                        pass
                    elif i == 2 and isclose(results[i], ref[i], abs_tol=1e-6):
                        # Total energy is tight
                        pass
                    else:
                        flag = False
                except:
                    flag = False
            # Dipole moments
            try:
                for i in range(3):
                    results.append(float(log[5].split()[2+i*3]))
                results.append(float(log[6].split()[4]))
                for i in range(3):
                    results.append(float(log[9].split()[2+i*3]))
                results.append(float(log[10].split()[4]))
                for i in range(8):
                    if isclose(results[i+3], ref[i+3], abs_tol=1e-3):
                        pass
                    else:
                        flag = False
            except:
                flag = False
        except:
            flag = False
        if flag:    
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")
    
##############
#  Excited   #
##############
if Excited:
    for sample, ref in Excited_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        CDIR = os.getcwd()
        flag = True
        log = str(sh.grep("Final:", sample_log)).split("\n")
        results = None
        try:
            results = []
            for _log in log:
                log_list = _log.split()
                if len(log_list) < 6:
                    continue
                try:
                    results.append(float(log_list[3]))
                    results.append(float(log_list[6]))
                except:
                    flag = False
            for i in range(len(results)): 
                if i%2 == 0 and isclose(ref[i], results[i], abs_tol=1e-6):
                    # Energy criteria
                    pass
                elif i%2 == 1 and isclose(ref[i], results[i], abs_tol=1e-4):
                    # S2 criteria
                    pass
                else:
                    flag = False
        except:
            flag = False
        if flag:    
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")
    
    
################
#  k-UpCCGSD   #
################
if UpCCGSD:
    for sample, ref in UpCCGSD_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        flag = True
        try:
            log = str(sh.grep("Final:", sample_log)).split("\n")
            results = float(log[0].split()[3])
            if isclose(ref, results, abs_tol=1e-6):
                pass
            else:
                flag = False
        except:
            flag = False
            results = None
        if flag:    
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")


############
#  QITE    #
############
if QITE:
    for sample, ref in QITE_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        flag = True
        results = None
        try:
            log = str(sh.grep("Final:", sample_log)).split("\n")
            results = float(log[0].split()[3])
            if isclose(ref, results, abs_tol=1e-6):
                pass
            else:
                flag = False
        except:
            flag = False
        if flag:    
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")



############
#  MSQITE  #
############
if MSQITE:
    for sample, ref in MSQITE_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        flag = True
        results = None
        try:
            log = str(sh.grep("Final:", '-A 10', sample_log)).split()
            results = []
            for i in range(len(log)):
                if "MSQITE" in log[i]:
                    results.append(float(log[i+2]))
            for correct, calc in zip(ref, results):
                try:
                    if isclose(correct, calc, abs_tol=1e-6):
                        pass
                    else:
                        flag = False
                except:
                    flag = False
        except:
            flag = False
        if flag:    
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")

############
#  OPT     #
############
if OPT:
    for sample, ref in OPT_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        flag = True
        results = None
        try:
            log = str(sh.grep("Final:", sample_log)).split("\n")
            results = float(log[-2].split()[3])
            if isclose(ref, results, abs_tol=1e-6):
                pass
            else:
                flag = False
        except:
            flag = False
        if flag:    
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")

############
#  OO      #
############
if OO:
    for sample, ref in OO_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        flag = True
        results = None
        try:
            log = str(sh.grep("Final:", sample_log)).split("\n")
            results = float(log[-2].split()[3])
            if isclose(ref, results, abs_tol=1e-6):
                pass
            else:
                flag = False
        except:
            flag = False
        if flag:    
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")

##################
#  USER-DEFINED  #
##################
if USER:
    for sample, ref in User_defined_test_list.items():
        print(sample,"", end="", flush=True)
        sample_log = run(sample)
        flag = True
        results = None
        try:
            log = str(sh.grep("Final:", sample_log)).split("\n")
            results = float(log[-2].split()[3])
            if isclose(ref, results, abs_tol=1e-4):
                pass
            else:
                flag = False
        except:
            flag = False
        if flag:    
            print(" passed")
            delete_err(sample)
        else: 
            print(f" failed:\n Reference = {ref}   \n This code = {results}")

