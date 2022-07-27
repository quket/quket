#!/usr/local/bin/python3.8
"""relax.py
This test code gets Relax density matrix (1-body) by computing 
expectation values of perturbed Hamiltonians.

This program relies on the Quket log format where
the final converged energy appears as

Final: E[uccsd] = -7.882391426828  ... 
                  ~~~~~~~~~~~~~~~
                  
It also requires a modified version of PySCF 
(installed only on Ravel) to add perturbation 
to a certain one-body term in the atomic orbital basis. 

HOW TO USE:
    Simply run this code with a Quket input.

        e.g.)   python3.8 relax.py uccsd_LiH > rdm &

    Then, this program will create ***_dm.inp file and 
    submit to perform several jobs. Computing NBasis*(NBasis+1)/2
    jobs may take some time, but if you already have ***_dm.log 
    finished in your directory, you can bypass the re-calculation
    by setting the option "bypass".
"""

### CHANGE ACCORDINGLY ###
delta = 0.0000001
nprocs = 28
nthread = 2
QDIR = "/home/tsuchimochi/Quket/master"
bypass = False



### YOU MAY MODIFY BELOW TO GET WHAT YOU WANT ###

import os
import re
import sys
import time
import sh
import glob
import numpy as np
np.set_printoptions(precision=10, suppress=True)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=2000)

def Submit(jobname, nprocs=1, nthread=1):
    """Function
    Submit job. Track the submitted job until its finished.
    """
    job = '.job'
    job_script = f"{input_dir}/{jobname+job}"
    ### Make job script
    with open(job_script, "w") as f:
        print("#!/bin/bash", file=f)
        print("#PBS -N ", jobname, file=f)
        print("#PBS -l select=1:ncpus={}:mpiprocs={}".format(nprocs, nprocs), file=f)
        print("", file=f)
        print("cd ${PBS_O_WORKDIR}", file=f)
        print("", file=f)
        print("export OMP_NUM_THREADS={}".format(nthread), file=f)
        print("mpirun -np {} -genv I_MPI_FABRICS tcp /usr/local/bin/python3.8 -m mpi4py {}/main.py {} -nt {}".format(nprocs, QDIR, jobname, nthread), file=f)
    
    
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

def ReadMatrix(NDim, Mat_name, filename):
    """Function
    Get Matrix(NDim, NDim) from file.
    """
    Nlines = ((NDim//10) + 1) * (NDim + 2)
    log = str(sh.grep("-A", Nlines, Mat_name, filename)).split("\n")
    Mat = np.zeros((NDim, NDim), dtype=float)
    
    for p in range(NDim):
        for qq in range(NDim//10):
            row = (NDim+2)*qq + 3
            log_line = log[p+row].split()
            for q in range(10):
    #            print(log_line)
    #            print('({}, {}) = {}'.format(p,qq*10+q,log_line[q+1]))
                Mat[p, qq*10 + q] = float(log_line[q+1])
        qq = NDim//10       
        row = (NDim+2)*qq + 3
        log_line = log[p+row].split()
        for q in range(NDim%10):
            #print(log_line)
    #        print('({}, {}) = {}'.format(p,q+qq*10,log_line[q+1]))
            Mat[p, qq*10 + q] = float(log_line[q+1])
    return Mat    



################################################################
#                   Setting for input and output               #
################################################################
len_argv = len(sys.argv)
if len_argv == 1:
    print("Error! No input loaded.")
    exit()

# First argument = Input file,  either "***.inp" or "***" is allowed.
input_file=sys.argv[1]
input_dir, base_name = os.path.split(input_file)
input_dir = "." if input_dir == "" else input_dir
input_name, ext = os.path.splitext(base_name)

if ext == "":
    ext = ".inp"
elif ext not in ".inp":
    input_name += ext
    ext = ".inp"
input_file = f"{input_dir}/{input_name+ext}"
test = "_test"
dm = "_dm"
tmp = "tmpfile"
job = ".job"
log = ".log"
test_job = f"{input_name+test}" 
dm_job = f"{input_name+dm}"
dm_job_path = f"{input_dir}/{input_name+dm+ext}"
dm_job_log = f"{input_dir}/{input_name+dm+log}"
test_job_path = f"{input_dir}/{test_job+ext}"
test_job_log = f"{input_dir}/{test_job+log}"
tmpfile = f"{input_dir}/{tmp}"


try:
    if sys.argv[2] == 'bypass' and os.path.isfile(test_job_log):
        bypass = True
except:
    bypass = False


# Prepare input files and run calculations... 
### Get Input file
num_lines = sum(
    1 for line in open(input_file)
)  # number of lines in the input file
f = open(input_file)
lines = f.readlines()

if bypass:
    pass
else:
    print('Running test job to get NBasis')
    with open(test_job_path, "w") as f:
        for i in lines:
            print(i.replace('\n',''), file=f)
        print('MaxIter = 0', file=f)
        print('debug = True', file=f)
        #print('pyscf_guess = minao', file=f)
        print('pyscf_guess = read', file=f)
        print('theta_guess = zero', file=f)
        print('kappa_guess = zero', file=f)
    Submit(test_job, nprocs=nprocs, nthread=nthread)
    print('Test job Done')

### Open test.log and get NBasis
num_lines = sum(
    1 for line in open(test_job_log)
)  # number of lines in the input file
f = open(test_job_log)
test_lines = f.readlines()
iline = 0
NBasis = None
nf = None
nc = None
nact = None
while iline < num_lines or NBasis is None or nf is None or nc is None or nact is None:
    line = test_lines[iline].replace("=", " ")
    print(line)
    words = [x.strip() for x in line.split() if not line.strip() == ""]
    print('word ',words)
    len_words = len(words)
    if len_words > 0 and words[0][0] not in {"!", "#"}:
        if words[0].lower() == "nbasis":
            NBasis = int(words[1])
        elif words[0].lower() == "n_frozen_orbitals":
            nf = int(words[1])
        elif words[0].lower() == "n_core_orbitals":
            nc = int(words[1])
        elif words[0].lower() == "n_active_orbitals":
            nact = int(words[1])
    iline += 1

### Get MO_coeff and Overlap matrices
MO_coeff = ReadMatrix(NBasis, "MO coeff", test_job_log)
Overlap = ReadMatrix(NBasis, "Overlap", test_job_log)
print("MO_coeff = \n",MO_coeff)
print("Overlap = \n",Overlap)

MO_coeff_act = MO_coeff[:,(nf+nc):(nf+nc+nact)] 
print("MO_coeff = \n",MO_coeff_act)


#Delete files...
wildcard = "*"
sh.cp(test_job_log, tmpfile) 
sh.rm(sh.glob(f"{test_job+wildcard}"))
sh.cp(tmpfile, test_job_log) 
sh.rm(sh.glob(tmpfile))

print("NBasis = ",NBasis)
print('Creating {}'.format(dm_job_path))

if bypass:
    print(f"Will bypass computing {dm_job}")
else:
    pq = 0
    for p in range(NBasis):
        for q in range(p+1):
            for s in range(2):
                if s == 0:
                    sign = '+'
                else:
                    sign = '-'
                str_pq = str(pq)
                dm_job = f"{input_name+dm+sign+str_pq}"
                dm_job_path = f"{input_dir}/{input_name+dm+sign+str_pq+ext}"
                with open(dm_job_path, "w") as f:
                    for i in lines:
                        print(i.replace('\n',''), file=f)
                    print('theta_guess = zero', file=f)
                    print('kappa_guess = zero', file=f)
                    print('IntPQ = 0 0 ', delta, file=f)
                    print('debug = False', file=f)
                    print('1RDM = False', file=f)
                    print('ftol = 1e-14', file=f)
                    print('gtol = 1e-6', file=f)
                    if sign == '+':
                        print('IntPQ = {} {} {} '.format(p, q, delta), file=f)
                    elif sign == '-':
                        print('IntPQ = {} {} {} '.format(p, q, -delta), file=f)
               
                Submit(dm_job, nprocs=nprocs)
                dm_job_log = f"{input_dir}/{input_name+dm+sign+str_pq+log}"
                sh.cp(dm_job_log, tmpfile) 
                sh.rm(sh.glob(f"{dm_job+wildcard}"))
                sh.cp(tmpfile, dm_job_log) 
                sh.rm(sh.glob(tmpfile))
            pq += 1
    print('end of job: now compute relax density matrix')

DM_AO = np.zeros((NBasis, NBasis), dtype=float)
pq = 0
for p in range(NBasis):
    for q in range(p+1):
        str_pq = str(pq)
        dm_job_log = f"{input_name+dm+'+'+str_pq+log}"
        final = str(sh.grep("Final:", dm_job_log)).split("\n")
        Ep = float(final[0].split()[3])
        dm_job_log = f"{input_name+dm+'-'+str_pq+log}"
        final = str(sh.grep("Final:", dm_job_log)).split("\n")
        Em = float(final[0].split()[3])
        print(f"{p} {q}  {Ep}  {Em}")
        DM_AO[p,q] = (Ep - Em) / (4*delta)
        DM_AO[q,p] = DM_AO[p,q]
        pq += 1

print("Relaxed DM in AO (alpha+beta) \n",DM_AO)

### Compute DM in MO 
DM_MO_act =  MO_coeff_act.T @ Overlap @ DM_AO @ Overlap @ MO_coeff_act

print("Relaxed DM in MO (alpha+beta): active\n",DM_MO_act)
print("Relaxed DM in MO alpha: active\n",DM_MO_act/2)

### Compute DM in MO 
DM_MO =  MO_coeff.T @ Overlap @ DM_AO @ Overlap @ MO_coeff
print("Relaxed DM in MO (alpha+beta): whole\n",DM_MO)
print("Relaxed DM in MO alpha: whole\n",DM_MO/2)
