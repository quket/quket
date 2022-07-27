#!/usr/local/bin/python3.8
import os
"""gradient.py
This test code gets nulcear gradients by computing 
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

        e.g.)   python3.8 gradient.py uccsd_LiH > grad &

    Then, this program will create ***_geom.inp file and 
    submit to perform several jobs. Computing NBasis*(NBasis+1)/2
    jobs may take some time, but if you already have ***_geom.log 
    finished in your directory, you can bypass the re-calculation
    by setting the option "bypass".
"""

### CHANGE ACCORDINGLY ###
ang = 1/1.8897259886 # Bohr
delta = 0.001 
nprocs = 16
nthread = 2
#QDIR = "/home/tsuchimochi/Quket/master"
QDIR = os.getcwd()
bypass = False



### YOU MAY MODIFY BELOW TO GET WHAT YOU WANT ###

import re
import sys
import time
import sh
import glob
import numpy as np
np.set_printoptions(precision=10, suppress=True)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=2000)
python = sh.which('python3.8') 

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
        print("mpirun -np {} -genv I_MPI_FABRICS tcp {} -m mpi4py {}/main.py {} -nt {}".format(nprocs, python, QDIR, jobname, nthread), file=f)
    
    
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
geom = "_geom"
tmp = "tmpfile"
job = ".job"
log = ".log"
test_job = f"{input_name+test}" 
geom_job = f"{input_name+geom}"
geom_job_path = f"{input_dir}/{input_name+geom+ext}"
geom_job_log = f"{input_dir}/{input_name+geom+log}"
test_job_path = f"{input_dir}/{test_job+ext}"
test_job_log = f"{input_dir}/{test_job+log}"
tmpfile = f"{input_dir}/{tmp}"


try:
    if sys.argv[2] == 'bypass':
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
### Check geometry
for i, iline in enumerate(lines):
    if 'geometry' in iline.replace(' ','').lower():
        geometry = []
        for j in range(i+1, len(lines)):
            next_line = get_line(lines[j])
            if not isinstance(next_line, str):
                break
            atom_info = next_line.split(" ")
            if len(atom_info) != 4:
                break
            atom = atom_info[0]
            xyz = tuple(map(float, atom_info[1:4]))
            geometry.append((atom, xyz))
    def get_line(line):
        # Removal of certain symbols and convert multi-space to single-space.
        remstr = ":,'()"
        line = line.translate(str.maketrans(remstr, " "*len(remstr), ""))
        line = re.sub(r" +", r" ", line).rstrip("\n").strip()
        if len(line) == 0:
            # Ignore blank line.
            return
        if line[0] in ["!", "#"]:
            # Ignore comment line.
            return

        if "=" not in line:
            return line

        key, value = line.split("=",1)
        key = key.strip().lower()
        value = value.strip()
        value = value.split(" ")
        if len(value) == 1:
            return key, value[0]
        else:
            return key, value
###
natm = len(geometry)

wildcard = "*"

if bypass:
    print(f"Will bypass computing {geom_job}")
else:
    for p in range(natm):
        for xyz in ['x','y','z']:
            strpxyz = str(p)+ xyz

            for s in range(2):
                if s == 0:
                    sign = '+'
                else:
                    sign = '-'
                geom_job = f"{input_name+geom+strpxyz+sign}"
                geom_job_path = f"{input_dir}/{input_name+geom+strpxyz+sign+ext}"
                with open(geom_job_path, "w") as f:
                    for i in lines:
                        print(i.replace('\n',''), file=f)
                    print('theta_guess = zero', file=f)
                    print('kappa_guess = zero', file=f)
                    print('debug = False', file=f)
                    print('1RDM = False', file=f)
                    print('ftol = 1e-14', file=f)
                    print('gtol = 1e-6', file=f)
                    print('geometry', file=f)
                    if sign == '+':
                        for i, igeom in enumerate(geometry):
                            if i == p:
                                x = igeom[1][0]
                                y = igeom[1][1]
                                z = igeom[1][2]
                                if xyz == 'x':
                                    x += delta * ang
                                elif xyz == 'y':
                                    y += delta * ang
                                elif xyz == 'z':
                                    z += delta * ang
                                print(igeom[0],'  ', x, y, z, file=f) 
                            else:
                                print(igeom[0],'  ', igeom[1][0], igeom[1][1], igeom[1][2], file=f)
                    elif sign == '-':
                        for i, igeom in enumerate(geometry):
                            if i == p:
                                x = igeom[1][0]
                                y = igeom[1][1]
                                z = igeom[1][2]
                                if xyz == 'x':
                                    x -= delta * ang
                                elif xyz == 'y':
                                    y -= delta * ang
                                elif xyz == 'z':
                                    z -= delta * ang
                                print(igeom[0],'  ', x, y, z, file=f) 
                            else:
                                print(igeom[0],'  ', igeom[1][0], igeom[1][1], igeom[1][2], file=f)
               
                print(geom_job)
                Submit(geom_job, nprocs=nprocs)
                geom_job_log = f"{input_dir}/{input_name+geom+strpxyz+sign+log}"
                sh.cp(geom_job_log, tmpfile) 
                sh.rm(sh.glob(f"{geom_job+wildcard}"))
                sh.cp(tmpfile, geom_job_log) 
                sh.rm(sh.glob(tmpfile))
    print('end of job: now compute relax density matrix')


grad = np.zeros((natm, 3))
for p in range(natm):
    ixyz = 0
    for ixyz, xyz in enumerate(['x','y','z']):
        strpxyz = str(p)+ xyz
        geom_job_log = f"{input_name+geom+strpxyz+'+'+log}"
        final = str(sh.grep("Final:", geom_job_log)).split("\n")
        Ep = float(final[0].split()[3])
        geom_job_log = f"{input_name+geom+strpxyz+'-'+log}"
        final = str(sh.grep("Final:", geom_job_log)).split("\n")
        Em = float(final[0].split()[3])
        print(f"{p} {xyz}  {Ep}  {Em}")
        grad[p,ixyz] = (Ep - Em) / (2*delta)
        ixyz += 1

print("Nuclear Gradients\n",grad)
