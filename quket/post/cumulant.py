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

cumulant.py

Cumulant-related subroutines.

"""
import numpy as np
import scipy as sp
import time
import copy
import math

from qulacs.state import inner_product

from quket.mpilib import mpilib as mpi

from quket.fileio import prints, print_state, error, printmat, printmath
from quket.opelib import FermionOperator_from_list
from quket.lib import FermionOperator, normal_ordered

def Decompose_3body_CU(HA, D):
    ### 3-particle operator is cummlant-decomposed
    Hn1 = 0
    for op, coef in HA.terms.items():
        if len(op) == 6:
            # 3-body term.
            p = op[0][0]
            q = op[1][0]
            r = op[2][0]
            u = op[3][0]
            t = op[4][0]
            s = op[5][0]
            Dup = D[u,p]
            Dtq = D[t,q]
            Dsr = D[s,r]
            Dsp = D[s,p]
            Duq = D[u,q]
            Dtr = D[t,r]
            Dtp = D[t,p]
            Dsq = D[s,q]
            Dur = D[u,r]
            op_cum =   Dsp * FermionOperator_from_list([q,r],[u,t]) \
                     - Dsq * FermionOperator_from_list([p,r],[u,t]) \
                     - Dsr * FermionOperator_from_list([q,p],[u,t]) \
                     - Dtp * FermionOperator_from_list([q,r],[u,s]) \
                     + Dtq * FermionOperator_from_list([p,r],[u,s]) \
                     + Dtr * FermionOperator_from_list([q,p],[u,s]) \
                     - Dup * FermionOperator_from_list([q,r],[s,t]) \
                     + Duq * FermionOperator_from_list([p,r],[s,t]) \
                     + Dur * FermionOperator_from_list([q,p],[s,t]) \
                     - 2/3 * (Dsp * Dtq - Dtp * Dsq) * FermionOperator_from_list([r],[u]) \
                     + 2/3 * (Dsr * Dtq - Dsq * Dtr) * FermionOperator_from_list([p],[u]) \
                     + 2/3 * (Dsp * Dtr - Dtp * Dsr) * FermionOperator_from_list([q],[u]) \
                     + 2/3 * (Dsp * Duq - Dup * Dsq) * FermionOperator_from_list([r],[t]) \
                     - 2/3 * (Dsr * Duq - Dsq * Dur) * FermionOperator_from_list([p],[t]) \
                     - 2/3 * (Dsp * Dur - Dup * Dsr) * FermionOperator_from_list([q],[t]) \
                     + 2/3 * (Dup * Dtq - Dtp * Duq) * FermionOperator_from_list([r],[s]) \
                     - 2/3 * (Dur * Dtq - Duq * Dtr) * FermionOperator_from_list([p],[s]) \
                     - 2/3 * (Dup * Dtr - Dtp * Dur) * FermionOperator_from_list([q],[s])

            Hn1 += coef * op_cum
        else:
            Hn1 += FermionOperator(op, coef)
    if Hn1 != 0:
        Hn1 = normal_ordered(Hn1)
    return Hn1

def Decompose_3body_MK(HA, D1, D2, cum_list=None):
    ### 3-particle operator is cummlant-decomposed
    #
    #  p^ q^ r^ u t s =  +Dsp [q^ r^ u t]
    #                    -Dsq [p^ r^ u t]
    #                    -Dsr [q^ p^ u t]
    #                    -Dtp [q^ r^ u s]
    #                    +Dtq [p^ r^ u s]
    #                    +Dtr [q^ p^ u s]
    #                    -Dup [q^ r^ s t]
    #                    +Duq [p^ r^ s t]
    #                    +Dur [q^ p^ s t]
    #                    +Xpqst [r^ u]
    #                    -Xrqst [p^ u]
    #                    -Xprst [q^ u]
    #                    -Xpqut [r^ s]
    #                    +Xrqut [p^ s]
    #                    +Xprut [q^ s]
    #                    -Xpqsu [r^ t]
    #                    +Xrqsu [p^ t]
    #                    +Xprsu [q^ t]
    #                    -Ypqst Dur
    #                    +Yrqst Dup
    #                    +Yprst Duq
    #                    +Ypqut Dsr
    #                    -Yrqut Dsp
    #                    -Yprut Dsq
    #                    +Ypqsu Dtr
    #                    -Yrqsu Dtp
    #                    -Yprsu Dtq
    #
    # where
    #
    #   Xpqst  = D2[p,q,t,s] - 2 (D1[p,s] D1[q,t] - D1[p,t] D1[q,s])
    #   Ypqst = D2[p,q,t,s] - 4/3 (D1[p,s] D1[q,t] - D1[p,t] D1[q,s])
    #

    Hn1 = 0
    for op, coef in HA.terms.items():
        if len(op) > 6:
            if abs(coef) > 1e-6:
                raise ValueError(f'Operator {op} has too many rank.')
            op_cum = 0
        elif len(op) == 6:
            if abs(coef) < 1e-7:
                continue
            # 3-body term.
            p = op[0][0]
            q = op[1][0]
            r = op[2][0]
            u = op[3][0]
            t = op[4][0]
            s = op[5][0]
            if cum_list is None:
                Dsp = D1[p,s]
                Dsq = D1[q,s]
                Dsr = D1[r,s]
                Dtp = D1[p,t]
                Dtq = D1[q,t]
                Dtr = D1[r,t]
                Dup = D1[p,u]
                Duq = D1[q,u]
                Dur = D1[r,u]
                Xpqst = D2[p,q,t,s]  - 2 * (Dsp * Dtq - Dsq * Dtp)
                Xrqst = D2[r,q,t,s]  - 2 * (Dsr * Dtq - Dsq * Dtr)
                Xprst = D2[p,r,t,s]  - 2 * (Dsp * Dtr - Dsr * Dtp)
                Xpqut = D2[p,q,t,u]  - 2 * (Dup * Dtq - Duq * Dtp)
                Xrqut = D2[r,q,t,u]  - 2 * (Dur * Dtq - Duq * Dtr)
                Xprut = D2[p,r,t,u]  - 2 * (Dup * Dtr - Dur * Dtp)
                Xpqsu = D2[p,q,u,s]  - 2 * (Dsp * Duq - Dsq * Dup)
                Xrqsu = D2[r,q,u,s]  - 2 * (Dsr * Duq - Dsq * Dur)
                Xprsu = D2[p,r,u,s]  - 2 * (Dsp * Dur - Dsr * Dup)
                Ypqst = D2[p,q,t,s]  - 4/3 * (Dsp * Dtq - Dsq * Dtp)
                Yrqst = D2[r,q,t,s]  - 4/3 * (Dsr * Dtq - Dsq * Dtr)
                Yprst = D2[p,r,t,s]  - 4/3 * (Dsp * Dtr - Dsr * Dtp)
                Ypqut = D2[p,q,t,u]  - 4/3 * (Dup * Dtq - Duq * Dtp)
                Yrqut = D2[r,q,t,u]  - 4/3 * (Dur * Dtq - Duq * Dtr)
                Yprut = D2[p,r,t,u]  - 4/3 * (Dup * Dtr - Dur * Dtp)
                Ypqsu = D2[p,q,u,s]  - 4/3 * (Dsp * Duq - Dsq * Dup)
                Yrqsu = D2[r,q,u,s]  - 4/3 * (Dsr * Duq - Dsq * Dur)
                Yprsu = D2[p,r,u,s]  - 4/3 * (Dsp * Dur - Dsr * Dup)

                op_cum =   Dsp * FermionOperator_from_list([q,r],[u,t]) \
                         - Dsq * FermionOperator_from_list([p,r],[u,t]) \
                         - Dsr * FermionOperator_from_list([q,p],[u,t]) \
                         - Dtp * FermionOperator_from_list([q,r],[u,s]) \
                         + Dtq * FermionOperator_from_list([p,r],[u,s]) \
                         + Dtr * FermionOperator_from_list([q,p],[u,s]) \
                         - Dup * FermionOperator_from_list([q,r],[s,t]) \
                         + Duq * FermionOperator_from_list([p,r],[s,t]) \
                         + Dur * FermionOperator_from_list([q,p],[s,t]) \
                         + Xpqst * FermionOperator_from_list([r],[u]) \
                         - Xrqst * FermionOperator_from_list([p],[u]) \
                         - Xprst * FermionOperator_from_list([q],[u]) \
                         - Xpqut * FermionOperator_from_list([r],[s]) \
                         + Xrqut * FermionOperator_from_list([p],[s]) \
                         + Xprut * FermionOperator_from_list([q],[s]) \
                         - Xpqsu * FermionOperator_from_list([r],[t]) \
                         + Xrqsu * FermionOperator_from_list([p],[t]) \
                         + Xprsu * FermionOperator_from_list([q],[t]) \
                         - Ypqst * Dur \
                         + Yrqst * Dup \
                         + Yprst * Duq \
                         + Ypqut * Dsr \
                         - Yrqut * Dsp \
                         - Yprut * Dsq \
                         + Ypqsu * Dtr \
                         - Yrqsu * Dtp \
                         - Yprsu * Dtq
            else:
                n = D1.shape[0]    
                pqr = p * (p-1) * (p-2) // 6 + q * (q-1) // 2 + r 
                stu = u * (u-1) * (u-2) // 6 + t * (t-1) // 2 + s 
                op_cum = cum_list[pqr][stu]
                #prints(f"'{p}^ {q}^ {r}^ {u} {t} {s}'  -->  {op_cum}")

            Hn1 += coef * op_cum
        else:
            Hn1 += FermionOperator(op, coef)
    if Hn1 != 0:
        Hn1 = normal_ordered(Hn1)
    return Hn1

def store_Decomposed_3body(D1, D2, method='MK'):
    """
    Pre-compute the cumulant approximation
       [p^ q^ r^ u t s]  ~  Dup [q^ r^ t s] - ...
    for all possible combinations, for the latter use.
    """

    n_qubits = D1.shape[0]    
    cum_index = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits), dtype=int)
    my_cum_index = np.ones((n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits), dtype=int)
    my_cum_index *= -1
    norbs = n_qubits//2

    ### Count the number of spin-allowed 3-body operators
    NAAA = norbs * (norbs-1) * (norbs-2) // (3*2) 
    NBAA = norbs * norbs * (norbs-1) // 2 
    NBBB = NAAA
    NBBA = NBAA
    Npqruts =  NAAA * NAAA \
             + NBAA * NBAA \
             + NBBA * NBBA \
             + NBBB * NBBB

    ### Perform cumulant decomposition
    Ndim = n_qubits * (n_qubits-1) * (n_qubits-2) // (3*2)
    pqruts = -1
    ipos, my_ndim = mpi.myrange(Ndim*Ndim)
    my_cum_list = []
    pqr = -1
    for p in range(n_qubits):
        for q in range(p):
            for r in range(q):
                pqr += 1
                uts = -1
                for s in range(n_qubits):
                    for t in range(s):
                        for u in range(t):
                            uts += 1
                            pqruts += 1
                            if pqruts >= ipos and pqruts < ipos+my_ndim:
                                if (p%2+q%2+r%2)  != (s%2+t%2+u%2):
                                    op_pqruts = 0
                                else:
                                    op_pqruts = FermionOperator_from_list([p,q,r],[s,t,u])
                                    if method == 'MK':
                                        op_pqruts = Decompose_3body_MK(op_pqruts, D1, D2)
                                    elif method == 'CU':
                                        op_pqruts = Decompose_3body_CU(op_pqruts, D1)
                                my_cum_list.append(op_pqruts)
    data = mpi.gather(my_cum_list, root=0)
    if mpi.rank == 0:
        cum_list = [x for l in data for x in l]
    else:
        cum_list = None
    cum_list = mpi.bcast(cum_list, root=0)
    ### Convert to 2d list
    cum_2d_list = []
    for pqr in range(Ndim):
        cum_2d_list.append([])
        for stu in range(Ndim):
            cum_2d_list[pqr].append(cum_list[pqr*Ndim + stu])

    return cum_2d_list


def cumulant_3RDM(D1, D2):
    """
    Form cumulant 3-body density matrix using 1-body and 2-body density matrices, D1 and D2.
    This is an inefficient way of doing things, but allows for easy-coding.
    """

    n_qubits = D1.shape[0]    

    ### First, let's get X2 = D2 - 2/3 D1.D1
    
    X2 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits), dtype=float)
    my_X2 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits), dtype=float)
    pqst = 0
    pq = -1
    for p in range(n_qubits):
        for q in range(p):
            pq += 1
            st = -1
            for s in range(n_qubits):
                for t in range(s):
                    st += 1
                    if pq < st or (p%2+q%2)  != (s%2+t%2):
                        continue
                        pqst += 1
                        if pqst % mpi.nprocs == mpi.rank:
                            Dsp = D1[p,s]
                            Dsq = D1[q,s]
                            Dtp = D1[p,t]
                            Dtq = D1[q,t]
                            val = D2[p,q,t,s]  - 2/3 * (Dsp * Dtq - Dsq * Dtp)
                            my_X2[p,q,t,s] = val
                            my_X2[q,p,t,s] = -val
                            my_X2[p,q,s,t] = -val
                            my_X2[q,p,s,t] = val
                            my_X2[t,s,p,q] = val
                            my_X2[t,s,q,p] = -val
                            my_X2[s,t,p,q] = -val
                            my_X2[s,t,q,p] = val

    X2 = mpi.allreduce(my_X2, mpi.MPI.SUM)

    D3 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits), dtype=float)
    my_D3 = np.ones((n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits), dtype=float)
    ndim = n_qubits * (n_qubits - 1) * (n_qubits - 2) // 6
    ndim2 = ndim * (ndim + 1) // 2
    pqruts = 0 
    pqr = -1
    for p in range(n_qubits):
        for q in range(p):
            for r in range(q):
                pqr += 1
                uts = -1
                for s in range(n_qubits):
                    for t in range(s):
                        for u in range(t):
                            uts += 1
                            if pqr < uts or (p%2+q%2+r%2)  != (s%2+t%2+u%2):
                                continue

                            pqruts += 1
                            if pqruts % mpi.nprocs == mpi.rank:
                                val = X2[p,q,t,s] * D1[u,r] \
                                     -X2[r,q,t,s] * D1[u,p] \
                                     -X2[p,r,t,s] * D1[u,q] \
                                     -X2[p,q,t,u] * D1[s,r] \
                                     +X2[r,q,t,u] * D1[s,p] \
                                     +X2[p,r,t,u] * D1[s,q] \
                                     -X2[p,q,u,s] * D1[t,r] \
                                     +X2[r,q,u,s] * D1[t,p] \
                                     +X2[p,r,u,s] * D1[t,q] 
####
# put the element in D3 and anti-symmetrize!
####
                                # pqr -> pqr (+)
                                my_D3[p,q,r,u,t,s] = val  # uts -> uts (+)
                                my_D3[p,q,r,u,s,t] =-val  # uts -> ust (-)
                                my_D3[p,q,r,s,u,t] = val  # uts -> sut (+)
                                my_D3[p,q,r,t,u,s] =-val  # uts -> tus (-)
                                my_D3[p,q,r,t,s,u] = val  # uts -> tsu (+)
                                my_D3[p,q,r,s,t,u] =-val  # uts -> stu (-)

                                # pqr -> prq (-)
                                my_D3[p,r,q,u,t,s] =-val  # uts -> uts (+) 
                                my_D3[p,r,q,u,s,t] = val  # uts -> ust (-) 
                                my_D3[p,r,q,s,u,t] =-val  # uts -> sut (+) 
                                my_D3[p,r,q,t,u,s] = val  # uts -> tus (-) 
                                my_D3[p,r,q,t,s,u] =-val  # uts -> tsu (+) 
                                my_D3[p,r,q,s,t,u] = val  # uts -> stu (-) 

                                # pqr -> qrp (+)
                                my_D3[q,r,p,u,t,s] = val  # uts -> uts (+) 
                                my_D3[q,r,p,u,s,t] =-val  # uts -> ust (-) 
                                my_D3[q,r,p,s,u,t] = val  # uts -> sut (+) 
                                my_D3[q,r,p,t,u,s] =-val  # uts -> tus (-) 
                                my_D3[q,r,p,t,s,u] = val  # uts -> tsu (+) 
                                my_D3[q,r,p,s,t,u] =-val  # uts -> stu (-) 

                                # pqr -> qpr (-) 
                                my_D3[q,p,r,u,t,s] =-val  # uts -> uts (+) 
                                my_D3[q,p,r,u,s,t] = val  # uts -> ust (-) 
                                my_D3[q,p,r,s,u,t] =-val  # uts -> sut (+) 
                                my_D3[q,p,r,s,t,u] = val  # uts -> tus (-) 
                                my_D3[q,p,r,t,s,u] =-val  # uts -> tsu (+) 
                                my_D3[q,p,r,t,u,s] = val  # uts -> stu (-) 

                                # pqr -> rpq (+) 
                                my_D3[r,p,q,u,t,s] = val  # uts -> uts (+) 
                                my_D3[r,p,q,u,s,t] =-val  # uts -> ust (-) 
                                my_D3[r,p,q,s,u,t] = val  # uts -> sut (+) 
                                my_D3[r,p,q,s,t,u] =-val  # uts -> tus (-) 
                                my_D3[r,p,q,t,s,u] = val  # uts -> tsu (+) 
                                my_D3[r,p,q,t,u,s] =-val  # uts -> stu (-) 

                                # pqr -> rqp (-) 
                                my_D3[r,q,p,u,t,s] =-val  # uts -> uts (+) 
                                my_D3[r,q,p,u,s,t] = val  # uts -> ust (-) 
                                my_D3[r,q,p,s,u,t] =-val  # uts -> sut (+) 
                                my_D3[r,q,p,s,t,u] = val  # uts -> tus (-) 
                                my_D3[r,q,p,t,s,u] =-val  # uts -> tsu (+) 
                                my_D3[r,q,p,t,u,s] = val  # uts -> stu (-) 


                                # pqr <-> uts
                                # uts -> uts (+)
                                my_D3[u,t,s,p,q,r] = val  # pqr -> pqr (+)
                                my_D3[u,t,s,p,r,q] =-val  # pqr -> prq (-)
                                my_D3[u,t,s,r,p,q] = val  # pqr -> rpq (+)
                                my_D3[u,t,s,q,p,r] =-val  # pqr -> qpr (-)
                                my_D3[u,t,s,q,r,p] = val  # pqr -> qrp (+)
                                my_D3[u,t,s,r,q,p] =-val  # pqr -> rqp (-)

                                # uts -> ust (-)
                                my_D3[u,s,t,p,q,r] =-val  # pqr -> pqr (+) 
                                my_D3[u,s,t,p,r,q] = val  # pqr -> prq (-) 
                                my_D3[u,s,t,r,p,q] =-val  # pqr -> rpq (+) 
                                my_D3[u,s,t,q,p,r] = val  # pqr -> qpr (-) 
                                my_D3[u,s,t,q,r,p] =-val  # pqr -> qrp (+) 
                                my_D3[u,s,t,r,q,p] = val  # pqr -> rqp (-) 

                                # uts -> tsu (+)
                                my_D3[t,s,u,p,q,r] = val  # pqr -> pqr (+) 
                                my_D3[t,s,u,p,r,q] =-val  # pqr -> prq (-) 
                                my_D3[t,s,u,r,p,q] = val  # pqr -> rpq (+) 
                                my_D3[t,s,u,q,p,r] =-val  # pqr -> qpr (-) 
                                my_D3[t,s,u,q,r,p] = val  # pqr -> qrp (+) 
                                my_D3[t,s,u,r,q,p] =-val  # pqr -> rqp (-) 

                                # uts -> tus (-) 
                                my_D3[t,u,s,p,q,r] =-val  # pqr -> pqr (+) 
                                my_D3[t,u,s,p,r,q] = val  # pqr -> prq (-) 
                                my_D3[t,u,s,r,p,q] =-val  # pqr -> rpq (+) 
                                my_D3[t,u,s,r,q,p] = val  # pqr -> qpr (-) 
                                my_D3[t,u,s,q,r,p] =-val  # pqr -> qrp (+) 
                                my_D3[t,u,s,q,p,r] = val  # pqr -> rqp (-) 

                                # uts -> sut (+) 
                                my_D3[s,u,t,p,q,r] = val  # pqr -> pqr (+) 
                                my_D3[s,u,t,p,r,q] =-val  # pqr -> prq (-) 
                                my_D3[s,u,t,r,p,q] = val  # pqr -> rpq (+) 
                                my_D3[s,u,t,r,q,p] =-val  # pqr -> qpr (-) 
                                my_D3[s,u,t,q,r,p] = val  # pqr -> qrp (+) 
                                my_D3[s,u,t,q,p,r] =-val  # pqr -> rqp (-) 

                                # uts -> stu (-) 
                                my_D3[s,t,u,p,q,r] =-val  # pqr -> pqr (+) 
                                my_D3[s,t,u,p,r,q] = val  # pqr -> prq (-) 
                                my_D3[s,t,u,r,p,q] =-val  # pqr -> rpq (+) 
                                my_D3[s,t,u,r,q,p] = val  # pqr -> qpr (-) 
                                my_D3[s,t,u,q,r,p] =-val  # pqr -> qrp (+) 
                                my_D3[s,t,u,q,p,r] = val  # pqr -> rqp (-) 



    D3 = mpi.allreduce(my_D3, mpi.MPI.SUM)
    return D3


def cumulant_4RDM(D1,D2,D3):
    """
    Form cumulant 4-body density matrix using 1-body, 2-body, and 3-body density matrices, D1, D2 and D3.
    This is an inefficient way of doing things, but allows for easy-coding.
    """
    n_qubits = D1.shape[0]
    D4 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits), dtype=float)
    my_D4 = np.ones((n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits, n_qubits), dtype=float)
    t1 = time.time()
    cyc = 0
    pqrs = -1
    for p in range(n_qubits):
        for q in range(p):
            for r in range(q):  
                for s in range(r):                  
                    pqrs += 1
                    tuvw = -1
                    for t in range(n_qubits):
                        for u in range(t):
                            for v in range(u):     
                                for w in range(v):
                                    tuvw += 1
                                    if pqrs < tuvw or (p%2+q%2+r%2+s%2!=t%2+u%2+v%2+w%2):
                                        continue
                                    if pqruts % mpi.nprocs == mpi.rank:
                                        val = D1[p,w] * D3[q,r,s,t,u,v]\
                                        -D1[q,w] * D3[p,r,s,t,u,v]\
                                        -D1[r,w] * D3[q,p,s,t,u,v]\
                                        -D1[s,w] * D3[q,r,p,t,u,v]\
                                        -D1[p,v] * D3[q,r,s,t,u,w]\
                                        +D1[q,v] * D3[p,r,s,t,u,w]\
                                        +D1[r,v] * D3[q,p,s,t,u,w]\
                                        +D1[s,v] * D3[q,r,p,t,u,w]\
                                        -D1[p,u] * D3[q,r,s,t,w,v]\
                                        +D1[q,u] * D3[p,r,s,t,w,v]\
                                        +D1[r,u] * D3[q,p,s,t,w,v]\
                                        +D1[s,u] * D3[q,r,p,t,w,v]\
                                        -D1[p,t] * D3[q,r,s,w,u,v]\
                                        +D1[q,t] * D3[p,r,s,w,u,v]\
                                        +D1[r,t] * D3[q,p,s,w,u,v]\
                                        +D1[s,t] * D3[q,r,p,w,u,v]\
                                        #
                                        val+= \
                                        +D2[p,q,v,w] * D2[r,s,t,u]\
                                        -D2[p,q,v,u] * D2[r,s,t,w]\
                                        -D2[p,q,v,t] * D2[r,s,w,u]\
                                        -D2[p,q,u,w] * D2[r,s,t,v]\
                                        -D2[p,q,t,w] * D2[r,s,v,u]\
                                        +D2[p,q,t,u] * D2[r,s,v,w]\
                                        -D2[p,s,v,w] * D2[r,q,t,u]\
                                        +D2[p,s,v,u] * D2[r,q,t,w]\
                                        +D2[p,s,v,t] * D2[r,q,w,u]\
                                        +D2[p,s,u,w] * D2[r,q,t,v]\
                                        +D2[p,s,t,w] * D2[r,q,v,u]\
                                        -D2[p,s,t,u] * D2[r,q,v,w]\
                                        -D2[p,r,v,w] * D2[q,s,t,u]\
                                        +D2[p,r,v,u] * D2[q,s,t,w]\
                                        +D2[p,r,v,t] * D2[q,s,w,u]\
                                        +D2[p,r,u,w] * D2[q,s,t,v]\
                                        +D2[p,r,t,w] * D2[q,s,v,u]\
                                        -D2[p,r,t,u] * D2[q,s,v,w]\
                                        #
                                        val+= -2*(\
                                            +(D1[p,w] * D1[q,v] - D1[p,v] * D1[q,w]) * D2[r,s,t,u]\
                                            -(D1[p,w] * D1[q,u] - D1[p,u] * D1[q,w]) * D2[r,s,t,v]\
                                            -(D1[p,w] * D1[q,t] - D1[p,t] * D1[q,w]) * D2[r,s,v,u]\
                                            -(D1[p,u] * D1[q,v] - D1[p,v] * D1[q,u]) * D2[r,s,t,w]\
                                            -(D1[p,t] * D1[q,v] - D1[p,v] * D1[q,t]) * D2[r,s,w,u]\
                                            +(D1[p,t] * D1[q,u] - D1[p,u] * D1[q,t]) * D2[r,s,w,v]\
                                            -(D1[p,w] * D1[s,v] - D1[p,v] * D1[s,w]) * D2[r,q,t,u]\
                                            +(D1[p,w] * D1[s,u] - D1[p,u] * D1[s,w]) * D2[r,q,t,v]\
                                            +(D1[p,w] * D1[s,t] - D1[p,t] * D1[s,w]) * D2[r,q,v,u]\
                                            +(D1[p,u] * D1[s,v] - D1[p,v] * D1[s,u]) * D2[r,q,t,w]\
                                            +(D1[p,t] * D1[s,v] - D1[p,v] * D1[s,t]) * D2[r,q,w,u]\
                                            -(D1[p,t] * D1[s,u] - D1[p,u] * D1[s,t]) * D2[r,q,w,v]\
                                            -(D1[p,w] * D1[r,v] - D1[p,v] * D1[r,w]) * D2[q,s,t,u]\
                                            +(D1[p,w] * D1[r,u] - D1[p,u] * D1[r,w]) * D2[q,s,t,v]\
                                            +(D1[p,w] * D1[r,t] - D1[p,t] * D1[r,w]) * D2[q,s,v,u]\
                                            +(D1[p,u] * D1[r,v] - D1[p,v] * D1[r,u]) * D2[q,s,t,w]\
                                            +(D1[p,t] * D1[r,v] - D1[p,v] * D1[r,t]) * D2[q,s,w,u]\
                                            -(D1[p,t] * D1[r,u] - D1[p,u] * D1[r,t]) * D2[q,s,w,v]\
                                            -(D1[s,w] * D1[q,v] - D1[s,v] * D1[q,w]) * D2[r,p,t,u]\
                                            +(D1[s,w] * D1[q,u] - D1[s,u] * D1[q,w]) * D2[r,p,t,v]\
                                            +(D1[s,w] * D1[q,t] - D1[s,t] * D1[q,w]) * D2[r,p,v,u]\
                                            +(D1[s,u] * D1[q,v] - D1[s,v] * D1[q,u]) * D2[r,p,t,w]\
                                            +(D1[s,t] * D1[q,v] - D1[s,v] * D1[q,t]) * D2[r,p,w,u]\
                                            -(D1[s,t] * D1[q,u] - D1[s,u] * D1[q,t]) * D2[r,p,w,v]\
                                            -(D1[r,w] * D1[q,v] - D1[r,v] * D1[q,w]) * D2[p,s,t,u]\
                                            +(D1[r,w] * D1[q,u] - D1[r,u] * D1[q,w]) * D2[p,s,t,v]\
                                            +(D1[r,w] * D1[q,t] - D1[r,t] * D1[q,w]) * D2[p,s,v,u]\
                                            +(D1[r,u] * D1[q,v] - D1[r,v] * D1[q,u]) * D2[p,s,t,w]\
                                            +(D1[r,t] * D1[q,v] - D1[r,v] * D1[q,t]) * D2[p,s,w,u]\
                                            -(D1[r,t] * D1[q,u] - D1[r,u] * D1[q,t]) * D2[p,s,w,v]\
                                            +(D1[r,w] * D1[s,v] - D1[r,v] * D1[s,w]) * D2[p,q,t,u]\
                                            -(D1[r,w] * D1[s,u] - D1[r,u] * D1[s,w]) * D2[p,q,t,v]\
                                            -(D1[r,w] * D1[s,t] - D1[r,t] * D1[s,w]) * D2[p,q,v,u]\
                                            -(D1[r,u] * D1[s,v] - D1[r,v] * D1[s,u]) * D2[p,q,t,w]\
                                            -(D1[r,t] * D1[s,v] - D1[r,v] * D1[s,t]) * D2[p,q,w,u]\
                                            +(D1[r,t] * D1[s,u] - D1[r,u] * D1[s,t]) * D2[p,q,w,v]\
                                 )
                                        val += 6 * (\
                                               +D1[p,w] * D1[q,v] * D1[r,u] * D1[s,t]\
                                               -D1[p,w] * D1[q,v] * D1[r,t] * D1[s,u]\
                                               -D1[p,w] * D1[q,t] * D1[r,u] * D1[s,v]\
                                               +D1[p,w] * D1[q,t] * D1[r,v] * D1[s,u]\
                                               -D1[p,w] * D1[q,u] * D1[r,v] * D1[s,t]\
                                               +D1[p,w] * D1[q,u] * D1[r,t] * D1[s,v]\
                                               -D1[q,w] * D1[p,v] * D1[r,u] * D1[s,t]\
                                               +D1[q,w] * D1[p,v] * D1[r,t] * D1[s,u]\
                                               +D1[q,w] * D1[p,t] * D1[r,u] * D1[s,v]\
                                               -D1[q,w] * D1[p,t] * D1[r,v] * D1[s,u]\
                                               +D1[q,w] * D1[p,u] * D1[r,v] * D1[s,t]\
                                               -D1[q,w] * D1[p,u] * D1[r,t] * D1[s,v]\
                                               -D1[r,w] * D1[q,v] * D1[p,u] * D1[s,t]\
                                               +D1[r,w] * D1[q,v] * D1[p,t] * D1[s,u]\
                                               +D1[r,w] * D1[q,t] * D1[p,u] * D1[s,v]\
                                               -D1[r,w] * D1[q,t] * D1[p,v] * D1[s,u]\
                                               +D1[r,w] * D1[q,u] * D1[p,v] * D1[s,t]\
                                               -D1[r,w] * D1[q,u] * D1[p,t] * D1[s,v]\
                                               -D1[s,w] * D1[q,v] * D1[r,u] * D1[p,t]\
                                               +D1[s,w] * D1[q,v] * D1[r,t] * D1[p,u]\
                                               +D1[s,w] * D1[q,t] * D1[r,u] * D1[p,v]\
                                               -D1[s,w] * D1[q,t] * D1[r,v] * D1[p,u]\
                                               +D1[s,w] * D1[q,u] * D1[r,v] * D1[p,t]\
                                               -D1[s,w] * D1[q,u] * D1[r,t] * D1[p,v]\
                                              )
                                
####
# put the element in D4 and anti-symmetrize!
####
# p,q,r,s
                                        par = 1
                                        my_D4[p, q, r, s, t, u, v, w] = val*par
                                        my_D4[p, q, r, s, t, u, w, v] =-val*par
                                        my_D4[p, q, r, s, t, v, u, w] =-val*par
                                        my_D4[p, q, r, s, t, v, w, u] = val*par
                                        my_D4[p, q, r, s, t, w, v, u] =-val*par
                                        my_D4[p, q, r, s, t, w, u, v] = val*par
                                        my_D4[p, q, r, s, u, t, v, w] =-val*par
                                        my_D4[p, q, r, s, u, t, w, v] =+val*par
                                        my_D4[p, q, r, s, u, v, t, w] =+val*par
                                        my_D4[p, q, r, s, u, v, w, t] =-val*par
                                        my_D4[p, q, r, s, u, w, v, t] =+val*par
                                        my_D4[p, q, r, s, u, w, t, v] =-val*par
                                        my_D4[p, q, r, s, v, u, t, w] =-val*par
                                        my_D4[p, q, r, s, v, u, w, t] = val*par
                                        my_D4[p, q, r, s, v, t, u, w] = val*par
                                        my_D4[p, q, r, s, v, t, w, u] =-val*par
                                        my_D4[p, q, r, s, v, w, t, u] = val*par
                                        my_D4[p, q, r, s, v, w, u, t] =-val*par
                                        my_D4[p, q, r, s, w, u, v, t] =-val*par
                                        my_D4[p, q, r, s, w, u, t, v] = val*par
                                        my_D4[p, q, r, s, w, v, u, t] = val*par
                                        my_D4[p, q, r, s, w, v, t, u] =-val*par
                                        my_D4[p, q, r, s, w, t, v, u] = val*par
                                        my_D4[p, q, r, s, w, t, u, v] =-val*par                                   
                                        
                                        par = -1
                                        my_D4[p, q, s, r, t, u, v, w] = val*par
                                        my_D4[p, q, s, r, t, u, w, v] =-val*par
                                        my_D4[p, q, s, r, t, v, u, w] =-val*par
                                        my_D4[p, q, s, r, t, v, w, u] = val*par
                                        my_D4[p, q, s, r, t, w, v, u] =-val*par
                                        my_D4[p, q, s, r, t, w, u, v] = val*par
                                        my_D4[p, q, s, r, u, t, v, w] =-val*par
                                        my_D4[p, q, s, r, u, t, w, v] =+val*par
                                        my_D4[p, q, s, r, u, v, t, w] =+val*par
                                        my_D4[p, q, s, r, u, v, w, t] =-val*par 
                                        my_D4[p, q, s, r, u, w, v, t] =+val*par
                                        my_D4[p, q, s, r, u, w, t, v] =-val*par
                                        my_D4[p, q, s, r, v, u, t, w] =-val*par
                                        my_D4[p, q, s, r, v, u, w, t] = val*par
                                        my_D4[p, q, s, r, v, t, u, w] = val*par
                                        my_D4[p, q, s, r, v, t, w, u] =-val*par
                                        my_D4[p, q, s, r, v, w, t, u] = val*par
                                        my_D4[p, q, s, r, v, w, u, t] =-val*par
                                        my_D4[p, q, s, r, w, u, v, t] =-val*par
                                        my_D4[p, q, s, r, w, u, t, v] = val*par
                                        my_D4[p, q, s, r, w, v, u, t] = val*par
                                        my_D4[p, q, s, r, w, v, t, u] =-val*par
                                        my_D4[p, q, s, r, w, t, v, u] = val*par
                                        my_D4[p, q, s, r, w, t, u, v] =-val*par      

                                        par = -1
                                        my_D4[p, r, q, s, t, u, v, w] = val*par
                                        my_D4[p, r, q, s, t, u, w, v] =-val*par
                                        my_D4[p, r, q, s, t, v, u, w] =-val*par
                                        my_D4[p, r, q, s, t, v, w, u] = val*par
                                        my_D4[p, r, q, s, t, w, v, u] =-val*par
                                        my_D4[p, r, q, s, t, w, u, v] = val*par
                                        my_D4[p, r, q, s, u, t, v, w] =-val*par
                                        my_D4[p, r, q, s, u, t, w, v] =+val*par
                                        my_D4[p, r, q, s, u, v, t, w] =+val*par
                                        my_D4[p, r, q, s, u, v, w, t] =-val*par
                                        my_D4[p, r, q, s, u, w, v, t] =+val*par
                                        my_D4[p, r, q, s, u, w, t, v] =-val*par
                                        my_D4[p, r, q, s, v, u, t, w] =-val*par
                                        my_D4[p, r, q, s, v, u, w, t] = val*par
                                        my_D4[p, r, q, s, v, t, u, w] = val*par
                                        my_D4[p, r, q, s, v, t, w, u] =-val*par
                                        my_D4[p, r, q, s, v, w, t, u] = val*par
                                        my_D4[p, r, q, s, v, w, u, t] =-val*par
                                        my_D4[p, r, q, s, w, u, v, t] =-val*par
                                        my_D4[p, r, q, s, w, u, t, v] = val*par
                                        my_D4[p, r, q, s, w, v, u, t] = val*par
                                        my_D4[p, r, q, s, w, v, t, u] =-val*par
                                        my_D4[p, r, q, s, w, t, v, u] = val*par
                                        my_D4[p, r, q, s, w, t, u, v] =-val*par                                   

                                        par = 1
                                        my_D4[p, r, s, q, t, u, v, w] = val*par
                                        my_D4[p, r, s, q, t, u, w, v] =-val*par
                                        my_D4[p, r, s, q, t, v, u, w] =-val*par
                                        my_D4[p, r, s, q, t, v, w, u] = val*par
                                        my_D4[p, r, s, q, t, w, v, u] =-val*par
                                        my_D4[p, r, s, q, t, w, u, v] = val*par
                                        my_D4[p, r, s, q, u, t, v, w] =-val*par
                                        my_D4[p, r, s, q, u, t, w, v] =+val*par
                                        my_D4[p, r, s, q, u, v, t, w] =+val*par
                                        my_D4[p, r, s, q, u, v, w, t] =-val*par
                                        my_D4[p, r, s, q, u, w, v, t] =+val*par
                                        my_D4[p, r, s, q, u, w, t, v] =-val*par
                                        my_D4[p, r, s, q, v, u, t, w] =-val*par
                                        my_D4[p, r, s, q, v, u, w, t] = val*par
                                        my_D4[p, r, s, q, v, t, u, w] = val*par
                                        my_D4[p, r, s, q, v, t, w, u] =-val*par
                                        my_D4[p, r, s, q, v, w, t, u] = val*par
                                        my_D4[p, r, s, q, v, w, u, t] =-val*par
                                        my_D4[p, r, s, q, w, u, v, t] =-val*par
                                        my_D4[p, r, s, q, w, u, t, v] = val*par
                                        my_D4[p, r, s, q, w, v, u, t] = val*par
                                        my_D4[p, r, s, q, w, v, t, u] =-val*par
                                        my_D4[p, r, s, q, w, t, v, u] = val*par
                                        my_D4[p, r, s, q, w, t, u, v] =-val*par   
                                        
                                        par = -1
                                        my_D4[p, s, r, q, t, u, v, w] = val*par
                                        my_D4[p, s, r, q, t, u, w, v] =-val*par
                                        my_D4[p, s, r, q, t, v, u, w] =-val*par
                                        my_D4[p, s, r, q, t, v, w, u] = val*par
                                        my_D4[p, s, r, q, t, w, v, u] =-val*par
                                        my_D4[p, s, r, q, t, w, u, v] = val*par
                                        my_D4[p, s, r, q, u, t, v, w] =-val*par
                                        my_D4[p, s, r, q, u, t, w, v] =+val*par
                                        my_D4[p, s, r, q, u, v, t, w] =+val*par
                                        my_D4[p, s, r, q, u, v, w, t] =-val*par 
                                        my_D4[p, s, r, q, u, w, v, t] =+val*par
                                        my_D4[p, s, r, q, u, w, t, v] =-val*par
                                        my_D4[p, s, r, q, v, u, t, w] =-val*par
                                        my_D4[p, s, r, q, v, u, w, t] = val*par
                                        my_D4[p, s, r, q, v, t, u, w] = val*par
                                        my_D4[p, s, r, q, v, t, w, u] =-val*par
                                        my_D4[p, s, r, q, v, w, t, u] = val*par
                                        my_D4[p, s, r, q, v, w, u, t] =-val*par
                                        my_D4[p, s, r, q, w, u, v, t] =-val*par
                                        my_D4[p, s, r, q, w, u, t, v] = val*par
                                        my_D4[p, s, r, q, w, v, u, t] = val*par
                                        my_D4[p, s, r, q, w, v, t, u] =-val*par
                                        my_D4[p, s, r, q, w, t, v, u] = val*par
                                        my_D4[p, s, r, q, w, t, u, v] =-val*par   
                                        
                                        par = 1
                                        my_D4[p, s, q, r, t, u, v, w] = val*par
                                        my_D4[p, s, q, r, t, u, w, v] =-val*par
                                        my_D4[p, s, q, r, t, v, u, w] =-val*par
                                        my_D4[p, s, q, r, t, v, w, u] = val*par
                                        my_D4[p, s, q, r, t, w, v, u] =-val*par
                                        my_D4[p, s, q, r, t, w, u, v] = val*par
                                        my_D4[p, s, q, r, u, t, v, w] =-val*par
                                        my_D4[p, s, q, r, u, t, w, v] =+val*par
                                        my_D4[p, s, q, r, u, v, t, w] =+val*par
                                        my_D4[p, s, q, r, u, v, w, t] =-val*par 
                                        my_D4[p, s, q, r, u, w, v, t] =+val*par
                                        my_D4[p, s, q, r, u, w, t, v] =-val*par
                                        my_D4[p, s, q, r, v, u, t, w] =-val*par
                                        my_D4[p, s, q, r, v, u, w, t] = val*par
                                        my_D4[p, s, q, r, v, t, u, w] = val*par
                                        my_D4[p, s, q, r, v, t, w, u] =-val*par
                                        my_D4[p, s, q, r, v, w, t, u] = val*par
                                        my_D4[p, s, q, r, v, w, u, t] =-val*par
                                        my_D4[p, s, q, r, w, u, v, t] =-val*par
                                        my_D4[p, s, q, r, w, u, t, v] = val*par
                                        my_D4[p, s, q, r, w, v, u, t] = val*par
                                        my_D4[p, s, q, r, w, v, t, u] =-val*par
                                        my_D4[p, s, q, r, w, t, v, u] = val*par
                                        my_D4[p, s, q, r, w, t, u, v] =-val*par            
                                        
### q, [p, r, s] ...                                   
                                        par = -1
                                        my_D4[q, p, r, s, t, u, v, w] = val*par
                                        my_D4[q, p, r, s, t, u, w, v] =-val*par
                                        my_D4[q, p, r, s, t, v, u, w] =-val*par
                                        my_D4[q, p, r, s, t, v, w, u] = val*par
                                        my_D4[q, p, r, s, t, w, v, u] =-val*par
                                        my_D4[q, p, r, s, t, w, u, v] = val*par
                                        my_D4[q, p, r, s, u, t, v, w] =-val*par
                                        my_D4[q, p, r, s, u, t, w, v] =+val*par
                                        my_D4[q, p, r, s, u, v, t, w] =+val*par
                                        my_D4[q, p, r, s, u, v, w, t] =-val*par
                                        my_D4[q, p, r, s, u, w, v, t] =+val*par
                                        my_D4[q, p, r, s, u, w, t, v] =-val*par
                                        my_D4[q, p, r, s, v, u, t, w] =-val*par
                                        my_D4[q, p, r, s, v, u, w, t] = val*par
                                        my_D4[q, p, r, s, v, t, u, w] = val*par
                                        my_D4[q, p, r, s, v, t, w, u] =-val*par
                                        my_D4[q, p, r, s, v, w, t, u] = val*par
                                        my_D4[q, p, r, s, v, w, u, t] =-val*par
                                        my_D4[q, p, r, s, w, u, v, t] =-val*par
                                        my_D4[q, p, r, s, w, u, t, v] = val*par
                                        my_D4[q, p, r, s, w, v, u, t] = val*par
                                        my_D4[q, p, r, s, w, v, t, u] =-val*par
                                        my_D4[q, p, r, s, w, t, v, u] = val*par
                                        my_D4[q, p, r, s, w, t, u, v] =-val*par                                   
                                        
                                        par = 1
                                        my_D4[q, p, s, r, t, u, v, w] = val*par
                                        my_D4[q, p, s, r, t, u, w, v] =-val*par
                                        my_D4[q, p, s, r, t, v, u, w] =-val*par
                                        my_D4[q, p, s, r, t, v, w, u] = val*par
                                        my_D4[q, p, s, r, t, w, v, u] =-val*par
                                        my_D4[q, p, s, r, t, w, u, v] = val*par
                                        my_D4[q, p, s, r, u, t, v, w] =-val*par
                                        my_D4[q, p, s, r, u, t, w, v] =+val*par
                                        my_D4[q, p, s, r, u, v, t, w] =+val*par
                                        my_D4[q, p, s, r, u, v, w, t] =-val*par 
                                        my_D4[q, p, s, r, u, w, v, t] =+val*par
                                        my_D4[q, p, s, r, u, w, t, v] =-val*par
                                        my_D4[q, p, s, r, v, u, t, w] =-val*par
                                        my_D4[q, p, s, r, v, u, w, t] = val*par
                                        my_D4[q, p, s, r, v, t, u, w] = val*par
                                        my_D4[q, p, s, r, v, t, w, u] =-val*par
                                        my_D4[q, p, s, r, v, w, t, u] = val*par
                                        my_D4[q, p, s, r, v, w, u, t] =-val*par
                                        my_D4[q, p, s, r, w, u, v, t] =-val*par
                                        my_D4[q, p, s, r, w, u, t, v] = val*par
                                        my_D4[q, p, s, r, w, v, u, t] = val*par
                                        my_D4[q, p, s, r, w, v, t, u] =-val*par
                                        my_D4[q, p, s, r, w, t, v, u] = val*par
                                        my_D4[q, p, s, r, w, t, u, v] =-val*par      

                                        par = 1
                                        my_D4[q, r, p, s, t, u, v, w] = val*par
                                        my_D4[q, r, p, s, t, u, w, v] =-val*par
                                        my_D4[q, r, p, s, t, v, u, w] =-val*par
                                        my_D4[q, r, p, s, t, v, w, u] = val*par
                                        my_D4[q, r, p, s, t, w, v, u] =-val*par
                                        my_D4[q, r, p, s, t, w, u, v] = val*par
                                        my_D4[q, r, p, s, u, t, v, w] =-val*par
                                        my_D4[q, r, p, s, u, t, w, v] =+val*par
                                        my_D4[q, r, p, s, u, v, t, w] =+val*par
                                        my_D4[q, r, p, s, u, v, w, t] =-val*par
                                        my_D4[q, r, p, s, u, w, v, t] =+val*par
                                        my_D4[q, r, p, s, u, w, t, v] =-val*par
                                        my_D4[q, r, p, s, v, u, t, w] =-val*par
                                        my_D4[q, r, p, s, v, u, w, t] = val*par
                                        my_D4[q, r, p, s, v, t, u, w] = val*par
                                        my_D4[q, r, p, s, v, t, w, u] =-val*par
                                        my_D4[q, r, p, s, v, w, t, u] = val*par
                                        my_D4[q, r, p, s, v, w, u, t] =-val*par
                                        my_D4[q, r, p, s, w, u, v, t] =-val*par
                                        my_D4[q, r, p, s, w, u, t, v] = val*par
                                        my_D4[q, r, p, s, w, v, u, t] = val*par
                                        my_D4[q, r, p, s, w, v, t, u] =-val*par
                                        my_D4[q, r, p, s, w, t, v, u] = val*par
                                        my_D4[q, r, p, s, w, t, u, v] =-val*par                                   

                                        par = -1
                                        my_D4[q, r, s, p, t, u, v, w] = val*par
                                        my_D4[q, r, s, p, t, u, w, v] =-val*par
                                        my_D4[q, r, s, p, t, v, u, w] =-val*par
                                        my_D4[q, r, s, p, t, v, w, u] = val*par
                                        my_D4[q, r, s, p, t, w, v, u] =-val*par
                                        my_D4[q, r, s, p, t, w, u, v] = val*par
                                        my_D4[q, r, s, p, u, t, v, w] =-val*par
                                        my_D4[q, r, s, p, u, t, w, v] =+val*par
                                        my_D4[q, r, s, p, u, v, t, w] =+val*par
                                        my_D4[q, r, s, p, u, v, w, t] =-val*par
                                        my_D4[q, r, s, p, u, w, v, t] =+val*par
                                        my_D4[q, r, s, p, u, w, t, v] =-val*par
                                        my_D4[q, r, s, p, v, u, t, w] =-val*par
                                        my_D4[q, r, s, p, v, u, w, t] = val*par
                                        my_D4[q, r, s, p, v, t, u, w] = val*par
                                        my_D4[q, r, s, p, v, t, w, u] =-val*par
                                        my_D4[q, r, s, p, v, w, t, u] = val*par
                                        my_D4[q, r, s, p, v, w, u, t] =-val*par
                                        my_D4[q, r, s, p, w, u, v, t] =-val*par
                                        my_D4[q, r, s, p, w, u, t, v] = val*par
                                        my_D4[q, r, s, p, w, v, u, t] = val*par
                                        my_D4[q, r, s, p, w, v, t, u] =-val*par
                                        my_D4[q, r, s, p, w, t, v, u] = val*par
                                        my_D4[q, r, s, p, w, t, u, v] =-val*par   
                                        
                                        par = 1
                                        my_D4[q, s, r, p, t, u, v, w] = val*par
                                        my_D4[q, s, r, p, t, u, w, v] =-val*par
                                        my_D4[q, s, r, p, t, v, u, w] =-val*par
                                        my_D4[q, s, r, p, t, v, w, u] = val*par
                                        my_D4[q, s, r, p, t, w, v, u] =-val*par
                                        my_D4[q, s, r, p, t, w, u, v] = val*par
                                        my_D4[q, s, r, p, u, t, v, w] =-val*par
                                        my_D4[q, s, r, p, u, t, w, v] =+val*par
                                        my_D4[q, s, r, p, u, v, t, w] =+val*par
                                        my_D4[q, s, r, p, u, v, w, t] =-val*par 
                                        my_D4[q, s, r, p, u, w, v, t] =+val*par
                                        my_D4[q, s, r, p, u, w, t, v] =-val*par
                                        my_D4[q, s, r, p, v, u, t, w] =-val*par
                                        my_D4[q, s, r, p, v, u, w, t] = val*par
                                        my_D4[q, s, r, p, v, t, u, w] = val*par
                                        my_D4[q, s, r, p, v, t, w, u] =-val*par
                                        my_D4[q, s, r, p, v, w, t, u] = val*par
                                        my_D4[q, s, r, p, v, w, u, t] =-val*par
                                        my_D4[q, s, r, p, w, u, v, t] =-val*par
                                        my_D4[q, s, r, p, w, u, t, v] = val*par
                                        my_D4[q, s, r, p, w, v, u, t] = val*par
                                        my_D4[q, s, r, p, w, v, t, u] =-val*par
                                        my_D4[q, s, r, p, w, t, v, u] = val*par
                                        my_D4[q, s, r, p, w, t, u, v] =-val*par   
                                        
                                        par = -1
                                        my_D4[q, s, p, r, t, u, v, w] = val*par
                                        my_D4[q, s, p, r, t, u, w, v] =-val*par
                                        my_D4[q, s, p, r, t, v, u, w] =-val*par
                                        my_D4[q, s, p, r, t, v, w, u] = val*par
                                        my_D4[q, s, p, r, t, w, v, u] =-val*par
                                        my_D4[q, s, p, r, t, w, u, v] = val*par
                                        my_D4[q, s, p, r, u, t, v, w] =-val*par
                                        my_D4[q, s, p, r, u, t, w, v] =+val*par
                                        my_D4[q, s, p, r, u, v, t, w] =+val*par
                                        my_D4[q, s, p, r, u, v, w, t] =-val*par 
                                        my_D4[q, s, p, r, u, w, v, t] =+val*par
                                        my_D4[q, s, p, r, u, w, t, v] =-val*par
                                        my_D4[q, s, p, r, v, u, t, w] =-val*par
                                        my_D4[q, s, p, r, v, u, w, t] = val*par
                                        my_D4[q, s, p, r, v, t, u, w] = val*par
                                        my_D4[q, s, p, r, v, t, w, u] =-val*par
                                        my_D4[q, s, p, r, v, w, t, u] = val*par
                                        my_D4[q, s, p, r, v, w, u, t] =-val*par
                                        my_D4[q, s, p, r, w, u, v, t] =-val*par
                                        my_D4[q, s, p, r, w, u, t, v] = val*par
                                        my_D4[q, s, p, r, w, v, u, t] = val*par
                                        my_D4[q, s, p, r, w, v, t, u] =-val*par
                                        my_D4[q, s, p, r, w, t, v, u] = val*par
                                        my_D4[q, s, p, r, w, t, u, v] =-val*par                                         
# r, [p,q,s]
                                        par = -1
                                        my_D4[r, q, p, s, t, u, v, w] = val*par
                                        my_D4[r, q, p, s, t, u, w, v] =-val*par
                                        my_D4[r, q, p, s, t, v, u, w] =-val*par
                                        my_D4[r, q, p, s, t, v, w, u] = val*par
                                        my_D4[r, q, p, s, t, w, v, u] =-val*par
                                        my_D4[r, q, p, s, t, w, u, v] = val*par
                                        my_D4[r, q, p, s, u, t, v, w] =-val*par
                                        my_D4[r, q, p, s, u, t, w, v] =+val*par
                                        my_D4[r, q, p, s, u, v, t, w] =+val*par
                                        my_D4[r, q, p, s, u, v, w, t] =-val*par
                                        my_D4[r, q, p, s, u, w, v, t] =+val*par
                                        my_D4[r, q, p, s, u, w, t, v] =-val*par
                                        my_D4[r, q, p, s, v, u, t, w] =-val*par
                                        my_D4[r, q, p, s, v, u, w, t] = val*par
                                        my_D4[r, q, p, s, v, t, u, w] = val*par
                                        my_D4[r, q, p, s, v, t, w, u] =-val*par
                                        my_D4[r, q, p, s, v, w, t, u] = val*par
                                        my_D4[r, q, p, s, v, w, u, t] =-val*par
                                        my_D4[r, q, p, s, w, u, v, t] =-val*par
                                        my_D4[r, q, p, s, w, u, t, v] = val*par
                                        my_D4[r, q, p, s, w, v, u, t] = val*par
                                        my_D4[r, q, p, s, w, v, t, u] =-val*par
                                        my_D4[r, q, p, s, w, t, v, u] = val*par
                                        my_D4[r, q, p, s, w, t, u, v] =-val*par                                   
                                        
                                        par = 1
                                        my_D4[r, q, s, p, t, u, v, w] = val*par
                                        my_D4[r, q, s, p, t, u, w, v] =-val*par
                                        my_D4[r, q, s, p, t, v, u, w] =-val*par
                                        my_D4[r, q, s, p, t, v, w, u] = val*par
                                        my_D4[r, q, s, p, t, w, v, u] =-val*par
                                        my_D4[r, q, s, p, t, w, u, v] = val*par
                                        my_D4[r, q, s, p, u, t, v, w] =-val*par
                                        my_D4[r, q, s, p, u, t, w, v] =+val*par
                                        my_D4[r, q, s, p, u, v, t, w] =+val*par
                                        my_D4[r, q, s, p, u, v, w, t] =-val*par 
                                        my_D4[r, q, s, p, u, w, v, t] =+val*par
                                        my_D4[r, q, s, p, u, w, t, v] =-val*par
                                        my_D4[r, q, s, p, v, u, t, w] =-val*par
                                        my_D4[r, q, s, p, v, u, w, t] = val*par
                                        my_D4[r, q, s, p, v, t, u, w] = val*par
                                        my_D4[r, q, s, p, v, t, w, u] =-val*par
                                        my_D4[r, q, s, p, v, w, t, u] = val*par
                                        my_D4[r, q, s, p, v, w, u, t] =-val*par
                                        my_D4[r, q, s, p, w, u, v, t] =-val*par
                                        my_D4[r, q, s, p, w, u, t, v] = val*par
                                        my_D4[r, q, s, p, w, v, u, t] = val*par
                                        my_D4[r, q, s, p, w, v, t, u] =-val*par
                                        my_D4[r, q, s, p, w, t, v, u] = val*par
                                        my_D4[r, q, s, p, w, t, u, v] =-val*par      

                                        par = 1
                                        my_D4[r, p, q, s, t, u, v, w] = val*par
                                        my_D4[r, p, q, s, t, u, w, v] =-val*par
                                        my_D4[r, p, q, s, t, v, u, w] =-val*par
                                        my_D4[r, p, q, s, t, v, w, u] = val*par
                                        my_D4[r, p, q, s, t, w, v, u] =-val*par
                                        my_D4[r, p, q, s, t, w, u, v] = val*par
                                        my_D4[r, p, q, s, u, t, v, w] =-val*par
                                        my_D4[r, p, q, s, u, t, w, v] =+val*par
                                        my_D4[r, p, q, s, u, v, t, w] =+val*par
                                        my_D4[r, p, q, s, u, v, w, t] =-val*par
                                        my_D4[r, p, q, s, u, w, v, t] =+val*par
                                        my_D4[r, p, q, s, u, w, t, v] =-val*par
                                        my_D4[r, p, q, s, v, u, t, w] =-val*par
                                        my_D4[r, p, q, s, v, u, w, t] = val*par
                                        my_D4[r, p, q, s, v, t, u, w] = val*par
                                        my_D4[r, p, q, s, v, t, w, u] =-val*par
                                        my_D4[r, p, q, s, v, w, t, u] = val*par
                                        my_D4[r, p, q, s, v, w, u, t] =-val*par
                                        my_D4[r, p, q, s, w, u, v, t] =-val*par
                                        my_D4[r, p, q, s, w, u, t, v] = val*par
                                        my_D4[r, p, q, s, w, v, u, t] = val*par
                                        my_D4[r, p, q, s, w, v, t, u] =-val*par
                                        my_D4[r, p, q, s, w, t, v, u] = val*par
                                        my_D4[r, p, q, s, w, t, u, v] =-val*par                                   

                                        par = -1
                                        my_D4[r, p, s, q, t, u, v, w] = val*par
                                        my_D4[r, p, s, q, t, u, w, v] =-val*par
                                        my_D4[r, p, s, q, t, v, u, w] =-val*par
                                        my_D4[r, p, s, q, t, v, w, u] = val*par
                                        my_D4[r, p, s, q, t, w, v, u] =-val*par
                                        my_D4[r, p, s, q, t, w, u, v] = val*par
                                        my_D4[r, p, s, q, u, t, v, w] =-val*par
                                        my_D4[r, p, s, q, u, t, w, v] =+val*par
                                        my_D4[r, p, s, q, u, v, t, w] =+val*par
                                        my_D4[r, p, s, q, u, v, w, t] =-val*par
                                        my_D4[r, p, s, q, u, w, v, t] =+val*par
                                        my_D4[r, p, s, q, u, w, t, v] =-val*par
                                        my_D4[r, p, s, q, v, u, t, w] =-val*par
                                        my_D4[r, p, s, q, v, u, w, t] = val*par
                                        my_D4[r, p, s, q, v, t, u, w] = val*par
                                        my_D4[r, p, s, q, v, t, w, u] =-val*par
                                        my_D4[r, p, s, q, v, w, t, u] = val*par
                                        my_D4[r, p, s, q, v, w, u, t] =-val*par
                                        my_D4[r, p, s, q, w, u, v, t] =-val*par
                                        my_D4[r, p, s, q, w, u, t, v] = val*par
                                        my_D4[r, p, s, q, w, v, u, t] = val*par
                                        my_D4[r, p, s, q, w, v, t, u] =-val*par
                                        my_D4[r, p, s, q, w, t, v, u] = val*par
                                        my_D4[r, p, s, q, w, t, u, v] =-val*par   
                                        
                                        par = 1
                                        my_D4[r, s, p, q, t, u, v, w] = val*par
                                        my_D4[r, s, p, q, t, u, w, v] =-val*par
                                        my_D4[r, s, p, q, t, v, u, w] =-val*par
                                        my_D4[r, s, p, q, t, v, w, u] = val*par
                                        my_D4[r, s, p, q, t, w, v, u] =-val*par
                                        my_D4[r, s, p, q, t, w, u, v] = val*par
                                        my_D4[r, s, p, q, u, t, v, w] =-val*par
                                        my_D4[r, s, p, q, u, t, w, v] =+val*par
                                        my_D4[r, s, p, q, u, v, t, w] =+val*par
                                        my_D4[r, s, p, q, u, v, w, t] =-val*par 
                                        my_D4[r, s, p, q, u, w, v, t] =+val*par
                                        my_D4[r, s, p, q, u, w, t, v] =-val*par
                                        my_D4[r, s, p, q, v, u, t, w] =-val*par
                                        my_D4[r, s, p, q, v, u, w, t] = val*par
                                        my_D4[r, s, p, q, v, t, u, w] = val*par
                                        my_D4[r, s, p, q, v, t, w, u] =-val*par
                                        my_D4[r, s, p, q, v, w, t, u] = val*par
                                        my_D4[r, s, p, q, v, w, u, t] =-val*par
                                        my_D4[r, s, p, q, w, u, v, t] =-val*par
                                        my_D4[r, s, p, q, w, u, t, v] = val*par
                                        my_D4[r, s, p, q, w, v, u, t] = val*par
                                        my_D4[r, s, p, q, w, v, t, u] =-val*par
                                        my_D4[r, s, p, q, w, t, v, u] = val*par
                                        my_D4[r, s, p, q, w, t, u, v] =-val*par   
                                        
                                        par = -1
                                        my_D4[r, s, q, p, t, u, v, w] = val*par
                                        my_D4[r, s, q, p, t, u, w, v] =-val*par
                                        my_D4[r, s, q, p, t, v, u, w] =-val*par
                                        my_D4[r, s, q, p, t, v, w, u] = val*par
                                        my_D4[r, s, q, p, t, w, v, u] =-val*par
                                        my_D4[r, s, q, p, t, w, u, v] = val*par
                                        my_D4[r, s, q, p, u, t, v, w] =-val*par
                                        my_D4[r, s, q, p, u, t, w, v] =+val*par
                                        my_D4[r, s, q, p, u, v, t, w] =+val*par
                                        my_D4[r, s, q, p, u, v, w, t] =-val*par 
                                        my_D4[r, s, q, p, u, w, v, t] =+val*par
                                        my_D4[r, s, q, p, u, w, t, v] =-val*par
                                        my_D4[r, s, q, p, v, u, t, w] =-val*par
                                        my_D4[r, s, q, p, v, u, w, t] = val*par
                                        my_D4[r, s, q, p, v, t, u, w] = val*par
                                        my_D4[r, s, q, p, v, t, w, u] =-val*par
                                        my_D4[r, s, q, p, v, w, t, u] = val*par
                                        my_D4[r, s, q, p, v, w, u, t] =-val*par
                                        my_D4[r, s, q, p, w, u, v, t] =-val*par
                                        my_D4[r, s, q, p, w, u, t, v] = val*par
                                        my_D4[r, s, q, p, w, v, u, t] = val*par
                                        my_D4[r, s, q, p, w, v, t, u] =-val*par
                                        my_D4[r, s, q, p, w, t, v, u] = val*par
                                        my_D4[r, s, q, p, w, t, u, v] =-val*par     
# s, [p,q,r]
                                        par = -1
                                        my_D4[s, q, r, p, t, u, v, w] = val*par
                                        my_D4[s, q, r, p, t, u, w, v] =-val*par
                                        my_D4[s, q, r, p, t, v, u, w] =-val*par
                                        my_D4[s, q, r, p, t, v, w, u] = val*par
                                        my_D4[s, q, r, p, t, w, v, u] =-val*par
                                        my_D4[s, q, r, p, t, w, u, v] = val*par
                                        my_D4[s, q, r, p, u, t, v, w] =-val*par
                                        my_D4[s, q, r, p, u, t, w, v] =+val*par
                                        my_D4[s, q, r, p, u, v, t, w] =+val*par
                                        my_D4[s, q, r, p, u, v, w, t] =-val*par
                                        my_D4[s, q, r, p, u, w, v, t] =+val*par
                                        my_D4[s, q, r, p, u, w, t, v] =-val*par
                                        my_D4[s, q, r, p, v, u, t, w] =-val*par
                                        my_D4[s, q, r, p, v, u, w, t] = val*par
                                        my_D4[s, q, r, p, v, t, u, w] = val*par
                                        my_D4[s, q, r, p, v, t, w, u] =-val*par
                                        my_D4[s, q, r, p, v, w, t, u] = val*par
                                        my_D4[s, q, r, p, v, w, u, t] =-val*par
                                        my_D4[s, q, r, p, w, u, v, t] =-val*par
                                        my_D4[s, q, r, p, w, u, t, v] = val*par
                                        my_D4[s, q, r, p, w, v, u, t] = val*par
                                        my_D4[s, q, r, p, w, v, t, u] =-val*par
                                        my_D4[s, q, r, p, w, t, v, u] = val*par
                                        my_D4[s, q, r, p, w, t, u, v] =-val*par                                   
                                        
                                        par = 1
                                        my_D4[s, q, p, r, t, u, v, w] = val*par
                                        my_D4[s, q, p, r, t, u, w, v] =-val*par
                                        my_D4[s, q, p, r, t, v, u, w] =-val*par
                                        my_D4[s, q, p, r, t, v, w, u] = val*par
                                        my_D4[s, q, p, r, t, w, v, u] =-val*par
                                        my_D4[s, q, p, r, t, w, u, v] = val*par
                                        my_D4[s, q, p, r, u, t, v, w] =-val*par
                                        my_D4[s, q, p, r, u, t, w, v] =+val*par
                                        my_D4[s, q, p, r, u, v, t, w] =+val*par
                                        my_D4[s, q, p, r, u, v, w, t] =-val*par 
                                        my_D4[s, q, p, r, u, w, v, t] =+val*par
                                        my_D4[s, q, p, r, u, w, t, v] =-val*par
                                        my_D4[s, q, p, r, v, u, t, w] =-val*par
                                        my_D4[s, q, p, r, v, u, w, t] = val*par
                                        my_D4[s, q, p, r, v, t, u, w] = val*par
                                        my_D4[s, q, p, r, v, t, w, u] =-val*par
                                        my_D4[s, q, p, r, v, w, t, u] = val*par
                                        my_D4[s, q, p, r, v, w, u, t] =-val*par
                                        my_D4[s, q, p, r, w, u, v, t] =-val*par
                                        my_D4[s, q, p, r, w, u, t, v] = val*par
                                        my_D4[s, q, p, r, w, v, u, t] = val*par
                                        my_D4[s, q, p, r, w, v, t, u] =-val*par
                                        my_D4[s, q, p, r, w, t, v, u] = val*par
                                        my_D4[s, q, p, r, w, t, u, v] =-val*par      

                                        par = 1
                                        my_D4[s, r, q, p, t, u, v, w] = val*par
                                        my_D4[s, r, q, p, t, u, w, v] =-val*par
                                        my_D4[s, r, q, p, t, v, u, w] =-val*par
                                        my_D4[s, r, q, p, t, v, w, u] = val*par
                                        my_D4[s, r, q, p, t, w, v, u] =-val*par
                                        my_D4[s, r, q, p, t, w, u, v] = val*par
                                        my_D4[s, r, q, p, u, t, v, w] =-val*par
                                        my_D4[s, r, q, p, u, t, w, v] =+val*par
                                        my_D4[s, r, q, p, u, v, t, w] =+val*par
                                        my_D4[s, r, q, p, u, v, w, t] =-val*par
                                        my_D4[s, r, q, p, u, w, v, t] =+val*par
                                        my_D4[s, r, q, p, u, w, t, v] =-val*par
                                        my_D4[s, r, q, p, v, u, t, w] =-val*par
                                        my_D4[s, r, q, p, v, u, w, t] = val*par
                                        my_D4[s, r, q, p, v, t, u, w] = val*par
                                        my_D4[s, r, q, p, v, t, w, u] =-val*par
                                        my_D4[s, r, q, p, v, w, t, u] = val*par
                                        my_D4[s, r, q, p, v, w, u, t] =-val*par
                                        my_D4[s, r, q, p, w, u, v, t] =-val*par
                                        my_D4[s, r, q, p, w, u, t, v] = val*par
                                        my_D4[s, r, q, p, w, v, u, t] = val*par
                                        my_D4[s, r, q, p, w, v, t, u] =-val*par
                                        my_D4[s, r, q, p, w, t, v, u] = val*par
                                        my_D4[s, r, q, p, w, t, u, v] =-val*par                                   

                                        par = -1
                                        my_D4[s, r, p, q, t, u, v, w] = val*par
                                        my_D4[s, r, p, q, t, u, w, v] =-val*par
                                        my_D4[s, r, p, q, t, v, u, w] =-val*par
                                        my_D4[s, r, p, q, t, v, w, u] = val*par
                                        my_D4[s, r, p, q, t, w, v, u] =-val*par
                                        my_D4[s, r, p, q, t, w, u, v] = val*par
                                        my_D4[s, r, p, q, u, t, v, w] =-val*par
                                        my_D4[s, r, p, q, u, t, w, v] =+val*par
                                        my_D4[s, r, p, q, u, v, t, w] =+val*par
                                        my_D4[s, r, p, q, u, v, w, t] =-val*par
                                        my_D4[s, r, p, q, u, w, v, t] =+val*par
                                        my_D4[s, r, p, q, u, w, t, v] =-val*par
                                        my_D4[s, r, p, q, v, u, t, w] =-val*par
                                        my_D4[s, r, p, q, v, u, w, t] = val*par
                                        my_D4[s, r, p, q, v, t, u, w] = val*par
                                        my_D4[s, r, p, q, v, t, w, u] =-val*par
                                        my_D4[s, r, p, q, v, w, t, u] = val*par
                                        my_D4[s, r, p, q, v, w, u, t] =-val*par
                                        my_D4[s, r, p, q, w, u, v, t] =-val*par
                                        my_D4[s, r, p, q, w, u, t, v] = val*par
                                        my_D4[s, r, p, q, w, v, u, t] = val*par
                                        my_D4[s, r, p, q, w, v, t, u] =-val*par
                                        my_D4[s, r, p, q, w, t, v, u] = val*par
                                        my_D4[s, r, p, q, w, t, u, v] =-val*par   
                                        
                                        par = 1
                                        my_D4[s, p, r, q, t, u, v, w] = val*par
                                        my_D4[s, p, r, q, t, u, w, v] =-val*par
                                        my_D4[s, p, r, q, t, v, u, w] =-val*par
                                        my_D4[s, p, r, q, t, v, w, u] = val*par
                                        my_D4[s, p, r, q, t, w, v, u] =-val*par
                                        my_D4[s, p, r, q, t, w, u, v] = val*par
                                        my_D4[s, p, r, q, u, t, v, w] =-val*par
                                        my_D4[s, p, r, q, u, t, w, v] =+val*par
                                        my_D4[s, p, r, q, u, v, t, w] =+val*par
                                        my_D4[s, p, r, q, u, v, w, t] =-val*par 
                                        my_D4[s, p, r, q, u, w, v, t] =+val*par
                                        my_D4[s, p, r, q, u, w, t, v] =-val*par
                                        my_D4[s, p, r, q, v, u, t, w] =-val*par
                                        my_D4[s, p, r, q, v, u, w, t] = val*par
                                        my_D4[s, p, r, q, v, t, u, w] = val*par
                                        my_D4[s, p, r, q, v, t, w, u] =-val*par
                                        my_D4[s, p, r, q, v, w, t, u] = val*par
                                        my_D4[s, p, r, q, v, w, u, t] =-val*par
                                        my_D4[s, p, r, q, w, u, v, t] =-val*par
                                        my_D4[s, p, r, q, w, u, t, v] = val*par
                                        my_D4[s, p, r, q, w, v, u, t] = val*par
                                        my_D4[s, p, r, q, w, v, t, u] =-val*par
                                        my_D4[s, p, r, q, w, t, v, u] = val*par
                                        my_D4[s, p, r, q, w, t, u, v] =-val*par   
                                        
                                        par = -1
                                        my_D4[s, p, q, r, t, u, v, w] = val*par
                                        my_D4[s, p, q, r, t, u, w, v] =-val*par
                                        my_D4[s, p, q, r, t, v, u, w] =-val*par
                                        my_D4[s, p, q, r, t, v, w, u] = val*par
                                        my_D4[s, p, q, r, t, w, v, u] =-val*par
                                        my_D4[s, p, q, r, t, w, u, v] = val*par
                                        my_D4[s, p, q, r, u, t, v, w] =-val*par
                                        my_D4[s, p, q, r, u, t, w, v] =+val*par
                                        my_D4[s, p, q, r, u, v, t, w] =+val*par
                                        my_D4[s, p, q, r, u, v, w, t] =-val*par 
                                        my_D4[s, p, q, r, u, w, v, t] =+val*par
                                        my_D4[s, p, q, r, u, w, t, v] =-val*par
                                        my_D4[s, p, q, r, v, u, t, w] =-val*par
                                        my_D4[s, p, q, r, v, u, w, t] = val*par
                                        my_D4[s, p, q, r, v, t, u, w] = val*par
                                        my_D4[s, p, q, r, v, t, w, u] =-val*par
                                        my_D4[s, p, q, r, v, w, t, u] = val*par
                                        my_D4[s, p, q, r, v, w, u, t] =-val*par
                                        my_D4[s, p, q, r, w, u, v, t] =-val*par
                                        my_D4[s, p, q, r, w, u, t, v] = val*par
                                        my_D4[s, p, q, r, w, v, u, t] = val*par
                                        my_D4[s, p, q, r, w, v, t, u] =-val*par
                                        my_D4[s, p, q, r, w, t, v, u] = val*par
                                        my_D4[s, p, q, r, w, t, u, v] =-val*par   
### pqrs <-> tuvw
# p,q,r,s
                                        par = 1
                                        my_D4[t, u, v, w, p, q, r, s] = val*par
                                        my_D4[t, u, w, v, p, q, r, s] =-val*par
                                        my_D4[t, v, u, w, p, q, r, s] =-val*par
                                        my_D4[t, v, w, u, p, q, r, s] = val*par
                                        my_D4[t, w, v, u, p, q, r, s] =-val*par
                                        my_D4[t, w, u, v, p, q, r, s] = val*par
                                        my_D4[u, t, v, w, p, q, r, s] =-val*par
                                        my_D4[u, t, w, v, p, q, r, s] =+val*par
                                        my_D4[u, v, t, w, p, q, r, s] =+val*par
                                        my_D4[u, v, w, t, p, q, r, s] =-val*par
                                        my_D4[u, w, v, t, p, q, r, s] =+val*par
                                        my_D4[u, w, t, v, p, q, r, s] =-val*par
                                        my_D4[v, u, t, w, p, q, r, s] =-val*par
                                        my_D4[v, u, w, t, p, q, r, s] = val*par
                                        my_D4[v, t, u, w, p, q, r, s] = val*par
                                        my_D4[v, t, w, u, p, q, r, s] =-val*par
                                        my_D4[v, w, t, u, p, q, r, s] = val*par
                                        my_D4[v, w, u, t, p, q, r, s] =-val*par
                                        my_D4[w, u, v, t, p, q, r, s] =-val*par
                                        my_D4[w, u, t, v, p, q, r, s] = val*par
                                        my_D4[w, v, u, t, p, q, r, s] = val*par
                                        my_D4[w, v, t, u, p, q, r, s] =-val*par
                                        my_D4[w, t, v, u, p, q, r, s] = val*par
                                        my_D4[w, t, u, v, p, q, r, s] =-val*par                                   
                                        
                                        par = -1
                                        my_D4[t, u, v, w, p, q, s, r] = val*par
                                        my_D4[t, u, w, v, p, q, s, r] =-val*par
                                        my_D4[t, v, u, w, p, q, s, r] =-val*par
                                        my_D4[t, v, w, u, p, q, s, r] = val*par
                                        my_D4[t, w, v, u, p, q, s, r] =-val*par
                                        my_D4[t, w, u, v, p, q, s, r] = val*par
                                        my_D4[u, t, v, w, p, q, s, r] =-val*par
                                        my_D4[u, t, w, v, p, q, s, r] =+val*par
                                        my_D4[u, v, t, w, p, q, s, r] =+val*par
                                        my_D4[u, v, w, t, p, q, s, r] =-val*par 
                                        my_D4[u, w, v, t, p, q, s, r] =+val*par
                                        my_D4[u, w, t, v, p, q, s, r] =-val*par
                                        my_D4[v, u, t, w, p, q, s, r] =-val*par
                                        my_D4[v, u, w, t, p, q, s, r] = val*par
                                        my_D4[v, t, u, w, p, q, s, r] = val*par
                                        my_D4[v, t, w, u, p, q, s, r] =-val*par
                                        my_D4[v, w, t, u, p, q, s, r] = val*par
                                        my_D4[v, w, u, t, p, q, s, r] =-val*par
                                        my_D4[w, u, v, t, p, q, s, r] =-val*par
                                        my_D4[w, u, t, v, p, q, s, r] = val*par
                                        my_D4[w, v, u, t, p, q, s, r] = val*par
                                        my_D4[w, v, t, u, p, q, s, r] =-val*par
                                        my_D4[w, t, v, u, p, q, s, r] = val*par
                                        my_D4[w, t, u, v, p, q, s, r] =-val*par      

                                        par = -1
                                        my_D4[t, u, v, w, p, r, q, s] = val*par
                                        my_D4[t, u, w, v, p, r, q, s] =-val*par
                                        my_D4[t, v, u, w, p, r, q, s] =-val*par
                                        my_D4[t, v, w, u, p, r, q, s] = val*par
                                        my_D4[t, w, v, u, p, r, q, s] =-val*par
                                        my_D4[t, w, u, v, p, r, q, s] = val*par
                                        my_D4[u, t, v, w, p, r, q, s] =-val*par
                                        my_D4[u, t, w, v, p, r, q, s] =+val*par
                                        my_D4[u, v, t, w, p, r, q, s] =+val*par
                                        my_D4[u, v, w, t, p, r, q, s] =-val*par
                                        my_D4[u, w, v, t, p, r, q, s] =+val*par
                                        my_D4[u, w, t, v, p, r, q, s] =-val*par
                                        my_D4[v, u, t, w, p, r, q, s] =-val*par
                                        my_D4[v, u, w, t, p, r, q, s] = val*par
                                        my_D4[v, t, u, w, p, r, q, s] = val*par
                                        my_D4[v, t, w, u, p, r, q, s] =-val*par
                                        my_D4[v, w, t, u, p, r, q, s] = val*par
                                        my_D4[v, w, u, t, p, r, q, s] =-val*par
                                        my_D4[w, u, v, t, p, r, q, s] =-val*par
                                        my_D4[w, u, t, v, p, r, q, s] = val*par
                                        my_D4[w, v, u, t, p, r, q, s] = val*par
                                        my_D4[w, v, t, u, p, r, q, s] =-val*par
                                        my_D4[w, t, v, u, p, r, q, s] = val*par
                                        my_D4[w, t, u, v, p, r, q, s] =-val*par                                   

                                        par = 1
                                        my_D4[t, u, v, w, p, r, s, q] = val*par
                                        my_D4[t, u, w, v, p, r, s, q] =-val*par
                                        my_D4[t, v, u, w, p, r, s, q] =-val*par
                                        my_D4[t, v, w, u, p, r, s, q] = val*par
                                        my_D4[t, w, v, u, p, r, s, q] =-val*par
                                        my_D4[t, w, u, v, p, r, s, q] = val*par
                                        my_D4[u, t, v, w, p, r, s, q] =-val*par
                                        my_D4[u, t, w, v, p, r, s, q] =+val*par
                                        my_D4[u, v, t, w, p, r, s, q] =+val*par
                                        my_D4[u, v, w, t, p, r, s, q] =-val*par
                                        my_D4[u, w, v, t, p, r, s, q] =+val*par
                                        my_D4[u, w, t, v, p, r, s, q] =-val*par
                                        my_D4[v, u, t, w, p, r, s, q] =-val*par
                                        my_D4[v, u, w, t, p, r, s, q] = val*par
                                        my_D4[v, t, u, w, p, r, s, q] = val*par
                                        my_D4[v, t, w, u, p, r, s, q] =-val*par
                                        my_D4[v, w, t, u, p, r, s, q] = val*par
                                        my_D4[v, w, u, t, p, r, s, q] =-val*par
                                        my_D4[w, u, v, t, p, r, s, q] =-val*par
                                        my_D4[w, u, t, v, p, r, s, q] = val*par
                                        my_D4[w, v, u, t, p, r, s, q] = val*par
                                        my_D4[w, v, t, u, p, r, s, q] =-val*par
                                        my_D4[w, t, v, u, p, r, s, q] = val*par
                                        my_D4[w, t, u, v, p, r, s, q] =-val*par   
                                        
                                        par = -1
                                        my_D4[t, u, v, w, p, s, r, q] = val*par
                                        my_D4[t, u, w, v, p, s, r, q] =-val*par
                                        my_D4[t, v, u, w, p, s, r, q] =-val*par
                                        my_D4[t, v, w, u, p, s, r, q] = val*par
                                        my_D4[t, w, v, u, p, s, r, q] =-val*par
                                        my_D4[t, w, u, v, p, s, r, q] = val*par
                                        my_D4[u, t, v, w, p, s, r, q] =-val*par
                                        my_D4[u, t, w, v, p, s, r, q] =+val*par
                                        my_D4[u, v, t, w, p, s, r, q] =+val*par
                                        my_D4[u, v, w, t, p, s, r, q] =-val*par 
                                        my_D4[u, w, v, t, p, s, r, q] =+val*par
                                        my_D4[u, w, t, v, p, s, r, q] =-val*par
                                        my_D4[v, u, t, w, p, s, r, q] =-val*par
                                        my_D4[v, u, w, t, p, s, r, q] = val*par
                                        my_D4[v, t, u, w, p, s, r, q] = val*par
                                        my_D4[v, t, w, u, p, s, r, q] =-val*par
                                        my_D4[v, w, t, u, p, s, r, q] = val*par
                                        my_D4[v, w, u, t, p, s, r, q] =-val*par
                                        my_D4[w, u, v, t, p, s, r, q] =-val*par
                                        my_D4[w, u, t, v, p, s, r, q] = val*par
                                        my_D4[w, v, u, t, p, s, r, q] = val*par
                                        my_D4[w, v, t, u, p, s, r, q] =-val*par
                                        my_D4[w, t, v, u, p, s, r, q] = val*par
                                        my_D4[w, t, u, v, p, s, r, q] =-val*par   
                                        
                                        par = 1
                                        my_D4[t, u, v, w, p, s, q, r] = val*par
                                        my_D4[t, u, w, v, p, s, q, r] =-val*par
                                        my_D4[t, v, u, w, p, s, q, r] =-val*par
                                        my_D4[t, v, w, u, p, s, q, r] = val*par
                                        my_D4[t, w, v, u, p, s, q, r] =-val*par
                                        my_D4[t, w, u, v, p, s, q, r] = val*par
                                        my_D4[u, t, v, w, p, s, q, r] =-val*par
                                        my_D4[u, t, w, v, p, s, q, r] =+val*par
                                        my_D4[u, v, t, w, p, s, q, r] =+val*par
                                        my_D4[u, v, w, t, p, s, q, r] =-val*par 
                                        my_D4[u, w, v, t, p, s, q, r] =+val*par
                                        my_D4[u, w, t, v, p, s, q, r] =-val*par
                                        my_D4[v, u, t, w, p, s, q, r] =-val*par
                                        my_D4[v, u, w, t, p, s, q, r] = val*par
                                        my_D4[v, t, u, w, p, s, q, r] = val*par
                                        my_D4[v, t, w, u, p, s, q, r] =-val*par
                                        my_D4[v, w, t, u, p, s, q, r] = val*par
                                        my_D4[v, w, u, t, p, s, q, r] =-val*par
                                        my_D4[w, u, v, t, p, s, q, r] =-val*par
                                        my_D4[w, u, t, v, p, s, q, r] = val*par
                                        my_D4[w, v, u, t, p, s, q, r] = val*par
                                        my_D4[w, v, t, u, p, s, q, r] =-val*par
                                        my_D4[w, t, v, u, p, s, q, r] = val*par
                                        my_D4[w, t, u, v, p, s, q, r] =-val*par            
                                        
### q, [p, r, s] ...                        ,           
                                        par = -1
                                        my_D4[t, u, v, w, q, p, r, s] = val*par
                                        my_D4[t, u, w, v, q, p, r, s] =-val*par
                                        my_D4[t, v, u, w, q, p, r, s] =-val*par
                                        my_D4[t, v, w, u, q, p, r, s] = val*par
                                        my_D4[t, w, v, u, q, p, r, s] =-val*par
                                        my_D4[t, w, u, v, q, p, r, s] = val*par
                                        my_D4[u, t, v, w, q, p, r, s] =-val*par
                                        my_D4[u, t, w, v, q, p, r, s] =+val*par
                                        my_D4[u, v, t, w, q, p, r, s] =+val*par
                                        my_D4[u, v, w, t, q, p, r, s] =-val*par
                                        my_D4[u, w, v, t, q, p, r, s] =+val*par
                                        my_D4[u, w, t, v, q, p, r, s] =-val*par
                                        my_D4[v, u, t, w, q, p, r, s] =-val*par
                                        my_D4[v, u, w, t, q, p, r, s] = val*par
                                        my_D4[v, t, u, w, q, p, r, s] = val*par
                                        my_D4[v, t, w, u, q, p, r, s] =-val*par
                                        my_D4[v, w, t, u, q, p, r, s] = val*par
                                        my_D4[v, w, u, t, q, p, r, s] =-val*par
                                        my_D4[w, u, v, t, q, p, r, s] =-val*par
                                        my_D4[w, u, t, v, q, p, r, s] = val*par
                                        my_D4[w, v, u, t, q, p, r, s] = val*par
                                        my_D4[w, v, t, u, q, p, r, s] =-val*par
                                        my_D4[w, t, v, u, q, p, r, s] = val*par
                                        my_D4[w, t, u, v, q, p, r, s] =-val*par                                   
                                        
                                        par = 1
                                        my_D4[t, u, v, w, q, p, s, r] = val*par
                                        my_D4[t, u, w, v, q, p, s, r] =-val*par
                                        my_D4[t, v, u, w, q, p, s, r] =-val*par
                                        my_D4[t, v, w, u, q, p, s, r] = val*par
                                        my_D4[t, w, v, u, q, p, s, r] =-val*par
                                        my_D4[t, w, u, v, q, p, s, r] = val*par
                                        my_D4[u, t, v, w, q, p, s, r] =-val*par
                                        my_D4[u, t, w, v, q, p, s, r] =+val*par
                                        my_D4[u, v, t, w, q, p, s, r] =+val*par
                                        my_D4[u, v, w, t, q, p, s, r] =-val*par 
                                        my_D4[u, w, v, t, q, p, s, r] =+val*par
                                        my_D4[u, w, t, v, q, p, s, r] =-val*par
                                        my_D4[v, u, t, w, q, p, s, r] =-val*par
                                        my_D4[v, u, w, t, q, p, s, r] = val*par
                                        my_D4[v, t, u, w, q, p, s, r] = val*par
                                        my_D4[v, t, w, u, q, p, s, r] =-val*par
                                        my_D4[v, w, t, u, q, p, s, r] = val*par
                                        my_D4[v, w, u, t, q, p, s, r] =-val*par
                                        my_D4[w, u, v, t, q, p, s, r] =-val*par
                                        my_D4[w, u, t, v, q, p, s, r] = val*par
                                        my_D4[w, v, u, t, q, p, s, r] = val*par
                                        my_D4[w, v, t, u, q, p, s, r] =-val*par
                                        my_D4[w, t, v, u, q, p, s, r] = val*par
                                        my_D4[w, t, u, v, q, p, s, r] =-val*par      

                                        par = 1
                                        my_D4[t, u, v, w, q, r, p, s] = val*par
                                        my_D4[t, u, w, v, q, r, p, s] =-val*par
                                        my_D4[t, v, u, w, q, r, p, s] =-val*par
                                        my_D4[t, v, w, u, q, r, p, s] = val*par
                                        my_D4[t, w, v, u, q, r, p, s] =-val*par
                                        my_D4[t, w, u, v, q, r, p, s] = val*par
                                        my_D4[u, t, v, w, q, r, p, s] =-val*par
                                        my_D4[u, t, w, v, q, r, p, s] =+val*par
                                        my_D4[u, v, t, w, q, r, p, s] =+val*par
                                        my_D4[u, v, w, t, q, r, p, s] =-val*par
                                        my_D4[u, w, v, t, q, r, p, s] =+val*par
                                        my_D4[u, w, t, v, q, r, p, s] =-val*par
                                        my_D4[v, u, t, w, q, r, p, s] =-val*par
                                        my_D4[v, u, w, t, q, r, p, s] = val*par
                                        my_D4[v, t, u, w, q, r, p, s] = val*par
                                        my_D4[v, t, w, u, q, r, p, s] =-val*par
                                        my_D4[v, w, t, u, q, r, p, s] = val*par
                                        my_D4[v, w, u, t, q, r, p, s] =-val*par
                                        my_D4[w, u, v, t, q, r, p, s] =-val*par
                                        my_D4[w, u, t, v, q, r, p, s] = val*par
                                        my_D4[w, v, u, t, q, r, p, s] = val*par
                                        my_D4[w, v, t, u, q, r, p, s] =-val*par
                                        my_D4[w, t, v, u, q, r, p, s] = val*par
                                        my_D4[w, t, u, v, q, r, p, s] =-val*par                                   

                                        par = -1
                                        my_D4[t, u, v, w, q, r, s, p] = val*par
                                        my_D4[t, u, w, v, q, r, s, p] =-val*par
                                        my_D4[t, v, u, w, q, r, s, p] =-val*par
                                        my_D4[t, v, w, u, q, r, s, p] = val*par
                                        my_D4[t, w, v, u, q, r, s, p] =-val*par
                                        my_D4[t, w, u, v, q, r, s, p] = val*par
                                        my_D4[u, t, v, w, q, r, s, p] =-val*par
                                        my_D4[u, t, w, v, q, r, s, p] =+val*par
                                        my_D4[u, v, t, w, q, r, s, p] =+val*par
                                        my_D4[u, v, w, t, q, r, s, p] =-val*par
                                        my_D4[u, w, v, t, q, r, s, p] =+val*par
                                        my_D4[u, w, t, v, q, r, s, p] =-val*par
                                        my_D4[v, u, t, w, q, r, s, p] =-val*par
                                        my_D4[v, u, w, t, q, r, s, p] = val*par
                                        my_D4[v, t, u, w, q, r, s, p] = val*par
                                        my_D4[v, t, w, u, q, r, s, p] =-val*par
                                        my_D4[v, w, t, u, q, r, s, p] = val*par
                                        my_D4[v, w, u, t, q, r, s, p] =-val*par
                                        my_D4[w, u, v, t, q, r, s, p] =-val*par
                                        my_D4[w, u, t, v, q, r, s, p] = val*par
                                        my_D4[w, v, u, t, q, r, s, p] = val*par
                                        my_D4[w, v, t, u, q, r, s, p] =-val*par
                                        my_D4[w, t, v, u, q, r, s, p] = val*par
                                        my_D4[w, t, u, v, q, r, s, p] =-val*par   
                                        
                                        par = 1
                                        my_D4[t, u, v, w, q, s, r, p] = val*par
                                        my_D4[t, u, w, v, q, s, r, p] =-val*par
                                        my_D4[t, v, u, w, q, s, r, p] =-val*par
                                        my_D4[t, v, w, u, q, s, r, p] = val*par
                                        my_D4[t, w, v, u, q, s, r, p] =-val*par
                                        my_D4[t, w, u, v, q, s, r, p] = val*par
                                        my_D4[u, t, v, w, q, s, r, p] =-val*par
                                        my_D4[u, t, w, v, q, s, r, p] =+val*par
                                        my_D4[u, v, t, w, q, s, r, p] =+val*par
                                        my_D4[u, v, w, t, q, s, r, p] =-val*par 
                                        my_D4[u, w, v, t, q, s, r, p] =+val*par
                                        my_D4[u, w, t, v, q, s, r, p] =-val*par
                                        my_D4[v, u, t, w, q, s, r, p] =-val*par
                                        my_D4[v, u, w, t, q, s, r, p] = val*par
                                        my_D4[v, t, u, w, q, s, r, p] = val*par
                                        my_D4[v, t, w, u, q, s, r, p] =-val*par
                                        my_D4[v, w, t, u, q, s, r, p] = val*par
                                        my_D4[v, w, u, t, q, s, r, p] =-val*par
                                        my_D4[w, u, v, t, q, s, r, p] =-val*par
                                        my_D4[w, u, t, v, q, s, r, p] = val*par
                                        my_D4[w, v, u, t, q, s, r, p] = val*par
                                        my_D4[w, v, t, u, q, s, r, p] =-val*par
                                        my_D4[w, t, v, u, q, s, r, p] = val*par
                                        my_D4[w, t, u, v, q, s, r, p] =-val*par   
                                        
                                        par = -1
                                        my_D4[t, u, v, w, q, s, p, r] = val*par
                                        my_D4[t, u, w, v, q, s, p, r] =-val*par
                                        my_D4[t, v, u, w, q, s, p, r] =-val*par
                                        my_D4[t, v, w, u, q, s, p, r] = val*par
                                        my_D4[t, w, v, u, q, s, p, r] =-val*par
                                        my_D4[t, w, u, v, q, s, p, r] = val*par
                                        my_D4[u, t, v, w, q, s, p, r] =-val*par
                                        my_D4[u, t, w, v, q, s, p, r] =+val*par
                                        my_D4[u, v, t, w, q, s, p, r] =+val*par
                                        my_D4[u, v, w, t, q, s, p, r] =-val*par 
                                        my_D4[u, w, v, t, q, s, p, r] =+val*par
                                        my_D4[u, w, t, v, q, s, p, r] =-val*par
                                        my_D4[v, u, t, w, q, s, p, r] =-val*par
                                        my_D4[v, u, w, t, q, s, p, r] = val*par
                                        my_D4[v, t, u, w, q, s, p, r] = val*par
                                        my_D4[v, t, w, u, q, s, p, r] =-val*par
                                        my_D4[v, w, t, u, q, s, p, r] = val*par
                                        my_D4[v, w, u, t, q, s, p, r] =-val*par
                                        my_D4[w, u, v, t, q, s, p, r] =-val*par
                                        my_D4[w, u, t, v, q, s, p, r] = val*par
                                        my_D4[w, v, u, t, q, s, p, r] = val*par
                                        my_D4[w, v, t, u, q, s, p, r] =-val*par
                                        my_D4[w, t, v, u, q, s, p, r] = val*par
                                        my_D4[w, t, u, v, q, s, p, r] =-val*par                                         
# r, [p,q,s]
                                        par = -1
                                        my_D4[t, u, v, w, r, q, p, s] = val*par
                                        my_D4[t, u, w, v, r, q, p, s] =-val*par
                                        my_D4[t, v, u, w, r, q, p, s] =-val*par
                                        my_D4[t, v, w, u, r, q, p, s] = val*par
                                        my_D4[t, w, v, u, r, q, p, s] =-val*par
                                        my_D4[t, w, u, v, r, q, p, s] = val*par
                                        my_D4[u, t, v, w, r, q, p, s] =-val*par
                                        my_D4[u, t, w, v, r, q, p, s] =+val*par
                                        my_D4[u, v, t, w, r, q, p, s] =+val*par
                                        my_D4[u, v, w, t, r, q, p, s] =-val*par
                                        my_D4[u, w, v, t, r, q, p, s] =+val*par
                                        my_D4[u, w, t, v, r, q, p, s] =-val*par
                                        my_D4[v, u, t, w, r, q, p, s] =-val*par
                                        my_D4[v, u, w, t, r, q, p, s] = val*par
                                        my_D4[v, t, u, w, r, q, p, s] = val*par
                                        my_D4[v, t, w, u, r, q, p, s] =-val*par
                                        my_D4[v, w, t, u, r, q, p, s] = val*par
                                        my_D4[v, w, u, t, r, q, p, s] =-val*par
                                        my_D4[w, u, v, t, r, q, p, s] =-val*par
                                        my_D4[w, u, t, v, r, q, p, s] = val*par
                                        my_D4[w, v, u, t, r, q, p, s] = val*par
                                        my_D4[w, v, t, u, r, q, p, s] =-val*par
                                        my_D4[w, t, v, u, r, q, p, s] = val*par
                                        my_D4[w, t, u, v, r, q, p, s] =-val*par                                   
                                        
                                        par = 1
                                        my_D4[t, u, v, w, r, q, s, p] = val*par
                                        my_D4[t, u, w, v, r, q, s, p] =-val*par
                                        my_D4[t, v, u, w, r, q, s, p] =-val*par
                                        my_D4[t, v, w, u, r, q, s, p] = val*par
                                        my_D4[t, w, v, u, r, q, s, p] =-val*par
                                        my_D4[t, w, u, v, r, q, s, p] = val*par
                                        my_D4[u, t, v, w, r, q, s, p] =-val*par
                                        my_D4[u, t, w, v, r, q, s, p] =+val*par
                                        my_D4[u, v, t, w, r, q, s, p] =+val*par
                                        my_D4[u, v, w, t, r, q, s, p] =-val*par 
                                        my_D4[u, w, v, t, r, q, s, p] =+val*par
                                        my_D4[u, w, t, v, r, q, s, p] =-val*par
                                        my_D4[v, u, t, w, r, q, s, p] =-val*par
                                        my_D4[v, u, w, t, r, q, s, p] = val*par
                                        my_D4[v, t, u, w, r, q, s, p] = val*par
                                        my_D4[v, t, w, u, r, q, s, p] =-val*par
                                        my_D4[v, w, t, u, r, q, s, p] = val*par
                                        my_D4[v, w, u, t, r, q, s, p] =-val*par
                                        my_D4[w, u, v, t, r, q, s, p] =-val*par
                                        my_D4[w, u, t, v, r, q, s, p] = val*par
                                        my_D4[w, v, u, t, r, q, s, p] = val*par
                                        my_D4[w, v, t, u, r, q, s, p] =-val*par
                                        my_D4[w, t, v, u, r, q, s, p] = val*par
                                        my_D4[w, t, u, v, r, q, s, p] =-val*par      

                                        par = 1
                                        my_D4[t, u, v, w, r, p, q, s] = val*par
                                        my_D4[t, u, w, v, r, p, q, s] =-val*par
                                        my_D4[t, v, u, w, r, p, q, s] =-val*par
                                        my_D4[t, v, w, u, r, p, q, s] = val*par
                                        my_D4[t, w, v, u, r, p, q, s] =-val*par
                                        my_D4[t, w, u, v, r, p, q, s] = val*par
                                        my_D4[u, t, v, w, r, p, q, s] =-val*par
                                        my_D4[u, t, w, v, r, p, q, s] =+val*par
                                        my_D4[u, v, t, w, r, p, q, s] =+val*par
                                        my_D4[u, v, w, t, r, p, q, s] =-val*par
                                        my_D4[u, w, v, t, r, p, q, s] =+val*par
                                        my_D4[u, w, t, v, r, p, q, s] =-val*par
                                        my_D4[v, u, t, w, r, p, q, s] =-val*par
                                        my_D4[v, u, w, t, r, p, q, s] = val*par
                                        my_D4[v, t, u, w, r, p, q, s] = val*par
                                        my_D4[v, t, w, u, r, p, q, s] =-val*par
                                        my_D4[v, w, t, u, r, p, q, s] = val*par
                                        my_D4[v, w, u, t, r, p, q, s] =-val*par
                                        my_D4[w, u, v, t, r, p, q, s] =-val*par
                                        my_D4[w, u, t, v, r, p, q, s] = val*par
                                        my_D4[w, v, u, t, r, p, q, s] = val*par
                                        my_D4[w, v, t, u, r, p, q, s] =-val*par
                                        my_D4[w, t, v, u, r, p, q, s] = val*par
                                        my_D4[w, t, u, v, r, p, q, s] =-val*par                                   

                                        par = -1
                                        my_D4[t, u, v, w, r, p, s, q] = val*par
                                        my_D4[t, u, w, v, r, p, s, q] =-val*par
                                        my_D4[t, v, u, w, r, p, s, q] =-val*par
                                        my_D4[t, v, w, u, r, p, s, q] = val*par
                                        my_D4[t, w, v, u, r, p, s, q] =-val*par
                                        my_D4[t, w, u, v, r, p, s, q] = val*par
                                        my_D4[u, t, v, w, r, p, s, q] =-val*par
                                        my_D4[u, t, w, v, r, p, s, q] =+val*par
                                        my_D4[u, v, t, w, r, p, s, q] =+val*par
                                        my_D4[u, v, w, t, r, p, s, q] =-val*par
                                        my_D4[u, w, v, t, r, p, s, q] =+val*par
                                        my_D4[u, w, t, v, r, p, s, q] =-val*par
                                        my_D4[v, u, t, w, r, p, s, q] =-val*par
                                        my_D4[v, u, w, t, r, p, s, q] = val*par
                                        my_D4[v, t, u, w, r, p, s, q] = val*par
                                        my_D4[v, t, w, u, r, p, s, q] =-val*par
                                        my_D4[v, w, t, u, r, p, s, q] = val*par
                                        my_D4[v, w, u, t, r, p, s, q] =-val*par
                                        my_D4[w, u, v, t, r, p, s, q] =-val*par
                                        my_D4[w, u, t, v, r, p, s, q] = val*par
                                        my_D4[w, v, u, t, r, p, s, q] = val*par
                                        my_D4[w, v, t, u, r, p, s, q] =-val*par
                                        my_D4[w, t, v, u, r, p, s, q] = val*par
                                        my_D4[w, t, u, v, r, p, s, q] =-val*par   
                                        
                                        par = 1
                                        my_D4[t, u, v, w, r, s, p, q] = val*par
                                        my_D4[t, u, w, v, r, s, p, q] =-val*par
                                        my_D4[t, v, u, w, r, s, p, q] =-val*par
                                        my_D4[t, v, w, u, r, s, p, q] = val*par
                                        my_D4[t, w, v, u, r, s, p, q] =-val*par
                                        my_D4[t, w, u, v, r, s, p, q] = val*par
                                        my_D4[u, t, v, w, r, s, p, q] =-val*par
                                        my_D4[u, t, w, v, r, s, p, q] =+val*par
                                        my_D4[u, v, t, w, r, s, p, q] =+val*par
                                        my_D4[u, v, w, t, r, s, p, q] =-val*par 
                                        my_D4[u, w, v, t, r, s, p, q] =+val*par
                                        my_D4[u, w, t, v, r, s, p, q] =-val*par
                                        my_D4[v, u, t, w, r, s, p, q] =-val*par
                                        my_D4[v, u, w, t, r, s, p, q] = val*par
                                        my_D4[v, t, u, w, r, s, p, q] = val*par
                                        my_D4[v, t, w, u, r, s, p, q] =-val*par
                                        my_D4[v, w, t, u, r, s, p, q] = val*par
                                        my_D4[v, w, u, t, r, s, p, q] =-val*par
                                        my_D4[w, u, v, t, r, s, p, q] =-val*par
                                        my_D4[w, u, t, v, r, s, p, q] = val*par
                                        my_D4[w, v, u, t, r, s, p, q] = val*par
                                        my_D4[w, v, t, u, r, s, p, q] =-val*par
                                        my_D4[w, t, v, u, r, s, p, q] = val*par
                                        my_D4[w, t, u, v, r, s, p, q] =-val*par   
                                        
                                        par = -1
                                        my_D4[t, u, v, w, r, s, q, p] = val*par
                                        my_D4[t, u, w, v, r, s, q, p] =-val*par
                                        my_D4[t, v, u, w, r, s, q, p] =-val*par
                                        my_D4[t, v, w, u, r, s, q, p] = val*par
                                        my_D4[t, w, v, u, r, s, q, p] =-val*par
                                        my_D4[t, w, u, v, r, s, q, p] = val*par
                                        my_D4[u, t, v, w, r, s, q, p] =-val*par
                                        my_D4[u, t, w, v, r, s, q, p] =+val*par
                                        my_D4[u, v, t, w, r, s, q, p] =+val*par
                                        my_D4[u, v, w, t, r, s, q, p] =-val*par 
                                        my_D4[u, w, v, t, r, s, q, p] =+val*par
                                        my_D4[u, w, t, v, r, s, q, p] =-val*par
                                        my_D4[v, u, t, w, r, s, q, p] =-val*par
                                        my_D4[v, u, w, t, r, s, q, p] = val*par
                                        my_D4[v, t, u, w, r, s, q, p] = val*par
                                        my_D4[v, t, w, u, r, s, q, p] =-val*par
                                        my_D4[v, w, t, u, r, s, q, p] = val*par
                                        my_D4[v, w, u, t, r, s, q, p] =-val*par
                                        my_D4[w, u, v, t, r, s, q, p] =-val*par
                                        my_D4[w, u, t, v, r, s, q, p] = val*par
                                        my_D4[w, v, u, t, r, s, q, p] = val*par
                                        my_D4[w, v, t, u, r, s, q, p] =-val*par
                                        my_D4[w, t, v, u, r, s, q, p] = val*par
                                        my_D4[w, t, u, v, r, s, q, p] =-val*par     
# s, [p,q,r]
                                        par = -1
                                        my_D4[t, u, v, w, s, q, r, p] = val*par
                                        my_D4[t, u, w, v, s, q, r, p] =-val*par
                                        my_D4[t, v, u, w, s, q, r, p] =-val*par
                                        my_D4[t, v, w, u, s, q, r, p] = val*par
                                        my_D4[t, w, v, u, s, q, r, p] =-val*par
                                        my_D4[t, w, u, v, s, q, r, p] = val*par
                                        my_D4[u, t, v, w, s, q, r, p] =-val*par
                                        my_D4[u, t, w, v, s, q, r, p] =+val*par
                                        my_D4[u, v, t, w, s, q, r, p] =+val*par
                                        my_D4[u, v, w, t, s, q, r, p] =-val*par
                                        my_D4[u, w, v, t, s, q, r, p] =+val*par
                                        my_D4[u, w, t, v, s, q, r, p] =-val*par
                                        my_D4[v, u, t, w, s, q, r, p] =-val*par
                                        my_D4[v, u, w, t, s, q, r, p] = val*par
                                        my_D4[v, t, u, w, s, q, r, p] = val*par
                                        my_D4[v, t, w, u, s, q, r, p] =-val*par
                                        my_D4[v, w, t, u, s, q, r, p] = val*par
                                        my_D4[v, w, u, t, s, q, r, p] =-val*par
                                        my_D4[w, u, v, t, s, q, r, p] =-val*par
                                        my_D4[w, u, t, v, s, q, r, p] = val*par
                                        my_D4[w, v, u, t, s, q, r, p] = val*par
                                        my_D4[w, v, t, u, s, q, r, p] =-val*par
                                        my_D4[w, t, v, u, s, q, r, p] = val*par
                                        my_D4[w, t, u, v, s, q, r, p] =-val*par                                   
                                        
                                        par = 1
                                        my_D4[t, u, v, w, s, q, p, r] = val*par
                                        my_D4[t, u, w, v, s, q, p, r] =-val*par
                                        my_D4[t, v, u, w, s, q, p, r] =-val*par
                                        my_D4[t, v, w, u, s, q, p, r] = val*par
                                        my_D4[t, w, v, u, s, q, p, r] =-val*par
                                        my_D4[t, w, u, v, s, q, p, r] = val*par
                                        my_D4[u, t, v, w, s, q, p, r] =-val*par
                                        my_D4[u, t, w, v, s, q, p, r] =+val*par
                                        my_D4[u, v, t, w, s, q, p, r] =+val*par
                                        my_D4[u, v, w, t, s, q, p, r] =-val*par 
                                        my_D4[u, w, v, t, s, q, p, r] =+val*par
                                        my_D4[u, w, t, v, s, q, p, r] =-val*par
                                        my_D4[v, u, t, w, s, q, p, r] =-val*par
                                        my_D4[v, u, w, t, s, q, p, r] = val*par
                                        my_D4[v, t, u, w, s, q, p, r] = val*par
                                        my_D4[v, t, w, u, s, q, p, r] =-val*par
                                        my_D4[v, w, t, u, s, q, p, r] = val*par
                                        my_D4[v, w, u, t, s, q, p, r] =-val*par
                                        my_D4[w, u, v, t, s, q, p, r] =-val*par
                                        my_D4[w, u, t, v, s, q, p, r] = val*par
                                        my_D4[w, v, u, t, s, q, p, r] = val*par
                                        my_D4[w, v, t, u, s, q, p, r] =-val*par
                                        my_D4[w, t, v, u, s, q, p, r] = val*par
                                        my_D4[w, t, u, v, s, q, p, r] =-val*par      

                                        par = 1
                                        my_D4[t, u, v, w, s, r, q, p] = val*par
                                        my_D4[t, u, w, v, s, r, q, p] =-val*par
                                        my_D4[t, v, u, w, s, r, q, p] =-val*par
                                        my_D4[t, v, w, u, s, r, q, p] = val*par
                                        my_D4[t, w, v, u, s, r, q, p] =-val*par
                                        my_D4[t, w, u, v, s, r, q, p] = val*par
                                        my_D4[u, t, v, w, s, r, q, p] =-val*par
                                        my_D4[u, t, w, v, s, r, q, p] =+val*par
                                        my_D4[u, v, t, w, s, r, q, p] =+val*par
                                        my_D4[u, v, w, t, s, r, q, p] =-val*par
                                        my_D4[u, w, v, t, s, r, q, p] =+val*par
                                        my_D4[u, w, t, v, s, r, q, p] =-val*par
                                        my_D4[v, u, t, w, s, r, q, p] =-val*par
                                        my_D4[v, u, w, t, s, r, q, p] = val*par
                                        my_D4[v, t, u, w, s, r, q, p] = val*par
                                        my_D4[v, t, w, u, s, r, q, p] =-val*par
                                        my_D4[v, w, t, u, s, r, q, p] = val*par
                                        my_D4[v, w, u, t, s, r, q, p] =-val*par
                                        my_D4[w, u, v, t, s, r, q, p] =-val*par
                                        my_D4[w, u, t, v, s, r, q, p] = val*par
                                        my_D4[w, v, u, t, s, r, q, p] = val*par
                                        my_D4[w, v, t, u, s, r, q, p] =-val*par
                                        my_D4[w, t, v, u, s, r, q, p] = val*par
                                        my_D4[w, t, u, v, s, r, q, p] =-val*par                                   

                                        par = -1
                                        my_D4[t, u, v, w, s, r, p, q] = val*par
                                        my_D4[t, u, w, v, s, r, p, q] =-val*par
                                        my_D4[t, v, u, w, s, r, p, q] =-val*par
                                        my_D4[t, v, w, u, s, r, p, q] = val*par
                                        my_D4[t, w, v, u, s, r, p, q] =-val*par
                                        my_D4[t, w, u, v, s, r, p, q] = val*par
                                        my_D4[u, t, v, w, s, r, p, q] =-val*par
                                        my_D4[u, t, w, v, s, r, p, q] =+val*par
                                        my_D4[u, v, t, w, s, r, p, q] =+val*par
                                        my_D4[u, v, w, t, s, r, p, q] =-val*par
                                        my_D4[u, w, v, t, s, r, p, q] =+val*par
                                        my_D4[u, w, t, v, s, r, p, q] =-val*par
                                        my_D4[v, u, t, w, s, r, p, q] =-val*par
                                        my_D4[v, u, w, t, s, r, p, q] = val*par
                                        my_D4[v, t, u, w, s, r, p, q] = val*par
                                        my_D4[v, t, w, u, s, r, p, q] =-val*par
                                        my_D4[v, w, t, u, s, r, p, q] = val*par
                                        my_D4[v, w, u, t, s, r, p, q] =-val*par
                                        my_D4[w, u, v, t, s, r, p, q] =-val*par
                                        my_D4[w, u, t, v, s, r, p, q] = val*par
                                        my_D4[w, v, u, t, s, r, p, q] = val*par
                                        my_D4[w, v, t, u, s, r, p, q] =-val*par
                                        my_D4[w, t, v, u, s, r, p, q] = val*par
                                        my_D4[w, t, u, v, s, r, p, q] =-val*par   
                                        
                                        par = 1
                                        my_D4[t, u, v, w, s, p, r, q] = val*par
                                        my_D4[t, u, w, v, s, p, r, q] =-val*par
                                        my_D4[t, v, u, w, s, p, r, q] =-val*par
                                        my_D4[t, v, w, u, s, p, r, q] = val*par
                                        my_D4[t, w, v, u, s, p, r, q] =-val*par
                                        my_D4[t, w, u, v, s, p, r, q] = val*par
                                        my_D4[u, t, v, w, s, p, r, q] =-val*par
                                        my_D4[u, t, w, v, s, p, r, q] =+val*par
                                        my_D4[u, v, t, w, s, p, r, q] =+val*par
                                        my_D4[u, v, w, t, s, p, r, q] =-val*par 
                                        my_D4[u, w, v, t, s, p, r, q] =+val*par
                                        my_D4[u, w, t, v, s, p, r, q] =-val*par
                                        my_D4[v, u, t, w, s, p, r, q] =-val*par
                                        my_D4[v, u, w, t, s, p, r, q] = val*par
                                        my_D4[v, t, u, w, s, p, r, q] = val*par
                                        my_D4[v, t, w, u, s, p, r, q] =-val*par
                                        my_D4[v, w, t, u, s, p, r, q] = val*par
                                        my_D4[v, w, u, t, s, p, r, q] =-val*par
                                        my_D4[w, u, v, t, s, p, r, q] =-val*par
                                        my_D4[w, u, t, v, s, p, r, q] = val*par
                                        my_D4[w, v, u, t, s, p, r, q] = val*par
                                        my_D4[w, v, t, u, s, p, r, q] =-val*par
                                        my_D4[w, t, v, u, s, p, r, q] = val*par
                                        my_D4[w, t, u, v, s, p, r, q] =-val*par   
                                        
                                        par = -1
                                        my_D4[t, u, v, w, s, p, q, r] = val*par
                                        my_D4[t, u, w, v, s, p, q, r] =-val*par
                                        my_D4[t, v, u, w, s, p, q, r] =-val*par
                                        my_D4[t, v, w, u, s, p, q, r] = val*par
                                        my_D4[t, w, v, u, s, p, q, r] =-val*par
                                        my_D4[t, w, u, v, s, p, q, r] = val*par
                                        my_D4[u, t, v, w, s, p, q, r] =-val*par
                                        my_D4[u, t, w, v, s, p, q, r] =+val*par
                                        my_D4[u, v, t, w, s, p, q, r] =+val*par
                                        my_D4[u, v, w, t, s, p, q, r] =-val*par 
                                        my_D4[u, w, v, t, s, p, q, r] =+val*par
                                        my_D4[u, w, t, v, s, p, q, r] =-val*par
                                        my_D4[v, u, t, w, s, p, q, r] =-val*par
                                        my_D4[v, u, w, t, s, p, q, r] = val*par
                                        my_D4[v, t, u, w, s, p, q, r] = val*par
                                        my_D4[v, t, w, u, s, p, q, r] =-val*par
                                        my_D4[v, w, t, u, s, p, q, r] = val*par
                                        my_D4[v, w, u, t, s, p, q, r] =-val*par
                                        my_D4[w, u, v, t, s, p, q, r] =-val*par
                                        my_D4[w, u, t, v, s, p, q, r] = val*par
                                        my_D4[w, v, u, t, s, p, q, r] = val*par
                                        my_D4[w, v, t, u, s, p, q, r] =-val*par
                                        my_D4[w, t, v, u, s, p, q, r] = val*par
                                        my_D4[w, t, u, v, s, p, q, r] =-val*par   

    D4 = mpi.allreduce(my_D4, mpi.MPI.SUM)
    return D4
