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

orbital/hess.py

Functions related to orbital hessian

"""
import numpy as np
import time

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.linalg import vectorize_skew, skew
from quket.fileio import error, prints, printmat, print_state
from .hess_vec import *
from .misc import *

def get_HF_orbital_hessian(Quket):
    """ Function
    Compute orbital Hessian Aaijb of Hartree-Fock (whole orbital space).
    Args: 
        Quket (QuketData)
    Returns:
        Aaibj (2darray): Hessian matrix

    Author(s): Taisei Nishimaki, Takashi Tsuchimochi
    """
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    ncore = Quket.n_frozen_orbitals + Quket.n_core_orbitals
    nact = Quket.n_active_orbitals
    norbs = Quket.n_orbitals
    nsec = norbs - ncore - nact 
    NOA = noa + ncore
    NOB = nob + ncore
    NVA = nva + nsec
    NVB = nvb + nsec
    NOAVA = NOA*NVA
    NOBVB = NOB*NVB

    # Set matrix for each spin
    Aaa = np.zeros((NOAVA, NOAVA))
    Abb = np.zeros((NOBVB, NOBVB))
    Aba = np.zeros((NOBVB, NOAVA))
    # Set orbital energies
    mo_coeff = Quket.mo_coeff
    # Note: With frozen-core orbitals, we have to 'shift' the index by ncore
    #       in order to actually handle the target orbitals i,j,a,b
    #       (i.e., i <- i + ncore)
    mo_energy = Quket.mo_energy
    h_pqrs = Quket.two_body_integrals

    #print("\n == Aaibj ==")
    ### Aaa ###
    bj = 0
    for j in range(NOA):
        for b in range(NVA):
            ai = 0
            for i in range(NOA):
                for a in range(NVA):
                    # 1st term
                    if a == b and i == j:
                        E_mol = mo_energy[a+NOA] - mo_energy[i]
                    else:
                        E_mol = .0
                    # 2nd,3rd term
                    # <aj||ib> + <ab||ij> = <aj|ib> - <aj|bi> + <ab|ij> - <ab|ji>
                    #                     = (ai|jb) - (ab|ji) + (ai||bj) - (aj|bi)
                    #                     = 2(ai|bj) - (ab|ij) - (aj|bi)
                    two_e_integral = (h_pqrs[a+NOA, i, j, b+NOA]
                                      - h_pqrs[a+NOA, b+NOA, j, i]
                                      + h_pqrs[a+NOA, i, b+NOA, j]
                                      - h_pqrs[a+NOA, j, b+NOA, i])
                    Aaa[ai, bj] = E_mol + two_e_integral
                    ai+=1
            bj+=1
    ### Abb ###
    bj = 0
    for j in range(NOB):
        for b in range(NVB):
            ai = 0
            for i in range(NOB):
                for a in range(NVB):
                    if a == b and i == j:
                        E_mol = mo_energy[a+NOB] - mo_energy[i]
                    else:
                        E_mol = .0
                    two_e_integral = (h_pqrs[a+NOB, i, j, b+NOB]
                                      - h_pqrs[a+NOB, b+NOB, j, i]
                                      + h_pqrs[a+NOB, i, b+NOB, j]
                                      - h_pqrs[a+NOB, j, b+NOB, i])
                    Abb[ai, bj] = E_mol + two_e_integral
                    ai+=1
            bj+=1
    ### Aba ###
    bj = 0
    for j in range(NOA):
        for b in range(NVA):
            ai = 0
            for i in range(NOB):
                for a in range(NVB):
                    two_e_integral = (h_pqrs[a+NOB, i, j, b+NOA]
                                      + h_pqrs[a+NOB, i, b+NOA, j])
                    Aba[ai, bj] = two_e_integral
                    ai+=1
            bj+=1
    Aaibj = np.block([[Aaa, Aba.T],
                      [Aba, Abb]])
    return  Aaibj


def Hx(x, Quket, DA=None, DB=None, Daaaa=None, Dbbbb=None, Dbaab=None, state=None, spinless=True):
    """
    Matrix-vector multiplication between orbital Hessian H[tu,vw] = <[[H, Etu - Eut], Evw - Ewv]> and a vector x
    using 1- and 2-RDMs.
    Here we consider the most general form of Hessian,

              [AC, AC]     [AC, AA]     [AC, VC]     [AC, VA]
              [AA, AC]     [AA, AA]     [AA, VC]     [AA, VA]
              [VC, AC]     [VC, AA]     [VC, VC]     [VC, VA]
              [VA, AC]     [VA, AA]     [VA, VC]     [VA, VA]
   
    so its size is 
         ncore*nact + nact*(nact-1)//2 + nsec*ncore + nsec*nact 
         (ncore: frozen core #,  nact: active #,  nsec: frozen virtual #)

    By default, Hessian is computed by using 1- and 2-RDMs of Quket.state (stored as Quket.DA, Quket.DB, etc.).
    If Quket.DA etc. are None, then they are computed here.
    
    If 1- and 2-RDMs of a specific state are passed, then the code will use them.
    This is useful for state-average calculations by substituting state-averaged density matrices. 
    Note that DA, DB are (norbs, norbs) whereas Daaaa, Dbbbb, Dbaaab are (nact, nact, nact, nact), where nact is the number of active orbitals (n_qubits/2).
    Also, the definitions of DA and Daaaa here are
    DA[p,q] = <p^ q>
    Daaaa[p,q,r,s] = <p^ q^ r s>
    
    If 'state' is given, its 1- and 2-RDMs are explicitly re-computed and used.
    'state' takes a higher preference over 1- and 2-RDMs.

    Args:
        Quket (QuketData): 
        DA (2darray, optional): Alpha 1-particle density matrix (norbs, norbs)
        DB (2darray, optional): Beta 1-particle density matrix (norbs, norbs)
        Daaaa (4darray, optional): Alpha 2-particle density matrix in the active space (nact, nact, nact, nact)
        Dbbbb (4darray, optional): Beta 2-particle density matrix (nact, nact, nact, nact)
        Dbaab (4darray, optional): Beta-Alpha 2-particle density matrix (nact, nact, nact, nact)
        state (QuantumState, optional): QuantumState for which orbital Hessian is computed.
        spinless (bool, optional): If true, spin is integrated. (dimension of Hx is reduced by 1/2)
    Returns:
        Hx (1darray): Hessian @ X 

    Author(s): Takashi Tsuchimochi
    """
    from quket.post import get_1RDM, get_2RDM
    ncore = Quket.n_frozen_orbitals + Quket.n_core_orbitals
    nact = Quket.n_active_orbitals
    norbs = Quket.n_orbitals
    nsec = norbs - ncore - nact 
    if state is not None:
        # Use the inserted quantum state 
        DA, DB = get_1RDM(Quket, state=state)
        Daaaa, Dbaab, Dbbbb = get_2RDM(Quket, state=state)
    elif DA is DB is Daaaa is Dbbbb is Dbaab is None:
        if Quket.DA is None:
            Quket.get_1RDM()
        if Quket.Daaaa is None:
            Quket.get_2RDM()
        DA = Quket.DA
        DB = Quket.DB
        Daaaa = Quket.Daaaa
        Dbaab = Quket.Dbaab
        Dbbbb = Quket.Dbbbb
    elif DA is None or DB is None or Daaaa is None or Dbbbb is None or Dbaab is None:
        raise ValueError(f'Some density matrices are plugged in but not all that are required:'
                         f'DA = {type(DA)}   DB = {type(DB)}  Daaaa = {type(Daaaa)}  Dbbbb = {type(Dbbbb)}  Dbaab = {type(Dbaab)}')

    # Active Space 1RDM
    Daa = DA[ncore:ncore+nact,ncore:ncore+nact]
    Dbb = DB[ncore:ncore+nact,ncore:ncore+nact]


    natt = nact*(nact-1)//2
    nott = norbs*(norbs-1)//2
    
    # Decompose x to each space
    if spinless:
        XA_AC, XA_AA, XA_VC, XA_VA = decompose_CAV(x, ncore, nact, nsec)
        XB_AC = XA_AC.copy() 
        XB_AA = XA_AA.copy()
        XB_VC = XA_VC.copy()
        XB_VA = XA_VA.copy()
    else:
        XA_AC, XA_AA, XA_VC, XA_VA = decompose_CAV(x[:len(x)//2], ncore, nact, nsec)
        XB_AC, XB_AA, XB_VC, XB_VA = decompose_CAV(x[len(x)//2:], ncore, nact, nsec)
    XA_AC = XA_AC.reshape((nact, ncore))
    XA_AA = skew(XA_AA)
    XA_VC = XA_VC.reshape((nsec, ncore))
    XA_VA = XA_VA.reshape((nsec, nact))
    XB_AC = XB_AC.reshape((nact, ncore))
    XB_AA = skew(XB_AA)
    XB_VC = XB_VC.reshape((nsec, ncore))
    XB_VA = XB_VA.reshape((nsec, nact))

    h1 = Quket.one_body_integrals
    Ecore, htilde = get_htilde(Quket)

    Ftilde_A, Ftilde_B = get_Fock(Quket.one_body_integrals, Quket.two_body_integrals, DA, DB)
    # Dzy h~yx
    Dh_A = np.einsum('zy,yx->zx',DA,htilde)
    Dh_B = np.einsum('zy,yx->zx',DB,htilde)  

    ########################
    ## Act-core, Act-core ##
    ########################
    t0_ACAC = time.time()
    ACAC_A, ACAC_B = Hess_ACAC_X(XA_AC, XB_AC, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    ########################
    ## Act-core, Act-Act  ##
    ########################
    t0_ACAA = time.time()
    ACAA_A, ACAA_B = Hess_ACAA_X(XA_AA, XB_AA, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    ########################
    ## Act-core, Vir-core ##
    ########################
    t0_ACVC = time.time()
    ACVC_A, ACVC_B = Hess_ACVC_X(XA_VC, XB_VC, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    ########################
    ## Act-core, Vir-Act  ##
    ########################
    t0_ACVA = time.time()
    ACVA_A, ACVA_B = Hess_ACVA_X(XA_VA, XB_VA, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    ########################
    ## Act-Act, Act-core  ##
    ########################
    t0_AAAC = time.time()
    AAAC_A, AAAC_B = Hess_AAAC_X(XA_AC, XB_AC, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    ########################
    ## Act-Act, Act-Act   ##
    ########################
    t0_AAAA = time.time()
    AAAA_A, AAAA_B = Hess_AAAA_X(XA_AA, XB_AA, ncore, nact, nsec, htilde, Quket.two_body_integrals, Daa, Dbb, Daaaa, Dbaab, Dbbbb)
           
    ########################
    ## Act-Act, Vir-core  ##
    ########################
    t0_AAVC = time.time()
    AAVC_A, AAVC_B = Hess_AAVC_X(XA_VC, XB_VC, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    ########################
    ## Act-Act, Vir-Act   ##
    ########################
    t0_AAVA = time.time()
    AAVA_A, AAVA_B = Hess_AAVA_X(XA_VA, XB_VA, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    #########################
    ## Vir-Core, Act-Core  ##
    #########################
    t0_VCAC = time.time()
    VCAC_A, VCAC_B = Hess_VCAC_X(XA_AC, XB_AC, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    
           
    #########################
    ## Vir-Core, Act-Act   ##
    #########################               
    t0_VCAA = time.time()
    VCAA_A, VCAA_B = Hess_VCAA_X(XA_AA, XB_AA, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    #########################
    ## Vir-Core, Vir-Core  ##
    #########################
    t0_VCVC = time.time()
    VCVC_A, VCVC_B = Hess_VCVC_X(XA_VC, XB_VC, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    #########################
    ## Vir-Core, Vir-Act   ##
    #########################
    t0_VCVA = time.time()
    VCVA_A, VCVA_B = Hess_VCVA_X(XA_VA, XB_VA, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    ########################
    ## Vir-Act, Act-Core  ##
    ########################
    t0_VAAC = time.time()
    VAAC_A, VAAC_B = Hess_VAAC_X(XA_AC, XB_AC, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    
            
    #######################
    ## Vir-Act, Act-Act  ##
    #######################
    t0_VAAA = time.time()
    VAAA_A, VAAA_B = Hess_VAAA_X(XA_AA, XB_AA, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    ########################
    ## Vir-Act, Vir-Core  ##
    ########################   
    t0_VAVC = time.time()
    VAVC_A, VAVC_B = Hess_VAVC_X(XA_VC, XB_VC, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    
    
    #######################
    ## Vir-Act, Vir-Act  ##
    #######################               
    t0_VAVA = time.time()
    VAVA_A, VAVA_B = Hess_VAVA_X(XA_VA, XB_VA, ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    
    t1 = time.time()


    # Multiply by Missing Factor 2
    # and add each contribution to get the entire vector Hess * X
    AC_A = 2* ( ACAC_A + ACAA_A + ACVC_A + ACVA_A)
    AA_A = 2* ( AAAC_A + AAAA_A + AAVC_A + AAVA_A)
    VC_A = 2* ( VCAC_A + VCAA_A + VCVC_A + VCVA_A)
    VA_A = 2* ( VAAC_A + VAAA_A + VAVC_A + VAVA_A)

    AC_B = 2* ( ACAC_B + ACAA_B + ACVC_B + ACVA_B)
    AA_B = 2* ( AAAC_B + AAAA_B + AAVC_B + AAVA_B)
    VC_B = 2* ( VCAC_B + VCAA_B + VCVC_B + VCVA_B)
    VA_B = 2* ( VAAC_B + VAAA_B + VAVC_B + VAVA_B)

    HX_A = compose_CAV(AC_A, AA_A, VC_A, VA_A, ncore, nact, nsec)
    HX_B = compose_CAV(AC_B, AA_B, VC_B, VA_B, ncore, nact, nsec)

    if spinless:
        # Spin-average
        HX_A = (HX_A + HX_B) / 2
        HX_B = HX_A.copy()

    return HX_A, HX_B

 
def AHx(x, Quket, g, DA=None, DB=None, Daaaa=None, Dbbbb=None, Dbaab=None, state=None, spinless=True):
    """
    Augmented Hessian
    """
    t0 = time.time()
    ncore = Quket.n_frozen_orbitals + Quket.n_core_orbitals
    nact = Quket.n_active_orbitals
    norbs = Quket.n_orbitals
    nsec = norbs - ncore - nact 
    natt = nact*(nact-1)//2
    nott = norbs*(norbs-1)//2

    if spinless:
        gx_A = g.T @ x[1:]
        gx_B = gx_A
    else:
        gx_A = g[:nott].T @ x[1:1+nott]  
        gx_B = g[nott:].T @ x[1+nott:]  
    
    HX_A, HX_B = Hx(x[1:], Quket, DA=DA, DB=DB, Daaaa=Daaaa, Dbbbb=Dbbbb, Dbaab=Dbaab, state=state, spinless=spinless)

    for pq in range(nott):
        HX_A[pq] += x[0] * g[pq]
    if spinless:
        HX_A = (HX_A + HX_B)
        HX_B = HX_A
    else:
        for pq in range(nott):
            HX_B[pq] += x[0] * g[pq+nott]

    AHX_A = np.hstack([gx_A, HX_A])
    AHX_B = np.hstack([gx_B, HX_B])
    t1 = time.time()

    return AHX_A, AHX_B
