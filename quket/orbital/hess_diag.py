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

orbital/hess_diag.py

Functions related to orbital hessian (diagonal elements)

"""
import numpy as np
import time

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.linalg import vectorize_skew, skew
from quket.fileio import error, prints, printmat, print_state

def Hess_D_AAAA(ncore, nact, nsec, htilde, hpqrs, Daa, Dbb, Daaaa, Dbaab, Dbbbb):    
    ######################
    ## Act-Act, Act-Act ##
    ######################
    '''
    (1)
    - h[u,t] D[ut] - h[u,t] D[u,t] + h[t,t] D[u,u] + h[u,u] D[t,t]
    (2)
    + (it|jt) D[ij,uu] + (ui|uj) D[tt,ij] - (it|ju) D[ij,tu] - (ui|tj) D[ut,ij]
    (3)
    + ((ut|ij) - (uj|it)) D[ti,uj]  + ((ut|ij) - (uj|it)) D[ti,uj]
    - ((tt|ij) - (tj|it)) D[ui,uj]  - ((uu|ij) - (uj|iu)) D[ti,tj]
    (4)
    - delta[u,u]  (D[t,i] h[i,t] - (ik|jt) D[ij,kt])
    - delta[t,t]  (h[u,i] D[i,u] - (uj|ik) D[ui,jk])
    '''
    natt = (nact-1)*nact//2
    h1act = htilde[ncore:ncore+nact, ncore:ncore+nact]
    h_wxyz = hpqrs[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]

    AA_D_A = np.zeros(natt)
    AA_D_B = np.zeros(natt)
    AA_D_BA = np.zeros(natt)
    # (it|jt) D[ij,uu] + (ui|uj) D[tt,ij] - (it|ju) D[ij,tu] - (ui|tj) D[ut,ij]
    # Int1[tu] = (it|jt) D[ij,uu] 
    # Int2[tu] = (it|ju) D[ij,tu]
    Int1_A = np.einsum('itjt,ijuu->tu',h_wxyz, Daaaa)
    Int2_A = np.einsum('itju,ijtu->tu',h_wxyz, Daaaa)
    Int1_B = np.einsum('itjt,ijuu->tu',h_wxyz, Dbbbb)
    Int2_B = np.einsum('itju,ijtu->tu',h_wxyz, Dbbbb)
    '''
    (2)
    + (iBtB|jAtA) D[iBjA,uAuB] + (uBiB|uAjA) D[tAtB,iBjA] - (iBtB|jAuA) D[iBjA,tAuB] - (uBiB|tAjA) D[uAtB,iBjA]
    '''
    Int1_BA = np.einsum('itjt,ijuu->tu',h_wxyz, Dbaab) 
    Int2_BA = np.einsum('itju,ijtu->tu',h_wxyz, Dbaab)

    '''
    (3)
    +2 ((ut|ij) - (uj|it)) D[ti,uj]  
    #   aa aa     aa aa     aa aa   
    #   aa bb               ab ab   
    - ((tt|ij) - (tj|it)) D[ui,uj]  - ((uu|ij) - (uj|iu)) D[ti,tj]
    #   aa aa     aa aa     aa aa       aa aa     aa aa     aa aa
    #   aa bb               ab ab       aa bb               ab ab
    '''
    Int3_sum = -2*np.einsum('utij,ituj->tu',h_wxyz, Daaaa+Dbaab)
    Int3_A = Int3_sum - 2* np.einsum('ujit,tiuj->tu',h_wxyz, Daaaa) 
    Int3_B = Int3_sum - 2* np.einsum('ujit,tiuj->tu',h_wxyz, Dbbbb) 
    Int4_A = np.einsum('ttij,iuuj->tu', h_wxyz, Daaaa + Dbaab) + np.einsum('tjit,uiuj->tu', h_wxyz, Daaaa)
    Int4_B = np.einsum('ttij,uiju->tu', h_wxyz, Daaaa + Dbaab) + np.einsum('tjit,uiuj->tu', h_wxyz, Dbbbb)

    '''
    - (uAjA|iBtB)) D[tAiB,uBjA]  - (uBjB|iAtA) D[tBiA,uAjB]
    + (tAjA|iBtB)) D[uAiB,uBjA]  + (uBjB|iAuA)) D[tBiA,tAjB]
    '''
    Int3_BA = -np.einsum('ujit,itju->tu', h_wxyz, Dbaab) - np.einsum('ujit,tiuj->tu',h_wxyz, Dbaab)
    Int4_BA = np.einsum('tjit,iuju->tu', h_wxyz, Dbaab) + np.einsum('ujiu,titj->tu', h_wxyz, Dbaab)

    '''
    (4)
    -  (D[t,i] h[i,t] - (ik|jt) D[ij,kt])
                         bb aa    ba ba
    -  (h[u,i] D[i,u] - (uj|ik) D[ui,jk])
                         aa aa    
    '''
    # (iAkA|jAtA) D[iAjA,kAwA] + (iBkB|jAtA) D[iBjA,kBwA]
    Dh_A = Daa @ h1act
    Dh_B = Daa @ h1act
    Int5_A = np.einsum('ikjt,ijtk->t',h_wxyz, Daaaa+Dbaab)
    Int5_B = np.einsum('ikjt,jikt->t',h_wxyz, Dbbbb+Dbaab)
    tu = 0
    for t in range(nact):
        for u in range(t):
            AA_D_A[tu] += - 2* h1act[u,t] * Daa[u,t] + h1act[t,t] * Daa[u,u] + h1act[u,u] * Daa[t,t]
            AA_D_A[tu] += Int1_A[t,u] + Int1_A[u,t] - Int2_A[t,u] -  Int2_A[u,t]  
            AA_D_A[tu] += Int3_A[t,u] + Int4_A[t,u] + Int4_A[u,t] 
            AA_D_A[tu] -= Dh_A[t,t] + Dh_A[u,u] + Int5_A[t] + Int5_A[u]
            AA_D_B[tu] += - 2* h1act[u,t] * Dbb[u,t] + h1act[t,t] * Dbb[u,u] + h1act[u,u] * Dbb[t,t]
            AA_D_B[tu] += Int1_B[t,u] + Int1_B[u,t] - Int2_B[t,u] -  Int2_B[u,t]  
            AA_D_B[tu] += Int3_B[t,u] + Int4_B[t,u] + Int4_B[u,t] 
            AA_D_B[tu] -= Dh_B[t,t] + Dh_B[u,u] + Int5_B[t] + Int5_B[u]
            AA_D_BA[tu] = Int1_BA[t,u] + Int1_BA[u,t] - Int2_BA[t,u] - Int2_BA[u,t] + Int3_BA[t,u] + Int4_BA[t,u] 
            tu += 1
    return AA_D_A, AA_D_BA, AA_D_B

def Hess_D_ACAC(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):    
    ########################
    ## Act-core, Act-core ##
    ########################
    h_IJxy = hpqrs[:ncore,:ncore,ncore:ncore+nact,ncore:ncore+nact]    
    h_IxJy = hpqrs[:ncore,ncore:ncore+nact,:ncore,ncore:ncore+nact]
    h_wxyz = hpqrs[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]

    ACAC_D_A = np.zeros((nact*ncore))
    ACAC_D_B = np.zeros((nact*ncore)) 
    ACAC_D_BA = np.zeros((nact*ncore)) 
    ### Intermediates
    # ((Ix|Iy) - (II|yx)) Dyx = Int1_AA[x,I]
    Int1_AA = np.einsum('IyIx, yx->xI', h_IxJy, Daa) - np.einsum('IIxy,yx->xI',h_IJxy, Daa)
    Int1_BB = np.einsum('IyIx, yx->xI', h_IxJy, Dbb) - np.einsum('IIxy,yx->xI',h_IJxy, Dbb)

    Int1_BA = 2* np.einsum('IxIy, xy->xI', h_IxJy, Daa + Dbb) 

    # ((II|yw) - (Iw|Iy)) Dxy,wx
    # ((IaJa|yawa) - (Iawa|Jaya)) Dxaya,waza  + (IaJa|ybwb) Dxayb,wbza 
    # = ((IaJa|yawa) (Dxaya,waza + Dxayb,wbza)  - (Iawa|Jaya) Dxaya,waza 
    # = ((IaJa|yawa) (Dyxa,zawa + Dybxa,zawb)  - (Iawa|Jaya) Dxaya,waza 
    Int3_AA = np.einsum('IIyw,yxxw->xI',h_IJxy, Daaaa+Dbaab) - np.einsum('IwIy,yxxw->xI',h_IxJy, Daaaa)
    # ((IbJb|ybwb) - (Ibwb|Jbyb)) Dxbyb,wbzb  + (IbJb|yawa) Dxbya,wazb
    # = ((IbJb|ybwb) (Dxbyb,zbwb + Dxbya,wazb)  - (Ibwb|Jbyb) Dxbyb,zbwb   
    Int3_BB = np.einsum('IIyw,xywx->xI',h_IJxy, Dbbbb+Dbaab) - np.einsum('IwIy,yxxw->xI',h_IxJy, Dbbbb)    
    Int3_BA = np.einsum('IwIy,xyxw->xI',h_IxJy, Dbaab)
    # (Iy|Iw) Dxx,wy
    # (Iaya|Iawa) Dxaxa,waya
    Int4_AA = np.einsum('IyIw,xxwy->xI',h_IxJy, Daaaa)
    Int4_BB = np.einsum('IyIw,xxwy->xI',h_IxJy, Dbbbb) 
    Int4_BA = -np.einsum('IyIw,xxwy->xI',h_IxJy, Dbaab)
    # (yv|wx) Dyw,xv
    # (yava|waxa) Dyawa,xava +     (ybvb|waxa) Dybwa,xavb  
    Int5_AA = np.einsum('yvwx,ywxv->x',h_wxyz, Daaaa + Dbaab) 
    # (ybvb|wbxb) Dybwb,xbvb +     (yava|wbxb) Dyawb,xbva 
    Int5_BB = np.einsum('yvwx,wyvx->x',h_wxyz, Dbbbb + Dbaab)     
    xI = 0
    for x in range(nact):
        for I in range(ncore):
            ACAC_D_A[xI] = htilde[I,I] * Daa[x,x] 
            ACAC_D_A[xI] += 2*h_IxJy[I,x,I,x] - h_IxJy[I,x,I,x] - h_IJxy[I,I,x,x]
            ACAC_D_A[xI] += - Int1_AA[x,I] - Int1_AA[x,I]
            ACAC_D_A[xI] += + Int3_AA[x,I] 
            ACAC_D_A[xI] += - Int4_AA[x,I]
            
            ACAC_D_B[xI] = htilde[I,I] * Dbb[x,x] 
            ACAC_D_B[xI] += 2*h_IxJy[I,x,I,x] - h_IxJy[I,x,I,x] - h_IJxy[I,I,x,x]
            ACAC_D_B[xI] += - Int1_BB[x,I] - Int1_BB[x,I]
            ACAC_D_B[xI] += + Int3_BB[x,I] 
            ACAC_D_B[xI] += - Int4_BB[x,I]

            
            ACAC_D_A[xI] += - Dh_A[x+ncore,x+ncore]  + Ftilde_A[x+ncore,x+ncore] - Int5_AA[x]
            ACAC_D_B[xI] += - Dh_B[x+ncore,x+ncore]  + Ftilde_B[x+ncore,x+ncore] - Int5_BB[x]
            ACAC_D_A[xI] -= Ftilde_A[I,I]
            ACAC_D_B[xI] -= Ftilde_B[I,I]

            ACAC_D_BA[xI] = 2*h_IxJy[I,x,I,x] - Int1_BA[x,I] + Int3_BA[x,I] - Int4_BA[x,I]
            xI += 1
    return ACAC_D_A, ACAC_D_BA, ACAC_D_B

def Hess_D_VCVC(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    ########################
    ## Vir-core, Vir-core ##
    ########################
    h_AIBJ = hpqrs[ncore+nact:, :ncore, ncore+nact:, :ncore]
    h_ABIJ = hpqrs[ncore+nact:, ncore+nact:, :ncore, :ncore]
    VCVC_D_A = np.zeros(nsec*ncore)
    VCVC_D_B = np.zeros(nsec*ncore)
    VCVC_D_BA = np.zeros(nsec*ncore)
    AI = 0
    for A in range(nsec):
        for I in range(ncore):
            VCVC_D_A[AI] = h_AIBJ[A,I,A,I] - h_ABIJ[A,A,I,I]
            VCVC_D_B[AI] = VCVC_D_A[AI]
            VCVC_D_A[AI] += Ftilde_A[A+ncore+nact,A+ncore+nact]
            VCVC_D_B[AI] += Ftilde_B[A+ncore+nact,A+ncore+nact]
            VCVC_D_A[AI] -= Ftilde_A[I,I]
            VCVC_D_B[AI] -= Ftilde_B[I,I]
            VCVC_D_BA[AI] = 2* h_AIBJ[A,I,A,I] 
            AI += 1
    return VCVC_D_A, VCVC_D_BA, VCVC_D_B

def Hess_D_VAVA(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #######################
    ## Vir-Act, Vir-Act  ##
    #######################   
    VAVA_D_A = np.zeros((nact*nsec), dtype=float)
    VAVA_D_B = np.zeros((nact*nsec), dtype=float)
    VAVA_D_BA = np.zeros((nact*nsec), dtype=float)
    h_ABxy = hpqrs[ ncore+nact:, ncore+nact:, ncore:ncore+nact, ncore:ncore+nact]
    h_AxBy = hpqrs[ ncore+nact:, ncore:ncore+nact, ncore+nact:, ncore:ncore+nact]
    h_wxyz = hpqrs[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    # (At|Au) Dtu,yy
    Int1_AA = np.einsum('AtAu, tuyy -> Ay', h_AxBy, Daaaa)
    Int1_BB = np.einsum('AtAu, tuyy -> Ay', h_AxBy, Dbbbb)    
    Int1_BA = -np.einsum('AtAu, tuyy -> Ay', h_AxBy, Dbaab)
    # ((AA|tu) - (At|Au)) Dyt,uy
    # (AaAa|taua) Dyata,uaya + (AaAa|tbub) Dyatb,ubya - (Aata|Aaua) Dyata,uaya
    Int2_AA = np.einsum('AAtu, tyyu->Ay',h_ABxy, Daaaa + Dbaab) - np.einsum('AtAu,ytuy->Ay',h_AxBy, Daaaa)
    Int2_BB = np.einsum('AAtu, ytuy->Ay',h_ABxy, Dbbbb + Dbaab) - np.einsum('AtAu,ytuy->Ay',h_AxBy, Dbbbb)
    Int2_AB = np.einsum('AtAu,ytyu->Ay',h_AxBy, Dbaab)  
    # (yt|vu) Dyv,ut
    # (yata|vaua) Dyava, uata + (yata|vbub) Dyavb,ubta
    Int3_AA = np.einsum('ytvu, vytu->y',h_wxyz,Daaaa+Dbaab)
    Int3_BB = np.einsum('ytvu, yvut->y',h_wxyz,Daaaa+Dbaab)    
    Ay = 0
    for A in range(nsec):
        for y in range(nact):
            VAVA_D_A[Ay] = htilde[A+ncore+nact,A+ncore+nact] * Daa[y,y] - Int1_AA[A,y] + Int2_AA[A,y]
            VAVA_D_B[Ay] = htilde[A+ncore+nact,A+ncore+nact] * Dbb[y,y] - Int1_BB[A,y] + Int2_BB[A,y]
            VAVA_D_A[Ay] -= ( Dh_A[y+ncore,y+ncore] + Int3_AA[y] )
            VAVA_D_B[Ay] -= ( Dh_B[y+ncore,y+ncore] + Int3_BB[y] )
            VAVA_D_BA[Ay] = -Int1_BA[A,y] + Int2_AB[A,y]
            Ay += 1
    return VAVA_D_A, VAVA_D_BA, VAVA_D_B

def Hess_diag(Quket, DA=None, DB=None, Daaaa=None, Dbbbb=None, Dbaab=None, state=None, spinless=True):
    """
    Compute the diagonal elements of orbital Hessian.
    """
    from quket.post import get_1RDM, get_2RDM
    from .misc import get_htilde, get_Fock, compose_CAV
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
    h1 = Quket.one_body_integrals
    Ecore, htilde = get_htilde(Quket)
    Ftilde_A, Ftilde_B = get_Fock(Quket.one_body_integrals, Quket.two_body_integrals, DA, DB)
    # Dzy h~yx
    Dh_A = np.einsum('zy,yx->zx',DA,htilde)
    Dh_B = np.einsum('zy,yx->zx',DB,htilde)  

    AC_D_A, AC_D_BA, AC_D_B = Hess_D_ACAC(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb) 
    AA_D_A, AA_D_BA, AA_D_B = Hess_D_AAAA(ncore, nact, nsec, htilde, Quket.two_body_integrals, Daa, Dbb, Daaaa, Dbaab, Dbbbb)
    VC_D_A, VC_D_BA, VC_D_B = Hess_D_VCVC(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)
    VA_D_A, VA_D_BA, VA_D_B = Hess_D_VAVA(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)
    
    # Sort vector and multiply by 2
    HD_A = 2* compose_CAV(AC_D_A, AA_D_A, VC_D_A, VA_D_A, ncore, nact, nsec)
    HD_B = 2* compose_CAV(AC_D_B, AA_D_B, VC_D_B, VA_D_B, ncore, nact, nsec)
    HD_BA =2* compose_CAV(AC_D_BA, AA_D_BA, VC_D_BA, VA_D_BA, ncore, nact, nsec)
    if spinless:
       HD_A = (HD_A + HD_B + HD_BA *2)/2
       HD_BA = HD_A
       HD_B = HD_A
    return HD_A, HD_BA, HD_B
