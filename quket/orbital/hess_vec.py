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

orbital/hess_vec.py

Functions related to orbital hessian (matrix-vector multiplication)

"""
import numpy as np
import time

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.linalg import vectorize_skew, skew
from quket.fileio import error, prints, printmat, print_state
from .misc import *

###############################################
# Orbital Hessian for subspaces (Contraction) #
###############################################

def Hess_AAAA_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Daa, Dbb, Daaaa, Dbaab, Dbbbb):    
    ######################
    ## Act-Act, Act-Act ##
    ######################
    '''
    (1)
    - h[w,t] D[uv] - h[u,v] D[w,t] + h[v,t] D[u,w] + h[u,w] D[v,t]
    (2)
    + (it|jv) D[ij,wu] + (ui|wj) D[vt,ij] - (it|jw) D[ij,vu] - (ui|vj) D[wt,ij]
    (3)
    + ((wt|ij) - (wj|it)) D[vi,uj]  + ((uv|ij) - (uj|iv)) D[ti,wj]
    - ((vt|ij) - (vj|it)) D[wi,uj]  - ((uw|ij) - (uj|iw)) D[ti,vj]
    (4)
    + delta[v,u]  (D[w,i] h[i,t] - (ik|jt) D[ij,kw])
    + delta[t,w]  (h[u,i] D[i,v] - (uj|ik) D[vi,jk])
    - delta[v,u]  (D[v,i] h[i,t] - (ik|jt) D[ij,kv])
    - delta[t,v]  (h[u,i] D[i,w] - (uj|ik) D[wi,jk])
    '''

    natt = (nact-1)*nact//2
    AAAA_A = np.zeros((natt))
    AAAA_B = np.zeros((natt))
    
    h1act = htilde[ncore:ncore+nact, ncore:ncore+nact]
    h_wxyz = hpqrs[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    '''
    (1)
    - h[w,t] D[u,v] Xvw - h[u,v] D[w,t] Xvw + h[v,t] D[u,w] Xvw + h[u,w] D[v,t] Xvw
    '''
    # - hwt Xvw Duv - huv Xvw Dwt + hvt Xvw Duw + huw Xvw Dvt
    # = -DXh_ut - hXD_ut 
    DXh_A = Daa @ XA @ h1act
    DXh_B = Dbb @ XB @ h1act
    hXD_A = h1act @ XA @ Daa
    hXD_B = h1act @ XB @ Dbb
    AAAA_A = - DXh_A.T #- hXD_A.T 
    AAAA_B = - DXh_B.T #- hXD_B.T 

    '''
    (2)
    + (it|jv) Xvw D[ij,wu] + (ui|wj) Xvw D[vt,ij] - (it|jw) Xvw D[ij,vu] - (ui|vj) Xvw D[wt,ij]
    '''
    # =  Int1[i,t,j,w] Dij,wu  -  (t<->u)
    # <-  Int1[ia,ta,ja,wa] Diaja,waua + Int1[ia,ta,jb,wb] Diajb,wbua   -  (t<->u)
    # Int1[i,t,j,w] = (it|jv) Xvw
    Int1_A = np.einsum('itjv,vw->itjw', h_wxyz, XA)
    Int1_B = np.einsum('itjv,vw->itjw', h_wxyz, XB)
    AAAA_A += np.einsum('itjw, ijwu -> tu', Int1_A, Daaaa)\
            + np.einsum('itjw, jiuw -> tu', Int1_B, Dbaab )
    AAAA_B += np.einsum('itjw, ijwu -> tu', Int1_B, Daaaa)\
            + np.einsum('itjw, ijwu -> tu', Int1_A, Dbaab )
             

    '''
    (3)
    + ((wt|ij) - (wj|it)) Xvw D[vi,uj]  + ((uv|ij) - (uj|iv)) Xvw D[ti,wj]
    - ((vt|ij) - (vj|it)) Xvw D[wi,uj]  + ((uw|ij) - (uj|iw)) Xvw D[ti,vj]
    '''
    # + ((ij|tw) - (it|jw)) Xvw D[vi,uj]  -   (t <-> u) 
    # = -Int1[i,j,t,v] D[vi,uj]  + Int1[i,t,j,v] D[vi,uj]  
    # <- - Int1[ia,ja,ta,va] Dvaia,uaja - Int1[ib,jb,ta,va] Dvaib,uajb  + Int1[ia,ta,ja,va] Dvaia,uaja + Int1[ia,ta,jb,vb] Dvbia,uajb
    AAAA_A += np.einsum('ijtv,ivuj->tu', Int1_A, Daaaa+Dbaab)\
            + np.einsum('itjv,viuj->tu', Int1_A, Daaaa)\
            + np.einsum('itjv,viuj->tu', Int1_B, Dbaab)
    AAAA_B += np.einsum('ijtv,viju->tu', Int1_B, Dbbbb+Dbaab)\
            + np.einsum('itjv,viuj->tu', Int1_B, Dbbbb)\
            + np.einsum('itjv,ivju->tu', Int1_A, Dbaab)

    '''
    (4)
    + 0.5 delta[v,u]  (Dwi hit + Dti hiw - (ik|jt) D[ij,kw] - (ik|jw) D[ij,kt]) Xvw
    + 0.5 delta[t,w]  (hui Div + hvi Diu - (uj|ik) D[vi,jk] - (vj|ik) D[ui,jk]) Xvw
    - 0.5 delta[v,u]  (Dvi hit + Dti hiv - (ik|jt) D[ij,kv] - (ik|jv) D[ij,kt]) Xvw
    - 0.5 delta[t,v]  (hui Diw + hwi Diu - (uj|ik) D[wi,jk] - (wj|ik) D[ui,jk]) Xvw
    '''
    hD_A = h1act @ Daa
    hD_B = h1act @ Dbb
    # tu=A, vw=A
    # (iAkA|jAtA) D[iAjA,kAwA] + (iBkB|jAtA) D[iBjA,kBwA]
    Int2_A = np.einsum('ikjt,ijkw->tw',h_wxyz, Daaaa) - np.einsum('ikjt,ijwk->tw',h_wxyz, Dbaab)
    # tu=B, vw=B 
    # (iBkB|jBtB) D[iBjB,kBwB] + (iAkA|jBtB) D[iAjB,kAwB]
    Int2_B = np.einsum('ikjt,ijkw->tw',h_wxyz, Dbbbb) - np.einsum('ikjt,jikw->tw',h_wxyz, Dbaab)
    AAAA_A += 0.5 * np.einsum('tw,uw-> tu', (hD_A + hD_A.T - Int2_A - Int2_A.T), XA) 
    AAAA_B += 0.5 * np.einsum('tw,uw-> tu', (hD_B + hD_B.T - Int2_B - Int2_B.T), XB) 

    AAAA_A = vectorize_skew(AAAA_A - AAAA_A.T)
    AAAA_B = vectorize_skew(AAAA_B - AAAA_B.T)
    return AAAA_A, AAAA_B

def Hess_ACAC_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):    
    ########################
    ## Act-core, Act-core ##
    ########################
    h_IJxy = hpqrs[:ncore,:ncore,ncore:ncore+nact,ncore:ncore+nact]    
    h_IxJy = hpqrs[:ncore,ncore:ncore+nact,:ncore,ncore:ncore+nact]
    h_wxyz = hpqrs[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]

    ACAC_A = np.zeros((nact*ncore))
    ACAC_B = np.zeros((nact*ncore)) 
    # Dxz XzJ hIJ
    DXh_A = Daa @ XA @ htilde[:ncore,:ncore]
    DXh_B = Dbb @ XB @ htilde[:ncore,:ncore]
    ACAC_A = DXh_A 
    ACAC_B = DXh_B 
    DhX_A = Dh_A[ncore:ncore+nact, ncore:ncore+nact] @ XA
    DhX_B = Dh_B[ncore:ncore+nact, ncore:ncore+nact] @ XB
    # (2(Ix|Jz) - (Iz|Jx) - (IJ|xz)) XzJ
    ACAC_A += 2* np.einsum('IxJz,zJ->xI',h_IxJy, XA+XB) - np.einsum('IzJx,zJ->xI',h_IxJy, XA) - np.einsum('IJxz,zJ->xI',h_IJxy, XA)
    ACAC_B += 2* np.einsum('IxJz,zJ->xI',h_IxJy, XA+XB) - np.einsum('IzJx,zJ->xI',h_IxJy, XB) - np.einsum('IJxz,zJ->xI',h_IJxy, XB)

    ### Intermediates
    # (2(Ix|Jy) - (IJ|xy) - (Iy|Jx)) XzJ Dzy
    Int1_A = 2* np.einsum('IxJy, zy->xIzJ', h_IxJy, Daa) 
    Int1_B = 2* np.einsum('IxJy, zy->xIzJ', h_IxJy, Dbb) 
    Int2_BA = 2* np.einsum('IyJz, yx->xIzJ', h_IxJy, Dbb) 
    Int1_AA = Int1_A - np.einsum('IJxy,zy->xIzJ',h_IJxy, Daa) - np.einsum('JxIy,zy->xIzJ',h_IxJy,Daa)
    Int1_BB = Int1_B - np.einsum('IJxy,zy->xIzJ',h_IJxy, Dbb) - np.einsum('JxIy,zy->xIzJ',h_IxJy,Dbb)
    ACAC_A -= np.einsum('xIzJ,zJ->xI',Int1_AA, XA) \
            + np.einsum('zJxI,zJ->xI',Int1_AA, XA) \
            + np.einsum('xIzJ,zJ->xI',Int1_B, XB)\
            + np.einsum('zJxI,zJ->xI',Int1_A, XB)
    ACAC_B -= np.einsum('xIzJ,zJ->xI',Int1_BB, XB) \
            + np.einsum('zJxI,zJ->xI',Int1_BB, XB) \
            + np.einsum('xIzJ,zJ->xI',Int1_A, XA)\
            + np.einsum('zJxI,zJ->xI',Int1_B, XA)

    # ((IJ|yw) - (Iw|Jy)) XzJ Dxy,wz
    # Int3[Ia,za,ya,wa] Dxaya,waza + Int3[Ia,za,yb,wb] Dxayb,wbza - Int4[Ia,za,ya,wa] Dxaya,waza  - Int4[Ia,zb,yb,wa] Dxayb,wazb
    Int3_A = np.einsum('IJyw,zJ->Izyw', h_IJxy, XA)
    Int3_B = np.einsum('IJyw,zJ->Izyw', h_IJxy, XB)
    Int4_A = np.einsum('IwJy,zJ->Izyw', h_IxJy, XA)
    Int4_B = np.einsum('IwJy,zJ->Izyw', h_IxJy, XB)
    ACAC_A += np.einsum('Izyw,yxzw->xI', Int3_A, Daaaa+Dbaab)\
            - np.einsum('Izyw,xywz->xI', Int4_A, Daaaa)\
            + np.einsum('Izyw,yxwz->xI', Int4_B, Dbaab)
    ACAC_B += np.einsum('Izyw,xywz->xI', Int3_B, Dbbbb+Dbaab)\
            - np.einsum('Izyw,xywz->xI', Int4_B, Dbbbb)\
            + np.einsum('Izyw,xyzw->xI', Int4_A, Dbaab)

    # -(Iy|Jw) XzJ Dzx,wy
    # = -Int4[I,z,w,y] Dzx,wy
    # <- -Int4[Ia,za,wa,ya] Dzaxa,waya - Int4[Ia,zb,wb,ya] Dzbxa,wbya
    ACAC_A += np.einsum('Izwy,zxyw->xI',Int4_A, Daaaa)+ np.einsum('Izwy,zxyw->xI',Int4_B,Dbaab)
    ACAC_B += np.einsum('Izwy,zxyw->xI',Int4_B, Dbbbb)+ np.einsum('Izwy,xzwy->xI',Int4_A,Dbaab)
    Int4_BA = -np.einsum('IyJw,xzwy->xIzJ',h_IxJy, Dbaab)

    # 0.5 delta_IJ (- Dh_zx - Dh_xz + Fzx + Fxz - (yv|wx) Dyw,zv - (yv|wz) Dyw,xv) XzJ
    # (yv|wx) Dyw,zv
    # (yava|waxa) Dyawa,zava +     (ybvb|waxa) Dybwa,zavb  
    Int5_AA = np.einsum('yvwx,ywzv->zx',h_wxyz, Daaaa + Dbaab) 
    # (ybvb|wbxb) Dybwb,zbvb +     (yava|wbxb) Dyawb,zbva 
    Int5_BB = np.einsum('yvwx,wyvz->zx',h_wxyz, Dbbbb + Dbaab)     

    Dh_A_act = Dh_A[ncore:ncore+nact, ncore:ncore+nact]
    Dh_B_act = Dh_B[ncore:ncore+nact, ncore:ncore+nact]
    Ftilde_A_act = Ftilde_A[ncore:ncore+nact, ncore:ncore+nact]
    Ftilde_B_act = Ftilde_B[ncore:ncore+nact, ncore:ncore+nact]
    ACAC_A += 0.5 * (- Dh_A_act - Dh_A_act.T + Ftilde_A_act + Ftilde_A_act.T - Int5_AA - Int5_AA.T) @ XA
    ACAC_B += 0.5 * (- Dh_B_act - Dh_B_act.T + Ftilde_B_act + Ftilde_B_act.T - Int5_BB - Int5_BB.T) @ XB

    # -0.5 (FIJ + FJI) XzJ
    ACAC_A -= 0.5 * XA @ (Ftilde_A[:ncore, :ncore] + Ftilde_A[:ncore, :ncore].T)
    ACAC_B -= 0.5 * XB @ (Ftilde_B[:ncore, :ncore] + Ftilde_B[:ncore, :ncore].T)
    ACAC_A = ACAC_A.reshape(nact*ncore) 
    ACAC_B = ACAC_B.reshape(nact*ncore) 
    return ACAC_A, ACAC_B

def Hess_ACAA_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    ########################
    ## Act-Core, Act-Act  ##
    ########################    
    natt = nact*(nact-1)//2
    h_Ixyz = hpqrs[:ncore,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
   
    ACAA_A = np.zeros((nact*ncore), dtype=float)
    ACAA_B = np.zeros((nact*ncore), dtype=float)
    # Dzx Xxy hyJ 
    DXh_A = Daa @ XA @ htilde[ncore:ncore+nact, :ncore]
    DXh_B = Dbb @ XB @ htilde[ncore:ncore+nact, :ncore]
    ACAA_A = DXh_A
    ACAA_B = DXh_B

    # ( 2(Jz|xw) - (Jx|wz) - (Jw|xz) ) (XD)xw
    XD_A = XA @ Daa
    XD_B = XB @ Dbb
    ACAA_A += 2* np.einsum('Jzxw,xw->zJ', h_Ixyz, XD_A+XD_B) - np.einsum('Jxwz,xw->zJ',h_Ixyz, XD_A) - np.einsum('Jwxz, xw->zJ', h_Ixyz, XD_A)
    ACAA_B += 2* np.einsum('Jzxw,xw->zJ', h_Ixyz, XD_A+XD_B) - np.einsum('Jxwz,xw->zJ',h_Ixyz, XD_B) - np.einsum('Jwxz, xw->zJ', h_Ixyz, XD_B)

    # - (Jw|yv) Xxy Dzx,wv
    # = -Int1[J,w,x,v] Dzx,wv
    # <- -Int1[Ja,wa,xa,va] Dzaxa,wava - Int1[Ja,wa,xb,vb] Dzaxb,wavb
    Int1_A = np.einsum('Jwyv,xy->Jwxv', h_Ixyz, XA)
    Int1_B = np.einsum('Jwyv,xy->Jwxv', h_Ixyz, XB)
    ACAA_A += np.einsum('Jwxv,xzwv->zJ', Int1_A, Daaaa)\
            + np.einsum('Jwxv,xzwv->zJ', Int1_B, Dbaab)
    ACAA_B += np.einsum('Jwxv,xzwv->zJ', Int1_B, Dbbbb)\
            + np.einsum('Jwxv,zxvw->zJ', Int1_A, Dbaab)


    # - ((Jx|vw) - (Jw|vx)) Xxy Dzv,wy
    # = - (Jx|vw) Xxy Dzv,wy + (Jw|yv) Xxy Dzv,wx
    # = - Int2[J,y,v,w] Dzv,wy + Int1[J,w,x,v] Dzv,wx
    # = - Int2[Ja,ya,va,wa] Dzava,waya - Int2[Ja,ya,vb,wb] Dzavb,wbya + Int1[Ja,wa,xa,va] Dzava,waxa + Int1[Ja,wa,xb,vb] Dzavb,waxb
    #
    Int2_A = np.einsum('Jxvw,xy->Jyvw',h_Ixyz, XA)
    Int2_B = np.einsum('Jxvw,xy->Jyvw',h_Ixyz, XB)
    ACAA_A -= np.einsum('Jyvw,vzyw->zJ',Int2_A, Daaaa+ Dbaab) \
            + np.einsum('Jwxv,zvwx->zJ',Int1_A, Daaaa) \
            - np.einsum('Jwxv,vzwx->zJ',Int1_B, Dbaab)
    ACAA_B -= np.einsum('Jyvw,zvwy->zJ',Int2_B, Dbbbb+ Dbaab) \
            + np.einsum('Jwxv,zvwx->zJ',Int1_B, Dbbbb) \
            - np.einsum('Jwxv,zvxw->zJ',Int1_A, Dbaab)

    # 0.5 delta_zy Xxy (FJx + Dh_xJ + (Jt|wu) Dxw,ut)  
    # (Jt|wu) Dxw,ut
    # (Jata|waua) Dxawa,uata + (Jata|wbub) Dxawb,ubta    
    Int4_AA = np.einsum('Jtwu, wxtu -> xJ', h_Ixyz, Daaaa+Dbaab)
    Int4_BB = np.einsum('Jtwu, xwut -> xJ', h_Ixyz, Dbbbb+Dbaab)    
    ACAA_A -= 0.5* XA @ (Ftilde_A[:ncore, ncore:ncore+nact].T + Dh_A[ncore:ncore+nact, :ncore] + Int4_AA)
    ACAA_B -= 0.5* XB @ (Ftilde_B[:ncore, ncore:ncore+nact].T + Dh_B[ncore:ncore+nact, :ncore] + Int4_BB)
    ACAA_A = ACAA_A.reshape(nact*ncore)
    ACAA_B = ACAA_B.reshape(nact*ncore)
    return ACAA_A, ACAA_B

def Hess_AAAC_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    ########################
    ## Act-Act, Act-core  ##
    ########################    
    norbs = ncore + nact + nsec
    natt = nact*(nact-1)//2
    h_Ixyz = hpqrs[:ncore,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
   
    AAAC_A = np.zeros((natt), dtype=float)
    AAAC_B = np.zeros((natt), dtype=float)

    # htilde[y,J] D[x,z] X[z,J]
    DXh_A = Daa @ XA @ htilde[:ncore, ncore:ncore+nact]
    DXh_B = Dbb @ XB @ htilde[:ncore, ncore:ncore+nact]

    # X[x,J] Ftilde[J,y]
    XF_A = XA @ Ftilde_A[:ncore, ncore:ncore+nact]
    XF_B = XB @ Ftilde_B[:ncore, ncore:ncore+nact]
    XDh_A = XA @ (Dh_A[ncore:ncore+nact, :ncore]).T
    XDh_B = XB @ (Dh_B[ncore:ncore+nact, :ncore]).T
   
    # (Jt|wu) Dxw,ut
    # (Jata|waua) Dxawa,uata + (Jata|wbub) Dxawb,ubta    
    Int_A = np.einsum('Jtwu, wxtu -> Jx', h_Ixyz, Daaaa+Dbaab)
    Int_B = np.einsum('Jtwu, xwut -> Jx', h_Ixyz, Dbbbb+Dbaab)    
    Int_A = XA @ Int_A
    Int_B = XB @ Int_B

    AAAC_A = + DXh_A - DXh_A.T
    AAAC_B = + DXh_B - DXh_B.T
    AAAC_A+= - 0.5 * (XF_A - XF_A.T + XDh_A - XDh_A.T + Int_A - Int_A.T)
    AAAC_B+= - 0.5 * (XF_B - XF_B.T + XDh_B - XDh_B.T + Int_B - Int_B.T)

    # ( 2(Jz|xw) - (Jx|wz) - (Jw|xz) ) Dyw XzJ
    Int1_BA = 2* np.einsum('Jzxw,zJ->xw', h_Ixyz, XA) 
    Int1_AB = 2* np.einsum('Jzxw,zJ->xw', h_Ixyz, XB) 
    Int1_AA = Int1_BA - np.einsum('Jxwz,zJ->xw',h_Ixyz,XA) - np.einsum('Jwxz,zJ->xw',h_Ixyz,XA)
    Int1_BB = Int1_AB - np.einsum('Jxwz,zJ->xw',h_Ixyz,XB) - np.einsum('Jwxz,zJ->xw',h_Ixyz,XB)
    # Int1_A[xw] Dwy 
    Int1_AA = Int1_AA @ Daa
    Int1_BB = Int1_BB @ Dbb
    Int1_BA = Int1_BA @ Daa
    Int1_AB = Int1_AB @ Dbb
    AAAC_A += Int1_AA - Int1_AA.T + Int1_AB - Int1_AB.T
    AAAC_B += Int1_BB - Int1_BB.T + Int1_BA - Int1_BA.T

    # (Jw|yv) Dzx,wv
    # XzJ (Jw|yv)
    # XzaJa(Jawa|yava)
    Int2_A = np.einsum('zJ,Jwyv->zwyv',XA, h_Ixyz) 
    # XzbJb(Jbwb|ybvb)
    Int2_B = np.einsum('zJ,Jwyv->zwyv',XB, h_Ixyz) 
    # XzaJa(Jawa|yava) Dzaxa,wava
    Int2_AA = np.einsum('zwyv,zxwv->xy', Int2_A, Daaaa)
    # XzbJb(Jbwb|ybvb) Dzbxb,wbvb
    Int2_BB = np.einsum('zwyv,zxwv->xy', Int2_B, Dbbbb)
    # XzaJa(Jawa|ybvb) Dzaxb,wavb
    Int2_BA = -np.einsum('zwyv,xzwv->xy', Int2_A, Dbaab)
    # XzbJb(Jbwb|yava) Dzbxa,wbva
    Int2_AB = -np.einsum('zwyv,zxvw->xy', Int2_B, Dbaab)
    AAAC_A -= Int2_AA - Int2_AA.T + Int2_AB - Int2_AB.T
    AAAC_B -= Int2_BB - Int2_BB.T + Int2_BA - Int2_BA.T

    # XzJ((Jx|vw) - (Jw|vx)) Dzv,wy
    # Int3[z,x,v,w] = XzJ (Jx|vw)
    Int3A = np.einsum('zJ,Jxvw->zxvw',XA, h_Ixyz) 
    Int3B = np.einsum('zJ,Jxvw->zxvw',XB, h_Ixyz)
    # <- Int3[z,x,v,w] Dzv,wy
    Int3_AA = np.einsum('zxvw, vzyw-> xy', Int3A, Daaaa+Dbaab) - np.einsum('zwvx,vzyw->xy', Int3A, Daaaa)
    Int3_BB = np.einsum('zxvw, zvwy-> xy', Int3B, Dbbbb+Dbaab) - np.einsum('zwvx,vzyw->xy', Int3B, Dbbbb)
    Int3_BA = np.einsum('zwvx, vzwy-> xy', Int3A, Dbaab)
    Int3_AB = np.einsum('zwvx, zvyw-> xy', Int3B, Dbaab)
    AAAC_A -= Int3_AA - Int3_AA.T + Int3_AB - Int3_AB.T
    AAAC_B -= Int3_BB - Int3_BB.T + Int3_BA - Int3_BA.T

    return vectorize_skew(AAAC_A), vectorize_skew(AAAC_B)

def Hess_AAVA_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #######################
    ## Vir-Act, Act-Act  ##
    #######################    

    natt = nact*(nact-1)//2
    h_Axyz = hpqrs[ncore+nact:,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
   
    # hwA XAy Dyz - h.c.
    hXD_A = htilde[ncore:ncore+nact, ncore+nact:] @ XA @ Daa
    hXD_B = htilde[ncore:ncore+nact, ncore+nact:] @ XB @ Dbb
    AAVA_A = hXD_A 
    AAVA_B = hXD_B 

    # - ((At|uz) XAy Dtu,yw - h.c.)
    # = Int1[y,t,u,z] Dtu,yw
    # <- Int1[ya, ta, ua, za] Dtaua,yawa + Int1[yb, tb, ua, za] Dtbua,ybwa
    Int1_A = np.einsum('Atuz,Ay->ytuz', h_Axyz, XA) 
    Int1_B = np.einsum('Atuz,Ay->ytuz', h_Axyz, XB) 
    AAVA_A -= np.einsum('ytuz,tuyw->zw', Int1_A, Daaaa) - np.einsum('ytuz,tuwy->zw', Int1_B, Dbaab)
    AAVA_B -= np.einsum('ytuz,tuyw->zw', Int1_B, Dbbbb) - np.einsum('ytuz,utyw->zw', Int1_A, Dbaab)

    # ((Az|tu) - (At|uz)) XAy Dwt,uy - h.c.
    # = (Az|tu) XAy Dwt,uy - (At|uz) XAy Dwt,uy - h.c. 
    # = Int2[y,z,t,u] Dwt,uy - Int1[y,t,u,z] Dwt,uy - h.c.
    # <- Int2[ya,za,ta,ua] Dwata,uaya + Int2[ya,za,tb,ub] Dwatb,ubya  -  Int1[ya,ta,ua,za] Dwata,uaya - Int1[yb,tb,ua,za] Dwatb,uayb
    Int2_A = np.einsum('Aztu,Ay->yztu', h_Axyz, XA)
    Int2_B = np.einsum('Aztu,Ay->yztu', h_Axyz, XB)
    AAVA_A += np.einsum('yztu, twyu->zw', Int2_A, Daaaa+Dbaab) \
            - np.einsum('ytuz, wtuy->zw', Int1_A, Daaaa)\
            + np.einsum('ytuz, twuy->zw', Int1_B, Dbaab)
    AAVA_B += np.einsum('yztu, wtuy->zw', Int2_B, Dbbbb+Dbaab) \
            - np.einsum('ytuz, wtuy->zw', Int1_B, Dbbbb)\
            + np.einsum('ytuz, wtyu->zw', Int1_A, Dbaab)

    #-0.5 delta_zy (Dh_zA + (Au|tv) Dtu,zv) XAy
    #+0.5 delta_zw (Dh_wA + (Au|tv) Dtu,wv) XAy 

    # (Au|tv) Dtu,wv
    # (Aaua|tava) Dtaua,wava + (Aaua|tbvb) Dtbua,wavb
    Int3_A = np.einsum('Autv, tuwv -> wA', h_Axyz, Daaaa+Dbaab)
    Int3_B = np.einsum('Autv, utvw -> wA', h_Axyz, Dbbbb+Dbaab)    

    AAVA_A -= 0.5 * (Dh_A[ncore:ncore+nact, ncore+nact:] + Int3_A) @ XA
    AAVA_B -= 0.5 * (Dh_B[ncore:ncore+nact, ncore+nact:] + Int3_B) @ XB

    AAVA_A = vectorize_skew(AAVA_A - AAVA_A.T)
    AAVA_B = vectorize_skew(AAVA_B - AAVA_B.T)
    return AAVA_A, AAVA_B

def Hess_VCAC_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #########################
    ## Vir-Core, Act-Core  ##
    #########################               
    VCAC_A = np.zeros((nsec,ncore), dtype=float)
    VCAC_B = np.zeros((nsec,ncore), dtype=float)
    h_AIxJ = hpqrs[ ncore+nact:, :ncore,  ncore:ncore+nact, :ncore]
    h_AxIJ = hpqrs[ ncore+nact:, ncore:ncore+nact, :ncore,:ncore]
    h_Axyz = hpqrs[ncore+nact:,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]    

    # ( 2(AI|zJ) - (Az|IJ) - (AJ|zI) XzJ
    Int0 = 2* np.einsum('AIzJ,zJ->AI',h_AIxJ,XA+XB)
    VCAC_A = Int0 - np.einsum('AzIJ, zJ->AI', h_AxIJ, XA) - np.einsum('AJzI, zJ->AI', h_AIxJ, XA)
    VCAC_B = Int0 - np.einsum('AzIJ, zJ->AI', h_AxIJ, XB) - np.einsum('AJzI, zJ->AI', h_AIxJ, XB)
     
    # ( 2(AI|vJ) - (AJ|vI) - (Av|IJ) ) Dvz XzJ
    # = Int1[A,I,z,J] XzJ
    Int1_AA =2* np.einsum('AIvJ, vz -> AIzJ', h_AIxJ, Daa) - np.einsum('AJvI, vz -> AIzJ',h_AIxJ, Daa) - np.einsum('AvIJ, vz -> AIzJ', h_AxIJ, Daa)
    Int1_BB =2* np.einsum('AIvJ, vz -> AIzJ', h_AIxJ, Dbb) - np.einsum('AJvI, vz -> AIzJ',h_AIxJ, Dbb) - np.einsum('AvIJ, vz -> AIzJ', h_AxIJ, Dbb)
    Int1_BA =2* np.einsum('AIvJ, vz -> AIzJ', h_AIxJ, Daa)
    Int1_AB =2* np.einsum('AIvJ, vz -> AIzJ', h_AIxJ, Dbb)
    VCAC_A -= np.einsum('AIzJ,zJ->AI',Int1_AA,XA) \
            + np.einsum('AIzJ,zJ->AI',Int1_AB,XB)
    VCAC_B -= np.einsum('AIzJ,zJ->AI',Int1_BA,XA) \
            + np.einsum('AIzJ,zJ->AI',Int1_BB,XB)

    # (Au|tv) Dtuzv
    # (Aaua|tava) Dtaua,zava + (Aaua|tbvb) Dtbua,zavb
    Int2_AA = np.einsum('Autv, tuzv -> zA', h_Axyz, Daaaa+Dbaab)
    Int2_BB = np.einsum('Autv, utvz -> zA', h_Axyz, Dbbbb+Dbaab)    
    VCAC_A -= 0.5* np.einsum('zA,zI->AI', Int2_AA, XA)
    VCAC_B -= 0.5* np.einsum('zA,zI->AI', Int2_BB, XB)
     

    # delta_IJ 0.5 (FAz XzI + FzA XzI - Dh_zA XzI)
    VCAC_A += 0.5* (np.einsum('Az,zI->AI',Ftilde_A[ncore+nact:,ncore:ncore+nact], XA) \
                   +np.einsum('zA,zI->AI',Ftilde_A[ncore:ncore+nact,ncore+nact:], XA) \
                   -np.einsum('zA,zI->AI',Dh_A[ncore:ncore+nact,ncore+nact:],XA))
    VCAC_B += 0.5* (np.einsum('Az,zI->AI',Ftilde_B[ncore+nact:,ncore:ncore+nact], XB) \
                   +np.einsum('zA,zI->AI',Ftilde_B[ncore:ncore+nact,ncore+nact:], XB) \
                   -np.einsum('zA,zI->AI',Dh_B[ncore:ncore+nact,ncore+nact:],XB))

    VCAC_A = VCAC_A.reshape(nsec*ncore)
    VCAC_B = VCAC_B.reshape(nsec*ncore)
    
    return VCAC_A, VCAC_B

def Hess_VCAA_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #########################
    ## Vir-Core, Act-Act   ##
    #########################
    natt = nact*(nact-1)//2
    VCAA_A = np.zeros((nsec,ncore), dtype=float)
    VCAA_B = np.zeros((nsec,ncore), dtype=float)
    h_AIxy = hpqrs[ncore+nact:,:ncore,ncore:ncore+nact, ncore:ncore+nact]
    h_AxyI = hpqrs[ncore+nact:, ncore:ncore+nact,ncore:ncore+nact,  :ncore]
 
    # ( 2(AI|zv) - (Az|vI) - (Av|zI) ) Dwv Xzw
    #=( 2(AI|zv) - (Az|vI) - (Av|zI) ) XDzv
    XD_A = XA @ Daa
    XD_B = XB @ Dbb
    Int1_AA =2* np.einsum('AIzv, zv -> AI', h_AIxy, XD_A) - np.einsum('AzvI, zv -> AI',h_AxyI, XD_A) - np.einsum('AvzI, zv -> AI', h_AxyI, XD_A)
    Int1_BB =2* np.einsum('AIzv, zv -> AI', h_AIxy, XD_B) - np.einsum('AzvI, zv -> AI',h_AxyI, XD_B) - np.einsum('AvzI, zv -> AI', h_AxyI, XD_B)
    Int1_BA =2* np.einsum('AIzv, zv -> AI', h_AIxy, XD_A) 
    Int1_AB =2* np.einsum('AIzv, zv -> AI', h_AIxy, XD_B) 
    VCAA_A = Int1_AA + Int1_AB
    VCAA_B = Int1_BB + Int1_BA
    VCAA_A = VCAA_A.reshape(nsec*ncore)
    VCAA_B = VCAA_B.reshape(nsec*ncore)
    return VCAA_A, VCAA_B 

def Hess_VCVC_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    ########################
    ## Vir-core, Vir-core ##
    ########################
    h_AIBJ = hpqrs[ncore+nact:, :ncore, ncore+nact:, :ncore]
    h_ABIJ = hpqrs[ncore+nact:, ncore+nact:, :ncore, :ncore]
    VCVC_AA = np.zeros((nsec*ncore, nsec*ncore))
    VCVC_BA = np.zeros((nsec*ncore, nsec*ncore))
    VCVC_BB = np.zeros((nsec*ncore, nsec*ncore))

    # (2 (AI|BJ) - (AJ|BI) - (AB|IJ)) X_BJ
    AIBJ_X = 2* np.einsum('AIBJ,BJ->AI',h_AIBJ, XA+XB)
    VCVC_A = AIBJ_X - np.einsum('AJBI,BJ->AI',h_AIBJ, XA) - np.einsum('ABIJ,BJ->AI',h_ABIJ, XA) 
    VCVC_B = AIBJ_X - np.einsum('AJBI,BJ->AI',h_AIBJ, XB) - np.einsum('ABIJ,BJ->AI',h_ABIJ, XB) 

    # delta_IJ FBA XBJ  
    VCVC_A += np.einsum('BA,BI->AI', Ftilde_A[ncore+nact:, ncore+nact:], XA)
    VCVC_B += np.einsum('BA,BI->AI', Ftilde_B[ncore+nact:, ncore+nact:], XB)
    # delta_AB -FIJ XAJ
    VCVC_A -= np.einsum('AJ,IJ->AI', XA, Ftilde_A[:ncore, :ncore])
    VCVC_B -= np.einsum('AJ,IJ->AI', XB, Ftilde_B[:ncore, :ncore])
    
    VCVC_A = VCVC_A.reshape((nsec*ncore))
    VCVC_B = VCVC_B.reshape((nsec*ncore))
    return VCVC_A, VCVC_B

def Hess_VAAC_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    ########################
    ## Vir-Act, Act-Core  ##
    ########################               
    VAAC_A = np.zeros((nact*nsec), dtype=float)
    VAAC_B = np.zeros((nact*nsec), dtype=float)
    h_AxIy = hpqrs[ncore+nact:,ncore:ncore+nact,:ncore, ncore:ncore+nact]
    h_AIxy = hpqrs[ncore+nact:,:ncore,ncore:ncore+nact,ncore:ncore+nact]

    # -htilde_JA Dyz XzJ = DXh_yA
    DXh_A = Daa @ XA @ htilde[:ncore, ncore+nact:] 
    DXh_B = Dbb @ XB @ htilde[:ncore, ncore+nact:] 
    VAAC_A = - DXh_A.T  
    VAAC_B = - DXh_B.T  

    # ( 2(Aw|Jz) - (Az|Jw) - (AJ|wz) ) Dyw XzJ
    # = Int1[Aw] Dyw
    # Int1 = ( 2(Aw|Jz) - (Az|Jw) - (AJ|wz) ) XzJ
    Int1 = 2 * np.einsum('AwJz, zJ -> Aw', h_AxIy, XA+XB)
    Int1_A = Int1 -  np.einsum('AzJw, zJ -> Aw', h_AxIy, XA) - np.einsum('AJwz, zJ -> Aw', h_AIxy, XA) 
    Int1_B = Int1 -  np.einsum('AzJw, zJ -> Aw', h_AxIy, XB) - np.einsum('AJwz, zJ -> Aw', h_AIxy, XB) 
    # Int1[Aw]Dyw
    VAAC_A += np.einsum('Aw,yw->Ay', Int1_A, Daa)
    VAAC_B += np.einsum('Aw,yw->Ay', Int1_B, Dbb)
    
    # (Av|Jw) XzJ Dvw,yz 
    # = Int2[A,v,z,w] Dvw,yz
    # Int2 = (Av|Jw) XzJ
    Int2_A = np.einsum('AvJw,zJ->Avzw',h_AxIy, XA)
    Int2_B = np.einsum('AvJw,zJ->Avzw',h_AxIy, XB)
    # Int2[A,v,z,w] Dvw,yz
    # Int2[Aa,va,za,wa] Dvawa,yaza + Int2[Aa,va, zb, wb] Dvawb,yazb
    VAAC_A += np.einsum('Avzw, vwyz->Ay', Int2_A, Daaaa) - np.einsum('Avzw,wvyz->Ay',Int2_B,Dbaab)
    VAAC_B += np.einsum('Avzw, vwyz->Ay', Int2_B, Dbbbb) - np.einsum('Avzw,vwzy->Ay',Int2_A,Dbaab)

    # ((AJ|vw) - (Av|Jw)) Dzv,wy XzJ
    # = Int3[A,z,v,w] Dzv,wy
    # Int3[A,z,v,w] = ((AJ|vw) - (Av|Jw)) XzJ
    # A <- (AaJa|vawa) XzaJa Dzava,waya + (AaJa|vbwb) XzaJa Dzavb,wbya - (Aava|Jawa) XzaJa Dzava,waya - (Aava|Jbwb) XzbJb Dzbva,wbya
    #     = Int3_A[A,z,v,w] (Dzava,waya + Dzavb,wbya) - Int4_A[A,z,v,w] Dzava,waya - Int4_B[A,z,v,w] Dzbva,wbya
    Int3_A =   np.einsum('AJvw,zJ->Azvw', h_AIxy, XA) 
    Int4_A = - np.einsum('AvJw,zJ->Azvw', h_AxIy, XA) 
    Int3_B =   np.einsum('AJvw,zJ->Azvw', h_AIxy, XB) 
    Int4_B = - np.einsum('AvJw,zJ->Azvw', h_AxIy, XB)
    ## Int3[Aa,za,va,wa] Dzava,waya 
    VAAC_A += - np.einsum('Azvw, vzyw->Ay', Int3_A, Daaaa+Dbaab) 
    VAAC_A += - np.einsum('Azvw, zvwy->Ay', Int4_A, Daaaa)
    VAAC_A +=   np.einsum('Azvw, zvyw->Ay', Int4_B, Dbaab)
    VAAC_B += - np.einsum('Azvw, zvwy->Ay', Int3_B, Daaaa+Dbaab) 
    VAAC_B += - np.einsum('Azvw, zvwy->Ay', Int4_B, Dbbbb)
    VAAC_B +=   np.einsum('Azvw, vzwy->Ay', Int4_A, Dbaab)


    # delta_zy 0.5 F_JA XyJ
    VAAC_A += 0.5 * np.einsum('JA,yJ->Ay', Ftilde_A[:ncore, ncore+nact:], XA)
    VAAC_B += 0.5 * np.einsum('JA,yJ->Ay', Ftilde_B[:ncore, ncore+nact:], XB)

    VAAC_A = VAAC_A.reshape(nsec*nact)
    VAAC_B = VAAC_B.reshape(nsec*nact)
    
    return VAAC_A, VAAC_B

def Hess_VAAA_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #######################
    ## Vir-Act, Act-Act  ##
    #######################    

    natt = nact*(nact-1)//2
    h_Axyz = hpqrs[ncore+nact:,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    
    VAAA_A = np.zeros((nact*nsec), dtype=float)
    VAAA_B = np.zeros((nact*nsec), dtype=float)

    # -h_wA Dyz Xzw = -DXh_yA
    DXh_A = Daa @ XA @ htilde[ncore:ncore+nact, ncore+nact:]
    DXh_B = Dbb @ XB @ htilde[ncore:ncore+nact, ncore+nact:]
    VAAA_A = - DXh_A.T
    VAAA_B = - DXh_B.T

    # (At|uz) Dtu,yw Xzw
    # = Int1[A,t,u,w] Dtu,yw
    # Int1[A,t,u,w] = (At|uz) Xzw
    Int1_A = np.einsum('Atuz, zw-> Atuw', h_Axyz, XA)
    Int1_B = np.einsum('Atuz, zw-> Atuw', h_Axyz, XB)
    # Int1[Aa,ta,ua,wa] Dtaua,yawa + Int1[Aa,ta,ub,wb] Dtaub,yawb 
    VAAA_A -= np.einsum('Atuw, tuyw-> Ay', Int1_A, Daaaa) - np.einsum('Atuw,utyw->Ay',Int1_B, Dbaab) 
    VAAA_B -= np.einsum('Atuw, tuyw-> Ay', Int1_B, Dbbbb) - np.einsum('Atuw,tuwy->Ay',Int1_A, Dbaab) 


    # ((Az|tu) - (At|uz)) Xzw Dwt,uy
    # = Int2[A,w,t,u] Dwt,uy - Int1[A,t,u,w] Dwt,uy
    # Int2[A,w,t,u] = (Az|tu) Xzw

    Int2_A = np.einsum('Aztu,zw->Awtu', h_Axyz, XA) 
    Int2_B = np.einsum('Aztu,zw->Awtu', h_Axyz, XB) 

    # <- Int2[Aa,wa,ta,ua] Dwata,uaya  + Int2[Aa, wa, tb, ub] Dwatb,ubya - Int1[Aa,ta,ua,wa]  Dwata,uaya - Int1[Aa,ta,ub,wb] Dwbta,ubya
    VAAA_A += np.einsum('Awtu,twyu->Ay',Int2_A, Daaaa+Dbaab)\
            - np.einsum('Atuw,wtuy->Ay',Int1_A, Daaaa)\
            + np.einsum('Atuw,wtyu->Ay',Int1_B, Dbaab)
    VAAA_B += np.einsum('Awtu,wtuy->Ay',Int2_B, Daaaa+Dbaab)\
            - np.einsum('Atuw,wtuy->Ay',Int1_B, Daaaa)\
            + np.einsum('Atuw,twuy->Ay',Int1_A, Dbaab)

    # delta_zy 0.5 (Xzw Dh_wA + (Au|tv) Dtu,wv Xzw)
    # delta_wy 0.5 (
    XDh_A = XA @ Dh_A[ncore:ncore+nact,ncore+nact:]
    XDh_B = XB @ Dh_B[ncore:ncore+nact,ncore+nact:]
    # (Au|tv) Dtu,wv
    # (Aaua|tava) Dtaua,wava + (Aaua|tbvb) Dtbua,wavb
    Int3_AA = np.einsum('Autv, tuwv -> Aw', h_Axyz, Daaaa+Dbaab)
    Int3_BB = np.einsum('Autv, utvw -> Aw', h_Axyz, Dbbbb+Dbaab)    
    VAAA_A += 0.5 * (XDh_A.T + np.einsum('Aw, yw->Ay', Int3_AA, XA))   
    VAAA_B += 0.5 * (XDh_B.T + np.einsum('Aw, yw->Ay', Int3_BB, XB)) 

    VAAA_A = VAAA_A.reshape(nact*nsec)
    VAAA_B = VAAA_B.reshape(nact*nsec)
    #Ay = 0
    #for A in range(nsec):
    #    for y in range(nact):
    #        zw = 0
    #        for z in range(nact):
    #            for w in range(z):
    #                if z==y:
    #                    VAAA_A[Ay] += 0.5 * (Dh_A[w+ncore,A+ncore+nact] + Int3_AA[A,w]) * XA[z,w]
    #                    VAAA_B[Ay] += 0.5 * (Dh_B[w+ncore,A+ncore+nact] + Int3_BB[A,w]) * XB[z,w]
    #                if w==y:
    #                    VAAA_A[Ay] -= 0.5 * (Dh_A[z+ncore,A+ncore+nact] + Int3_AA[A,z]) * XA[z,w]
    #                    VAAA_B[Ay] -= 0.5 * (Dh_B[z+ncore,A+ncore+nact] + Int3_BB[A,z]) * XB[z,w]
    #                   
    #                zw += 1
    #        Ay += 1
    return VAAA_A, VAAA_B

def Hess_VAVC_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #########################
    ## Vir-Act, Vir-Core   ##
    #########################
    VAVC_A = np.zeros((nact*nsec), dtype=float)
    VAVC_B = np.zeros((nact*nsec), dtype=float)
    h_AxBJ = hpqrs[ncore+nact:, ncore:ncore+nact, ncore+nact:, :ncore]
    h_ABxJ = hpqrs[ncore+nact:, ncore+nact:, ncore:ncore+nact, :ncore]
    h_xyzJ = hpqrs[ncore:ncore+nact, ncore:ncore+nact, ncore:ncore+nact, :ncore]    

    # ( 2(Av|BJ) - (AJ|Bv) - (AB|vJ) ) XBJ Dyv
    Int1_A = 2*np.einsum('AvBJ,BJ->Av',h_AxBJ, XA)
    Int1_B = 2*np.einsum('AvBJ,BJ->Av',h_AxBJ, XB)
    Int1_AA = Int1_A - np.einsum('BvAJ,BJ->Av',h_AxBJ,XA) - np.einsum('ABvJ,BJ->Av',h_ABxJ,XA)
    Int1_BB = Int1_B - np.einsum('BvAJ,BJ->Av',h_AxBJ,XB) - np.einsum('ABvJ,BJ->Av',h_ABxJ,XB)
    VAVC_A = np.einsum('Av,yv->Ay',Int1_AA + Int1_B, Daa)  
    VAVC_B = np.einsum('Av,yv->Ay',Int1_BB + Int1_A, Dbb)  

    # -0.5 delta_AB (FyJ + Dh_yJ + (vu|tJ) Dyv,ut) XBJ
    # (vu|tJ) XzJ Dyv,ut
    # (vaua|taJa) Dyava,uata + (vbub|taJa) Dyavb,ubta 
    Int2_AA = np.einsum('vutJ, vytu -> yJ', h_xyzJ, Daaaa + Dbaab)
    Int2_BB = np.einsum('vutJ, yvut -> yJ', h_xyzJ, Dbbbb + Dbaab)
    VAVC_A -= 0.5 * XA @ (Ftilde_A[ncore:ncore+nact, :ncore] + Dh_A[ncore:ncore+nact, :ncore] + Int2_AA).T 
    VAVC_B -= 0.5 * XB @ (Ftilde_B[ncore:ncore+nact, :ncore] + Dh_B[ncore:ncore+nact, :ncore] + Int2_BB).T 
    
    VAVC_A = VAVC_A.reshape(nact*nsec)
    VAVC_B = VAVC_B.reshape(nact*nsec)
    return VAVC_A, VAVC_B

def Hess_VCVA_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #########################
    ## Vir-Act, Vir-Core   ##
    #########################
    VAVC_AA = np.zeros((nact*nsec), dtype=float)
    VAVC_BA = np.zeros((nact*nsec), dtype=float)
    VAVC_AB = np.zeros((nact*nsec), dtype=float)    
    VAVC_BB = np.zeros((nact*nsec), dtype=float)
    h_AxBJ = hpqrs[ncore+nact:, ncore:ncore+nact, ncore+nact:, :ncore]
    h_ABxJ = hpqrs[ncore+nact:, ncore+nact:, ncore:ncore+nact, :ncore]
    h_xyzJ = hpqrs[ncore:ncore+nact, ncore:ncore+nact, ncore:ncore+nact, :ncore]    

    # ( 2(Av|BJ) - (Bv|AJ) - (AB|vJ) ) Dyv
    Int1_AA = 2* np.einsum('AvBJ, yv -> AyBJ', h_AxBJ, Daa) - np.einsum('BvAJ, yv -> AyBJ', h_AxBJ, Daa) - np.einsum('ABvJ, yv -> AyBJ',h_ABxJ, Daa)
    Int1_BB = 2* np.einsum('AvBJ, yv -> AyBJ', h_AxBJ, Dbb) - np.einsum('BvAJ, yv -> AyBJ', h_AxBJ, Dbb) - np.einsum('ABvJ, yv -> AyBJ',h_ABxJ, Dbb)
    Int1_BA = 2* np.einsum('AvBJ, yv -> AyBJ', h_AxBJ, Daa)
    Int1_AB = 2* np.einsum('AvBJ, yv -> AyBJ', h_AxBJ, Dbb)
    # (vu|tJ) Dyv,ut
    # (vaua|taJa) Dyava,uata + (vbub|taJa) Dyavb,ubta 
    Int2_AA = np.einsum('vutJ, vytu -> yJ', h_xyzJ, Daaaa + Dbaab)
    Int2_BB = np.einsum('vutJ, yvut -> yJ', h_xyzJ, Dbbbb + Dbaab)
    
    Ay = 0
    for A in range(nsec):
        for y in range(nact):
            BJ = 0
            for B in range(nsec):
                for J in range(ncore):
                    VAVC_AA[Ay,BJ] = Int1_AA[A,y,B,J]
                    VAVC_BB[Ay,BJ] = Int1_BB[A,y,B,J]
                    VAVC_BA[Ay,BJ] = Int1_BA[A,y,B,J]
                    VAVC_AB[Ay,BJ] = Int1_AB[A,y,B,J]
                    if A==B:
                        VAVC_AA[Ay,BJ] -= 0.5 * (Ftilde_A[y+ncore,J] + Dh_A[y+ncore,J] + Int2_AA[y,J])
                        VAVC_BB[Ay,BJ] -= 0.5 * (Ftilde_B[y+ncore,J] + Dh_B[y+ncore,J] + Int2_BB[y,J])
                        
                    BJ += 1
            Ay += 1
    return VAVC_AA, VAVC_BA, VAVC_AB, VAVC_BB

def Hess_VAVA_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #######################
    ## Vir-Act, Vir-Act  ##
    #######################   
    VAVA_A = np.zeros((nact*nsec), dtype=float)
    VAVA_B = np.zeros((nact*nsec), dtype=float)
    h_ABxy = hpqrs[ ncore+nact:, ncore+nact:, ncore:ncore+nact, ncore:ncore+nact]
    h_AxBy = hpqrs[ ncore+nact:, ncore:ncore+nact, ncore+nact:, ncore:ncore+nact]
    h_wxyz = hpqrs[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    # hBA Dyw XBw = hXD
    hXD_A = htilde[ncore+nact:, ncore+nact:] @ XA @ Daa
    hXD_B = htilde[ncore+nact:, ncore+nact:] @ XB @ Dbb
    VAVA_A = hXD_A
    VAVA_B = hXD_B
    # (At|Bu) XBw Dtu,yw
    # = Int1[A,t,w,u] Dtu,yw
    # <- Int1[Aa,ta,wa,ua] Dtaua,yawa  + Int1[Aa,ta,wb,ub] Dtaub,yawb
    Int1_A = np.einsum('AtBu,Bw->Atwu',h_AxBy, XA)
    Int1_B = np.einsum('AtBu,Bw->Atwu',h_AxBy, XB)
    VAVA_A += np.einsum('Atwu,utyw->Ay',Int1_A, Daaaa) + np.einsum('Atwu,utyw->Ay',Int1_B, Dbaab)
    VAVA_B += np.einsum('Atwu,utyw->Ay',Int1_B, Dbbbb) + np.einsum('Atwu,tuwy->Ay',Int1_A, Dbaab)

    # ((AB|tu) - (At|Bu)) XBw Dwt,uy
    # = (AB|tu) XBw Dwt,uy  - (At|Bu) XBw Dwt,uy
    # = Int2[A,w,t,u] Dwt,uy - Int1[A,t,w,u] Dwt,uy
    # <- Int2[Aa,wa,ta,ua] Dwata,uaya + Int2[Aa,wa,tb,ub] Dwatb,ub,ya - Int1[Aa,ta,wa,ua] Dwata,uaya - Int1[Aa,ta,wb,ub] Dwbta,ubya
    Int2_A = np.einsum('ABtu,Bw->Awtu', h_ABxy, XA)
    Int2_B = np.einsum('ABtu,Bw->Awtu', h_ABxy, XB)
    VAVA_A += np.einsum('Awtu,twyu->Ay', Int2_A, Daaaa+Dbaab)\
            - np.einsum('Atwu,wtuy->Ay', Int1_A, Daaaa)\
            + np.einsum('Atwu,wtyu->Ay', Int1_B, Dbaab)
    VAVA_B += np.einsum('Awtu,wtuy->Ay', Int2_B, Dbbbb+Dbaab)\
            - np.einsum('Atwu,wtuy->Ay', Int1_B, Dbbbb)\
            + np.einsum('Atwu,twuy->Ay', Int1_A, Dbaab)
    
    # -0.5 delta_AB ( Dh_wy + Dh_yw + (yt|vu) Dwv,ut + (wt|vu) Dyv,ut ) XBw
    # (yata|vaua) Dwava, uata + (yata|vbub) Dwavb,ubta
    Int3_AA = np.einsum('ytvu, vwtu->yw',h_wxyz,Daaaa+Dbaab)
    Int3_BB = np.einsum('ytvu, wvut->yw',h_wxyz,Daaaa+Dbaab)    
    VAVA_A -= 0.5 * XA @ ( Dh_A[ncore:ncore+nact, ncore:ncore+nact] + Dh_A[ncore:ncore+nact, ncore:ncore+nact].T + Int3_AA + Int3_AA.T)
    VAVA_B -= 0.5 * XB @ ( Dh_B[ncore:ncore+nact, ncore:ncore+nact] + Dh_B[ncore:ncore+nact, ncore:ncore+nact].T + Int3_BB + Int3_BB.T)
    VAVA_A = VAVA_A.reshape(nact*nsec)
    VAVA_B = VAVA_B.reshape(nact*nsec)
    return VAVA_A, VAVA_B 

def Hess_ACVC_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #########################
    ## Act-Core,  Vir-Core ##
    #########################               
    ACVC_A = np.zeros((nact*ncore), dtype=float)
    ACVC_B = np.zeros((nact*ncore), dtype=float)
    h_AIxJ = hpqrs[ ncore+nact:, :ncore,  ncore:ncore+nact, :ncore]
    h_AxIJ = hpqrs[ ncore+nact:, ncore:ncore+nact, :ncore,:ncore]
    h_Axyz = hpqrs[ncore+nact:,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]    

    # ( 2(AI|zJ) - (Az|IJ) - (AJ|zI) ) XAI
    Int1_sum = 2* np.einsum('AIzJ, AI -> zJ', h_AIxJ, XA+XB)
    Int1_A = Int1_sum - np.einsum('AJzI,AI->zJ',h_AIxJ, XA) - np.einsum('AzIJ,AI->zJ',h_AxIJ, XA)
    Int1_B = Int1_sum - np.einsum('AJzI,AI->zJ',h_AIxJ, XB) - np.einsum('AzIJ,AI->zJ',h_AxIJ, XB)
    ACVC_A = Int1_A
    ACVC_B = Int1_B

    # ( 2(AI|vJ) - (AJ|vI) - (Av|IJ) ) XAI Dvz
    # = Int1[v,J] Dvz
    ACVC_A -= np.einsum('vz,vJ->zJ',Daa,Int1_A)
    ACVC_B -= np.einsum('vz,vJ->zJ',Dbb,Int1_B)


    # 0.5 delta_IJ (FAz + FzA - Dh_zA - (Au|tv) Dtu,zv) XAI
    # (Au|tv) Dtuzv
    # (Aaua|tava) Dtaua,zava + (Aaua|tbvb) Dtbua,zavb
    Int2_AA = np.einsum('Autv, tuzv -> zA', h_Axyz, Daaaa+Dbaab)
    Int2_BB = np.einsum('Autv, utvz -> zA', h_Axyz, Dbbbb+Dbaab)    
    ACVC_A += 0.5 * (Ftilde_A[ncore+nact:,ncore:ncore+nact].T + Ftilde_A[ncore:ncore+nact,ncore+nact:] - Dh_A[ncore:ncore+nact,ncore+nact:] - Int2_AA) @ XA
    ACVC_B += 0.5 * (Ftilde_B[ncore+nact:,ncore:ncore+nact].T + Ftilde_B[ncore:ncore+nact,ncore+nact:] - Dh_B[ncore:ncore+nact,ncore+nact:] - Int2_BB) @ XB
    
    ACVC_A = ACVC_A.reshape(nact*ncore)
    ACVC_B = ACVC_B.reshape(nact*ncore)
    return ACVC_A, ACVC_B

def Hess_AAVC_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #########################
    ## Act-Act,  Vir-Core  ##
    #########################
    natt = nact*(nact-1)//2
    AAVC_AA = np.zeros((natt), dtype=float)
    AAVC_BA = np.zeros((natt), dtype=float)
    AAVC_AB = np.zeros((natt), dtype=float)    
    AAVC_BB = np.zeros((natt), dtype=float)
    h_AIxy = hpqrs[ncore+nact:,:ncore,ncore:ncore+nact, ncore:ncore+nact]
    h_AxyI = hpqrs[ncore+nact:, ncore:ncore+nact,ncore:ncore+nact,  :ncore]
 
    # ( 2(AI|zv) - (Az|vI) - (Av|zI) ) XAI Dwv
    # Int1[zv] Dwv
    Int1_sum = 2* np.einsum('AIzv, AI -> zv', h_AIxy, XA + XB) 
    Int1_A = Int1_sum - np.einsum('AzvI, AI -> zv',h_AxyI, XA) - np.einsum('AvzI, AI -> zv', h_AxyI, XA)
    Int1_B = Int1_sum - np.einsum('AzvI, AI -> zv',h_AxyI, XB) - np.einsum('AvzI, AI -> zv', h_AxyI, XB)
    
    AAVC_A_ = np.einsum('zv,wv->zw',Int1_A, Daa) 
    AAVC_B_ = np.einsum('zv,wv->zw',Int1_B, Dbb) 
    AAVC_A = AAVC_A_ - AAVC_A_.T
    AAVC_B = AAVC_B_ - AAVC_B_.T
    return vectorize_skew(AAVC_A), vectorize_skew(AAVC_B)


def Hess_ACVA_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    ########################
    ## Act-Core, Vir-Act  ##
    ########################               
    h_AxIy = hpqrs[ncore+nact:,ncore:ncore+nact,:ncore, ncore:ncore+nact]
    h_AIxy = hpqrs[ncore+nact:,:ncore,ncore:ncore+nact,ncore:ncore+nact]
   
    # -hJA Dyz XAy = - hXD[J,z]
    hXD_A = htilde[:ncore, ncore+nact:] @ XA @ Daa
    hXD_B = htilde[:ncore, ncore+nact:] @ XB @ Dbb
    ACVA_A = - hXD_A.T
    ACVA_B = - hXD_B.T
     
    # ( 2(Aw|Jz) - (Az|Jw) - (AJ|wz) ) XAy Dyw
    # 
    XD_A = XA @ Daa
    XD_B = XB @ Dbb
    Int1_sum = 2 * np.einsum('AwJz,Aw->zJ', h_AxIy, XD_A + XD_B) 
    ACVA_A += Int1_sum - np.einsum('AzJw,Aw->zJ', h_AxIy, XD_A) - np.einsum('AJwz,Aw->zJ', h_AIxy, XD_A) 
    ACVA_B += Int1_sum - np.einsum('AzJw,Aw->zJ', h_AxIy, XD_B) - np.einsum('AJwz,Aw->zJ', h_AIxy, XD_B) 

    # (Av|Jw) XAy Dvw,yz
    # = Int2[y,v,J,w] Dvw,yz
    #<- Int2[ya,va,Ja,wa] Dvawa,yaza + Int2[yb,vb,Ja,wa] Dvbwa,ybza
    Int2_A = np.einsum('AvJw,Ay->yvJw', h_AxIy, XA) 
    Int2_B = np.einsum('AvJw,Ay->yvJw', h_AxIy, XB) 
    ACVA_A -= np.einsum('yvJw,vwzy->zJ', Int2_A, Daaaa) + np.einsum('yvJw,vwzy->zJ', Int2_B, Dbaab)
    ACVA_B -= np.einsum('yvJw,vwzy->zJ', Int2_B, Dbbbb) + np.einsum('yvJw,wvyz->zJ', Int2_A, Dbaab)

    # - ((AJ|vw) - (Av|Jw)) XAy Dzv,wy
    # = - ( (AJ|vw) XAy - (Av|Jw) XAy ) Dzv,wy 
    # = - ( Int3[y,J,v,w] - Int2[y,v,J,w] ) Dzv,wy 
    Int3_A = np.einsum('AJvw,Ay->yJvw',h_AIxy, XA)
    Int3_B = np.einsum('AJvw,Ay->yJvw',h_AIxy, XB)
    # <- - ( Int3[ya,Ja,va,wa] Dzava,waya + Int3[ya,Ja,vb,wb] Dzavb,wbya - Int2[ya,va,Ja,wa] Dzava,waya - Int2[yb,vb,Ja,wa] Dzavb,wayb )
    ACVA_A -= np.einsum('yJvw,vzyw->zJ', Int3_A, Daaaa+Dbaab)\
            - np.einsum('yvJw,zvwy->zJ', Int2_A, Daaaa)\
            + np.einsum('yvJw,vzwy->zJ', Int2_B, Dbaab)
    ACVA_B -= np.einsum('yJvw,zvwy->zJ', Int3_B, Dbbbb+Dbaab)\
            - np.einsum('yvJw,zvwy->zJ', Int2_B, Dbbbb)\
            + np.einsum('yvJw,zvyw->zJ', Int2_A, Dbaab)
   
    # 0.5 delta_zy FJA XAy
    ACVA_A += 0.5 * np.einsum('JA,Az->zJ',Ftilde_A[:ncore, ncore+nact:], XA)
    ACVA_B += 0.5 * np.einsum('JA,Az->zJ',Ftilde_B[:ncore, ncore+nact:], XB)
    ACVA_A = ACVA_A.reshape(nact*ncore) 
    ACVA_B = ACVA_B.reshape(nact*ncore) 
    return ACVA_A, ACVA_B


def Hess_VCVA_X(XA, XB, ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #########################
    ## Vir-Core, Vir-Act   ##
    #########################
    h_AxBJ = hpqrs[ncore+nact:, ncore:ncore+nact, ncore+nact:, :ncore]
    h_ABxJ = hpqrs[ncore+nact:, ncore+nact:, ncore:ncore+nact, :ncore]
    h_xyzJ = hpqrs[ncore:ncore+nact, ncore:ncore+nact, ncore:ncore+nact, :ncore]    

    # ( 2(Av|BJ) - (Bv|AJ) - (AB|vJ) ) XAy Dyv
    XD_A = XA @ Daa
    XD_B = XB @ Dbb
    VCVA_A = 2* np.einsum('AvBJ, Av -> BJ', h_AxBJ, XD_A+XD_B)\
             -  np.einsum('BvAJ, Av -> BJ', h_AxBJ, XD_A)\
             -  np.einsum('ABvJ, Av -> BJ', h_ABxJ, XD_A)
    VCVA_B = 2* np.einsum('AvBJ, Av -> BJ', h_AxBJ, XD_A+XD_B)\
             -  np.einsum('BvAJ, Av -> BJ', h_AxBJ, XD_B)\
             -  np.einsum('ABvJ, Av -> BJ', h_ABxJ, XD_B)
   
    # -0.5 delta_AB XBy (FyJ + Dh_yJ (vu|tJ) Dyv,ut) 
    # (vu|tJ) Dyv,ut
    # (vaua|taJa) Dyava,uata + (vbub|taJa) Dyavb,ubta 
    Int2_A = np.einsum('vutJ, vytu -> yJ', h_xyzJ, Daaaa + Dbaab)
    Int2_B = np.einsum('vutJ, yvut -> yJ', h_xyzJ, Dbbbb + Dbaab)
    VCVA_A -= 0.5 * XA @ (Ftilde_A[ncore:ncore+nact, :ncore] + Dh_A[ncore:ncore+nact, :ncore] + Int2_A)
    VCVA_B -= 0.5 * XB @ (Ftilde_B[ncore:ncore+nact, :ncore] + Dh_B[ncore:ncore+nact, :ncore] + Int2_B)
    
    VCVA_A = VCVA_A.reshape(nsec*ncore)
    VCVA_B = VCVA_B.reshape(nsec*ncore)
    return VCVA_A, VCVA_B

