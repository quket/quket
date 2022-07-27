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
# For debug purpose. Compute whole Hessian (including redundant rotation).

"""
#######################
#        quket        #
#######################

orbital/hess_whole.py

Functions related to orbital hessian

"""
import numpy as np
import time

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.linalg import vectorize_skew, skew
from quket.fileio import error, prints, printmat, print_state
from .misc import *

#################################
# Orbital Hessian for subspaces #
#################################
def Hess_AAAA(ncore, nact, nsec, htilde, hpqrs, Daa, Dbb, Daaaa, Dbaab, Dbbbb):    
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
    - delta[w,u]  (D[v,i] h[i,t] - (ik|jt) D[ij,kv])
    - delta[t,v]  (h[u,i] D[i,w] - (uj|ik) D[wi,jk])
    '''

    '''
    (2)
    - (it|jv) D[ij,wu] - (ui|wj) D[vt,ij] + (it|jw) D[ij,vu] + (ui|vj) D[wt,ij]
    '''
    natt = (nact-1)*nact//2
    AAAA_AA = np.zeros((natt, natt))
    AAAA_BA = np.zeros((natt, natt))
    AAAA_BB = np.zeros((natt, natt))
    
    h1act = htilde[ncore:ncore+nact, ncore:ncore+nact]
    h_wxyz = hpqrs[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    # tu=A, vw=A
    # (iAtA|jAvA) D[iAjA,wAuA]
    HD2_AA = np.einsum('itjv,ijwu->tvwu',h_wxyz,Daaaa)
    # tu=B, vw=A
    # (iBtB|jAvA) D[iBjA,wAuB]
    HD2_BA = np.einsum('itjv,ijwu->tvwu',h_wxyz,Dbaab)
    # tu=B, vw=B
    # (iBtB|jBvB) D[iBjB,wBuB]
    HD2_BB = np.einsum('itjv,ijwu->tvwu',h_wxyz,Dbbbb)
    '''
    (3)
    - ((wt|ij) - (wj|it)) D[vi,uj]  - ((uv|ij) - (uj|iv)) D[ti,wj]
    + ((vt|ij) - (vj|it)) D[wi,uj]  + ((uw|ij) - (uj|iw)) D[ti,vj]
    '''
    # tu=A, vw=A
    # ((wAtA|iAjA) - (wAjA|iAtA)) D[vAiA,uAjA] + (wAtA|iBjB) D[vAiB, uAjB]
    HD3_AA = np.einsum('wtij,viuj->wtvu',h_wxyz,Daaaa) - np.einsum('wjit,viuj->wtvu',h_wxyz,Daaaa)\
              - np.einsum('wtij,ivuj-> wtvu', h_wxyz, Dbaab)
    # tu=B, vw=A
    # - (wAjA|iBtB) D[vAiB,uBjA]  = - (wAjA|iBtB) D[iBvA, jAuB]
    HD3_BA = -np.einsum('wjit,ivju->wtvu',h_wxyz, Dbaab)
    HD3_AB = -np.einsum('wjit,viuj->wtvu',h_wxyz, Dbaab)

    # tu=B, vw=B
    # ((wBtB|iBjB) - (wBjB|iBtB)) D[vBiB,uBjB] + (wBtB|iAjA) D[vBiA, uBjA]
    HD3_BB = np.einsum('wtij,viuj->wtvu',h_wxyz,Dbbbb) - np.einsum('wjit,viuj->wtvu',h_wxyz,Dbbbb)\
              - np.einsum('wtij,viju-> wtvu', h_wxyz, Dbaab)
    '''
    (4)
    + delta[v,u]  (D[w,i] h[i,t] - (ik|jt) D[ij,kw])
    + delta[t,w]  (h[u,i] D[i,v] - (uj|ik) D[vi,jk])
    - delta[v,u]  (D[v,i] h[i,t] - (ik|jt) D[ij,kv])
    - delta[t,v]  (h[u,i] D[i,w] - (uj|ik) D[wi,jk])
    '''
    HD1_A = np.einsum('ui,iv->uv',h1act, Daa)
    HD1_B = np.einsum('ui,iv->uv',h1act, Dbb)
    # tu=A, vw=A
    # (iAkA|jAtA) D[iAjA,kAwA] + (iBkB|jAtA) D[iBjA,kBwA]
    HD4_AA = np.einsum('ikjt,ijkw->tw',h_wxyz, Daaaa) - np.einsum('ikjt,ijwk->tw',h_wxyz, Dbaab)
    # tu=B, vw=B 
    # (iBkB|jBtB) D[iBjB,kBwB] + (iAkA|jBtB) D[iAjB,kAwB]
    HD4_BB = np.einsum('ikjt,ijkw->tw',h_wxyz, Dbbbb) - np.einsum('ikjt,jikw->tw',h_wxyz, Dbaab)
    # tu=B, vw=A  --> 0
    tu = 0
    for t in range(nact):
        for u in range(t):
            vw = 0
            for v in range(nact):
                for w in range(v):
                    AAAA_AA[tu, vw] = - h1act[w,t] * Daa[u,v] \
                                       - h1act[u,v] * Daa[w,t] \
                                       + h1act[v,t] * Daa[u,w] \
                                       + h1act[u,w] * Daa[v,t] \
                                       + HD2_AA[t,v,w,u] \
                                       + HD2_AA[u,w,v,t] \
                                       - HD2_AA[t,w,v,u] \
                                       - HD2_AA[u,v,w,t] \
                                       + HD3_AA[w,t,v,u] \
                                       + HD3_AA[u,v,t,w] \
                                       - HD3_AA[v,t,w,u] \
                                       - HD3_AA[u,w,t,v] 
                    AAAA_BB[tu, vw] =   - h1act[w,t] * Dbb[u,v] \
                                       - h1act[u,v] * Dbb[w,t] \
                                       + h1act[v,t] * Dbb[u,w] \
                                       + h1act[u,w] * Dbb[v,t] \
                                       + HD2_BB[t,v,w,u] \
                                       + HD2_BB[u,w,v,t] \
                                       - HD2_BB[t,w,v,u] \
                                       - HD2_BB[u,v,w,t] \
                                       + HD3_BB[w,t,v,u] \
                                       + HD3_BB[u,v,t,w] \
                                       - HD3_BB[v,t,w,u] \
                                       - HD3_BB[u,w,t,v] 
                    AAAA_BA[tu, vw] =  + HD2_BA[t,v,w,u] \
                                       + HD2_BA[u,w,v,t] \
                                       - HD2_BA[t,w,v,u] \
                                       - HD2_BA[u,v,w,t] \
                                       + HD3_BA[w,t,v,u] \
                                       + HD3_AB[u,v,t,w] \
                                       - HD3_BA[v,t,w,u] \
                                       - HD3_AB[u,w,t,v]                         
                    if v==u:
                        AAAA_AA[tu, vw] += HD1_A[t,w] - HD4_AA[t,w]
                        AAAA_BB[tu, vw] += HD1_B[t,w] - HD4_BB[t,w]
                    if t==w:
                        AAAA_AA[tu, vw] += HD1_A[u,v] - HD4_AA[u,v]
                        AAAA_BB[tu, vw] += HD1_B[u,v] - HD4_BB[u,v]
                    if w==u:
                        AAAA_AA[tu, vw] -= HD1_A[t,v] - HD4_AA[t,v]
                        AAAA_BB[tu, vw] -= HD1_B[t,v] - HD4_BB[t,v]
                    if t==v:
                        AAAA_AA[tu, vw] -= HD1_A[u,w] - HD4_AA[u,w]
                        AAAA_BB[tu, vw] -= HD1_B[u,w] - HD4_BB[u,w]
                    vw += 1
            tu += 1
    AAAA_AA = 0.5 * (AAAA_AA + AAAA_AA.T)
    AAAA_BB = 0.5 * (AAAA_BB + AAAA_BB.T)  
    return AAAA_AA, AAAA_BA, AAAA_BB

def Hess_ACAC(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):    
    ########################
    ## Act-core, Act-core ##
    ########################
    h_IJxy = hpqrs[:ncore,:ncore,ncore:ncore+nact,ncore:ncore+nact]    
    h_IxJy = hpqrs[:ncore,ncore:ncore+nact,:ncore,ncore:ncore+nact]
    h_wxyz = hpqrs[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]

    ACAC_AA = np.zeros((nact*ncore, nact*ncore))
    ACAC_BA = np.zeros((nact*ncore, nact*ncore)) 
    ACAC_BB = np.zeros((nact*ncore, nact*ncore))    
    ### Intermediates
    # (2(Ix|Jy) - (IJ|xy) - (Iy|Jx)) Dzy
    Int1_AA = 2* np.einsum('IxJy, zy->xIzJ', h_IxJy, Daa) - np.einsum('IJxy,zy->xIzJ',h_IJxy, Daa) - np.einsum('JxIy,zy->xIzJ',h_IxJy,Daa)
    Int1_BB = 2* np.einsum('IxJy, zy->xIzJ', h_IxJy, Dbb) - np.einsum('IJxy,zy->xIzJ',h_IJxy, Dbb) - np.einsum('JxIy,zy->xIzJ',h_IxJy,Dbb)
    Int1_BA = 2* np.einsum('IxJy, zy->xIzJ', h_IxJy, Daa) 
    # (2(Iy|Jz) - (IJ|yz) - (Iz|Jy)) Dyx = Int1_AA[z,J,x,I]
    # (2(Ibyb|Jaza) - (IJ|yz) - (Ibza|Jaya)) Dybxb 
    Int2_BA = 2* np.einsum('IyJz, yx->xIzJ', h_IxJy, Dbb) 
    # ((IJ|yw) - (Iw|Jy)) Dxy,wz
    # ((IaJa|yawa) - (Iawa|Jaya)) Dxaya,waza  + (IaJa|ybwb) Dxayb,wbza 
    # = ((IaJa|yawa) (Dxaya,waza + Dxayb,wbza)  - (Iawa|Jaya) Dxaya,waza 
    # = ((IaJa|yawa) (Dyxa,zawa + Dybxa,zawb)  - (Iawa|Jaya) Dxaya,waza 
    Int3_AA = np.einsum('IJyw,yxzw->xIzJ',h_IJxy, Daaaa+Dbaab) - np.einsum('IwJy,yxzw->xIzJ',h_IxJy, Daaaa)
    # ((IbJb|ybwb) - (Ibwb|Jbyb)) Dxbyb,wbzb  + (IbJb|yawa) Dxbya,wazb
    # = ((IbJb|ybwb) (Dxbyb,zbwb + Dxbya,wazb)  - (Ibwb|Jbyb) Dxbyb,zbwb   
    Int3_BB = np.einsum('IJyw,xywz->x IzJ',h_IJxy, Dbbbb+Dbaab) - np.einsum('IwJy,yxzw->xIzJ',h_IxJy, Dbbbb)    
    # ( - (Ibwb|Jaya)) Dxbya,wbza = (Ibwb|Jaya)) Dxbya,zawb
    Int3_BA = np.einsum('IwJy,xyzw->xIzJ',h_IxJy, Dbaab)
    # ( - (Iawa|Jbyb)) Dxayb,zbwa 
    Int3_AB = np.einsum('IwJy,yxwz->xIzJ',h_IxJy, Dbaab)    
    # (Iy|Jw) Dzx,wy
    # (Iaya|Jawa) Dzaxa,waya
    Int4_AA = np.einsum('IyJw,zxwy->xIzJ',h_IxJy, Daaaa)
    Int4_BB = np.einsum('IyJw,zxwy->xIzJ',h_IxJy, Dbbbb) 
    # (Ibyb|Jawa) Dzaxb,wayb
    Int4_BA = -np.einsum('IyJw,xzwy->xIzJ',h_IxJy, Dbaab)
    # (yv|wx) Dyw,zv
    # (yava|waxa) Dyawa,zava +     (ybvb|waxa) Dybwa,zavb  
    Int5_AA = np.einsum('yvwx,ywzv->zx',h_wxyz, Daaaa + Dbaab) 
    # (ybvb|wbxb) Dybwb,zbvb +     (yava|wbxb) Dyawb,zbva 
    Int5_BB = np.einsum('yvwx,wyvz->zx',h_wxyz, Dbbbb + Dbaab)     
    xI = 0
    for x in range(nact):
        for I in range(ncore):
            zJ = 0
            for z in range(nact):
                for J in range(ncore):
                    ACAC_AA[xI, zJ] = htilde[I,J] * Daa[z,x] 
                    ACAC_AA[xI, zJ] += 2*h_IxJy[I,x,J,z] - h_IxJy[J,x,I,z] - h_IJxy[I,J,x,z]
                    ACAC_AA[xI, zJ] += - Int1_AA[x,I,z,J] - Int1_AA[z,J,x,I]
                    ACAC_AA[xI, zJ] += + Int3_AA[x,I,z,J] 
                    ACAC_AA[xI, zJ] += - Int4_AA[x,I,z,J]
                    
                    ACAC_BB[xI, zJ] = htilde[I,J] * Dbb[z,x] 
                    ACAC_BB[xI, zJ] += 2*h_IxJy[I,x,J,z] - h_IxJy[J,x,I,z] - h_IJxy[I,J,x,z]
                    ACAC_BB[xI, zJ] += - Int1_BB[x,I,z,J] - Int1_BB[z,J,x,I]
                    ACAC_BB[xI, zJ] += + Int3_BB[x,I,z,J] 
                    ACAC_BB[xI, zJ] += - Int4_BB[x,I,z,J]
                    
                    ACAC_BA[xI, zJ] += 2*h_IxJy[I,x,J,z] 
                    ACAC_BA[xI, zJ] += - Int1_BA[x,I,z,J] - Int2_BA[x,I,z,J]
                    ACAC_BA[xI, zJ] += + Int3_BA[x,I,z,J] - Int4_BA[x,I,z,J]
                    
                    if I==J:
                        ACAC_AA[xI, zJ] += - Dh_A[z+ncore,x+ncore]  + Ftilde_A[z+ncore,x+ncore] - Int5_AA[z,x]
                        ACAC_BB[xI, zJ] += - Dh_B[z+ncore,x+ncore]  + Ftilde_B[z+ncore,x+ncore] - Int5_BB[z,x]
                    if x==z:
                        ACAC_AA[xI, zJ] -= Ftilde_A[I,J]
                        ACAC_BB[xI, zJ] -= Ftilde_B[I,J]
                    zJ += 1
            xI += 1
    ACAC_AA = 0.5 * (ACAC_AA + ACAC_AA.T)
    ACAC_BB = 0.5 * (ACAC_BB + ACAC_BB.T)  
    return ACAC_AA, ACAC_BA, ACAC_BB

def Hess_AAAC(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    ########################
    ## Act-Act, Act-core  ##
    ########################    
    natt = nact*(nact-1)//2
    h_Ixyz = hpqrs[:ncore,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
   
    AAAC_AA = np.zeros((natt, nact*ncore), dtype=float)
    AAAC_BA = np.zeros((natt, nact*ncore), dtype=float)
    AAAC_AB = np.zeros((natt, nact*ncore), dtype=float)    
    AAAC_BB = np.zeros((natt, nact*ncore), dtype=float) 
    # ( 2(Jz|xw) - (Jx|wz) - (Jw|xz) ) Dyw
    Int1_AA = 2* np.einsum('Jzxw,yw->xyzJ',h_Ixyz,Daa) - np.einsum('Jxwz,yw->xyzJ',h_Ixyz,Daa) - np.einsum('Jwxz,yw->xyzJ',h_Ixyz,Daa)
    Int1_BB = 2* np.einsum('Jzxw,yw->xyzJ',h_Ixyz,Dbb) - np.einsum('Jxwz,yw->xyzJ',h_Ixyz,Dbb) - np.einsum('Jwxz,yw->xyzJ',h_Ixyz,Dbb)    
    Int1_BA = 2* np.einsum('Jzxw,yw->xyzJ',h_Ixyz,Daa) 
    Int1_AB = 2* np.einsum('Jzxw,yw->xyzJ',h_Ixyz,Dbb)
    # (Jw|yv) Dzx,wv
    # (Jawa|yava) Dzaxa,wava
    Int2_AA = np.einsum('Jwyv,zxwv->xyzJ',h_Ixyz, Daaaa)
    # (Jbwb|ybvb) Dzbxb,wbvb    
    Int2_BB = np.einsum('Jwyv,zxwv->xyzJ',h_Ixyz, Dbbbb)    
    # (Jawa|ybvb) Dzaxb,wavb    
    Int2_BA = -np.einsum('Jwyv, xzwv->xyzJ',h_Ixyz, Dbaab)
    # (Jbwb|yava) Dzbxa,wbva    
    Int2_AB = -np.einsum('Jwyv, zxvw->xyzJ',h_Ixyz, Dbaab)    
    # ((Jx|vw) - (Jw|vx)) Dzv,wy
    # (Jaxa|vawa) Dzava,waya + (Jaxa|vbwb) Dzavb,wbya - (Jawa|vaxa) Dzava,waya
    Int3_AA = np.einsum('Jxvw, vzyw-> xyzJ',h_Ixyz, Daaaa+Dbaab) - np.einsum('Jwvx,vzyw->xyzJ', h_Ixyz, Daaaa)
    # (Jbxb|vbwb) Dzbvb,wbyb + (Jbxb|vawa) Dzbva,wayb - (Jbwb|vbxb) Dzbvb,wbyb
    Int3_BB = np.einsum('Jxvw, zvwy-> xyzJ',h_Ixyz, Dbbbb+Dbaab) - np.einsum('Jwvx,vzyw->xyzJ', h_Ixyz, Dbbbb)   
    # - (Jawa|vbxb)) Dzavb,wayb =  (Jawa|vbxb)) Dvbza,wayb
    Int3_BA = np.einsum('Jwvx, vzwy -> xyzJ',h_Ixyz, Dbaab)
    # - (Jbwb|vaxa)) Dzbva,wbya =  (Jbwb|vaxa)) Dzbva,yawb
    Int3_AB = np.einsum('Jwvx, zvyw -> xyzJ',h_Ixyz, Dbaab)
    # (Jt|wu) Dxw,ut
    # (Jata|waua) Dxawa,uata + (Jata|wbub) Dxawb,ubta    
    Int4_AA = np.einsum('Jtwu, wxtu -> xJ', h_Ixyz, Daaaa+Dbaab)
    Int4_BB = np.einsum('Jtwu, xwut -> xJ', h_Ixyz, Dbbbb+Dbaab)    
    xy = 0
    for x in range(nact):
        for y in range(x):
            zJ = 0
            for z in range(nact):
                for J in range(ncore):
                    AAAC_AA[xy, zJ]  = - htilde[J,x+ncore] * Daa[y,z] + htilde[y+ncore,J] * Daa[z,x]
                    AAAC_AA[xy, zJ] +=   Int1_AA[x,y,z,J] - Int1_AA[y,x,z,J]
                    AAAC_AA[xy, zJ] += - Int2_AA[x,y,z,J] + Int2_AA[y,x,z,J]
                    AAAC_AA[xy, zJ] += - Int3_AA[x,y,z,J] + Int3_AA[y,x,z,J]
                    AAAC_BB[xy, zJ]  = - htilde[J,x+ncore] * Dbb[y,z] + htilde[y+ncore,J] * Dbb[z,x]
                    AAAC_BB[xy, zJ] +=   Int1_BB[x,y,z,J] - Int1_BB[y,x,z,J]
                    AAAC_BB[xy, zJ] += - Int2_BB[x,y,z,J] + Int2_BB[y,x,z,J]
                    AAAC_BB[xy, zJ] += - Int3_BB[x,y,z,J] + Int3_BB[y,x,z,J]
                    
                    AAAC_BA[xy, zJ]  =   Int1_BA[x,y,z,J] - Int1_BA[y,x,z,J]
                    AAAC_BA[xy, zJ] += - Int2_BA[x,y,z,J] + Int2_BA[y,x,z,J]
                    AAAC_BA[xy, zJ] += - Int3_BA[x,y,z,J] + Int3_BA[y,x,z,J]   
                    AAAC_AB[xy, zJ]  =   Int1_AB[x,y,z,J] - Int1_AB[y,x,z,J]                    
                    AAAC_AB[xy, zJ] += - Int2_AB[x,y,z,J] + Int2_AB[y,x,z,J]
                    AAAC_AB[xy, zJ] += - Int3_AB[x,y,z,J] + Int3_AB[y,x,z,J]                       
                    if z==y:
                        AAAC_AA[xy, zJ] += 0.5 * (Ftilde_A[J,x+ncore] + Dh_A[x+ncore,J] + Int4_AA[x,J])
                        AAAC_BB[xy, zJ] += 0.5 * (Ftilde_B[J,x+ncore] + Dh_B[x+ncore,J] + Int4_BB[x,J])
                    if z==x:
                        AAAC_AA[xy, zJ] -= 0.5 * (Ftilde_A[J,y+ncore] + Dh_A[y+ncore,J] + Int4_AA[y,J])
                        AAAC_BB[xy, zJ] -= 0.5 * (Ftilde_B[J,y+ncore] + Dh_B[y+ncore,J] + Int4_BB[y,J])
                    zJ += 1
            xy += 1
    return AAAC_AA, AAAC_BA, AAAC_AB, AAAC_BB

def Hess_VCAC(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #########################
    ## Vir-Core, Act-Core  ##
    #########################               
    VCAC_AA = np.zeros((ncore*nsec, nact*ncore), dtype=float)
    VCAC_BA = np.zeros((ncore*nsec, nact*ncore), dtype=float)
    VCAC_AB = np.zeros((ncore*nsec, nact*ncore), dtype=float)    
    VCAC_BB = np.zeros((ncore*nsec, nact*ncore), dtype=float)
    h_AIxJ = hpqrs[ ncore+nact:, :ncore,  ncore:ncore+nact, :ncore]
    h_AxIJ = hpqrs[ ncore+nact:, ncore:ncore+nact, :ncore,:ncore]
    h_Axyz = hpqrs[ncore+nact:,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]    
     
    # ( 2(AI|vJ) - (AJ|vI) - (Av|IJ) ) Dvz
    Int1_AA =2* np.einsum('AIvJ, vz -> AIzJ', h_AIxJ, Daa) - np.einsum('AJvI, vz -> AIzJ',h_AIxJ, Daa) - np.einsum('AvIJ, vz -> AIzJ', h_AxIJ, Daa)
    Int1_BB =2* np.einsum('AIvJ, vz -> AIzJ', h_AIxJ, Dbb) - np.einsum('AJvI, vz -> AIzJ',h_AIxJ, Dbb) - np.einsum('AvIJ, vz -> AIzJ', h_AxIJ, Dbb)
    Int1_BA =2* np.einsum('AIvJ, vz -> AIzJ', h_AIxJ, Daa)
    Int1_AB =2* np.einsum('AIvJ, vz -> AIzJ', h_AIxJ, Dbb)
    # (Au|tv) Dtuzv
    # (Aaua|tava) Dtaua,zava + (Aaua|tbvb) Dtbua,zavb
    Int2_AA = np.einsum('Autv, tuzv -> zA', h_Axyz, Daaaa+Dbaab)
    Int2_BB = np.einsum('Autv, utvz -> zA', h_Axyz, Dbbbb+Dbaab)    
    
    AI = 0
    for A in range(nsec):
        for I in range(ncore):
            zJ = 0
            for z in range(nact):
                for J in range(ncore):
                    VCAC_AA[AI,zJ] = 2 * h_AIxJ[A,I,z,J] - h_AxIJ[A,z,I,J] - h_AIxJ[A,J,z,I] - Int1_AA[A,I,z,J]
                    VCAC_BB[AI,zJ] = 2 * h_AIxJ[A,I,z,J] - h_AxIJ[A,z,I,J] - h_AIxJ[A,J,z,I] - Int1_BB[A,I,z,J]                    
                    VCAC_BA[AI,zJ] = 2 * h_AIxJ[A,I,z,J] - Int1_BA[A,I,z,J]
                    VCAC_AB[AI,zJ] = 2 * h_AIxJ[A,I,z,J] - Int1_AB[A,I,z,J]                    
                    if I==J:
                        VCAC_AA[AI,zJ] += 0.5*(Ftilde_A[A+ncore+nact,z+ncore] + Ftilde_A[z+ncore,A+ncore+nact] - Dh_A[z+ncore,A+ncore+nact] - Int2_AA[z,A])
                        VCAC_BB[AI,zJ] += 0.5*(Ftilde_B[A+ncore+nact,z+ncore] + Ftilde_B[z+ncore,A+ncore+nact] - Dh_B[z+ncore,A+ncore+nact] - Int2_BB[z,A])
                    zJ += 1
            AI += 1
    return VCAC_AA, VCAC_BA, VCAC_AB, VCAC_BB

def Hess_VCAA(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #########################
    ## Vir-Core, Act-Act   ##
    #########################
    natt = nact*(nact-1)//2
    VCAA_AA = np.zeros((ncore*nsec, natt), dtype=float)
    VCAA_BA = np.zeros((ncore*nsec, natt), dtype=float)
    VCAA_AB = np.zeros((ncore*nsec, natt), dtype=float)    
    VCAA_BB = np.zeros((ncore*nsec, natt), dtype=float)
    h_AIxy = hpqrs[ncore+nact:,:ncore,ncore:ncore+nact, ncore:ncore+nact]
    h_AxyI = hpqrs[ncore+nact:, ncore:ncore+nact,ncore:ncore+nact,  :ncore]
 
    # ( 2(AI|zv) - (Az|vI) - (Av|zI) ) Dwv
    Int1_AA =2* np.einsum('AIzv, wv -> AIzw', h_AIxy, Daa) - np.einsum('AzvI, wv -> AIzw',h_AxyI, Daa) - np.einsum('AvzI, wv -> AIzw', h_AxyI, Daa)
    Int1_BB =2* np.einsum('AIzv, wv -> AIzw', h_AIxy, Dbb) - np.einsum('AzvI, wv -> AIzw',h_AxyI, Dbb) - np.einsum('AvzI, wv -> AIzw', h_AxyI, Dbb)
    Int1_BA =2* np.einsum('AIzv, wv -> AIzw', h_AIxy, Daa) 
    Int1_AB =2* np.einsum('AIzv, wv -> AIzw', h_AIxy, Dbb) 
    AI = 0
    for A in range(nsec):
        for I in range(ncore):
            zw = 0
            for z in range(nact):
                for w in range(z):
                    VCAA_AA[AI,zw] = Int1_AA[A,I,z,w] -  Int1_AA[A,I,w,z]
                    VCAA_BB[AI,zw] = Int1_BB[A,I,z,w] -  Int1_BB[A,I,w,z]                    
                    VCAA_BA[AI,zw] = Int1_BA[A,I,z,w] -  Int1_BA[A,I,w,z]
                    VCAA_AB[AI,zw] = Int1_AB[A,I,z,w] -  Int1_AB[A,I,w,z]
                    zw += 1
            AI += 1
    return VCAA_AA, VCAA_BA, VCAA_AB, VCAA_BB    

def Hess_VCVC(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    ########################
    ## Vir-core, Vir-core ##
    ########################
    h_AIBJ = hpqrs[ncore+nact:, :ncore, ncore+nact:, :ncore]
    h_ABIJ = hpqrs[ncore+nact:, ncore+nact:, :ncore, :ncore]
    VCVC_AA = np.zeros((nsec*ncore, nsec*ncore))
    VCVC_BA = np.zeros((nsec*ncore, nsec*ncore))
    VCVC_BB = np.zeros((nsec*ncore, nsec*ncore))
    AI = 0
    for A in range(nsec):
        for I in range(ncore):
            BJ = 0
            for B in range(nsec):
                for J in range(ncore):
                    VCVC_AA[AI,BJ] = 2* h_AIBJ[A,I,B,J] - h_AIBJ[A,J,B,I] - h_ABIJ[A,B,I,J]
                    VCVC_BB[AI,BJ] = VCVC_AA[AI,BJ]
                    VCVC_BA[AI,BJ] = 2* h_AIBJ[A,I,B,J] 
                    if I==J:
                        VCVC_AA[AI,BJ] += Ftilde_A[B+ncore+nact,A+ncore+nact]
                        VCVC_BB[AI,BJ] += Ftilde_B[B+ncore+nact,A+ncore+nact]
                    if A==B:
                        VCVC_AA[AI,BJ] -= Ftilde_A[I,J]
                        VCVC_BB[AI,BJ] -= Ftilde_B[I,J]
                    BJ += 1
            AI += 1
    return VCVC_AA, VCVC_BA, VCVC_BB 


def Hess_VAAC(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    ########################
    ## Vir-Act, Act-Core  ##
    ########################               
    VAAC_AA = np.zeros((nact*nsec, nact*ncore), dtype=float)
    VAAC_BA = np.zeros((nact*nsec, nact*ncore), dtype=float)
    VAAC_AB = np.zeros((nact*nsec, nact*ncore), dtype=float)    
    VAAC_BB = np.zeros((nact*nsec, nact*ncore), dtype=float)
    h_AxIy = hpqrs[ncore+nact:,ncore:ncore+nact,:ncore, ncore:ncore+nact]
    h_AIxy = hpqrs[ncore+nact:,:ncore,ncore:ncore+nact,ncore:ncore+nact]
    
    # ( 2(Aw|Jz) - (Az|Jw) - (AJ|wz) ) Dyw
    Int1_AA = 2 * np.einsum('AwJz, yw -> AyzJ', h_AxIy, Daa) -  np.einsum('AzJw, yw -> AyzJ', h_AxIy, Daa) - np.einsum('AJwz, yw -> AyzJ', h_AIxy, Daa) 
    Int1_BB = 2 * np.einsum('AwJz, yw -> AyzJ', h_AxIy, Dbb) -  np.einsum('AzJw, yw -> AyzJ', h_AxIy, Dbb) - np.einsum('AJwz, yw -> AyzJ', h_AIxy, Dbb) 
    Int1_BA = 2 * np.einsum('AwJz, yw -> AyzJ', h_AxIy, Dbb) 
    Int1_AB = 2 * np.einsum('AwJz, yw -> AyzJ', h_AxIy, Daa)     
    # (Av|Jw) Dvw,yz
    Int2_AA = np.einsum('AvJw, vwyz->AyzJ',h_AxIy, Daaaa)
    Int2_BB = np.einsum('AvJw, vwyz->AyzJ',h_AxIy, Dbbbb)    
    # (Abvb|Jawa) Dvbwa,ybza
    Int2_BA = -np.einsum('AvJw, vwzy->AyzJ',h_AxIy, Dbaab)
    # (Aava|Jbwb) Dvawb,yazb
    Int2_AB = -np.einsum('AvJw, wvyz->AyzJ',h_AxIy, Dbaab)
    # ((AJ|vw) - (Av|Jw)) Dzv,wy
    # (AaJa|vawa) Dzava,waya + (AaJa|vbwb) Dzavb,wbya - (Aava|Jawa) Dzava,waya
    Int3_AA = np.einsum('AJvw, vzyw -> AyzJ', h_AIxy, Daaaa+Dbaab ) - np.einsum('AvJw, zvwy->AyzJ', h_AxIy, Daaaa)
    Int3_BB = np.einsum('AJvw, zvwy -> AyzJ', h_AIxy, Dbbbb+Dbaab ) - np.einsum('AvJw, zvwy->AyzJ', h_AxIy, Dbbbb)    
    # - (Abvb|Jawa) Dzavb,wayb = (Abvb|Jawa) Dvbza,wayb
    Int3_BA = np.einsum('AvJw,vzwy->AyzJ',h_AxIy, Dbaab)
    # - (Aava|Jbwb) Dzbva,wbya = (Aava|Jbwb) Dzbva,yawb
    Int3_AB = np.einsum('AvJw,zvyw->AyzJ',h_AxIy, Dbaab)    
    
    Ay = 0
    for A in range(nsec):
        for y in range(nact):
            zJ = 0
            for z in range(nact):
                for J in range(ncore):
                    VAAC_AA[Ay, zJ] = -htilde[J,A+ncore+nact] * Daa[y,z] + Int1_AA[A,y,z,J] + Int2_AA[A,y,z,J] - Int3_AA[A,y,z,J]
                    VAAC_BB[Ay, zJ] = -htilde[J,A+ncore+nact] * Dbb[y,z] + Int1_BB[A,y,z,J] + Int2_BB[A,y,z,J] - Int3_BB[A,y,z,J]                    
                    VAAC_BA[Ay, zJ] = Int1_BA[A,y,z,J] + Int2_BA[A,y,z,J] - Int3_BA[A,y,z,J]
                    VAAC_AB[Ay, zJ] = Int1_AB[A,y,z,J] + Int2_AB[A,y,z,J] - Int3_AB[A,y,z,J]                    
                    if z==y:
                        VAAC_AA[Ay, zJ] += 0.5 * Ftilde_A[J,A+ncore+nact] 
                        VAAC_BB[Ay, zJ] += 0.5 * Ftilde_B[J,A+ncore+nact]
                    zJ += 1
            Ay += 1
    return VAAC_AA, VAAC_BA, VAAC_AB, VAAC_BB

def Hess_VAAA(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #######################
    ## Vir-Act, Act-Act  ##
    #######################    

    natt = nact*(nact-1)//2
    h_Axyz = hpqrs[ncore+nact:,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    
    VAAA_AA = np.zeros((nact*nsec, natt), dtype=float)
    VAAA_BA = np.zeros((nact*nsec, natt), dtype=float)
    VAAA_AB = np.zeros((nact*nsec, natt), dtype=float)    
    VAAA_BB = np.zeros((nact*nsec, natt), dtype=float)

    # (At|uz) Dtu,yw
    # (Aata|uaza) Dtaua,yawa
    Int1_AA = np.einsum('Atuz, tuyw-> Ayzw', h_Axyz, Daaaa)
    Int1_BB = np.einsum('Atuz, tuyw-> Ayzw', h_Axyz, Daaaa)
    # (Abtb|uaza) Dtbua,ybwa
    Int1_BA = - np.einsum('Atuz,tuwy->Ayzw', h_Axyz, Dbaab)
    # (Aata|ubzb) Dtaub,yawb
    Int1_AB = - np.einsum('Atuz,utyw->Ayzw', h_Axyz, Dbaab)    
    # ((Az|tu) - (At|zu)) Dwt,uy
    # (Aaza|taua) Dwata,uaya +  (Aaza|tbub) Dwatb,ubya - (Aata|zaua) Dwata,uaya
    Int2_AA = np.einsum('Aztu, twyu -> Ayzw', h_Axyz, Daaaa+Dbaab) - np.einsum('Atzu, wtuy->Ayzw',h_Axyz,Daaaa)
    Int2_BB = np.einsum('Aztu, wtuy -> Ayzw', h_Axyz, Dbbbb+Dbaab) - np.einsum('Atzu, wtuy->Ayzw',h_Axyz,Dbbbb)
    # - (Abtb|zaua) Dwatb,uayb
    Int2_BA = np.einsum('Atzu, twuy -> Ayzw', h_Axyz, Dbaab)
    # - (Aata|zbub) Dwbta,ubya   
    Int2_AB = np.einsum('Atzu, wtyu -> Ayzw', h_Axyz, Dbaab)
    # (Au|tv) Dtu,wv
    # (Aaua|tava) Dtaua,wava + (Aaua|tbvb) Dtbua,wavb
    Int3_AA = np.einsum('Autv, tuwv -> Aw', h_Axyz, Daaaa+Dbaab)
    Int3_BB = np.einsum('Autv, utvw -> Aw', h_Axyz, Dbbbb+Dbaab)    
    Ay = 0
    for A in range(nsec):
        for y in range(nact):
            zw = 0
            for z in range(nact):
                for w in range(z):
                    VAAA_AA[Ay, zw]  = - htilde[w+ncore,A+ncore+nact] * Daa[y,z] + htilde[z+ncore,A+ncore+nact] * Daa[y,w] 
                    VAAA_AA[Ay, zw] -= Int1_AA[A,y,z,w] - Int1_AA[A,y,w,z]
                    VAAA_AA[Ay, zw] += Int2_AA[A,y,z,w] - Int2_AA[A,y,w,z]   
                    VAAA_BB[Ay, zw]  = - htilde[w+ncore,A+ncore+nact] * Dbb[y,z] + htilde[z+ncore,A+ncore+nact] * Dbb[y,w] 
                    VAAA_BB[Ay, zw] -= Int1_BB[A,y,z,w] - Int1_BB[A,y,w,z]
                    VAAA_BB[Ay, zw] += Int2_BB[A,y,z,w] - Int2_BB[A,y,w,z] 
                    VAAA_BA[Ay, zw] -= Int1_BA[A,y,z,w] - Int1_BA[A,y,w,z]                    
                    VAAA_BA[Ay, zw] += Int2_BA[A,y,z,w] - Int2_BA[A,y,w,z]
                    VAAA_AB[Ay, zw] -= Int1_AB[A,y,z,w] - Int1_AB[A,y,w,z]                          
                    VAAA_AB[Ay, zw] += Int2_AB[A,y,z,w] - Int2_AB[A,y,w,z]      
                    if z==y:
                        VAAA_AA[Ay, zw] += 0.5 * (Dh_A[w+ncore,A+ncore+nact] + Int3_AA[A,w])
                        VAAA_BB[Ay, zw] += 0.5 * (Dh_B[w+ncore,A+ncore+nact] + Int3_BB[A,w])
                    if w==y:
                        VAAA_AA[Ay, zw] -= 0.5 * (Dh_A[z+ncore,A+ncore+nact] + Int3_AA[A,z])
                        VAAA_BB[Ay, zw] -= 0.5 * (Dh_B[z+ncore,A+ncore+nact] + Int3_BB[A,z]) 
                       
                    zw += 1
            Ay += 1
    return VAAA_AA, VAAA_BA, VAAA_AB, VAAA_BB

def Hess_VAVC(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #########################
    ## Vir-Act, Vir-Core   ##
    #########################
    VAVC_AA = np.zeros((nact*nsec, nsec*ncore), dtype=float)
    VAVC_BA = np.zeros((nact*nsec, nsec*ncore), dtype=float)
    VAVC_AB = np.zeros((nact*nsec, nsec*ncore), dtype=float)    
    VAVC_BB = np.zeros((nact*nsec, nsec*ncore), dtype=float)
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

def Hess_VAVA(ncore, nact, nsec, htilde, hpqrs, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb):
    #######################
    ## Vir-Act, Vir-Act  ##
    #######################   
    VAVA_AA = np.zeros((nact*nsec, nact*nsec), dtype=float)
    VAVA_BA = np.zeros((nact*nsec, nact*nsec), dtype=float)
    VAVA_AB = np.zeros((nact*nsec, nact*nsec), dtype=float)    
    VAVA_BB = np.zeros((nact*nsec, nact*nsec), dtype=float)
    h_ABxy = hpqrs[ ncore+nact:, ncore+nact:, ncore:ncore+nact, ncore:ncore+nact]
    h_AxBy = hpqrs[ ncore+nact:, ncore:ncore+nact, ncore+nact:, ncore:ncore+nact]
    h_wxyz = hpqrs[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    # (At|Bu) Dtu,yw
    Int1_AA = np.einsum('AtBu, tuyw -> AyBw', h_AxBy, Daaaa)
    Int1_BB = np.einsum('AtBu, tuyw -> AyBw', h_AxBy, Dbbbb)    
    # (Abtb|Baua) Dtbua,ybwa 
    Int1_BA = -np.einsum('AtBu, tuwy -> AyBw', h_AxBy, Dbaab)
    Int1_AB = -np.einsum('AtBu, utyw -> AyBw', h_AxBy, Dbaab)
    # ((AB|tu) - (At|Bu)) Dwt,uy
    # (AaBa|taua) Dwata,uaya + (AaBa|tbub) Dwatb,ubya - (Aata|Baua) Dwata,uaya
    Int2_AA = np.einsum('ABtu, twyu->AyBw',h_ABxy, Daaaa + Dbaab) - np.einsum('AtBu,wtuy->AyBw',h_AxBy, Daaaa)
    Int2_BB = np.einsum('ABtu, wtuy->AyBw',h_ABxy, Dbbbb + Dbaab) - np.einsum('AtBu,wtuy->AyBw',h_AxBy, Dbbbb)
    # -(Abtb|Baua) Dwatb,uayb = (Abtb|Baua) Dtbwa,uayb
    Int2_BA = np.einsum('AtBu,twuy->AyBw',h_AxBy, Dbaab)
    Int2_AB = np.einsum('AtBu,wtyu->AyBw',h_AxBy, Dbaab)  
    # (yt|vu) Dwv,ut
    # (yata|vaua) Dwava, uata + (yata|vbub) Dwavb,ubta
    Int3_AA = np.einsum('ytvu, vwtu->yw',h_wxyz,Daaaa+Dbaab)
    Int3_BB = np.einsum('ytvu, wvut->yw',h_wxyz,Daaaa+Dbaab)    
    Ay = 0
    for A in range(nsec):
        for y in range(nact):
            Bw = 0
            for B in range(nsec):
                for w in range(nact):
                    VAVA_AA[Ay,Bw] = htilde[B+ncore+nact,A+ncore+nact] * Daa[y,w] - Int1_AA[A,y,B,w] + Int2_AA[A,y,B,w]
                    VAVA_BB[Ay,Bw] = htilde[B+ncore+nact,A+ncore+nact] * Dbb[y,w] - Int1_BB[A,y,B,w] + Int2_BB[A,y,B,w]
                    VAVA_BA[Ay,Bw] = -Int1_BA[A,y,B,w] + Int2_AB[A,y,B,w]
                    if A==B:
                        VAVA_AA[Ay,Bw] -= 0.5*( Dh_A[w+ncore,y+ncore] + Dh_A[y+ncore,w+ncore] + Int3_AA[y,w] + Int3_AA[w,y])
                        VAVA_BB[Ay,Bw] -= 0.5*( Dh_B[w+ncore,y+ncore] + Dh_B[y+ncore,w+ncore] + Int3_BB[y,w] + Int3_BB[w,y])
                    Bw += 1
            Ay += 1
    return VAVA_AA, VAVA_BA, VAVA_BB    

def get_orbital_hessian(Quket, DA=None, DB=None, Daaaa=None, Dbbbb=None, Dbaab=None, state=None):
    """
    Inefficiently compute 'whole' orbital Hessian H[tu,vw] = <[[H, Etu - Eut], Evw - Ewv]>
    using 1- and 2-RDMs.
    The size of Hessian is (2*nott, 2*nott) where nott = norbs * (norbs-1)/2 (# of 'tu' orbital pairs for each spin).
    
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
    Returns:
        Hess (2darray): Orbital Hessian (2*nott, 2*nott)

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
    Hess = np.zeros((nott*2, nott*2), dtype=float)  # Currently spin is explicitly considered 
    


    h1 = Quket.one_body_integrals
    Ecore, htilde = get_htilde(Quket)

    #############
    ## Act-Act ##
    #############
    AAAA_AA, AAAA_BA, AAAA_BB = Hess_AAAA(ncore, nact, nsec, htilde, Quket.two_body_integrals, Daa, Dbb, Daaaa, Dbaab, Dbbbb)
    

    ###################
    ## Other sectors ##
    ###################
    
    Ftilde_A, Ftilde_B = get_Fock(Quket.one_body_integrals, Quket.two_body_integrals, DA, DB)
    # Dzy h~yx
    Dh_A = np.einsum('zy,yx->zx',DA,htilde)
    Dh_B = np.einsum('zy,yx->zx',DB,htilde)  

    ########################
    ## Act-core, Act-core ##
    ########################
    ACAC_AA, ACAC_BA, ACAC_BB = Hess_ACAC(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    

    ########################
    ## Act-Act, Act-core  ##
    ########################
    AAAC_AA, AAAC_BA, AAAC_AB, AAAC_BB = Hess_AAAC(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    
           
    #########################
    ## Vir-Core, Act-Core  ##
    #########################
    VCAC_AA, VCAC_BA, VCAC_AB, VCAC_BB = Hess_VCAC(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)
            
    #########################
    ## Vir-Core, Act-Act   ##
    #########################               
    VCAA_AA, VCAA_BA, VCAA_AB, VCAA_BB = Hess_VCAA(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)

    #########################
    ## Vir-Core, Vir-Core  ##
    #########################
    VCVC_AA, VCVC_BA, VCVC_BB = Hess_VCVC(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)

    ########################
    ## Vir-Act, Act-Core  ##
    ########################
    VAAC_AA, VAAC_BA, VAAC_AB, VAAC_BB = Hess_VAAC(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)    
            
    #######################
    ## Vir-Act, Act-Act  ##
    #######################
    VAAA_AA, VAAA_BA, VAAA_AB, VAAA_BB = Hess_VAAA(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)

    ########################
    ## Vir-Act, Vir-Core  ##
    ########################   
    VAVC_AA, VAVC_BA, VAVC_AB, VAVC_BB = Hess_VAVC(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)
    
    #######################
    ## Vir-Act, Vir-Act  ##
    #######################               
    VAVA_AA, VAVA_BA, VAVA_BB = Hess_VAVA(ncore, nact, nsec, htilde, Quket.two_body_integrals, Dh_A, Dh_B, Ftilde_A, Ftilde_B, Daa, Dbb, Daaaa, Dbaab, Dbbbb)

       
####################
### Put together ###
####################
    #######################
    ## Act-Act, Act-Act  ##
    ####################### 
    xy = 0
    for x in range(nact):
        for y in range(x):
            x0 = x + ncore
            y0 = y + ncore
            xy0A = (x0-1)*x0//2 + y0
            xy0B = xy0A + nott
            zw = 0
            for z in range(nact):
                for w in range(z):
                    z0 = z + ncore
                    w0 = w + ncore
                    zw0A = (z0-1)*z0//2 + w0
                    zw0B = zw0A + nott
                    Hess[xy0A, zw0A] = AAAA_AA[xy, zw]
                    Hess[xy0B, zw0B] = AAAA_BB[xy, zw]
                    Hess[xy0B, zw0A] = AAAA_BA[xy, zw]
                    Hess[xy0A, zw0B] = AAAA_BA[zw, xy]
#                    if abs(AAAA_AA[xy, zw] - corr[xy0A, zw0A]) + abs(AAAA_BB[xy, zw] - corr[xy0B, zw0B]) + abs(AAAA_BA[xy, zw] - corr[xy0B, zw0A]) > 1e-6:
#                        print(f'AAAA   {x,y,z,w}   AA = {AAAA_AA[xy, zw] - corr[xy0A, zw0A]:+12.10f} ,   BB = {AAAA_BB[xy, zw] - corr[xy0B, zw0B]:+12.10f} ,   BA = {AAAA_BA[xy, zw] -corr[xy0B, zw0A]:+12.10f} ')
                    zw += 1
            xy += 1    
    ########################
    ## Act-Core, Act-Core ##
    ######################## 
    xI = 0
    for x in range(nact):
        for I in range(ncore):
            x0 = x + ncore
            I0 = I
            xI0A = (x0-1)*x0//2 + I0
            xI0B = xI0A + nott
            zJ = 0
            for z in range(nact):
                for J in range(ncore):
                    z0 = z + ncore
                    J0 = J
                    zJ0A = (z0-1)*z0//2 + J0
                    zJ0B = zJ0A + nott
                    Hess[xI0A, zJ0A] = ACAC_AA[xI, zJ]
                    Hess[xI0B, zJ0B] = ACAC_BB[xI, zJ]
                    Hess[xI0B, zJ0A] = ACAC_BA[xI, zJ]
                    Hess[xI0A, zJ0B] = ACAC_BA[zJ, xI]
#                    if abs(ACAC_AA[xI, zJ] - corr[xI0A, zJ0A]) + abs(ACAC_BB[xI, zJ] - corr[xI0B, zJ0B]) + abs(ACAC_BA[xI, zJ] - corr[xI0B, zJ0A]) > 1e-6:
#                        print(f'ACAC   {x0,I0,z0,J0}    AA = {ACAC_AA[xI, zJ] - corr[xI0A, zJ0A]:+12.10f} ,   BB = {ACAC_BB[xI, zJ] - corr[xI0B, zJ0B]:+12.10f} ,   BA = {ACAC_BA[xI, zJ] - corr[xI0B, zJ0A]:+12.10f} ')
                    zJ += 1
            xI += 1    
            
    ########################
    ## Act-Act, Act-Core  ##
    ########################             
    xy = 0
    for x in range(nact):
        for y in range(x):
            x0 = x + ncore
            y0 = y + ncore
            xy0A = (x0-1)*x0//2 + y0
            xy0B = xy0A + nott
            zJ = 0
            for z in range(nact):
                for J in range(ncore):
                    z0 = z + ncore
                    J0 = J
                    zJ0A = (z0-1)*z0//2 + J0
                    zJ0B = zJ0A + nott
                    Hess[xy0A, zJ0A] = Hess[zJ0A, xy0A] = AAAC_AA[xy, zJ]
                    Hess[xy0B, zJ0B] = Hess[zJ0B, xy0B] = AAAC_BB[xy, zJ]
                    Hess[xy0B, zJ0A] = Hess[zJ0A, xy0B] = AAAC_BA[xy, zJ]
                    Hess[xy0A, zJ0B] = Hess[zJ0B, xy0A] = AAAC_AB[xy, zJ]

#                    if abs(AAAC_AA[xy, zJ] - corr[xy0A, zJ0A]) + abs(AAAC_BB[xy, zJ] - corr[xy0B, zJ0B]) + abs(AAAC_BA[xy, zJ] - corr[xy0B, zJ0A]) + abs(AAAC_AB[xy, zJ] - corr[xy0A, zJ0B]) > 1e-6:
#                        print(f'AAAC {x0} {y0} {z0} {J0}     AA = {AAAC_AA[xy, zJ] - corr[xy0A, zJ0A]:+12.10f} ,   BB = {AAAC_BB[xy, zJ] - corr[xy0B, zJ0B]:+12.10f} ,   BA = {AAAC_BA[xy, zJ] - corr[xy0B, zJ0A]:+12.10f},   AB = {AAAC_AB[xy, zJ] - corr[xy0A, zJ0B]:+12.10f}')
                    zJ += 1
            xy += 1    

    #########################
    ## Vir-Core, Act-Core  ##
    #########################               
    AI = 0
    for A in range(nsec):
        for I in range(ncore):
            A0 = A + ncore + nact
            I0 = I 
            AI0A = (A0-1)*A0//2 + I0
            AI0B = AI0A + nott
            zJ = 0
            for z in range(nact):
                for J in range(ncore):
                    z0 = z + ncore
                    J0 = J
                    zJ0A = (z0-1)*z0//2 + J0
                    zJ0B = zJ0A + nott
                    Hess[AI0A, zJ0A] = Hess[zJ0A, AI0A] = VCAC_AA[AI, zJ]
                    Hess[AI0B, zJ0B] = Hess[zJ0B, AI0B] = VCAC_BB[AI, zJ]
                    Hess[AI0B, zJ0A] = Hess[zJ0A, AI0B] = VCAC_BA[AI, zJ]
                    Hess[AI0A, zJ0B] = Hess[zJ0B, AI0A] = VCAC_AB[AI, zJ]

#                    if abs(VCAC_AA[AI, zJ] - corr[AI0A, zJ0A]) + abs(VCAC_BB[AI, zJ] - corr[AI0B, zJ0B]) + abs(VCAC_BA[AI, zJ] - corr[AI0B, zJ0A]) + abs(VCAC_AB[AI, zJ] - corr[AI0A, zJ0B]) > 1e-6:
#                        print(f'VCAC {A0} {I0} {z0} {J0}     AA = {VCAC_AA[AI, zJ] - corr[AI0A, zJ0A]:+12.10f} ,   BB = {VCAC_BB[AI, zJ] - corr[AI0B, zJ0B]:+12.10f} ,   BA = {VCAC_BA[AI, zJ] - corr[AI0B, zJ0A]:+12.10f},   AB = {VCAC_AB[AI, zJ] - corr[AI0A, zJ0B]:+12.10f}')                    
                    zJ += 1
            AI += 1

    #########################
    ## Vir-Core, Act-Act   ##
    #########################
    AI = 0
    for A in range(nsec):
        for I in range(ncore):
            A0 = A + ncore + nact
            I0 = I 
            AI0A = (A0-1)*A0//2 + I0
            AI0B = AI0A + nott
            zw = 0
            for z in range(nact):
                for w in range(z):
                    z0 = z+ncore
                    w0 = w+ncore
                    zw0A = z0*(z0-1)//2 + w0  
                    zw0B = zw0A + nott
                    Hess[AI0A, zw0A] = Hess[zw0A, AI0A] = VCAA_AA[AI, zw]
                    Hess[AI0B, zw0B] = Hess[zw0B, AI0B] = VCAA_BB[AI, zw]
                    Hess[AI0B, zw0A] = Hess[zw0A, AI0B] = VCAA_BA[AI, zw]
                    Hess[AI0A, zw0B] = Hess[zw0B, AI0A] = VCAA_AB[AI, zw]
                    
#                    if abs(VCAA_AA[AI, zw]-corr[AI0A, zw0A]) + abs(VCAA_BB[AI, zw] - corr[AI0A, zw0A]) + abs(VCAA_BA[AI, zw] - corr[AI0B, zw0A]) + abs(VCAA_AB[AI, zw] - corr[AI0A, zw0B])> 1e-6:
#                        print(f'VCAA {A0} {I0} {z0} {w0}   AA = {VCAA_AA[AI, zw]-corr[AI0A, zw0A]:+12.10f} ,   BB = {VCAA_BB[AI, zw] - corr[AI0A, zw0A]:+12.10f} ,   BA = {VCAA_BA[AI, zw] - corr[AI0B, zw0A]:+12.10f} ,   AB = {VCAA_AB[AI, zw] - corr[AI0A, zw0B]:+12.10f}  ')    
                    zw += 1
            AI += 1
            
    #########################    
    ## Vir-Core, Vir-Core  ##
    #########################
    AI = 0
    for A in range(nsec):
        for I in range(ncore):
            A0 = A + ncore + nact
            I0 = I 
            AI0A = (A0-1)*A0//2 + I0
            AI0B = AI0A + nott
            BJ = 0
            for B in range(nsec):
                for J in range(ncore):
                    B0 = B + ncore + nact
                    J0 = J 
                    BJ0A = (B0-1)*B0//2 + J0
                    BJ0B = BJ0A + nott
                    Hess[AI0A, BJ0A] = VCVC_AA[AI, BJ]
                    Hess[AI0B, BJ0B] = VCVC_BB[AI, BJ]
                    Hess[AI0B, BJ0A] = VCVC_BA[AI, BJ]
                    Hess[AI0A, BJ0B] = VCVC_BA[BJ, AI]
                    
#                    if abs(VCVC_AA[AI, BJ]-corr[AI0A, BJ0A]) + abs(VCVC_BB[AI, BJ] - corr[AI0A, BJ0A]) + abs(VCVC_BA[AI, BJ] - corr[AI0B, BJ0A])  > 1e-6:
#                        print(f'VCVC {A0} {I0} {B0} {J0}   AA = {VCVC_AA[AI, BJ]-corr[AI0A, BJ0A]:+12.10f} ,   BB = {VCVC_BB[AI, BJ] - corr[AI0A, BJ0A]:+12.10f} ,   BA = {VCVC_BA[AI, BJ] - corr[AI0B, BJ0A]:+12.10f}')    
                    
                    BJ += 1
            AI += 1

    ########################
    ## Vir-Act, Act-Core  ##
    ######################## 
    Ay = 0
    for A in range(nsec):
        for y in range(nact):
            A0 = A + ncore + nact
            y0 = y + ncore
            Ay0A = A0 * (A0-1)//2 + y0
            Ay0B = Ay0A + nott
            zJ = 0
            for z in range(nact):
                for J in range(ncore):
                    z0 = z + ncore
                    J0 = J
                    zJ0A = (z0-1)*z0//2 + J0
                    zJ0B = zJ0A + nott
                    Hess[Ay0A, zJ0A] = Hess[zJ0A, Ay0A] = VAAC_AA[Ay, zJ]
                    Hess[Ay0B, zJ0B] = Hess[zJ0B, Ay0B] = VAAC_BB[Ay, zJ]
                    Hess[Ay0B, zJ0A] = Hess[zJ0A, Ay0B] = VAAC_BA[Ay, zJ]
                    Hess[Ay0A, zJ0B] = Hess[zJ0B, Ay0A] = VAAC_AB[Ay, zJ]

#                    if abs(VAAC_AA[Ay, zJ] - corr[Ay0A, zJ0A]) + abs(VAAC_BB[Ay, zJ] - corr[Ay0B, zJ0B]) + abs(VAAC_BA[Ay, zJ] - corr[Ay0B, zJ0A]) + abs(VAAC_AB[Ay, zJ] - corr[Ay0A, zJ0B]) > 1e-6:
#                        print(f'VAAC {A0} {y0} {z0} {J0}     AA = {VAAC_AA[Ay, zJ] - corr[Ay0A, zJ0A]:+12.10f} ,   BB = {VAAC_BB[Ay, zJ] - corr[Ay0B, zJ0B]:+12.10f} ,   BA = {VAAC_BA[Ay, zJ] - corr[Ay0B, zJ0A]:+12.10f},   AB = {VAAC_AB[Ay, zJ] - corr[Ay0A, zJ0B]:+12.10f}')                    
                    zJ += 1
            Ay += 1


    ########################
    ## Vir-Act, Act-Act   ##
    ######################## 
    Ay = 0
    for A in range(nsec):
        for y in range(nact):
            A0 = A + ncore + nact
            y0 = y + ncore
            Ay0A = A0 * (A0-1)//2 + y0
            Ay0B = Ay0A + nott   
            zw = 0
            for z in range(nact):
                for w in range(z):
                    z0 = z+ncore
                    w0 = w+ncore
                    zw0A = z0*(z0-1)//2 + w0
                    zw0B = zw0A + nott
                    Hess[Ay0A, zw0A] = Hess[zw0A, Ay0A] = VAAA_AA[Ay, zw]
                    Hess[Ay0B, zw0B] = Hess[zw0B, Ay0B] = VAAA_BB[Ay, zw]
                    Hess[Ay0B, zw0A] = Hess[zw0A, Ay0B] = VAAA_BA[Ay, zw]
                    Hess[Ay0A, zw0B] = Hess[zw0B, Ay0A] = VAAA_AB[Ay, zw]
                    
#                    if abs(VAAA_AA[Ay, zw]-corr[Ay0A, zw0A]) + abs(VAAA_BB[Ay, zw] - corr[Ay0A, zw0A]) + abs(VAAA_BA[Ay, zw] - corr[Ay0B, zw0A]) + abs(VAAA_AB[Ay, zw] - corr[Ay0A, zw0B])> 1e-6:
#                        print(f'VAAA {A0} {y0} {z0} {w0}   AA = {VAAA_AA[Ay, zw]-corr[Ay0A, zw0A]:+12.10f} ,   BB = {VAAA_BB[Ay, zw] - corr[Ay0A, zw0A]:+12.10f} ,   BA = {VAAA_BA[Ay, zw] - corr[Ay0B, zw0A]:+12.10f} ,   AB = {VAAA_AB[Ay, zw] - corr[Ay0A, zw0B]:+12.10f}  ')    
                    zw += 1
            Ay += 1
            
    ########################
    ## Vir-Act, Vir-Core  ##
    ########################
    Ay = 0
    for A in range(nsec):
        for y in range(nact):
            A0 = A + ncore + nact
            y0 = y + ncore
            Ay0A = A0 * (A0-1)//2 + y0
            Ay0B = Ay0A + nott   
            BJ = 0
            for B in range(nsec):
                for J in range(ncore):
                    B0 = B + ncore + nact
                    J0 = J 
                    BJ0A = (B0-1)*B0//2 + J0
                    BJ0B = BJ0A + nott
                    Hess[Ay0A, BJ0A] = Hess[BJ0A, Ay0A] = VAVC_AA[Ay, BJ]
                    Hess[Ay0B, BJ0B] = Hess[BJ0B, Ay0B] = VAVC_BB[Ay, BJ]
                    Hess[Ay0B, BJ0A] = Hess[BJ0A, Ay0B] = VAVC_BA[Ay, BJ]
                    Hess[Ay0A, BJ0B] = Hess[BJ0B, Ay0A] = VAVC_AB[Ay, BJ]
#                    if abs(VAVC_AA[Ay, BJ]-corr[Ay0A, BJ0A]) + abs(VAVC_BB[Ay, BJ] - corr[Ay0A, BJ0A]) + abs(VAVC_BA[Ay, BJ] - corr[Ay0B, BJ0A]) + abs(VAVC_AB[Ay, BJ] - corr[Ay0A, BJ0B])> 1e-6:
#                        print(f'VAVC {A0} {y0} {B0} {J0}   AA = {VAVC_AA[Ay, BJ]-corr[Ay0A, BJ0A]:+12.10f} ,   BB = {VAVC_BB[Ay, BJ] - corr[Ay0A, BJ0A]:+12.10f} ,   BA = {VAVC_BA[Ay, BJ] - corr[Ay0B, BJ0A]:+12.10f} ,   AB = {VAVC_AB[Ay, BJ] - corr[Ay0A, BJ0B]:+12.10f}  ')    
                    BJ += 1
            Ay += 1

    ########################
    ## Vir-Act, Vir-Act   ##
    ########################
    Ay = 0
    for A in range(nsec):
        for y in range(nact):
            A0 = A + ncore + nact
            y0 = y + ncore
            Ay0A = A0 * (A0-1)//2 + y0
            Ay0B = Ay0A + nott            
            Bw = 0
            for B in range(nsec):
                for w in range(nact):
                    B0 = B+ncore+nact
                    w0 = w+ncore
                    Bw0A = B0*(B0-1)//2 + w0  
                    Bw0B = Bw0A + nott
                    Hess[Ay0A, Bw0A] = VAVA_AA[Ay, Bw]
                    Hess[Ay0B, Bw0B] = VAVA_BB[Ay, Bw]
                    Hess[Ay0B, Bw0A] = VAVA_BA[Ay, Bw]
                    Hess[Ay0A, Bw0B] = VAVA_BA[Bw, Ay]
                    
#                    if abs(VAVA_AA[Ay, Bw]-corr[Ay0A, Bw0A]) + abs(VAVA_BB[Ay, Bw] - corr[Ay0A, Bw0A]) + abs(VAVA_BA[Ay, Bw] - corr[Ay0B, Bw0A])>1e-6:    
#                        print(f'VAVA {A0} {y0} {B0} {w0}   AA = {VAVA_AA[Ay, Bw]-corr[Ay0A, Bw0A]:+12.10f} ,   BB = {VAVA_BB[Ay, Bw] - corr[Ay0A, Bw0A]:+12.10f} ,   BA = {VAVA_BA[Ay, Bw] - corr[Ay0B, Bw0A]:+12.10f}')    
                    Bw += 1
            Ay += 1

    return 2*Hess

