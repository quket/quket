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
from quket import config as cf
from quket.fileio import prints


def get_ndims(Quket):
    if Quket.model not in ("chemical", "hubbard"):
        return 0, 0, 0
    ansatz = Quket.ansatz
    from_vir = Quket.from_vir

    ######################
    # Arrange parameters #
    ######################
    if ansatz in ("ic_mrucc", "ic_mrucc_spinfree"):
        arrange_params_ic_mrucc(Quket)
    elif ansatz in (None, "user-defined"):
        return 0, 0, 0

    nca = ncb = Quket.nc
    if not from_vir:
        noa = Quket.noa
        nob = Quket.nob
        nva = Quket.nva
        nvb = Quket.nvb
    else:
        _noa = Quket.noa
        _nob = Quket.nob
        _nva = Quket.nva
        _nvb = Quket.nvb
        noa = _noa + _nva
        nob = _nob + _nvb
        nva = _noa + _nva
        nvb = _nob + _nvb
    nsa = nsb = Quket.ns
    norbs = Quket.n_active_orbitals
    include = Quket.include

    #################
    # Default ndims #
    #################
    #ndim1 = noa*nva + nob*nvb
    #ndim2aa = noa*(noa-1)*nva*(nva-1)//4
    #ndim2ab = noa*nob*nva*nvb
    #ndim2bb = nob*(nob-1)*nvb*(nvb-1)//4
    #ndim2 = ndim2aa + ndim2ab + ndim2bb
    #ndim = ndim1 + ndim2
    ndim1 = ndim1a = ndim1b = 0
    ndim2 = ndim2aa = ndim2ab = ndim2ba = ndim2bb = 0
    ndim = 0
    for from_, to in include.items():
        if from_ == "c":
            if "a" in to:
                ndim1a += nca*nva
                ndim1b += ncb*nvb
            if "s" in to:
                ndim1a += nca*nsa
                ndim1b += ncb*nsb
        elif from_ == "a":
            if "a" in to:
                ndim1a += noa*nva
                ndim1b += nob*nvb
            if "s" in to:
                ndim1a += noa*nsa
                ndim1b += nob*nsb
        elif from_ == "cc":
            if "aa" in to:
                ndim2aa += nca*(nca-1)*nva*(nva-1)//4
                ndim2ab += nca*ncb*nva*nvb
                ndim2ba += 0
                ndim2bb += ncb*(ncb-1)*nvb*(nvb-1)//4
            if "as" in to:
                ndim2aa += nca*(nca-1)*nva*nsa//2
                ndim2ab += nca*ncb*nva*nsb
                ndim2ba += ncb*nca*nvb*nsa
                ndim2bb += ncb*(ncb-1)*nvb*nsb//2
            if "ss" in to:
                ndim2aa += nca*(nca-1)*nsa*(nsa-1)//4
                ndim2ab += nca*ncb*nsa*nsb
                ndim2ba += 0
                ndim2bb += ncb*(ncb-1)*nsb*(nsb-1)//4
        elif from_ == "ca":
            if "aa" in to:
                ndim2aa += nca*noa*nva*(nva-1)//2
                ndim2ab += nca*nob*nva*nvb
                ndim2ba += ncb*noa*nvb*nva
                ndim2bb += ncb*nob*nvb*(nvb-1)//2
            if "as" in to:
                ndim2aa += nca*noa*nva*nsa
                ndim2ab += nca*nob*(nva*nsb + nsa*nvb)
                ndim2ba += ncb*noa*(nvb*nsa + nsb*nva)
                ndim2bb += ncb*nob*nvb*nsb
            if "ss" in to:
                ndim2aa += nca*noa*nsa*(nsa-1)//2
                ndim2ab += nca*nob*nsa*nsb
                ndim2ba += ncb*noa*nsb*nsa
                ndim2bb += ncb*nob*nsb*(nsb-1)//2
        elif from_ == "aa":
            if "aa" in to:
                ndim2aa += noa*(noa-1)*nva*(nva-1)//4
                ndim2ab += noa*nob*nva*nvb
                ndim2ba += 0
                ndim2bb += nob*(nob-1)*nvb*(nvb-1)//4
            if "as" in to:
                ndim2aa += noa*(noa-1)*nva*nsa//2
                ndim2ab += noa*nob*nva*nsb
                ndim2ba += nob*noa*nvb*nsa
                ndim2bb += nob*(nob-1)*nvb*nsb//2
            if "ss" in to:
                ndim2aa += noa*(noa-1)*nsa*(nsa-1)//4
                ndim2ab += noa*nob*nsa*nsb
                ndim2ba += 0
                ndim2bb += nob*(nob-1)*nsb*(nsb-1)//4
    ndim1 = ndim1a + ndim1b
    ndim2 = ndim2aa + ndim2ab + ndim2ba + ndim2bb

    #################
    # Arrange ndims #
    #################
    if ansatz in ("uhf", "phf", "suhf"):
        ndim = ndim1
        ndim2 = 0
    elif ansatz in ("uccd", "puccd"):
        ndim1 = 0
        ndim = ndim2
    elif ansatz in ("uccsd", "opt_puccd", "puccsd"):
        ndim = ndim1 + ndim2
    elif ansatz in ("uccgd", "uccgsd"):
        # TODO: This is ugly and we need to get rid of ndim.py in future
        from .utils import Gdoubles_list
        r_list, u_list, parity_list = Gdoubles_list(norbs)
        ndim2 = 0
        for ilist in range(len(u_list)):
            for b, a, j, i in u_list[ilist]:
                ndim2 += 1
        if ansatz == "uccgsd":
            ndim1 = norbs*(norbs-1)
        else:
            ndim1 = 0
        ndim = ndim1 + ndim2
    elif ansatz == "sghf":
        ndim1 = (noa+nob)*(nva+nvb)
        ndim2 = 0
        ndim = ndim1
    elif ansatz == "opt_psauccd":
        ndim2 = noa*nva*(noa*nva + 1)//2
        ndim = ndim1 + ndim2
    elif ansatz == "sauccsd":
        ndim1 = noa*nva
        ndim2 = ndim1*(ndim1+1)//2
        ndim = ndim1 + ndim2
    elif "bcs" in ansatz:
        if "ebcs" in ansatz:
            k_param = ansatz[0:nsatz.find("-ebcs")]
        else:
            k_param = ansatz[0:ansatz.find("-bcs")]
        if not k_param.isdecimal():
            prints(f"Unrecognized k: {k_param}")
            error("k-BCS without specifying k.")
        k_param = int(k_param)
        if k_param < 1:
            error("0-bcs is just HF!")
        ndim1 = norbs*(norbs-1)//2
        ndim2 = norbs
        ndim = k_param*(ndim1+ndim2)
    elif "pccgsd" in ansatz:
        if "upccgsd" in ansatz:
            k_param = ansatz[0:ansatz.find("-upccgsd")]
        elif "epccgsd" in ansatz:
            k_param = ansatz[0:ansatz.find("-epccgsd")]
        if not k_param.isdecimal():
            prints(f"Unrecognized k: {k_param}")
            error("k-UpCCGSD without specifying k.")
        k_param = int(k_param)
        if k_param < 1:
            error("0-upccgsd is just HF!")
        ndim1 = norbs*(norbs-1)//2
        ndim2 = norbs*(norbs-1)//2
        ndim = k_param*(ndim1+ndim2)
        if "epccgsd" in ansatz:
            ndim += ndim1
    elif ansatz == "jmucc":
        if Quket.multi.nstates == 0:
            error("JM-UCC specified without state specification!")
        ndim = Quket.multi.nstates*(ndim1+ndim2)
    elif ansatz == "ic_mrucc":
        # Note; c=core, a=act, v=vir, A=Alpha, B=Beta
        #        cA->aA  + cA->vA  +     aA->aA     + aA->vA
        ndim1 = (nca*noa + nca*nva + noa*(noa-1)//2 + noa*nva
        #        cB->aB  + cB->vB  +     aB->aB     + aB->vB
               + ncb*nob + ncb*nvb + nob*(nob-1)//2 + nob*nvb)
        if Quket.multi.act2act_opt:
            # Include aa -> aa excitation.
            #                cAcA      +  cAaA   +      aAaA
            ndim2aa = ((nca*(nca-1)//2 + nca*noa + noa*(noa-1)//2)
            #       ->       aAaA      +  aAvA   +      vAvA
                      *(noa*(noa-1)//2 + noa*nva + nva*(nva-1)//2)
            # Get rid of duplicate counts; aAaA -> aAaA.
                      - noa*(noa-1)//2)
            #            cAcB   +  cAaB   +  aAcB   +  aAaB
            ndim2ab = ((nca*ncb + nca*nob + noa*ncb + noa*nob)
            #       ->   aAaB   +  aAvB   +  vAaB   +  vAvB
                      *(noa*nob + noa*nvb + nva*nob + nva*nvb)
            # Get rid of duplicate counts; aAaB -> aAaB.
                      - noa*nob)
            #                cBcB      +  cBaB   +      aBaB
            ndim2bb = ((ncb*(ncb-1)//2 + ncb*nob + nob*(nob-1)//2)
            #       ->       aBaB      +  aBvB   +      vBvB
                      *(nob*(nob-1)//2 + nob*nvb + nvb*(nvb-1)//2)
            # Get rid of duplicate counts; aBaB -> aBaB.
                      - nob*(nob-1)//2)
        else:
            # Exclude aa -> aa excitation.
            #                cAcA      +  cAaA   +      aAaA
            ndim2aa = ((nca*(nca-1)//2 + nca*noa + noa*(noa-1)//2)
            #       ->       aAaA      +  aAvA   +      vAvA
                      *(noa*(noa-1)//2 + noa*nva + nva*(nva-1)//2)
            # Get rid of counts completely; aAaA -> aAaA.
                      - noa*(noa-1)*noa*(noa-1)//4)
            #            cAcB   +  cAaB   +  aAcB   +  aAaB
            ndim2ab = ((nca*ncb + nca*nob + noa*ncb + noa*nob)
            #       ->   aAaB   +  aAvB   +  vAaB   +  vAvB
                      *(noa*nob + noa*nvb + nva*nob + nva*nvb)
            # Get rid of counts completely; aAaB -> aAaB.
                      - noa*nob*noa*nob)
            #                cBcB      +  cBaB   +      aBaB
            ndim2bb = ((ncb*(ncb-1)//2 + ncb*nob + nob*(nob-1)//2)
            #       ->       aBaB      +  aBvB   +      vBvB
                      *(nob*(nob-1)//2 + nob*nvb + nvb*(nvb-1)//2)
            # Get rid of counts completely; aBaB -> aBaB.
                      - nob*(nob-1)*nob*(nob-1)//4)
        ndim2 = ndim2aa + ndim2ab + ndim2bb
        ndim = ndim1 + ndim2

        if cf.debug:
            prints(f"vir: {nva}, act: {noa}, core: {nca}")
            prints(f"act to act: {Quket.multi.act2act_opt}  "
                   f"[ndim1: {ndim1}, ndim2: {ndim2}]")
    elif ansatz == "ic_mrucc_spinfree":
        #        c->a   +  c->v   +      a->a      +  a->v
        ndim1 = nca*noa + nca*nva + noa*(noa-1)//2 + noa*nva
        #                   ca   ->   av
        excite_pattern = (nca+noa)*(noa+nva)
        # Total number of excitation pattern.
        ndim2 = excite_pattern*(excite_pattern+1)//2
        if Quket.multi.act2act_opt:
            # Get rid of duplicate counts; aa -> aa.
            ndim2 -= noa*(noa+1)//2
        else:
            # Get rid of counts completely; aa -> aa.
            ndim2 -= (noa*noa)*(noa*noa + 1)//2
        ndim = ndim1 + ndim2

        if cf.debug:
            prints(f"vir: {nva, nvb}, act: {noa, nob}")
            prints(f"act to act: {Quket.multi.act2act_opt}  "
                   f"[ndim1: {ndim1}, ndim2: {ndim2}]")

    if cf.debug:
        prints(f"secondary: {nsa}, vir: {nva, nvb}, "
               f"act: {noa, nob}, core: {nca}")
        prints(f"ndim1: {ndim1}, ndim2: {ndim2}")

    return ndim1, ndim2, ndim

def arrange_params_ic_mrucc(Quket):
    states = Quket.multi.init_states_info[:]
    nstates = Quket.multi.nstates
    n_qubits = Quket.n_qubits

    opt = f"0{n_qubits}b"
    nc = nv = n_qubits
    for state_ in states:
        for state in state_:
            # Note; state must be integer.
            statestr = format(state[1], opt)
            core_id = statestr.rfind("0")
            vir_id = statestr.find("1")
            nc = min(nc, n_qubits-core_id-1)
            nv = min(nv, vir_id)
    nc //= 2
    nv //= 2
    na = n_qubits//2 - nc - nv

    Quket.nc = nc
    Quket.noa = Quket.nob = na
    Quket.nva = Quket.nvb = nv
