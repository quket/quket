from math import fsum
from itertools import product, groupby, chain
from operator import itemgetter

from numpy import einsum, array, zeros, isclose, outer, where

from quket.fileio.fileio import prints


# Frozen || Core | Active || Secondary
# nf || nc | na || ns
#
# integrals nf || nc | na || ns
# integrals_active || | na ||
# H_tlide nf || nc | na || ns
# 1234RDM || | na ||
# excitation_list || nc | na || ns


#### This is the main driver that call subroutines ###########################


def do_spinorbital(Quket, HD, y_excitation_list, x_excitation_list):

    from quket.post.lucc_auxiliary.interface import \
            sort_by_space_spin, \
            fullspin_to_spaceorbital, \
            shift_index_by_n, \
            get_bound_for_active_space

    #=== Setting some constants ===============================================#
    # No need to multiply by 2
    nCor = Quket.n_core_orbitals
    nAct = Quket.n_active_orbitals
    ActFloor, ActCeil = get_bound_for_active_space(Quket)

    #=== Prepare a matrix to contain einsum results ===========================#
    y_excitation_list = tuple( tuple(x) for x in y_excitation_list)
    x_excitation_list = tuple( tuple(x) for x in x_excitation_list)
    Amat_dimension = len(y_excitation_list), len(x_excitation_list)
    Amat = zeros(Amat_dimension)

    #=== Analysising the excitation_list ======================================#
    # sort and transpose
    tv_excitation_list, \
    tv_spacetypes, \
    tv_spintypes, \
    tv_paritys = zip(*(
        sort_by_space_spin(2*nCor, 2*(nCor+nAct), e)
        for e in y_excitation_list) )

    pr_excitation_list, \
    pr_spacetypes, \
    pr_spintypes, \
    pr_paritys = zip(*(
        sort_by_space_spin(2*nCor, 2*(nCor+nAct), e)
        for e in x_excitation_list) )

    # Compress index from space-orbital to spin-orbital
    tv_excitation_list = fullspin_to_spaceorbital(tv_excitation_list)
    pr_excitation_list = fullspin_to_spaceorbital(pr_excitation_list)

    #=== Do Einsums ===========================================================#

    #--- [H, tv]
    for i, (tv_excite, tv_spin, p1) in enumerate(zip(tv_excitation_list, tv_spintypes, tv_paritys)):

        #--- [ [H, pr] , tv]
        for j, (pr_excite, pr_spin, p2) in enumerate(zip(pr_excitation_list, pr_spintypes, pr_paritys)):

            #--- Compute the actually value of double commutator
            spintype = (pr_spin, tv_spin)
            func_key = '_'.join(spintype)

            if func_key in FUNCMAPPER:
                e = (pr_excite, tv_excite)

                Amat[i,j] = -FUNCMAPPER[func_key](HD, *e) \
                            if p1^p2 else \
                            FUNCMAPPER[func_key](HD, *e)

    return 0.5 * Amat


#### Subroutinues are under this line ########################################

def lucc_aa(HD, *e):
    (p,r) = e
    result = 0
    result += fsum(( -HD['Haaaa123Daaaa123'][p,r] ,
                     +HD['Haaaa123Daaaa123'][r,p] ))
    result += 2 * fsum(( +HD['Haa1Daa1'][p,r] ,
                         -HD['Haa1Daa1'][r,p] ,
                         -HD['Habab123Dabab123'][p,r] ,
                         +HD['Habab123Dabab123'][r,p] ))
    return result


def lucc_aa_aa(HD, *e):
    (p,r),(t,v) = e
    result = 0
    result += fsum(( -HD['Haaaa23Daaaa23'][r,v,p,t] ,
                     +HD['Haaaa23Daaaa23'][r,t,p,v] ,
                     +HD['Haaaa23Daaaa23'][p,v,r,t] ,
                     -HD['Haaaa23Daaaa23'][p,t,r,v] ))
    result += 2 * fsum(( +HD['Haa'][p,t] * HD['Daa'][r,v] ,
                         -HD['Haa'][p,v] * HD['Daa'][r,t] ,
                         -HD['Haa'][r,t] * HD['Daa'][p,v] ,
                         +HD['Haa'][r,v] * HD['Daa'][p,t] ,
                         -HD['Haaaa13Daaaa13'][p,t,r,v] ,
                         +HD['Haaaa13Daaaa13'][p,v,r,t] ,
                         +HD['Haaaa13Daaaa13'][r,t,p,v] ,
                         -HD['Haaaa13Daaaa13'][r,v,p,t] ,
                         -HD['Habab13Dabab13'][p,t,r,v] ,
                         +HD['Habab13Dabab13'][p,v,r,t] ,
                         +HD['Habab13Dabab13'][r,t,p,v] ,
                         -HD['Habab13Dabab13'][r,v,p,t] ))
    if r==v:
        result += +HD['Haaaa123Daaaa123'][p,t]
        result += 2 * fsum(( -HD['Haa1Daa1'][p,t] ,
                             +HD['Habab123Dabab123'][p,t] ))
    if r==t:
        result += -HD['Haaaa123Daaaa123'][p,v]
        result += 2 * fsum(( +HD['Haa1Daa1'][p,v] ,
                             -HD['Habab123Dabab123'][p,v] ))
    if p==v:
        result += -HD['Haaaa123Daaaa123'][r,t]
        result += 2 * fsum(( +HD['Haa1Daa1'][r,t] ,
                             -HD['Habab123Dabab123'][r,t] ))
    if p==t:
        result += +HD['Haaaa123Daaaa123'][r,v]
        result += 2 * fsum(( -HD['Haa1Daa1'][r,v] ,
                             +HD['Habab123Dabab123'][r,v] ))
    return result


def lucc_aa_bb(HD, *e):
    (p,r),(t,v) = e
    result = 0
    result += 2 * fsum(( -HD['Habba13Dabba13'][p,t,r,v] ,
                         +HD['Habba13Dabba13'][p,v,r,t] ,
                         +HD['Habba13Dabba13'][r,t,p,v] ,
                         -HD['Habba13Dabba13'][r,v,p,t] ,
                         +HD['Hbaab23Dbaab23'][v,p,t,r] ,
                         -HD['Hbaab23Dbaab23'][t,p,v,r] ,
                         -HD['Hbaab23Dbaab23'][v,r,t,p] ,
                         +HD['Hbaab23Dbaab23'][t,r,v,p] ))
    return result


def lucc_aa_aaaa(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += fsum(( +HD['Haaaa23Daaaaaa45'][p,w,r,t,u,v] ,
                     -HD['Haaaa23Daaaaaa45'][p,v,r,t,u,w] ,
                     +HD['Haaaa23Daaaaaa45'][p,t,r,v,w,u] ,
                     -HD['Haaaa23Daaaaaa45'][p,u,r,v,w,t] ,
                     -HD['Haaaa23Daaaaaa45'][r,w,p,t,u,v] ,
                     +HD['Haaaa23Daaaaaa45'][r,v,p,t,u,w] ,
                     -HD['Haaaa23Daaaaaa45'][r,t,p,v,w,u] ,
                     +HD['Haaaa23Daaaaaa45'][r,u,p,v,w,t] ))
    result += 2 * fsum(( +HD['Haa'][p,t] * HD['Daaaa'][r,u,v,w] ,
                         -HD['Haa'][p,u] * HD['Daaaa'][r,t,v,w] ,
                         +HD['Haa'][p,w] * HD['Daaaa'][r,v,t,u] ,
                         -HD['Haa'][p,v] * HD['Daaaa'][r,w,t,u] ,
                         -HD['Haa'][r,t] * HD['Daaaa'][p,u,v,w] ,
                         +HD['Haa'][r,u] * HD['Daaaa'][p,t,v,w] ,
                         -HD['Haa'][r,w] * HD['Daaaa'][p,v,t,u] ,
                         +HD['Haa'][r,v] * HD['Daaaa'][p,w,t,u] ,
                         +HD['Haaaa13Daaaaaa25'][p,t,r,u,v,w] ,
                         -HD['Haaaa13Daaaaaa25'][p,u,r,t,v,w] ,
                         +HD['Haaaa13Daaaaaa25'][p,w,r,v,t,u] ,
                         -HD['Haaaa13Daaaaaa25'][p,v,r,w,t,u] ,
                         -HD['Haaaa13Daaaaaa25'][r,t,p,u,v,w] ,
                         +HD['Haaaa13Daaaaaa25'][r,u,p,t,v,w] ,
                         -HD['Haaaa13Daaaaaa25'][r,w,p,v,t,u] ,
                         +HD['Haaaa13Daaaaaa25'][r,v,p,w,t,u] ,
                         +HD['Haaaa3Daaaa3'][t,u,p,v,w,r] ,
                         +HD['Haaaa3Daaaa3'][v,w,r,t,u,p] ,
                         -HD['Haaaa3Daaaa3'][v,w,p,t,u,r] ,
                         -HD['Haaaa3Daaaa3'][t,u,r,v,w,p] ,
                         +HD['Habab13Daabaab25'][p,t,r,u,v,w] ,
                         -HD['Habab13Daabaab25'][p,u,r,t,v,w] ,
                         +HD['Habab13Daabaab25'][p,w,r,v,t,u] ,
                         -HD['Habab13Daabaab25'][p,v,r,w,t,u] ,
                         -HD['Habab13Daabaab25'][r,t,p,u,v,w] ,
                         +HD['Habab13Daabaab25'][r,u,p,t,v,w] ,
                         -HD['Habab13Daabaab25'][r,w,p,v,t,u] ,
                         +HD['Habab13Daabaab25'][r,v,p,w,t,u] ))
    if r==w:
        result += fsum(( -HD['Haaaa123Daaaaaa245'][p,t,u,v] ,
                         +HD['Haaaa23Daaaa23'][p,v,t,u] ))
        result += 2 * fsum(( -HD['Haa1Daaaa3'][p,t,u,v] ,
                             -HD['Habab123Daabaab245'][p,t,u,v] ))
    if r==v:
        result += fsum(( +HD['Haaaa123Daaaaaa245'][p,t,u,w] ,
                         -HD['Haaaa23Daaaa23'][p,w,t,u] ))
        result += 2 * fsum(( +HD['Haa1Daaaa3'][p,t,u,w] ,
                             +HD['Habab123Daabaab245'][p,t,u,w] ))
    if r==t:
        result += fsum(( -HD['Haaaa123Daaaaaa245'][p,v,w,u] ,
                         +HD['Haaaa23Daaaa23'][p,u,v,w] ))
        result += 2 * fsum(( -HD['Haa1Daaaa3'][p,v,w,u] ,
                             -HD['Habab123Daabaab245'][p,v,w,u] ))
    if r==u:
        result += fsum(( +HD['Haaaa123Daaaaaa245'][p,v,w,t] ,
                         -HD['Haaaa23Daaaa23'][p,t,v,w] ))
        result += 2 * fsum(( +HD['Haa1Daaaa3'][p,v,w,t] ,
                             +HD['Habab123Daabaab245'][p,v,w,t] ))
    if p==w:
        result += fsum(( +HD['Haaaa123Daaaaaa245'][r,t,u,v] ,
                         -HD['Haaaa23Daaaa23'][r,v,t,u] ))
        result += 2 * fsum(( +HD['Haa1Daaaa3'][r,t,u,v] ,
                             +HD['Habab123Daabaab245'][r,t,u,v] ))
    if p==v:
        result += fsum(( -HD['Haaaa123Daaaaaa245'][r,t,u,w] ,
                         +HD['Haaaa23Daaaa23'][r,w,t,u] ))
        result += 2 * fsum(( -HD['Haa1Daaaa3'][r,t,u,w] ,
                             -HD['Habab123Daabaab245'][r,t,u,w] ))
    if p==t:
        result += fsum(( +HD['Haaaa123Daaaaaa245'][r,v,w,u] ,
                         -HD['Haaaa23Daaaa23'][r,u,v,w] ))
        result += 2 * fsum(( +HD['Haa1Daaaa3'][r,v,w,u] ,
                             +HD['Habab123Daabaab245'][r,v,w,u] ))
    if p==u:
        result += fsum(( -HD['Haaaa123Daaaaaa245'][r,v,w,t] ,
                         +HD['Haaaa23Daaaa23'][r,t,v,w] ))
        result += 2 * fsum(( -HD['Haa1Daaaa3'][r,v,w,t] ,
                             -HD['Habab123Daabaab245'][r,v,w,t] ))
    return result


def lucc_aa_baba(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += fsum(( +HD['Haaaa23Dbaabaa45'][r,w,t,p,u,v] ,
                     -HD['Haaaa23Dbaabaa45'][r,u,v,p,w,t] ,
                     -HD['Haaaa23Dbaabaa45'][p,w,t,r,u,v] ,
                     +HD['Haaaa23Dbaabaa45'][p,u,v,r,w,t] ))
    result += 2 * fsum(( -HD['Haa'][p,u] * HD['Dbaab'][t,r,w,v] ,
                         +HD['Haa'][p,w] * HD['Dbaab'][v,r,u,t] ,
                         +HD['Haa'][r,u] * HD['Dbaab'][t,p,w,v] ,
                         -HD['Haa'][r,w] * HD['Dbaab'][v,p,u,t] ,
                         -HD['Haaaa13Dbaaaba25'][p,u,t,r,w,v] ,
                         +HD['Haaaa13Dbaaaba25'][p,w,v,r,u,t] ,
                         +HD['Haaaa13Dbaaaba25'][r,u,t,p,w,v] ,
                         -HD['Haaaa13Dbaaaba25'][r,w,v,p,u,t] ,
                         -HD['Habab13Dbababb25'][r,w,v,p,u,t] ,
                         -HD['Habab13Dbababb25'][p,u,t,r,w,v] ,
                         +HD['Habab13Dbababb25'][r,u,t,p,w,v] ,
                         +HD['Habab13Dbababb25'][p,w,v,r,u,t] ,
                         -HD['Habba13Daababa25'][p,t,r,u,w,v] ,
                         -HD['Habba13Daababa25'][r,v,p,w,u,t] ,
                         +HD['Habba13Daababa25'][p,v,r,w,u,t] ,
                         +HD['Habba13Daababa25'][r,t,p,u,w,v] ,
                         +HD['Hbaab23Dbaaaab45'][v,r,t,p,u,w] ,
                         -HD['Hbaab23Dbaaaab45'][v,p,t,r,u,w] ,
                         -HD['Hbaab23Dbaaaab45'][t,r,v,p,w,u] ,
                         +HD['Hbaab23Dbaaaab45'][t,p,v,r,w,u] ,
                         +HD['Hbaab3Dbaab3'][t,u,p,v,w,r] ,
                         -HD['Hbaab3Dbaab3'][t,u,r,v,w,p] ,
                         -HD['Hbaab3Dbaab3'][v,w,p,t,u,r] ,
                         +HD['Hbaab3Dbaab3'][v,w,r,t,u,p] ))
    if r==w:
        result += -HD['Haaaa123Dbaabaa245'][p,t,u,v]
        result += 2 * fsum(( -HD['Haa1Dbaba3'][p,t,u,v] ,
                             -HD['Habab123Dbabbab245'][p,t,u,v] ,
                             -HD['Hbaab23Dbaab23'][v,p,t,u] ))
    if r==u:
        result += +HD['Haaaa123Dbaabaa245'][p,v,w,t]
        result += 2 * fsum(( +HD['Haa1Dbaba3'][p,v,w,t] ,
                             +HD['Habab123Dbabbab245'][p,v,w,t] ,
                             +HD['Hbaab23Dbaab23'][t,p,v,w] ))
    if p==w:
        result += +HD['Haaaa123Dbaabaa245'][r,t,u,v]
        result += 2 * fsum(( +HD['Haa1Dbaba3'][r,t,u,v] ,
                             +HD['Habab123Dbabbab245'][r,t,u,v] ,
                             +HD['Hbaab23Dbaab23'][v,r,t,u] ))
    if p==u:
        result += -HD['Haaaa123Dbaabaa245'][r,v,w,t]
        result += 2 * fsum(( -HD['Haa1Dbaba3'][r,v,w,t] ,
                             -HD['Habab123Dbabbab245'][r,v,w,t] ,
                             -HD['Hbaab23Dbaab23'][t,r,v,w] ))
    return result


def lucc_aa_bbbb(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += 2 * fsum(( -HD['Habba13Dbabbba25'][p,t,u,r,v,w] ,
                         +HD['Habba13Dbabbba25'][p,u,t,r,v,w] ,
                         -HD['Habba13Dbabbba25'][p,w,v,r,t,u] ,
                         +HD['Habba13Dbabbba25'][p,v,w,r,t,u] ,
                         +HD['Habba13Dbabbba25'][r,t,u,p,v,w] ,
                         -HD['Habba13Dbabbba25'][r,u,t,p,v,w] ,
                         +HD['Habba13Dbabbba25'][r,w,v,p,t,u] ,
                         -HD['Habba13Dbabbba25'][r,v,w,p,t,u] ,
                         -HD['Hbaab23Dbbabab45'][w,p,t,u,r,v] ,
                         +HD['Hbaab23Dbbabab45'][v,p,t,u,r,w] ,
                         -HD['Hbaab23Dbbabab45'][t,p,v,w,r,u] ,
                         +HD['Hbaab23Dbbabab45'][u,p,v,w,r,t] ,
                         +HD['Hbaab23Dbbabab45'][w,r,t,u,p,v] ,
                         -HD['Hbaab23Dbbabab45'][v,r,t,u,p,w] ,
                         +HD['Hbaab23Dbbabab45'][t,r,v,w,p,u] ,
                         -HD['Hbaab23Dbbabab45'][u,r,v,w,p,t] ))
    return result


def lucc_bb(HD, *e):
    (p,r) = e
    result = 0
    result += fsum(( -HD['Hbbbb123Dbbbb123'][p,r] ,
                     +HD['Hbbbb123Dbbbb123'][r,p] ))
    result += 2 * fsum(( -HD['Hbaab123Dbaab123'][p,r] ,
                         +HD['Hbaab123Dbaab123'][r,p] ,
                         +HD['Hbb1Dbb1'][p,r] ,
                         -HD['Hbb1Dbb1'][r,p] ))
    return result


def lucc_bb_aa(HD, *e):
    (p,r),(t,v) = e
    result = 0
    result += 2 * fsum(( -HD['Hbaab13Dbaab13'][p,t,r,v] ,
                         +HD['Hbaab13Dbaab13'][p,v,r,t] ,
                         +HD['Hbaab13Dbaab13'][r,t,p,v] ,
                         -HD['Hbaab13Dbaab13'][r,v,p,t] ,
                         +HD['Hbaab23Dbaab23'][p,v,r,t] ,
                         -HD['Hbaab23Dbaab23'][p,t,r,v] ,
                         -HD['Hbaab23Dbaab23'][r,v,p,t] ,
                         +HD['Hbaab23Dbaab23'][r,t,p,v] ))
    return result


def lucc_bb_bb(HD, *e):
    (p,r),(t,v) = e
    result = 0
    result += fsum(( +HD['Hbbbb23Dbbbb23'][p,v,r,t] ,
                     -HD['Hbbbb23Dbbbb23'][p,t,r,v] ,
                     -HD['Hbbbb23Dbbbb23'][r,v,p,t] ,
                     +HD['Hbbbb23Dbbbb23'][r,t,p,v] ))
    result += 2 * fsum(( -HD['Hbaba13Dbaba13'][p,t,r,v] ,
                         +HD['Hbaba13Dbaba13'][p,v,r,t] ,
                         +HD['Hbaba13Dbaba13'][r,t,p,v] ,
                         -HD['Hbaba13Dbaba13'][r,v,p,t] ,
                         +HD['Hbb'][p,t] * HD['Dbb'][r,v] ,
                         -HD['Hbb'][p,v] * HD['Dbb'][r,t] ,
                         -HD['Hbb'][r,t] * HD['Dbb'][p,v] ,
                         +HD['Hbb'][r,v] * HD['Dbb'][p,t] ,
                         -HD['Hbbbb13Dbbbb13'][p,t,r,v] ,
                         +HD['Hbbbb13Dbbbb13'][p,v,r,t] ,
                         +HD['Hbbbb13Dbbbb13'][r,t,p,v] ,
                         -HD['Hbbbb13Dbbbb13'][r,v,p,t] ))
    if r==v:
        result += +HD['Hbbbb123Dbbbb123'][p,t]
        result += 2 * fsum(( +HD['Hbaab123Dbaab123'][p,t] ,
                             -HD['Hbb1Dbb1'][p,t] ))
    if r==t:
        result += -HD['Hbbbb123Dbbbb123'][p,v]
        result += 2 * fsum(( -HD['Hbaab123Dbaab123'][p,v] ,
                             +HD['Hbb1Dbb1'][p,v] ))
    if p==v:
        result += -HD['Hbbbb123Dbbbb123'][r,t]
        result += 2 * fsum(( -HD['Hbaab123Dbaab123'][r,t] ,
                             +HD['Hbb1Dbb1'][r,t] ))
    if p==t:
        result += +HD['Hbbbb123Dbbbb123'][r,v]
        result += 2 * fsum(( +HD['Hbaab123Dbaab123'][r,v] ,
                             -HD['Hbb1Dbb1'][r,v] ))
    return result


def lucc_bb_aaaa(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += 2 * fsum(( +HD['Hbaab13Dbaaaab25'][p,t,r,u,v,w] ,
                         -HD['Hbaab13Dbaaaab25'][p,u,r,t,v,w] ,
                         +HD['Hbaab13Dbaaaab25'][p,w,r,v,t,u] ,
                         -HD['Hbaab13Dbaaaab25'][p,v,r,w,t,u] ,
                         -HD['Hbaab13Dbaaaab25'][r,t,p,u,v,w] ,
                         +HD['Hbaab13Dbaaaab25'][r,u,p,t,v,w] ,
                         -HD['Hbaab13Dbaaaab25'][r,w,p,v,t,u] ,
                         +HD['Hbaab13Dbaaaab25'][r,v,p,w,t,u] ,
                         +HD['Hbaab23Dbaaaab45'][p,w,r,t,u,v] ,
                         -HD['Hbaab23Dbaaaab45'][p,v,r,t,u,w] ,
                         +HD['Hbaab23Dbaaaab45'][p,t,r,v,w,u] ,
                         -HD['Hbaab23Dbaaaab45'][p,u,r,v,w,t] ,
                         -HD['Hbaab23Dbaaaab45'][r,w,p,t,u,v] ,
                         +HD['Hbaab23Dbaaaab45'][r,v,p,t,u,w] ,
                         -HD['Hbaab23Dbaaaab45'][r,t,p,v,w,u] ,
                         +HD['Hbaab23Dbaaaab45'][r,u,p,v,w,t] ))
    return result


def lucc_bb_baba(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += fsum(( +HD['Hbbbb23Dbbaabb45'][r,v,p,t,u,w] ,
                     -HD['Hbbbb23Dbbaabb45'][r,t,p,v,w,u] ,
                     -HD['Hbbbb23Dbbaabb45'][p,v,r,t,u,w] ,
                     +HD['Hbbbb23Dbbaabb45'][p,t,r,v,w,u] ))
    result += 2 * fsum(( -HD['Hbaab13Dbbaabb25'][p,w,r,v,u,t] ,
                         +HD['Hbaab13Dbbaabb25'][r,w,p,v,u,t] ,
                         +HD['Hbaab13Dbbaabb25'][p,u,r,t,w,v] ,
                         -HD['Hbaab13Dbbaabb25'][r,u,p,t,w,v] ,
                         +HD['Hbaab23Dbbabab45'][r,u,p,v,w,t] ,
                         -HD['Hbaab23Dbbabab45'][p,u,r,v,w,t] ,
                         -HD['Hbaab23Dbbabab45'][r,w,p,t,u,v] ,
                         +HD['Hbaab23Dbbabab45'][p,w,r,t,u,v] ,
                         -HD['Hbaba13Dbaaaba25'][r,v,p,w,u,t] ,
                         +HD['Hbaba13Dbaaaba25'][p,v,r,w,u,t] ,
                         +HD['Hbaba13Dbaaaba25'][r,t,p,u,w,v] ,
                         -HD['Hbaba13Dbaaaba25'][p,t,r,u,w,v] ,
                         -HD['Hbaba3Dbaba3'][v,w,p,t,u,r] ,
                         +HD['Hbaba3Dbaba3'][v,w,r,t,u,p] ,
                         +HD['Hbaba3Dbaba3'][t,u,p,v,w,r] ,
                         -HD['Hbaba3Dbaba3'][t,u,r,v,w,p] ,
                         -HD['Hbb'][p,t] * HD['Dbaab'][r,u,w,v] ,
                         +HD['Hbb'][p,v] * HD['Dbaab'][r,w,u,t] ,
                         +HD['Hbb'][r,t] * HD['Dbaab'][p,u,w,v] ,
                         -HD['Hbb'][r,v] * HD['Dbaab'][p,w,u,t] ,
                         -HD['Hbbbb13Dbababb25'][p,t,r,u,w,v] ,
                         +HD['Hbbbb13Dbababb25'][p,v,r,w,u,t] ,
                         +HD['Hbbbb13Dbababb25'][r,t,p,u,w,v] ,
                         -HD['Hbbbb13Dbababb25'][r,v,p,w,u,t] ))
    if r==v:
        result += +HD['Hbbbb123Dbababb245'][p,t,u,w]
        result += 2 * fsum(( +HD['Hbaab123Dbaaaab245'][p,t,u,w] ,
                             -HD['Hbaab23Dbaab23'][p,w,t,u] ,
                             +HD['Hbb1Dbaab3'][p,t,u,w] ))
    if r==t:
        result += -HD['Hbbbb123Dbababb245'][p,v,w,u]
        result += 2 * fsum(( -HD['Hbaab123Dbaaaab245'][p,v,w,u] ,
                             +HD['Hbaab23Dbaab23'][p,u,v,w] ,
                             -HD['Hbb1Dbaab3'][p,v,w,u] ))
    if p==v:
        result += -HD['Hbbbb123Dbababb245'][r,t,u,w]
        result += 2 * fsum(( -HD['Hbaab123Dbaaaab245'][r,t,u,w] ,
                             +HD['Hbaab23Dbaab23'][r,w,t,u] ,
                             -HD['Hbb1Dbaab3'][r,t,u,w] ))
    if p==t:
        result += +HD['Hbbbb123Dbababb245'][r,v,w,u]
        result += 2 * fsum(( +HD['Hbaab123Dbaaaab245'][r,v,w,u] ,
                             -HD['Hbaab23Dbaab23'][r,u,v,w] ,
                             +HD['Hbb1Dbaab3'][r,v,w,u] ))
    return result


def lucc_bb_bbbb(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += fsum(( -HD['Hbbbb23Dbbbbbb45'][r,t,p,v,w,u] ,
                     +HD['Hbbbb23Dbbbbbb45'][r,u,p,v,w,t] ,
                     +HD['Hbbbb23Dbbbbbb45'][p,w,r,t,u,v] ,
                     -HD['Hbbbb23Dbbbbbb45'][p,v,r,t,u,w] ,
                     +HD['Hbbbb23Dbbbbbb45'][p,t,r,v,w,u] ,
                     -HD['Hbbbb23Dbbbbbb45'][p,u,r,v,w,t] ,
                     -HD['Hbbbb23Dbbbbbb45'][r,w,p,t,u,v] ,
                     +HD['Hbbbb23Dbbbbbb45'][r,v,p,t,u,w] ))
    result += 2 * fsum(( +HD['Hbaba13Dbbabba25'][p,t,r,u,v,w] ,
                         -HD['Hbaba13Dbbabba25'][p,u,r,t,v,w] ,
                         +HD['Hbaba13Dbbabba25'][p,w,r,v,t,u] ,
                         -HD['Hbaba13Dbbabba25'][p,v,r,w,t,u] ,
                         -HD['Hbaba13Dbbabba25'][r,t,p,u,v,w] ,
                         +HD['Hbaba13Dbbabba25'][r,u,p,t,v,w] ,
                         -HD['Hbaba13Dbbabba25'][r,w,p,v,t,u] ,
                         +HD['Hbaba13Dbbabba25'][r,v,p,w,t,u] ,
                         +HD['Hbb'][p,t] * HD['Dbbbb'][r,u,v,w] ,
                         -HD['Hbb'][p,u] * HD['Dbbbb'][r,t,v,w] ,
                         +HD['Hbb'][p,w] * HD['Dbbbb'][r,v,t,u] ,
                         -HD['Hbb'][p,v] * HD['Dbbbb'][r,w,t,u] ,
                         -HD['Hbb'][r,t] * HD['Dbbbb'][p,u,v,w] ,
                         +HD['Hbb'][r,u] * HD['Dbbbb'][p,t,v,w] ,
                         -HD['Hbb'][r,w] * HD['Dbbbb'][p,v,t,u] ,
                         +HD['Hbb'][r,v] * HD['Dbbbb'][p,w,t,u] ,
                         +HD['Hbbbb13Dbbbbbb25'][p,t,r,u,v,w] ,
                         -HD['Hbbbb13Dbbbbbb25'][p,u,r,t,v,w] ,
                         +HD['Hbbbb13Dbbbbbb25'][p,w,r,v,t,u] ,
                         -HD['Hbbbb13Dbbbbbb25'][p,v,r,w,t,u] ,
                         -HD['Hbbbb13Dbbbbbb25'][r,t,p,u,v,w] ,
                         +HD['Hbbbb13Dbbbbbb25'][r,u,p,t,v,w] ,
                         -HD['Hbbbb13Dbbbbbb25'][r,w,p,v,t,u] ,
                         +HD['Hbbbb13Dbbbbbb25'][r,v,p,w,t,u] ,
                         +HD['Hbbbb3Dbbbb3'][t,u,p,v,w,r] ,
                         +HD['Hbbbb3Dbbbb3'][v,w,r,t,u,p] ,
                         -HD['Hbbbb3Dbbbb3'][v,w,p,t,u,r] ,
                         -HD['Hbbbb3Dbbbb3'][t,u,r,v,w,p] ))
    if r==w:
        result += fsum(( -HD['Hbbbb123Dbbbbbb245'][p,t,u,v] ,
                         +HD['Hbbbb23Dbbbb23'][p,v,t,u] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][p,t,u,v] ,
                             -HD['Hbb1Dbbbb3'][p,t,u,v] ))
    if r==v:
        result += fsum(( +HD['Hbbbb123Dbbbbbb245'][p,t,u,w] ,
                         -HD['Hbbbb23Dbbbb23'][p,w,t,u] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][p,t,u,w] ,
                             +HD['Hbb1Dbbbb3'][p,t,u,w] ))
    if r==t:
        result += fsum(( -HD['Hbbbb123Dbbbbbb245'][p,v,w,u] ,
                         +HD['Hbbbb23Dbbbb23'][p,u,v,w] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][p,v,w,u] ,
                             -HD['Hbb1Dbbbb3'][p,v,w,u] ))
    if r==u:
        result += fsum(( +HD['Hbbbb123Dbbbbbb245'][p,v,w,t] ,
                         -HD['Hbbbb23Dbbbb23'][p,t,v,w] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][p,v,w,t] ,
                             +HD['Hbb1Dbbbb3'][p,v,w,t] ))
    if p==w:
        result += fsum(( +HD['Hbbbb123Dbbbbbb245'][r,t,u,v] ,
                         -HD['Hbbbb23Dbbbb23'][r,v,t,u] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][r,t,u,v] ,
                             +HD['Hbb1Dbbbb3'][r,t,u,v] ))
    if p==v:
        result += fsum(( -HD['Hbbbb123Dbbbbbb245'][r,t,u,w] ,
                         +HD['Hbbbb23Dbbbb23'][r,w,t,u] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][r,t,u,w] ,
                             -HD['Hbb1Dbbbb3'][r,t,u,w] ))
    if p==t:
        result += fsum(( +HD['Hbbbb123Dbbbbbb245'][r,v,w,u] ,
                         -HD['Hbbbb23Dbbbb23'][r,u,v,w] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][r,v,w,u] ,
                             +HD['Hbb1Dbbbb3'][r,v,w,u] ))
    if p==u:
        result += fsum(( -HD['Hbbbb123Dbbbbbb245'][r,v,w,t] ,
                         +HD['Hbbbb23Dbbbb23'][r,t,v,w] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][r,v,w,t] ,
                             -HD['Hbb1Dbbbb3'][r,v,w,t] ))
    return result


def lucc_aaaa(HD, *e):
    (p,q,r,s) = e
    result = 0
    result += fsum(( -HD['Haaaa123Daaaaaa245'][p,r,s,q] ,
                     +HD['Haaaa123Daaaaaa245'][q,r,s,p] ,
                     -HD['Haaaa123Daaaaaa245'][s,p,q,r] ,
                     +HD['Haaaa123Daaaaaa245'][r,p,q,s] ,
                     +HD['Haaaa23Daaaa23'][p,q,r,s] ,
                     -HD['Haaaa23Daaaa23'][r,s,p,q] ))
    result += 2 * fsum(( -HD['Haa1Daaaa3'][p,r,s,q] ,
                         +HD['Haa1Daaaa3'][q,r,s,p] ,
                         -HD['Haa1Daaaa3'][s,p,q,r] ,
                         +HD['Haa1Daaaa3'][r,p,q,s] ,
                         -HD['Habab123Daabaab245'][p,r,s,q] ,
                         +HD['Habab123Daabaab245'][q,r,s,p] ,
                         -HD['Habab123Daabaab245'][s,p,q,r] ,
                         +HD['Habab123Daabaab245'][r,p,q,s] ))
    return result


def lucc_aaaa_aa(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += fsum(( +HD['Haaaa23Daaaaaa45'][p,v,r,s,t,q] ,
                     -HD['Haaaa23Daaaaaa45'][p,t,r,s,v,q] ,
                     -HD['Haaaa23Daaaaaa45'][q,v,r,s,t,p] ,
                     +HD['Haaaa23Daaaaaa45'][q,t,r,s,v,p] ,
                     +HD['Haaaa23Daaaaaa45'][s,v,p,q,t,r] ,
                     -HD['Haaaa23Daaaaaa45'][s,t,p,q,v,r] ,
                     -HD['Haaaa23Daaaaaa45'][r,v,p,q,t,s] ,
                     +HD['Haaaa23Daaaaaa45'][r,t,p,q,v,s] ))
    result += 2 * fsum(( -HD['Haa'][p,t] * HD['Daaaa'][q,v,r,s] ,
                         +HD['Haa'][p,v] * HD['Daaaa'][q,t,r,s] ,
                         +HD['Haa'][q,t] * HD['Daaaa'][p,v,r,s] ,
                         -HD['Haa'][q,v] * HD['Daaaa'][p,t,r,s] ,
                         -HD['Haa'][s,t] * HD['Daaaa'][p,q,r,v] ,
                         +HD['Haa'][s,v] * HD['Daaaa'][p,q,r,t] ,
                         +HD['Haa'][r,t] * HD['Daaaa'][p,q,s,v] ,
                         -HD['Haa'][r,v] * HD['Daaaa'][p,q,s,t] ,
                         -HD['Haaaa13Daaaaaa25'][p,t,r,s,q,v] ,
                         +HD['Haaaa13Daaaaaa25'][p,v,r,s,q,t] ,
                         +HD['Haaaa13Daaaaaa25'][q,t,r,s,p,v] ,
                         -HD['Haaaa13Daaaaaa25'][q,v,r,s,p,t] ,
                         -HD['Haaaa13Daaaaaa25'][s,t,p,q,r,v] ,
                         +HD['Haaaa13Daaaaaa25'][s,v,p,q,r,t] ,
                         +HD['Haaaa13Daaaaaa25'][r,t,p,q,s,v] ,
                         -HD['Haaaa13Daaaaaa25'][r,v,p,q,s,t] ,
                         +HD['Haaaa3Daaaa3'][p,q,t,r,s,v] ,
                         -HD['Haaaa3Daaaa3'][p,q,v,r,s,t] ,
                         -HD['Haaaa3Daaaa3'][r,s,t,p,q,v] ,
                         +HD['Haaaa3Daaaa3'][r,s,v,p,q,t] ,
                         -HD['Habab13Daabaab25'][p,t,r,s,q,v] ,
                         +HD['Habab13Daabaab25'][p,v,r,s,q,t] ,
                         +HD['Habab13Daabaab25'][q,t,r,s,p,v] ,
                         -HD['Habab13Daabaab25'][q,v,r,s,p,t] ,
                         -HD['Habab13Daabaab25'][s,t,p,q,r,v] ,
                         +HD['Habab13Daabaab25'][s,v,p,q,r,t] ,
                         +HD['Habab13Daabaab25'][r,t,p,q,s,v] ,
                         -HD['Habab13Daabaab25'][r,v,p,q,s,t] ))
    if q==t:
        result += fsum(( -HD['Haaaa123Daaaaaa245'][p,r,s,v] ,
                         -HD['Haaaa123Daaaaaa245'][s,p,v,r] ,
                         +HD['Haaaa123Daaaaaa245'][r,p,v,s] ,
                         -HD['Haaaa23Daaaa23'][r,s,p,v] ))
        result += 2 * fsum(( -HD['Haa1Daaaa3'][p,r,s,v] ,
                             -HD['Haa1Daaaa3'][s,p,v,r] ,
                             +HD['Haa1Daaaa3'][r,p,v,s] ,
                             -HD['Habab123Daabaab245'][s,p,v,r] ,
                             +HD['Habab123Daabaab245'][r,p,v,s] ,
                             -HD['Habab123Daabaab245'][p,r,s,v] ))
    if r==v:
        result += fsum(( +HD['Haaaa123Daaaaaa245'][s,p,q,t] ,
                         -HD['Haaaa123Daaaaaa245'][p,s,t,q] ,
                         +HD['Haaaa123Daaaaaa245'][q,s,t,p] ,
                         +HD['Haaaa23Daaaa23'][p,q,s,t] ))
        result += 2 * fsum(( -HD['Haa1Daaaa3'][p,s,t,q] ,
                             +HD['Haa1Daaaa3'][q,s,t,p] ,
                             +HD['Haa1Daaaa3'][s,p,q,t] ,
                             +HD['Habab123Daabaab245'][s,p,q,t] ,
                             -HD['Habab123Daabaab245'][p,s,t,q] ,
                             +HD['Habab123Daabaab245'][q,s,t,p] ))
    if s==v:
        result += fsum(( +HD['Haaaa123Daaaaaa245'][p,r,t,q] ,
                         -HD['Haaaa123Daaaaaa245'][r,p,q,t] ,
                         -HD['Haaaa123Daaaaaa245'][q,r,t,p] ,
                         -HD['Haaaa23Daaaa23'][p,q,r,t] ))
        result += 2 * fsum(( +HD['Haa1Daaaa3'][p,r,t,q] ,
                             -HD['Haa1Daaaa3'][q,r,t,p] ,
                             -HD['Haa1Daaaa3'][r,p,q,t] ,
                             -HD['Habab123Daabaab245'][r,p,q,t] ,
                             +HD['Habab123Daabaab245'][p,r,t,q] ,
                             -HD['Habab123Daabaab245'][q,r,t,p] ))
    if q==v:
        result += fsum(( +HD['Haaaa123Daaaaaa245'][s,p,t,r] ,
                         +HD['Haaaa123Daaaaaa245'][p,r,s,t] ,
                         -HD['Haaaa123Daaaaaa245'][r,p,t,s] ,
                         +HD['Haaaa23Daaaa23'][r,s,p,t] ))
        result += 2 * fsum(( +HD['Haa1Daaaa3'][p,r,s,t] ,
                             +HD['Haa1Daaaa3'][s,p,t,r] ,
                             -HD['Haa1Daaaa3'][r,p,t,s] ,
                             +HD['Habab123Daabaab245'][s,p,t,r] ,
                             -HD['Habab123Daabaab245'][r,p,t,s] ,
                             +HD['Habab123Daabaab245'][p,r,s,t] ))
    if r==t:
        result += fsum(( -HD['Haaaa123Daaaaaa245'][s,p,q,v] ,
                         -HD['Haaaa123Daaaaaa245'][q,s,v,p] ,
                         +HD['Haaaa123Daaaaaa245'][p,s,v,q] ,
                         -HD['Haaaa23Daaaa23'][p,q,s,v] ))
        result += 2 * fsum(( +HD['Haa1Daaaa3'][p,s,v,q] ,
                             -HD['Haa1Daaaa3'][q,s,v,p] ,
                             -HD['Haa1Daaaa3'][s,p,q,v] ,
                             -HD['Habab123Daabaab245'][q,s,v,p] ,
                             -HD['Habab123Daabaab245'][s,p,q,v] ,
                             +HD['Habab123Daabaab245'][p,s,v,q] ))
    if s==t:
        result += fsum(( +HD['Haaaa123Daaaaaa245'][r,p,q,v] ,
                         -HD['Haaaa123Daaaaaa245'][p,r,v,q] ,
                         +HD['Haaaa123Daaaaaa245'][q,r,v,p] ,
                         +HD['Haaaa23Daaaa23'][p,q,r,v] ))
        result += 2 * fsum(( -HD['Haa1Daaaa3'][p,r,v,q] ,
                             +HD['Haa1Daaaa3'][q,r,v,p] ,
                             +HD['Haa1Daaaa3'][r,p,q,v] ,
                             +HD['Habab123Daabaab245'][q,r,v,p] ,
                             +HD['Habab123Daabaab245'][r,p,q,v] ,
                             -HD['Habab123Daabaab245'][p,r,v,q] ))
    if p==t:
        result += fsum(( +HD['Haaaa123Daaaaaa245'][s,q,v,r] ,
                         -HD['Haaaa123Daaaaaa245'][r,q,v,s] ,
                         +HD['Haaaa123Daaaaaa245'][q,r,s,v] ,
                         +HD['Haaaa23Daaaa23'][r,s,q,v] ))
        result += 2 * fsum(( +HD['Haa1Daaaa3'][q,r,s,v] ,
                             +HD['Haa1Daaaa3'][s,q,v,r] ,
                             -HD['Haa1Daaaa3'][r,q,v,s] ,
                             +HD['Habab123Daabaab245'][s,q,v,r] ,
                             -HD['Habab123Daabaab245'][r,q,v,s] ,
                             +HD['Habab123Daabaab245'][q,r,s,v] ))
    if p==v:
        result += fsum(( -HD['Haaaa123Daaaaaa245'][s,q,t,r] ,
                         +HD['Haaaa123Daaaaaa245'][r,q,t,s] ,
                         -HD['Haaaa123Daaaaaa245'][q,r,s,t] ,
                         -HD['Haaaa23Daaaa23'][r,s,q,t] ))
        result += 2 * fsum(( -HD['Haa1Daaaa3'][q,r,s,t] ,
                             -HD['Haa1Daaaa3'][s,q,t,r] ,
                             +HD['Haa1Daaaa3'][r,q,t,s] ,
                             -HD['Habab123Daabaab245'][q,r,s,t] ,
                             -HD['Habab123Daabaab245'][s,q,t,r] ,
                             +HD['Habab123Daabaab245'][r,q,t,s] ))
    return result


def lucc_aaaa_bb(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += 2 * fsum(( -HD['Habba13Daababa25'][p,t,r,s,q,v] ,
                         +HD['Habba13Daababa25'][p,v,r,s,q,t] ,
                         +HD['Habba13Daababa25'][q,t,r,s,p,v] ,
                         -HD['Habba13Daababa25'][q,v,r,s,p,t] ,
                         -HD['Habba13Daababa25'][s,t,p,q,r,v] ,
                         +HD['Habba13Daababa25'][s,v,p,q,r,t] ,
                         +HD['Habba13Daababa25'][r,t,p,q,s,v] ,
                         -HD['Habba13Daababa25'][r,v,p,q,s,t] ,
                         -HD['Hbaab23Dbaaaab45'][v,p,t,r,s,q] ,
                         +HD['Hbaab23Dbaaaab45'][t,p,v,r,s,q] ,
                         +HD['Hbaab23Dbaaaab45'][v,q,t,r,s,p] ,
                         -HD['Hbaab23Dbaaaab45'][t,q,v,r,s,p] ,
                         -HD['Hbaab23Dbaaaab45'][v,s,t,p,q,r] ,
                         +HD['Hbaab23Dbaaaab45'][t,s,v,p,q,r] ,
                         +HD['Hbaab23Dbaaaab45'][v,r,t,p,q,s] ,
                         -HD['Hbaab23Dbaaaab45'][t,r,v,p,q,s] ))
    return result


def lucc_aaaa_aaaa(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( +HD['Haaaa23Daaaaaaaa67'][r,t,p,q,v,w,s,u] ,
                     +HD['Haaaa23Daaaaaaaa67'][q,w,r,s,t,u,p,v] ,
                     -HD['Haaaa23Daaaaaaaa67'][q,v,r,s,t,u,p,w] ,
                     -HD['Haaaa23Daaaaaaaa67'][r,u,p,q,v,w,s,t] ,
                     +HD['Haaaa23Daaaaaaaa67'][q,t,r,s,v,w,p,u] ,
                     -HD['Haaaa23Daaaaaaaa67'][q,u,r,s,v,w,p,t] ,
                     -HD['Haaaa23Daaaaaaaa67'][s,w,p,q,t,u,r,v] ,
                     +HD['Haaaa23Daaaaaaaa67'][s,v,p,q,t,u,r,w] ,
                     -HD['Haaaa23Daaaaaaaa67'][r,v,p,q,t,u,s,w] ,
                     -HD['Haaaa23Daaaaaaaa67'][s,t,p,q,v,w,r,u] ,
                     -HD['Haaaa23Daaaaaaaa67'][p,w,r,s,t,u,q,v] ,
                     +HD['Haaaa23Daaaaaaaa67'][p,v,r,s,t,u,q,w] ,
                     +HD['Haaaa23Daaaaaaaa67'][s,u,p,q,v,w,r,t] ,
                     +HD['Haaaa23Daaaaaaaa67'][r,w,p,q,t,u,s,v] ,
                     -HD['Haaaa23Daaaaaaaa67'][p,t,r,s,v,w,q,u] ,
                     +HD['Haaaa23Daaaaaaaa67'][p,u,r,s,v,w,q,t] ))
    result += 2 * fsum(( +HD['Haa'][p,t] * HD['Daaaaaa'][q,v,w,r,s,u] ,
                         -HD['Haa'][p,u] * HD['Daaaaaa'][q,v,w,r,s,t] ,
                         +HD['Haa'][p,w] * HD['Daaaaaa'][q,t,u,r,s,v] ,
                         -HD['Haa'][p,v] * HD['Daaaaaa'][q,t,u,r,s,w] ,
                         -HD['Haa'][q,t] * HD['Daaaaaa'][p,v,w,r,s,u] ,
                         +HD['Haa'][q,u] * HD['Daaaaaa'][p,v,w,r,s,t] ,
                         -HD['Haa'][q,w] * HD['Daaaaaa'][p,t,u,r,s,v] ,
                         +HD['Haa'][q,v] * HD['Daaaaaa'][p,t,u,r,s,w] ,
                         +HD['Haa'][s,t] * HD['Daaaaaa'][p,q,u,r,v,w] ,
                         -HD['Haa'][s,u] * HD['Daaaaaa'][p,q,t,r,v,w] ,
                         +HD['Haa'][s,w] * HD['Daaaaaa'][p,q,v,r,t,u] ,
                         -HD['Haa'][s,v] * HD['Daaaaaa'][p,q,w,r,t,u] ,
                         -HD['Haa'][r,t] * HD['Daaaaaa'][p,q,u,s,v,w] ,
                         +HD['Haa'][r,u] * HD['Daaaaaa'][p,q,t,s,v,w] ,
                         -HD['Haa'][r,w] * HD['Daaaaaa'][p,q,v,s,t,u] ,
                         +HD['Haa'][r,v] * HD['Daaaaaa'][p,q,w,s,t,u] ,
                         +HD['Haaaa13Daaaaaaaa37'][q,w,r,s,v,p,t,u] ,
                         -HD['Haaaa13Daaaaaaaa37'][q,v,r,s,w,p,t,u] ,
                         -HD['Haaaa13Daaaaaaaa37'][s,t,p,q,u,r,v,w] ,
                         +HD['Haaaa13Daaaaaaaa37'][s,u,p,q,t,r,v,w] ,
                         -HD['Haaaa13Daaaaaaaa37'][s,w,p,q,v,r,t,u] ,
                         +HD['Haaaa13Daaaaaaaa37'][s,v,p,q,w,r,t,u] ,
                         +HD['Haaaa13Daaaaaaaa37'][r,t,p,q,u,s,v,w] ,
                         -HD['Haaaa13Daaaaaaaa37'][r,u,p,q,t,s,v,w] ,
                         +HD['Haaaa13Daaaaaaaa37'][r,w,p,q,v,s,t,u] ,
                         -HD['Haaaa13Daaaaaaaa37'][r,v,p,q,w,s,t,u] ,
                         -HD['Haaaa13Daaaaaaaa37'][p,t,r,s,u,q,v,w] ,
                         +HD['Haaaa13Daaaaaaaa37'][p,u,r,s,t,q,v,w] ,
                         -HD['Haaaa13Daaaaaaaa37'][p,w,r,s,v,q,t,u] ,
                         +HD['Haaaa13Daaaaaaaa37'][p,v,r,s,w,q,t,u] ,
                         +HD['Haaaa13Daaaaaaaa37'][q,t,r,s,u,p,v,w] ,
                         -HD['Haaaa13Daaaaaaaa37'][q,u,r,s,t,p,v,w] ,
                         -HD['Haaaa3Daaaaaa5'][r,s,t,p,q,u,v,w] ,
                         +HD['Haaaa3Daaaaaa5'][r,s,u,p,q,t,v,w] ,
                         -HD['Haaaa3Daaaaaa5'][r,s,w,p,q,v,t,u] ,
                         +HD['Haaaa3Daaaaaa5'][r,s,v,p,q,w,t,u] ,
                         +HD['Haaaa3Daaaaaa5'][t,u,s,r,v,w,p,q] ,
                         -HD['Haaaa3Daaaaaa5'][v,w,s,r,t,u,p,q] ,
                         -HD['Haaaa3Daaaaaa5'][t,u,r,s,v,w,p,q] ,
                         +HD['Haaaa3Daaaaaa5'][p,q,t,r,s,u,v,w] ,
                         -HD['Haaaa3Daaaaaa5'][p,q,u,r,s,t,v,w] ,
                         +HD['Haaaa3Daaaaaa5'][v,w,r,s,t,u,p,q] ,
                         +HD['Haaaa3Daaaaaa5'][p,q,w,r,s,v,t,u] ,
                         -HD['Haaaa3Daaaaaa5'][p,q,v,r,s,w,t,u] ,
                         +HD['Haaaa3Daaaaaa5'][t,u,p,q,v,w,r,s] ,
                         -HD['Haaaa3Daaaaaa5'][v,w,p,q,t,u,r,s] ,
                         -HD['Haaaa3Daaaaaa5'][t,u,q,p,v,w,r,s] ,
                         +HD['Haaaa3Daaaaaa5'][v,w,q,p,t,u,r,s] ,
                         -HD['Haaaa'][p,q,t,u] * HD['Daaaa'][r,s,v,w] ,
                         +HD['Haaaa'][r,s,t,u] * HD['Daaaa'][p,q,v,w] ,
                         +HD['Haaaa'][p,q,v,w] * HD['Daaaa'][r,s,t,u] ,
                         -HD['Haaaa'][r,s,v,w] * HD['Daaaa'][p,q,t,u] ,
                         +HD['Habab13Daaabaaab37'][r,t,p,q,u,s,v,w] ,
                         -HD['Habab13Daaabaaab37'][r,u,p,q,t,s,v,w] ,
                         +HD['Habab13Daaabaaab37'][r,w,p,q,v,s,t,u] ,
                         -HD['Habab13Daaabaaab37'][r,v,p,q,w,s,t,u] ,
                         -HD['Habab13Daaabaaab37'][p,t,r,s,u,q,v,w] ,
                         +HD['Habab13Daaabaaab37'][p,u,r,s,t,q,v,w] ,
                         -HD['Habab13Daaabaaab37'][p,w,r,s,v,q,t,u] ,
                         +HD['Habab13Daaabaaab37'][p,v,r,s,w,q,t,u] ,
                         +HD['Habab13Daaabaaab37'][q,t,r,s,u,p,v,w] ,
                         -HD['Habab13Daaabaaab37'][q,u,r,s,t,p,v,w] ,
                         +HD['Habab13Daaabaaab37'][q,w,r,s,v,p,t,u] ,
                         -HD['Habab13Daaabaaab37'][q,v,r,s,w,p,t,u] ,
                         -HD['Habab13Daaabaaab37'][s,t,p,q,u,r,v,w] ,
                         +HD['Habab13Daaabaaab37'][s,u,p,q,t,r,v,w] ,
                         -HD['Habab13Daaabaaab37'][s,w,p,q,v,r,t,u] ,
                         +HD['Habab13Daaabaaab37'][s,v,p,q,w,r,t,u] ))
    if q==u:
        result += fsum(( -HD['Haaaa123Daaaaaaaa367'][p,r,s,t,v,w] ,
                         -HD['Haaaa123Daaaaaaaa367'][s,p,v,w,r,t] ,
                         +HD['Haaaa123Daaaaaaaa367'][r,p,v,w,s,t] ,
                         -HD['Haaaa23Daaaaaa45'][r,s,p,v,w,t] ,
                         -HD['Haaaa23Daaaaaa45'][s,t,p,v,w,r] ,
                         +HD['Haaaa23Daaaaaa45'][r,t,p,v,w,s] ))
        result += 2 * fsum(( -HD['Haa1Daaaaaa5'][r,p,v,w,s,t] ,
                             +HD['Haa1Daaaaaa5'][p,r,s,t,v,w] ,
                             +HD['Haa1Daaaaaa5'][s,p,v,w,r,t] ,
                             -HD['Haa'][p,t] * HD['Daaaa'][r,s,v,w] ,
                             -HD['Haaaa13Daaaaaa25'][p,t,r,s,v,w] ,
                             -HD['Habab123Daaabaaab367'][p,r,s,t,v,w] ,
                             -HD['Habab123Daaabaaab367'][s,p,v,w,r,t] ,
                             +HD['Habab123Daaabaaab367'][r,p,v,w,s,t] ,
                             -HD['Habab13Daabaab25'][p,t,r,s,v,w] ))
        if p==t:
            result += fsum(( +HD['Haaaa123Daaaaaa245'][s,v,w,r] ,
                             -HD['Haaaa123Daaaaaa245'][r,v,w,s] ,
                             +HD['Haaaa23Daaaa23'][r,s,v,w] ))
            result += 2 * fsum(( +HD['Haa1Daaaa3'][s,v,w,r] ,
                                 -HD['Haa1Daaaa3'][r,v,w,s] ,
                                 -HD['Habab123Daabaab245'][r,v,w,s] ,
                                 +HD['Habab123Daabaab245'][s,v,w,r] ))
    if q==t:
        result += fsum(( +HD['Haaaa123Daaaaaaaa367'][s,p,v,w,r,u] ,
                         -HD['Haaaa123Daaaaaaaa367'][r,p,v,w,s,u] ,
                         +HD['Haaaa123Daaaaaaaa367'][p,r,s,u,v,w] ,
                         +HD['Haaaa23Daaaaaa45'][r,s,p,v,w,u] ,
                         +HD['Haaaa23Daaaaaa45'][s,u,p,v,w,r] ,
                         -HD['Haaaa23Daaaaaa45'][r,u,p,v,w,s] ))
        result += 2 * fsum(( -HD['Haa1Daaaaaa5'][p,r,s,u,v,w] ,
                             -HD['Haa1Daaaaaa5'][s,p,v,w,r,u] ,
                             +HD['Haa1Daaaaaa5'][r,p,v,w,s,u] ,
                             +HD['Haa'][p,u] * HD['Daaaa'][r,s,v,w] ,
                             +HD['Haaaa13Daaaaaa25'][p,u,r,s,v,w] ,
                             +HD['Habab123Daaabaaab367'][p,r,s,u,v,w] ,
                             +HD['Habab123Daaabaaab367'][s,p,v,w,r,u] ,
                             -HD['Habab123Daaabaaab367'][r,p,v,w,s,u] ,
                             +HD['Habab13Daabaab25'][p,u,r,s,v,w] ))
        if p==u:
            result += fsum(( -HD['Haaaa123Daaaaaa245'][s,v,w,r] ,
                             +HD['Haaaa123Daaaaaa245'][r,v,w,s] ,
                             -HD['Haaaa23Daaaa23'][r,s,v,w] ))
            result += 2 * fsum(( -HD['Haa1Daaaa3'][s,v,w,r] ,
                                 +HD['Haa1Daaaa3'][r,v,w,s] ,
                                 -HD['Habab123Daabaab245'][s,v,w,r] ,
                                 +HD['Habab123Daabaab245'][r,v,w,s] ))
    if s==v:
        result += fsum(( -HD['Haaaa123Daaaaaaaa367'][p,r,t,u,q,w] ,
                         +HD['Haaaa123Daaaaaaaa367'][q,r,t,u,p,w] ,
                         +HD['Haaaa123Daaaaaaaa367'][r,p,q,w,t,u] ,
                         -HD['Haaaa23Daaaaaa45'][p,w,r,t,u,q] ,
                         +HD['Haaaa23Daaaaaa45'][q,w,r,t,u,p] ,
                         +HD['Haaaa23Daaaaaa45'][p,q,r,t,u,w] ))
        result += 2 * fsum(( -HD['Haa1Daaaaaa5'][q,r,t,u,p,w] ,
                             +HD['Haa1Daaaaaa5'][p,r,t,u,q,w] ,
                             -HD['Haa1Daaaaaa5'][r,p,q,w,t,u] ,
                             +HD['Haa'][r,w] * HD['Daaaa'][p,q,t,u] ,
                             +HD['Haaaa13Daaaaaa25'][r,w,p,q,t,u] ,
                             +HD['Habab123Daaabaaab367'][q,r,t,u,p,w] ,
                             -HD['Habab123Daaabaaab367'][p,r,t,u,q,w] ,
                             +HD['Habab123Daaabaaab367'][r,p,q,w,t,u] ,
                             +HD['Habab13Daabaab25'][r,w,p,q,t,u] ))
        if r==w:
            result += fsum(( +HD['Haaaa123Daaaaaa245'][p,t,u,q] ,
                             -HD['Haaaa123Daaaaaa245'][q,t,u,p] ,
                             -HD['Haaaa23Daaaa23'][p,q,t,u] ))
            result += 2 * fsum(( +HD['Haa1Daaaa3'][p,t,u,q] ,
                                 -HD['Haa1Daaaa3'][q,t,u,p] ,
                                 +HD['Habab123Daabaab245'][p,t,u,q] ,
                                 -HD['Habab123Daabaab245'][q,t,u,p] ))
    if r==w:
        result += fsum(( +HD['Haaaa123Daaaaaaaa367'][q,s,t,u,p,v] ,
                         +HD['Haaaa123Daaaaaaaa367'][s,p,q,v,t,u] ,
                         -HD['Haaaa123Daaaaaaaa367'][p,s,t,u,q,v] ,
                         +HD['Haaaa23Daaaaaa45'][q,v,s,t,u,p] ,
                         +HD['Haaaa23Daaaaaa45'][p,q,s,t,u,v] ,
                         -HD['Haaaa23Daaaaaa45'][p,v,s,t,u,q] ))
        result += 2 * fsum(( -HD['Haa1Daaaaaa5'][s,p,q,v,t,u] ,
                             +HD['Haa1Daaaaaa5'][p,s,t,u,q,v] ,
                             -HD['Haa1Daaaaaa5'][q,s,t,u,p,v] ,
                             +HD['Haa'][s,v] * HD['Daaaa'][p,q,t,u] ,
                             +HD['Haaaa13Daaaaaa25'][s,v,p,q,t,u] ,
                             -HD['Habab123Daaabaaab367'][p,s,t,u,q,v] ,
                             +HD['Habab123Daaabaaab367'][s,p,q,v,t,u] ,
                             +HD['Habab123Daaabaaab367'][q,s,t,u,p,v] ,
                             +HD['Habab13Daabaab25'][s,v,p,q,t,u] ))
    if s==w:
        result += fsum(( +HD['Haaaa123Daaaaaaaa367'][p,r,t,u,q,v] ,
                         -HD['Haaaa123Daaaaaaaa367'][q,r,t,u,p,v] ,
                         -HD['Haaaa123Daaaaaaaa367'][r,p,q,v,t,u] ,
                         -HD['Haaaa23Daaaaaa45'][q,v,r,t,u,p] ,
                         -HD['Haaaa23Daaaaaa45'][p,q,r,t,u,v] ,
                         +HD['Haaaa23Daaaaaa45'][p,v,r,t,u,q] ))
        result += 2 * fsum(( +HD['Haa1Daaaaaa5'][q,r,t,u,p,v] ,
                             -HD['Haa1Daaaaaa5'][p,r,t,u,q,v] ,
                             +HD['Haa1Daaaaaa5'][r,p,q,v,t,u] ,
                             -HD['Haa'][r,v] * HD['Daaaa'][p,q,t,u] ,
                             -HD['Haaaa13Daaaaaa25'][r,v,p,q,t,u] ,
                             -HD['Habab123Daaabaaab367'][q,r,t,u,p,v] ,
                             +HD['Habab123Daaabaaab367'][p,r,t,u,q,v] ,
                             -HD['Habab123Daaabaaab367'][r,p,q,v,t,u] ,
                             -HD['Habab13Daabaab25'][r,v,p,q,t,u] ))
        if r==v:
            result += fsum(( +HD['Haaaa123Daaaaaa245'][q,t,u,p] ,
                             -HD['Haaaa123Daaaaaa245'][p,t,u,q] ,
                             +HD['Haaaa23Daaaa23'][p,q,t,u] ))
            result += 2 * fsum(( +HD['Haa1Daaaa3'][q,t,u,p] ,
                                 -HD['Haa1Daaaa3'][p,t,u,q] ,
                                 -HD['Habab123Daabaab245'][p,t,u,q] ,
                                 +HD['Habab123Daabaab245'][q,t,u,p] ))
    if r==v:
        result += fsum(( -HD['Haaaa123Daaaaaaaa367'][q,s,t,u,p,w] ,
                         -HD['Haaaa123Daaaaaaaa367'][s,p,q,w,t,u] ,
                         +HD['Haaaa123Daaaaaaaa367'][p,s,t,u,q,w] ,
                         -HD['Haaaa23Daaaaaa45'][q,w,s,t,u,p] ,
                         -HD['Haaaa23Daaaaaa45'][p,q,s,t,u,w] ,
                         +HD['Haaaa23Daaaaaa45'][p,w,s,t,u,q] ))
        result += 2 * fsum(( +HD['Haa1Daaaaaa5'][s,p,q,w,t,u] ,
                             +HD['Haa1Daaaaaa5'][q,s,t,u,p,w] ,
                             -HD['Haa1Daaaaaa5'][p,s,t,u,q,w] ,
                             -HD['Haa'][s,w] * HD['Daaaa'][p,q,t,u] ,
                             -HD['Haaaa13Daaaaaa25'][s,w,p,q,t,u] ,
                             -HD['Habab123Daaabaaab367'][s,p,q,w,t,u] ,
                             +HD['Habab123Daaabaaab367'][p,s,t,u,q,w] ,
                             -HD['Habab123Daaabaaab367'][q,s,t,u,p,w] ,
                             -HD['Habab13Daabaab25'][s,w,p,q,t,u] ))
    if q==v:
        result += fsum(( -HD['Haaaa123Daaaaaaaa367'][s,p,t,u,r,w] ,
                         +HD['Haaaa123Daaaaaaaa367'][r,p,t,u,s,w] ,
                         -HD['Haaaa123Daaaaaaaa367'][p,r,s,w,t,u] ,
                         +HD['Haaaa23Daaaaaa45'][r,w,p,t,u,s] ,
                         -HD['Haaaa23Daaaaaa45'][r,s,p,t,u,w] ,
                         -HD['Haaaa23Daaaaaa45'][s,w,p,t,u,r] ))
        result += 2 * fsum(( +HD['Haa1Daaaaaa5'][s,p,t,u,r,w] ,
                             +HD['Haa1Daaaaaa5'][p,r,s,w,t,u] ,
                             -HD['Haa1Daaaaaa5'][r,p,t,u,s,w] ,
                             -HD['Haa'][p,w] * HD['Daaaa'][r,s,t,u] ,
                             -HD['Haaaa13Daaaaaa25'][p,w,r,s,t,u] ,
                             -HD['Habab123Daaabaaab367'][p,r,s,w,t,u] ,
                             +HD['Habab123Daaabaaab367'][r,p,t,u,s,w] ,
                             -HD['Habab123Daaabaaab367'][s,p,t,u,r,w] ,
                             -HD['Habab13Daabaab25'][p,w,r,s,t,u] ))
        if p==w:
            result += fsum(( +HD['Haaaa123Daaaaaa245'][s,t,u,r] ,
                             -HD['Haaaa123Daaaaaa245'][r,t,u,s] ,
                             +HD['Haaaa23Daaaa23'][r,s,t,u] ))
            result += 2 * fsum(( -HD['Haa1Daaaa3'][r,t,u,s] ,
                                 +HD['Haa1Daaaa3'][s,t,u,r] ,
                                 -HD['Habab123Daabaab245'][r,t,u,s] ,
                                 +HD['Habab123Daabaab245'][s,t,u,r] ))
    if q==w:
        result += fsum(( -HD['Haaaa123Daaaaaaaa367'][r,p,t,u,s,v] ,
                         +HD['Haaaa123Daaaaaaaa367'][p,r,s,v,t,u] ,
                         +HD['Haaaa123Daaaaaaaa367'][s,p,t,u,r,v] ,
                         -HD['Haaaa23Daaaaaa45'][r,v,p,t,u,s] ,
                         +HD['Haaaa23Daaaaaa45'][r,s,p,t,u,v] ,
                         +HD['Haaaa23Daaaaaa45'][s,v,p,t,u,r] ))
        result += 2 * fsum(( -HD['Haa1Daaaaaa5'][s,p,t,u,r,v] ,
                             -HD['Haa1Daaaaaa5'][p,r,s,v,t,u] ,
                             +HD['Haa1Daaaaaa5'][r,p,t,u,s,v] ,
                             +HD['Haa'][p,v] * HD['Daaaa'][r,s,t,u] ,
                             +HD['Haaaa13Daaaaaa25'][p,v,r,s,t,u] ,
                             +HD['Habab123Daaabaaab367'][p,r,s,v,t,u] ,
                             -HD['Habab123Daaabaaab367'][r,p,t,u,s,v] ,
                             +HD['Habab123Daaabaaab367'][s,p,t,u,r,v] ,
                             +HD['Habab13Daabaab25'][p,v,r,s,t,u] ))
        if p==v:
            result += fsum(( +HD['Haaaa123Daaaaaa245'][r,t,u,s] ,
                             -HD['Haaaa123Daaaaaa245'][s,t,u,r] ,
                             -HD['Haaaa23Daaaa23'][r,s,t,u] ))
            result += 2 * fsum(( +HD['Haa1Daaaa3'][r,t,u,s] ,
                                 -HD['Haa1Daaaa3'][s,t,u,r] ,
                                 +HD['Habab123Daabaab245'][r,t,u,s] ,
                                 -HD['Habab123Daabaab245'][s,t,u,r] ))
    if s==u:
        result += fsum(( -HD['Haaaa123Daaaaaaaa367'][p,r,v,w,q,t] ,
                         +HD['Haaaa123Daaaaaaaa367'][q,r,v,w,p,t] ,
                         +HD['Haaaa123Daaaaaaaa367'][r,p,q,t,v,w] ,
                         +HD['Haaaa23Daaaaaa45'][p,q,r,v,w,t] ,
                         -HD['Haaaa23Daaaaaa45'][p,t,r,v,w,q] ,
                         +HD['Haaaa23Daaaaaa45'][q,t,r,v,w,p] ))
        result += 2 * fsum(( +HD['Haa1Daaaaaa5'][p,r,v,w,q,t] ,
                             -HD['Haa1Daaaaaa5'][r,p,q,t,v,w] ,
                             -HD['Haa1Daaaaaa5'][q,r,v,w,p,t] ,
                             +HD['Haa'][r,t] * HD['Daaaa'][p,q,v,w] ,
                             +HD['Haaaa13Daaaaaa25'][r,t,p,q,v,w] ,
                             +HD['Habab123Daaabaaab367'][r,p,q,t,v,w] ,
                             +HD['Habab123Daaabaaab367'][q,r,v,w,p,t] ,
                             -HD['Habab123Daaabaaab367'][p,r,v,w,q,t] ,
                             +HD['Habab13Daabaab25'][r,t,p,q,v,w] ))
        if r==t:
            result += fsum(( +HD['Haaaa123Daaaaaa245'][p,v,w,q] ,
                             -HD['Haaaa123Daaaaaa245'][q,v,w,p] ,
                             -HD['Haaaa23Daaaa23'][p,q,v,w] ))
            result += 2 * fsum(( -HD['Haa1Daaaa3'][q,v,w,p] ,
                                 +HD['Haa1Daaaa3'][p,v,w,q] ,
                                 -HD['Habab123Daabaab245'][q,v,w,p] ,
                                 +HD['Habab123Daabaab245'][p,v,w,q] ))
    if r==t:
        result += fsum(( -HD['Haaaa123Daaaaaaaa367'][p,s,v,w,q,u] ,
                         +HD['Haaaa123Daaaaaaaa367'][q,s,v,w,p,u] ,
                         +HD['Haaaa123Daaaaaaaa367'][s,p,q,u,v,w] ,
                         +HD['Haaaa23Daaaaaa45'][p,q,s,v,w,u] ,
                         -HD['Haaaa23Daaaaaa45'][p,u,s,v,w,q] ,
                         +HD['Haaaa23Daaaaaa45'][q,u,s,v,w,p] ))
        result += 2 * fsum(( -HD['Haa1Daaaaaa5'][q,s,v,w,p,u] ,
                             +HD['Haa1Daaaaaa5'][p,s,v,w,q,u] ,
                             -HD['Haa1Daaaaaa5'][s,p,q,u,v,w] ,
                             +HD['Haa'][s,u] * HD['Daaaa'][p,q,v,w] ,
                             +HD['Haaaa13Daaaaaa25'][s,u,p,q,v,w] ,
                             +HD['Habab123Daaabaaab367'][q,s,v,w,p,u] ,
                             +HD['Habab123Daaabaaab367'][s,p,q,u,v,w] ,
                             -HD['Habab123Daaabaaab367'][p,s,v,w,q,u] ,
                             +HD['Habab13Daabaab25'][s,u,p,q,v,w] ))
    if r==u:
        result += fsum(( +HD['Haaaa123Daaaaaaaa367'][p,s,v,w,q,t] ,
                         -HD['Haaaa123Daaaaaaaa367'][q,s,v,w,p,t] ,
                         -HD['Haaaa123Daaaaaaaa367'][s,p,q,t,v,w] ,
                         -HD['Haaaa23Daaaaaa45'][p,q,s,v,w,t] ,
                         +HD['Haaaa23Daaaaaa45'][p,t,s,v,w,q] ,
                         -HD['Haaaa23Daaaaaa45'][q,t,s,v,w,p] ))
        result += 2 * fsum(( +HD['Haa1Daaaaaa5'][q,s,v,w,p,t] ,
                             -HD['Haa1Daaaaaa5'][p,s,v,w,q,t] ,
                             +HD['Haa1Daaaaaa5'][s,p,q,t,v,w] ,
                             -HD['Haa'][s,t] * HD['Daaaa'][p,q,v,w] ,
                             -HD['Haaaa13Daaaaaa25'][s,t,p,q,v,w] ,
                             -HD['Habab123Daaabaaab367'][q,s,v,w,p,t] ,
                             -HD['Habab123Daaabaaab367'][s,p,q,t,v,w] ,
                             +HD['Habab123Daaabaaab367'][p,s,v,w,q,t] ,
                             -HD['Habab13Daabaab25'][s,t,p,q,v,w] ))
        if s==t:
            result += fsum(( -HD['Haaaa123Daaaaaa245'][p,v,w,q] ,
                             +HD['Haaaa123Daaaaaa245'][q,v,w,p] ,
                             +HD['Haaaa23Daaaa23'][p,q,v,w] ))
            result += 2 * fsum(( +HD['Haa1Daaaa3'][q,v,w,p] ,
                                 -HD['Haa1Daaaa3'][p,v,w,q] ,
                                 +HD['Habab123Daabaab245'][q,v,w,p] ,
                                 -HD['Habab123Daabaab245'][p,v,w,q] ))
    if s==t:
        result += fsum(( -HD['Haaaa123Daaaaaaaa367'][q,r,v,w,p,u] ,
                         +HD['Haaaa123Daaaaaaaa367'][p,r,v,w,q,u] ,
                         -HD['Haaaa123Daaaaaaaa367'][r,p,q,u,v,w] ,
                         -HD['Haaaa23Daaaaaa45'][p,q,r,v,w,u] ,
                         +HD['Haaaa23Daaaaaa45'][p,u,r,v,w,q] ,
                         -HD['Haaaa23Daaaaaa45'][q,u,r,v,w,p] ))
        result += 2 * fsum(( +HD['Haa1Daaaaaa5'][r,p,q,u,v,w] ,
                             +HD['Haa1Daaaaaa5'][q,r,v,w,p,u] ,
                             -HD['Haa1Daaaaaa5'][p,r,v,w,q,u] ,
                             -HD['Haa'][r,u] * HD['Daaaa'][p,q,v,w] ,
                             -HD['Haaaa13Daaaaaa25'][r,u,p,q,v,w] ,
                             -HD['Habab123Daaabaaab367'][r,p,q,u,v,w] ,
                             -HD['Habab123Daaabaaab367'][q,r,v,w,p,u] ,
                             +HD['Habab123Daaabaaab367'][p,r,v,w,q,u] ,
                             -HD['Habab13Daabaab25'][r,u,p,q,v,w] ))
    if p==u:
        result += fsum(( +HD['Haaaa123Daaaaaaaa367'][s,q,v,w,r,t] ,
                         -HD['Haaaa123Daaaaaaaa367'][r,q,v,w,s,t] ,
                         +HD['Haaaa123Daaaaaaaa367'][q,r,s,t,v,w] ,
                         +HD['Haaaa23Daaaaaa45'][r,s,q,v,w,t] ,
                         +HD['Haaaa23Daaaaaa45'][s,t,q,v,w,r] ,
                         -HD['Haaaa23Daaaaaa45'][r,t,q,v,w,s] ))
        result += 2 * fsum(( -HD['Haa1Daaaaaa5'][q,r,s,t,v,w] ,
                             -HD['Haa1Daaaaaa5'][s,q,v,w,r,t] ,
                             +HD['Haa1Daaaaaa5'][r,q,v,w,s,t] ,
                             +HD['Haa'][q,t] * HD['Daaaa'][r,s,v,w] ,
                             +HD['Haaaa13Daaaaaa25'][q,t,r,s,v,w] ,
                             +HD['Habab123Daaabaaab367'][s,q,v,w,r,t] ,
                             +HD['Habab123Daaabaaab367'][q,r,s,t,v,w] ,
                             -HD['Habab123Daaabaaab367'][r,q,v,w,s,t] ,
                             +HD['Habab13Daabaab25'][q,t,r,s,v,w] ))
    if p==t:
        result += fsum(( -HD['Haaaa123Daaaaaaaa367'][s,q,v,w,r,u] ,
                         +HD['Haaaa123Daaaaaaaa367'][r,q,v,w,s,u] ,
                         -HD['Haaaa123Daaaaaaaa367'][q,r,s,u,v,w] ,
                         -HD['Haaaa23Daaaaaa45'][r,s,q,v,w,u] ,
                         -HD['Haaaa23Daaaaaa45'][s,u,q,v,w,r] ,
                         +HD['Haaaa23Daaaaaa45'][r,u,q,v,w,s] ))
        result += 2 * fsum(( +HD['Haa1Daaaaaa5'][s,q,v,w,r,u] ,
                             -HD['Haa1Daaaaaa5'][r,q,v,w,s,u] ,
                             +HD['Haa1Daaaaaa5'][q,r,s,u,v,w] ,
                             -HD['Haa'][q,u] * HD['Daaaa'][r,s,v,w] ,
                             -HD['Haaaa13Daaaaaa25'][q,u,r,s,v,w] ,
                             -HD['Habab123Daaabaaab367'][s,q,v,w,r,u] ,
                             +HD['Habab123Daaabaaab367'][r,q,v,w,s,u] ,
                             -HD['Habab123Daaabaaab367'][q,r,s,u,v,w] ,
                             -HD['Habab13Daabaab25'][q,u,r,s,v,w] ))
    if p==v:
        result += fsum(( -HD['Haaaa123Daaaaaaaa367'][r,q,t,u,s,w] ,
                         +HD['Haaaa123Daaaaaaaa367'][q,r,s,w,t,u] ,
                         +HD['Haaaa123Daaaaaaaa367'][s,q,t,u,r,w] ,
                         -HD['Haaaa23Daaaaaa45'][r,w,q,t,u,s] ,
                         +HD['Haaaa23Daaaaaa45'][r,s,q,t,u,w] ,
                         +HD['Haaaa23Daaaaaa45'][s,w,q,t,u,r] ))
        result += 2 * fsum(( -HD['Haa1Daaaaaa5'][s,q,t,u,r,w] ,
                             -HD['Haa1Daaaaaa5'][q,r,s,w,t,u] ,
                             +HD['Haa1Daaaaaa5'][r,q,t,u,s,w] ,
                             +HD['Haa'][q,w] * HD['Daaaa'][r,s,t,u] ,
                             +HD['Haaaa13Daaaaaa25'][q,w,r,s,t,u] ,
                             +HD['Habab123Daaabaaab367'][q,r,s,w,t,u] ,
                             -HD['Habab123Daaabaaab367'][r,q,t,u,s,w] ,
                             +HD['Habab123Daaabaaab367'][s,q,t,u,r,w] ,
                             +HD['Habab13Daabaab25'][q,w,r,s,t,u] ))
    if p==w:
        result += fsum(( -HD['Haaaa123Daaaaaaaa367'][q,r,s,v,t,u] ,
                         -HD['Haaaa123Daaaaaaaa367'][s,q,t,u,r,v] ,
                         +HD['Haaaa123Daaaaaaaa367'][r,q,t,u,s,v] ,
                         -HD['Haaaa23Daaaaaa45'][r,s,q,t,u,v] ,
                         -HD['Haaaa23Daaaaaa45'][s,v,q,t,u,r] ,
                         +HD['Haaaa23Daaaaaa45'][r,v,q,t,u,s] ))
        result += 2 * fsum(( +HD['Haa1Daaaaaa5'][s,q,t,u,r,v] ,
                             +HD['Haa1Daaaaaa5'][q,r,s,v,t,u] ,
                             -HD['Haa1Daaaaaa5'][r,q,t,u,s,v] ,
                             -HD['Haa'][q,v] * HD['Daaaa'][r,s,t,u] ,
                             -HD['Haaaa13Daaaaaa25'][q,v,r,s,t,u] ,
                             -HD['Habab123Daaabaaab367'][q,r,s,v,t,u] ,
                             +HD['Habab123Daaabaaab367'][r,q,t,u,s,v] ,
                             -HD['Habab123Daaabaaab367'][s,q,t,u,r,v] ,
                             -HD['Habab13Daabaab25'][q,v,r,s,t,u] ))
    return result


def lucc_aaaa_baba(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( -HD['Haaaa23Dbaaaabaa67'][s,w,t,p,q,u,r,v] ,
                     +HD['Haaaa23Dbaaaabaa67'][s,u,v,p,q,w,r,t] ,
                     +HD['Haaaa23Dbaaaabaa67'][r,w,t,p,q,u,s,v] ,
                     -HD['Haaaa23Dbaaaabaa67'][r,u,v,p,q,w,s,t] ,
                     -HD['Haaaa23Dbaaaabaa67'][p,w,t,r,s,u,q,v] ,
                     +HD['Haaaa23Dbaaaabaa67'][p,u,v,r,s,w,q,t] ,
                     +HD['Haaaa23Dbaaaabaa67'][q,w,t,r,s,u,p,v] ,
                     -HD['Haaaa23Dbaaaabaa67'][q,u,v,r,s,w,p,t] ))
    result += 2 * fsum(( +HD['Haa'][p,u] * HD['Dbaaaab'][v,q,w,r,s,t] ,
                         -HD['Haa'][p,w] * HD['Dbaaaab'][t,q,u,r,s,v] ,
                         -HD['Haa'][q,u] * HD['Dbaaaab'][v,p,w,r,s,t] ,
                         +HD['Haa'][q,w] * HD['Dbaaaab'][t,p,u,r,s,v] ,
                         +HD['Haa'][s,u] * HD['Dbaaaab'][t,p,q,r,w,v] ,
                         -HD['Haa'][s,w] * HD['Dbaaaab'][v,p,q,r,u,t] ,
                         -HD['Haa'][r,u] * HD['Dbaaaab'][t,p,q,s,w,v] ,
                         +HD['Haa'][r,w] * HD['Dbaaaab'][v,p,q,s,u,t] ,
                         -HD['Haaaa13Dbaaaaaba37'][r,w,v,p,q,s,u,t] ,
                         -HD['Haaaa13Dbaaaaaba37'][p,u,t,r,s,q,w,v] ,
                         +HD['Haaaa13Dbaaaaaba37'][p,w,v,r,s,q,u,t] ,
                         +HD['Haaaa13Dbaaaaaba37'][q,u,t,r,s,p,w,v] ,
                         -HD['Haaaa13Dbaaaaaba37'][q,w,v,r,s,p,u,t] ,
                         -HD['Haaaa13Dbaaaaaba37'][s,u,t,p,q,r,w,v] ,
                         +HD['Haaaa13Dbaaaaaba37'][s,w,v,p,q,r,u,t] ,
                         +HD['Haaaa13Dbaaaaaba37'][r,u,t,p,q,s,w,v] ,
                         +HD['Haaaa3Dbaaaba5'][r,s,w,v,p,q,u,t] ,
                         -HD['Haaaa3Dbaaaba5'][r,s,u,t,p,q,w,v] ,
                         +HD['Haaaa3Dbaaaba5'][p,q,u,t,r,s,w,v] ,
                         -HD['Haaaa3Dbaaaba5'][p,q,w,v,r,s,u,t] ,
                         -HD['Habab13Dbaabaabb37'][p,u,t,r,s,q,w,v] ,
                         +HD['Habab13Dbaabaabb37'][s,w,v,p,q,r,u,t] ,
                         +HD['Habab13Dbaabaabb37'][p,w,v,r,s,q,u,t] ,
                         +HD['Habab13Dbaabaabb37'][q,u,t,r,s,p,w,v] ,
                         +HD['Habab13Dbaabaabb37'][r,u,t,p,q,s,w,v] ,
                         -HD['Habab13Dbaabaabb37'][q,w,v,r,s,p,u,t] ,
                         -HD['Habab13Dbaabaabb37'][r,w,v,p,q,s,u,t] ,
                         -HD['Habab13Dbaabaabb37'][s,u,t,p,q,r,w,v] ,
                         +HD['Habba13Daaabaaba37'][p,t,r,s,u,q,w,v] ,
                         -HD['Habba13Daaabaaba37'][s,v,p,q,w,r,u,t] ,
                         -HD['Habba13Daaabaaba37'][p,v,r,s,w,q,u,t] ,
                         -HD['Habba13Daaabaaba37'][q,t,r,s,u,p,w,v] ,
                         -HD['Habba13Daaabaaba37'][r,t,p,q,u,s,w,v] ,
                         +HD['Habba13Daaabaaba37'][q,v,r,s,w,p,u,t] ,
                         +HD['Habba13Daaabaaba37'][s,t,p,q,u,r,w,v] ,
                         +HD['Habba13Daaabaaba37'][r,v,p,q,w,s,u,t] ,
                         -HD['Hbaab23Dbaaaaaab67'][v,s,t,p,q,u,r,w] ,
                         -HD['Hbaab23Dbaaaaaab67'][v,p,t,r,s,u,q,w] ,
                         +HD['Hbaab23Dbaaaaaab67'][t,s,v,p,q,w,r,u] ,
                         +HD['Hbaab23Dbaaaaaab67'][t,p,v,r,s,w,q,u] ,
                         +HD['Hbaab23Dbaaaaaab67'][v,q,t,r,s,u,p,w] ,
                         +HD['Hbaab23Dbaaaaaab67'][v,r,t,p,q,u,s,w] ,
                         -HD['Hbaab23Dbaaaaaab67'][t,q,v,r,s,w,p,u] ,
                         -HD['Hbaab23Dbaaaaaab67'][t,r,v,p,q,w,s,u] ,
                         -HD['Hbaab3Dbaaaab5'][t,u,p,v,q,w,r,s] ,
                         +HD['Hbaab3Dbaaaab5'][v,w,s,t,r,u,p,q] ,
                         +HD['Hbaab3Dbaaaab5'][v,w,p,t,q,u,r,s] ,
                         +HD['Hbaab3Dbaaaab5'][t,u,q,v,p,w,r,s] ,
                         +HD['Hbaab3Dbaaaab5'][t,u,r,v,s,w,p,q] ,
                         -HD['Hbaab3Dbaaaab5'][v,w,q,t,p,u,r,s] ,
                         -HD['Hbaab3Dbaaaab5'][v,w,r,t,s,u,p,q] ,
                         -HD['Hbaab3Dbaaaab5'][t,u,s,v,r,w,p,q] ))
    if q==u:
        result += fsum(( +HD['Haaaa123Dbaaaabaa367'][p,t,r,s,w,v] ,
                         +HD['Haaaa123Dbaaaabaa367'][s,v,p,w,r,t] ,
                         -HD['Haaaa123Dbaaaabaa367'][r,v,p,w,s,t] ,
                         +HD['Haaaa23Dbaabaa45'][r,s,v,p,w,t] ))
        result += 2 * fsum(( -HD['Haa1Dbaaaba5'][p,t,r,s,w,v] ,
                             +HD['Haa1Dbaaaba5'][r,v,p,w,s,t] ,
                             -HD['Haa1Dbaaaba5'][s,v,p,w,r,t] ,
                             -HD['Habab123Dbaababab367'][r,v,p,w,s,t] ,
                             +HD['Habab123Dbaababab367'][s,v,p,w,r,t] ,
                             +HD['Habab123Dbaababab367'][p,t,r,s,w,v] ,
                             +HD['Habba13Daababa25'][p,t,r,s,w,v] ,
                             +HD['Hbaab23Dbaaaab45'][t,r,v,p,w,s] ,
                             -HD['Hbaab23Dbaaaab45'][t,s,v,p,w,r] ))
    if r==w:
        result += fsum(( -HD['Haaaa123Dbaaaabaa367'][q,t,s,u,p,v] ,
                         -HD['Haaaa123Dbaaaabaa367'][s,v,p,q,u,t] ,
                         +HD['Haaaa123Dbaaaabaa367'][p,t,s,u,q,v] ,
                         -HD['Haaaa23Dbaabaa45'][p,q,t,s,u,v] ))
        result += 2 * fsum(( -HD['Haa1Dbaaaba5'][p,t,s,u,q,v] ,
                             +HD['Haa1Dbaaaba5'][q,t,s,u,p,v] ,
                             +HD['Haa1Dbaaaba5'][s,v,p,q,u,t] ,
                             +HD['Habab123Dbaababab367'][p,t,s,u,q,v] ,
                             -HD['Habab123Dbaababab367'][s,v,p,q,u,t] ,
                             -HD['Habab123Dbaababab367'][q,t,s,u,p,v] ,
                             -HD['Habba13Daababa25'][s,v,p,q,u,t] ,
                             +HD['Hbaab23Dbaaaab45'][v,q,t,s,u,p] ,
                             -HD['Hbaab23Dbaaaab45'][v,p,t,s,u,q] ))
    if s==w:
        result += fsum(( +HD['Haaaa123Dbaaaabaa367'][q,t,r,u,p,v] ,
                         -HD['Haaaa123Dbaaaabaa367'][p,t,r,u,q,v] ,
                         +HD['Haaaa123Dbaaaabaa367'][r,v,p,q,u,t] ,
                         +HD['Haaaa23Dbaabaa45'][p,q,t,r,u,v] ))
        result += 2 * fsum(( +HD['Haa1Dbaaaba5'][p,t,r,u,q,v] ,
                             -HD['Haa1Dbaaaba5'][q,t,r,u,p,v] ,
                             -HD['Haa1Dbaaaba5'][r,v,p,q,u,t] ,
                             +HD['Habab123Dbaababab367'][r,v,p,q,u,t] ,
                             +HD['Habab123Dbaababab367'][q,t,r,u,p,v] ,
                             -HD['Habab123Dbaababab367'][p,t,r,u,q,v] ,
                             +HD['Habba13Daababa25'][r,v,p,q,u,t] ,
                             -HD['Hbaab23Dbaaaab45'][v,q,t,r,u,p] ,
                             +HD['Hbaab23Dbaaaab45'][v,p,t,r,u,q] ))
    if q==w:
        result += fsum(( -HD['Haaaa123Dbaaaabaa367'][s,t,p,u,r,v] ,
                         -HD['Haaaa123Dbaaaabaa367'][p,v,r,s,u,t] ,
                         +HD['Haaaa123Dbaaaabaa367'][r,t,p,u,s,v] ,
                         -HD['Haaaa23Dbaabaa45'][r,s,t,p,u,v] ))
        result += 2 * fsum(( -HD['Haa1Dbaaaba5'][r,t,p,u,s,v] ,
                             +HD['Haa1Dbaaaba5'][p,v,r,s,u,t] ,
                             +HD['Haa1Dbaaaba5'][s,t,p,u,r,v] ,
                             -HD['Habab123Dbaababab367'][p,v,r,s,u,t] ,
                             -HD['Habab123Dbaababab367'][s,t,p,u,r,v] ,
                             +HD['Habab123Dbaababab367'][r,t,p,u,s,v] ,
                             -HD['Habba13Daababa25'][p,v,r,s,u,t] ,
                             +HD['Hbaab23Dbaaaab45'][v,s,t,p,u,r] ,
                             -HD['Hbaab23Dbaaaab45'][v,r,t,p,u,s] ))
    if r==u:
        result += fsum(( +HD['Haaaa123Dbaaaabaa367'][s,t,p,q,w,v] ,
                         -HD['Haaaa123Dbaaaabaa367'][p,v,s,w,q,t] ,
                         +HD['Haaaa123Dbaaaabaa367'][q,v,s,w,p,t] ,
                         +HD['Haaaa23Dbaabaa45'][p,q,v,s,w,t] ))
        result += 2 * fsum(( +HD['Haa1Dbaaaba5'][p,v,s,w,q,t] ,
                             -HD['Haa1Dbaaaba5'][q,v,s,w,p,t] ,
                             -HD['Haa1Dbaaaba5'][s,t,p,q,w,v] ,
                             +HD['Habab123Dbaababab367'][s,t,p,q,w,v] ,
                             -HD['Habab123Dbaababab367'][p,v,s,w,q,t] ,
                             +HD['Habab123Dbaababab367'][q,v,s,w,p,t] ,
                             +HD['Habba13Daababa25'][s,t,p,q,w,v] ,
                             +HD['Hbaab23Dbaaaab45'][t,p,v,s,w,q] ,
                             -HD['Hbaab23Dbaaaab45'][t,q,v,s,w,p] ))
    if s==u:
        result += fsum(( -HD['Haaaa123Dbaaaabaa367'][r,t,p,q,w,v] ,
                         +HD['Haaaa123Dbaaaabaa367'][p,v,r,w,q,t] ,
                         -HD['Haaaa123Dbaaaabaa367'][q,v,r,w,p,t] ,
                         -HD['Haaaa23Dbaabaa45'][p,q,v,r,w,t] ))
        result += 2 * fsum(( +HD['Haa1Dbaaaba5'][r,t,p,q,w,v] ,
                             -HD['Haa1Dbaaaba5'][p,v,r,w,q,t] ,
                             +HD['Haa1Dbaaaba5'][q,v,r,w,p,t] ,
                             +HD['Habab123Dbaababab367'][p,v,r,w,q,t] ,
                             -HD['Habab123Dbaababab367'][r,t,p,q,w,v] ,
                             -HD['Habab123Dbaababab367'][q,v,r,w,p,t] ,
                             -HD['Habba13Daababa25'][r,t,p,q,w,v] ,
                             -HD['Hbaab23Dbaaaab45'][t,p,v,r,w,q] ,
                             +HD['Hbaab23Dbaaaab45'][t,q,v,r,w,p] ))
    if p==u:
        result += fsum(( +HD['Haaaa123Dbaaaabaa367'][r,v,q,w,s,t] ,
                         -HD['Haaaa123Dbaaaabaa367'][s,v,q,w,r,t] ,
                         -HD['Haaaa123Dbaaaabaa367'][q,t,r,s,w,v] ,
                         -HD['Haaaa23Dbaabaa45'][r,s,v,q,w,t] ))
        result += 2 * fsum(( +HD['Haa1Dbaaaba5'][q,t,r,s,w,v] ,
                             +HD['Haa1Dbaaaba5'][s,v,q,w,r,t] ,
                             -HD['Haa1Dbaaaba5'][r,v,q,w,s,t] ,
                             +HD['Habab123Dbaababab367'][r,v,q,w,s,t] ,
                             -HD['Habab123Dbaababab367'][q,t,r,s,w,v] ,
                             -HD['Habab123Dbaababab367'][s,v,q,w,r,t] ,
                             -HD['Habba13Daababa25'][q,t,r,s,w,v] ,
                             -HD['Hbaab23Dbaaaab45'][t,r,v,q,w,s] ,
                             +HD['Hbaab23Dbaaaab45'][t,s,v,q,w,r] ))
    if p==w:
        result += fsum(( +HD['Haaaa123Dbaaaabaa367'][q,v,r,s,u,t] ,
                         +HD['Haaaa123Dbaaaabaa367'][s,t,q,u,r,v] ,
                         -HD['Haaaa123Dbaaaabaa367'][r,t,q,u,s,v] ,
                         +HD['Haaaa23Dbaabaa45'][r,s,t,q,u,v] ))
        result += 2 * fsum(( +HD['Haa1Dbaaaba5'][r,t,q,u,s,v] ,
                             -HD['Haa1Dbaaaba5'][q,v,r,s,u,t] ,
                             -HD['Haa1Dbaaaba5'][s,t,q,u,r,v] ,
                             +HD['Habab123Dbaababab367'][s,t,q,u,r,v] ,
                             +HD['Habab123Dbaababab367'][q,v,r,s,u,t] ,
                             -HD['Habab123Dbaababab367'][r,t,q,u,s,v] ,
                             +HD['Habba13Daababa25'][q,v,r,s,u,t] ,
                             -HD['Hbaab23Dbaaaab45'][v,s,t,q,u,r] ,
                             +HD['Hbaab23Dbaaaab45'][v,r,t,q,u,s] ))
    return result


def lucc_aaaa_bbbb(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += 2 * fsum(( -HD['Habba13Dbaababba37'][p,t,u,r,s,q,v,w] ,
                         +HD['Habba13Dbaababba37'][p,u,t,r,s,q,v,w] ,
                         -HD['Habba13Dbaababba37'][p,w,v,r,s,q,t,u] ,
                         +HD['Habba13Dbaababba37'][p,v,w,r,s,q,t,u] ,
                         +HD['Habba13Dbaababba37'][q,t,u,r,s,p,v,w] ,
                         -HD['Habba13Dbaababba37'][q,u,t,r,s,p,v,w] ,
                         +HD['Habba13Dbaababba37'][q,w,v,r,s,p,t,u] ,
                         -HD['Habba13Dbaababba37'][q,v,w,r,s,p,t,u] ,
                         -HD['Habba13Dbaababba37'][s,t,u,p,q,r,v,w] ,
                         +HD['Habba13Dbaababba37'][s,u,t,p,q,r,v,w] ,
                         -HD['Habba13Dbaababba37'][s,w,v,p,q,r,t,u] ,
                         +HD['Habba13Dbaababba37'][s,v,w,p,q,r,t,u] ,
                         +HD['Habba13Dbaababba37'][r,t,u,p,q,s,v,w] ,
                         -HD['Habba13Dbaababba37'][r,u,t,p,q,s,v,w] ,
                         +HD['Habba13Dbaababba37'][r,w,v,p,q,s,t,u] ,
                         -HD['Habba13Dbaababba37'][r,v,w,p,q,s,t,u] ,
                         +HD['Hbaab23Dbbaaabab67'][w,p,t,u,r,s,q,v] ,
                         -HD['Hbaab23Dbbaaabab67'][v,p,t,u,r,s,q,w] ,
                         +HD['Hbaab23Dbbaaabab67'][t,p,v,w,r,s,q,u] ,
                         -HD['Hbaab23Dbbaaabab67'][u,p,v,w,r,s,q,t] ,
                         -HD['Hbaab23Dbbaaabab67'][w,q,t,u,r,s,p,v] ,
                         +HD['Hbaab23Dbbaaabab67'][v,q,t,u,r,s,p,w] ,
                         -HD['Hbaab23Dbbaaabab67'][t,q,v,w,r,s,p,u] ,
                         +HD['Hbaab23Dbbaaabab67'][u,q,v,w,r,s,p,t] ,
                         +HD['Hbaab23Dbbaaabab67'][w,s,t,u,p,q,r,v] ,
                         -HD['Hbaab23Dbbaaabab67'][v,s,t,u,p,q,r,w] ,
                         +HD['Hbaab23Dbbaaabab67'][t,s,v,w,p,q,r,u] ,
                         -HD['Hbaab23Dbbaaabab67'][u,s,v,w,p,q,r,t] ,
                         -HD['Hbaab23Dbbaaabab67'][w,r,t,u,p,q,s,v] ,
                         +HD['Hbaab23Dbbaaabab67'][v,r,t,u,p,q,s,w] ,
                         -HD['Hbaab23Dbbaaabab67'][t,r,v,w,p,q,s,u] ,
                         +HD['Hbaab23Dbbaaabab67'][u,r,v,w,p,q,s,t] ))
    return result


def lucc_baba(HD, *e):
    (p,q,r,s) = e
    result = 0
    result += fsum(( +HD['Haaaa123Dbaabaa245'][q,r,s,p] ,
                     -HD['Haaaa123Dbaabaa245'][s,p,q,r] ,
                     -HD['Hbbbb123Dbababb245'][p,r,s,q] ,
                     +HD['Hbbbb123Dbababb245'][r,p,q,s] ))
    result += 2 * fsum(( +HD['Haa1Dbaba3'][q,r,s,p] ,
                         -HD['Haa1Dbaba3'][s,p,q,r] ,
                         -HD['Habab123Dbabbab245'][s,p,q,r] ,
                         +HD['Habab123Dbabbab245'][q,r,s,p] ,
                         +HD['Hbaab123Dbaaaab245'][r,p,q,s] ,
                         -HD['Hbaab123Dbaaaab245'][p,r,s,q] ,
                         +HD['Hbaab23Dbaab23'][p,q,r,s] ,
                         -HD['Hbaab23Dbaab23'][r,s,p,q] ,
                         -HD['Hbb1Dbaab3'][p,r,s,q] ,
                         +HD['Hbb1Dbaab3'][r,p,q,s] ))
    return result


def lucc_baba_aa(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += fsum(( +HD['Haaaa23Dbaabaa45'][s,v,p,q,t,r] ,
                     -HD['Haaaa23Dbaabaa45'][s,t,p,q,v,r] ,
                     -HD['Haaaa23Dbaabaa45'][q,v,r,s,t,p] ,
                     +HD['Haaaa23Dbaabaa45'][q,t,r,s,v,p] ))
    result += 2 * fsum(( -HD['Haa'][q,t] * HD['Dbaab'][p,v,s,r] ,
                         +HD['Haa'][q,v] * HD['Dbaab'][p,t,s,r] ,
                         +HD['Haa'][s,t] * HD['Dbaab'][p,q,v,r] ,
                         -HD['Haa'][s,v] * HD['Dbaab'][p,q,t,r] ,
                         -HD['Haaaa13Dbaaaba25'][q,t,r,s,v,p] ,
                         +HD['Haaaa13Dbaaaba25'][q,v,r,s,t,p] ,
                         +HD['Haaaa13Dbaaaba25'][s,t,p,q,v,r] ,
                         -HD['Haaaa13Dbaaaba25'][s,v,p,q,t,r] ,
                         +HD['Habab13Dbababb25'][s,t,p,q,v,r] ,
                         -HD['Habab13Dbababb25'][s,v,p,q,t,r] ,
                         -HD['Habab13Dbababb25'][q,t,r,s,v,p] ,
                         +HD['Habab13Dbababb25'][q,v,r,s,t,p] ,
                         -HD['Hbaab13Dbaaaab25'][p,t,r,s,q,v] ,
                         +HD['Hbaab13Dbaaaab25'][p,v,r,s,q,t] ,
                         +HD['Hbaab13Dbaaaab25'][r,t,p,q,s,v] ,
                         -HD['Hbaab13Dbaaaab25'][r,v,p,q,s,t] ,
                         -HD['Hbaab23Dbaaaab45'][r,v,p,q,t,s] ,
                         +HD['Hbaab23Dbaaaab45'][p,v,r,s,t,q] ,
                         -HD['Hbaab23Dbaaaab45'][p,t,r,s,v,q] ,
                         +HD['Hbaab23Dbaaaab45'][r,t,p,q,v,s] ,
                         +HD['Hbaab3Dbaab3'][p,q,t,r,s,v] ,
                         -HD['Hbaab3Dbaab3'][p,q,v,r,s,t] ,
                         -HD['Hbaab3Dbaab3'][r,s,t,p,q,v] ,
                         +HD['Hbaab3Dbaab3'][r,s,v,p,q,t] ))
    if s==v:
        result += fsum(( -HD['Haaaa123Dbaabaa245'][q,r,t,p] ,
                         +HD['Hbbbb123Dbababb245'][p,r,t,q] ,
                         -HD['Hbbbb123Dbababb245'][r,p,q,t] ))
        result += 2 * fsum(( -HD['Haa1Dbaba3'][q,r,t,p] ,
                             -HD['Habab123Dbabbab245'][q,r,t,p] ,
                             +HD['Hbaab123Dbaaaab245'][p,r,t,q] ,
                             -HD['Hbaab123Dbaaaab245'][r,p,q,t] ,
                             -HD['Hbaab23Dbaab23'][p,q,r,t] ,
                             +HD['Hbb1Dbaab3'][p,r,t,q] ,
                             -HD['Hbb1Dbaab3'][r,p,q,t] ))
    if s==t:
        result += fsum(( +HD['Haaaa123Dbaabaa245'][q,r,v,p] ,
                         +HD['Hbbbb123Dbababb245'][r,p,q,v] ,
                         -HD['Hbbbb123Dbababb245'][p,r,v,q] ))
        result += 2 * fsum(( +HD['Haa1Dbaba3'][q,r,v,p] ,
                             +HD['Habab123Dbabbab245'][q,r,v,p] ,
                             -HD['Hbaab123Dbaaaab245'][p,r,v,q] ,
                             +HD['Hbaab123Dbaaaab245'][r,p,q,v] ,
                             +HD['Hbaab23Dbaab23'][p,q,r,v] ,
                             -HD['Hbb1Dbaab3'][p,r,v,q] ,
                             +HD['Hbb1Dbaab3'][r,p,q,v] ))
    if q==v:
        result += fsum(( +HD['Haaaa123Dbaabaa245'][s,p,t,r] ,
                         -HD['Hbbbb123Dbababb245'][r,p,t,s] ,
                         +HD['Hbbbb123Dbababb245'][p,r,s,t] ))
        result += 2 * fsum(( +HD['Haa1Dbaba3'][s,p,t,r] ,
                             +HD['Habab123Dbabbab245'][s,p,t,r] ,
                             +HD['Hbaab123Dbaaaab245'][p,r,s,t] ,
                             -HD['Hbaab123Dbaaaab245'][r,p,t,s] ,
                             +HD['Hbaab23Dbaab23'][r,s,p,t] ,
                             +HD['Hbb1Dbaab3'][p,r,s,t] ,
                             -HD['Hbb1Dbaab3'][r,p,t,s] ))
    if q==t:
        result += fsum(( -HD['Haaaa123Dbaabaa245'][s,p,v,r] ,
                         +HD['Hbbbb123Dbababb245'][r,p,v,s] ,
                         -HD['Hbbbb123Dbababb245'][p,r,s,v] ))
        result += 2 * fsum(( -HD['Haa1Dbaba3'][s,p,v,r] ,
                             -HD['Habab123Dbabbab245'][s,p,v,r] ,
                             -HD['Hbaab123Dbaaaab245'][p,r,s,v] ,
                             +HD['Hbaab123Dbaaaab245'][r,p,v,s] ,
                             -HD['Hbaab23Dbaab23'][r,s,p,v] ,
                             -HD['Hbb1Dbaab3'][p,r,s,v] ,
                             +HD['Hbb1Dbaab3'][r,p,v,s] ))
    return result


def lucc_baba_bb(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += fsum(( +HD['Hbbbb23Dbbaabb45'][r,v,p,t,q,s] ,
                     -HD['Hbbbb23Dbbaabb45'][r,t,p,v,q,s] ,
                     -HD['Hbbbb23Dbbaabb45'][p,v,r,t,s,q] ,
                     +HD['Hbbbb23Dbbaabb45'][p,t,r,v,s,q] ))
    result += 2 * fsum(( +HD['Habba13Dbabbba25'][q,t,r,s,p,v] ,
                         +HD['Habba13Dbabbba25'][s,v,p,q,r,t] ,
                         -HD['Habba13Dbabbba25'][q,v,r,s,p,t] ,
                         -HD['Habba13Dbabbba25'][s,t,p,q,r,v] ,
                         -HD['Hbaab23Dbbabab45'][v,q,r,t,s,p] ,
                         +HD['Hbaab23Dbbabab45'][t,q,r,v,s,p] ,
                         -HD['Hbaab23Dbbabab45'][t,s,p,v,q,r] ,
                         +HD['Hbaab23Dbbabab45'][v,s,p,t,q,r] ,
                         +HD['Hbaba13Dbaaaba25'][r,t,p,q,s,v] ,
                         -HD['Hbaba13Dbaaaba25'][r,v,p,q,s,t] ,
                         -HD['Hbaba13Dbaaaba25'][p,t,r,s,q,v] ,
                         +HD['Hbaba13Dbaaaba25'][p,v,r,s,q,t] ,
                         -HD['Hbaba3Dbaba3'][r,s,t,p,q,v] ,
                         +HD['Hbaba3Dbaba3'][r,s,v,p,q,t] ,
                         +HD['Hbaba3Dbaba3'][p,q,t,r,s,v] ,
                         -HD['Hbaba3Dbaba3'][p,q,v,r,s,t] ,
                         -HD['Hbb'][p,t] * HD['Dbaab'][v,q,s,r] ,
                         +HD['Hbb'][p,v] * HD['Dbaab'][t,q,s,r] ,
                         +HD['Hbb'][r,t] * HD['Dbaab'][p,q,s,v] ,
                         -HD['Hbb'][r,v] * HD['Dbaab'][p,q,s,t] ,
                         -HD['Hbbbb13Dbababb25'][p,t,r,s,q,v] ,
                         +HD['Hbbbb13Dbababb25'][p,v,r,s,q,t] ,
                         +HD['Hbbbb13Dbababb25'][r,t,p,q,s,v] ,
                         -HD['Hbbbb13Dbababb25'][r,v,p,q,s,t] ))
    if p==t:
        result += fsum(( -HD['Haaaa123Dbaabaa245'][s,v,q,r] ,
                         +HD['Haaaa123Dbaabaa245'][q,r,s,v] ,
                         +HD['Hbbbb123Dbababb245'][r,v,q,s] ))
        result += 2 * fsum(( +HD['Haa1Dbaba3'][q,r,s,v] ,
                             -HD['Haa1Dbaba3'][s,v,q,r] ,
                             +HD['Habab123Dbabbab245'][q,r,s,v] ,
                             -HD['Habab123Dbabbab245'][s,v,q,r] ,
                             +HD['Hbaab123Dbaaaab245'][r,v,q,s] ,
                             -HD['Hbaab23Dbaab23'][r,s,v,q] ,
                             +HD['Hbb1Dbaab3'][r,v,q,s] ))
    if r==v:
        result += fsum(( -HD['Haaaa123Dbaabaa245'][q,t,s,p] ,
                         +HD['Haaaa123Dbaabaa245'][s,p,q,t] ,
                         +HD['Hbbbb123Dbababb245'][p,t,s,q] ))
        result += 2 * fsum(( -HD['Haa1Dbaba3'][q,t,s,p] ,
                             +HD['Haa1Dbaba3'][s,p,q,t] ,
                             -HD['Habab123Dbabbab245'][q,t,s,p] ,
                             +HD['Habab123Dbabbab245'][s,p,q,t] ,
                             +HD['Hbaab123Dbaaaab245'][p,t,s,q] ,
                             -HD['Hbaab23Dbaab23'][p,q,t,s] ,
                             +HD['Hbb1Dbaab3'][p,t,s,q] ))
    if p==v:
        result += fsum(( -HD['Haaaa123Dbaabaa245'][q,r,s,t] ,
                         +HD['Haaaa123Dbaabaa245'][s,t,q,r] ,
                         -HD['Hbbbb123Dbababb245'][r,t,q,s] ))
        result += 2 * fsum(( -HD['Haa1Dbaba3'][q,r,s,t] ,
                             +HD['Haa1Dbaba3'][s,t,q,r] ,
                             -HD['Habab123Dbabbab245'][q,r,s,t] ,
                             +HD['Habab123Dbabbab245'][s,t,q,r] ,
                             -HD['Hbaab123Dbaaaab245'][r,t,q,s] ,
                             +HD['Hbaab23Dbaab23'][r,s,t,q] ,
                             -HD['Hbb1Dbaab3'][r,t,q,s] ))
    if r==t:
        result += fsum(( +HD['Haaaa123Dbaabaa245'][q,v,s,p] ,
                         -HD['Haaaa123Dbaabaa245'][s,p,q,v] ,
                         -HD['Hbbbb123Dbababb245'][p,v,s,q] ))
        result += 2 * fsum(( +HD['Haa1Dbaba3'][q,v,s,p] ,
                             -HD['Haa1Dbaba3'][s,p,q,v] ,
                             +HD['Habab123Dbabbab245'][q,v,s,p] ,
                             -HD['Habab123Dbabbab245'][s,p,q,v] ,
                             -HD['Hbaab123Dbaaaab245'][p,v,s,q] ,
                             +HD['Hbaab23Dbaab23'][p,q,v,s] ,
                             -HD['Hbb1Dbaab3'][p,v,s,q] ))
    return result


def lucc_baba_aaaa(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( +HD['Haaaa23Dbaaaabaa67'][s,t,p,q,v,w,u,r] ,
                     -HD['Haaaa23Dbaaaabaa67'][s,u,p,q,v,w,t,r] ,
                     -HD['Haaaa23Dbaaaabaa67'][q,w,r,s,t,u,v,p] ,
                     +HD['Haaaa23Dbaaaabaa67'][q,v,r,s,t,u,w,p] ,
                     -HD['Haaaa23Dbaaaabaa67'][q,t,r,s,v,w,u,p] ,
                     +HD['Haaaa23Dbaaaabaa67'][q,u,r,s,v,w,t,p] ,
                     +HD['Haaaa23Dbaaaabaa67'][s,w,p,q,t,u,v,r] ,
                     -HD['Haaaa23Dbaaaabaa67'][s,v,p,q,t,u,w,r] ))
    result += 2 * fsum(( -HD['Haa'][q,t] * HD['Dbaaaab'][p,v,w,s,u,r] ,
                         +HD['Haa'][q,u] * HD['Dbaaaab'][p,v,w,s,t,r] ,
                         -HD['Haa'][q,w] * HD['Dbaaaab'][p,t,u,s,v,r] ,
                         +HD['Haa'][q,v] * HD['Dbaaaab'][p,t,u,s,w,r] ,
                         +HD['Haa'][s,t] * HD['Dbaaaab'][p,q,u,v,w,r] ,
                         -HD['Haa'][s,u] * HD['Dbaaaab'][p,q,t,v,w,r] ,
                         +HD['Haa'][s,w] * HD['Dbaaaab'][p,q,v,t,u,r] ,
                         -HD['Haa'][s,v] * HD['Dbaaaab'][p,q,w,t,u,r] ,
                         +HD['Haaaa13Dbaaaaaba37'][q,t,r,s,u,v,w,p] ,
                         -HD['Haaaa13Dbaaaaaba37'][q,u,r,s,t,v,w,p] ,
                         +HD['Haaaa13Dbaaaaaba37'][q,w,r,s,v,t,u,p] ,
                         -HD['Haaaa13Dbaaaaaba37'][q,v,r,s,w,t,u,p] ,
                         -HD['Haaaa13Dbaaaaaba37'][s,t,p,q,u,v,w,r] ,
                         +HD['Haaaa13Dbaaaaaba37'][s,u,p,q,t,v,w,r] ,
                         -HD['Haaaa13Dbaaaaaba37'][s,w,p,q,v,t,u,r] ,
                         +HD['Haaaa13Dbaaaaaba37'][s,v,p,q,w,t,u,r] ,
                         +HD['Haaaa3Dbaaaba5'][v,w,s,r,t,u,q,p] ,
                         -HD['Haaaa3Dbaaaba5'][v,w,q,p,t,u,s,r] ,
                         -HD['Haaaa3Dbaaaba5'][t,u,s,r,v,w,q,p] ,
                         +HD['Haaaa3Dbaaaba5'][t,u,q,p,v,w,s,r] ,
                         +HD['Habab13Dbaabaabb37'][q,t,r,s,u,v,w,p] ,
                         -HD['Habab13Dbaabaabb37'][q,u,r,s,t,v,w,p] ,
                         +HD['Habab13Dbaabaabb37'][q,w,r,s,v,t,u,p] ,
                         -HD['Habab13Dbaabaabb37'][q,v,r,s,w,t,u,p] ,
                         -HD['Habab13Dbaabaabb37'][s,t,p,q,u,v,w,r] ,
                         +HD['Habab13Dbaabaabb37'][s,u,p,q,t,v,w,r] ,
                         -HD['Habab13Dbaabaabb37'][s,w,p,q,v,t,u,r] ,
                         +HD['Habab13Dbaabaabb37'][s,v,p,q,w,t,u,r] ,
                         +HD['Hbaab13Dbaaaaaab37'][p,u,r,s,t,q,v,w] ,
                         -HD['Hbaab13Dbaaaaaab37'][r,u,p,q,t,s,v,w] ,
                         -HD['Hbaab13Dbaaaaaab37'][p,w,r,s,v,q,t,u] ,
                         +HD['Hbaab13Dbaaaaaab37'][p,v,r,s,w,q,t,u] ,
                         +HD['Hbaab13Dbaaaaaab37'][r,w,p,q,v,s,t,u] ,
                         -HD['Hbaab13Dbaaaaaab37'][r,v,p,q,w,s,t,u] ,
                         +HD['Hbaab13Dbaaaaaab37'][r,t,p,q,u,s,v,w] ,
                         -HD['Hbaab13Dbaaaaaab37'][p,t,r,s,u,q,v,w] ,
                         -HD['Hbaab23Dbaaaaaab67'][p,w,r,s,t,u,q,v] ,
                         +HD['Hbaab23Dbaaaaaab67'][p,v,r,s,t,u,q,w] ,
                         +HD['Hbaab23Dbaaaaaab67'][r,w,p,q,t,u,s,v] ,
                         -HD['Hbaab23Dbaaaaaab67'][r,v,p,q,t,u,s,w] ,
                         -HD['Hbaab23Dbaaaaaab67'][p,t,r,s,v,w,q,u] ,
                         +HD['Hbaab23Dbaaaaaab67'][p,u,r,s,v,w,q,t] ,
                         +HD['Hbaab23Dbaaaaaab67'][r,t,p,q,v,w,s,u] ,
                         -HD['Hbaab23Dbaaaaaab67'][r,u,p,q,v,w,s,t] ,
                         -HD['Hbaab3Dbaaaab5'][r,s,t,p,q,u,v,w] ,
                         +HD['Hbaab3Dbaaaab5'][r,s,u,p,q,t,v,w] ,
                         -HD['Hbaab3Dbaaaab5'][r,s,w,p,q,v,t,u] ,
                         +HD['Hbaab3Dbaaaab5'][r,s,v,p,q,w,t,u] ,
                         +HD['Hbaab3Dbaaaab5'][p,q,t,r,s,u,v,w] ,
                         -HD['Hbaab3Dbaaaab5'][p,q,u,r,s,t,v,w] ,
                         +HD['Hbaab3Dbaaaab5'][p,q,w,r,s,v,t,u] ,
                         -HD['Hbaab3Dbaaaab5'][p,q,v,r,s,w,t,u] ))
    if s==w:
        result += fsum(( +HD['Haaaa123Dbaaaabaa367'][q,r,t,u,v,p] ,
                         -HD['Haaaa23Dbaabaa45'][q,v,r,t,u,p] ,
                         -HD['Hbbbb123Dbaabaabb367'][r,p,q,v,t,u] ,
                         +HD['Hbbbb123Dbaabaabb367'][p,r,t,u,q,v] ))
        result += 2 * fsum(( -HD['Haa1Dbaaaba5'][q,r,t,u,v,p] ,
                             +HD['Habab123Dbaababab367'][q,r,t,u,v,p] ,
                             -HD['Hbaab123Dbaaaaaab367'][r,p,q,v,t,u] ,
                             +HD['Hbaab123Dbaaaaaab367'][p,r,t,u,q,v] ,
                             -HD['Hbaab13Dbaaaab25'][r,v,p,q,t,u] ,
                             -HD['Hbaab23Dbaaaab45'][p,q,r,t,u,v] ,
                             +HD['Hbaab23Dbaaaab45'][p,v,r,t,u,q] ,
                             +HD['Hbb1Dbaaaab5'][r,p,q,v,t,u] ,
                             -HD['Hbb1Dbaaaab5'][p,r,t,u,q,v] ))
    if s==v:
        result += fsum(( -HD['Haaaa123Dbaaaabaa367'][q,r,t,u,w,p] ,
                         +HD['Haaaa23Dbaabaa45'][q,w,r,t,u,p] ,
                         +HD['Hbbbb123Dbaabaabb367'][r,p,q,w,t,u] ,
                         -HD['Hbbbb123Dbaabaabb367'][p,r,t,u,q,w] ))
        result += 2 * fsum(( +HD['Haa1Dbaaaba5'][q,r,t,u,w,p] ,
                             -HD['Habab123Dbaababab367'][q,r,t,u,w,p] ,
                             +HD['Hbaab123Dbaaaaaab367'][r,p,q,w,t,u] ,
                             -HD['Hbaab123Dbaaaaaab367'][p,r,t,u,q,w] ,
                             +HD['Hbaab13Dbaaaab25'][r,w,p,q,t,u] ,
                             +HD['Hbaab23Dbaaaab45'][p,q,r,t,u,w] ,
                             -HD['Hbaab23Dbaaaab45'][p,w,r,t,u,q] ,
                             -HD['Hbb1Dbaaaab5'][r,p,q,w,t,u] ,
                             +HD['Hbb1Dbaaaab5'][p,r,t,u,q,w] ))
    if s==t:
        result += fsum(( +HD['Haaaa123Dbaaaabaa367'][q,r,v,w,u,p] ,
                         -HD['Haaaa23Dbaabaa45'][q,u,r,v,w,p] ,
                         +HD['Hbbbb123Dbaabaabb367'][p,r,v,w,q,u] ,
                         -HD['Hbbbb123Dbaabaabb367'][r,p,q,u,v,w] ))
        result += 2 * fsum(( -HD['Haa1Dbaaaba5'][q,r,v,w,u,p] ,
                             +HD['Habab123Dbaababab367'][q,r,v,w,u,p] ,
                             -HD['Hbaab123Dbaaaaaab367'][r,p,q,u,v,w] ,
                             +HD['Hbaab123Dbaaaaaab367'][p,r,v,w,q,u] ,
                             -HD['Hbaab13Dbaaaab25'][r,u,p,q,v,w] ,
                             -HD['Hbaab23Dbaaaab45'][p,q,r,v,w,u] ,
                             +HD['Hbaab23Dbaaaab45'][p,u,r,v,w,q] ,
                             -HD['Hbb1Dbaaaab5'][p,r,v,w,q,u] ,
                             +HD['Hbb1Dbaaaab5'][r,p,q,u,v,w] ))
    if s==u:
        result += fsum(( -HD['Haaaa123Dbaaaabaa367'][q,r,v,w,t,p] ,
                         +HD['Haaaa23Dbaabaa45'][q,t,r,v,w,p] ,
                         -HD['Hbbbb123Dbaabaabb367'][p,r,v,w,q,t] ,
                         +HD['Hbbbb123Dbaabaabb367'][r,p,q,t,v,w] ))
        result += 2 * fsum(( +HD['Haa1Dbaaaba5'][q,r,v,w,t,p] ,
                             -HD['Habab123Dbaababab367'][q,r,v,w,t,p] ,
                             -HD['Hbaab123Dbaaaaaab367'][p,r,v,w,q,t] ,
                             +HD['Hbaab123Dbaaaaaab367'][r,p,q,t,v,w] ,
                             +HD['Hbaab13Dbaaaab25'][r,t,p,q,v,w] ,
                             -HD['Hbaab23Dbaaaab45'][p,t,r,v,w,q] ,
                             +HD['Hbaab23Dbaaaab45'][p,q,r,v,w,t] ,
                             +HD['Hbb1Dbaaaab5'][p,r,v,w,q,t] ,
                             -HD['Hbb1Dbaaaab5'][r,p,q,t,v,w] ))
    if q==w:
        result += fsum(( -HD['Haaaa123Dbaaaabaa367'][s,p,t,u,v,r] ,
                         +HD['Haaaa23Dbaabaa45'][s,v,p,t,u,r] ,
                         +HD['Hbbbb123Dbaabaabb367'][p,r,s,v,t,u] ,
                         -HD['Hbbbb123Dbaabaabb367'][r,p,t,u,s,v] ))
        result += 2 * fsum(( +HD['Haa1Dbaaaba5'][s,p,t,u,v,r] ,
                             -HD['Habab123Dbaababab367'][s,p,t,u,v,r] ,
                             +HD['Hbaab123Dbaaaaaab367'][p,r,s,v,t,u] ,
                             -HD['Hbaab123Dbaaaaaab367'][r,p,t,u,s,v] ,
                             +HD['Hbaab13Dbaaaab25'][p,v,r,s,t,u] ,
                             -HD['Hbaab23Dbaaaab45'][r,v,p,t,u,s] ,
                             +HD['Hbaab23Dbaaaab45'][r,s,p,t,u,v] ,
                             +HD['Hbb1Dbaaaab5'][r,p,t,u,s,v] ,
                             -HD['Hbb1Dbaaaab5'][p,r,s,v,t,u] ))
    if q==v:
        result += fsum(( +HD['Haaaa123Dbaaaabaa367'][s,p,t,u,w,r] ,
                         -HD['Haaaa23Dbaabaa45'][s,w,p,t,u,r] ,
                         -HD['Hbbbb123Dbaabaabb367'][p,r,s,w,t,u] ,
                         +HD['Hbbbb123Dbaabaabb367'][r,p,t,u,s,w] ))
        result += 2 * fsum(( -HD['Haa1Dbaaaba5'][s,p,t,u,w,r] ,
                             +HD['Habab123Dbaababab367'][s,p,t,u,w,r] ,
                             -HD['Hbaab123Dbaaaaaab367'][p,r,s,w,t,u] ,
                             +HD['Hbaab123Dbaaaaaab367'][r,p,t,u,s,w] ,
                             -HD['Hbaab13Dbaaaab25'][p,w,r,s,t,u] ,
                             +HD['Hbaab23Dbaaaab45'][r,w,p,t,u,s] ,
                             -HD['Hbaab23Dbaaaab45'][r,s,p,t,u,w] ,
                             +HD['Hbb1Dbaaaab5'][p,r,s,w,t,u] ,
                             -HD['Hbb1Dbaaaab5'][r,p,t,u,s,w] ))
    if q==t:
        result += fsum(( -HD['Haaaa123Dbaaaabaa367'][s,p,v,w,u,r] ,
                         +HD['Haaaa23Dbaabaa45'][s,u,p,v,w,r] ,
                         -HD['Hbbbb123Dbaabaabb367'][r,p,v,w,s,u] ,
                         +HD['Hbbbb123Dbaabaabb367'][p,r,s,u,v,w] ))
        result += 2 * fsum(( +HD['Haa1Dbaaaba5'][s,p,v,w,u,r] ,
                             -HD['Habab123Dbaababab367'][s,p,v,w,u,r] ,
                             +HD['Hbaab123Dbaaaaaab367'][p,r,s,u,v,w] ,
                             -HD['Hbaab123Dbaaaaaab367'][r,p,v,w,s,u] ,
                             +HD['Hbaab13Dbaaaab25'][p,u,r,s,v,w] ,
                             +HD['Hbaab23Dbaaaab45'][r,s,p,v,w,u] ,
                             -HD['Hbaab23Dbaaaab45'][r,u,p,v,w,s] ,
                             +HD['Hbb1Dbaaaab5'][r,p,v,w,s,u] ,
                             -HD['Hbb1Dbaaaab5'][p,r,s,u,v,w] ))
    if q==u:
        result += fsum(( +HD['Haaaa123Dbaaaabaa367'][s,p,v,w,t,r] ,
                         -HD['Haaaa23Dbaabaa45'][s,t,p,v,w,r] ,
                         +HD['Hbbbb123Dbaabaabb367'][r,p,v,w,s,t] ,
                         -HD['Hbbbb123Dbaabaabb367'][p,r,s,t,v,w] ))
        result += 2 * fsum(( -HD['Haa1Dbaaaba5'][s,p,v,w,t,r] ,
                             +HD['Habab123Dbaababab367'][s,p,v,w,t,r] ,
                             -HD['Hbaab123Dbaaaaaab367'][p,r,s,t,v,w] ,
                             +HD['Hbaab123Dbaaaaaab367'][r,p,v,w,s,t] ,
                             -HD['Hbaab13Dbaaaab25'][p,t,r,s,v,w] ,
                             +HD['Hbaab23Dbaaaab45'][r,t,p,v,w,s] ,
                             -HD['Hbaab23Dbaaaab45'][r,s,p,v,w,t] ,
                             -HD['Hbb1Dbaaaab5'][r,p,v,w,s,t] ,
                             +HD['Hbb1Dbaaaab5'][p,r,s,t,v,w] ))
    return result


def lucc_baba_baba(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( +HD['Haaaa23Dbbaabbaa67'][q,u,r,v,s,w,p,t] ,
                     +HD['Haaaa23Dbbaabbaa67'][s,w,p,t,q,u,r,v] ,
                     -HD['Haaaa23Dbbaabbaa67'][s,u,p,v,q,w,r,t] ,
                     -HD['Haaaa23Dbbaabbaa67'][q,w,r,t,s,u,p,v] ,
                     -HD['Hbbbb23Dbbaaaabb67'][p,v,r,t,s,u,q,w] ,
                     +HD['Hbbbb23Dbbaaaabb67'][p,t,r,v,s,w,q,u] ,
                     +HD['Hbbbb23Dbbaaaabb67'][r,v,p,t,q,u,s,w] ,
                     -HD['Hbbbb23Dbbaaaabb67'][r,t,p,v,q,w,s,u] ))
    result += 2 * fsum(( -HD['Haa'][q,u] * HD['Dbbaabb'][p,v,w,s,r,t] ,
                         +HD['Haa'][s,u] * HD['Dbbaabb'][p,t,q,w,r,v] ,
                         -HD['Haa'][s,w] * HD['Dbbaabb'][p,v,q,u,r,t] ,
                         +HD['Haa'][q,w] * HD['Dbbaabb'][p,t,u,s,r,v] ,
                         -HD['Haaaa13Dbbaaabba37'][q,w,r,v,s,u,p,t] ,
                         -HD['Haaaa13Dbbaaabba37'][s,u,p,t,q,w,r,v] ,
                         +HD['Haaaa13Dbbaaabba37'][s,w,p,v,q,u,r,t] ,
                         +HD['Haaaa13Dbbaaabba37'][q,u,r,t,s,w,p,v] ,
                         +HD['Habab13Dbbababbb37'][s,w,p,v,q,u,r,t] ,
                         -HD['Habab13Dbbababbb37'][q,w,r,v,s,u,p,t] ,
                         -HD['Habab13Dbbababbb37'][s,u,p,t,q,w,r,v] ,
                         +HD['Habab13Dbbababbb37'][q,u,r,t,s,w,p,v] ,
                         -HD['Habba13Dbaababba37'][q,v,r,s,w,u,p,t] ,
                         +HD['Habba13Dbaababba37'][s,v,p,q,w,u,r,t] ,
                         -HD['Habba13Dbaababba37'][s,t,p,q,u,w,r,v] ,
                         +HD['Habba13Dbaababba37'][q,t,r,s,u,w,p,v] ,
                         -HD['Hbaab13Dbbaaaabb37'][r,u,p,t,q,s,w,v] ,
                         -HD['Hbaab13Dbbaaaabb37'][p,w,r,v,s,q,u,t] ,
                         +HD['Hbaab13Dbbaaaabb37'][r,w,p,v,q,s,u,t] ,
                         +HD['Hbaab13Dbbaaaabb37'][p,u,r,t,s,q,w,v] ,
                         -HD['Hbaab23Dbbaaabab67'][t,q,r,v,s,w,u,p] ,
                         -HD['Hbaab23Dbbaaabab67'][v,s,p,t,q,u,w,r] ,
                         +HD['Hbaab23Dbbaaabab67'][p,w,r,t,s,u,q,v] ,
                         +HD['Hbaab23Dbbaaabab67'][t,s,p,v,q,w,u,r] ,
                         -HD['Hbaab23Dbbaaabab67'][p,u,r,v,s,w,q,t] ,
                         +HD['Hbaab23Dbbaaabab67'][v,q,r,t,s,u,w,p] ,
                         -HD['Hbaab23Dbbaaabab67'][r,w,p,t,q,u,s,v] ,
                         +HD['Hbaab23Dbbaaabab67'][r,u,p,v,q,w,s,t] ,
                         +HD['Hbaab3Dbbaabb5'][r,s,u,p,t,q,w,v] ,
                         -HD['Hbaab3Dbbaabb5'][r,s,w,p,v,q,u,t] ,
                         -HD['Hbaab3Dbbaabb5'][p,q,u,r,t,s,w,v] ,
                         -HD['Hbaab3Dbbaabb5'][t,u,s,r,v,w,q,p] ,
                         +HD['Hbaab3Dbbaabb5'][p,q,w,r,v,s,u,t] ,
                         +HD['Hbaab3Dbbaabb5'][v,w,s,r,t,u,q,p] ,
                         +HD['Hbaab3Dbbaabb5'][t,u,q,p,v,w,s,r] ,
                         -HD['Hbaab3Dbbaabb5'][v,w,q,p,t,u,s,r] ,
                         -HD['Hbaab'][p,q,u,t] * HD['Dbaab'][r,s,w,v] ,
                         +HD['Hbaab'][r,s,u,t] * HD['Dbaab'][p,q,w,v] ,
                         +HD['Hbaab'][p,q,w,v] * HD['Dbaab'][r,s,u,t] ,
                         -HD['Hbaab'][r,s,w,v] * HD['Dbaab'][p,q,u,t] ,
                         +HD['Hbaba13Dbaaaaaba37'][p,t,r,s,u,q,w,v] ,
                         -HD['Hbaba13Dbaaaaaba37'][p,v,r,s,w,q,u,t] ,
                         +HD['Hbaba13Dbaaaaaba37'][r,v,p,q,w,s,u,t] ,
                         -HD['Hbaba13Dbaaaaaba37'][r,t,p,q,u,s,w,v] ,
                         +HD['Hbaba3Dbaaaba5'][r,s,t,p,q,u,w,v] ,
                         -HD['Hbaba3Dbaaaba5'][p,q,t,r,s,u,w,v] ,
                         -HD['Hbaba3Dbaaaba5'][r,s,v,p,q,w,u,t] ,
                         +HD['Hbaba3Dbaaaba5'][p,q,v,r,s,w,u,t] ,
                         +HD['Hbaba3Dbaaaba5'][t,u,p,v,q,w,s,r] ,
                         -HD['Hbaba3Dbaaaba5'][v,w,p,t,q,u,s,r] ,
                         -HD['Hbaba3Dbaaaba5'][t,u,r,v,s,w,q,p] ,
                         +HD['Hbaba3Dbaaaba5'][v,w,r,t,s,u,q,p] ,
                         -HD['Hbb'][r,v] * HD['Dbaaaab'][p,q,w,s,u,t] ,
                         +HD['Hbb'][p,v] * HD['Dbaaaab'][t,q,u,s,w,r] ,
                         -HD['Hbb'][p,t] * HD['Dbaaaab'][v,q,w,s,u,r] ,
                         +HD['Hbb'][r,t] * HD['Dbaaaab'][p,q,u,s,w,v] ,
                         +HD['Hbbbb13Dbaabaabb37'][p,t,r,s,u,q,w,v] ,
                         -HD['Hbbbb13Dbaabaabb37'][p,v,r,s,w,q,u,t] ,
                         -HD['Hbbbb13Dbaabaabb37'][r,t,p,q,u,s,w,v] ,
                         +HD['Hbbbb13Dbaabaabb37'][r,v,p,q,w,s,u,t] ))
    if p==t:
        result += fsum(( +HD['Haaaa123Dbaaaabaa367'][q,r,s,u,w,v] ,
                         -HD['Haaaa123Dbaaaabaa367'][s,v,q,w,u,r] ,
                         +HD['Haaaa23Dbaabaa45'][s,u,v,q,w,r] ,
                         -HD['Hbbbb123Dbaabaabb367'][r,v,q,w,s,u] ))
        result += 2 * fsum(( -HD['Haa1Dbaaaba5'][q,r,s,u,w,v] ,
                             +HD['Haa1Dbaaaba5'][s,v,q,w,u,r] ,
                             +HD['Haa'][q,u] * HD['Dbaab'][r,s,w,v] ,
                             +HD['Haaaa13Dbaaaba25'][q,u,r,s,w,v] ,
                             -HD['Habab123Dbaababab367'][s,v,q,w,u,r] ,
                             +HD['Habab123Dbaababab367'][q,r,s,u,w,v] ,
                             +HD['Habab13Dbababb25'][q,u,r,s,w,v] ,
                             -HD['Hbaab123Dbaaaaaab367'][r,v,q,w,s,u] ,
                             -HD['Hbaab23Dbaaaab45'][r,u,v,q,w,s] ,
                             +HD['Hbaab23Dbaaaab45'][r,s,v,q,w,u] ,
                             +HD['Hbb1Dbaaaab5'][r,v,q,w,s,u] ))
        if q==u:
            result += fsum(( +HD['Haaaa123Dbaabaa245'][s,v,w,r] ,
                             -HD['Hbbbb123Dbababb245'][r,v,w,s] ))
            result += 2 * fsum(( +HD['Haa1Dbaba3'][s,v,w,r] ,
                                 +HD['Habab123Dbabbab245'][s,v,w,r] ,
                                 -HD['Hbaab123Dbaaaab245'][r,v,w,s] ,
                                 +HD['Hbaab23Dbaab23'][r,s,v,w] ,
                                 -HD['Hbb1Dbaab3'][r,v,w,s] ))
    if s==w:
        result += fsum(( -HD['Haaaa123Dbbaabbaa367'][q,r,t,u,p,v] ,
                         +HD['Hbbbb123Dbbababbb367'][p,r,t,u,q,v] ,
                         -HD['Hbbbb123Dbbababbb367'][r,p,v,q,u,t] ,
                         +HD['Hbbbb23Dbbaabb45'][p,v,r,t,u,q] ))
        result += 2 * fsum(( +HD['Haa1Dbbabba5'][q,r,t,u,p,v] ,
                             -HD['Habab123Dbbabbbab367'][q,r,t,u,p,v] ,
                             +HD['Hbaab123Dbbaaabab367'][p,r,t,u,q,v] ,
                             -HD['Hbaab123Dbbaaabab367'][r,p,v,q,u,t] ,
                             -HD['Hbaab23Dbbabab45'][p,q,r,t,u,v] ,
                             +HD['Hbaab23Dbbabab45'][v,q,r,t,u,p] ,
                             +HD['Hbaba13Dbaaaba25'][r,v,p,q,u,t] ,
                             -HD['Hbb1Dbbaabb5'][p,r,t,u,q,v] ,
                             +HD['Hbb1Dbbaabb5'][r,p,v,q,u,t] ,
                             +HD['Hbb'][r,v] * HD['Dbaab'][p,q,u,t] ,
                             +HD['Hbbbb13Dbababb25'][r,v,p,q,u,t] ))
        if r==v:
            result += fsum(( +HD['Haaaa123Dbaabaa245'][q,t,u,p] ,
                             -HD['Hbbbb123Dbababb245'][p,t,u,q] ))
            result += 2 * fsum(( +HD['Haa1Dbaba3'][q,t,u,p] ,
                                 +HD['Habab123Dbabbab245'][q,t,u,p] ,
                                 -HD['Hbaab123Dbaaaab245'][p,t,u,q] ,
                                 +HD['Hbaab23Dbaab23'][p,q,t,u] ,
                                 -HD['Hbb1Dbaab3'][p,t,u,q] ))
    if r==v:
        result += fsum(( -HD['Haaaa123Dbaaaabaa367'][q,t,s,u,w,p] ,
                         +HD['Haaaa123Dbaaaabaa367'][s,p,q,w,u,t] ,
                         +HD['Haaaa23Dbaabaa45'][q,w,t,s,u,p] ,
                         -HD['Hbbbb123Dbaabaabb367'][p,t,s,u,q,w] ))
        result += 2 * fsum(( +HD['Haa1Dbaaaba5'][q,t,s,u,w,p] ,
                             -HD['Haa1Dbaaaba5'][s,p,q,w,u,t] ,
                             +HD['Haa'][s,w] * HD['Dbaab'][p,q,u,t] ,
                             +HD['Haaaa13Dbaaaba25'][s,w,p,q,u,t] ,
                             -HD['Habab123Dbaababab367'][q,t,s,u,w,p] ,
                             +HD['Habab123Dbaababab367'][s,p,q,w,u,t] ,
                             +HD['Habab13Dbababb25'][s,w,p,q,u,t] ,
                             -HD['Hbaab123Dbaaaaaab367'][p,t,s,u,q,w] ,
                             -HD['Hbaab23Dbaaaab45'][p,w,t,s,u,q] ,
                             +HD['Hbaab23Dbaaaab45'][p,q,t,s,u,w] ,
                             +HD['Hbb1Dbaaaab5'][p,t,s,u,q,w] ))
    if p==v:
        result += fsum(( +HD['Haaaa123Dbaaaabaa367'][s,t,q,u,w,r] ,
                         -HD['Haaaa123Dbaaaabaa367'][q,r,s,w,u,t] ,
                         -HD['Haaaa23Dbaabaa45'][s,w,t,q,u,r] ,
                         +HD['Hbbbb123Dbaabaabb367'][r,t,q,u,s,w] ))
        result += 2 * fsum(( +HD['Haa1Dbaaaba5'][q,r,s,w,u,t] ,
                             -HD['Haa1Dbaaaba5'][s,t,q,u,w,r] ,
                             -HD['Haa'][q,w] * HD['Dbaab'][r,s,u,t] ,
                             -HD['Haaaa13Dbaaaba25'][q,w,r,s,u,t] ,
                             -HD['Habab123Dbaababab367'][q,r,s,w,u,t] ,
                             +HD['Habab123Dbaababab367'][s,t,q,u,w,r] ,
                             -HD['Habab13Dbababb25'][q,w,r,s,u,t] ,
                             +HD['Hbaab123Dbaaaaaab367'][r,t,q,u,s,w] ,
                             -HD['Hbaab23Dbaaaab45'][r,s,t,q,u,w] ,
                             +HD['Hbaab23Dbaaaab45'][r,w,t,q,u,s] ,
                             -HD['Hbb1Dbaaaab5'][r,t,q,u,s,w] ))
        if q==w:
            result += fsum(( -HD['Haaaa123Dbaabaa245'][s,t,u,r] ,
                             +HD['Hbbbb123Dbababb245'][r,t,u,s] ))
            result += 2 * fsum(( -HD['Haa1Dbaba3'][s,t,u,r] ,
                                 -HD['Habab123Dbabbab245'][s,t,u,r] ,
                                 +HD['Hbaab123Dbaaaab245'][r,t,u,s] ,
                                 -HD['Hbaab23Dbaab23'][r,s,t,u] ,
                                 +HD['Hbb1Dbaab3'][r,t,u,s] ))
    if s==u:
        result += fsum(( +HD['Haaaa123Dbbaabbaa367'][q,r,v,w,p,t] ,
                         -HD['Hbbbb123Dbbababbb367'][p,r,v,w,q,t] ,
                         +HD['Hbbbb123Dbbababbb367'][r,p,t,q,w,v] ,
                         -HD['Hbbbb23Dbbaabb45'][p,t,r,v,w,q] ))
        result += 2 * fsum(( -HD['Haa1Dbbabba5'][q,r,v,w,p,t] ,
                             +HD['Habab123Dbbabbbab367'][q,r,v,w,p,t] ,
                             -HD['Hbaab123Dbbaaabab367'][p,r,v,w,q,t] ,
                             +HD['Hbaab123Dbbaaabab367'][r,p,t,q,w,v] ,
                             -HD['Hbaab23Dbbabab45'][t,q,r,v,w,p] ,
                             +HD['Hbaab23Dbbabab45'][p,q,r,v,w,t] ,
                             -HD['Hbaba13Dbaaaba25'][r,t,p,q,w,v] ,
                             +HD['Hbb1Dbbaabb5'][p,r,v,w,q,t] ,
                             -HD['Hbb1Dbbaabb5'][r,p,t,q,w,v] ,
                             -HD['Hbb'][r,t] * HD['Dbaab'][p,q,w,v] ,
                             -HD['Hbbbb13Dbababb25'][r,t,p,q,w,v] ))
        if r==t:
            result += fsum(( -HD['Haaaa123Dbaabaa245'][q,v,w,p] ,
                             +HD['Hbbbb123Dbababb245'][p,v,w,q] ))
            result += 2 * fsum(( -HD['Haa1Dbaba3'][q,v,w,p] ,
                                 -HD['Habab123Dbabbab245'][q,v,w,p] ,
                                 +HD['Hbaab123Dbaaaab245'][p,v,w,q] ,
                                 -HD['Hbaab23Dbaab23'][p,q,v,w] ,
                                 +HD['Hbb1Dbaab3'][p,v,w,q] ))
    if r==t:
        result += fsum(( -HD['Haaaa123Dbaaaabaa367'][s,p,q,u,w,v] ,
                         +HD['Haaaa123Dbaaaabaa367'][q,v,s,w,u,p] ,
                         -HD['Haaaa23Dbaabaa45'][q,u,v,s,w,p] ,
                         +HD['Hbbbb123Dbaabaabb367'][p,v,s,w,q,u] ))
        result += 2 * fsum(( -HD['Haa1Dbaaaba5'][q,v,s,w,u,p] ,
                             +HD['Haa1Dbaaaba5'][s,p,q,u,w,v] ,
                             -HD['Haa'][s,u] * HD['Dbaab'][p,q,w,v] ,
                             -HD['Haaaa13Dbaaaba25'][s,u,p,q,w,v] ,
                             -HD['Habab123Dbaababab367'][s,p,q,u,w,v] ,
                             +HD['Habab123Dbaababab367'][q,v,s,w,u,p] ,
                             -HD['Habab13Dbababb25'][s,u,p,q,w,v] ,
                             +HD['Hbaab123Dbaaaaaab367'][p,v,s,w,q,u] ,
                             -HD['Hbaab23Dbaaaab45'][p,q,v,s,w,u] ,
                             +HD['Hbaab23Dbaaaab45'][p,u,v,s,w,q] ,
                             -HD['Hbb1Dbaaaab5'][p,v,s,w,q,u] ))
    if q==w:
        result += fsum(( +HD['Haaaa123Dbbaabbaa367'][s,p,t,u,r,v] ,
                         +HD['Hbbbb123Dbbababbb367'][p,r,v,s,u,t] ,
                         -HD['Hbbbb123Dbbababbb367'][r,p,t,u,s,v] ,
                         -HD['Hbbbb23Dbbaabb45'][r,v,p,t,u,s] ))
        result += 2 * fsum(( -HD['Haa1Dbbabba5'][s,p,t,u,r,v] ,
                             +HD['Habab123Dbbabbbab367'][s,p,t,u,r,v] ,
                             +HD['Hbaab123Dbbaaabab367'][p,r,v,s,u,t] ,
                             -HD['Hbaab123Dbbaaabab367'][r,p,t,u,s,v] ,
                             +HD['Hbaab23Dbbabab45'][r,s,p,t,u,v] ,
                             -HD['Hbaab23Dbbabab45'][v,s,p,t,u,r] ,
                             -HD['Hbaba13Dbaaaba25'][p,v,r,s,u,t] ,
                             +HD['Hbb1Dbbaabb5'][r,p,t,u,s,v] ,
                             -HD['Hbb1Dbbaabb5'][p,r,v,s,u,t] ,
                             -HD['Hbb'][p,v] * HD['Dbaab'][r,s,u,t] ,
                             -HD['Hbbbb13Dbababb25'][p,v,r,s,u,t] ))
    if q==u:
        result += fsum(( -HD['Haaaa123Dbbaabbaa367'][s,p,v,w,r,t] ,
                         +HD['Hbbbb123Dbbababbb367'][r,p,v,w,s,t] ,
                         -HD['Hbbbb123Dbbababbb367'][p,r,t,s,w,v] ,
                         +HD['Hbbbb23Dbbaabb45'][r,t,p,v,w,s] ))
        result += 2 * fsum(( +HD['Haa1Dbbabba5'][s,p,v,w,r,t] ,
                             -HD['Habab123Dbbabbbab367'][s,p,v,w,r,t] ,
                             -HD['Hbaab123Dbbaaabab367'][p,r,t,s,w,v] ,
                             +HD['Hbaab123Dbbaaabab367'][r,p,v,w,s,t] ,
                             +HD['Hbaab23Dbbabab45'][t,s,p,v,w,r] ,
                             -HD['Hbaab23Dbbabab45'][r,s,p,v,w,t] ,
                             +HD['Hbaba13Dbaaaba25'][p,t,r,s,w,v] ,
                             +HD['Hbb1Dbbaabb5'][p,r,t,s,w,v] ,
                             -HD['Hbb1Dbbaabb5'][r,p,v,w,s,t] ,
                             +HD['Hbb'][p,t] * HD['Dbaab'][r,s,w,v] ,
                             +HD['Hbbbb13Dbababb25'][p,t,r,s,w,v] ))
    return result


def lucc_baba_bbbb(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( -HD['Hbbbb23Dbbbaabbb67'][p,w,r,t,u,s,q,v] ,
                     +HD['Hbbbb23Dbbbaabbb67'][p,v,r,t,u,s,q,w] ,
                     -HD['Hbbbb23Dbbbaabbb67'][p,t,r,v,w,s,q,u] ,
                     +HD['Hbbbb23Dbbbaabbb67'][p,u,r,v,w,s,q,t] ,
                     +HD['Hbbbb23Dbbbaabbb67'][r,w,p,t,u,q,s,v] ,
                     -HD['Hbbbb23Dbbbaabbb67'][r,v,p,t,u,q,s,w] ,
                     +HD['Hbbbb23Dbbbaabbb67'][r,t,p,v,w,q,s,u] ,
                     -HD['Hbbbb23Dbbbaabbb67'][r,u,p,v,w,q,s,t] ))
    result += 2 * fsum(( -HD['Habba13Dbbabbbba37'][q,w,r,v,s,p,t,u] ,
                         +HD['Habba13Dbbabbbba37'][q,v,r,w,s,p,t,u] ,
                         +HD['Habba13Dbbabbbba37'][s,t,p,u,q,r,v,w] ,
                         -HD['Habba13Dbbabbbba37'][s,u,p,t,q,r,v,w] ,
                         +HD['Habba13Dbbabbbba37'][s,w,p,v,q,r,t,u] ,
                         -HD['Habba13Dbbabbbba37'][s,v,p,w,q,r,t,u] ,
                         -HD['Habba13Dbbabbbba37'][q,t,r,u,s,p,v,w] ,
                         +HD['Habba13Dbbabbbba37'][q,u,r,t,s,p,v,w] ,
                         -HD['Hbaab23Dbbbabbab67'][t,q,r,v,w,s,p,u] ,
                         +HD['Hbaab23Dbbbabbab67'][u,q,r,v,w,s,p,t] ,
                         +HD['Hbaab23Dbbbabbab67'][w,s,p,t,u,q,r,v] ,
                         -HD['Hbaab23Dbbbabbab67'][v,s,p,t,u,q,r,w] ,
                         +HD['Hbaab23Dbbbabbab67'][t,s,p,v,w,q,r,u] ,
                         -HD['Hbaab23Dbbbabbab67'][u,s,p,v,w,q,r,t] ,
                         -HD['Hbaab23Dbbbabbab67'][w,q,r,t,u,s,p,v] ,
                         +HD['Hbaab23Dbbbabbab67'][v,q,r,t,u,s,p,w] ,
                         -HD['Hbaba13Dbbaaabba37'][r,t,p,u,q,s,v,w] ,
                         +HD['Hbaba13Dbbaaabba37'][r,u,p,t,q,s,v,w] ,
                         -HD['Hbaba13Dbbaaabba37'][r,w,p,v,q,s,t,u] ,
                         +HD['Hbaba13Dbbaaabba37'][r,v,p,w,q,s,t,u] ,
                         +HD['Hbaba13Dbbaaabba37'][p,t,r,u,s,q,v,w] ,
                         -HD['Hbaba13Dbbaaabba37'][p,u,r,t,s,q,v,w] ,
                         +HD['Hbaba13Dbbaaabba37'][p,w,r,v,s,q,t,u] ,
                         -HD['Hbaba13Dbbaaabba37'][p,v,r,w,s,q,t,u] ,
                         +HD['Hbaba3Dbbabba5'][r,s,t,p,u,q,v,w] ,
                         -HD['Hbaba3Dbbabba5'][r,s,u,p,t,q,v,w] ,
                         +HD['Hbaba3Dbbabba5'][r,s,w,p,v,q,t,u] ,
                         -HD['Hbaba3Dbbabba5'][r,s,v,p,w,q,t,u] ,
                         -HD['Hbaba3Dbbabba5'][p,q,t,r,u,s,v,w] ,
                         +HD['Hbaba3Dbbabba5'][p,q,u,r,t,s,v,w] ,
                         -HD['Hbaba3Dbbabba5'][p,q,w,r,v,s,t,u] ,
                         +HD['Hbaba3Dbbabba5'][p,q,v,r,w,s,t,u] ,
                         -HD['Hbb'][p,t] * HD['Dbbaabb'][v,w,q,s,r,u] ,
                         +HD['Hbb'][p,u] * HD['Dbbaabb'][v,w,q,s,r,t] ,
                         -HD['Hbb'][p,w] * HD['Dbbaabb'][t,u,q,s,r,v] ,
                         +HD['Hbb'][p,v] * HD['Dbbaabb'][t,u,q,s,r,w] ,
                         +HD['Hbb'][r,t] * HD['Dbbaabb'][p,u,q,s,v,w] ,
                         -HD['Hbb'][r,u] * HD['Dbbaabb'][p,t,q,s,v,w] ,
                         +HD['Hbb'][r,w] * HD['Dbbaabb'][p,v,q,s,t,u] ,
                         -HD['Hbb'][r,v] * HD['Dbbaabb'][p,w,q,s,t,u] ,
                         +HD['Hbbbb13Dbbababbb37'][p,t,r,u,s,q,v,w] ,
                         -HD['Hbbbb13Dbbababbb37'][p,u,r,t,s,q,v,w] ,
                         +HD['Hbbbb13Dbbababbb37'][p,w,r,v,s,q,t,u] ,
                         -HD['Hbbbb13Dbbababbb37'][p,v,r,w,s,q,t,u] ,
                         -HD['Hbbbb13Dbbababbb37'][r,t,p,u,q,s,v,w] ,
                         +HD['Hbbbb13Dbbababbb37'][r,u,p,t,q,s,v,w] ,
                         -HD['Hbbbb13Dbbababbb37'][r,w,p,v,q,s,t,u] ,
                         +HD['Hbbbb13Dbbababbb37'][r,v,p,w,q,s,t,u] ,
                         -HD['Hbbbb3Dbbaabb5'][t,u,p,v,w,q,s,r] ,
                         -HD['Hbbbb3Dbbaabb5'][v,w,r,t,u,s,q,p] ,
                         +HD['Hbbbb3Dbbaabb5'][t,u,r,v,w,s,q,p] ,
                         +HD['Hbbbb3Dbbaabb5'][v,w,p,t,u,q,s,r] ))
    if p==t:
        result += fsum(( +HD['Haaaa123Dbbaabbaa367'][q,r,u,s,v,w] ,
                         -HD['Haaaa123Dbbaabbaa367'][s,v,w,q,r,u] ,
                         +HD['Hbbbb123Dbbababbb367'][r,v,w,q,s,u] ,
                         +HD['Hbbbb23Dbbaabb45'][r,u,v,w,q,s] ))
        result += 2 * fsum(( -HD['Haa1Dbbabba5'][q,r,u,s,v,w] ,
                             +HD['Haa1Dbbabba5'][s,v,w,q,r,u] ,
                             +HD['Habab123Dbbabbbab367'][q,r,u,s,v,w] ,
                             -HD['Habab123Dbbabbbab367'][s,v,w,q,r,u] ,
                             -HD['Habba13Dbabbba25'][q,u,r,s,v,w] ,
                             +HD['Hbaab123Dbbaaabab367'][r,v,w,q,s,u] ,
                             -HD['Hbaab23Dbbabab45'][r,s,v,w,q,u] ,
                             +HD['Hbaab23Dbbabab45'][u,s,v,w,q,r] ,
                             -HD['Hbb1Dbbaabb5'][r,v,w,q,s,u] ))
    if p==u:
        result += fsum(( -HD['Haaaa123Dbbaabbaa367'][q,r,t,s,v,w] ,
                         +HD['Haaaa123Dbbaabbaa367'][s,v,w,q,r,t] ,
                         -HD['Hbbbb123Dbbababbb367'][r,v,w,q,s,t] ,
                         -HD['Hbbbb23Dbbaabb45'][r,t,v,w,q,s] ))
        result += 2 * fsum(( +HD['Haa1Dbbabba5'][q,r,t,s,v,w] ,
                             -HD['Haa1Dbbabba5'][s,v,w,q,r,t] ,
                             -HD['Habab123Dbbabbbab367'][q,r,t,s,v,w] ,
                             +HD['Habab123Dbbabbbab367'][s,v,w,q,r,t] ,
                             +HD['Habba13Dbabbba25'][q,t,r,s,v,w] ,
                             -HD['Hbaab123Dbbaaabab367'][r,v,w,q,s,t] ,
                             -HD['Hbaab23Dbbabab45'][t,s,v,w,q,r] ,
                             +HD['Hbaab23Dbbabab45'][r,s,v,w,q,t] ,
                             +HD['Hbb1Dbbaabb5'][r,v,w,q,s,t] ))
    if r==w:
        result += fsum(( +HD['Haaaa123Dbbaabbaa367'][q,t,u,s,p,v] ,
                         -HD['Haaaa123Dbbaabbaa367'][s,p,v,q,t,u] ,
                         -HD['Hbbbb123Dbbababbb367'][p,t,u,s,q,v] ,
                         -HD['Hbbbb23Dbbaabb45'][p,v,t,u,s,q] ))
        result += 2 * fsum(( -HD['Haa1Dbbabba5'][q,t,u,s,p,v] ,
                             +HD['Haa1Dbbabba5'][s,p,v,q,t,u] ,
                             +HD['Habab123Dbbabbbab367'][q,t,u,s,p,v] ,
                             -HD['Habab123Dbbabbbab367'][s,p,v,q,t,u] ,
                             +HD['Habba13Dbabbba25'][s,v,p,q,t,u] ,
                             -HD['Hbaab123Dbbaaabab367'][p,t,u,s,q,v] ,
                             -HD['Hbaab23Dbbabab45'][v,q,t,u,s,p] ,
                             +HD['Hbaab23Dbbabab45'][p,q,t,u,s,v] ,
                             +HD['Hbb1Dbbaabb5'][p,t,u,s,q,v] ))
    if r==v:
        result += fsum(( +HD['Haaaa123Dbbaabbaa367'][s,p,w,q,t,u] ,
                         -HD['Haaaa123Dbbaabbaa367'][q,t,u,s,p,w] ,
                         +HD['Hbbbb123Dbbababbb367'][p,t,u,s,q,w] ,
                         +HD['Hbbbb23Dbbaabb45'][p,w,t,u,s,q] ))
        result += 2 * fsum(( +HD['Haa1Dbbabba5'][q,t,u,s,p,w] ,
                             -HD['Haa1Dbbabba5'][s,p,w,q,t,u] ,
                             -HD['Habab123Dbbabbbab367'][q,t,u,s,p,w] ,
                             +HD['Habab123Dbbabbbab367'][s,p,w,q,t,u] ,
                             -HD['Habba13Dbabbba25'][s,w,p,q,t,u] ,
                             +HD['Hbaab123Dbbaaabab367'][p,t,u,s,q,w] ,
                             +HD['Hbaab23Dbbabab45'][w,q,t,u,s,p] ,
                             -HD['Hbaab23Dbbabab45'][p,q,t,u,s,w] ,
                             -HD['Hbb1Dbbaabb5'][p,t,u,s,q,w] ))
    if p==w:
        result += fsum(( +HD['Haaaa123Dbbaabbaa367'][q,r,v,s,t,u] ,
                         -HD['Haaaa123Dbbaabbaa367'][s,t,u,q,r,v] ,
                         +HD['Hbbbb123Dbbababbb367'][r,t,u,q,s,v] ,
                         +HD['Hbbbb23Dbbaabb45'][r,v,t,u,q,s] ))
        result += 2 * fsum(( -HD['Haa1Dbbabba5'][q,r,v,s,t,u] ,
                             +HD['Haa1Dbbabba5'][s,t,u,q,r,v] ,
                             -HD['Habab123Dbbabbbab367'][s,t,u,q,r,v] ,
                             +HD['Habab123Dbbabbbab367'][q,r,v,s,t,u] ,
                             -HD['Habba13Dbabbba25'][q,v,r,s,t,u] ,
                             +HD['Hbaab123Dbbaaabab367'][r,t,u,q,s,v] ,
                             -HD['Hbaab23Dbbabab45'][r,s,t,u,q,v] ,
                             +HD['Hbaab23Dbbabab45'][v,s,t,u,q,r] ,
                             -HD['Hbb1Dbbaabb5'][r,t,u,q,s,v] ))
    if p==v:
        result += fsum(( -HD['Haaaa123Dbbaabbaa367'][q,r,w,s,t,u] ,
                         +HD['Haaaa123Dbbaabbaa367'][s,t,u,q,r,w] ,
                         -HD['Hbbbb123Dbbababbb367'][r,t,u,q,s,w] ,
                         -HD['Hbbbb23Dbbaabb45'][r,w,t,u,q,s] ))
        result += 2 * fsum(( +HD['Haa1Dbbabba5'][q,r,w,s,t,u] ,
                             -HD['Haa1Dbbabba5'][s,t,u,q,r,w] ,
                             +HD['Habab123Dbbabbbab367'][s,t,u,q,r,w] ,
                             -HD['Habab123Dbbabbbab367'][q,r,w,s,t,u] ,
                             +HD['Habba13Dbabbba25'][q,w,r,s,t,u] ,
                             -HD['Hbaab123Dbbaaabab367'][r,t,u,q,s,w] ,
                             +HD['Hbaab23Dbbabab45'][r,s,t,u,q,w] ,
                             -HD['Hbaab23Dbbabab45'][w,s,t,u,q,r] ,
                             +HD['Hbb1Dbbaabb5'][r,t,u,q,s,w] ))
    if r==t:
        result += fsum(( +HD['Haaaa123Dbbaabbaa367'][q,v,w,s,p,u] ,
                         -HD['Haaaa123Dbbaabbaa367'][s,p,u,q,v,w] ,
                         -HD['Hbbbb123Dbbababbb367'][p,v,w,s,q,u] ,
                         -HD['Hbbbb23Dbbaabb45'][p,u,v,w,s,q] ))
        result += 2 * fsum(( -HD['Haa1Dbbabba5'][q,v,w,s,p,u] ,
                             +HD['Haa1Dbbabba5'][s,p,u,q,v,w] ,
                             -HD['Habab123Dbbabbbab367'][s,p,u,q,v,w] ,
                             +HD['Habab123Dbbabbbab367'][q,v,w,s,p,u] ,
                             +HD['Habba13Dbabbba25'][s,u,p,q,v,w] ,
                             -HD['Hbaab123Dbbaaabab367'][p,v,w,s,q,u] ,
                             -HD['Hbaab23Dbbabab45'][u,q,v,w,s,p] ,
                             +HD['Hbaab23Dbbabab45'][p,q,v,w,s,u] ,
                             +HD['Hbb1Dbbaabb5'][p,v,w,s,q,u] ))
    if r==u:
        result += fsum(( -HD['Haaaa123Dbbaabbaa367'][q,v,w,s,p,t] ,
                         +HD['Haaaa123Dbbaabbaa367'][s,p,t,q,v,w] ,
                         +HD['Hbbbb123Dbbababbb367'][p,v,w,s,q,t] ,
                         +HD['Hbbbb23Dbbaabb45'][p,t,v,w,s,q] ))
        result += 2 * fsum(( +HD['Haa1Dbbabba5'][q,v,w,s,p,t] ,
                             -HD['Haa1Dbbabba5'][s,p,t,q,v,w] ,
                             +HD['Habab123Dbbabbbab367'][s,p,t,q,v,w] ,
                             -HD['Habab123Dbbabbbab367'][q,v,w,s,p,t] ,
                             -HD['Habba13Dbabbba25'][s,t,p,q,v,w] ,
                             +HD['Hbaab123Dbbaaabab367'][p,v,w,s,q,t] ,
                             +HD['Hbaab23Dbbabab45'][t,q,v,w,s,p] ,
                             -HD['Hbaab23Dbbabab45'][p,q,v,w,s,t] ,
                             -HD['Hbb1Dbbaabb5'][p,v,w,s,q,t] ))
    return result


def lucc_bbbb(HD, *e):
    (p,q,r,s) = e
    result = 0
    result += fsum(( -HD['Hbbbb123Dbbbbbb245'][p,r,s,q] ,
                     +HD['Hbbbb123Dbbbbbb245'][q,r,s,p] ,
                     -HD['Hbbbb123Dbbbbbb245'][s,p,q,r] ,
                     +HD['Hbbbb123Dbbbbbb245'][r,p,q,s] ,
                     +HD['Hbbbb23Dbbbb23'][p,q,r,s] ,
                     -HD['Hbbbb23Dbbbb23'][r,s,p,q] ))
    result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][p,r,s,q] ,
                         +HD['Hbaab123Dbbabab245'][q,r,s,p] ,
                         -HD['Hbaab123Dbbabab245'][s,p,q,r] ,
                         +HD['Hbaab123Dbbabab245'][r,p,q,s] ,
                         -HD['Hbb1Dbbbb3'][p,r,s,q] ,
                         +HD['Hbb1Dbbbb3'][q,r,s,p] ,
                         -HD['Hbb1Dbbbb3'][s,p,q,r] ,
                         +HD['Hbb1Dbbbb3'][r,p,q,s] ))
    return result


def lucc_bbbb_aa(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += 2 * fsum(( +HD['Hbaab13Dbbaabb25'][p,t,r,s,v,q] ,
                         -HD['Hbaab13Dbbaabb25'][p,v,r,s,t,q] ,
                         -HD['Hbaab13Dbbaabb25'][q,t,r,s,v,p] ,
                         +HD['Hbaab13Dbbaabb25'][q,v,r,s,t,p] ,
                         +HD['Hbaab13Dbbaabb25'][s,t,p,q,v,r] ,
                         -HD['Hbaab13Dbbaabb25'][s,v,p,q,t,r] ,
                         -HD['Hbaab13Dbbaabb25'][r,t,p,q,v,s] ,
                         +HD['Hbaab13Dbbaabb25'][r,v,p,q,t,s] ,
                         +HD['Hbaab23Dbbabab45'][p,v,r,s,t,q] ,
                         -HD['Hbaab23Dbbabab45'][p,t,r,s,v,q] ,
                         -HD['Hbaab23Dbbabab45'][q,v,r,s,t,p] ,
                         +HD['Hbaab23Dbbabab45'][q,t,r,s,v,p] ,
                         +HD['Hbaab23Dbbabab45'][s,v,p,q,t,r] ,
                         -HD['Hbaab23Dbbabab45'][s,t,p,q,v,r] ,
                         -HD['Hbaab23Dbbabab45'][r,v,p,q,t,s] ,
                         +HD['Hbaab23Dbbabab45'][r,t,p,q,v,s] ))
    return result


def lucc_bbbb_bb(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += fsum(( -HD['Hbbbb23Dbbbbbb45'][r,v,p,q,t,s] ,
                     +HD['Hbbbb23Dbbbbbb45'][r,t,p,q,v,s] ,
                     +HD['Hbbbb23Dbbbbbb45'][p,v,r,s,t,q] ,
                     -HD['Hbbbb23Dbbbbbb45'][p,t,r,s,v,q] ,
                     -HD['Hbbbb23Dbbbbbb45'][q,v,r,s,t,p] ,
                     +HD['Hbbbb23Dbbbbbb45'][q,t,r,s,v,p] ,
                     +HD['Hbbbb23Dbbbbbb45'][s,v,p,q,t,r] ,
                     -HD['Hbbbb23Dbbbbbb45'][s,t,p,q,v,r] ))
    result += 2 * fsum(( -HD['Hbaba13Dbbabba25'][p,t,r,s,q,v] ,
                         +HD['Hbaba13Dbbabba25'][p,v,r,s,q,t] ,
                         +HD['Hbaba13Dbbabba25'][q,t,r,s,p,v] ,
                         -HD['Hbaba13Dbbabba25'][q,v,r,s,p,t] ,
                         -HD['Hbaba13Dbbabba25'][s,t,p,q,r,v] ,
                         +HD['Hbaba13Dbbabba25'][s,v,p,q,r,t] ,
                         +HD['Hbaba13Dbbabba25'][r,t,p,q,s,v] ,
                         -HD['Hbaba13Dbbabba25'][r,v,p,q,s,t] ,
                         -HD['Hbb'][p,t] * HD['Dbbbb'][q,v,r,s] ,
                         +HD['Hbb'][p,v] * HD['Dbbbb'][q,t,r,s] ,
                         +HD['Hbb'][q,t] * HD['Dbbbb'][p,v,r,s] ,
                         -HD['Hbb'][q,v] * HD['Dbbbb'][p,t,r,s] ,
                         -HD['Hbb'][s,t] * HD['Dbbbb'][p,q,r,v] ,
                         +HD['Hbb'][s,v] * HD['Dbbbb'][p,q,r,t] ,
                         +HD['Hbb'][r,t] * HD['Dbbbb'][p,q,s,v] ,
                         -HD['Hbb'][r,v] * HD['Dbbbb'][p,q,s,t] ,
                         -HD['Hbbbb13Dbbbbbb25'][p,t,r,s,q,v] ,
                         +HD['Hbbbb13Dbbbbbb25'][p,v,r,s,q,t] ,
                         +HD['Hbbbb13Dbbbbbb25'][q,t,r,s,p,v] ,
                         -HD['Hbbbb13Dbbbbbb25'][q,v,r,s,p,t] ,
                         -HD['Hbbbb13Dbbbbbb25'][s,t,p,q,r,v] ,
                         +HD['Hbbbb13Dbbbbbb25'][s,v,p,q,r,t] ,
                         +HD['Hbbbb13Dbbbbbb25'][r,t,p,q,s,v] ,
                         -HD['Hbbbb13Dbbbbbb25'][r,v,p,q,s,t] ,
                         +HD['Hbbbb3Dbbbb3'][p,q,t,r,s,v] ,
                         -HD['Hbbbb3Dbbbb3'][p,q,v,r,s,t] ,
                         -HD['Hbbbb3Dbbbb3'][r,s,t,p,q,v] ,
                         +HD['Hbbbb3Dbbbb3'][r,s,v,p,q,t] ))
    if q==t:
        result += fsum(( -HD['Hbbbb123Dbbbbbb245'][p,r,s,v] ,
                         -HD['Hbbbb123Dbbbbbb245'][s,p,v,r] ,
                         +HD['Hbbbb123Dbbbbbb245'][r,p,v,s] ,
                         -HD['Hbbbb23Dbbbb23'][r,s,p,v] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][p,r,s,v] ,
                             -HD['Hbaab123Dbbabab245'][s,p,v,r] ,
                             +HD['Hbaab123Dbbabab245'][r,p,v,s] ,
                             -HD['Hbb1Dbbbb3'][p,r,s,v] ,
                             -HD['Hbb1Dbbbb3'][s,p,v,r] ,
                             +HD['Hbb1Dbbbb3'][r,p,v,s] ))
    if r==v:
        result += fsum(( +HD['Hbbbb123Dbbbbbb245'][s,p,q,t] ,
                         -HD['Hbbbb123Dbbbbbb245'][p,s,t,q] ,
                         +HD['Hbbbb123Dbbbbbb245'][q,s,t,p] ,
                         +HD['Hbbbb23Dbbbb23'][p,q,s,t] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][p,s,t,q] ,
                             +HD['Hbaab123Dbbabab245'][q,s,t,p] ,
                             +HD['Hbaab123Dbbabab245'][s,p,q,t] ,
                             -HD['Hbb1Dbbbb3'][p,s,t,q] ,
                             +HD['Hbb1Dbbbb3'][q,s,t,p] ,
                             +HD['Hbb1Dbbbb3'][s,p,q,t] ))
    if s==v:
        result += fsum(( -HD['Hbbbb123Dbbbbbb245'][q,r,t,p] ,
                         +HD['Hbbbb123Dbbbbbb245'][p,r,t,q] ,
                         -HD['Hbbbb123Dbbbbbb245'][r,p,q,t] ,
                         -HD['Hbbbb23Dbbbb23'][p,q,r,t] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][p,r,t,q] ,
                             -HD['Hbaab123Dbbabab245'][q,r,t,p] ,
                             -HD['Hbaab123Dbbabab245'][r,p,q,t] ,
                             +HD['Hbb1Dbbbb3'][p,r,t,q] ,
                             -HD['Hbb1Dbbbb3'][q,r,t,p] ,
                             -HD['Hbb1Dbbbb3'][r,p,q,t] ))
    if q==v:
        result += fsum(( +HD['Hbbbb123Dbbbbbb245'][s,p,t,r] ,
                         +HD['Hbbbb123Dbbbbbb245'][p,r,s,t] ,
                         -HD['Hbbbb123Dbbbbbb245'][r,p,t,s] ,
                         +HD['Hbbbb23Dbbbb23'][r,s,p,t] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][p,r,s,t] ,
                             +HD['Hbaab123Dbbabab245'][s,p,t,r] ,
                             -HD['Hbaab123Dbbabab245'][r,p,t,s] ,
                             +HD['Hbb1Dbbbb3'][p,r,s,t] ,
                             -HD['Hbb1Dbbbb3'][r,p,t,s] ,
                             +HD['Hbb1Dbbbb3'][s,p,t,r] ))
    if r==t:
        result += fsum(( -HD['Hbbbb123Dbbbbbb245'][s,p,q,v] ,
                         +HD['Hbbbb123Dbbbbbb245'][p,s,v,q] ,
                         -HD['Hbbbb123Dbbbbbb245'][q,s,v,p] ,
                         -HD['Hbbbb23Dbbbb23'][p,q,s,v] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][p,s,v,q] ,
                             -HD['Hbaab123Dbbabab245'][q,s,v,p] ,
                             -HD['Hbaab123Dbbabab245'][s,p,q,v] ,
                             +HD['Hbb1Dbbbb3'][p,s,v,q] ,
                             -HD['Hbb1Dbbbb3'][q,s,v,p] ,
                             -HD['Hbb1Dbbbb3'][s,p,q,v] ))
    if s==t:
        result += fsum(( +HD['Hbbbb123Dbbbbbb245'][r,p,q,v] ,
                         -HD['Hbbbb123Dbbbbbb245'][p,r,v,q] ,
                         +HD['Hbbbb123Dbbbbbb245'][q,r,v,p] ,
                         +HD['Hbbbb23Dbbbb23'][p,q,r,v] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][p,r,v,q] ,
                             +HD['Hbaab123Dbbabab245'][q,r,v,p] ,
                             +HD['Hbaab123Dbbabab245'][r,p,q,v] ,
                             -HD['Hbb1Dbbbb3'][p,r,v,q] ,
                             +HD['Hbb1Dbbbb3'][q,r,v,p] ,
                             +HD['Hbb1Dbbbb3'][r,p,q,v] ))
    if p==t:
        result += fsum(( +HD['Hbbbb123Dbbbbbb245'][s,q,v,r] ,
                         +HD['Hbbbb123Dbbbbbb245'][q,r,s,v] ,
                         -HD['Hbbbb123Dbbbbbb245'][r,q,v,s] ,
                         +HD['Hbbbb23Dbbbb23'][r,s,q,v] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][q,r,s,v] ,
                             +HD['Hbaab123Dbbabab245'][s,q,v,r] ,
                             -HD['Hbaab123Dbbabab245'][r,q,v,s] ,
                             +HD['Hbb1Dbbbb3'][q,r,s,v] ,
                             -HD['Hbb1Dbbbb3'][r,q,v,s] ,
                             +HD['Hbb1Dbbbb3'][s,q,v,r] ))
    if p==v:
        result += fsum(( -HD['Hbbbb123Dbbbbbb245'][s,q,t,r] ,
                         +HD['Hbbbb123Dbbbbbb245'][r,q,t,s] ,
                         -HD['Hbbbb123Dbbbbbb245'][q,r,s,t] ,
                         -HD['Hbbbb23Dbbbb23'][r,s,q,t] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][q,r,s,t] ,
                             -HD['Hbaab123Dbbabab245'][s,q,t,r] ,
                             +HD['Hbaab123Dbbabab245'][r,q,t,s] ,
                             +HD['Hbb1Dbbbb3'][r,q,t,s] ,
                             -HD['Hbb1Dbbbb3'][q,r,s,t] ,
                             -HD['Hbb1Dbbbb3'][s,q,t,r] ))
    return result


def lucc_bbbb_aaaa(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += 2 * fsum(( -HD['Hbaab13Dbbaaaabb37'][p,t,r,s,u,v,w,q] ,
                         +HD['Hbaab13Dbbaaaabb37'][p,u,r,s,t,v,w,q] ,
                         -HD['Hbaab13Dbbaaaabb37'][p,w,r,s,v,t,u,q] ,
                         +HD['Hbaab13Dbbaaaabb37'][p,v,r,s,w,t,u,q] ,
                         +HD['Hbaab13Dbbaaaabb37'][q,t,r,s,u,v,w,p] ,
                         -HD['Hbaab13Dbbaaaabb37'][q,u,r,s,t,v,w,p] ,
                         +HD['Hbaab13Dbbaaaabb37'][q,w,r,s,v,t,u,p] ,
                         -HD['Hbaab13Dbbaaaabb37'][q,v,r,s,w,t,u,p] ,
                         -HD['Hbaab13Dbbaaaabb37'][s,t,p,q,u,v,w,r] ,
                         +HD['Hbaab13Dbbaaaabb37'][s,u,p,q,t,v,w,r] ,
                         -HD['Hbaab13Dbbaaaabb37'][s,w,p,q,v,t,u,r] ,
                         +HD['Hbaab13Dbbaaaabb37'][s,v,p,q,w,t,u,r] ,
                         +HD['Hbaab13Dbbaaaabb37'][r,t,p,q,u,v,w,s] ,
                         -HD['Hbaab13Dbbaaaabb37'][r,u,p,q,t,v,w,s] ,
                         +HD['Hbaab13Dbbaaaabb37'][r,w,p,q,v,t,u,s] ,
                         -HD['Hbaab13Dbbaaaabb37'][r,v,p,q,w,t,u,s] ,
                         +HD['Hbaab23Dbbaaabab67'][p,w,r,s,t,u,v,q] ,
                         -HD['Hbaab23Dbbaaabab67'][p,v,r,s,t,u,w,q] ,
                         +HD['Hbaab23Dbbaaabab67'][p,t,r,s,v,w,u,q] ,
                         -HD['Hbaab23Dbbaaabab67'][p,u,r,s,v,w,t,q] ,
                         -HD['Hbaab23Dbbaaabab67'][q,w,r,s,t,u,v,p] ,
                         +HD['Hbaab23Dbbaaabab67'][q,v,r,s,t,u,w,p] ,
                         -HD['Hbaab23Dbbaaabab67'][q,t,r,s,v,w,u,p] ,
                         +HD['Hbaab23Dbbaaabab67'][q,u,r,s,v,w,t,p] ,
                         +HD['Hbaab23Dbbaaabab67'][s,w,p,q,t,u,v,r] ,
                         -HD['Hbaab23Dbbaaabab67'][s,v,p,q,t,u,w,r] ,
                         +HD['Hbaab23Dbbaaabab67'][s,t,p,q,v,w,u,r] ,
                         -HD['Hbaab23Dbbaaabab67'][s,u,p,q,v,w,t,r] ,
                         -HD['Hbaab23Dbbaaabab67'][r,w,p,q,t,u,v,s] ,
                         +HD['Hbaab23Dbbaaabab67'][r,v,p,q,t,u,w,s] ,
                         -HD['Hbaab23Dbbaaabab67'][r,t,p,q,v,w,u,s] ,
                         +HD['Hbaab23Dbbaaabab67'][r,u,p,q,v,w,t,s] ))
    return result


def lucc_bbbb_baba(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( +HD['Hbbbb23Dbbbaabbb67'][q,v,r,s,t,u,w,p] ,
                     -HD['Hbbbb23Dbbbaabbb67'][q,t,r,s,v,w,u,p] ,
                     -HD['Hbbbb23Dbbbaabbb67'][s,v,p,q,t,u,w,r] ,
                     +HD['Hbbbb23Dbbbaabbb67'][s,t,p,q,v,w,u,r] ,
                     +HD['Hbbbb23Dbbbaabbb67'][r,v,p,q,t,u,w,s] ,
                     -HD['Hbbbb23Dbbbaabbb67'][r,t,p,q,v,w,u,s] ,
                     -HD['Hbbbb23Dbbbaabbb67'][p,v,r,s,t,u,w,q] ,
                     +HD['Hbbbb23Dbbbaabbb67'][p,t,r,s,v,w,u,q] ))
    result += 2 * fsum(( +HD['Hbaab13Dbbbaabbb37'][s,u,p,q,t,w,r,v] ,
                         -HD['Hbaab13Dbbbaabbb37'][s,w,p,q,v,u,r,t] ,
                         -HD['Hbaab13Dbbbaabbb37'][r,u,p,q,t,w,s,v] ,
                         +HD['Hbaab13Dbbbaabbb37'][r,w,p,q,v,u,s,t] ,
                         +HD['Hbaab13Dbbbaabbb37'][p,u,r,s,t,w,q,v] ,
                         -HD['Hbaab13Dbbbaabbb37'][p,w,r,s,v,u,q,t] ,
                         -HD['Hbaab13Dbbbaabbb37'][q,u,r,s,t,w,p,v] ,
                         +HD['Hbaab13Dbbbaabbb37'][q,w,r,s,v,u,p,t] ,
                         -HD['Hbaab23Dbbbabbab67'][s,w,p,q,t,u,r,v] ,
                         +HD['Hbaab23Dbbbabbab67'][s,u,p,q,v,w,r,t] ,
                         +HD['Hbaab23Dbbbabbab67'][r,w,p,q,t,u,s,v] ,
                         -HD['Hbaab23Dbbbabbab67'][r,u,p,q,v,w,s,t] ,
                         -HD['Hbaab23Dbbbabbab67'][p,w,r,s,t,u,q,v] ,
                         +HD['Hbaab23Dbbbabbab67'][p,u,r,s,v,w,q,t] ,
                         +HD['Hbaab23Dbbbabbab67'][q,w,r,s,t,u,p,v] ,
                         -HD['Hbaab23Dbbbabbab67'][q,u,r,s,v,w,p,t] ,
                         -HD['Hbaba13Dbbaaabba37'][s,t,p,q,u,w,r,v] ,
                         +HD['Hbaba13Dbbaaabba37'][s,v,p,q,w,u,r,t] ,
                         +HD['Hbaba13Dbbaaabba37'][r,t,p,q,u,w,s,v] ,
                         -HD['Hbaba13Dbbaaabba37'][r,v,p,q,w,u,s,t] ,
                         -HD['Hbaba13Dbbaaabba37'][p,t,r,s,u,w,q,v] ,
                         +HD['Hbaba13Dbbaaabba37'][p,v,r,s,w,u,q,t] ,
                         +HD['Hbaba13Dbbaaabba37'][q,t,r,s,u,w,p,v] ,
                         -HD['Hbaba13Dbbaaabba37'][q,v,r,s,w,u,p,t] ,
                         +HD['Hbaba3Dbbabba5'][t,u,s,r,v,w,p,q] ,
                         -HD['Hbaba3Dbbabba5'][v,w,s,r,t,u,p,q] ,
                         -HD['Hbaba3Dbbabba5'][t,u,r,s,v,w,p,q] ,
                         +HD['Hbaba3Dbbabba5'][v,w,r,s,t,u,p,q] ,
                         +HD['Hbaba3Dbbabba5'][t,u,p,q,v,w,r,s] ,
                         -HD['Hbaba3Dbbabba5'][v,w,p,q,t,u,r,s] ,
                         -HD['Hbaba3Dbbabba5'][t,u,q,p,v,w,r,s] ,
                         +HD['Hbaba3Dbbabba5'][v,w,q,p,t,u,r,s] ,
                         +HD['Hbb'][p,t] * HD['Dbbaabb'][q,v,w,u,r,s] ,
                         -HD['Hbb'][p,v] * HD['Dbbaabb'][q,t,u,w,r,s] ,
                         -HD['Hbb'][q,t] * HD['Dbbaabb'][p,v,w,u,r,s] ,
                         +HD['Hbb'][q,v] * HD['Dbbaabb'][p,t,u,w,r,s] ,
                         +HD['Hbb'][s,t] * HD['Dbbaabb'][p,q,u,w,r,v] ,
                         -HD['Hbb'][s,v] * HD['Dbbaabb'][p,q,w,u,r,t] ,
                         -HD['Hbb'][r,t] * HD['Dbbaabb'][p,q,u,w,s,v] ,
                         +HD['Hbb'][r,v] * HD['Dbbaabb'][p,q,w,u,s,t] ,
                         +HD['Hbbbb13Dbbababbb37'][q,t,r,s,u,w,p,v] ,
                         -HD['Hbbbb13Dbbababbb37'][q,v,r,s,w,u,p,t] ,
                         -HD['Hbbbb13Dbbababbb37'][s,t,p,q,u,w,r,v] ,
                         +HD['Hbbbb13Dbbababbb37'][s,v,p,q,w,u,r,t] ,
                         +HD['Hbbbb13Dbbababbb37'][r,t,p,q,u,w,s,v] ,
                         -HD['Hbbbb13Dbbababbb37'][r,v,p,q,w,u,s,t] ,
                         -HD['Hbbbb13Dbbababbb37'][p,t,r,s,u,w,q,v] ,
                         +HD['Hbbbb13Dbbababbb37'][p,v,r,s,w,u,q,t] ,
                         -HD['Hbbbb3Dbbaabb5'][p,q,t,r,s,u,w,v] ,
                         +HD['Hbbbb3Dbbaabb5'][p,q,v,r,s,w,u,t] ,
                         +HD['Hbbbb3Dbbaabb5'][r,s,t,p,q,u,w,v] ,
                         -HD['Hbbbb3Dbbaabb5'][r,s,v,p,q,w,u,t] ))
    if q==t:
        result += fsum(( -HD['Hbbbb123Dbbababbb367'][p,r,s,u,w,v] ,
                         +HD['Hbbbb123Dbbababbb367'][r,p,v,w,u,s] ,
                         -HD['Hbbbb123Dbbababbb367'][s,p,v,w,u,r] ,
                         +HD['Hbbbb23Dbbaabb45'][r,s,p,v,w,u] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbaaabab367'][p,r,s,u,w,v] ,
                             +HD['Hbaab123Dbbaaabab367'][r,p,v,w,u,s] ,
                             -HD['Hbaab123Dbbaaabab367'][s,p,v,w,u,r] ,
                             -HD['Hbaab13Dbbaabb25'][p,u,r,s,w,v] ,
                             -HD['Hbaab23Dbbabab45'][r,u,p,v,w,s] ,
                             +HD['Hbaab23Dbbabab45'][s,u,p,v,w,r] ,
                             +HD['Hbb1Dbbaabb5'][p,r,s,u,w,v] ,
                             -HD['Hbb1Dbbaabb5'][r,p,v,w,u,s] ,
                             +HD['Hbb1Dbbaabb5'][s,p,v,w,u,r] ))
    if r==v:
        result += fsum(( +HD['Hbbbb123Dbbababbb367'][q,s,t,u,w,p] ,
                         -HD['Hbbbb123Dbbababbb367'][p,s,t,u,w,q] ,
                         +HD['Hbbbb123Dbbababbb367'][s,p,q,w,u,t] ,
                         -HD['Hbbbb23Dbbaabb45'][p,q,s,t,u,w] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbaaabab367'][q,s,t,u,w,p] ,
                             -HD['Hbaab123Dbbaaabab367'][p,s,t,u,w,q] ,
                             +HD['Hbaab123Dbbaaabab367'][s,p,q,w,u,t] ,
                             +HD['Hbaab13Dbbaabb25'][s,w,p,q,u,t] ,
                             +HD['Hbaab23Dbbabab45'][p,w,s,t,u,q] ,
                             -HD['Hbaab23Dbbabab45'][q,w,s,t,u,p] ,
                             +HD['Hbb1Dbbaabb5'][p,s,t,u,w,q] ,
                             -HD['Hbb1Dbbaabb5'][q,s,t,u,w,p] ,
                             -HD['Hbb1Dbbaabb5'][s,p,q,w,u,t] ))
    if s==v:
        result += fsum(( -HD['Hbbbb123Dbbababbb367'][r,p,q,w,u,t] ,
                         -HD['Hbbbb123Dbbababbb367'][q,r,t,u,w,p] ,
                         +HD['Hbbbb123Dbbababbb367'][p,r,t,u,w,q] ,
                         +HD['Hbbbb23Dbbaabb45'][p,q,r,t,u,w] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbaaabab367'][p,r,t,u,w,q] ,
                             -HD['Hbaab123Dbbaaabab367'][r,p,q,w,u,t] ,
                             -HD['Hbaab123Dbbaaabab367'][q,r,t,u,w,p] ,
                             -HD['Hbaab13Dbbaabb25'][r,w,p,q,u,t] ,
                             -HD['Hbaab23Dbbabab45'][p,w,r,t,u,q] ,
                             +HD['Hbaab23Dbbabab45'][q,w,r,t,u,p] ,
                             -HD['Hbb1Dbbaabb5'][p,r,t,u,w,q] ,
                             +HD['Hbb1Dbbaabb5'][r,p,q,w,u,t] ,
                             +HD['Hbb1Dbbaabb5'][q,r,t,u,w,p] ))
    if q==v:
        result += fsum(( +HD['Hbbbb123Dbbababbb367'][s,p,t,u,w,r] ,
                         +HD['Hbbbb123Dbbababbb367'][p,r,s,w,u,t] ,
                         -HD['Hbbbb123Dbbababbb367'][r,p,t,u,w,s] ,
                         -HD['Hbbbb23Dbbaabb45'][r,s,p,t,u,w] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbaaabab367'][r,p,t,u,w,s] ,
                             +HD['Hbaab123Dbbaaabab367'][p,r,s,w,u,t] ,
                             +HD['Hbaab123Dbbaaabab367'][s,p,t,u,w,r] ,
                             +HD['Hbaab13Dbbaabb25'][p,w,r,s,u,t] ,
                             +HD['Hbaab23Dbbabab45'][r,w,p,t,u,s] ,
                             -HD['Hbaab23Dbbabab45'][s,w,p,t,u,r] ,
                             +HD['Hbb1Dbbaabb5'][r,p,t,u,w,s] ,
                             -HD['Hbb1Dbbaabb5'][p,r,s,w,u,t] ,
                             -HD['Hbb1Dbbaabb5'][s,p,t,u,w,r] ))
    if r==t:
        result += fsum(( -HD['Hbbbb123Dbbababbb367'][s,p,q,u,w,v] ,
                         -HD['Hbbbb123Dbbababbb367'][q,s,v,w,u,p] ,
                         +HD['Hbbbb123Dbbababbb367'][p,s,v,w,u,q] ,
                         +HD['Hbbbb23Dbbaabb45'][p,q,s,v,w,u] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbaaabab367'][q,s,v,w,u,p] ,
                             -HD['Hbaab123Dbbaaabab367'][s,p,q,u,w,v] ,
                             +HD['Hbaab123Dbbaaabab367'][p,s,v,w,u,q] ,
                             -HD['Hbaab13Dbbaabb25'][s,u,p,q,w,v] ,
                             +HD['Hbaab23Dbbabab45'][q,u,s,v,w,p] ,
                             -HD['Hbaab23Dbbabab45'][p,u,s,v,w,q] ,
                             +HD['Hbb1Dbbaabb5'][s,p,q,u,w,v] ,
                             -HD['Hbb1Dbbaabb5'][p,s,v,w,u,q] ,
                             +HD['Hbb1Dbbaabb5'][q,s,v,w,u,p] ))
    if s==t:
        result += fsum(( -HD['Hbbbb123Dbbababbb367'][p,r,v,w,u,q] ,
                         +HD['Hbbbb123Dbbababbb367'][r,p,q,u,w,v] ,
                         +HD['Hbbbb123Dbbababbb367'][q,r,v,w,u,p] ,
                         -HD['Hbbbb23Dbbaabb45'][p,q,r,v,w,u] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbaaabab367'][r,p,q,u,w,v] ,
                             +HD['Hbaab123Dbbaaabab367'][q,r,v,w,u,p] ,
                             -HD['Hbaab123Dbbaaabab367'][p,r,v,w,u,q] ,
                             +HD['Hbaab13Dbbaabb25'][r,u,p,q,w,v] ,
                             -HD['Hbaab23Dbbabab45'][q,u,r,v,w,p] ,
                             +HD['Hbaab23Dbbabab45'][p,u,r,v,w,q] ,
                             +HD['Hbb1Dbbaabb5'][p,r,v,w,u,q] ,
                             -HD['Hbb1Dbbaabb5'][q,r,v,w,u,p] ,
                             -HD['Hbb1Dbbaabb5'][r,p,q,u,w,v] ))
    if p==t:
        result += fsum(( +HD['Hbbbb123Dbbababbb367'][q,r,s,u,w,v] ,
                         -HD['Hbbbb123Dbbababbb367'][r,q,v,w,u,s] ,
                         +HD['Hbbbb123Dbbababbb367'][s,q,v,w,u,r] ,
                         -HD['Hbbbb23Dbbaabb45'][r,s,q,v,w,u] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbaaabab367'][r,q,v,w,u,s] ,
                             +HD['Hbaab123Dbbaaabab367'][q,r,s,u,w,v] ,
                             +HD['Hbaab123Dbbaaabab367'][s,q,v,w,u,r] ,
                             +HD['Hbaab13Dbbaabb25'][q,u,r,s,w,v] ,
                             +HD['Hbaab23Dbbabab45'][r,u,q,v,w,s] ,
                             -HD['Hbaab23Dbbabab45'][s,u,q,v,w,r] ,
                             -HD['Hbb1Dbbaabb5'][q,r,s,u,w,v] ,
                             +HD['Hbb1Dbbaabb5'][r,q,v,w,u,s] ,
                             -HD['Hbb1Dbbaabb5'][s,q,v,w,u,r] ))
    if p==v:
        result += fsum(( +HD['Hbbbb123Dbbababbb367'][r,q,t,u,w,s] ,
                         -HD['Hbbbb123Dbbababbb367'][s,q,t,u,w,r] ,
                         -HD['Hbbbb123Dbbababbb367'][q,r,s,w,u,t] ,
                         +HD['Hbbbb23Dbbaabb45'][r,s,q,t,u,w] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbaaabab367'][q,r,s,w,u,t] ,
                             +HD['Hbaab123Dbbaaabab367'][r,q,t,u,w,s] ,
                             -HD['Hbaab123Dbbaaabab367'][s,q,t,u,w,r] ,
                             -HD['Hbaab13Dbbaabb25'][q,w,r,s,u,t] ,
                             -HD['Hbaab23Dbbabab45'][r,w,q,t,u,s] ,
                             +HD['Hbaab23Dbbabab45'][s,w,q,t,u,r] ,
                             +HD['Hbb1Dbbaabb5'][q,r,s,w,u,t] ,
                             +HD['Hbb1Dbbaabb5'][s,q,t,u,w,r] ,
                             -HD['Hbb1Dbbaabb5'][r,q,t,u,w,s] ))
    return result


def lucc_bbbb_bbbb(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( -HD['Hbbbb23Dbbbbbbbb67'][s,w,p,q,t,u,r,v] ,
                     +HD['Hbbbb23Dbbbbbbbb67'][s,v,p,q,t,u,r,w] ,
                     -HD['Hbbbb23Dbbbbbbbb67'][s,t,p,q,v,w,r,u] ,
                     -HD['Hbbbb23Dbbbbbbbb67'][p,w,r,s,t,u,q,v] ,
                     +HD['Hbbbb23Dbbbbbbbb67'][p,v,r,s,t,u,q,w] ,
                     +HD['Hbbbb23Dbbbbbbbb67'][s,u,p,q,v,w,r,t] ,
                     +HD['Hbbbb23Dbbbbbbbb67'][r,w,p,q,t,u,s,v] ,
                     -HD['Hbbbb23Dbbbbbbbb67'][p,t,r,s,v,w,q,u] ,
                     +HD['Hbbbb23Dbbbbbbbb67'][p,u,r,s,v,w,q,t] ,
                     -HD['Hbbbb23Dbbbbbbbb67'][r,v,p,q,t,u,s,w] ,
                     +HD['Hbbbb23Dbbbbbbbb67'][r,t,p,q,v,w,s,u] ,
                     +HD['Hbbbb23Dbbbbbbbb67'][q,w,r,s,t,u,p,v] ,
                     -HD['Hbbbb23Dbbbbbbbb67'][q,v,r,s,t,u,p,w] ,
                     -HD['Hbbbb23Dbbbbbbbb67'][r,u,p,q,v,w,s,t] ,
                     +HD['Hbbbb23Dbbbbbbbb67'][q,t,r,s,v,w,p,u] ,
                     -HD['Hbbbb23Dbbbbbbbb67'][q,u,r,s,v,w,p,t] ))
    result += 2 * fsum(( -HD['Hbaba13Dbbbabbba37'][p,t,r,s,u,q,v,w] ,
                         +HD['Hbaba13Dbbbabbba37'][p,u,r,s,t,q,v,w] ,
                         -HD['Hbaba13Dbbbabbba37'][p,w,r,s,v,q,t,u] ,
                         +HD['Hbaba13Dbbbabbba37'][p,v,r,s,w,q,t,u] ,
                         +HD['Hbaba13Dbbbabbba37'][q,t,r,s,u,p,v,w] ,
                         -HD['Hbaba13Dbbbabbba37'][q,u,r,s,t,p,v,w] ,
                         +HD['Hbaba13Dbbbabbba37'][q,w,r,s,v,p,t,u] ,
                         -HD['Hbaba13Dbbbabbba37'][q,v,r,s,w,p,t,u] ,
                         -HD['Hbaba13Dbbbabbba37'][s,t,p,q,u,r,v,w] ,
                         +HD['Hbaba13Dbbbabbba37'][s,u,p,q,t,r,v,w] ,
                         -HD['Hbaba13Dbbbabbba37'][s,w,p,q,v,r,t,u] ,
                         +HD['Hbaba13Dbbbabbba37'][s,v,p,q,w,r,t,u] ,
                         +HD['Hbaba13Dbbbabbba37'][r,t,p,q,u,s,v,w] ,
                         -HD['Hbaba13Dbbbabbba37'][r,u,p,q,t,s,v,w] ,
                         +HD['Hbaba13Dbbbabbba37'][r,w,p,q,v,s,t,u] ,
                         -HD['Hbaba13Dbbbabbba37'][r,v,p,q,w,s,t,u] ,
                         +HD['Hbb'][p,t] * HD['Dbbbbbb'][q,v,w,r,s,u] ,
                         -HD['Hbb'][p,u] * HD['Dbbbbbb'][q,v,w,r,s,t] ,
                         +HD['Hbb'][p,w] * HD['Dbbbbbb'][q,t,u,r,s,v] ,
                         -HD['Hbb'][p,v] * HD['Dbbbbbb'][q,t,u,r,s,w] ,
                         -HD['Hbb'][q,t] * HD['Dbbbbbb'][p,v,w,r,s,u] ,
                         +HD['Hbb'][q,u] * HD['Dbbbbbb'][p,v,w,r,s,t] ,
                         -HD['Hbb'][q,w] * HD['Dbbbbbb'][p,t,u,r,s,v] ,
                         +HD['Hbb'][q,v] * HD['Dbbbbbb'][p,t,u,r,s,w] ,
                         +HD['Hbb'][s,t] * HD['Dbbbbbb'][p,q,u,r,v,w] ,
                         -HD['Hbb'][s,u] * HD['Dbbbbbb'][p,q,t,r,v,w] ,
                         +HD['Hbb'][s,w] * HD['Dbbbbbb'][p,q,v,r,t,u] ,
                         -HD['Hbb'][s,v] * HD['Dbbbbbb'][p,q,w,r,t,u] ,
                         -HD['Hbb'][r,t] * HD['Dbbbbbb'][p,q,u,s,v,w] ,
                         +HD['Hbb'][r,u] * HD['Dbbbbbb'][p,q,t,s,v,w] ,
                         -HD['Hbb'][r,w] * HD['Dbbbbbb'][p,q,v,s,t,u] ,
                         +HD['Hbb'][r,v] * HD['Dbbbbbb'][p,q,w,s,t,u] ,
                         -HD['Hbbbb13Dbbbbbbbb37'][r,u,p,q,t,s,v,w] ,
                         +HD['Hbbbb13Dbbbbbbbb37'][r,w,p,q,v,s,t,u] ,
                         -HD['Hbbbb13Dbbbbbbbb37'][r,v,p,q,w,s,t,u] ,
                         -HD['Hbbbb13Dbbbbbbbb37'][p,t,r,s,u,q,v,w] ,
                         +HD['Hbbbb13Dbbbbbbbb37'][p,u,r,s,t,q,v,w] ,
                         -HD['Hbbbb13Dbbbbbbbb37'][p,w,r,s,v,q,t,u] ,
                         +HD['Hbbbb13Dbbbbbbbb37'][p,v,r,s,w,q,t,u] ,
                         +HD['Hbbbb13Dbbbbbbbb37'][q,t,r,s,u,p,v,w] ,
                         -HD['Hbbbb13Dbbbbbbbb37'][q,u,r,s,t,p,v,w] ,
                         +HD['Hbbbb13Dbbbbbbbb37'][q,w,r,s,v,p,t,u] ,
                         -HD['Hbbbb13Dbbbbbbbb37'][q,v,r,s,w,p,t,u] ,
                         -HD['Hbbbb13Dbbbbbbbb37'][s,t,p,q,u,r,v,w] ,
                         +HD['Hbbbb13Dbbbbbbbb37'][s,u,p,q,t,r,v,w] ,
                         -HD['Hbbbb13Dbbbbbbbb37'][s,w,p,q,v,r,t,u] ,
                         +HD['Hbbbb13Dbbbbbbbb37'][s,v,p,q,w,r,t,u] ,
                         +HD['Hbbbb13Dbbbbbbbb37'][r,t,p,q,u,s,v,w] ,
                         +HD['Hbbbb3Dbbbbbb5'][p,q,t,r,s,u,v,w] ,
                         -HD['Hbbbb3Dbbbbbb5'][p,q,u,r,s,t,v,w] ,
                         +HD['Hbbbb3Dbbbbbb5'][v,w,r,s,t,u,p,q] ,
                         +HD['Hbbbb3Dbbbbbb5'][p,q,w,r,s,v,t,u] ,
                         -HD['Hbbbb3Dbbbbbb5'][p,q,v,r,s,w,t,u] ,
                         +HD['Hbbbb3Dbbbbbb5'][t,u,p,q,v,w,r,s] ,
                         -HD['Hbbbb3Dbbbbbb5'][v,w,p,q,t,u,r,s] ,
                         -HD['Hbbbb3Dbbbbbb5'][t,u,q,p,v,w,r,s] ,
                         +HD['Hbbbb3Dbbbbbb5'][v,w,q,p,t,u,r,s] ,
                         -HD['Hbbbb3Dbbbbbb5'][r,s,t,p,q,u,v,w] ,
                         +HD['Hbbbb3Dbbbbbb5'][r,s,u,p,q,t,v,w] ,
                         -HD['Hbbbb3Dbbbbbb5'][r,s,w,p,q,v,t,u] ,
                         +HD['Hbbbb3Dbbbbbb5'][r,s,v,p,q,w,t,u] ,
                         +HD['Hbbbb3Dbbbbbb5'][t,u,s,r,v,w,p,q] ,
                         -HD['Hbbbb3Dbbbbbb5'][v,w,s,r,t,u,p,q] ,
                         -HD['Hbbbb3Dbbbbbb5'][t,u,r,s,v,w,p,q] ,
                         -HD['Hbbbb'][p,q,t,u] * HD['Dbbbb'][r,s,v,w] ,
                         +HD['Hbbbb'][r,s,t,u] * HD['Dbbbb'][p,q,v,w] ,
                         +HD['Hbbbb'][p,q,v,w] * HD['Dbbbb'][r,s,t,u] ,
                         -HD['Hbbbb'][r,s,v,w] * HD['Dbbbb'][p,q,t,u] ))
    if q==u:
        result += fsum(( -HD['Hbbbb123Dbbbbbbbb367'][s,p,v,w,r,t] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][p,r,s,t,v,w] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][r,p,v,w,s,t] ,
                         +HD['Hbbbb23Dbbbbbb45'][r,t,p,v,w,s] ,
                         -HD['Hbbbb23Dbbbbbb45'][r,s,p,v,w,t] ,
                         -HD['Hbbbb23Dbbbbbb45'][s,t,p,v,w,r] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbbabbab367'][r,p,v,w,s,t] ,
                             -HD['Hbaab123Dbbbabbab367'][p,r,s,t,v,w] ,
                             -HD['Hbaab123Dbbbabbab367'][s,p,v,w,r,t] ,
                             -HD['Hbaba13Dbbabba25'][p,t,r,s,v,w] ,
                             +HD['Hbb1Dbbbbbb5'][p,r,s,t,v,w] ,
                             +HD['Hbb1Dbbbbbb5'][s,p,v,w,r,t] ,
                             -HD['Hbb1Dbbbbbb5'][r,p,v,w,s,t] ,
                             -HD['Hbb'][p,t] * HD['Dbbbb'][r,s,v,w] ,
                             -HD['Hbbbb13Dbbbbbb25'][p,t,r,s,v,w] ))
        if p==t:
            result += fsum(( +HD['Hbbbb123Dbbbbbb245'][s,v,w,r] ,
                             -HD['Hbbbb123Dbbbbbb245'][r,v,w,s] ,
                             +HD['Hbbbb23Dbbbb23'][r,s,v,w] ))
            result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][s,v,w,r] ,
                                 -HD['Hbaab123Dbbabab245'][r,v,w,s] ,
                                 +HD['Hbb1Dbbbb3'][s,v,w,r] ,
                                 -HD['Hbb1Dbbbb3'][r,v,w,s] ))
    if q==t:
        result += fsum(( +HD['Hbbbb123Dbbbbbbbb367'][s,p,v,w,r,u] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][p,r,s,u,v,w] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][r,p,v,w,s,u] ,
                         +HD['Hbbbb23Dbbbbbb45'][r,s,p,v,w,u] ,
                         +HD['Hbbbb23Dbbbbbb45'][s,u,p,v,w,r] ,
                         -HD['Hbbbb23Dbbbbbb45'][r,u,p,v,w,s] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbbabbab367'][r,p,v,w,s,u] ,
                             +HD['Hbaab123Dbbbabbab367'][p,r,s,u,v,w] ,
                             +HD['Hbaab123Dbbbabbab367'][s,p,v,w,r,u] ,
                             +HD['Hbaba13Dbbabba25'][p,u,r,s,v,w] ,
                             -HD['Hbb1Dbbbbbb5'][p,r,s,u,v,w] ,
                             -HD['Hbb1Dbbbbbb5'][s,p,v,w,r,u] ,
                             +HD['Hbb1Dbbbbbb5'][r,p,v,w,s,u] ,
                             +HD['Hbb'][p,u] * HD['Dbbbb'][r,s,v,w] ,
                             +HD['Hbbbb13Dbbbbbb25'][p,u,r,s,v,w] ))
        if p==u:
            result += fsum(( -HD['Hbbbb123Dbbbbbb245'][s,v,w,r] ,
                             +HD['Hbbbb123Dbbbbbb245'][r,v,w,s] ,
                             -HD['Hbbbb23Dbbbb23'][r,s,v,w] ))
            result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][s,v,w,r] ,
                                 +HD['Hbaab123Dbbabab245'][r,v,w,s] ,
                                 -HD['Hbb1Dbbbb3'][s,v,w,r] ,
                                 +HD['Hbb1Dbbbb3'][r,v,w,s] ))
    if s==v:
        result += fsum(( -HD['Hbbbb123Dbbbbbbbb367'][p,r,t,u,q,w] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][q,r,t,u,p,w] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][r,p,q,w,t,u] ,
                         -HD['Hbbbb23Dbbbbbb45'][p,w,r,t,u,q] ,
                         +HD['Hbbbb23Dbbbbbb45'][q,w,r,t,u,p] ,
                         +HD['Hbbbb23Dbbbbbb45'][p,q,r,t,u,w] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbbabbab367'][r,p,q,w,t,u] ,
                             +HD['Hbaab123Dbbbabbab367'][q,r,t,u,p,w] ,
                             -HD['Hbaab123Dbbbabbab367'][p,r,t,u,q,w] ,
                             +HD['Hbaba13Dbbabba25'][r,w,p,q,t,u] ,
                             -HD['Hbb1Dbbbbbb5'][r,p,q,w,t,u] ,
                             -HD['Hbb1Dbbbbbb5'][q,r,t,u,p,w] ,
                             +HD['Hbb1Dbbbbbb5'][p,r,t,u,q,w] ,
                             +HD['Hbb'][r,w] * HD['Dbbbb'][p,q,t,u] ,
                             +HD['Hbbbb13Dbbbbbb25'][r,w,p,q,t,u] ))
        if r==w:
            result += fsum(( +HD['Hbbbb123Dbbbbbb245'][p,t,u,q] ,
                             -HD['Hbbbb123Dbbbbbb245'][q,t,u,p] ,
                             -HD['Hbbbb23Dbbbb23'][p,q,t,u] ))
            result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][p,t,u,q] ,
                                 -HD['Hbaab123Dbbabab245'][q,t,u,p] ,
                                 +HD['Hbb1Dbbbb3'][p,t,u,q] ,
                                 -HD['Hbb1Dbbbb3'][q,t,u,p] ))
    if r==w:
        result += fsum(( +HD['Hbbbb123Dbbbbbbbb367'][q,s,t,u,p,v] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][s,p,q,v,t,u] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][p,s,t,u,q,v] ,
                         +HD['Hbbbb23Dbbbbbb45'][q,v,s,t,u,p] ,
                         +HD['Hbbbb23Dbbbbbb45'][p,q,s,t,u,v] ,
                         -HD['Hbbbb23Dbbbbbb45'][p,v,s,t,u,q] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbbabbab367'][s,p,q,v,t,u] ,
                             -HD['Hbaab123Dbbbabbab367'][p,s,t,u,q,v] ,
                             +HD['Hbaab123Dbbbabbab367'][q,s,t,u,p,v] ,
                             +HD['Hbaba13Dbbabba25'][s,v,p,q,t,u] ,
                             +HD['Hbb1Dbbbbbb5'][p,s,t,u,q,v] ,
                             -HD['Hbb1Dbbbbbb5'][q,s,t,u,p,v] ,
                             -HD['Hbb1Dbbbbbb5'][s,p,q,v,t,u] ,
                             +HD['Hbb'][s,v] * HD['Dbbbb'][p,q,t,u] ,
                             +HD['Hbbbb13Dbbbbbb25'][s,v,p,q,t,u] ))
    if s==w:
        result += fsum(( +HD['Hbbbb123Dbbbbbbbb367'][p,r,t,u,q,v] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][q,r,t,u,p,v] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][r,p,q,v,t,u] ,
                         -HD['Hbbbb23Dbbbbbb45'][q,v,r,t,u,p] ,
                         -HD['Hbbbb23Dbbbbbb45'][p,q,r,t,u,v] ,
                         +HD['Hbbbb23Dbbbbbb45'][p,v,r,t,u,q] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbbabbab367'][q,r,t,u,p,v] ,
                             +HD['Hbaab123Dbbbabbab367'][p,r,t,u,q,v] ,
                             -HD['Hbaab123Dbbbabbab367'][r,p,q,v,t,u] ,
                             -HD['Hbaba13Dbbabba25'][r,v,p,q,t,u] ,
                             +HD['Hbb1Dbbbbbb5'][r,p,q,v,t,u] ,
                             +HD['Hbb1Dbbbbbb5'][q,r,t,u,p,v] ,
                             -HD['Hbb1Dbbbbbb5'][p,r,t,u,q,v] ,
                             -HD['Hbb'][r,v] * HD['Dbbbb'][p,q,t,u] ,
                             -HD['Hbbbb13Dbbbbbb25'][r,v,p,q,t,u] ))
        if r==v:
            result += fsum(( +HD['Hbbbb123Dbbbbbb245'][q,t,u,p] ,
                             -HD['Hbbbb123Dbbbbbb245'][p,t,u,q] ,
                             +HD['Hbbbb23Dbbbb23'][p,q,t,u] ))
            result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][q,t,u,p] ,
                                 -HD['Hbaab123Dbbabab245'][p,t,u,q] ,
                                 +HD['Hbb1Dbbbb3'][q,t,u,p] ,
                                 -HD['Hbb1Dbbbb3'][p,t,u,q] ))
    if r==v:
        result += fsum(( -HD['Hbbbb123Dbbbbbbbb367'][q,s,t,u,p,w] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][s,p,q,w,t,u] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][p,s,t,u,q,w] ,
                         -HD['Hbbbb23Dbbbbbb45'][q,w,s,t,u,p] ,
                         -HD['Hbbbb23Dbbbbbb45'][p,q,s,t,u,w] ,
                         +HD['Hbbbb23Dbbbbbb45'][p,w,s,t,u,q] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbbabbab367'][q,s,t,u,p,w] ,
                             -HD['Hbaab123Dbbbabbab367'][s,p,q,w,t,u] ,
                             +HD['Hbaab123Dbbbabbab367'][p,s,t,u,q,w] ,
                             -HD['Hbaba13Dbbabba25'][s,w,p,q,t,u] ,
                             +HD['Hbb1Dbbbbbb5'][s,p,q,w,t,u] ,
                             +HD['Hbb1Dbbbbbb5'][q,s,t,u,p,w] ,
                             -HD['Hbb1Dbbbbbb5'][p,s,t,u,q,w] ,
                             -HD['Hbb'][s,w] * HD['Dbbbb'][p,q,t,u] ,
                             -HD['Hbbbb13Dbbbbbb25'][s,w,p,q,t,u] ))
    if q==v:
        result += fsum(( -HD['Hbbbb123Dbbbbbbbb367'][s,p,t,u,r,w] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][r,p,t,u,s,w] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][p,r,s,w,t,u] ,
                         +HD['Hbbbb23Dbbbbbb45'][r,w,p,t,u,s] ,
                         -HD['Hbbbb23Dbbbbbb45'][r,s,p,t,u,w] ,
                         -HD['Hbbbb23Dbbbbbb45'][s,w,p,t,u,r] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbbabbab367'][s,p,t,u,r,w] ,
                             -HD['Hbaab123Dbbbabbab367'][p,r,s,w,t,u] ,
                             +HD['Hbaab123Dbbbabbab367'][r,p,t,u,s,w] ,
                             -HD['Hbaba13Dbbabba25'][p,w,r,s,t,u] ,
                             +HD['Hbb1Dbbbbbb5'][s,p,t,u,r,w] ,
                             +HD['Hbb1Dbbbbbb5'][p,r,s,w,t,u] ,
                             -HD['Hbb1Dbbbbbb5'][r,p,t,u,s,w] ,
                             -HD['Hbb'][p,w] * HD['Dbbbb'][r,s,t,u] ,
                             -HD['Hbbbb13Dbbbbbb25'][p,w,r,s,t,u] ))
        if p==w:
            result += fsum(( +HD['Hbbbb123Dbbbbbb245'][s,t,u,r] ,
                             -HD['Hbbbb123Dbbbbbb245'][r,t,u,s] ,
                             +HD['Hbbbb23Dbbbb23'][r,s,t,u] ))
            result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][r,t,u,s] ,
                                 +HD['Hbaab123Dbbabab245'][s,t,u,r] ,
                                 -HD['Hbb1Dbbbb3'][r,t,u,s] ,
                                 +HD['Hbb1Dbbbb3'][s,t,u,r] ))
    if q==w:
        result += fsum(( -HD['Hbbbb123Dbbbbbbbb367'][r,p,t,u,s,v] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][p,r,s,v,t,u] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][s,p,t,u,r,v] ,
                         -HD['Hbbbb23Dbbbbbb45'][r,v,p,t,u,s] ,
                         +HD['Hbbbb23Dbbbbbb45'][r,s,p,t,u,v] ,
                         +HD['Hbbbb23Dbbbbbb45'][s,v,p,t,u,r] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbbabbab367'][s,p,t,u,r,v] ,
                             +HD['Hbaab123Dbbbabbab367'][p,r,s,v,t,u] ,
                             -HD['Hbaab123Dbbbabbab367'][r,p,t,u,s,v] ,
                             +HD['Hbaba13Dbbabba25'][p,v,r,s,t,u] ,
                             -HD['Hbb1Dbbbbbb5'][s,p,t,u,r,v] ,
                             -HD['Hbb1Dbbbbbb5'][p,r,s,v,t,u] ,
                             +HD['Hbb1Dbbbbbb5'][r,p,t,u,s,v] ,
                             +HD['Hbb'][p,v] * HD['Dbbbb'][r,s,t,u] ,
                             +HD['Hbbbb13Dbbbbbb25'][p,v,r,s,t,u] ))
        if p==v:
            result += fsum(( +HD['Hbbbb123Dbbbbbb245'][r,t,u,s] ,
                             -HD['Hbbbb123Dbbbbbb245'][s,t,u,r] ,
                             -HD['Hbbbb23Dbbbb23'][r,s,t,u] ))
            result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][r,t,u,s] ,
                                 -HD['Hbaab123Dbbabab245'][s,t,u,r] ,
                                 +HD['Hbb1Dbbbb3'][r,t,u,s] ,
                                 -HD['Hbb1Dbbbb3'][s,t,u,r] ))
    if s==u:
        result += fsum(( -HD['Hbbbb123Dbbbbbbbb367'][p,r,v,w,q,t] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][q,r,v,w,p,t] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][r,p,q,t,v,w] ,
                         +HD['Hbbbb23Dbbbbbb45'][p,q,r,v,w,t] ,
                         -HD['Hbbbb23Dbbbbbb45'][p,t,r,v,w,q] ,
                         +HD['Hbbbb23Dbbbbbb45'][q,t,r,v,w,p] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbbabbab367'][r,p,q,t,v,w] ,
                             +HD['Hbaab123Dbbbabbab367'][q,r,v,w,p,t] ,
                             -HD['Hbaab123Dbbbabbab367'][p,r,v,w,q,t] ,
                             +HD['Hbaba13Dbbabba25'][r,t,p,q,v,w] ,
                             -HD['Hbb1Dbbbbbb5'][r,p,q,t,v,w] ,
                             -HD['Hbb1Dbbbbbb5'][q,r,v,w,p,t] ,
                             +HD['Hbb1Dbbbbbb5'][p,r,v,w,q,t] ,
                             +HD['Hbb'][r,t] * HD['Dbbbb'][p,q,v,w] ,
                             +HD['Hbbbb13Dbbbbbb25'][r,t,p,q,v,w] ))
        if r==t:
            result += fsum(( +HD['Hbbbb123Dbbbbbb245'][p,v,w,q] ,
                             -HD['Hbbbb123Dbbbbbb245'][q,v,w,p] ,
                             -HD['Hbbbb23Dbbbb23'][p,q,v,w] ))
            result += 2 * fsum(( -HD['Hbaab123Dbbabab245'][q,v,w,p] ,
                                 +HD['Hbaab123Dbbabab245'][p,v,w,q] ,
                                 -HD['Hbb1Dbbbb3'][q,v,w,p] ,
                                 +HD['Hbb1Dbbbb3'][p,v,w,q] ))
    if r==t:
        result += fsum(( -HD['Hbbbb123Dbbbbbbbb367'][p,s,v,w,q,u] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][q,s,v,w,p,u] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][s,p,q,u,v,w] ,
                         -HD['Hbbbb23Dbbbbbb45'][p,u,s,v,w,q] ,
                         +HD['Hbbbb23Dbbbbbb45'][p,q,s,v,w,u] ,
                         +HD['Hbbbb23Dbbbbbb45'][q,u,s,v,w,p] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbbabbab367'][q,s,v,w,p,u] ,
                             -HD['Hbaab123Dbbbabbab367'][p,s,v,w,q,u] ,
                             +HD['Hbaab123Dbbbabbab367'][s,p,q,u,v,w] ,
                             +HD['Hbaba13Dbbabba25'][s,u,p,q,v,w] ,
                             -HD['Hbb1Dbbbbbb5'][q,s,v,w,p,u] ,
                             +HD['Hbb1Dbbbbbb5'][p,s,v,w,q,u] ,
                             -HD['Hbb1Dbbbbbb5'][s,p,q,u,v,w] ,
                             +HD['Hbb'][s,u] * HD['Dbbbb'][p,q,v,w] ,
                             +HD['Hbbbb13Dbbbbbb25'][s,u,p,q,v,w] ))
    if r==u:
        result += fsum(( +HD['Hbbbb123Dbbbbbbbb367'][p,s,v,w,q,t] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][s,p,q,t,v,w] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][q,s,v,w,p,t] ,
                         -HD['Hbbbb23Dbbbbbb45'][p,q,s,v,w,t] ,
                         +HD['Hbbbb23Dbbbbbb45'][p,t,s,v,w,q] ,
                         -HD['Hbbbb23Dbbbbbb45'][q,t,s,v,w,p] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbbabbab367'][q,s,v,w,p,t] ,
                             +HD['Hbaab123Dbbbabbab367'][p,s,v,w,q,t] ,
                             -HD['Hbaab123Dbbbabbab367'][s,p,q,t,v,w] ,
                             -HD['Hbaba13Dbbabba25'][s,t,p,q,v,w] ,
                             +HD['Hbb1Dbbbbbb5'][q,s,v,w,p,t] ,
                             -HD['Hbb1Dbbbbbb5'][p,s,v,w,q,t] ,
                             +HD['Hbb1Dbbbbbb5'][s,p,q,t,v,w] ,
                             -HD['Hbb'][s,t] * HD['Dbbbb'][p,q,v,w] ,
                             -HD['Hbbbb13Dbbbbbb25'][s,t,p,q,v,w] ))
        if s==t:
            result += fsum(( -HD['Hbbbb123Dbbbbbb245'][p,v,w,q] ,
                             +HD['Hbbbb123Dbbbbbb245'][q,v,w,p] ,
                             +HD['Hbbbb23Dbbbb23'][p,q,v,w] ))
            result += 2 * fsum(( +HD['Hbaab123Dbbabab245'][q,v,w,p] ,
                                 -HD['Hbaab123Dbbabab245'][p,v,w,q] ,
                                 +HD['Hbb1Dbbbb3'][q,v,w,p] ,
                                 -HD['Hbb1Dbbbb3'][p,v,w,q] ))
    if s==t:
        result += fsum(( -HD['Hbbbb123Dbbbbbbbb367'][q,r,v,w,p,u] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][p,r,v,w,q,u] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][r,p,q,u,v,w] ,
                         -HD['Hbbbb23Dbbbbbb45'][p,q,r,v,w,u] ,
                         +HD['Hbbbb23Dbbbbbb45'][p,u,r,v,w,q] ,
                         -HD['Hbbbb23Dbbbbbb45'][q,u,r,v,w,p] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbbabbab367'][r,p,q,u,v,w] ,
                             -HD['Hbaab123Dbbbabbab367'][q,r,v,w,p,u] ,
                             +HD['Hbaab123Dbbbabbab367'][p,r,v,w,q,u] ,
                             -HD['Hbaba13Dbbabba25'][r,u,p,q,v,w] ,
                             +HD['Hbb1Dbbbbbb5'][r,p,q,u,v,w] ,
                             +HD['Hbb1Dbbbbbb5'][q,r,v,w,p,u] ,
                             -HD['Hbb1Dbbbbbb5'][p,r,v,w,q,u] ,
                             -HD['Hbb'][r,u] * HD['Dbbbb'][p,q,v,w] ,
                             -HD['Hbbbb13Dbbbbbb25'][r,u,p,q,v,w] ))
    if p==u:
        result += fsum(( +HD['Hbbbb123Dbbbbbbbb367'][s,q,v,w,r,t] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][r,q,v,w,s,t] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][q,r,s,t,v,w] ,
                         +HD['Hbbbb23Dbbbbbb45'][r,s,q,v,w,t] ,
                         +HD['Hbbbb23Dbbbbbb45'][s,t,q,v,w,r] ,
                         -HD['Hbbbb23Dbbbbbb45'][r,t,q,v,w,s] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbbabbab367'][s,q,v,w,r,t] ,
                             +HD['Hbaab123Dbbbabbab367'][q,r,s,t,v,w] ,
                             -HD['Hbaab123Dbbbabbab367'][r,q,v,w,s,t] ,
                             +HD['Hbaba13Dbbabba25'][q,t,r,s,v,w] ,
                             -HD['Hbb1Dbbbbbb5'][s,q,v,w,r,t] ,
                             -HD['Hbb1Dbbbbbb5'][q,r,s,t,v,w] ,
                             +HD['Hbb1Dbbbbbb5'][r,q,v,w,s,t] ,
                             +HD['Hbb'][q,t] * HD['Dbbbb'][r,s,v,w] ,
                             +HD['Hbbbb13Dbbbbbb25'][q,t,r,s,v,w] ))
    if p==t:
        result += fsum(( -HD['Hbbbb123Dbbbbbbbb367'][s,q,v,w,r,u] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][r,q,v,w,s,u] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][q,r,s,u,v,w] ,
                         -HD['Hbbbb23Dbbbbbb45'][r,s,q,v,w,u] ,
                         -HD['Hbbbb23Dbbbbbb45'][s,u,q,v,w,r] ,
                         +HD['Hbbbb23Dbbbbbb45'][r,u,q,v,w,s] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbbabbab367'][s,q,v,w,r,u] ,
                             -HD['Hbaab123Dbbbabbab367'][q,r,s,u,v,w] ,
                             +HD['Hbaab123Dbbbabbab367'][r,q,v,w,s,u] ,
                             -HD['Hbaba13Dbbabba25'][q,u,r,s,v,w] ,
                             +HD['Hbb1Dbbbbbb5'][s,q,v,w,r,u] ,
                             +HD['Hbb1Dbbbbbb5'][q,r,s,u,v,w] ,
                             -HD['Hbb1Dbbbbbb5'][r,q,v,w,s,u] ,
                             -HD['Hbb'][q,u] * HD['Dbbbb'][r,s,v,w] ,
                             -HD['Hbbbb13Dbbbbbb25'][q,u,r,s,v,w] ))
    if p==v:
        result += fsum(( -HD['Hbbbb123Dbbbbbbbb367'][r,q,t,u,s,w] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][q,r,s,w,t,u] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][s,q,t,u,r,w] ,
                         -HD['Hbbbb23Dbbbbbb45'][r,w,q,t,u,s] ,
                         +HD['Hbbbb23Dbbbbbb45'][r,s,q,t,u,w] ,
                         +HD['Hbbbb23Dbbbbbb45'][s,w,q,t,u,r] ))
        result += 2 * fsum(( +HD['Hbaab123Dbbbabbab367'][q,r,s,w,t,u] ,
                             -HD['Hbaab123Dbbbabbab367'][r,q,t,u,s,w] ,
                             +HD['Hbaab123Dbbbabbab367'][s,q,t,u,r,w] ,
                             +HD['Hbaba13Dbbabba25'][q,w,r,s,t,u] ,
                             -HD['Hbb1Dbbbbbb5'][s,q,t,u,r,w] ,
                             -HD['Hbb1Dbbbbbb5'][q,r,s,w,t,u] ,
                             +HD['Hbb1Dbbbbbb5'][r,q,t,u,s,w] ,
                             +HD['Hbb'][q,w] * HD['Dbbbb'][r,s,t,u] ,
                             +HD['Hbbbb13Dbbbbbb25'][q,w,r,s,t,u] ))
    if p==w:
        result += fsum(( -HD['Hbbbb123Dbbbbbbbb367'][s,q,t,u,r,v] ,
                         -HD['Hbbbb123Dbbbbbbbb367'][q,r,s,v,t,u] ,
                         +HD['Hbbbb123Dbbbbbbbb367'][r,q,t,u,s,v] ,
                         -HD['Hbbbb23Dbbbbbb45'][r,s,q,t,u,v] ,
                         -HD['Hbbbb23Dbbbbbb45'][s,v,q,t,u,r] ,
                         +HD['Hbbbb23Dbbbbbb45'][r,v,q,t,u,s] ))
        result += 2 * fsum(( -HD['Hbaab123Dbbbabbab367'][s,q,t,u,r,v] ,
                             -HD['Hbaab123Dbbbabbab367'][q,r,s,v,t,u] ,
                             +HD['Hbaab123Dbbbabbab367'][r,q,t,u,s,v] ,
                             -HD['Hbaba13Dbbabba25'][q,v,r,s,t,u] ,
                             +HD['Hbb1Dbbbbbb5'][q,r,s,v,t,u] ,
                             -HD['Hbb1Dbbbbbb5'][r,q,t,u,s,v] ,
                             +HD['Hbb1Dbbbbbb5'][s,q,t,u,r,v] ,
                             -HD['Hbb'][q,v] * HD['Dbbbb'][r,s,t,u] ,
                             -HD['Hbbbb13Dbbbbbb25'][q,v,r,s,t,u] ))
    return result



FUNCMAPPER = {
'aa': lucc_aa , 'aa_aa': lucc_aa_aa , 'aa_bb': lucc_aa_bb , 'aa_aaaa': lucc_aa_aaaa ,
'aa_baba': lucc_aa_baba , 'aa_bbbb': lucc_aa_bbbb , 'bb': lucc_bb , 'bb_aa': lucc_bb_aa ,
'bb_bb': lucc_bb_bb , 'bb_aaaa': lucc_bb_aaaa , 'bb_baba': lucc_bb_baba , 'bb_bbbb': lucc_bb_bbbb ,
'aaaa': lucc_aaaa , 'aaaa_aa': lucc_aaaa_aa , 'aaaa_bb': lucc_aaaa_bb , 'aaaa_aaaa': lucc_aaaa_aaaa ,
'aaaa_baba': lucc_aaaa_baba , 'aaaa_bbbb': lucc_aaaa_bbbb , 'baba': lucc_baba , 'baba_aa': lucc_baba_aa ,
'baba_bb': lucc_baba_bb , 'baba_aaaa': lucc_baba_aaaa , 'baba_baba': lucc_baba_baba , 'baba_bbbb': lucc_baba_bbbb ,
'bbbb': lucc_bbbb , 'bbbb_aa': lucc_bbbb_aa , 'bbbb_bb': lucc_bbbb_bb , 'bbbb_aaaa': lucc_bbbb_aaaa ,
'bbbb_baba': lucc_bbbb_baba , 'bbbb_bbbb': lucc_bbbb_bbbb
}
