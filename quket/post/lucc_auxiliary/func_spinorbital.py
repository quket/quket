from math import fsum
from itertools import product, groupby, chain
from operator import itemgetter

from numpy import einsum, array, zeros, isclose, outer, where

from quket.fileio.fileio import prints


#                            Frozen || Core | Active || Secondary
#                              nf   ||  nc  |   na   ||    ns
#
#    integrals                 nf   ||  nc  |   na   ||    ns
#    integrals_active               ||      |   na   ||
#    H_tlide                   nf   ||  nc  |   na   ||    ns
#    1234RDM                        ||      |   na   ||
#    excitation_list                ||  nc  |   na   ||    ns


####  This is the main driver that call subroutines  ###########################


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


####  Subroutinues are under this line  ########################################

def lucc_aa(HD, *e):
    (p,r) = e
    result = 0
    result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaa'][r,:,:,:] ) ,
                     +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaa'][p,:,:,:] ) ))
    result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daa'][r,:] ) ,
                         -einsum( 'i,i->', HD['Haa'][r,:], HD['Daa'][p,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Dabab'][r,:,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Dabab'][p,:,:,:] ) ))
    return result


def lucc_aa_aa(HD, *e):
    (p,r),(t,v) = e
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['Haaaa'][r,v,:,:], HD['Daaaa'][p,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][r,t,:,:], HD['Daaaa'][p,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][p,v,:,:], HD['Daaaa'][r,t,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][p,t,:,:], HD['Daaaa'][r,v,:,:] ) ))
    result += 2 * fsum(( +HD['Haa'][p,t] * HD['Daa'][r,v] ,
                         -HD['Haa'][p,v] * HD['Daa'][r,t] ,
                         -HD['Haa'][r,t] * HD['Daa'][p,v] ,
                         +HD['Haa'][r,v] * HD['Daa'][p,t] ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,:,t,:], HD['Daaaa'][r,:,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,:,v,:], HD['Daaaa'][r,:,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,:,t,:], HD['Daaaa'][p,:,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,:,v,:], HD['Daaaa'][p,:,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][p,:,t,:], HD['Dabab'][r,:,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][p,:,v,:], HD['Dabab'][r,:,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][r,:,t,:], HD['Dabab'][p,:,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][r,:,v,:], HD['Dabab'][p,:,t,:] ) ))
    if r==v:
        result += +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaa'][t,:,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daa'][t,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Dabab'][t,:,:,:] ) ))
    if r==t:
        result += -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaa'][v,:,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daa'][v,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Dabab'][v,:,:,:] ) ))
    if p==v:
        result += -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaa'][t,:,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][r,:], HD['Daa'][t,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Dabab'][t,:,:,:] ) ))
    if p==t:
        result += +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaa'][v,:,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][r,:], HD['Daa'][v,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Dabab'][v,:,:,:] ) ))
    return result


def lucc_aa_bb(HD, *e):
    (p,r),(t,v) = e
    result = 0
    result += 2 * fsum(( -einsum( 'ij,ij->', HD['Habba'][p,:,t,:], HD['Dabba'][r,:,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][p,:,v,:], HD['Dabba'][r,:,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][r,:,t,:], HD['Dabba'][p,:,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][r,:,v,:], HD['Dabba'][p,:,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,p,:,:], HD['Dbaab'][t,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,p,:,:], HD['Dbaab'][v,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,r,:,:], HD['Dbaab'][t,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,r,:,:], HD['Dbaab'][v,p,:,:] ) ))
    return result


def lucc_aa_aaaa(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Haaaa'][p,w,:,:], HD['Daaaaaa'][r,t,u,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][p,v,:,:], HD['Daaaaaa'][r,t,u,w,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][p,t,:,:], HD['Daaaaaa'][r,v,w,u,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][p,u,:,:], HD['Daaaaaa'][r,v,w,t,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][r,w,:,:], HD['Daaaaaa'][p,t,u,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][r,v,:,:], HD['Daaaaaa'][p,t,u,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][r,t,:,:], HD['Daaaaaa'][p,v,w,u,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][r,u,:,:], HD['Daaaaaa'][p,v,w,t,:,:] ) ))
    result += 2 * fsum(( +HD['Haa'][p,t] * HD['Daaaa'][r,u,v,w] ,
                         -HD['Haa'][p,u] * HD['Daaaa'][r,t,v,w] ,
                         +HD['Haa'][p,w] * HD['Daaaa'][r,v,t,u] ,
                         -HD['Haa'][p,v] * HD['Daaaa'][r,w,t,u] ,
                         -HD['Haa'][r,t] * HD['Daaaa'][p,u,v,w] ,
                         +HD['Haa'][r,u] * HD['Daaaa'][p,t,v,w] ,
                         -HD['Haa'][r,w] * HD['Daaaa'][p,v,t,u] ,
                         +HD['Haa'][r,v] * HD['Daaaa'][p,w,t,u] ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,:,t,:], HD['Daaaaaa'][r,u,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,:,u,:], HD['Daaaaaa'][r,t,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,:,w,:], HD['Daaaaaa'][r,v,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,:,v,:], HD['Daaaaaa'][r,w,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,:,t,:], HD['Daaaaaa'][p,u,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,:,u,:], HD['Daaaaaa'][p,t,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,:,w,:], HD['Daaaaaa'][p,v,:,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,:,v,:], HD['Daaaaaa'][p,w,:,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][t,u,p,:], HD['Daaaa'][v,w,r,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][v,w,r,:], HD['Daaaa'][t,u,p,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][v,w,p,:], HD['Daaaa'][t,u,r,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][t,u,r,:], HD['Daaaa'][v,w,p,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][p,:,t,:], HD['Daabaab'][r,u,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][p,:,u,:], HD['Daabaab'][r,t,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][p,:,w,:], HD['Daabaab'][r,v,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][p,:,v,:], HD['Daabaab'][r,w,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][r,:,t,:], HD['Daabaab'][p,u,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][r,:,u,:], HD['Daabaab'][p,t,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][r,:,w,:], HD['Daabaab'][p,v,:,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][r,:,v,:], HD['Daabaab'][p,w,:,t,u,:] ) ))
    if r==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][t,u,:,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,v,:,:], HD['Daaaa'][t,u,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][t,u,v,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][t,u,:,v,:,:] ) ))
    if r==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][t,u,:,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,w,:,:], HD['Daaaa'][t,u,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][t,u,w,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][t,u,:,w,:,:] ) ))
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][v,w,:,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,u,:,:], HD['Daaaa'][v,w,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][v,w,u,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][v,w,:,u,:,:] ) ))
    if r==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][v,w,:,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,t,:,:], HD['Daaaa'][v,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][v,w,t,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][v,w,:,t,:,:] ) ))
    if p==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][t,u,:,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,v,:,:], HD['Daaaa'][t,u,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][t,u,v,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][t,u,:,v,:,:] ) ))
    if p==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][t,u,:,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,w,:,:], HD['Daaaa'][t,u,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][t,u,w,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][t,u,:,w,:,:] ) ))
    if p==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][v,w,:,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,u,:,:], HD['Daaaa'][v,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][v,w,u,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][v,w,:,u,:,:] ) ))
    if p==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][v,w,:,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,t,:,:], HD['Daaaa'][v,w,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][v,w,t,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][v,w,:,t,:,:] ) ))
    return result


def lucc_aa_baba(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Haaaa'][r,w,:,:], HD['Dbaabaa'][t,p,u,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][r,u,:,:], HD['Dbaabaa'][v,p,w,t,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][p,w,:,:], HD['Dbaabaa'][t,r,u,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][p,u,:,:], HD['Dbaabaa'][v,r,w,t,:,:] ) ))
    result += 2 * fsum(( -HD['Haa'][p,u] * HD['Dbaab'][t,r,w,v] ,
                         +HD['Haa'][p,w] * HD['Dbaab'][v,r,u,t] ,
                         +HD['Haa'][r,u] * HD['Dbaab'][t,p,w,v] ,
                         -HD['Haa'][r,w] * HD['Dbaab'][v,p,u,t] ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,:,u,:], HD['Dbaaaba'][t,r,:,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,:,w,:], HD['Dbaaaba'][v,r,:,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,:,u,:], HD['Dbaaaba'][t,p,:,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,:,w,:], HD['Dbaaaba'][v,p,:,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][r,:,w,:], HD['Dbababb'][v,p,:,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][p,:,u,:], HD['Dbababb'][t,r,:,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][r,:,u,:], HD['Dbababb'][t,p,:,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][p,:,w,:], HD['Dbababb'][v,r,:,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][p,:,t,:], HD['Daababa'][r,u,:,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][r,:,v,:], HD['Daababa'][p,w,:,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][p,:,v,:], HD['Daababa'][r,w,:,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][r,:,t,:], HD['Daababa'][p,u,:,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,r,:,:], HD['Dbaaaab'][t,p,u,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,p,:,:], HD['Dbaaaab'][t,r,u,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,r,:,:], HD['Dbaaaab'][v,p,w,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,p,:,:], HD['Dbaaaab'][v,r,w,u,:,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][t,u,p,:], HD['Dbaab'][v,w,r,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][t,u,r,:], HD['Dbaab'][v,w,p,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][v,w,p,:], HD['Dbaab'][t,u,r,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][v,w,r,:], HD['Dbaab'][t,u,p,:] ) ))
    if r==w:
        result += -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Dbaabaa'][t,u,:,v,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Dbaba'][t,u,v,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Dbabbab'][t,u,:,v,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][v,p,:,:], HD['Dbaab'][t,u,:,:] ) ))
    if r==u:
        result += +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Dbaabaa'][v,w,:,t,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Dbaba'][v,w,t,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Dbabbab'][v,w,:,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][t,p,:,:], HD['Dbaab'][v,w,:,:] ) ))
    if p==w:
        result += +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Dbaabaa'][t,u,:,v,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][r,:], HD['Dbaba'][t,u,v,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Dbabbab'][t,u,:,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][v,r,:,:], HD['Dbaab'][t,u,:,:] ) ))
    if p==u:
        result += -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Dbaabaa'][v,w,:,t,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][r,:], HD['Dbaba'][v,w,t,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Dbabbab'][v,w,:,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][t,r,:,:], HD['Dbaab'][v,w,:,:] ) ))
    return result


def lucc_aa_bbbb(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += 2 * fsum(( -einsum( 'ij,ij->', HD['Habba'][p,:,t,:], HD['Dbabbba'][u,r,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][p,:,u,:], HD['Dbabbba'][t,r,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][p,:,w,:], HD['Dbabbba'][v,r,:,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][p,:,v,:], HD['Dbabbba'][w,r,:,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][r,:,t,:], HD['Dbabbba'][u,p,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][r,:,u,:], HD['Dbabbba'][t,p,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][r,:,w,:], HD['Dbabbba'][v,p,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][r,:,v,:], HD['Dbabbba'][w,p,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][w,p,:,:], HD['Dbbabab'][t,u,r,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,p,:,:], HD['Dbbabab'][t,u,r,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,p,:,:], HD['Dbbabab'][v,w,r,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][u,p,:,:], HD['Dbbabab'][v,w,r,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][w,r,:,:], HD['Dbbabab'][t,u,p,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,r,:,:], HD['Dbbabab'][t,u,p,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,r,:,:], HD['Dbbabab'][v,w,p,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][u,r,:,:], HD['Dbbabab'][v,w,p,t,:,:] ) ))
    return result


def lucc_bb(HD, *e):
    (p,r) = e
    result = 0
    result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbb'][r,:,:,:] ) ,
                     +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbb'][p,:,:,:] ) ))
    result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaab'][r,:,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaab'][p,:,:,:] ) ,
                         +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbb'][r,:] ) ,
                         -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbb'][p,:] ) ))
    return result


def lucc_bb_aa(HD, *e):
    (p,r),(t,v) = e
    result = 0
    result += 2 * fsum(( -einsum( 'ij,ij->', HD['Hbaab'][p,:,t,:], HD['Dbaab'][r,:,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,:,v,:], HD['Dbaab'][r,:,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,t,:], HD['Dbaab'][p,:,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,v,:], HD['Dbaab'][p,:,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,v,:,:], HD['Dbaab'][r,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,t,:,:], HD['Dbaab'][r,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,v,:,:], HD['Dbaab'][p,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,t,:,:], HD['Dbaab'][p,v,:,:] ) ))
    return result


def lucc_bb_bb(HD, *e):
    (p,r),(t,v) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbbb'][r,t,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbbb'][r,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbbb'][p,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbbb'][p,v,:,:] ) ))
    result += 2 * fsum(( -einsum( 'ij,ij->', HD['Hbaba'][p,:,t,:], HD['Dbaba'][r,:,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][p,:,v,:], HD['Dbaba'][r,:,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,t,:], HD['Dbaba'][p,:,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,v,:], HD['Dbaba'][p,:,t,:] ) ,
                         +HD['Hbb'][p,t] * HD['Dbb'][r,v] ,
                         -HD['Hbb'][p,v] * HD['Dbb'][r,t] ,
                         -HD['Hbb'][r,t] * HD['Dbb'][p,v] ,
                         +HD['Hbb'][r,v] * HD['Dbb'][p,t] ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,t,:], HD['Dbbbb'][r,:,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,v,:], HD['Dbbbb'][r,:,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,t,:], HD['Dbbbb'][p,:,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,v,:], HD['Dbbbb'][p,:,t,:] ) ))
    if r==v:
        result += +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbb'][t,:,:,:] )
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaab'][t,:,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbb'][t,:] ) ))
    if r==t:
        result += -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbb'][v,:,:,:] )
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaab'][v,:,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbb'][v,:] ) ))
    if p==v:
        result += -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbb'][t,:,:,:] )
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaab'][t,:,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbb'][t,:] ) ))
    if p==t:
        result += +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbb'][v,:,:,:] )
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaab'][v,:,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbb'][v,:] ) ))
    return result


def lucc_bb_aaaa(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += 2 * fsum(( +einsum( 'ij,ij->', HD['Hbaab'][p,:,t,:], HD['Dbaaaab'][r,u,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,:,u,:], HD['Dbaaaab'][r,t,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,:,w,:], HD['Dbaaaab'][r,v,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,:,v,:], HD['Dbaaaab'][r,w,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,t,:], HD['Dbaaaab'][p,u,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,u,:], HD['Dbaaaab'][p,t,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,w,:], HD['Dbaaaab'][p,v,:,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,v,:], HD['Dbaaaab'][p,w,:,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,w,:,:], HD['Dbaaaab'][r,t,u,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,v,:,:], HD['Dbaaaab'][r,t,u,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,t,:,:], HD['Dbaaaab'][r,v,w,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,u,:,:], HD['Dbaaaab'][r,v,w,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,w,:,:], HD['Dbaaaab'][p,t,u,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,v,:,:], HD['Dbaaaab'][p,t,u,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,t,:,:], HD['Dbaaaab'][p,v,w,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,u,:,:], HD['Dbaaaab'][p,v,w,t,:,:] ) ))
    return result


def lucc_bb_baba(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbaabb'][p,t,u,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbaabb'][p,v,w,u,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbaabb'][r,t,u,w,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbaabb'][r,v,w,u,:,:] ) ))
    result += 2 * fsum(( -einsum( 'ij,ij->', HD['Hbaab'][p,:,w,:], HD['Dbbaabb'][r,v,:,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,w,:], HD['Dbbaabb'][p,v,:,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,:,u,:], HD['Dbbaabb'][r,t,:,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,u,:], HD['Dbbaabb'][p,t,:,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,u,:,:], HD['Dbbabab'][p,v,w,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,u,:,:], HD['Dbbabab'][r,v,w,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,w,:,:], HD['Dbbabab'][p,t,u,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,w,:,:], HD['Dbbabab'][r,t,u,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,v,:], HD['Dbaaaba'][p,w,:,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][p,:,v,:], HD['Dbaaaba'][r,w,:,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,t,:], HD['Dbaaaba'][p,u,:,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][p,:,t,:], HD['Dbaaaba'][r,u,:,w,v,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][v,w,p,:], HD['Dbaba'][t,u,r,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][v,w,r,:], HD['Dbaba'][t,u,p,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][t,u,p,:], HD['Dbaba'][v,w,r,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][t,u,r,:], HD['Dbaba'][v,w,p,:] ) ,
                         -HD['Hbb'][p,t] * HD['Dbaab'][r,u,w,v] ,
                         +HD['Hbb'][p,v] * HD['Dbaab'][r,w,u,t] ,
                         +HD['Hbb'][r,t] * HD['Dbaab'][p,u,w,v] ,
                         -HD['Hbb'][r,v] * HD['Dbaab'][p,w,u,t] ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,t,:], HD['Dbababb'][r,u,:,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,v,:], HD['Dbababb'][r,w,:,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,t,:], HD['Dbababb'][p,u,:,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,v,:], HD['Dbababb'][p,w,:,u,t,:] ) ))
    if r==v:
        result += +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbababb'][t,u,:,w,:,:] )
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaab'][t,u,:,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,w,:,:], HD['Dbaab'][t,u,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaab'][t,u,w,:] ) ))
    if r==t:
        result += -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbababb'][v,w,:,u,:,:] )
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaab'][v,w,:,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,u,:,:], HD['Dbaab'][v,w,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaab'][v,w,u,:] ) ))
    if p==v:
        result += -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbababb'][t,u,:,w,:,:] )
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaab'][t,u,:,w,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,w,:,:], HD['Dbaab'][t,u,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaab'][t,u,w,:] ) ))
    if p==t:
        result += +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbababb'][v,w,:,u,:,:] )
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaab'][v,w,:,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,u,:,:], HD['Dbaab'][v,w,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaab'][v,w,u,:] ) ))
    return result


def lucc_bb_bbbb(HD, *e):
    (p,r),(t,u,v,w) = e
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbbbbb'][p,v,w,u,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][r,u,:,:], HD['Dbbbbbb'][p,v,w,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,w,:,:], HD['Dbbbbbb'][r,t,u,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbbbbb'][r,t,u,w,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbbbbb'][r,v,w,u,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][p,u,:,:], HD['Dbbbbbb'][r,v,w,t,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][r,w,:,:], HD['Dbbbbbb'][p,t,u,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbbbbb'][p,t,u,w,:,:] ) ))
    result += 2 * fsum(( +einsum( 'ij,ij->', HD['Hbaba'][p,:,t,:], HD['Dbbabba'][r,u,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][p,:,u,:], HD['Dbbabba'][r,t,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][p,:,w,:], HD['Dbbabba'][r,v,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][p,:,v,:], HD['Dbbabba'][r,w,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,t,:], HD['Dbbabba'][p,u,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,u,:], HD['Dbbabba'][p,t,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,w,:], HD['Dbbabba'][p,v,:,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,v,:], HD['Dbbabba'][p,w,:,t,u,:] ) ,
                         +HD['Hbb'][p,t] * HD['Dbbbb'][r,u,v,w] ,
                         -HD['Hbb'][p,u] * HD['Dbbbb'][r,t,v,w] ,
                         +HD['Hbb'][p,w] * HD['Dbbbb'][r,v,t,u] ,
                         -HD['Hbb'][p,v] * HD['Dbbbb'][r,w,t,u] ,
                         -HD['Hbb'][r,t] * HD['Dbbbb'][p,u,v,w] ,
                         +HD['Hbb'][r,u] * HD['Dbbbb'][p,t,v,w] ,
                         -HD['Hbb'][r,w] * HD['Dbbbb'][p,v,t,u] ,
                         +HD['Hbb'][r,v] * HD['Dbbbb'][p,w,t,u] ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,t,:], HD['Dbbbbbb'][r,u,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,u,:], HD['Dbbbbbb'][r,t,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,w,:], HD['Dbbbbbb'][r,v,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,v,:], HD['Dbbbbbb'][r,w,:,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,t,:], HD['Dbbbbbb'][p,u,:,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,u,:], HD['Dbbbbbb'][p,t,:,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,w,:], HD['Dbbbbbb'][p,v,:,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,v,:], HD['Dbbbbbb'][p,w,:,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][t,u,p,:], HD['Dbbbb'][v,w,r,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][v,w,r,:], HD['Dbbbb'][t,u,p,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][v,w,p,:], HD['Dbbbb'][t,u,r,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][t,u,r,:], HD['Dbbbb'][v,w,p,:] ) ))
    if r==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][t,u,:,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbbb'][t,u,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][t,u,:,v,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][t,u,v,:] ) ))
    if r==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][t,u,:,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,w,:,:], HD['Dbbbb'][t,u,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][t,u,:,w,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][t,u,w,:] ) ))
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][v,w,:,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,u,:,:], HD['Dbbbb'][v,w,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][v,w,:,u,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][v,w,u,:] ) ))
    if r==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][v,w,:,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbbb'][v,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][v,w,:,t,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][v,w,t,:] ) ))
    if p==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][t,u,:,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbbb'][t,u,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][t,u,:,v,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][t,u,v,:] ) ))
    if p==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][t,u,:,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,w,:,:], HD['Dbbbb'][t,u,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][t,u,:,w,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][t,u,w,:] ) ))
    if p==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][v,w,:,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,u,:,:], HD['Dbbbb'][v,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][v,w,:,u,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][v,w,u,:] ) ))
    if p==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][v,w,:,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbbb'][v,w,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][v,w,:,t,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][v,w,t,:] ) ))
    return result


def lucc_aaaa(HD, *e):
    (p,q,r,s) = e
    result = 0
    result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][r,s,:,q,:,:] ) ,
                     +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][r,s,:,p,:,:] ) ,
                     -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][p,q,:,r,:,:] ) ,
                     +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][p,q,:,s,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][r,s,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][p,q,:,:] ) ))
    result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][r,s,q,:] ) ,
                         +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][r,s,p,:] ) ,
                         -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][p,q,r,:] ) ,
                         +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][p,q,s,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][r,s,:,q,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daabaab'][r,s,:,p,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daabaab'][p,q,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][p,q,:,s,:,:] ) ))
    return result


def lucc_aaaa_aa(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Haaaa'][p,v,:,:], HD['Daaaaaa'][r,s,t,q,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][p,t,:,:], HD['Daaaaaa'][r,s,v,q,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][q,v,:,:], HD['Daaaaaa'][r,s,t,p,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][q,t,:,:], HD['Daaaaaa'][r,s,v,p,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][s,v,:,:], HD['Daaaaaa'][p,q,t,r,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][s,t,:,:], HD['Daaaaaa'][p,q,v,r,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][r,v,:,:], HD['Daaaaaa'][p,q,t,s,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][r,t,:,:], HD['Daaaaaa'][p,q,v,s,:,:] ) ))
    result += 2 * fsum(( -HD['Haa'][p,t] * HD['Daaaa'][q,v,r,s] ,
                         +HD['Haa'][p,v] * HD['Daaaa'][q,t,r,s] ,
                         +HD['Haa'][q,t] * HD['Daaaa'][p,v,r,s] ,
                         -HD['Haa'][q,v] * HD['Daaaa'][p,t,r,s] ,
                         -HD['Haa'][s,t] * HD['Daaaa'][p,q,r,v] ,
                         +HD['Haa'][s,v] * HD['Daaaa'][p,q,r,t] ,
                         +HD['Haa'][r,t] * HD['Daaaa'][p,q,s,v] ,
                         -HD['Haa'][r,v] * HD['Daaaa'][p,q,s,t] ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,:,t,:], HD['Daaaaaa'][r,s,:,q,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,:,v,:], HD['Daaaaaa'][r,s,:,q,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,:,t,:], HD['Daaaaaa'][r,s,:,p,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,:,v,:], HD['Daaaaaa'][r,s,:,p,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,:,t,:], HD['Daaaaaa'][p,q,:,r,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,:,v,:], HD['Daaaaaa'][p,q,:,r,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,:,t,:], HD['Daaaaaa'][p,q,:,s,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,:,v,:], HD['Daaaaaa'][p,q,:,s,t,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][p,q,t,:], HD['Daaaa'][r,s,v,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][p,q,v,:], HD['Daaaa'][r,s,t,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][r,s,t,:], HD['Daaaa'][p,q,v,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][r,s,v,:], HD['Daaaa'][p,q,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][p,:,t,:], HD['Daabaab'][r,s,:,q,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][p,:,v,:], HD['Daabaab'][r,s,:,q,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][q,:,t,:], HD['Daabaab'][r,s,:,p,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][q,:,v,:], HD['Daabaab'][r,s,:,p,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][s,:,t,:], HD['Daabaab'][p,q,:,r,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][s,:,v,:], HD['Daabaab'][p,q,:,r,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][r,:,t,:], HD['Daabaab'][p,q,:,s,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][r,:,v,:], HD['Daabaab'][p,q,:,s,t,:] ) ))
    if q==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][r,s,:,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][p,v,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][p,v,:,s,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][p,v,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][r,s,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][p,v,r,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][p,v,s,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daabaab'][p,v,:,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][p,v,:,s,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][r,s,:,v,:,:] ) ))
    if r==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][p,q,:,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][s,t,:,q,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][s,t,:,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][s,t,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][s,t,q,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][s,t,p,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][p,q,t,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daabaab'][p,q,:,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][s,t,:,q,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daabaab'][s,t,:,p,:,:] ) ))
    if s==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][r,t,:,q,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][p,q,:,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][r,t,:,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][r,t,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][r,t,q,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][r,t,p,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][p,q,t,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][p,q,:,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][r,t,:,q,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daabaab'][r,t,:,p,:,:] ) ))
    if q==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][p,t,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][r,s,:,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][p,t,:,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][p,t,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][r,s,t,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][p,t,r,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][p,t,s,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daabaab'][p,t,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][p,t,:,s,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][r,s,:,t,:,:] ) ))
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][p,q,:,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][s,v,:,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][s,v,:,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][s,v,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][s,v,q,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][s,v,p,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][p,q,v,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daabaab'][s,v,:,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daabaab'][p,q,:,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][s,v,:,q,:,:] ) ))
    if s==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][p,q,:,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][r,v,:,q,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][r,v,:,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][r,v,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][r,v,q,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][r,v,p,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][p,q,v,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daabaab'][r,v,:,p,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][p,q,:,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][r,v,:,q,:,:] ) ))
    if p==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][q,v,:,r,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][q,v,:,s,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][r,s,:,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][q,v,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][r,s,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][q,v,r,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][q,v,s,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daabaab'][q,v,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][q,v,:,s,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daabaab'][r,s,:,v,:,:] ) ))
    if p==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][q,t,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][q,t,:,s,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][r,s,:,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][q,t,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][r,s,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][q,t,r,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][q,t,s,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daabaab'][r,s,:,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daabaab'][q,t,:,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][q,t,:,s,:,:] ) ))
    return result


def lucc_aaaa_bb(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += 2 * fsum(( -einsum( 'ij,ij->', HD['Habba'][p,:,t,:], HD['Daababa'][r,s,:,q,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][p,:,v,:], HD['Daababa'][r,s,:,q,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][q,:,t,:], HD['Daababa'][r,s,:,p,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][q,:,v,:], HD['Daababa'][r,s,:,p,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][s,:,t,:], HD['Daababa'][p,q,:,r,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][s,:,v,:], HD['Daababa'][p,q,:,r,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][r,:,t,:], HD['Daababa'][p,q,:,s,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][r,:,v,:], HD['Daababa'][p,q,:,s,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,p,:,:], HD['Dbaaaab'][t,r,s,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,p,:,:], HD['Dbaaaab'][v,r,s,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,q,:,:], HD['Dbaaaab'][t,r,s,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,q,:,:], HD['Dbaaaab'][v,r,s,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,s,:,:], HD['Dbaaaab'][t,p,q,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,s,:,:], HD['Dbaaaab'][v,p,q,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,r,:,:], HD['Dbaaaab'][t,p,q,s,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,r,:,:], HD['Dbaaaab'][v,p,q,s,:,:] ) ))
    return result


def lucc_aaaa_aaaa(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Haaaa'][r,t,:,:], HD['Daaaaaaaa'][p,q,v,w,s,u,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][q,w,:,:], HD['Daaaaaaaa'][r,s,t,u,p,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][q,v,:,:], HD['Daaaaaaaa'][r,s,t,u,p,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][r,u,:,:], HD['Daaaaaaaa'][p,q,v,w,s,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][q,t,:,:], HD['Daaaaaaaa'][r,s,v,w,p,u,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][q,u,:,:], HD['Daaaaaaaa'][r,s,v,w,p,t,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][s,w,:,:], HD['Daaaaaaaa'][p,q,t,u,r,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][s,v,:,:], HD['Daaaaaaaa'][p,q,t,u,r,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][r,v,:,:], HD['Daaaaaaaa'][p,q,t,u,s,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][s,t,:,:], HD['Daaaaaaaa'][p,q,v,w,r,u,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][p,w,:,:], HD['Daaaaaaaa'][r,s,t,u,q,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][p,v,:,:], HD['Daaaaaaaa'][r,s,t,u,q,w,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][s,u,:,:], HD['Daaaaaaaa'][p,q,v,w,r,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][r,w,:,:], HD['Daaaaaaaa'][p,q,t,u,s,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][p,t,:,:], HD['Daaaaaaaa'][r,s,v,w,q,u,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][p,u,:,:], HD['Daaaaaaaa'][r,s,v,w,q,t,:,:] ) ))
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
                         +einsum( 'ij,ij->', HD['Haaaa'][q,:,w,:], HD['Daaaaaaaa'][r,s,v,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,:,v,:], HD['Daaaaaaaa'][r,s,w,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,:,t,:], HD['Daaaaaaaa'][p,q,u,:,r,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,:,u,:], HD['Daaaaaaaa'][p,q,t,:,r,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,:,w,:], HD['Daaaaaaaa'][p,q,v,:,r,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,:,v,:], HD['Daaaaaaaa'][p,q,w,:,r,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,:,t,:], HD['Daaaaaaaa'][p,q,u,:,s,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,:,u,:], HD['Daaaaaaaa'][p,q,t,:,s,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,:,w,:], HD['Daaaaaaaa'][p,q,v,:,s,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,:,v,:], HD['Daaaaaaaa'][p,q,w,:,s,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,:,t,:], HD['Daaaaaaaa'][r,s,u,:,q,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,:,u,:], HD['Daaaaaaaa'][r,s,t,:,q,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,:,w,:], HD['Daaaaaaaa'][r,s,v,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,:,v,:], HD['Daaaaaaaa'][r,s,w,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,:,t,:], HD['Daaaaaaaa'][r,s,u,:,p,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,:,u,:], HD['Daaaaaaaa'][r,s,t,:,p,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][r,s,t,:], HD['Daaaaaa'][p,q,u,v,w,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][r,s,u,:], HD['Daaaaaa'][p,q,t,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][r,s,w,:], HD['Daaaaaa'][p,q,v,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][r,s,v,:], HD['Daaaaaa'][p,q,w,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][t,u,s,:], HD['Daaaaaa'][r,v,w,p,q,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][v,w,s,:], HD['Daaaaaa'][r,t,u,p,q,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][t,u,r,:], HD['Daaaaaa'][s,v,w,p,q,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][p,q,t,:], HD['Daaaaaa'][r,s,u,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][p,q,u,:], HD['Daaaaaa'][r,s,t,v,w,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][v,w,r,:], HD['Daaaaaa'][s,t,u,p,q,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][p,q,w,:], HD['Daaaaaa'][r,s,v,t,u,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][p,q,v,:], HD['Daaaaaa'][r,s,w,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][t,u,p,:], HD['Daaaaaa'][q,v,w,r,s,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][v,w,p,:], HD['Daaaaaa'][q,t,u,r,s,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][t,u,q,:], HD['Daaaaaa'][p,v,w,r,s,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][v,w,q,:], HD['Daaaaaa'][p,t,u,r,s,:] ) ,
                         -HD['Haaaa'][p,q,t,u] * HD['Daaaa'][r,s,v,w] ,
                         +HD['Haaaa'][r,s,t,u] * HD['Daaaa'][p,q,v,w] ,
                         +HD['Haaaa'][p,q,v,w] * HD['Daaaa'][r,s,t,u] ,
                         -HD['Haaaa'][r,s,v,w] * HD['Daaaa'][p,q,t,u] ,
                         +einsum( 'ij,ij->', HD['Habab'][r,:,t,:], HD['Daaabaaab'][p,q,u,:,s,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][r,:,u,:], HD['Daaabaaab'][p,q,t,:,s,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][r,:,w,:], HD['Daaabaaab'][p,q,v,:,s,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][r,:,v,:], HD['Daaabaaab'][p,q,w,:,s,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][p,:,t,:], HD['Daaabaaab'][r,s,u,:,q,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][p,:,u,:], HD['Daaabaaab'][r,s,t,:,q,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][p,:,w,:], HD['Daaabaaab'][r,s,v,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][p,:,v,:], HD['Daaabaaab'][r,s,w,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][q,:,t,:], HD['Daaabaaab'][r,s,u,:,p,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][q,:,u,:], HD['Daaabaaab'][r,s,t,:,p,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][q,:,w,:], HD['Daaabaaab'][r,s,v,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][q,:,v,:], HD['Daaabaaab'][r,s,w,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][s,:,t,:], HD['Daaabaaab'][p,q,u,:,r,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][s,:,u,:], HD['Daaabaaab'][p,q,t,:,r,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][s,:,w,:], HD['Daaabaaab'][p,q,v,:,r,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][s,:,v,:], HD['Daaabaaab'][p,q,w,:,r,t,u,:] ) ))
    if q==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,s,t,:,v,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,v,w,:,r,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,v,w,:,s,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][p,v,w,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,t,:,:], HD['Daaaaaa'][p,v,w,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,t,:,:], HD['Daaaaaa'][p,v,w,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,v,w,s,t,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,s,t,v,w,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,v,w,r,t,:] ) ,
                             -HD['Haa'][p,t] * HD['Daaaa'][r,s,v,w] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][p,:,t,:], HD['Daaaaaa'][r,s,:,v,w,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][r,s,t,:,v,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][p,v,w,:,r,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][p,v,w,:,s,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habab'][p,:,t,:], HD['Daabaab'][r,s,:,v,w,:] ) ))
        if p==t:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][v,w,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][v,w,:,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][v,w,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][v,w,r,:] ) ,
                                 -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][v,w,s,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][v,w,:,s,:,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daabaab'][v,w,:,r,:,:] ) ))
    if q==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,v,w,:,r,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,v,w,:,s,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,s,u,:,v,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][p,v,w,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,u,:,:], HD['Daaaaaa'][p,v,w,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,u,:,:], HD['Daaaaaa'][p,v,w,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,s,u,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,v,w,r,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,v,w,s,u,:] ) ,
                             +HD['Haa'][p,u] * HD['Daaaa'][r,s,v,w] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][p,:,u,:], HD['Daaaaaa'][r,s,:,v,w,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][r,s,u,:,v,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][p,v,w,:,r,u,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][p,v,w,:,s,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habab'][p,:,u,:], HD['Daabaab'][r,s,:,v,w,:] ) ))
        if p==u:
            result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][v,w,:,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][v,w,:,s,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][v,w,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][v,w,r,:] ) ,
                                 +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][v,w,s,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daabaab'][v,w,:,r,:,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][v,w,:,s,:,:] ) ))
    if s==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,t,u,:,q,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,t,u,:,p,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,q,w,:,t,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,w,:,:], HD['Daaaaaa'][r,t,u,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,w,:,:], HD['Daaaaaa'][r,t,u,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaaaa'][r,t,u,w,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,t,u,p,w,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,t,u,q,w,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,q,w,t,u,:] ) ,
                             +HD['Haa'][r,w] * HD['Daaaa'][p,q,t,u] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][r,:,w,:], HD['Daaaaaa'][p,q,:,t,u,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][r,t,u,:,p,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][r,t,u,:,q,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][p,q,w,:,t,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habab'][r,:,w,:], HD['Daabaab'][p,q,:,t,u,:] ) ))
        if r==w:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][t,u,:,q,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][t,u,:,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][t,u,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][t,u,q,:] ) ,
                                 -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][t,u,p,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][t,u,:,q,:,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daabaab'][t,u,:,p,:,:] ) ))
    if r==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][s,t,u,:,p,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,q,v,:,t,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][s,t,u,:,q,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,v,:,:], HD['Daaaaaa'][s,t,u,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaaaa'][s,t,u,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,v,:,:], HD['Daaaaaa'][s,t,u,q,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,q,v,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][s,t,u,q,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][s,t,u,p,v,:] ) ,
                             +HD['Haa'][s,v] * HD['Daaaa'][p,q,t,u] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][s,:,v,:], HD['Daaaaaa'][p,q,:,t,u,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][s,t,u,:,q,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][p,q,v,:,t,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][s,t,u,:,p,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habab'][s,:,v,:], HD['Daabaab'][p,q,:,t,u,:] ) ))
    if s==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,t,u,:,q,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,t,u,:,p,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,q,v,:,t,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,v,:,:], HD['Daaaaaa'][r,t,u,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaaaa'][r,t,u,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,v,:,:], HD['Daaaaaa'][r,t,u,q,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,t,u,p,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,t,u,q,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,q,v,t,u,:] ) ,
                             -HD['Haa'][r,v] * HD['Daaaa'][p,q,t,u] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][r,:,v,:], HD['Daaaaaa'][p,q,:,t,u,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][r,t,u,:,p,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][r,t,u,:,q,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][p,q,v,:,t,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habab'][r,:,v,:], HD['Daabaab'][p,q,:,t,u,:] ) ))
        if r==v:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][t,u,:,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][t,u,:,q,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][t,u,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][t,u,p,:] ) ,
                                 -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][t,u,q,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][t,u,:,q,:,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daabaab'][t,u,:,p,:,:] ) ))
    if r==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][s,t,u,:,p,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,q,w,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][s,t,u,:,q,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,w,:,:], HD['Daaaaaa'][s,t,u,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaaaa'][s,t,u,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,w,:,:], HD['Daaaaaa'][s,t,u,q,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,q,w,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][s,t,u,p,w,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][s,t,u,q,w,:] ) ,
                             -HD['Haa'][s,w] * HD['Daaaa'][p,q,t,u] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][s,:,w,:], HD['Daaaaaa'][p,q,:,t,u,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][p,q,w,:,t,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][s,t,u,:,q,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][s,t,u,:,p,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habab'][s,:,w,:], HD['Daabaab'][p,q,:,t,u,:] ) ))
    if q==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,t,u,:,r,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,t,u,:,s,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,s,w,:,t,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,w,:,:], HD['Daaaaaa'][p,t,u,s,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][p,t,u,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,w,:,:], HD['Daaaaaa'][p,t,u,r,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,t,u,r,w,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,s,w,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,t,u,s,w,:] ) ,
                             -HD['Haa'][p,w] * HD['Daaaa'][r,s,t,u] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][p,:,w,:], HD['Daaaaaa'][r,s,:,t,u,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][r,s,w,:,t,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][p,t,u,:,s,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][p,t,u,:,r,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habab'][p,:,w,:], HD['Daabaab'][r,s,:,t,u,:] ) ))
        if p==w:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][t,u,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][t,u,:,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][t,u,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][t,u,s,:] ) ,
                                 +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][t,u,r,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][t,u,:,s,:,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daabaab'][t,u,:,r,:,:] ) ))
    if q==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,t,u,:,s,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,s,v,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,t,u,:,r,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,v,:,:], HD['Daaaaaa'][p,t,u,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][p,t,u,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,v,:,:], HD['Daaaaaa'][p,t,u,r,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,t,u,r,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,s,v,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,t,u,s,v,:] ) ,
                             +HD['Haa'][p,v] * HD['Daaaa'][r,s,t,u] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][p,:,v,:], HD['Daaaaaa'][r,s,:,t,u,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][r,s,v,:,t,u,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][p,t,u,:,s,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][p,t,u,:,r,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habab'][p,:,v,:], HD['Daabaab'][r,s,:,t,u,:] ) ))
        if p==v:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][t,u,:,s,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][t,u,:,r,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][t,u,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][t,u,s,:] ) ,
                                 -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][t,u,r,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daabaab'][t,u,:,s,:,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daabaab'][t,u,:,r,:,:] ) ))
    if s==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,v,w,:,q,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,v,w,:,p,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,q,t,:,v,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaaaa'][r,v,w,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,t,:,:], HD['Daaaaaa'][r,v,w,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,t,:,:], HD['Daaaaaa'][r,v,w,p,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,v,w,q,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,q,t,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,v,w,p,t,:] ) ,
                             +HD['Haa'][r,t] * HD['Daaaa'][p,q,v,w] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][r,:,t,:], HD['Daaaaaa'][p,q,:,v,w,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][p,q,t,:,v,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][r,v,w,:,p,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][r,v,w,:,q,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habab'][r,:,t,:], HD['Daabaab'][p,q,:,v,w,:] ) ))
        if r==t:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][v,w,:,q,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][v,w,:,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][v,w,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][v,w,p,:] ) ,
                                 +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][v,w,q,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daabaab'][v,w,:,p,:,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][v,w,:,q,:,:] ) ))
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][s,v,w,:,q,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][s,v,w,:,p,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,q,u,:,v,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaaaa'][s,v,w,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,u,:,:], HD['Daaaaaa'][s,v,w,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,u,:,:], HD['Daaaaaa'][s,v,w,p,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][s,v,w,p,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][s,v,w,q,u,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,q,u,v,w,:] ) ,
                             +HD['Haa'][s,u] * HD['Daaaa'][p,q,v,w] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][s,:,u,:], HD['Daaaaaa'][p,q,:,v,w,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][s,v,w,:,p,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][p,q,u,:,v,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][s,v,w,:,q,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habab'][s,:,u,:], HD['Daabaab'][p,q,:,v,w,:] ) ))
    if r==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][s,v,w,:,q,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][s,v,w,:,p,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,q,t,:,v,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaaaa'][s,v,w,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,t,:,:], HD['Daaaaaa'][s,v,w,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,t,:,:], HD['Daaaaaa'][s,v,w,p,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][s,v,w,p,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][s,v,w,q,t,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,q,t,v,w,:] ) ,
                             -HD['Haa'][s,t] * HD['Daaaa'][p,q,v,w] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][s,:,t,:], HD['Daaaaaa'][p,q,:,v,w,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][s,v,w,:,p,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][p,q,t,:,v,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][s,v,w,:,q,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habab'][s,:,t,:], HD['Daabaab'][p,q,:,v,w,:] ) ))
        if s==t:
            result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][v,w,:,q,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][v,w,:,p,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][v,w,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][v,w,p,:] ) ,
                                 -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][v,w,q,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daabaab'][v,w,:,p,:,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daabaab'][v,w,:,q,:,:] ) ))
    if s==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,v,w,:,p,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,v,w,:,q,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,q,u,:,v,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaaaa'][r,v,w,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,u,:,:], HD['Daaaaaa'][r,v,w,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,u,:,:], HD['Daaaaaa'][r,v,w,p,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,q,u,v,w,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,v,w,p,u,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,v,w,q,u,:] ) ,
                             -HD['Haa'][r,u] * HD['Daaaa'][p,q,v,w] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][r,:,u,:], HD['Daaaaaa'][p,q,:,v,w,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][p,q,u,:,v,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][r,v,w,:,p,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Daaabaaab'][r,v,w,:,q,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habab'][r,:,u,:], HD['Daabaab'][p,q,:,v,w,:] ) ))
    if p==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][q,v,w,:,r,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][q,v,w,:,s,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,s,t,:,v,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][q,v,w,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,t,:,:], HD['Daaaaaa'][q,v,w,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,t,:,:], HD['Daaaaaa'][q,v,w,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,s,t,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][q,v,w,r,t,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][q,v,w,s,t,:] ) ,
                             +HD['Haa'][q,t] * HD['Daaaa'][r,s,v,w] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][q,:,t,:], HD['Daaaaaa'][r,s,:,v,w,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][q,v,w,:,r,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][r,s,t,:,v,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][q,v,w,:,s,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habab'][q,:,t,:], HD['Daabaab'][r,s,:,v,w,:] ) ))
    if p==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][q,v,w,:,r,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][q,v,w,:,s,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,s,u,:,v,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][q,v,w,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,u,:,:], HD['Daaaaaa'][q,v,w,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,u,:,:], HD['Daaaaaa'][q,v,w,s,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][q,v,w,r,u,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][q,v,w,s,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,s,u,v,w,:] ) ,
                             -HD['Haa'][q,u] * HD['Daaaa'][r,s,v,w] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][q,:,u,:], HD['Daaaaaa'][r,s,:,v,w,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][q,v,w,:,r,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][q,v,w,:,s,u,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][r,s,u,:,v,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habab'][q,:,u,:], HD['Daabaab'][r,s,:,v,w,:] ) ))
    if p==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][q,t,u,:,s,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,s,w,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][q,t,u,:,r,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,w,:,:], HD['Daaaaaa'][q,t,u,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][q,t,u,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,w,:,:], HD['Daaaaaa'][q,t,u,r,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][q,t,u,r,w,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,s,w,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][q,t,u,s,w,:] ) ,
                             +HD['Haa'][q,w] * HD['Daaaa'][r,s,t,u] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][q,:,w,:], HD['Daaaaaa'][r,s,:,t,u,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][r,s,w,:,t,u,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][q,t,u,:,s,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][q,t,u,:,r,w,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habab'][q,:,w,:], HD['Daabaab'][r,s,:,t,u,:] ) ))
    if p==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,s,v,:,t,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][q,t,u,:,r,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][q,t,u,:,s,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][q,t,u,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,v,:,:], HD['Daaaaaa'][q,t,u,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,v,:,:], HD['Daaaaaa'][q,t,u,s,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][q,t,u,r,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,s,v,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][q,t,u,s,v,:] ) ,
                             -HD['Haa'][q,v] * HD['Daaaa'][r,s,t,u] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][q,:,v,:], HD['Daaaaaa'][r,s,:,t,u,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Daaabaaab'][r,s,v,:,t,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Daaabaaab'][q,t,u,:,s,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Daaabaaab'][q,t,u,:,r,v,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habab'][q,:,v,:], HD['Daabaab'][r,s,:,t,u,:] ) ))
    return result


def lucc_aaaa_baba(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['Haaaa'][s,w,:,:], HD['Dbaaaabaa'][t,p,q,u,r,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][s,u,:,:], HD['Dbaaaabaa'][v,p,q,w,r,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][r,w,:,:], HD['Dbaaaabaa'][t,p,q,u,s,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][r,u,:,:], HD['Dbaaaabaa'][v,p,q,w,s,t,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][p,w,:,:], HD['Dbaaaabaa'][t,r,s,u,q,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][p,u,:,:], HD['Dbaaaabaa'][v,r,s,w,q,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][q,w,:,:], HD['Dbaaaabaa'][t,r,s,u,p,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][q,u,:,:], HD['Dbaaaabaa'][v,r,s,w,p,t,:,:] ) ))
    result += 2 * fsum(( +HD['Haa'][p,u] * HD['Dbaaaab'][v,q,w,r,s,t] ,
                         -HD['Haa'][p,w] * HD['Dbaaaab'][t,q,u,r,s,v] ,
                         -HD['Haa'][q,u] * HD['Dbaaaab'][v,p,w,r,s,t] ,
                         +HD['Haa'][q,w] * HD['Dbaaaab'][t,p,u,r,s,v] ,
                         +HD['Haa'][s,u] * HD['Dbaaaab'][t,p,q,r,w,v] ,
                         -HD['Haa'][s,w] * HD['Dbaaaab'][v,p,q,r,u,t] ,
                         -HD['Haa'][r,u] * HD['Dbaaaab'][t,p,q,s,w,v] ,
                         +HD['Haa'][r,w] * HD['Dbaaaab'][v,p,q,s,u,t] ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,:,w,:], HD['Dbaaaaaba'][v,p,q,:,s,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,:,u,:], HD['Dbaaaaaba'][t,r,s,:,q,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,:,w,:], HD['Dbaaaaaba'][v,r,s,:,q,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,:,u,:], HD['Dbaaaaaba'][t,r,s,:,p,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,:,w,:], HD['Dbaaaaaba'][v,r,s,:,p,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,:,u,:], HD['Dbaaaaaba'][t,p,q,:,r,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,:,w,:], HD['Dbaaaaaba'][v,p,q,:,r,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,:,u,:], HD['Dbaaaaaba'][t,p,q,:,s,w,v,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][r,s,w,:], HD['Dbaaaba'][v,p,q,u,t,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][r,s,u,:], HD['Dbaaaba'][t,p,q,w,v,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][p,q,u,:], HD['Dbaaaba'][t,r,s,w,v,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][p,q,w,:], HD['Dbaaaba'][v,r,s,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][p,:,u,:], HD['Dbaabaabb'][t,r,s,:,q,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][s,:,w,:], HD['Dbaabaabb'][v,p,q,:,r,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][p,:,w,:], HD['Dbaabaabb'][v,r,s,:,q,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][q,:,u,:], HD['Dbaabaabb'][t,r,s,:,p,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][r,:,u,:], HD['Dbaabaabb'][t,p,q,:,s,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][q,:,w,:], HD['Dbaabaabb'][v,r,s,:,p,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][r,:,w,:], HD['Dbaabaabb'][v,p,q,:,s,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][s,:,u,:], HD['Dbaabaabb'][t,p,q,:,r,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][p,:,t,:], HD['Daaabaaba'][r,s,u,:,q,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][s,:,v,:], HD['Daaabaaba'][p,q,w,:,r,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][p,:,v,:], HD['Daaabaaba'][r,s,w,:,q,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][q,:,t,:], HD['Daaabaaba'][r,s,u,:,p,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][r,:,t,:], HD['Daaabaaba'][p,q,u,:,s,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][q,:,v,:], HD['Daaabaaba'][r,s,w,:,p,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][s,:,t,:], HD['Daaabaaba'][p,q,u,:,r,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][r,:,v,:], HD['Daaabaaba'][p,q,w,:,s,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,s,:,:], HD['Dbaaaaaab'][t,p,q,u,r,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,p,:,:], HD['Dbaaaaaab'][t,r,s,u,q,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,s,:,:], HD['Dbaaaaaab'][v,p,q,w,r,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,p,:,:], HD['Dbaaaaaab'][v,r,s,w,q,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,q,:,:], HD['Dbaaaaaab'][t,r,s,u,p,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,r,:,:], HD['Dbaaaaaab'][t,p,q,u,s,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,q,:,:], HD['Dbaaaaaab'][v,r,s,w,p,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,r,:,:], HD['Dbaaaaaab'][v,p,q,w,s,u,:,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][t,u,p,:], HD['Dbaaaab'][v,q,w,r,s,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][v,w,s,:], HD['Dbaaaab'][t,r,u,p,q,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][v,w,p,:], HD['Dbaaaab'][t,q,u,r,s,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][t,u,q,:], HD['Dbaaaab'][v,p,w,r,s,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][t,u,r,:], HD['Dbaaaab'][v,s,w,p,q,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][v,w,q,:], HD['Dbaaaab'][t,p,u,r,s,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][v,w,r,:], HD['Dbaaaab'][t,s,u,p,q,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][t,u,s,:], HD['Dbaaaab'][v,r,w,p,q,:] ) ))
    if q==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Dbaaaabaa'][t,r,s,:,w,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][v,p,w,:,r,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Dbaaaabaa'][v,p,w,:,s,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Dbaabaa'][v,p,w,t,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Dbaaaba'][t,r,s,w,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Dbaaaba'][v,p,w,s,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][v,p,w,r,t,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Dbaababab'][v,p,w,:,s,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][v,p,w,:,r,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Dbaababab'][t,r,s,:,w,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habba'][p,:,t,:], HD['Daababa'][r,s,:,w,v,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][t,r,:,:], HD['Dbaaaab'][v,p,w,s,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][t,s,:,:], HD['Dbaaaab'][v,p,w,r,:,:] ) ))
    if r==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][t,s,u,:,p,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][v,p,q,:,u,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Dbaaaabaa'][t,s,u,:,q,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Dbaabaa'][t,s,u,v,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Dbaaaba'][t,s,u,q,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][t,s,u,p,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][v,p,q,u,t,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Dbaababab'][t,s,u,:,q,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][v,p,q,:,u,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][t,s,u,:,p,v,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habba'][s,:,v,:], HD['Daababa'][p,q,:,u,t,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][v,q,:,:], HD['Dbaaaab'][t,s,u,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][v,p,:,:], HD['Dbaaaab'][t,s,u,q,:,:] ) ))
    if s==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][t,r,u,:,p,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Dbaaaabaa'][t,r,u,:,q,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Dbaaaabaa'][v,p,q,:,u,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Dbaabaa'][t,r,u,v,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Dbaaaba'][t,r,u,q,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][t,r,u,p,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Dbaaaba'][v,p,q,u,t,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Dbaababab'][v,p,q,:,u,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][t,r,u,:,p,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Dbaababab'][t,r,u,:,q,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habba'][r,:,v,:], HD['Daababa'][p,q,:,u,t,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][v,q,:,:], HD['Dbaaaab'][t,r,u,p,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][v,p,:,:], HD['Dbaaaab'][t,r,u,q,:,:] ) ))
    if q==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][t,p,u,:,r,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Dbaaaabaa'][v,r,s,:,u,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Dbaaaabaa'][t,p,u,:,s,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Dbaabaa'][t,p,u,v,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][r,:], HD['Dbaaaba'][t,p,u,s,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][p,:], HD['Dbaaaba'][v,r,s,u,t,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][t,p,u,r,v,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Dbaababab'][v,r,s,:,u,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][t,p,u,:,r,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Dbaababab'][t,p,u,:,s,v,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habba'][p,:,v,:], HD['Daababa'][r,s,:,u,t,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][v,s,:,:], HD['Dbaaaab'][t,p,u,r,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][v,r,:,:], HD['Dbaaaab'][t,p,u,s,:,:] ) ))
    if r==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][t,p,q,:,w,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Dbaaaabaa'][v,s,w,:,q,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][v,s,w,:,p,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Dbaabaa'][v,s,w,t,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Dbaaaba'][v,s,w,q,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][v,s,w,p,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][t,p,q,w,v,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][t,p,q,:,w,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Dbaababab'][v,s,w,:,q,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][v,s,w,:,p,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habba'][s,:,t,:], HD['Daababa'][p,q,:,w,v,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][t,p,:,:], HD['Dbaaaab'][v,s,w,q,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][t,q,:,:], HD['Dbaaaab'][v,s,w,p,:,:] ) ))
    if s==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Dbaaaabaa'][t,p,q,:,w,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Dbaaaabaa'][v,r,w,:,q,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][v,r,w,:,p,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Dbaabaa'][v,r,w,t,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][r,:], HD['Dbaaaba'][t,p,q,w,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][p,:], HD['Dbaaaba'][v,r,w,q,t,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][v,r,w,p,t,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][p,:,:,:], HD['Dbaababab'][v,r,w,:,q,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Dbaababab'][t,p,q,:,w,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][v,r,w,:,p,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habba'][r,:,t,:], HD['Daababa'][p,q,:,w,v,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][t,p,:,:], HD['Dbaaaab'][v,r,w,q,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][t,q,:,:], HD['Dbaaaab'][v,r,w,p,:,:] ) ))
    if p==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Dbaaaabaa'][v,q,w,:,s,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][v,q,w,:,r,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][t,r,s,:,w,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Dbaabaa'][v,q,w,t,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][t,r,s,w,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][v,q,w,r,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Dbaaaba'][v,q,w,s,t,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Dbaababab'][v,q,w,:,s,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][t,r,s,:,w,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][v,q,w,:,r,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habba'][q,:,t,:], HD['Daababa'][r,s,:,w,v,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][t,r,:,:], HD['Dbaaaab'][v,q,w,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][t,s,:,:], HD['Dbaaaab'][v,q,w,r,:,:] ) ))
    if p==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][v,r,s,:,u,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][t,q,u,:,r,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Dbaaaabaa'][t,q,u,:,s,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Dbaabaa'][t,q,u,v,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][r,:], HD['Dbaaaba'][t,q,u,s,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][v,r,s,u,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][t,q,u,r,v,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][t,q,u,:,r,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][v,r,s,:,u,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][r,:,:,:], HD['Dbaababab'][t,q,u,:,s,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habba'][q,:,v,:], HD['Daababa'][r,s,:,u,t,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][v,s,:,:], HD['Dbaaaab'][t,q,u,r,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][v,r,:,:], HD['Dbaaaab'][t,q,u,s,:,:] ) ))
    return result


def lucc_aaaa_bbbb(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += 2 * fsum(( -einsum( 'ij,ij->', HD['Habba'][p,:,t,:], HD['Dbaababba'][u,r,s,:,q,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][p,:,u,:], HD['Dbaababba'][t,r,s,:,q,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][p,:,w,:], HD['Dbaababba'][v,r,s,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][p,:,v,:], HD['Dbaababba'][w,r,s,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][q,:,t,:], HD['Dbaababba'][u,r,s,:,p,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][q,:,u,:], HD['Dbaababba'][t,r,s,:,p,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][q,:,w,:], HD['Dbaababba'][v,r,s,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][q,:,v,:], HD['Dbaababba'][w,r,s,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][s,:,t,:], HD['Dbaababba'][u,p,q,:,r,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][s,:,u,:], HD['Dbaababba'][t,p,q,:,r,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][s,:,w,:], HD['Dbaababba'][v,p,q,:,r,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][s,:,v,:], HD['Dbaababba'][w,p,q,:,r,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][r,:,t,:], HD['Dbaababba'][u,p,q,:,s,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][r,:,u,:], HD['Dbaababba'][t,p,q,:,s,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][r,:,w,:], HD['Dbaababba'][v,p,q,:,s,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][r,:,v,:], HD['Dbaababba'][w,p,q,:,s,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][w,p,:,:], HD['Dbbaaabab'][t,u,r,s,q,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,p,:,:], HD['Dbbaaabab'][t,u,r,s,q,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,p,:,:], HD['Dbbaaabab'][v,w,r,s,q,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][u,p,:,:], HD['Dbbaaabab'][v,w,r,s,q,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][w,q,:,:], HD['Dbbaaabab'][t,u,r,s,p,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,q,:,:], HD['Dbbaaabab'][t,u,r,s,p,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,q,:,:], HD['Dbbaaabab'][v,w,r,s,p,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][u,q,:,:], HD['Dbbaaabab'][v,w,r,s,p,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][w,s,:,:], HD['Dbbaaabab'][t,u,p,q,r,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,s,:,:], HD['Dbbaaabab'][t,u,p,q,r,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,s,:,:], HD['Dbbaaabab'][v,w,p,q,r,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][u,s,:,:], HD['Dbbaaabab'][v,w,p,q,r,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][w,r,:,:], HD['Dbbaaabab'][t,u,p,q,s,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,r,:,:], HD['Dbbaaabab'][t,u,p,q,s,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,r,:,:], HD['Dbbaaabab'][v,w,p,q,s,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][u,r,:,:], HD['Dbbaaabab'][v,w,p,q,s,t,:,:] ) ))
    return result


def lucc_baba(HD, *e):
    (p,q,r,s) = e
    result = 0
    result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaabaa'][r,s,:,p,:,:] ) ,
                     -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaabaa'][p,q,:,r,:,:] ) ,
                     -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbababb'][r,s,:,q,:,:] ) ,
                     +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbababb'][p,q,:,s,:,:] ) ))
    result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaba'][r,s,p,:] ) ,
                         -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaba'][p,q,r,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbabbab'][p,q,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbabbab'][r,s,:,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaab'][p,q,:,s,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaab'][r,s,:,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaab'][r,s,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaab'][p,q,:,:] ) ,
                         -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaab'][r,s,q,:] ) ,
                         +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaab'][p,q,s,:] ) ))
    return result


def lucc_baba_aa(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Haaaa'][s,v,:,:], HD['Dbaabaa'][p,q,t,r,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][s,t,:,:], HD['Dbaabaa'][p,q,v,r,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][q,v,:,:], HD['Dbaabaa'][r,s,t,p,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][q,t,:,:], HD['Dbaabaa'][r,s,v,p,:,:] ) ))
    result += 2 * fsum(( -HD['Haa'][q,t] * HD['Dbaab'][p,v,s,r] ,
                         +HD['Haa'][q,v] * HD['Dbaab'][p,t,s,r] ,
                         +HD['Haa'][s,t] * HD['Dbaab'][p,q,v,r] ,
                         -HD['Haa'][s,v] * HD['Dbaab'][p,q,t,r] ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,:,t,:], HD['Dbaaaba'][r,s,:,v,p,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,:,v,:], HD['Dbaaaba'][r,s,:,t,p,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,:,t,:], HD['Dbaaaba'][p,q,:,v,r,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,:,v,:], HD['Dbaaaba'][p,q,:,t,r,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][s,:,t,:], HD['Dbababb'][p,q,:,v,r,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][s,:,v,:], HD['Dbababb'][p,q,:,t,r,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][q,:,t,:], HD['Dbababb'][r,s,:,v,p,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][q,:,v,:], HD['Dbababb'][r,s,:,t,p,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,:,t,:], HD['Dbaaaab'][r,s,:,q,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,:,v,:], HD['Dbaaaab'][r,s,:,q,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,t,:], HD['Dbaaaab'][p,q,:,s,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,v,:], HD['Dbaaaab'][p,q,:,s,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,v,:,:], HD['Dbaaaab'][p,q,t,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,v,:,:], HD['Dbaaaab'][r,s,t,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,t,:,:], HD['Dbaaaab'][r,s,v,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,t,:,:], HD['Dbaaaab'][p,q,v,s,:,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][p,q,t,:], HD['Dbaab'][r,s,v,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][p,q,v,:], HD['Dbaab'][r,s,t,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][r,s,t,:], HD['Dbaab'][p,q,v,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][r,s,v,:], HD['Dbaab'][p,q,t,:] ) ))
    if s==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaabaa'][r,t,:,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbababb'][r,t,:,q,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbababb'][p,q,:,t,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaba'][r,t,p,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbabbab'][r,t,:,p,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaab'][r,t,:,q,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaab'][p,q,:,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaab'][r,t,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaab'][r,t,q,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaab'][p,q,t,:] ) ))
    if s==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaabaa'][r,v,:,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbababb'][p,q,:,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbababb'][r,v,:,q,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaba'][r,v,p,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbabbab'][r,v,:,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaab'][r,v,:,q,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaab'][p,q,:,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaab'][r,v,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaab'][r,v,q,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaab'][p,q,v,:] ) ))
    if q==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaabaa'][p,t,:,r,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbababb'][p,t,:,s,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbababb'][r,s,:,t,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaba'][p,t,r,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbabbab'][p,t,:,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaab'][r,s,:,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaab'][p,t,:,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaab'][p,t,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaab'][r,s,t,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaab'][p,t,s,:] ) ))
    if q==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaabaa'][p,v,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbababb'][p,v,:,s,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbababb'][r,s,:,v,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaba'][p,v,r,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbabbab'][p,v,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaab'][r,s,:,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaab'][p,v,:,s,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaab'][p,v,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaab'][r,s,v,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaab'][p,v,s,:] ) ))
    return result


def lucc_baba_bb(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbaabb'][p,t,q,s,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbaabb'][p,v,q,s,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbaabb'][r,t,s,q,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbaabb'][r,v,s,q,:,:] ) ))
    result += 2 * fsum(( +einsum( 'ij,ij->', HD['Habba'][q,:,t,:], HD['Dbabbba'][r,s,:,p,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][s,:,v,:], HD['Dbabbba'][p,q,:,r,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][q,:,v,:], HD['Dbabbba'][r,s,:,p,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][s,:,t,:], HD['Dbabbba'][p,q,:,r,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,q,:,:], HD['Dbbabab'][r,t,s,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,q,:,:], HD['Dbbabab'][r,v,s,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,s,:,:], HD['Dbbabab'][p,v,q,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,s,:,:], HD['Dbbabab'][p,t,q,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,t,:], HD['Dbaaaba'][p,q,:,s,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,v,:], HD['Dbaaaba'][p,q,:,s,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][p,:,t,:], HD['Dbaaaba'][r,s,:,q,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][p,:,v,:], HD['Dbaaaba'][r,s,:,q,t,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][r,s,t,:], HD['Dbaba'][p,q,v,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][r,s,v,:], HD['Dbaba'][p,q,t,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][p,q,t,:], HD['Dbaba'][r,s,v,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][p,q,v,:], HD['Dbaba'][r,s,t,:] ) ,
                         -HD['Hbb'][p,t] * HD['Dbaab'][v,q,s,r] ,
                         +HD['Hbb'][p,v] * HD['Dbaab'][t,q,s,r] ,
                         +HD['Hbb'][r,t] * HD['Dbaab'][p,q,s,v] ,
                         -HD['Hbb'][r,v] * HD['Dbaab'][p,q,s,t] ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,t,:], HD['Dbababb'][r,s,:,q,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,v,:], HD['Dbababb'][r,s,:,q,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,t,:], HD['Dbababb'][p,q,:,s,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,v,:], HD['Dbababb'][p,q,:,s,t,:] ) ))
    if p==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaabaa'][v,q,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaabaa'][r,s,:,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbababb'][v,q,:,s,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaba'][r,s,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaba'][v,q,r,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbabbab'][r,s,:,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbabbab'][v,q,:,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaab'][v,q,:,s,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaab'][v,q,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaab'][v,q,s,:] ) ))
    if r==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaabaa'][t,s,:,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaabaa'][p,q,:,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbababb'][t,s,:,q,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaba'][t,s,p,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaba'][p,q,t,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbabbab'][t,s,:,p,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbabbab'][p,q,:,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaab'][t,s,:,q,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaab'][t,s,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaab'][t,s,q,:] ) ))
    if p==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaabaa'][r,s,:,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaabaa'][t,q,:,r,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbababb'][t,q,:,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaba'][r,s,t,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaba'][t,q,r,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbabbab'][r,s,:,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbabbab'][t,q,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaab'][t,q,:,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaab'][t,q,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaab'][t,q,s,:] ) ))
    if r==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaabaa'][v,s,:,p,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaabaa'][p,q,:,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbababb'][v,s,:,q,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaba'][v,s,p,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaba'][p,q,v,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbabbab'][v,s,:,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbabbab'][p,q,:,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaab'][v,s,:,q,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaab'][v,s,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaab'][v,s,q,:] ) ))
    return result


def lucc_baba_aaaa(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Haaaa'][s,t,:,:], HD['Dbaaaabaa'][p,q,v,w,u,r,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][s,u,:,:], HD['Dbaaaabaa'][p,q,v,w,t,r,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][q,w,:,:], HD['Dbaaaabaa'][r,s,t,u,v,p,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][q,v,:,:], HD['Dbaaaabaa'][r,s,t,u,w,p,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][q,t,:,:], HD['Dbaaaabaa'][r,s,v,w,u,p,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][q,u,:,:], HD['Dbaaaabaa'][r,s,v,w,t,p,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][s,w,:,:], HD['Dbaaaabaa'][p,q,t,u,v,r,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][s,v,:,:], HD['Dbaaaabaa'][p,q,t,u,w,r,:,:] ) ))
    result += 2 * fsum(( -HD['Haa'][q,t] * HD['Dbaaaab'][p,v,w,s,u,r] ,
                         +HD['Haa'][q,u] * HD['Dbaaaab'][p,v,w,s,t,r] ,
                         -HD['Haa'][q,w] * HD['Dbaaaab'][p,t,u,s,v,r] ,
                         +HD['Haa'][q,v] * HD['Dbaaaab'][p,t,u,s,w,r] ,
                         +HD['Haa'][s,t] * HD['Dbaaaab'][p,q,u,v,w,r] ,
                         -HD['Haa'][s,u] * HD['Dbaaaab'][p,q,t,v,w,r] ,
                         +HD['Haa'][s,w] * HD['Dbaaaab'][p,q,v,t,u,r] ,
                         -HD['Haa'][s,v] * HD['Dbaaaab'][p,q,w,t,u,r] ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,:,t,:], HD['Dbaaaaaba'][r,s,u,:,v,w,p,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,:,u,:], HD['Dbaaaaaba'][r,s,t,:,v,w,p,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,:,w,:], HD['Dbaaaaaba'][r,s,v,:,t,u,p,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,:,v,:], HD['Dbaaaaaba'][r,s,w,:,t,u,p,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,:,t,:], HD['Dbaaaaaba'][p,q,u,:,v,w,r,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,:,u,:], HD['Dbaaaaaba'][p,q,t,:,v,w,r,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,:,w,:], HD['Dbaaaaaba'][p,q,v,:,t,u,r,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,:,v,:], HD['Dbaaaaaba'][p,q,w,:,t,u,r,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][v,w,s,:], HD['Dbaaaba'][r,t,u,q,p,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][v,w,q,:], HD['Dbaaaba'][p,t,u,s,r,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][t,u,s,:], HD['Dbaaaba'][r,v,w,q,p,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][t,u,q,:], HD['Dbaaaba'][p,v,w,s,r,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][q,:,t,:], HD['Dbaabaabb'][r,s,u,:,v,w,p,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][q,:,u,:], HD['Dbaabaabb'][r,s,t,:,v,w,p,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][q,:,w,:], HD['Dbaabaabb'][r,s,v,:,t,u,p,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][q,:,v,:], HD['Dbaabaabb'][r,s,w,:,t,u,p,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][s,:,t,:], HD['Dbaabaabb'][p,q,u,:,v,w,r,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][s,:,u,:], HD['Dbaabaabb'][p,q,t,:,v,w,r,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][s,:,w,:], HD['Dbaabaabb'][p,q,v,:,t,u,r,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][s,:,v,:], HD['Dbaabaabb'][p,q,w,:,t,u,r,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,:,u,:], HD['Dbaaaaaab'][r,s,t,:,q,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,u,:], HD['Dbaaaaaab'][p,q,t,:,s,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,:,w,:], HD['Dbaaaaaab'][r,s,v,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,:,v,:], HD['Dbaaaaaab'][r,s,w,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,w,:], HD['Dbaaaaaab'][p,q,v,:,s,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,v,:], HD['Dbaaaaaab'][p,q,w,:,s,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,t,:], HD['Dbaaaaaab'][p,q,u,:,s,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,:,t,:], HD['Dbaaaaaab'][r,s,u,:,q,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,w,:,:], HD['Dbaaaaaab'][r,s,t,u,q,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,v,:,:], HD['Dbaaaaaab'][r,s,t,u,q,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,w,:,:], HD['Dbaaaaaab'][p,q,t,u,s,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,v,:,:], HD['Dbaaaaaab'][p,q,t,u,s,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,t,:,:], HD['Dbaaaaaab'][r,s,v,w,q,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,u,:,:], HD['Dbaaaaaab'][r,s,v,w,q,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,t,:,:], HD['Dbaaaaaab'][p,q,v,w,s,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,u,:,:], HD['Dbaaaaaab'][p,q,v,w,s,t,:,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][r,s,t,:], HD['Dbaaaab'][p,q,u,v,w,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][r,s,u,:], HD['Dbaaaab'][p,q,t,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][r,s,w,:], HD['Dbaaaab'][p,q,v,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][r,s,v,:], HD['Dbaaaab'][p,q,w,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][p,q,t,:], HD['Dbaaaab'][r,s,u,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][p,q,u,:], HD['Dbaaaab'][r,s,t,v,w,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][p,q,w,:], HD['Dbaaaab'][r,s,v,t,u,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][p,q,v,:], HD['Dbaaaab'][r,s,w,t,u,:] ) ))
    if s==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][r,t,u,:,v,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,v,:,:], HD['Dbaabaa'][r,t,u,p,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbaabaabb'][p,q,v,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbaabaabb'][r,t,u,:,q,v,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][r,t,u,v,p,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][r,t,u,:,v,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaaaab'][p,q,v,:,t,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaaaab'][r,t,u,:,q,v,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,:,v,:], HD['Dbaaaab'][p,q,:,t,u,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaaaab'][r,t,u,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,v,:,:], HD['Dbaaaab'][r,t,u,q,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaaaab'][p,q,v,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaaaab'][r,t,u,q,v,:] ) ))
    if s==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][r,t,u,:,w,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,w,:,:], HD['Dbaabaa'][r,t,u,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbaabaabb'][p,q,w,:,t,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbaabaabb'][r,t,u,:,q,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][r,t,u,w,p,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][r,t,u,:,w,p,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaaaab'][p,q,w,:,t,u,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaaaab'][r,t,u,:,q,w,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,:,w,:], HD['Dbaaaab'][p,q,:,t,u,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaaaab'][r,t,u,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,w,:,:], HD['Dbaaaab'][r,t,u,q,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaaaab'][p,q,w,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaaaab'][r,t,u,q,w,:] ) ))
    if s==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][r,v,w,:,u,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,u,:,:], HD['Dbaabaa'][r,v,w,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbaabaabb'][r,v,w,:,q,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbaabaabb'][p,q,u,:,v,w,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][r,v,w,u,p,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][r,v,w,:,u,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaaaab'][p,q,u,:,v,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaaaab'][r,v,w,:,q,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,:,u,:], HD['Dbaaaab'][p,q,:,v,w,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaaaab'][r,v,w,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,u,:,:], HD['Dbaaaab'][r,v,w,q,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaaaab'][r,v,w,q,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaaaab'][p,q,u,v,w,:] ) ))
    if s==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][r,v,w,:,t,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,t,:,:], HD['Dbaabaa'][r,v,w,p,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbaabaabb'][r,v,w,:,q,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbaabaabb'][p,q,t,:,v,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][r,v,w,t,p,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][r,v,w,:,t,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaaaab'][r,v,w,:,q,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaaaab'][p,q,t,:,v,w,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,:,t,:], HD['Dbaaaab'][p,q,:,v,w,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,t,:,:], HD['Dbaaaab'][r,v,w,q,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaaaab'][r,v,w,t,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaaaab'][r,v,w,q,t,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaaaab'][p,q,t,v,w,:] ) ))
    if q==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][p,t,u,:,v,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,v,:,:], HD['Dbaabaa'][p,t,u,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbaabaabb'][r,s,v,:,t,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbaabaabb'][p,t,u,:,s,v,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][p,t,u,v,r,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][p,t,u,:,v,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaaaab'][r,s,v,:,t,u,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaaaab'][p,t,u,:,s,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,:,v,:], HD['Dbaaaab'][r,s,:,t,u,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,v,:,:], HD['Dbaaaab'][p,t,u,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaaaab'][p,t,u,v,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaaaab'][p,t,u,s,v,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaaaab'][r,s,v,t,u,:] ) ))
    if q==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][p,t,u,:,w,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,w,:,:], HD['Dbaabaa'][p,t,u,r,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbaabaabb'][r,s,w,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbaabaabb'][p,t,u,:,s,w,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][p,t,u,w,r,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][p,t,u,:,w,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaaaab'][r,s,w,:,t,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaaaab'][p,t,u,:,s,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,:,w,:], HD['Dbaaaab'][r,s,:,t,u,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,w,:,:], HD['Dbaaaab'][p,t,u,s,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaaaab'][p,t,u,w,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaaaab'][r,s,w,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaaaab'][p,t,u,s,w,:] ) ))
    if q==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][p,v,w,:,u,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,u,:,:], HD['Dbaabaa'][p,v,w,r,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbaabaabb'][p,v,w,:,s,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbaabaabb'][r,s,u,:,v,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][p,v,w,u,r,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][p,v,w,:,u,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaaaab'][r,s,u,:,v,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaaaab'][p,v,w,:,s,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,:,u,:], HD['Dbaaaab'][r,s,:,v,w,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaaaab'][p,v,w,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,u,:,:], HD['Dbaaaab'][p,v,w,s,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaaaab'][p,v,w,s,u,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaaaab'][r,s,u,v,w,:] ) ))
    if q==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][p,v,w,:,t,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,t,:,:], HD['Dbaabaa'][p,v,w,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbaabaabb'][p,v,w,:,s,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbaabaabb'][r,s,t,:,v,w,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][p,v,w,t,r,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][p,v,w,:,t,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaaaab'][r,s,t,:,v,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaaaab'][p,v,w,:,s,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,:,t,:], HD['Dbaaaab'][r,s,:,v,w,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,t,:,:], HD['Dbaaaab'][p,v,w,s,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaaaab'][p,v,w,t,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaaaab'][p,v,w,s,t,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaaaab'][r,s,t,v,w,:] ) ))
    return result


def lucc_baba_baba(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Haaaa'][q,u,:,:], HD['Dbbaabbaa'][r,v,s,w,p,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][s,w,:,:], HD['Dbbaabbaa'][p,t,q,u,r,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][s,u,:,:], HD['Dbbaabbaa'][p,v,q,w,r,t,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][q,w,:,:], HD['Dbbaabbaa'][r,t,s,u,p,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbaaaabb'][r,t,s,u,q,w,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbaaaabb'][r,v,s,w,q,u,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbaaaabb'][p,t,q,u,s,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbaaaabb'][p,v,q,w,s,u,:,:] ) ))
    result += 2 * fsum(( -HD['Haa'][q,u] * HD['Dbbaabb'][p,v,w,s,r,t] ,
                         +HD['Haa'][s,u] * HD['Dbbaabb'][p,t,q,w,r,v] ,
                         -HD['Haa'][s,w] * HD['Dbbaabb'][p,v,q,u,r,t] ,
                         +HD['Haa'][q,w] * HD['Dbbaabb'][p,t,u,s,r,v] ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,:,w,:], HD['Dbbaaabba'][r,v,s,:,u,p,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,:,u,:], HD['Dbbaaabba'][p,t,q,:,w,r,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,:,w,:], HD['Dbbaaabba'][p,v,q,:,u,r,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,:,u,:], HD['Dbbaaabba'][r,t,s,:,w,p,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][s,:,w,:], HD['Dbbababbb'][p,v,q,:,u,r,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][q,:,w,:], HD['Dbbababbb'][r,v,s,:,u,p,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habab'][s,:,u,:], HD['Dbbababbb'][p,t,q,:,w,r,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habab'][q,:,u,:], HD['Dbbababbb'][r,t,s,:,w,p,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][q,:,v,:], HD['Dbaababba'][r,s,w,:,u,p,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][s,:,v,:], HD['Dbaababba'][p,q,w,:,u,r,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][s,:,t,:], HD['Dbaababba'][p,q,u,:,w,r,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][q,:,t,:], HD['Dbaababba'][r,s,u,:,w,p,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,u,:], HD['Dbbaaaabb'][p,t,q,:,s,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,:,w,:], HD['Dbbaaaabb'][r,v,s,:,q,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,w,:], HD['Dbbaaaabb'][p,v,q,:,s,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,:,u,:], HD['Dbbaaaabb'][r,t,s,:,q,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,q,:,:], HD['Dbbaaabab'][r,v,s,w,u,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,s,:,:], HD['Dbbaaabab'][p,t,q,u,w,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,w,:,:], HD['Dbbaaabab'][r,t,s,u,q,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,s,:,:], HD['Dbbaaabab'][p,v,q,w,u,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,u,:,:], HD['Dbbaaabab'][r,v,s,w,q,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,q,:,:], HD['Dbbaaabab'][r,t,s,u,w,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,w,:,:], HD['Dbbaaabab'][p,t,q,u,s,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,u,:,:], HD['Dbbaaabab'][p,v,q,w,s,t,:,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][r,s,u,:], HD['Dbbaabb'][p,t,q,w,v,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][r,s,w,:], HD['Dbbaabb'][p,v,q,u,t,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][p,q,u,:], HD['Dbbaabb'][r,t,s,w,v,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][t,u,s,:], HD['Dbbaabb'][r,v,w,q,p,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][p,q,w,:], HD['Dbbaabb'][r,v,s,u,t,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][v,w,s,:], HD['Dbbaabb'][r,t,u,q,p,:] ) ,
                         +einsum( 'i,i->', HD['Hbaab'][t,u,q,:], HD['Dbbaabb'][p,v,w,s,r,:] ) ,
                         -einsum( 'i,i->', HD['Hbaab'][v,w,q,:], HD['Dbbaabb'][p,t,u,s,r,:] ) ,
                         -HD['Hbaab'][p,q,u,t] * HD['Dbaab'][r,s,w,v] ,
                         +HD['Hbaab'][r,s,u,t] * HD['Dbaab'][p,q,w,v] ,
                         +HD['Hbaab'][p,q,w,v] * HD['Dbaab'][r,s,u,t] ,
                         -HD['Hbaab'][r,s,w,v] * HD['Dbaab'][p,q,u,t] ,
                         +einsum( 'ij,ij->', HD['Hbaba'][p,:,t,:], HD['Dbaaaaaba'][r,s,u,:,q,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][p,:,v,:], HD['Dbaaaaaba'][r,s,w,:,q,u,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,v,:], HD['Dbaaaaaba'][p,q,w,:,s,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,t,:], HD['Dbaaaaaba'][p,q,u,:,s,w,v,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][r,s,t,:], HD['Dbaaaba'][p,q,u,w,v,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][p,q,t,:], HD['Dbaaaba'][r,s,u,w,v,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][r,s,v,:], HD['Dbaaaba'][p,q,w,u,t,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][p,q,v,:], HD['Dbaaaba'][r,s,w,u,t,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][t,u,p,:], HD['Dbaaaba'][v,q,w,s,r,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][v,w,p,:], HD['Dbaaaba'][t,q,u,s,r,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][t,u,r,:], HD['Dbaaaba'][v,s,w,q,p,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][v,w,r,:], HD['Dbaaaba'][t,s,u,q,p,:] ) ,
                         -HD['Hbb'][r,v] * HD['Dbaaaab'][p,q,w,s,u,t] ,
                         +HD['Hbb'][p,v] * HD['Dbaaaab'][t,q,u,s,w,r] ,
                         -HD['Hbb'][p,t] * HD['Dbaaaab'][v,q,w,s,u,r] ,
                         +HD['Hbb'][r,t] * HD['Dbaaaab'][p,q,u,s,w,v] ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,t,:], HD['Dbaabaabb'][r,s,u,:,q,w,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,v,:], HD['Dbaabaabb'][r,s,w,:,q,u,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,t,:], HD['Dbaabaabb'][p,q,u,:,s,w,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,v,:], HD['Dbaabaabb'][p,q,w,:,s,u,t,:] ) ))
    if p==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][r,s,u,:,w,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][v,q,w,:,u,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,u,:,:], HD['Dbaabaa'][v,q,w,r,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbaabaabb'][v,q,w,:,s,u,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][r,s,u,w,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][v,q,w,u,r,:] ) ,
                             +HD['Haa'][q,u] * HD['Dbaab'][r,s,w,v] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][q,:,u,:], HD['Dbaaaba'][r,s,:,w,v,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][v,q,w,:,u,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][r,s,u,:,w,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habab'][q,:,u,:], HD['Dbababb'][r,s,:,w,v,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaaaab'][v,q,w,:,s,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,u,:,:], HD['Dbaaaab'][v,q,w,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaaaab'][v,q,w,u,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaaaab'][v,q,w,s,u,:] ) ))
        if q==u:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaabaa'][v,w,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbababb'][v,w,:,s,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaba'][v,w,r,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbabbab'][v,w,:,r,:,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaab'][v,w,:,s,:,:] ) ,
                                 +einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaab'][v,w,:,:] ) ,
                                 -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaab'][v,w,s,:] ) ))
    if s==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbbaabbaa'][r,t,u,:,p,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][r,t,u,:,q,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][p,v,q,:,u,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbaabb'][r,t,u,q,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbbabba'][r,t,u,p,v,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbbabbbab'][r,t,u,:,p,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][r,t,u,:,q,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][p,v,q,:,u,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbbabab'][r,t,u,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][v,q,:,:], HD['Dbbabab'][r,t,u,p,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaba'][r,:,v,:], HD['Dbaaaba'][p,q,:,u,t,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][r,t,u,q,v,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][p,v,q,u,t,:] ) ,
                             +HD['Hbb'][r,v] * HD['Dbaab'][p,q,u,t] ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][r,:,v,:], HD['Dbababb'][p,q,:,u,t,:] ) ))
        if r==v:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaabaa'][t,u,:,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbababb'][t,u,:,q,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaba'][t,u,p,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbabbab'][t,u,:,p,:,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaab'][t,u,:,q,:,:] ) ,
                                 +einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaab'][t,u,:,:] ) ,
                                 -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaab'][t,u,q,:] ) ))
    if r==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][t,s,u,:,w,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][p,q,w,:,u,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,w,:,:], HD['Dbaabaa'][t,s,u,p,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbaabaabb'][t,s,u,:,q,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][t,s,u,w,p,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][p,q,w,u,t,:] ) ,
                             +HD['Haa'][s,w] * HD['Dbaab'][p,q,u,t] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][s,:,w,:], HD['Dbaaaba'][p,q,:,u,t,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][t,s,u,:,w,p,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][p,q,w,:,u,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habab'][s,:,w,:], HD['Dbababb'][p,q,:,u,t,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaaaab'][t,s,u,:,q,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,w,:,:], HD['Dbaaaab'][t,s,u,q,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaaaab'][t,s,u,w,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaaaab'][t,s,u,q,w,:] ) ))
    if p==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][t,q,u,:,w,r,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][r,s,w,:,u,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,w,:,:], HD['Dbaabaa'][t,q,u,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbaabaabb'][t,q,u,:,s,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][r,s,w,u,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][t,q,u,w,r,:] ) ,
                             -HD['Haa'][q,w] * HD['Dbaab'][r,s,u,t] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][q,:,w,:], HD['Dbaaaba'][r,s,:,u,t,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][r,s,w,:,u,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][t,q,u,:,w,r,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habab'][q,:,w,:], HD['Dbababb'][r,s,:,u,t,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaaaab'][t,q,u,:,s,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaaaab'][t,q,u,w,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,w,:,:], HD['Dbaaaab'][t,q,u,s,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaaaab'][t,q,u,s,w,:] ) ))
        if q==w:
            result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaabaa'][t,u,:,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbababb'][t,u,:,s,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaba'][t,u,r,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbabbab'][t,u,:,r,:,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbaaaab'][t,u,:,s,:,:] ) ,
                                 -einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbaab'][t,u,:,:] ) ,
                                 +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbaab'][t,u,s,:] ) ))
    if s==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbbaabbaa'][r,v,w,:,p,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][r,v,w,:,q,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][p,t,q,:,w,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbaabb'][r,v,w,q,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbbabba'][r,v,w,p,t,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbbabbbab'][r,v,w,:,p,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][r,v,w,:,q,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][p,t,q,:,w,v,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][t,q,:,:], HD['Dbbabab'][r,v,w,p,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbbabab'][r,v,w,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaba'][r,:,t,:], HD['Dbaaaba'][p,q,:,w,v,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][r,v,w,q,t,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][p,t,q,w,v,:] ) ,
                             -HD['Hbb'][r,t] * HD['Dbaab'][p,q,w,v] ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][r,:,t,:], HD['Dbababb'][p,q,:,w,v,:] ) ))
        if r==t:
            result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaabaa'][v,w,:,p,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbababb'][v,w,:,q,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaba'][v,w,p,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbabbab'][v,w,:,p,:,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaab'][v,w,:,q,:,:] ) ,
                                 -einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaab'][v,w,:,:] ) ,
                                 +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaab'][v,w,q,:] ) ))
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbaaaabaa'][p,q,u,:,w,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbaaaabaa'][v,s,w,:,u,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,u,:,:], HD['Dbaabaa'][v,s,w,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbaabaabb'][v,s,w,:,q,u,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbaaaba'][v,s,w,u,p,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbaaaba'][p,q,u,w,v,:] ) ,
                             -HD['Haa'][s,u] * HD['Dbaab'][p,q,w,v] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][s,:,u,:], HD['Dbaaaba'][p,q,:,w,v,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbaababab'][p,q,u,:,w,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbaababab'][v,s,w,:,u,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habab'][s,:,u,:], HD['Dbababb'][p,q,:,w,v,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbaaaaaab'][v,s,w,:,q,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbaaaab'][v,s,w,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,u,:,:], HD['Dbaaaab'][v,s,w,q,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbaaaab'][v,s,w,q,u,:] ) ))
    if q==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbbaabbaa'][p,t,u,:,r,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][r,v,s,:,u,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][p,t,u,:,s,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbaabb'][p,t,u,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbbabba'][p,t,u,r,v,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbbabbbab'][p,t,u,:,r,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][r,v,s,:,u,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][p,t,u,:,s,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbbabab'][p,t,u,v,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][v,s,:,:], HD['Dbbabab'][p,t,u,r,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaba'][p,:,v,:], HD['Dbaaaba'][r,s,:,u,t,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][p,t,u,s,v,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][r,v,s,u,t,:] ) ,
                             -HD['Hbb'][p,v] * HD['Dbaab'][r,s,u,t] ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][p,:,v,:], HD['Dbababb'][r,s,:,u,t,:] ) ))
    if q==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbbaabbaa'][p,v,w,:,r,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][p,v,w,:,s,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][r,t,s,:,w,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbaabb'][p,v,w,s,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbbabba'][p,v,w,r,t,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbbabbbab'][p,v,w,:,r,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][r,t,s,:,w,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][p,v,w,:,s,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][t,s,:,:], HD['Dbbabab'][p,v,w,r,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbbabab'][p,v,w,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaba'][p,:,t,:], HD['Dbaaaba'][r,s,:,w,v,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][r,t,s,w,v,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][p,v,w,s,t,:] ) ,
                             +HD['Hbb'][p,t] * HD['Dbaab'][r,s,w,v] ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][p,:,t,:], HD['Dbababb'][r,s,:,w,v,:] ) ))
    return result


def lucc_baba_bbbb(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['Hbbbb'][p,w,:,:], HD['Dbbbaabbb'][r,t,u,s,q,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbbaabbb'][r,t,u,s,q,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbbaabbb'][r,v,w,s,q,u,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,u,:,:], HD['Dbbbaabbb'][r,v,w,s,q,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][r,w,:,:], HD['Dbbbaabbb'][p,t,u,q,s,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbbaabbb'][p,t,u,q,s,w,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbbaabbb'][p,v,w,q,s,u,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][r,u,:,:], HD['Dbbbaabbb'][p,v,w,q,s,t,:,:] ) ))
    result += 2 * fsum(( -einsum( 'ij,ij->', HD['Habba'][q,:,w,:], HD['Dbbabbbba'][r,v,s,:,p,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][q,:,v,:], HD['Dbbabbbba'][r,w,s,:,p,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][s,:,t,:], HD['Dbbabbbba'][p,u,q,:,r,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][s,:,u,:], HD['Dbbabbbba'][p,t,q,:,r,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][s,:,w,:], HD['Dbbabbbba'][p,v,q,:,r,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][s,:,v,:], HD['Dbbabbbba'][p,w,q,:,r,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Habba'][q,:,t,:], HD['Dbbabbbba'][r,u,s,:,p,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Habba'][q,:,u,:], HD['Dbbabbbba'][r,t,s,:,p,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][t,q,:,:], HD['Dbbbabbab'][r,v,w,s,p,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][u,q,:,:], HD['Dbbbabbab'][r,v,w,s,p,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][w,s,:,:], HD['Dbbbabbab'][p,t,u,q,r,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][v,s,:,:], HD['Dbbbabbab'][p,t,u,q,r,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][t,s,:,:], HD['Dbbbabbab'][p,v,w,q,r,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][u,s,:,:], HD['Dbbbabbab'][p,v,w,q,r,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][w,q,:,:], HD['Dbbbabbab'][r,t,u,s,p,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][v,q,:,:], HD['Dbbbabbab'][r,t,u,s,p,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,t,:], HD['Dbbaaabba'][p,u,q,:,s,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,u,:], HD['Dbbaaabba'][p,t,q,:,s,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,w,:], HD['Dbbaaabba'][p,v,q,:,s,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,v,:], HD['Dbbaaabba'][p,w,q,:,s,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][p,:,t,:], HD['Dbbaaabba'][r,u,s,:,q,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][p,:,u,:], HD['Dbbaaabba'][r,t,s,:,q,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][p,:,w,:], HD['Dbbaaabba'][r,v,s,:,q,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][p,:,v,:], HD['Dbbaaabba'][r,w,s,:,q,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][r,s,t,:], HD['Dbbabba'][p,u,q,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][r,s,u,:], HD['Dbbabba'][p,t,q,v,w,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][r,s,w,:], HD['Dbbabba'][p,v,q,t,u,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][r,s,v,:], HD['Dbbabba'][p,w,q,t,u,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][p,q,t,:], HD['Dbbabba'][r,u,s,v,w,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][p,q,u,:], HD['Dbbabba'][r,t,s,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][p,q,w,:], HD['Dbbabba'][r,v,s,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][p,q,v,:], HD['Dbbabba'][r,w,s,t,u,:] ) ,
                         -HD['Hbb'][p,t] * HD['Dbbaabb'][v,w,q,s,r,u] ,
                         +HD['Hbb'][p,u] * HD['Dbbaabb'][v,w,q,s,r,t] ,
                         -HD['Hbb'][p,w] * HD['Dbbaabb'][t,u,q,s,r,v] ,
                         +HD['Hbb'][p,v] * HD['Dbbaabb'][t,u,q,s,r,w] ,
                         +HD['Hbb'][r,t] * HD['Dbbaabb'][p,u,q,s,v,w] ,
                         -HD['Hbb'][r,u] * HD['Dbbaabb'][p,t,q,s,v,w] ,
                         +HD['Hbb'][r,w] * HD['Dbbaabb'][p,v,q,s,t,u] ,
                         -HD['Hbb'][r,v] * HD['Dbbaabb'][p,w,q,s,t,u] ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,t,:], HD['Dbbababbb'][r,u,s,:,q,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,u,:], HD['Dbbababbb'][r,t,s,:,q,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,w,:], HD['Dbbababbb'][r,v,s,:,q,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,v,:], HD['Dbbababbb'][r,w,s,:,q,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,t,:], HD['Dbbababbb'][p,u,q,:,s,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,u,:], HD['Dbbababbb'][p,t,q,:,s,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,w,:], HD['Dbbababbb'][p,v,q,:,s,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,v,:], HD['Dbbababbb'][p,w,q,:,s,t,u,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][t,u,p,:], HD['Dbbaabb'][v,w,q,s,r,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][v,w,r,:], HD['Dbbaabb'][t,u,s,q,p,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][t,u,r,:], HD['Dbbaabb'][v,w,s,q,p,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][v,w,p,:], HD['Dbbaabb'][t,u,q,s,r,:] ) ))
    if p==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbbaabbaa'][r,u,s,:,v,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbbaabbaa'][v,w,q,:,r,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][v,w,q,:,s,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,u,:,:], HD['Dbbaabb'][v,w,q,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbbabba'][r,u,s,v,w,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbbabba'][v,w,q,r,u,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbbabbbab'][r,u,s,:,v,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbbabbbab'][v,w,q,:,r,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habba'][q,:,u,:], HD['Dbabbba'][r,s,:,v,w,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][v,w,q,:,s,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbbabab'][v,w,q,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][u,s,:,:], HD['Dbbabab'][v,w,q,r,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][v,w,q,s,u,:] ) ))
    if p==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbbaabbaa'][r,t,s,:,v,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbbaabbaa'][v,w,q,:,r,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][v,w,q,:,s,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbaabb'][v,w,q,s,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbbabba'][r,t,s,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbbabba'][v,w,q,r,t,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbbabbbab'][r,t,s,:,v,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbbabbbab'][v,w,q,:,r,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habba'][q,:,t,:], HD['Dbabbba'][r,s,:,v,w,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][v,w,q,:,s,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][t,s,:,:], HD['Dbbabab'][v,w,q,r,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbbabab'][v,w,q,t,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][v,w,q,s,t,:] ) ))
    if r==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbbaabbaa'][t,u,s,:,p,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbbaabbaa'][p,v,q,:,t,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][t,u,s,:,q,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbaabb'][t,u,s,q,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbbabba'][t,u,s,p,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbbabba'][p,v,q,t,u,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbbabbbab'][t,u,s,:,p,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbbabbbab'][p,v,q,:,t,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habba'][s,:,v,:], HD['Dbabbba'][p,q,:,t,u,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][t,u,s,:,q,v,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][v,q,:,:], HD['Dbbabab'][t,u,s,p,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbbabab'][t,u,s,v,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][t,u,s,q,v,:] ) ))
    if r==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbbaabbaa'][p,w,q,:,t,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbbaabbaa'][t,u,s,:,p,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][t,u,s,:,q,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,w,:,:], HD['Dbbaabb'][t,u,s,q,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbbabba'][t,u,s,p,w,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbbabba'][p,w,q,t,u,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbbabbbab'][t,u,s,:,p,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbbabbbab'][p,w,q,:,t,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habba'][s,:,w,:], HD['Dbabbba'][p,q,:,t,u,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][t,u,s,:,q,w,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][w,q,:,:], HD['Dbbabab'][t,u,s,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbbabab'][t,u,s,w,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][t,u,s,q,w,:] ) ))
    if p==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbbaabbaa'][r,v,s,:,t,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbbaabbaa'][t,u,q,:,r,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][t,u,q,:,s,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbaabb'][t,u,q,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbbabba'][r,v,s,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbbabba'][t,u,q,r,v,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbbabbbab'][t,u,q,:,r,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbbabbbab'][r,v,s,:,t,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habba'][q,:,v,:], HD['Dbabbba'][r,s,:,t,u,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][t,u,q,:,s,v,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbbabab'][t,u,q,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][v,s,:,:], HD['Dbbabab'][t,u,q,r,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][t,u,q,s,v,:] ) ))
    if p==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbbaabbaa'][r,w,s,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbbaabbaa'][t,u,q,:,r,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][t,u,q,:,s,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,w,:,:], HD['Dbbaabb'][t,u,q,s,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbbabba'][r,w,s,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbbabba'][t,u,q,r,w,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbbabbbab'][t,u,q,:,r,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbbabbbab'][r,w,s,:,t,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habba'][q,:,w,:], HD['Dbabbba'][r,s,:,t,u,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][t,u,q,:,s,w,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,s,:,:], HD['Dbbabab'][t,u,q,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][w,s,:,:], HD['Dbbabab'][t,u,q,r,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][t,u,q,s,w,:] ) ))
    if r==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbbaabbaa'][v,w,s,:,p,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbbaabbaa'][p,u,q,:,v,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][v,w,s,:,q,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,u,:,:], HD['Dbbaabb'][v,w,s,q,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Dbbabba'][v,w,s,p,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Dbbabba'][p,u,q,v,w,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbbabbbab'][p,u,q,:,v,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbbabbbab'][v,w,s,:,p,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Habba'][s,:,u,:], HD['Dbabbba'][p,q,:,v,w,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][v,w,s,:,q,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][u,q,:,:], HD['Dbbabab'][v,w,s,p,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbbabab'][v,w,s,u,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][v,w,s,q,u,:] ) ))
    if r==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Dbbaabbaa'][v,w,s,:,p,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Dbbaabbaa'][p,t,q,:,v,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][v,w,s,:,q,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbaabb'][v,w,s,q,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Dbbabba'][v,w,s,p,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Dbbabba'][p,t,q,v,w,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Habab'][s,:,:,:], HD['Dbbabbbab'][p,t,q,:,v,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Habab'][q,:,:,:], HD['Dbbabbbab'][v,w,s,:,p,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Habba'][s,:,t,:], HD['Dbabbba'][p,q,:,v,w,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][v,w,s,:,q,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][t,q,:,:], HD['Dbbabab'][v,w,s,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,q,:,:], HD['Dbbabab'][v,w,s,t,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][v,w,s,q,t,:] ) ))
    return result


def lucc_bbbb(HD, *e):
    (p,q,r,s) = e
    result = 0
    result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][r,s,:,q,:,:] ) ,
                     +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbb'][r,s,:,p,:,:] ) ,
                     -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbb'][p,q,:,r,:,:] ) ,
                     +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][p,q,:,s,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbb'][r,s,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbb'][p,q,:,:] ) ))
    result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][r,s,:,q,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbabab'][r,s,:,p,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbabab'][p,q,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][p,q,:,s,:,:] ) ,
                         -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][r,s,q,:] ) ,
                         +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbb'][r,s,p,:] ) ,
                         -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbb'][p,q,r,:] ) ,
                         +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][p,q,s,:] ) ))
    return result


def lucc_bbbb_aa(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += 2 * fsum(( +einsum( 'ij,ij->', HD['Hbaab'][p,:,t,:], HD['Dbbaabb'][r,s,:,v,q,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,:,v,:], HD['Dbbaabb'][r,s,:,t,q,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][q,:,t,:], HD['Dbbaabb'][r,s,:,v,p,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][q,:,v,:], HD['Dbbaabb'][r,s,:,t,p,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][s,:,t,:], HD['Dbbaabb'][p,q,:,v,r,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][s,:,v,:], HD['Dbbaabb'][p,q,:,t,r,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,t,:], HD['Dbbaabb'][p,q,:,v,s,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,v,:], HD['Dbbaabb'][p,q,:,t,s,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,v,:,:], HD['Dbbabab'][r,s,t,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,t,:,:], HD['Dbbabab'][r,s,v,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][q,v,:,:], HD['Dbbabab'][r,s,t,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][q,t,:,:], HD['Dbbabab'][r,s,v,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][s,v,:,:], HD['Dbbabab'][p,q,t,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][s,t,:,:], HD['Dbbabab'][p,q,v,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,v,:,:], HD['Dbbabab'][p,q,t,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,t,:,:], HD['Dbbabab'][p,q,v,s,:,:] ) ))
    return result


def lucc_bbbb_bb(HD, *e):
    (p,q,r,s),(t,v) = e
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbbbbb'][p,q,t,s,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbbbbb'][p,q,v,s,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbbbbb'][r,s,t,q,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbbbbb'][r,s,v,q,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][q,v,:,:], HD['Dbbbbbb'][r,s,t,p,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][q,t,:,:], HD['Dbbbbbb'][r,s,v,p,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][s,v,:,:], HD['Dbbbbbb'][p,q,t,r,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][s,t,:,:], HD['Dbbbbbb'][p,q,v,r,:,:] ) ))
    result += 2 * fsum(( -einsum( 'ij,ij->', HD['Hbaba'][p,:,t,:], HD['Dbbabba'][r,s,:,q,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][p,:,v,:], HD['Dbbabba'][r,s,:,q,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][q,:,t,:], HD['Dbbabba'][r,s,:,p,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][q,:,v,:], HD['Dbbabba'][r,s,:,p,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][s,:,t,:], HD['Dbbabba'][p,q,:,r,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][s,:,v,:], HD['Dbbabba'][p,q,:,r,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,t,:], HD['Dbbabba'][p,q,:,s,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,v,:], HD['Dbbabba'][p,q,:,s,t,:] ) ,
                         -HD['Hbb'][p,t] * HD['Dbbbb'][q,v,r,s] ,
                         +HD['Hbb'][p,v] * HD['Dbbbb'][q,t,r,s] ,
                         +HD['Hbb'][q,t] * HD['Dbbbb'][p,v,r,s] ,
                         -HD['Hbb'][q,v] * HD['Dbbbb'][p,t,r,s] ,
                         -HD['Hbb'][s,t] * HD['Dbbbb'][p,q,r,v] ,
                         +HD['Hbb'][s,v] * HD['Dbbbb'][p,q,r,t] ,
                         +HD['Hbb'][r,t] * HD['Dbbbb'][p,q,s,v] ,
                         -HD['Hbb'][r,v] * HD['Dbbbb'][p,q,s,t] ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,t,:], HD['Dbbbbbb'][r,s,:,q,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,v,:], HD['Dbbbbbb'][r,s,:,q,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][q,:,t,:], HD['Dbbbbbb'][r,s,:,p,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][q,:,v,:], HD['Dbbbbbb'][r,s,:,p,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][s,:,t,:], HD['Dbbbbbb'][p,q,:,r,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][s,:,v,:], HD['Dbbbbbb'][p,q,:,r,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,t,:], HD['Dbbbbbb'][p,q,:,s,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,v,:], HD['Dbbbbbb'][p,q,:,s,t,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][p,q,t,:], HD['Dbbbb'][r,s,v,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][p,q,v,:], HD['Dbbbb'][r,s,t,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][r,s,t,:], HD['Dbbbb'][p,q,v,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][r,s,v,:], HD['Dbbbb'][p,q,t,:] ) ))
    if q==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][r,s,:,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbb'][p,v,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][p,v,:,s,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbb'][p,v,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][r,s,:,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbabab'][p,v,:,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][p,v,:,s,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][r,s,v,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbb'][p,v,r,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][p,v,s,:] ) ))
    if r==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbb'][p,q,:,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][s,t,:,q,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbb'][s,t,:,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbb'][s,t,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][s,t,:,q,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbabab'][s,t,:,p,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbabab'][p,q,:,t,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][s,t,q,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbb'][s,t,p,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbb'][p,q,t,:] ) ))
    if s==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbb'][r,t,:,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][r,t,:,q,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][p,q,:,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbb'][r,t,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][r,t,:,q,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbabab'][r,t,:,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][p,q,:,t,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][r,t,q,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbb'][r,t,p,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][p,q,t,:] ) ))
    if q==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbb'][p,t,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][r,s,:,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][p,t,:,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbb'][p,t,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][r,s,:,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbabab'][p,t,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][p,t,:,s,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][r,s,t,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][p,t,s,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbb'][p,t,r,:] ) ))
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbb'][p,q,:,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][s,v,:,q,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbb'][s,v,:,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbb'][s,v,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][s,v,:,q,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbabab'][s,v,:,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbabab'][p,q,:,v,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][s,v,q,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbb'][s,v,p,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbb'][p,q,v,:] ) ))
    if s==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][p,q,:,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][r,v,:,q,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbb'][r,v,:,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbb'][r,v,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][r,v,:,q,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbabab'][r,v,:,p,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][p,q,:,v,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][r,v,q,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbb'][r,v,p,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][p,q,v,:] ) ))
    if p==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbb'][q,v,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbb'][r,s,:,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][q,v,:,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbb'][q,v,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbabab'][r,s,:,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbabab'][q,v,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][q,v,:,s,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbb'][r,s,v,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][q,v,s,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbb'][q,v,r,:] ) ))
    if p==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbb'][q,t,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][q,t,:,s,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbb'][r,s,:,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbb'][q,t,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbabab'][r,s,:,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbabab'][q,t,:,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][q,t,:,s,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][q,t,s,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbb'][r,s,t,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbb'][q,t,r,:] ) ))
    return result


def lucc_bbbb_aaaa(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += 2 * fsum(( -einsum( 'ij,ij->', HD['Hbaab'][p,:,t,:], HD['Dbbaaaabb'][r,s,u,:,v,w,q,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,:,u,:], HD['Dbbaaaabb'][r,s,t,:,v,w,q,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,:,w,:], HD['Dbbaaaabb'][r,s,v,:,t,u,q,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,:,v,:], HD['Dbbaaaabb'][r,s,w,:,t,u,q,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][q,:,t,:], HD['Dbbaaaabb'][r,s,u,:,v,w,p,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][q,:,u,:], HD['Dbbaaaabb'][r,s,t,:,v,w,p,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][q,:,w,:], HD['Dbbaaaabb'][r,s,v,:,t,u,p,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][q,:,v,:], HD['Dbbaaaabb'][r,s,w,:,t,u,p,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][s,:,t,:], HD['Dbbaaaabb'][p,q,u,:,v,w,r,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][s,:,u,:], HD['Dbbaaaabb'][p,q,t,:,v,w,r,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][s,:,w,:], HD['Dbbaaaabb'][p,q,v,:,t,u,r,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][s,:,v,:], HD['Dbbaaaabb'][p,q,w,:,t,u,r,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,t,:], HD['Dbbaaaabb'][p,q,u,:,v,w,s,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,u,:], HD['Dbbaaaabb'][p,q,t,:,v,w,s,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,w,:], HD['Dbbaaaabb'][p,q,v,:,t,u,s,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,v,:], HD['Dbbaaaabb'][p,q,w,:,t,u,s,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,w,:,:], HD['Dbbaaabab'][r,s,t,u,v,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,v,:,:], HD['Dbbaaabab'][r,s,t,u,w,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,t,:,:], HD['Dbbaaabab'][r,s,v,w,u,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,u,:,:], HD['Dbbaaabab'][r,s,v,w,t,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][q,w,:,:], HD['Dbbaaabab'][r,s,t,u,v,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][q,v,:,:], HD['Dbbaaabab'][r,s,t,u,w,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][q,t,:,:], HD['Dbbaaabab'][r,s,v,w,u,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][q,u,:,:], HD['Dbbaaabab'][r,s,v,w,t,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][s,w,:,:], HD['Dbbaaabab'][p,q,t,u,v,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][s,v,:,:], HD['Dbbaaabab'][p,q,t,u,w,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][s,t,:,:], HD['Dbbaaabab'][p,q,v,w,u,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][s,u,:,:], HD['Dbbaaabab'][p,q,v,w,t,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,w,:,:], HD['Dbbaaabab'][p,q,t,u,v,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,v,:,:], HD['Dbbaaabab'][p,q,t,u,w,s,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,t,:,:], HD['Dbbaaabab'][p,q,v,w,u,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,u,:,:], HD['Dbbaaabab'][p,q,v,w,t,s,:,:] ) ))
    return result


def lucc_bbbb_baba(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Hbbbb'][q,v,:,:], HD['Dbbbaabbb'][r,s,t,u,w,p,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][q,t,:,:], HD['Dbbbaabbb'][r,s,v,w,u,p,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][s,v,:,:], HD['Dbbbaabbb'][p,q,t,u,w,r,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][s,t,:,:], HD['Dbbbaabbb'][p,q,v,w,u,r,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbbaabbb'][p,q,t,u,w,s,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbbaabbb'][p,q,v,w,u,s,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbbaabbb'][r,s,t,u,w,q,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbbaabbb'][r,s,v,w,u,q,:,:] ) ))
    result += 2 * fsum(( +einsum( 'ij,ij->', HD['Hbaab'][s,:,u,:], HD['Dbbbaabbb'][p,q,t,:,w,r,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][s,:,w,:], HD['Dbbbaabbb'][p,q,v,:,u,r,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,:,u,:], HD['Dbbbaabbb'][p,q,t,:,w,s,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,:,w,:], HD['Dbbbaabbb'][p,q,v,:,u,s,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,:,u,:], HD['Dbbbaabbb'][r,s,t,:,w,q,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,:,w,:], HD['Dbbbaabbb'][r,s,v,:,u,q,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][q,:,u,:], HD['Dbbbaabbb'][r,s,t,:,w,p,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][q,:,w,:], HD['Dbbbaabbb'][r,s,v,:,u,p,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][s,w,:,:], HD['Dbbbabbab'][p,q,t,u,r,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][s,u,:,:], HD['Dbbbabbab'][p,q,v,w,r,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][r,w,:,:], HD['Dbbbabbab'][p,q,t,u,s,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][r,u,:,:], HD['Dbbbabbab'][p,q,v,w,s,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][p,w,:,:], HD['Dbbbabbab'][r,s,t,u,q,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][p,u,:,:], HD['Dbbbabbab'][r,s,v,w,q,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaab'][q,w,:,:], HD['Dbbbabbab'][r,s,t,u,p,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaab'][q,u,:,:], HD['Dbbbabbab'][r,s,v,w,p,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][s,:,t,:], HD['Dbbaaabba'][p,q,u,:,w,r,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][s,:,v,:], HD['Dbbaaabba'][p,q,w,:,u,r,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,t,:], HD['Dbbaaabba'][p,q,u,:,w,s,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,v,:], HD['Dbbaaabba'][p,q,w,:,u,s,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][p,:,t,:], HD['Dbbaaabba'][r,s,u,:,w,q,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][p,:,v,:], HD['Dbbaaabba'][r,s,w,:,u,q,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][q,:,t,:], HD['Dbbaaabba'][r,s,u,:,w,p,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][q,:,v,:], HD['Dbbaaabba'][r,s,w,:,u,p,t,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][t,u,s,:], HD['Dbbabba'][r,v,w,p,q,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][v,w,s,:], HD['Dbbabba'][r,t,u,p,q,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][t,u,r,:], HD['Dbbabba'][s,v,w,p,q,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][v,w,r,:], HD['Dbbabba'][s,t,u,p,q,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][t,u,p,:], HD['Dbbabba'][q,v,w,r,s,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][v,w,p,:], HD['Dbbabba'][q,t,u,r,s,:] ) ,
                         -einsum( 'i,i->', HD['Hbaba'][t,u,q,:], HD['Dbbabba'][p,v,w,r,s,:] ) ,
                         +einsum( 'i,i->', HD['Hbaba'][v,w,q,:], HD['Dbbabba'][p,t,u,r,s,:] ) ,
                         +HD['Hbb'][p,t] * HD['Dbbaabb'][q,v,w,u,r,s] ,
                         -HD['Hbb'][p,v] * HD['Dbbaabb'][q,t,u,w,r,s] ,
                         -HD['Hbb'][q,t] * HD['Dbbaabb'][p,v,w,u,r,s] ,
                         +HD['Hbb'][q,v] * HD['Dbbaabb'][p,t,u,w,r,s] ,
                         +HD['Hbb'][s,t] * HD['Dbbaabb'][p,q,u,w,r,v] ,
                         -HD['Hbb'][s,v] * HD['Dbbaabb'][p,q,w,u,r,t] ,
                         -HD['Hbb'][r,t] * HD['Dbbaabb'][p,q,u,w,s,v] ,
                         +HD['Hbb'][r,v] * HD['Dbbaabb'][p,q,w,u,s,t] ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][q,:,t,:], HD['Dbbababbb'][r,s,u,:,w,p,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][q,:,v,:], HD['Dbbababbb'][r,s,w,:,u,p,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][s,:,t,:], HD['Dbbababbb'][p,q,u,:,w,r,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][s,:,v,:], HD['Dbbababbb'][p,q,w,:,u,r,t,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,t,:], HD['Dbbababbb'][p,q,u,:,w,s,v,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,v,:], HD['Dbbababbb'][p,q,w,:,u,s,t,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,t,:], HD['Dbbababbb'][r,s,u,:,w,q,v,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,v,:], HD['Dbbababbb'][r,s,w,:,u,q,t,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][p,q,t,:], HD['Dbbaabb'][r,s,u,w,v,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][p,q,v,:], HD['Dbbaabb'][r,s,w,u,t,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][r,s,t,:], HD['Dbbaabb'][p,q,u,w,v,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][r,s,v,:], HD['Dbbaabb'][p,q,w,u,t,:] ) ))
    if q==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][r,s,u,:,w,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][p,v,w,:,u,s,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbababbb'][p,v,w,:,u,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbaabb'][p,v,w,u,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][r,s,u,:,w,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][p,v,w,:,u,s,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbaaabab'][p,v,w,:,u,r,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,:,u,:], HD['Dbbaabb'][r,s,:,w,v,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,u,:,:], HD['Dbbabab'][p,v,w,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][s,u,:,:], HD['Dbbabab'][p,v,w,r,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][r,s,u,w,v,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][p,v,w,u,s,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbaabb'][p,v,w,u,r,:] ) ))
    if r==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbababbb'][s,t,u,:,w,p,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][s,t,u,:,w,q,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbababbb'][p,q,w,:,u,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbaabb'][s,t,u,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbaaabab'][s,t,u,:,w,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][s,t,u,:,w,q,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbaaabab'][p,q,w,:,u,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][s,:,w,:], HD['Dbbaabb'][p,q,:,u,t,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,w,:,:], HD['Dbbabab'][s,t,u,q,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][q,w,:,:], HD['Dbbabab'][s,t,u,p,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][s,t,u,w,q,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbaabb'][s,t,u,w,p,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbaabb'][p,q,w,u,t,:] ) ))
    if s==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][p,q,w,:,u,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbababbb'][r,t,u,:,w,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][r,t,u,:,w,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbaabb'][r,t,u,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][r,t,u,:,w,q,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][p,q,w,:,u,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbaaabab'][r,t,u,:,w,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,:,w,:], HD['Dbbaabb'][p,q,:,u,t,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,w,:,:], HD['Dbbabab'][r,t,u,q,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][q,w,:,:], HD['Dbbabab'][r,t,u,p,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][r,t,u,w,q,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][p,q,w,u,t,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbaabb'][r,t,u,w,p,:] ) ))
    if q==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbababbb'][p,t,u,:,w,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][r,s,w,:,u,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][p,t,u,:,w,s,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbaabb'][p,t,u,w,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][p,t,u,:,w,s,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][r,s,w,:,u,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbaaabab'][p,t,u,:,w,r,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,:,w,:], HD['Dbbaabb'][r,s,:,u,t,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,w,:,:], HD['Dbbabab'][p,t,u,s,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][s,w,:,:], HD['Dbbabab'][p,t,u,r,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][p,t,u,w,s,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][r,s,w,u,t,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbaabb'][p,t,u,w,r,:] ) ))
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbababbb'][p,q,u,:,w,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbababbb'][s,v,w,:,u,p,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][s,v,w,:,u,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbaabb'][s,v,w,u,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbaaabab'][s,v,w,:,u,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbaaabab'][p,q,u,:,w,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][s,v,w,:,u,q,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][s,:,u,:], HD['Dbbaabb'][p,q,:,w,v,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][q,u,:,:], HD['Dbbabab'][s,v,w,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][p,u,:,:], HD['Dbbabab'][s,v,w,q,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbaabb'][p,q,u,w,v,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][s,v,w,u,q,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbaabb'][s,v,w,u,p,:] ) ))
    if s==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbababbb'][r,v,w,:,u,q,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][p,q,u,:,w,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbababbb'][r,v,w,:,u,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbaabb'][r,v,w,u,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][p,q,u,:,w,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbaaabab'][r,v,w,:,u,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbaaabab'][r,v,w,:,u,q,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,:,u,:], HD['Dbbaabb'][p,q,:,w,v,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][q,u,:,:], HD['Dbbabab'][r,v,w,p,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][p,u,:,:], HD['Dbbabab'][r,v,w,q,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbaabb'][r,v,w,u,q,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbaabb'][r,v,w,u,p,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][p,q,u,w,v,:] ) ))
    if p==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbababbb'][r,s,u,:,w,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][q,v,w,:,u,s,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbababbb'][q,v,w,:,u,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbaabb'][q,v,w,u,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][q,v,w,:,u,s,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbaaabab'][r,s,u,:,w,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbaaabab'][q,v,w,:,u,r,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][q,:,u,:], HD['Dbbaabb'][r,s,:,w,v,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][r,u,:,:], HD['Dbbabab'][q,v,w,s,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][s,u,:,:], HD['Dbbabab'][q,v,w,r,:,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbaabb'][r,s,u,w,v,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][q,v,w,u,s,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbaabb'][q,v,w,u,r,:] ) ))
    if p==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbababbb'][q,t,u,:,w,s,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbababbb'][q,t,u,:,w,r,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbababbb'][r,s,w,:,u,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbaabb'][q,t,u,w,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbaaabab'][r,s,w,:,u,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbaaabab'][q,t,u,:,w,s,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbaaabab'][q,t,u,:,w,r,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][q,:,w,:], HD['Dbbaabb'][r,s,:,u,t,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaab'][r,w,:,:], HD['Dbbabab'][q,t,u,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaab'][s,w,:,:], HD['Dbbabab'][q,t,u,r,:,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbaabb'][r,s,w,u,t,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbaabb'][q,t,u,w,r,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbaabb'][q,t,u,w,s,:] ) ))
    return result


def lucc_bbbb_bbbb(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['Hbbbb'][s,w,:,:], HD['Dbbbbbbbb'][p,q,t,u,r,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][s,v,:,:], HD['Dbbbbbbbb'][p,q,t,u,r,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][s,t,:,:], HD['Dbbbbbbbb'][p,q,v,w,r,u,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][p,w,:,:], HD['Dbbbbbbbb'][r,s,t,u,q,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbbbbbbb'][r,s,t,u,q,w,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][s,u,:,:], HD['Dbbbbbbbb'][p,q,v,w,r,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][r,w,:,:], HD['Dbbbbbbbb'][p,q,t,u,s,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbbbbbbb'][r,s,v,w,q,u,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][p,u,:,:], HD['Dbbbbbbbb'][r,s,v,w,q,t,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbbbbbbb'][p,q,t,u,s,w,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbbbbbbb'][p,q,v,w,s,u,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][q,w,:,:], HD['Dbbbbbbbb'][r,s,t,u,p,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][q,v,:,:], HD['Dbbbbbbbb'][r,s,t,u,p,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][r,u,:,:], HD['Dbbbbbbbb'][p,q,v,w,s,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Hbbbb'][q,t,:,:], HD['Dbbbbbbbb'][r,s,v,w,p,u,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Hbbbb'][q,u,:,:], HD['Dbbbbbbbb'][r,s,v,w,p,t,:,:] ) ))
    result += 2 * fsum(( -einsum( 'ij,ij->', HD['Hbaba'][p,:,t,:], HD['Dbbbabbba'][r,s,u,:,q,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][p,:,u,:], HD['Dbbbabbba'][r,s,t,:,q,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][p,:,w,:], HD['Dbbbabbba'][r,s,v,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][p,:,v,:], HD['Dbbbabbba'][r,s,w,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][q,:,t,:], HD['Dbbbabbba'][r,s,u,:,p,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][q,:,u,:], HD['Dbbbabbba'][r,s,t,:,p,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][q,:,w,:], HD['Dbbbabbba'][r,s,v,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][q,:,v,:], HD['Dbbbabbba'][r,s,w,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][s,:,t,:], HD['Dbbbabbba'][p,q,u,:,r,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][s,:,u,:], HD['Dbbbabbba'][p,q,t,:,r,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][s,:,w,:], HD['Dbbbabbba'][p,q,v,:,r,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][s,:,v,:], HD['Dbbbabbba'][p,q,w,:,r,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,t,:], HD['Dbbbabbba'][p,q,u,:,s,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,u,:], HD['Dbbbabbba'][p,q,t,:,s,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbaba'][r,:,w,:], HD['Dbbbabbba'][p,q,v,:,s,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbaba'][r,:,v,:], HD['Dbbbabbba'][p,q,w,:,s,t,u,:] ) ,
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
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,u,:], HD['Dbbbbbbbb'][p,q,t,:,s,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,w,:], HD['Dbbbbbbbb'][p,q,v,:,s,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,:,v,:], HD['Dbbbbbbbb'][p,q,w,:,s,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,t,:], HD['Dbbbbbbbb'][r,s,u,:,q,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,u,:], HD['Dbbbbbbbb'][r,s,t,:,q,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,:,w,:], HD['Dbbbbbbbb'][r,s,v,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,:,v,:], HD['Dbbbbbbbb'][r,s,w,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][q,:,t,:], HD['Dbbbbbbbb'][r,s,u,:,p,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][q,:,u,:], HD['Dbbbbbbbb'][r,s,t,:,p,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][q,:,w,:], HD['Dbbbbbbbb'][r,s,v,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][q,:,v,:], HD['Dbbbbbbbb'][r,s,w,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][s,:,t,:], HD['Dbbbbbbbb'][p,q,u,:,r,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][s,:,u,:], HD['Dbbbbbbbb'][p,q,t,:,r,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][s,:,w,:], HD['Dbbbbbbbb'][p,q,v,:,r,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][s,:,v,:], HD['Dbbbbbbbb'][p,q,w,:,r,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,:,t,:], HD['Dbbbbbbbb'][p,q,u,:,s,v,w,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][p,q,t,:], HD['Dbbbbbb'][r,s,u,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][p,q,u,:], HD['Dbbbbbb'][r,s,t,v,w,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][v,w,r,:], HD['Dbbbbbb'][s,t,u,p,q,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][p,q,w,:], HD['Dbbbbbb'][r,s,v,t,u,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][p,q,v,:], HD['Dbbbbbb'][r,s,w,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][t,u,p,:], HD['Dbbbbbb'][q,v,w,r,s,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][v,w,p,:], HD['Dbbbbbb'][q,t,u,r,s,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][t,u,q,:], HD['Dbbbbbb'][p,v,w,r,s,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][v,w,q,:], HD['Dbbbbbb'][p,t,u,r,s,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][r,s,t,:], HD['Dbbbbbb'][p,q,u,v,w,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][r,s,u,:], HD['Dbbbbbb'][p,q,t,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][r,s,w,:], HD['Dbbbbbb'][p,q,v,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][r,s,v,:], HD['Dbbbbbb'][p,q,w,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Hbbbb'][t,u,s,:], HD['Dbbbbbb'][r,v,w,p,q,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][v,w,s,:], HD['Dbbbbbb'][r,t,u,p,q,:] ) ,
                         -einsum( 'i,i->', HD['Hbbbb'][t,u,r,:], HD['Dbbbbbb'][s,v,w,p,q,:] ) ,
                         -HD['Hbbbb'][p,q,t,u] * HD['Dbbbb'][r,s,v,w] ,
                         +HD['Hbbbb'][r,s,t,u] * HD['Dbbbb'][p,q,v,w] ,
                         +HD['Hbbbb'][p,q,v,w] * HD['Dbbbb'][r,s,t,u] ,
                         -HD['Hbbbb'][r,s,v,w] * HD['Dbbbb'][p,q,t,u] ))
    if q==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][p,v,w,:,r,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][r,s,t,:,v,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][p,v,w,:,s,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbbbbb'][p,v,w,s,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbbbb'][p,v,w,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][s,t,:,:], HD['Dbbbbbb'][p,v,w,r,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][p,v,w,:,s,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][r,s,t,:,v,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][p,v,w,:,r,t,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaba'][p,:,t,:], HD['Dbbabba'][r,s,:,v,w,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][r,s,t,v,w,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][p,v,w,r,t,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][p,v,w,s,t,:] ) ,
                             -HD['Hbb'][p,t] * HD['Dbbbb'][r,s,v,w] ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][p,:,t,:], HD['Dbbbbbb'][r,s,:,v,w,:] ) ))
        if p==t:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbb'][v,w,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][v,w,:,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbb'][v,w,:,:] ) ))
            result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbabab'][v,w,:,r,:,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][v,w,:,s,:,:] ) ,
                                 +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbb'][v,w,r,:] ) ,
                                 -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][v,w,s,:] ) ))
    if q==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][p,v,w,:,r,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][r,s,u,:,v,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][p,v,w,:,s,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbbbb'][p,v,w,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][s,u,:,:], HD['Dbbbbbb'][p,v,w,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,u,:,:], HD['Dbbbbbb'][p,v,w,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][p,v,w,:,s,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][r,s,u,:,v,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][p,v,w,:,r,u,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaba'][p,:,u,:], HD['Dbbabba'][r,s,:,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][r,s,u,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][p,v,w,r,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][p,v,w,s,u,:] ) ,
                             +HD['Hbb'][p,u] * HD['Dbbbb'][r,s,v,w] ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][p,:,u,:], HD['Dbbbbbb'][r,s,:,v,w,:] ) ))
        if p==u:
            result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbb'][v,w,:,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][v,w,:,s,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbb'][v,w,:,:] ) ))
            result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbabab'][v,w,:,r,:,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][v,w,:,s,:,:] ) ,
                                 -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbb'][v,w,r,:] ) ,
                                 +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][v,w,s,:] ) ))
    if s==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][r,t,u,:,q,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][r,t,u,:,p,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][p,q,w,:,t,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,w,:,:], HD['Dbbbbbb'][r,t,u,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][q,w,:,:], HD['Dbbbbbb'][r,t,u,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbbbb'][r,t,u,w,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][p,q,w,:,t,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][r,t,u,:,p,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][r,t,u,:,q,w,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaba'][r,:,w,:], HD['Dbbabba'][p,q,:,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][p,q,w,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][r,t,u,p,w,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][r,t,u,q,w,:] ) ,
                             +HD['Hbb'][r,w] * HD['Dbbbb'][p,q,t,u] ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][r,:,w,:], HD['Dbbbbbb'][p,q,:,t,u,:] ) ))
        if r==w:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][t,u,:,q,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbb'][t,u,:,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbb'][t,u,:,:] ) ))
            result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][t,u,:,q,:,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbabab'][t,u,:,p,:,:] ) ,
                                 +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][t,u,q,:] ) ,
                                 -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbb'][t,u,p,:] ) ))
    if r==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][s,t,u,:,p,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][p,q,v,:,t,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][s,t,u,:,q,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][q,v,:,:], HD['Dbbbbbb'][s,t,u,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbbbb'][s,t,u,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbbbbb'][s,t,u,q,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][p,q,v,:,t,u,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][s,t,u,:,q,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][s,t,u,:,p,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaba'][s,:,v,:], HD['Dbbabba'][p,q,:,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][s,t,u,q,v,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][s,t,u,p,v,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][p,q,v,t,u,:] ) ,
                             +HD['Hbb'][s,v] * HD['Dbbbb'][p,q,t,u] ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][s,:,v,:], HD['Dbbbbbb'][p,q,:,t,u,:] ) ))
    if s==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][r,t,u,:,q,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][r,t,u,:,p,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][p,q,v,:,t,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][q,v,:,:], HD['Dbbbbbb'][r,t,u,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbbbb'][r,t,u,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,v,:,:], HD['Dbbbbbb'][r,t,u,q,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][r,t,u,:,p,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][r,t,u,:,q,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][p,q,v,:,t,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaba'][r,:,v,:], HD['Dbbabba'][p,q,:,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][p,q,v,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][r,t,u,p,v,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][r,t,u,q,v,:] ) ,
                             -HD['Hbb'][r,v] * HD['Dbbbb'][p,q,t,u] ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][r,:,v,:], HD['Dbbbbbb'][p,q,:,t,u,:] ) ))
        if r==v:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbb'][t,u,:,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][t,u,:,q,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbb'][t,u,:,:] ) ))
            result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbabab'][t,u,:,p,:,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][t,u,:,q,:,:] ) ,
                                 +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbb'][t,u,p,:] ) ,
                                 -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][t,u,q,:] ) ))
    if r==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][s,t,u,:,p,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][p,q,w,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][s,t,u,:,q,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][q,w,:,:], HD['Dbbbbbb'][s,t,u,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbbbb'][s,t,u,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,w,:,:], HD['Dbbbbbb'][s,t,u,q,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][s,t,u,:,p,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][p,q,w,:,t,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][s,t,u,:,q,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaba'][s,:,w,:], HD['Dbbabba'][p,q,:,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][p,q,w,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][s,t,u,p,w,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][s,t,u,q,w,:] ) ,
                             -HD['Hbb'][s,w] * HD['Dbbbb'][p,q,t,u] ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][s,:,w,:], HD['Dbbbbbb'][p,q,:,t,u,:] ) ))
    if q==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][p,t,u,:,r,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][p,t,u,:,s,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][r,s,w,:,t,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,w,:,:], HD['Dbbbbbb'][p,t,u,s,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbbbb'][p,t,u,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][s,w,:,:], HD['Dbbbbbb'][p,t,u,r,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][p,t,u,:,r,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][r,s,w,:,t,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][p,t,u,:,s,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaba'][p,:,w,:], HD['Dbbabba'][r,s,:,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][p,t,u,r,w,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][r,s,w,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][p,t,u,s,w,:] ) ,
                             -HD['Hbb'][p,w] * HD['Dbbbb'][r,s,t,u] ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][p,:,w,:], HD['Dbbbbbb'][r,s,:,t,u,:] ) ))
        if p==w:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbb'][t,u,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][t,u,:,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbb'][t,u,:,:] ) ))
            result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][t,u,:,s,:,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbabab'][t,u,:,r,:,:] ) ,
                                 -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][t,u,s,:] ) ,
                                 +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbb'][t,u,r,:] ) ))
    if q==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][p,t,u,:,s,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][r,s,v,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][p,t,u,:,r,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbbbbb'][p,t,u,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbbbb'][p,t,u,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][s,v,:,:], HD['Dbbbbbb'][p,t,u,r,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][p,t,u,:,r,v,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][r,s,v,:,t,u,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][p,t,u,:,s,v,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaba'][p,:,v,:], HD['Dbbabba'][r,s,:,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][p,t,u,r,v,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][r,s,v,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][p,t,u,s,v,:] ) ,
                             +HD['Hbb'][p,v] * HD['Dbbbb'][r,s,t,u] ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][p,:,v,:], HD['Dbbbbbb'][r,s,:,t,u,:] ) ))
        if p==v:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbb'][t,u,:,s,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbb'][t,u,:,r,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbb'][t,u,:,:] ) ))
            result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbabab'][t,u,:,s,:,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbabab'][t,u,:,r,:,:] ) ,
                                 +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbb'][t,u,s,:] ) ,
                                 -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbb'][t,u,r,:] ) ))
    if s==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][r,v,w,:,q,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][r,v,w,:,p,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][p,q,t,:,v,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbbbb'][r,v,w,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbbbbb'][r,v,w,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][q,t,:,:], HD['Dbbbbbb'][r,v,w,p,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][p,q,t,:,v,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][r,v,w,:,p,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][r,v,w,:,q,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaba'][r,:,t,:], HD['Dbbabba'][p,q,:,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][p,q,t,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][r,v,w,p,t,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][r,v,w,q,t,:] ) ,
                             +HD['Hbb'][r,t] * HD['Dbbbb'][p,q,v,w] ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][r,:,t,:], HD['Dbbbbbb'][p,q,:,v,w,:] ) ))
        if r==t:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][v,w,:,q,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbb'][v,w,:,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbb'][v,w,:,:] ) ))
            result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbabab'][v,w,:,p,:,:] ) ,
                                 +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][v,w,:,q,:,:] ) ,
                                 -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbb'][v,w,p,:] ) ,
                                 +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][v,w,q,:] ) ))
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][s,v,w,:,q,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][s,v,w,:,p,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][p,q,u,:,v,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,u,:,:], HD['Dbbbbbb'][s,v,w,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbbbb'][s,v,w,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][q,u,:,:], HD['Dbbbbbb'][s,v,w,p,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][s,v,w,:,p,u,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][s,v,w,:,q,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][p,q,u,:,v,w,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaba'][s,:,u,:], HD['Dbbabba'][p,q,:,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][s,v,w,p,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][s,v,w,q,u,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][p,q,u,v,w,:] ) ,
                             +HD['Hbb'][s,u] * HD['Dbbbb'][p,q,v,w] ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][s,:,u,:], HD['Dbbbbbb'][p,q,:,v,w,:] ) ))
    if r==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][s,v,w,:,q,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][p,q,t,:,v,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][s,v,w,:,p,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbbbb'][s,v,w,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,t,:,:], HD['Dbbbbbb'][s,v,w,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][q,t,:,:], HD['Dbbbbbb'][s,v,w,p,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][s,v,w,:,p,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][s,v,w,:,q,t,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][p,q,t,:,v,w,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaba'][s,:,t,:], HD['Dbbabba'][p,q,:,v,w,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][s,v,w,p,t,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][s,v,w,q,t,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][p,q,t,v,w,:] ) ,
                             -HD['Hbb'][s,t] * HD['Dbbbb'][p,q,v,w] ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][s,:,t,:], HD['Dbbbbbb'][p,q,:,v,w,:] ) ))
        if s==t:
            result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbb'][v,w,:,q,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbb'][v,w,:,p,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbb'][v,w,:,:] ) ))
            result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbabab'][v,w,:,p,:,:] ) ,
                                 -einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbabab'][v,w,:,q,:,:] ) ,
                                 +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbb'][v,w,p,:] ) ,
                                 -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbb'][v,w,q,:] ) ))
    if s==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][r,v,w,:,p,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][p,:,:,:], HD['Dbbbbbbbb'][r,v,w,:,q,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][p,q,u,:,v,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][p,q,:,:], HD['Dbbbbbb'][r,v,w,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][p,u,:,:], HD['Dbbbbbb'][r,v,w,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][q,u,:,:], HD['Dbbbbbb'][r,v,w,p,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][p,q,u,:,v,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][r,v,w,:,p,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][p,:,:,:], HD['Dbbbabbab'][r,v,w,:,q,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaba'][r,:,u,:], HD['Dbbabba'][p,q,:,v,w,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][p,q,u,v,w,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][r,v,w,p,u,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][p,:], HD['Dbbbbbb'][r,v,w,q,u,:] ) ,
                             -HD['Hbb'][r,u] * HD['Dbbbb'][p,q,v,w] ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][r,:,u,:], HD['Dbbbbbb'][p,q,:,v,w,:] ) ))
    if p==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][q,v,w,:,r,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][q,v,w,:,s,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][r,s,t,:,v,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbbbb'][q,v,w,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][s,t,:,:], HD['Dbbbbbb'][q,v,w,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,t,:,:], HD['Dbbbbbb'][q,v,w,s,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][q,v,w,:,r,t,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][r,s,t,:,v,w,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][q,v,w,:,s,t,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaba'][q,:,t,:], HD['Dbbabba'][r,s,:,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][q,v,w,r,t,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][r,s,t,v,w,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][q,v,w,s,t,:] ) ,
                             +HD['Hbb'][q,t] * HD['Dbbbb'][r,s,v,w] ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][q,:,t,:], HD['Dbbbbbb'][r,s,:,v,w,:] ) ))
    if p==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][q,v,w,:,r,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][q,v,w,:,s,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][r,s,u,:,v,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbbbb'][q,v,w,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][s,u,:,:], HD['Dbbbbbb'][q,v,w,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,u,:,:], HD['Dbbbbbb'][q,v,w,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][q,v,w,:,r,u,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][r,s,u,:,v,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][q,v,w,:,s,u,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaba'][q,:,u,:], HD['Dbbabba'][r,s,:,v,w,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][q,v,w,r,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][r,s,u,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][q,v,w,s,u,:] ) ,
                             -HD['Hbb'][q,u] * HD['Dbbbb'][r,s,v,w] ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][q,:,u,:], HD['Dbbbbbb'][r,s,:,v,w,:] ) ))
    if p==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][q,t,u,:,s,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][r,s,w,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][q,t,u,:,r,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,w,:,:], HD['Dbbbbbb'][q,t,u,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbbbb'][q,t,u,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][s,w,:,:], HD['Dbbbbbb'][q,t,u,r,:,:] ) ))
        result += 2 * fsum(( +einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][r,s,w,:,t,u,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][q,t,u,:,s,w,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][q,t,u,:,r,w,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Hbaba'][q,:,w,:], HD['Dbbabba'][r,s,:,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][q,t,u,r,w,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][r,s,w,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][q,t,u,s,w,:] ) ,
                             +HD['Hbb'][q,w] * HD['Dbbbb'][r,s,t,u] ,
                             +einsum( 'ij,ij->', HD['Hbbbb'][q,:,w,:], HD['Dbbbbbb'][r,s,:,t,u,:] ) ))
    if p==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Hbbbb'][s,:,:,:], HD['Dbbbbbbbb'][q,t,u,:,r,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Hbbbb'][q,:,:,:], HD['Dbbbbbbbb'][r,s,v,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Hbbbb'][r,:,:,:], HD['Dbbbbbbbb'][q,t,u,:,s,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][r,s,:,:], HD['Dbbbbbb'][q,t,u,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Hbbbb'][s,v,:,:], HD['Dbbbbbb'][q,t,u,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Hbbbb'][r,v,:,:], HD['Dbbbbbb'][q,t,u,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'ijk,ijk->', HD['Hbaab'][s,:,:,:], HD['Dbbbabbab'][q,t,u,:,r,v,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Hbaab'][q,:,:,:], HD['Dbbbabbab'][r,s,v,:,t,u,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Hbaab'][r,:,:,:], HD['Dbbbabbab'][q,t,u,:,s,v,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Hbaba'][q,:,v,:], HD['Dbbabba'][r,s,:,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][q,:], HD['Dbbbbbb'][r,s,v,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Hbb'][r,:], HD['Dbbbbbb'][q,t,u,s,v,:] ) ,
                             +einsum( 'i,i->', HD['Hbb'][s,:], HD['Dbbbbbb'][q,t,u,r,v,:] ) ,
                             -HD['Hbb'][q,v] * HD['Dbbbb'][r,s,t,u] ,
                             -einsum( 'ij,ij->', HD['Hbbbb'][q,:,v,:], HD['Dbbbbbb'][r,s,:,t,u,:] ) ))
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
