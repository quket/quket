from math import fsum
from itertools import product, groupby

from numpy import einsum, array, zeros, isclose, outer, where

from quket.fileio.fileio import prints


####  This is the main driver that call subroutines  ###########################


def do_spinfree(Quket, excitations):
    prints('special_lucc_spinfree')
    H1 = Quket.one_body_integrals_active
    H2 = Quket.two_body_integrals_active
    H2 = 2*(H2.copy().transpose(0,1,3,2) - H2.copy())
    # H2 = <pq|rs>
    # H2 - H2.transpose =  <pq||rs> (for Hermitian?)
    # H2 = 1/2 <pq||rs>

    from ..rdm import get_1RDM_full, get_2RDM_full, get_3RDM_full, get_4RDM_full
    D1, \
    D2, \
    D3, \
    D4 = get_1RDM_full(Quket.state), \
            get_2RDM_full(Quket.state), \
            get_3RDM_full(Quket.state), \
            get_4RDM_full(Quket.state)

    HD = {'Haa':H1, 'Haaaa':H2,
          'Daa':D1, 'Daaaa':D2, 'Daaaaaa':D3, 'Daaaaaaaa':D4}

    excitations = tuple( tuple(x) for x in excitations)
    e_lens = tuple( len(x) for x in excitations)

    edge_len = len(excitations)
    Amat = zeros((edge_len,edge_len))

    #--- [H, tv]
    for i, (tv_excite, tv_len) in enumerate(zip(excitations, e_lens)):
        tv = 'a' * tv_len

        #--- [ [H, pr] , tv]
        for j, (pr_excite, pr_len) in enumerate(zip(excitations, e_lens)):
            pr = 'a' * pr_len

            #--- Compute the actually value of double commutator
            func_key = '_'.join([pr, tv])

            if func_key in FUNCMAPPER:
                Amat[i,j] = FUNCMAPPER[func_key](HD, pr_excite, tv_excite)

    return 0.5 * Amat


####  Subroutinues are under this line  ########################################


def lucc_aa(HD, *e):
    (p,r) = e
    result = 0
    result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaa'][r,:,:,:] ) ,
                     +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaa'][p,:,:,:] ) ))
    result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daa'][r,:] ) ,
                         -einsum( 'i,i->', HD['Haa'][r,:], HD['Daa'][p,:] ) ))
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
                         +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][p,q,s,:] ) ))
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
                         -einsum( 'ij,ij->', HD['Haaaa'][r,:,v,:], HD['Daaaa'][p,:,t,:] ) ))
    if r==v:
        result += +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaa'][t,:,:,:] )
        result += 2 * -einsum( 'i,i->', HD['Haa'][p,:], HD['Daa'][t,:] )
    if r==t:
        result += -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaa'][v,:,:,:] )
        result += 2 * +einsum( 'i,i->', HD['Haa'][p,:], HD['Daa'][v,:] )
    if p==v:
        result += -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaa'][t,:,:,:] )
        result += 2 * +einsum( 'i,i->', HD['Haa'][r,:], HD['Daa'][t,:] )
    if p==t:
        result += +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaa'][v,:,:,:] )
        result += 2 * -einsum( 'i,i->', HD['Haa'][r,:], HD['Daa'][v,:] )
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
                         -einsum( 'i,i->', HD['Haaaa'][t,u,r,:], HD['Daaaa'][v,w,p,:] ) ))
    if r==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][t,u,:,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,v,:,:], HD['Daaaa'][t,u,:,:] ) ))
        result += 2 * -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][t,u,v,:] )
    if r==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][t,u,:,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,w,:,:], HD['Daaaa'][t,u,:,:] ) ))
        result += 2 * +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][t,u,w,:] )
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][v,w,:,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,u,:,:], HD['Daaaa'][v,w,:,:] ) ))
        result += 2 * -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][v,w,u,:] )
    if r==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][v,w,:,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,t,:,:], HD['Daaaa'][v,w,:,:] ) ))
        result += 2 * +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][v,w,t,:] )
    if p==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][t,u,:,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,v,:,:], HD['Daaaa'][t,u,:,:] ) ))
        result += 2 * +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][t,u,v,:] )
    if p==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][t,u,:,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,w,:,:], HD['Daaaa'][t,u,:,:] ) ))
        result += 2 * -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][t,u,w,:] )
    if p==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][v,w,:,u,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,u,:,:], HD['Daaaa'][v,w,:,:] ) ))
        result += 2 * +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][v,w,u,:] )
    if p==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][v,w,:,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,t,:,:], HD['Daaaa'][v,w,:,:] ) ))
        result += 2 * -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][v,w,t,:] )
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
                         +einsum( 'i,i->', HD['Haaaa'][r,s,v,:], HD['Daaaa'][p,q,t,:] ) ))
    if q==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][r,s,:,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][p,v,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][p,v,:,s,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][p,v,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][r,s,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][p,v,r,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][p,v,s,:] ) ))
    if r==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][p,q,:,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][s,t,:,q,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][s,t,:,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][s,t,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][s,t,q,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][s,t,p,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][p,q,t,:] ) ))
    if s==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][r,t,:,q,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][p,q,:,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][r,t,:,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][r,t,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][r,t,q,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][r,t,p,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][p,q,t,:] ) ))
    if q==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][p,t,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][r,s,:,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][p,t,:,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][p,t,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][r,s,t,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][p,t,r,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][p,t,s,:] ) ))
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][p,q,:,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][s,v,:,q,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][s,v,:,p,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][s,v,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][s,v,q,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][s,v,p,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][p,q,v,:] ) ))
    if s==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][p,q,:,v,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][r,v,:,q,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][r,v,:,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][r,v,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][r,v,q,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][r,v,p,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][p,q,v,:] ) ))
    if p==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][q,v,:,r,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][q,v,:,s,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][r,s,:,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][q,v,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][r,s,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][q,v,r,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][q,v,s,:] ) ))
    if p==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][q,t,:,r,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][q,t,:,s,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][r,s,:,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][q,t,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][r,s,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][q,t,r,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][q,t,s,:] ) ))
    return result


def lucc_aaaa_aaaa(HD, *e):
    (p,q,r,s),(t,u,v,w) = e
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['Haaaa'][q,t,:,:], HD['Daaaaaaaa'][r,s,v,w,p,u,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][p,w,:,:], HD['Daaaaaaaa'][r,s,t,u,q,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][q,u,:,:], HD['Daaaaaaaa'][r,s,v,w,p,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][p,v,:,:], HD['Daaaaaaaa'][r,s,t,u,q,w,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][s,u,:,:], HD['Daaaaaaaa'][p,q,v,w,r,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][r,w,:,:], HD['Daaaaaaaa'][p,q,t,u,s,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][r,v,:,:], HD['Daaaaaaaa'][p,q,t,u,s,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][p,t,:,:], HD['Daaaaaaaa'][r,s,v,w,q,u,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][p,u,:,:], HD['Daaaaaaaa'][r,s,v,w,q,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][r,t,:,:], HD['Daaaaaaaa'][p,q,v,w,s,u,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][s,w,:,:], HD['Daaaaaaaa'][p,q,t,u,r,v,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][s,v,:,:], HD['Daaaaaaaa'][p,q,t,u,r,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][r,u,:,:], HD['Daaaaaaaa'][p,q,v,w,s,t,:,:] ) ,
                     +einsum( 'ij,ij->', HD['Haaaa'][q,w,:,:], HD['Daaaaaaaa'][r,s,t,u,p,v,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][q,v,:,:], HD['Daaaaaaaa'][r,s,t,u,p,w,:,:] ) ,
                     -einsum( 'ij,ij->', HD['Haaaa'][s,t,:,:], HD['Daaaaaaaa'][p,q,v,w,r,u,:,:] ) ))
    result += 2 * fsum(( +HD['Haa'][p,t] * HD['Daaaaaa'][q,v,w,r,s,u] ,
                         -HD['Haa'][p,u] * HD['Daaaaaa'][q,v,w,r,s,t] ,
                         -HD['Haa'][r,w] * HD['Daaaaaa'][p,q,v,s,t,u] ,
                         +HD['Haa'][r,v] * HD['Daaaaaa'][p,q,w,s,t,u] ,
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
                         -einsum( 'ij,ij->', HD['Haaaa'][s,:,t,:], HD['Daaaaaaaa'][p,q,u,:,r,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,:,w,:], HD['Daaaaaaaa'][r,s,v,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,:,v,:], HD['Daaaaaaaa'][r,s,w,:,q,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,:,u,:], HD['Daaaaaaaa'][p,q,t,:,r,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,:,w,:], HD['Daaaaaaaa'][p,q,v,:,s,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,:,v,:], HD['Daaaaaaaa'][p,q,w,:,s,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,:,w,:], HD['Daaaaaaaa'][p,q,v,:,r,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,:,t,:], HD['Daaaaaaaa'][p,q,u,:,s,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,:,v,:], HD['Daaaaaaaa'][p,q,w,:,r,t,u,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,:,t,:], HD['Daaaaaaaa'][r,s,u,:,p,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,:,u,:], HD['Daaaaaaaa'][r,s,t,:,p,v,w,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,:,t,:], HD['Daaaaaaaa'][r,s,u,:,q,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,:,u,:], HD['Daaaaaaaa'][r,s,t,:,q,v,w,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,:,w,:], HD['Daaaaaaaa'][r,s,v,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,:,v,:], HD['Daaaaaaaa'][r,s,w,:,p,t,u,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,:,u,:], HD['Daaaaaaaa'][p,q,t,:,s,v,w,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][t,u,s,:], HD['Daaaaaa'][r,v,w,p,q,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][v,w,p,:], HD['Daaaaaa'][q,t,u,r,s,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][v,w,r,:], HD['Daaaaaa'][s,t,u,p,q,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][p,q,t,:], HD['Daaaaaa'][r,s,u,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][p,q,u,:], HD['Daaaaaa'][r,s,t,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][v,w,s,:], HD['Daaaaaa'][r,t,u,p,q,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][t,u,q,:], HD['Daaaaaa'][p,v,w,r,s,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][p,q,w,:], HD['Daaaaaa'][r,s,v,t,u,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][p,q,v,:], HD['Daaaaaa'][r,s,w,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][t,u,p,:], HD['Daaaaaa'][q,v,w,r,s,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][v,w,q,:], HD['Daaaaaa'][p,t,u,r,s,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][t,u,r,:], HD['Daaaaaa'][s,v,w,p,q,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][r,s,t,:], HD['Daaaaaa'][p,q,u,v,w,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][r,s,u,:], HD['Daaaaaa'][p,q,t,v,w,:] ) ,
                         -einsum( 'i,i->', HD['Haaaa'][r,s,w,:], HD['Daaaaaa'][p,q,v,t,u,:] ) ,
                         +einsum( 'i,i->', HD['Haaaa'][r,s,v,:], HD['Daaaaaa'][p,q,w,t,u,:] ) ,
                         -HD['Haaaa'][r,s,v,w] * HD['Daaaa'][p,q,t,u] ,
                         +HD['Haaaa'][p,q,v,w] * HD['Daaaa'][r,s,t,u] ,
                         -HD['Haaaa'][p,q,t,u] * HD['Daaaa'][r,s,v,w] ,
                         +HD['Haaaa'][r,s,t,u] * HD['Daaaa'][p,q,v,w] ))
    if q==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,v,w,:,r,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,v,w,:,s,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,s,t,:,v,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][p,v,w,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,t,:,:], HD['Daaaaaa'][p,v,w,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,t,:,:], HD['Daaaaaa'][p,v,w,s,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,s,t,v,w,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,v,w,r,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,v,w,s,t,:] ) ,
                             -HD['Haa'][p,t] * HD['Daaaa'][r,s,v,w] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][p,:,t,:], HD['Daaaaaa'][r,s,:,v,w,:] ) ))
        if p==t:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][v,w,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][v,w,:,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][v,w,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][v,w,r,:] ) ,
                                 -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][v,w,s,:] ) ))
    if q==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,v,w,:,r,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,s,u,:,v,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,v,w,:,s,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][p,v,w,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,u,:,:], HD['Daaaaaa'][p,v,w,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,u,:,:], HD['Daaaaaa'][p,v,w,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,s,u,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,v,w,r,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,v,w,s,u,:] ) ,
                             +HD['Haa'][p,u] * HD['Daaaa'][r,s,v,w] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][p,:,u,:], HD['Daaaaaa'][r,s,:,v,w,:] ) ))
        if p==u:
            result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][v,w,:,r,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][v,w,:,s,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][v,w,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][v,w,r,:] ) ,
                                 +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][v,w,s,:] ) ))
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
                             +einsum( 'ij,ij->', HD['Haaaa'][r,:,w,:], HD['Daaaaaa'][p,q,:,t,u,:] ) ))
        if r==w:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][t,u,:,q,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][t,u,:,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][t,u,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][t,u,q,:] ) ,
                                 -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][t,u,p,:] ) ))
    if r==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][s,t,u,:,p,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,q,v,:,t,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][s,t,u,:,q,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,v,:,:], HD['Daaaaaa'][s,t,u,p,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaaaa'][s,t,u,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,v,:,:], HD['Daaaaaa'][s,t,u,q,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][s,t,u,q,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][s,t,u,p,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,q,v,t,u,:] ) ,
                             +HD['Haa'][s,v] * HD['Daaaa'][p,q,t,u] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][s,:,v,:], HD['Daaaaaa'][p,q,:,t,u,:] ) ))
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
                             -einsum( 'ij,ij->', HD['Haaaa'][r,:,v,:], HD['Daaaaaa'][p,q,:,t,u,:] ) ))
        if r==v:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][t,u,:,p,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][t,u,:,q,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][t,u,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][t,u,p,:] ) ,
                                 -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][t,u,q,:] ) ))
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
                             -einsum( 'ij,ij->', HD['Haaaa'][s,:,w,:], HD['Daaaaaa'][p,q,:,t,u,:] ) ))
    if q==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,t,u,:,r,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,t,u,:,s,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,s,w,:,t,u,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,w,:,:], HD['Daaaaaa'][p,t,u,s,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][p,t,u,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,w,:,:], HD['Daaaaaa'][p,t,u,r,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,s,w,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,t,u,s,w,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,t,u,r,w,:] ) ,
                             -HD['Haa'][p,w] * HD['Daaaa'][r,s,t,u] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][p,:,w,:], HD['Daaaaaa'][r,s,:,t,u,:] ) ))
        if p==w:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][t,u,:,r,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][t,u,:,s,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][t,u,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][t,u,s,:] ) ,
                                 +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][t,u,r,:] ) ))
    if q==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,t,u,:,s,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,s,v,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,t,u,:,r,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,v,:,:], HD['Daaaaaa'][p,t,u,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][p,t,u,v,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,v,:,:], HD['Daaaaaa'][p,t,u,r,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,s,v,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,t,u,s,v,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,t,u,r,v,:] ) ,
                             +HD['Haa'][p,v] * HD['Daaaa'][r,s,t,u] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][p,:,v,:], HD['Daaaaaa'][r,s,:,t,u,:] ) ))
        if p==v:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaa'][t,u,:,s,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaa'][t,u,:,r,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaa'][t,u,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaa'][t,u,s,:] ) ,
                                 -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaa'][t,u,r,:] ) ))
    if s==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][r,v,w,:,q,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,v,w,:,p,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][p,q,t,:,v,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaaaa'][r,v,w,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,t,:,:], HD['Daaaaaa'][r,v,w,q,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][q,t,:,:], HD['Daaaaaa'][r,v,w,p,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][p,q,t,v,w,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,v,w,p,t,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][r,v,w,q,t,:] ) ,
                             +HD['Haa'][r,t] * HD['Daaaa'][p,q,v,w] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][r,:,t,:], HD['Daaaaaa'][p,q,:,v,w,:] ) ))
        if r==t:
            result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][v,w,:,q,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][v,w,:,p,:,:] ) ,
                             -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][v,w,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][v,w,p,:] ) ,
                                 +einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][v,w,q,:] ) ))
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
                             +einsum( 'ij,ij->', HD['Haaaa'][s,:,u,:], HD['Daaaaaa'][p,q,:,v,w,:] ) ))
    if r==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaaaa'][s,v,w,:,q,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][p,q,t,:,v,w,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][s,v,w,:,p,t,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaaaa'][s,v,w,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][p,t,:,:], HD['Daaaaaa'][s,v,w,q,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][q,t,:,:], HD['Daaaaaa'][s,v,w,p,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][s,v,w,p,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaaaa'][s,v,w,q,t,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][p,q,t,v,w,:] ) ,
                             -HD['Haa'][s,t] * HD['Daaaa'][p,q,v,w] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][s,:,t,:], HD['Daaaaaa'][p,q,:,v,w,:] ) ))
        if s==t:
            result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaaaa'][v,w,:,q,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaa'][v,w,:,p,:,:] ) ,
                             +einsum( 'ij,ij->', HD['Haaaa'][p,q,:,:], HD['Daaaa'][v,w,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaa'][v,w,p,:] ) ,
                                 -einsum( 'i,i->', HD['Haa'][p,:], HD['Daaaa'][v,w,q,:] ) ))
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
                             -einsum( 'ij,ij->', HD['Haaaa'][r,:,u,:], HD['Daaaaaa'][p,q,:,v,w,:] ) ))
    if p==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][q,v,w,:,r,t,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][q,v,w,:,s,t,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,s,t,:,v,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][q,v,w,t,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,t,:,:], HD['Daaaaaa'][q,v,w,r,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,t,:,:], HD['Daaaaaa'][q,v,w,s,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][q,v,w,r,t,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][q,v,w,s,t,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,s,t,v,w,:] ) ,
                             +HD['Haa'][q,t] * HD['Daaaa'][r,s,v,w] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][q,:,t,:], HD['Daaaaaa'][r,s,:,v,w,:] ) ))
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
                             -einsum( 'ij,ij->', HD['Haaaa'][q,:,u,:], HD['Daaaaaa'][r,s,:,v,w,:] ) ))
    if p==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][q,t,u,:,s,w,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,s,w,:,t,u,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][q,t,u,:,r,w,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,w,:,:], HD['Daaaaaa'][q,t,u,s,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][q,t,u,w,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][s,w,:,:], HD['Daaaaaa'][q,t,u,r,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,s,w,t,u,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][q,t,u,s,w,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][q,t,u,r,w,:] ) ,
                             +HD['Haa'][q,w] * HD['Daaaa'][r,s,t,u] ,
                             +einsum( 'ij,ij->', HD['Haaaa'][q,:,w,:], HD['Daaaaaa'][r,s,:,t,u,:] ) ))
    if p==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][q,:,:,:], HD['Daaaaaaaa'][r,s,v,:,t,u,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['Haaaa'][s,:,:,:], HD['Daaaaaaaa'][q,t,u,:,r,v,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaaaaaa'][q,t,u,:,s,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][r,s,:,:], HD['Daaaaaa'][q,t,u,v,:,:] ) ,
                         -einsum( 'ij,ij->', HD['Haaaa'][s,v,:,:], HD['Daaaaaa'][q,t,u,r,:,:] ) ,
                         +einsum( 'ij,ij->', HD['Haaaa'][r,v,:,:], HD['Daaaaaa'][q,t,u,s,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][q,:], HD['Daaaaaa'][r,s,v,t,u,:] ) ,
                             -einsum( 'i,i->', HD['Haa'][r,:], HD['Daaaaaa'][q,t,u,s,v,:] ) ,
                             +einsum( 'i,i->', HD['Haa'][s,:], HD['Daaaaaa'][q,t,u,r,v,:] ) ,
                             -HD['Haa'][q,v] * HD['Daaaa'][r,s,t,u] ,
                             -einsum( 'ij,ij->', HD['Haaaa'][q,:,v,:], HD['Daaaaaa'][r,s,:,t,u,:] ) ))
    return result



FUNCMAPPER = {
    'aa_aa' : lucc_aa_aa, 'aa_aaaa' : lucc_aa_aaaa, 'aaaa_aa' : lucc_aaaa_aa, 'aaaa_aaaa' : lucc_aaaa_aaaa
}
