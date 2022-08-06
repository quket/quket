from math import fsum
from itertools import product, groupby, chain

from numpy import einsum, array, zeros, isclose, outer, where

from quket.fileio.fileio import prints


####  This is the main driver that call subroutines  ###########################


def do_spinfree_extended(Quket, excitations):
    from quket.post.rdm import get_1RDM_full, get_2RDM_full, get_3RDM_full, get_4RDM_full
    from quket.orbital.misc import get_htilde
    from quket.post.lucc_auxiliary.interface import sort_by_space_spin

    prints('general_lucc_spinfree')
    nCor = 2 * ( Quket.n_frozen_orbitals + Quket.n_core_orbitals )
    nAct = 2 * ( Quket.n_active_orbitals )
    nVir = Quket.n_orbitals - (nCor + nAct)
    print(nCor, nAct, nVir)

    ActFloor, \
    ActCeil = nCor, \
              nCor + nAct

    # For active space only
    H1 = Quket.one_body_integrals_active
    H2 = Quket.two_body_integrals_active
    H2 = 2*(H2.copy().transpose(0,1,3,2) - H2.copy())
    # H2 = <pq|rs>
    # H2 - H2.transpose =  <pq||rs> (for Hermitian?)
    # H2 = 1/2 <pq||rs>

    # For whole space
    H1s = Quket.one_body_integrals
    H1s_new = zeros((ActCeil,ActCeil))

    H2s = 0.5 * Quket.two_body_integrals.transpose(0,2,1,3)
    H2_new = zeros((ActCeil,ActCeil,ActCeil,ActCeil))

    for p,r in product(range(ActCeil//2), repeat=2):
        pa = p*2
        pb = pa+1
        ra = r*2
        rb = ra+1
        H1s_new[pa,ra] = H1s_new[pb,rb] = H1s[p,r]

        for q,s in product(range(ActCeil//2), repeat=2):
            qa = q*2
            qb = qa+1
            sa = s*2
            sb = sa+1
            H2_new[pa,qa,ra,sa] = H2_new[pb,qb,rb,sb] = H2_new[pb,qa,ra,sb] = H2_new[pa,qb,rb,sa] = H2s[p,q,s,r]  # pqrs = pqsr here

    H1s, \
    H2s = H1s_new, \
          2*(H2_new.copy().transpose(0,1,3,2) - H2_new.copy())

    # Check H2s == H2
    #print('H1s isclose to H1', isclose( H1s[ActFloor:, ActFloor:] , H1 ).all() )
    #print('H2s isclose to H2', isclose( H2s[ActFloor:, ActFloor:, ActFloor:, ActFloor:] , H2 ).all() )
    #print('H1s isequal to H1', ( (H1s[ActFloor:, ActFloor:]-H1).round(15) == 0 ).all() )
    #print('H2s isequal to H2', ( (H2s[ActFloor:, ActFloor:, ActFloor:, ActFloor:]-H2).round(15) == 0 ).all() )

    D1, \
    D2, \
    D3, \
    D4 = get_1RDM_full(Quket.state), \
            get_2RDM_full(Quket.state), \
            get_3RDM_full(Quket.state), \
            get_4RDM_full(Quket.state)

    print(H1s.shape, H1.shape, D1.shape, Quket.one_body_integrals.shape, Quket.one_body_integrals_active.shape)
    print(H2s.shape, H2.shape, D2.shape, Quket.two_body_integrals.shape, Quket.two_body_integrals_active.shape)


    HD = {'Haa':H1, 'Haaaa':H2,
          'HSaa':H1s, 'HSaaaa':H2s,
          'Daa':D1, 'Daaaa':D2, 'Daaaaaa':D3, 'Daaaaaaaa':D4}

    excitations = tuple( tuple(x) for x in excitations)
    excitations, spacetypes, spintypes, paritys = zip(*(sort_by_space_spin(ActFloor, ActCeil, e)
                                          for e in excitations) )

    edge_len = len(excitations)
    Amat = zeros((edge_len,edge_len))

    #--- [H, tv]
    for i, (tv_excite, tv_space, p1) in enumerate(zip(excitations, spacetypes, paritys)):

        #--- [ [H, pr] , tv]
        for j, (pr_excite, pr_space, p2) in enumerate(zip(excitations, spacetypes, paritys)):

            #--- Compute the actually value of double commutator
            spacetype = (pr_space, tv_space)
            func_key = '_'.join(spacetype)

            if func_key in FUNCMAPPER:
                e = (pr_excite, tv_excite)

                if {'V', 'C'}.isdisjoint(set(chain.from_iterable(spacetype))):
                    # Shifting full-space index to active-space
                    e = [[y-nCor for y in x] for x in e]
                    Amat[i,j] = -FUNCMAPPER[func_key](ActFloor, ActCeil, HD, *e) \
                                if p1^p2 else \
                                FUNCMAPPER[func_key](ActFloor, ActCeil, HD, *e)
                else:
                    Amat[i,j] = -FUNCMAPPER[func_key](ActFloor, ActCeil, HD, *e) \
                                if p1^p2 else \
                                FUNCMAPPER[func_key](ActFloor, ActCeil, HD, *e)

    return 0.5 * Amat


####  Subroutinues are under this line  ########################################


def lucc_AA(actfloor, actceil, HD, *e):
    (p,r) = e
    result = 0
    result += fsum(( -einsum( 'ijk,ijk->', HD['Haaaa'][p,:,:,:], HD['Daaaa'][r,:,:,:] ) ,
                     +einsum( 'ijk,ijk->', HD['Haaaa'][r,:,:,:], HD['Daaaa'][p,:,:,:] ) ))
    result += 2 * fsum(( +einsum( 'i,i->', HD['Haa'][p,:], HD['Daa'][r,:] ) ,
                         -einsum( 'i,i->', HD['Haa'][r,:], HD['Daa'][p,:] ) ))
    return result


def lucc_AA_AA(actfloor, actceil, HD, *e):
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


def lucc_AA_AAAA(actfloor, actceil, HD, *e):
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


def lucc_AAAA(actfloor, actceil, HD, *e):
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


def lucc_AAAA_AA(actfloor, actceil, HD, *e):
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


def lucc_AAAA_AAAA(actfloor, actceil, HD, *e):
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


def lucc_AA_CA(actfloor, actceil, HD, *e):
    (p,r),(T,v) = e
    Dp, Dr, Dv = p-actfloor, r-actfloor, v-actfloor
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['HSaaaa'][p,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dv,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][r,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dp,Dv,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][p,T] * HD['Daa'][Dr,Dv] ,
                         -HD['HSaa'][r,T] * HD['Daa'][Dp,Dv] ,
                         +HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                         -HD['Daa'][Dp,Dv] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dp,:,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,v,T,actfloor:actceil], HD['Daa'][Dp,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,T,v,actfloor:actceil], HD['Daa'][Dp,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][p,v,T,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][p,T,v,actfloor:actceil], HD['Daa'][Dr,:] ) ))
    if r==v:
        result += 2 * fsum(( -HD['HSaa'][p,T] ,
                             -einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daa'][:,:] ) ))
    if p==v:
        result += 2 * fsum(( +HD['HSaa'][r,T] ,
                             +einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daa'][:,:] ) ))
    return result


def lucc_AA_VA(actfloor, actceil, HD, *e):
    (p,r),(T,v) = e
    Dp, Dr, Dv = p-actfloor, r-actfloor, v-actfloor
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][T,p,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dv,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][T,r,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dp,Dv,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][p,T] * HD['Daa'][Dr,Dv] ,
                         -HD['HSaa'][r,T] * HD['Daa'][Dp,Dv] ,
                         -HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,:actfloor,T] ) ,
                         +HD['Daa'][Dp,Dv] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Dr,:,Dv,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Dp,:,Dv,:] ) ))
    return result


def lucc_AA_VC(actfloor, actceil, HD, *e):
    (p,r),(T,V) = e
    Dp, Dr = p-actfloor, r-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][p,V,actfloor:actceil,T], HD['Daa'][Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,p,V,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,V,actfloor:actceil,T], HD['Daa'][Dp,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,r,V,actfloor:actceil], HD['Daa'][Dp,:] ) ))
    return result


def lucc_CA_AA(actfloor, actceil, HD, *e):
    (P,r),(t,v) = e
    Dr, Dt, Dv = r-actfloor, t-actfloor, v-actfloor
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['HSaaaa'][v,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dt,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][t,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dv,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][P,t] * HD['Daa'][Dr,Dv] ,
                         -HD['HSaa'][P,v] * HD['Daa'][Dr,Dt] ,
                         -HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,t] ) ,
                         +HD['Daa'][Dr,Dt] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,t,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,v,actfloor:actceil], HD['Daaaa'][Dr,:,Dt,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][v,P,r,actfloor:actceil], HD['Daa'][Dt,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,t,P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][t,P,r,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,v,P,actfloor:actceil], HD['Daa'][Dt,:] ) ))
    if r==v:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,:,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Dt,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dt,:] ) ))
    if r==t:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ))
    return result


def lucc_CA_CA(actfloor, actceil, HD, *e):
    (P,r),(T,v) = e
    Dr, Dv = r-actfloor, v-actfloor
    result = 0
    result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dv,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Dr,Dv] ,
                         +HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,T,P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][v,P,T,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,r,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                         +HD['HSaaaa'][P,T,r,v] ,
                         +HD['HSaaaa'][v,P,T,r] ))
    if P==T:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             +HD['HSaa'][r,v] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             -einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,v] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,v,actfloor:actceil], HD['Daa'][:,:] ) ))
    if r==v:
        result += 2 * fsum(( -HD['HSaa'][P,T] ,
                             -einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daa'][:,:] ) ))
    return result


def lucc_CA_VA(actfloor, actceil, HD, *e):
    (P,r),(T,v) = e
    Dr, Dv = r-actfloor, v-actfloor
    result = 0
    result += +einsum( 'ij,ij->', HD['HSaaaa'][T,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dv,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Dr,Dv] ,
                         -HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,T], HD['Daaaa'][Dr,:,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,P,r,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,r,P,actfloor:actceil], HD['Daa'][Dv,:] ) ))
    return result


def lucc_CA_VC(actfloor, actceil, HD, *e):
    (P,r),(T,V) = e
    Dr = r-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daa'][Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,P,V,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                         -HD['HSaaaa'][T,P,V,r] ,
                         -HD['HSaaaa'][P,V,r,T] ))
    if P==V:
        result += 2 * fsum(( -HD['HSaa'][r,T] ,
                             +einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
    return result


def lucc_VA_AA(actfloor, actceil, HD, *e):
    (P,r),(t,v) = e
    Dr, Dt, Dv = r-actfloor, t-actfloor, v-actfloor
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][P,v,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dt,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][P,t,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dv,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][P,t] * HD['Daa'][Dr,Dv] ,
                         -HD['HSaa'][P,v] * HD['Daa'][Dr,Dt] ,
                         -HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,t] ) ,
                         +HD['Daa'][Dr,Dt] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,t,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dr,:,Dt,:] ) ))
    if r==v:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,:,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Dt,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dt,:] ) ))
    if r==t:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ))
    return result


def lucc_VA_CA(actfloor, actceil, HD, *e):
    (P,r),(T,v) = e
    Dr, Dv = r-actfloor, v-actfloor
    result = 0
    result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dv,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Dr,Dv] ,
                         +HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,v,T,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daa'][Dr,:] ) ))
    if r==v:
        result += 2 * fsum(( -HD['HSaa'][P,T] ,
                             -einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daa'][:,:] ) ))
    return result


def lucc_VA_VA(actfloor, actceil, HD, *e):
    (P,r),(T,v) = e
    Dr, Dv = r-actfloor, v-actfloor
    result = 0
    result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dv,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Dr,Dv] ,
                         -HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Dr,:,Dv,:] ) ))
    if P==T:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ))
    return result


def lucc_VA_VC(actfloor, actceil, HD, *e):
    (P,r),(T,V) = e
    Dr = r-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daa'][Dr,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,V,actfloor:actceil], HD['Daa'][Dr,:] ) ))
    if P==T:
        result += 2 * fsum(( -HD['HSaa'][r,V] ,
                             -einsum( 'II->', HD['HSaaaa'][r,:actfloor,V,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,V,actfloor:actceil], HD['Daa'][:,:] ) ))
    return result


def lucc_VC_AA(actfloor, actceil, HD, *e):
    (P,R),(t,v) = e
    Dt, Dv = t-actfloor, v-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][t,R,actfloor:actceil,P], HD['Daa'][Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,v,R,actfloor:actceil], HD['Daa'][Dt,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][v,R,actfloor:actceil,P], HD['Daa'][Dt,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,t,R,actfloor:actceil], HD['Daa'][Dv,:] ) ))
    return result


def lucc_VC_CA(actfloor, actceil, HD, *e):
    (P,R),(T,v) = e
    Dv = v-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][R,T,actfloor:actceil,P], HD['Daa'][Dv,:] ) ,
                         -HD['HSaaaa'][P,v,R,T] ,
                         -HD['HSaaaa'][P,T,R,v] ))
    if R==T:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             -HD['HSaa'][P,v] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             +einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daa'][:,:] ) ))
    return result


def lucc_VC_VA(actfloor, actceil, HD, *e):
    (P,R),(T,v) = e
    Dv = v-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,P], HD['Daa'][Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daa'][Dv,:] ) ))
    if P==T:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ))
    return result


def lucc_VC_VC(actfloor, actceil, HD, *e):
    (P,R),(T,V) = e
    result = 0
    result += 2 * fsum(( +HD['HSaaaa'][P,V,R,T] ,
                         +HD['HSaaaa'][P,T,R,V] ))
    if R==V:
        result += 2 * fsum(( +HD['HSaa'][P,T] ,
                             -einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
    if P==T:
        result += 2 * fsum(( -HD['HSaa'][R,V] ,
                             -einsum( 'II->', HD['HSaaaa'][R,:actfloor,V,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,V,actfloor:actceil], HD['Daa'][:,:] ) ))
    return result


def lucc_AA_CAAA(actfloor, actceil, HD, *e):
    (p,r),(T,u,v,w) = e
    Dp, Dr, Du, Dv, Dw = p-actfloor, r-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][p,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Du,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][r,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dv,Dw,Du,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][p,T] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                         -HD['HSaa'][r,T] * HD['Daaaa'][Dp,Du,Dv,Dw] ,
                         -HD['Daaaa'][Dp,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                         +HD['Daaaa'][Dr,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dv,Dw,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dp,Du,:,Dv,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,T,w,actfloor:actceil], HD['Daaaa'][Dp,Dv,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,T,v,actfloor:actceil], HD['Daaaa'][Dp,Dw,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][p,w,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][p,v,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][p,T,w,actfloor:actceil], HD['Daaaa'][Dr,Dv,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][p,T,v,actfloor:actceil], HD['Daaaa'][Dr,Dw,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,w,T,actfloor:actceil], HD['Daaaa'][Dp,Du,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,v,T,actfloor:actceil], HD['Daaaa'][Dp,Du,Dw,:] ) ,
                         -HD['HSaaaa'][p,T,v,w] * HD['Daa'][Dr,Du] ,
                         +HD['HSaaaa'][r,T,v,w] * HD['Daa'][Dp,Du] ))
    if r==w:
        result += 2 * fsum(( -HD['HSaa'][p,T] * HD['Daa'][Du,Dv] ,
                             -HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][p,v,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if r==v:
        result += 2 * fsum(( +HD['HSaa'][p,T] * HD['Daa'][Du,Dw] ,
                             +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][p,w,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if p==w:
        result += 2 * fsum(( +HD['HSaa'][r,T] * HD['Daa'][Du,Dv] ,
                             +HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,v,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if p==v:
        result += 2 * fsum(( -HD['HSaa'][r,T] * HD['Daa'][Du,Dw] ,
                             -HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,w,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    return result


def lucc_AA_CCAA(actfloor, actceil, HD, *e):
    (p,r),(T,U,v,w) = e
    Dp, Dr, Dv, Dw = p-actfloor, r-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dp,:] ) ,
                         -HD['HSaaaa'][p,w,T,U] * HD['Daa'][Dr,Dv] ,
                         +HD['HSaaaa'][p,v,T,U] * HD['Daa'][Dr,Dw] ,
                         +HD['HSaaaa'][r,w,T,U] * HD['Daa'][Dp,Dv] ,
                         -HD['HSaaaa'][r,v,T,U] * HD['Daa'][Dp,Dw] ))
    if r==w:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             -HD['HSaaaa'][p,v,T,U] ))
    if r==v:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                             +HD['HSaaaa'][p,w,T,U] ))
    if p==w:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             +HD['HSaaaa'][r,v,T,U] ))
    if p==v:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                             -HD['HSaaaa'][r,w,T,U] ))
    return result


def lucc_AA_VAAA(actfloor, actceil, HD, *e):
    (p,r),(T,u,v,w) = e
    Dp, Dr, Du, Dv, Dw = p-actfloor, r-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][T,r,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dv,Dw,Du,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][T,p,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Du,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][p,T] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                         -HD['HSaa'][r,T] * HD['Daaaa'][Dp,Du,Dv,Dw] ,
                         -HD['Daaaa'][Dr,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,:actfloor,T] ) ,
                         +HD['Daaaa'][Dp,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dr,Du,:,Dv,Dw,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dp,Du,:,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,u,p,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,u,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dp,:] ) ))
    if p==u:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][T,r,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    if r==u:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][T,p,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    return result


def lucc_AA_VACA(actfloor, actceil, HD, *e):
    (p,r),(T,u,V,w) = e
    Dp, Dr, Du, Dw = p-actfloor, r-actfloor, u-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][p,V,actfloor:actceil,T], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,p,V,actfloor:actceil], HD['Daaaa'][Dr,Dw,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,V,actfloor:actceil,T], HD['Daaaa'][Dp,Du,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,r,V,actfloor:actceil], HD['Daaaa'][Dp,Dw,Du,:] ) ,
                         -HD['HSaaaa'][p,V,u,T] * HD['Daa'][Dr,Dw] ,
                         +HD['HSaaaa'][r,V,u,T] * HD['Daa'][Dp,Dw] ))
    if r==u:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,p,V,actfloor:actceil], HD['Daa'][Dw,:] )
    if p==u:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,r,V,actfloor:actceil], HD['Daa'][Dw,:] )
    return result


def lucc_AA_VACC(actfloor, actceil, HD, *e):
    (p,r),(T,u,V,W) = e
    Dp, Dr, Du = p-actfloor, r-actfloor, u-actfloor
    result = 0
    result += 2 * fsum(( +HD['HSaaaa'][T,p,V,W] * HD['Daa'][Dr,Du] ,
                         -HD['HSaaaa'][T,r,V,W] * HD['Daa'][Dp,Du] ))
    if r==u:
        result += 2 * -HD['HSaaaa'][T,p,V,W]
    if p==u:
        result += 2 * +HD['HSaaaa'][T,r,V,W]
    return result


def lucc_AA_VVAA(actfloor, actceil, HD, *e):
    (p,r),(T,U,v,w) = e
    Dp, Dr, Dv, Dw = p-actfloor, r-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dp,:] ) ))
    return result


def lucc_AA_VVCA(actfloor, actceil, HD, *e):
    (p,r),(T,U,V,w) = e
    Dp, Dr, Dw = p-actfloor, r-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +HD['HSaaaa'][p,V,T,U] * HD['Daa'][Dr,Dw] ,
                         -HD['HSaaaa'][r,V,T,U] * HD['Daa'][Dp,Dw] ))
    return result


def lucc_AA_VVCC(actfloor, actceil, HD, *e):
    (p,r),(T,U,V,W) = e
    Dp, Dr = p-actfloor, r-actfloor
    result = 0
    return result


def lucc_CA_AAAA(actfloor, actceil, HD, *e):
    (P,r),(t,u,v,w) = e
    Dr, Dt, Du, Dv, Dw = r-actfloor, t-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['HSaaaa'][t,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Du,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][u,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dt,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][w,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dv,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][v,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dw,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][P,t] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                         -HD['HSaa'][P,u] * HD['Daaaa'][Dr,Dt,Dv,Dw] ,
                         +HD['HSaa'][P,w] * HD['Daaaa'][Dr,Dv,Dt,Du] ,
                         -HD['HSaa'][P,v] * HD['Daaaa'][Dr,Dw,Dt,Du] ,
                         +HD['Daaaa'][Dr,Dt,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,u] ) ,
                         -HD['Daaaa'][Dr,Dv,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                         +HD['Daaaa'][Dr,Dw,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                         -HD['Daaaa'][Dr,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,t] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,u,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,:,Dv,Dw,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,w,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,:,Dt,Du,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,v,actfloor:actceil], HD['Daaaaaa'][Dr,Dw,:,Dt,Du,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,t,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][t,u,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][v,w,P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dr,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][t,P,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][u,P,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,w,P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,v,P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][w,P,r,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][v,P,r,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,t,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,u,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] ) ))
    if r==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dv,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][v,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,Du,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] ) ))
    if r==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dw,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][w,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,Du,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] ) ))
    if r==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Du,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][u,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ))
    if r==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dt,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][t,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] ) ))
    return result


def lucc_CA_CAAA(actfloor, actceil, HD, *e):
    (P,r),(T,u,v,w) = e
    Dr, Du, Dv, Dw = r-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += +einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Du,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                         +HD['Daaaa'][Dr,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,T,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][w,P,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][v,P,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,w,actfloor:actceil], HD['Daaaa'][Dr,Dv,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daaaa'][Dr,Dw,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                         -HD['HSaaaa'][P,T,v,w] * HD['Daa'][Dr,Du] ,
                         +HD['HSaaaa'][P,T,r,w] * HD['Daa'][Du,Dv] ,
                         -HD['HSaaaa'][P,T,r,v] * HD['Daa'][Du,Dw] ,
                         +HD['HSaaaa'][w,P,T,r] * HD['Daa'][Du,Dv] ,
                         -HD['HSaaaa'][v,P,T,r] * HD['Daa'][Du,Dw] ))
    if P==T:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Du,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             +HD['HSaa'][r,w] * HD['Daa'][Du,Dv] ,
                             -HD['HSaa'][r,v] * HD['Daa'][Du,Dw] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             -HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,w] ) ,
                             +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,v] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,w,actfloor:actceil], HD['Daaaa'][Dv,:,Du,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dw,:,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,w,r,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if r==w:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daa'][Du,Dv] ,
                             -HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaa'][Du,:,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,P,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if r==v:
        result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Du,Dw] ,
                             +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][w,P,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    return result


def lucc_CA_CCAA(actfloor, actceil, HD, *e):
    (P,r),(T,U,v,w) = e
    Dr, Dv, Dw = r-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                         -HD['HSaaaa'][v,P,T,U] * HD['Daa'][Dr,Dw] ,
                         +HD['HSaaaa'][w,P,T,U] * HD['Daa'][Dr,Dv] ))
    if P==T:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][r,U,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][r,U,w,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,U,v,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                             -HD['HSaaaa'][r,U,v,w] ))
    if P==U:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][r,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][r,T,w,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,T,v,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                             +HD['HSaaaa'][r,T,v,w] ))
    if r==w:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             +HD['HSaaaa'][v,P,T,U] ))
    if r==v:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                             -HD['HSaaaa'][w,P,T,U] ))
    return result


def lucc_CA_VAAA(actfloor, actceil, HD, *e):
    (P,r),(T,u,v,w) = e
    Dr, Du, Dv, Dw = r-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += -einsum( 'ij,ij->', HD['HSaaaa'][T,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Du,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                         -HD['Daaaa'][Dr,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,T], HD['Daaaaaa'][Dr,Du,:,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,u,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,P,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,r,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ))
    if r==u:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][T,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    return result


def lucc_CA_VACA(actfloor, actceil, HD, *e):
    (P,r),(T,u,V,w) = e
    Dr, Du, Dw = r-actfloor, u-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,P,V,actfloor:actceil], HD['Daaaa'][Dr,Dw,Du,:] ) ,
                         -HD['HSaaaa'][P,V,u,T] * HD['Daa'][Dr,Dw] ,
                         +HD['HSaaaa'][T,P,V,r] * HD['Daa'][Du,Dw] ,
                         +HD['HSaaaa'][P,V,r,T] * HD['Daa'][Du,Dw] ))
    if P==V:
        result += 2 * fsum(( +HD['HSaa'][r,T] * HD['Daa'][Du,Dw] ,
                             -HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Du,:,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,u,r,actfloor:actceil], HD['Daa'][Dw,:] ) ))
    if r==u:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,P,V,actfloor:actceil], HD['Daa'][Dw,:] )
    return result


def lucc_CA_VACC(actfloor, actceil, HD, *e):
    (P,r),(T,u,V,W) = e
    Dr, Du = r-actfloor, u-actfloor
    result = 0
    result += 2 * +HD['HSaaaa'][T,P,V,W] * HD['Daa'][Dr,Du]
    if P==W:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][r,V,actfloor:actceil,T], HD['Daa'][Du,:] ) ,
                             +HD['HSaaaa'][r,V,u,T] ))
    if P==V:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][r,W,actfloor:actceil,T], HD['Daa'][Du,:] ) ,
                             -HD['HSaaaa'][r,W,u,T] ))
    if r==u:
        result += 2 * -HD['HSaaaa'][T,P,V,W]
    return result


def lucc_CA_VVAA(actfloor, actceil, HD, *e):
    (P,r),(T,U,v,w) = e
    Dr, Dv, Dw = r-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] )
    return result


def lucc_CA_VVCA(actfloor, actceil, HD, *e):
    (P,r),(T,U,V,w) = e
    Dr, Dw = r-actfloor, w-actfloor
    result = 0
    result += 2 * +HD['HSaaaa'][P,V,T,U] * HD['Daa'][Dr,Dw]
    if P==V:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daa'][Dw,:] )
    return result


def lucc_CA_VVCC(actfloor, actceil, HD, *e):
    (P,r),(T,U,V,W) = e
    Dr = r-actfloor
    result = 0
    if P==W:
        result += 2 * -HD['HSaaaa'][r,V,T,U]
    if P==V:
        result += 2 * +HD['HSaaaa'][r,W,T,U]
    return result


def lucc_VA_AAAA(actfloor, actceil, HD, *e):
    (P,r),(t,u,v,w) = e
    Dr, Dt, Du, Dv, Dw = r-actfloor, t-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][P,t,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Du,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][P,u,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dt,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][P,w,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dv,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][P,v,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dw,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][P,t] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                         -HD['HSaa'][P,u] * HD['Daaaa'][Dr,Dt,Dv,Dw] ,
                         +HD['HSaa'][P,w] * HD['Daaaa'][Dr,Dv,Dt,Du] ,
                         -HD['HSaa'][P,v] * HD['Daaaa'][Dr,Dw,Dt,Du] ,
                         -HD['Daaaa'][Dr,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,t] ) ,
                         +HD['Daaaa'][Dr,Dt,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,u] ) ,
                         -HD['Daaaa'][Dr,Dv,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                         +HD['Daaaa'][Dr,Dw,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,t,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dv,Dw,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,u,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,:,Dv,Dw,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,w,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,:,Dt,Du,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daaaaaa'][Dr,Dw,:,Dt,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][t,u,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][v,w,actfloor:actceil,P], HD['Daaaa'][Dt,Du,Dr,:] ) ))
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Du,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,u,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ))
    if r==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dt,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,t,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] ) ))
    if r==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dv,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,v,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,Du,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] ) ))
    if r==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dw,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,w,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,Du,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] ) ))
    return result


def lucc_VA_CAAA(actfloor, actceil, HD, *e):
    (P,r),(T,u,v,w) = e
    Dr, Du, Dv, Dw = r-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += +einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Du,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                         +HD['Daaaa'][Dr,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,w,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,v,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,w,actfloor:actceil], HD['Daaaa'][Dr,Dv,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daaaa'][Dr,Dw,Du,:] ) ,
                         -HD['HSaaaa'][P,T,v,w] * HD['Daa'][Dr,Du] ))
    if r==w:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daa'][Du,Dv] ,
                             -HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,v,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if r==v:
        result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Du,Dw] ,
                             +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,w,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    return result


def lucc_VA_CCAA(actfloor, actceil, HD, *e):
    (P,r),(T,U,v,w) = e
    Dr, Dv, Dw = r-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                         -HD['HSaaaa'][P,w,T,U] * HD['Daa'][Dr,Dv] ,
                         +HD['HSaaaa'][P,v,T,U] * HD['Daa'][Dr,Dw] ))
    if r==w:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daa'][Dv,:] ) ,
                             -HD['HSaaaa'][P,v,T,U] ))
    if r==v:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daa'][Dw,:] ) ,
                             +HD['HSaaaa'][P,w,T,U] ))
    return result


def lucc_VA_VAAA(actfloor, actceil, HD, *e):
    (P,r),(T,u,v,w) = e
    Dr, Du, Dv, Dw = r-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += +einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Du,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                         -HD['Daaaa'][Dr,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dr,Du,:,Dv,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,u,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Dr,:] ) ))
    if P==T:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Du,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][r,u,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ))
    if r==u:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    return result


def lucc_VA_VACA(actfloor, actceil, HD, *e):
    (P,r),(T,u,V,w) = e
    Dr, Du, Dw = r-actfloor, u-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,V,actfloor:actceil], HD['Daaaa'][Dr,Dw,Du,:] ) ,
                         -HD['HSaaaa'][P,V,u,T] * HD['Daa'][Dr,Dw] ))
    if P==T:
        result += 2 * fsum(( +HD['HSaa'][r,V] * HD['Daa'][Du,Dw] ,
                             +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,V,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,V,actfloor:actceil], HD['Daaaa'][Dw,:,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,u,V,actfloor:actceil], HD['Daa'][Dw,:] ) ))
    if r==u:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,T,V,actfloor:actceil], HD['Daa'][Dw,:] )
    return result


def lucc_VA_VACC(actfloor, actceil, HD, *e):
    (P,r),(T,u,V,W) = e
    Dr, Du = r-actfloor, u-actfloor
    result = 0
    result += 2 * -HD['HSaaaa'][P,T,V,W] * HD['Daa'][Dr,Du]
    if P==T:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][V,W,r,actfloor:actceil], HD['Daa'][Du,:] ) ,
                             +HD['HSaaaa'][r,u,V,W] ))
    if r==u:
        result += 2 * +HD['HSaaaa'][P,T,V,W]
    return result


def lucc_VA_VVAA(actfloor, actceil, HD, *e):
    (P,r),(T,U,v,w) = e
    Dr, Dv, Dw = r-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Dr,:] )
    if P==T:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][U,r,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    if P==U:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][T,r,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    return result


def lucc_VA_VVCA(actfloor, actceil, HD, *e):
    (P,r),(T,U,V,w) = e
    Dr, Dw = r-actfloor, w-actfloor
    result = 0
    result += 2 * +HD['HSaaaa'][P,V,T,U] * HD['Daa'][Dr,Dw]
    if P==T:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][U,r,V,actfloor:actceil], HD['Daa'][Dw,:] )
    if P==U:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,r,V,actfloor:actceil], HD['Daa'][Dw,:] )
    return result


def lucc_VA_VVCC(actfloor, actceil, HD, *e):
    (P,r),(T,U,V,W) = e
    Dr = r-actfloor
    result = 0
    if P==T:
        result += 2 * -HD['HSaaaa'][U,r,V,W]
    if P==U:
        result += 2 * +HD['HSaaaa'][T,r,V,W]
    return result


def lucc_VC_AAAA(actfloor, actceil, HD, *e):
    (P,R),(t,u,v,w) = e
    Dt, Du, Dv, Dw = t-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][t,R,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][u,R,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Dt,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,w,R,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,v,R,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][w,R,actfloor:actceil,P], HD['Daaaa'][Dt,Du,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][v,R,actfloor:actceil,P], HD['Daaaa'][Dt,Du,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,t,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,u,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] ) ))
    return result


def lucc_VC_CAAA(actfloor, actceil, HD, *e):
    (P,R),(T,u,v,w) = e
    Du, Dv, Dw = u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][R,T,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                         -HD['HSaaaa'][P,w,R,T] * HD['Daa'][Du,Dv] ,
                         +HD['HSaaaa'][P,v,R,T] * HD['Daa'][Du,Dw] ,
                         -HD['HSaaaa'][P,T,R,w] * HD['Daa'][Du,Dv] ,
                         +HD['HSaaaa'][P,T,R,v] * HD['Daa'][Du,Dw] ))
    if R==T:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Du,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             -HD['HSaa'][P,w] * HD['Daa'][Du,Dv] ,
                             +HD['HSaa'][P,v] * HD['Daa'][Du,Dw] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             +HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                             -HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,w,actfloor:actceil], HD['Daaaa'][Dv,:,Du,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dw,:,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,w,actfloor:actceil,P], HD['Daa'][Du,:] ) ))
    return result


def lucc_VC_CCAA(actfloor, actceil, HD, *e):
    (P,R),(T,U,v,w) = e
    Dv, Dw = v-actfloor, w-actfloor
    result = 0
    if R==T:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,U,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,U,w,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,U,v,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                             +HD['HSaaaa'][P,U,v,w] ))
    if R==U:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,T,w,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                             -HD['HSaaaa'][P,T,v,w] ))
    return result


def lucc_VC_VAAA(actfloor, actceil, HD, *e):
    (P,R),(T,u,v,w) = e
    Du, Dv, Dw = u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ))
    if P==T:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Du,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][u,R,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ))
    return result


def lucc_VC_VACA(actfloor, actceil, HD, *e):
    (P,R),(T,u,V,w) = e
    Du, Dw = u-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -HD['HSaaaa'][P,V,R,T] * HD['Daa'][Du,Dw] ,
                         -HD['HSaaaa'][P,T,R,V] * HD['Daa'][Du,Dw] ))
    if R==V:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daa'][Du,Dw] ,
                             +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Du,:,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,u,actfloor:actceil,P], HD['Daa'][Dw,:] ) ))
    if P==T:
        result += 2 * fsum(( +HD['HSaa'][R,V] * HD['Daa'][Du,Dw] ,
                             +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][R,:actfloor,V,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,V,actfloor:actceil], HD['Daaaa'][Dw,:,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][u,R,V,actfloor:actceil], HD['Daa'][Dw,:] ) ))
    return result


def lucc_VC_VACC(actfloor, actceil, HD, *e):
    (P,R),(T,u,V,W) = e
    Du = u-actfloor
    result = 0
    if R==W:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daa'][Du,:] ) ,
                             -HD['HSaaaa'][P,V,u,T] ))
    if R==V:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,W,actfloor:actceil,T], HD['Daa'][Du,:] ) ,
                             +HD['HSaaaa'][P,W,u,T] ))
    if P==T:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][V,W,R,actfloor:actceil], HD['Daa'][Du,:] ) ,
                             -HD['HSaaaa'][u,R,V,W] ))
    return result


def lucc_VC_VVAA(actfloor, actceil, HD, *e):
    (P,R),(T,U,v,w) = e
    Dv, Dw = v-actfloor, w-actfloor
    result = 0
    if P==T:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][U,R,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    if P==U:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][T,R,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    return result


def lucc_VC_VVCA(actfloor, actceil, HD, *e):
    (P,R),(T,U,V,w) = e
    Dw = w-actfloor
    result = 0
    if R==V:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daa'][Dw,:] )
    if P==T:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][U,R,V,actfloor:actceil], HD['Daa'][Dw,:] )
    if P==U:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,R,V,actfloor:actceil], HD['Daa'][Dw,:] )
    return result


def lucc_VC_VVCC(actfloor, actceil, HD, *e):
    (P,R),(T,U,V,W) = e
    result = 0
    if R==W:
        result += 2 * +HD['HSaaaa'][P,V,T,U]
    if R==V:
        result += 2 * -HD['HSaaaa'][P,W,T,U]
    if P==T:
        result += 2 * -HD['HSaaaa'][U,R,V,W]
    if P==U:
        result += 2 * +HD['HSaaaa'][T,R,V,W]
    return result


def lucc_AAAA_CA(actfloor, actceil, HD, *e):
    (p,q,r,s),(T,v) = e
    Dp, Dq, Dr, Ds, Dv = p-actfloor, q-actfloor, r-actfloor, s-actfloor, v-actfloor
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][q,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dp,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][s,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dv,Dr,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][r,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dv,Ds,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][p,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dq,:,:] ) ))
    result += 2 * fsum(( -HD['HSaa'][p,T] * HD['Daaaa'][Dq,Dv,Dr,Ds] ,
                         +HD['HSaa'][q,T] * HD['Daaaa'][Dp,Dv,Dr,Ds] ,
                         -HD['HSaa'][s,T] * HD['Daaaa'][Dp,Dq,Dr,Dv] ,
                         +HD['HSaa'][r,T] * HD['Daaaa'][Dp,Dq,Ds,Dv] ,
                         -HD['Daaaa'][Dp,Dq,Dr,Dv] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,T,:actfloor] ) ,
                         -HD['Daaaa'][Dq,Dv,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                         +HD['Daaaa'][Dp,Dq,Ds,Dv] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                         +HD['Daaaa'][Dp,Dv,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,:,Dr,Dv,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,:,Ds,Dv,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dq,Dv,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dp,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][p,q,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,s,T,actfloor:actceil], HD['Daaaa'][Dp,Dq,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][p,v,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][p,T,v,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dq,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][q,v,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dp,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][q,T,v,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dp,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][s,v,T,actfloor:actceil], HD['Daaaa'][Dp,Dq,Dr,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][s,T,v,actfloor:actceil], HD['Daaaa'][Dp,Dq,Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,v,T,actfloor:actceil], HD['Daaaa'][Dp,Dq,Ds,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,T,v,actfloor:actceil], HD['Daaaa'][Dp,Dq,Ds,:] ) ))
    if r==v:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][s,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dp,Dq,:,:] )
        result += 2 * fsum(( +HD['HSaa'][p,T] * HD['Daa'][Dq,Ds] ,
                             -HD['HSaa'][q,T] * HD['Daa'][Dp,Ds] ,
                             -HD['Daa'][Dp,Ds] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                             +HD['Daa'][Dq,Ds] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Ds,:,Dp,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Ds,:,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][p,q,T,actfloor:actceil], HD['Daa'][Ds,:] ) ))
    if s==v:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][r,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dp,Dq,:,:] )
        result += 2 * fsum(( -HD['HSaa'][p,T] * HD['Daa'][Dq,Dr] ,
                             +HD['HSaa'][q,T] * HD['Daa'][Dp,Dr] ,
                             +HD['Daa'][Dp,Dr] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                             -HD['Daa'][Dq,Dr] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dp,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][p,q,T,actfloor:actceil], HD['Daa'][Dr,:] ) ))
    if p==v:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][q,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] )
        result += 2 * fsum(( +HD['HSaa'][s,T] * HD['Daa'][Dq,Dr] ,
                             -HD['HSaa'][r,T] * HD['Daa'][Dq,Ds] ,
                             +HD['Daa'][Dq,Dr] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,T,:actfloor] ) ,
                             -HD['Daa'][Dq,Ds] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dq,:,Dr,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dq,:,Ds,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,T,actfloor:actceil], HD['Daa'][Dq,:] ) ))
    if q==v:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][p,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] )
        result += 2 * fsum(( -HD['HSaa'][s,T] * HD['Daa'][Dp,Dr] ,
                             +HD['HSaa'][r,T] * HD['Daa'][Dp,Ds] ,
                             -HD['Daa'][Dp,Dr] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,T,:actfloor] ) ,
                             +HD['Daa'][Dp,Ds] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dp,:,Dr,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dp,:,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,s,T,actfloor:actceil], HD['Daa'][Dp,:] ) ))
    return result


def lucc_AAAA_VA(actfloor, actceil, HD, *e):
    (p,q,r,s),(T,v) = e
    Dp, Dq, Dr, Ds, Dv = p-actfloor, q-actfloor, r-actfloor, s-actfloor, v-actfloor
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['HSaaaa'][T,r,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dv,Ds,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][T,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dp,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][T,s,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dv,Dr,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][T,p,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dq,:,:] ) ))
    result += 2 * fsum(( -HD['HSaa'][p,T] * HD['Daaaa'][Dq,Dv,Dr,Ds] ,
                         +HD['HSaa'][q,T] * HD['Daaaa'][Dp,Dv,Dr,Ds] ,
                         -HD['HSaa'][s,T] * HD['Daaaa'][Dp,Dq,Dr,Dv] ,
                         +HD['HSaa'][r,T] * HD['Daaaa'][Dp,Dq,Ds,Dv] ,
                         -HD['Daaaa'][Dp,Dv,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,T] ) ,
                         +HD['Daaaa'][Dp,Dq,Dr,Dv] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,T] ) ,
                         -HD['Daaaa'][Dp,Dq,Ds,Dv] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                         +HD['Daaaa'][Dq,Dv,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,:actfloor,T] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dp,Dq,:,Dr,Dv,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dp,Dq,:,Ds,Dv,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,:,Dq,Dv,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,:,Dp,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,s,actfloor:actceil,T], HD['Daaaa'][Dp,Dq,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][p,q,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dv,:] ) ))
    return result


def lucc_AAAA_VC(actfloor, actceil, HD, *e):
    (p,q,r,s),(T,V) = e
    Dp, Dq, Dr, Ds = p-actfloor, q-actfloor, r-actfloor, s-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][p,V,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,p,V,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dq,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][q,V,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dp,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,q,V,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dp,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][s,V,actfloor:actceil,T], HD['Daaaa'][Dp,Dq,Dr,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,s,V,actfloor:actceil], HD['Daaaa'][Dp,Dq,Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,V,actfloor:actceil,T], HD['Daaaa'][Dp,Dq,Ds,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,r,V,actfloor:actceil], HD['Daaaa'][Dp,Dq,Ds,:] ) ))
    return result


def lucc_CAAA_AA(actfloor, actceil, HD, *e):
    (P,q,r,s),(t,v) = e
    Dq, Dr, Ds, Dt, Dv = q-actfloor, r-actfloor, s-actfloor, t-actfloor, v-actfloor
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][t,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dq,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][v,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dt,Dq,:,:] ) ))
    result += 2 * fsum(( -HD['HSaa'][P,t] * HD['Daaaa'][Dq,Dv,Dr,Ds] ,
                         +HD['HSaa'][P,v] * HD['Daaaa'][Dq,Dt,Dr,Ds] ,
                         +HD['Daaaa'][Dq,Dv,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,t] ) ,
                         -HD['Daaaa'][Dq,Dt,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,t,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dq,Dv,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,v,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dq,Dt,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][v,P,r,actfloor:actceil], HD['Daaaa'][Ds,Dt,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][t,P,s,actfloor:actceil], HD['Daaaa'][Dr,Dv,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][s,v,P,actfloor:actceil], HD['Daaaa'][Dq,Dt,Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][v,P,s,actfloor:actceil], HD['Daaaa'][Dr,Dt,Dq,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][s,t,P,actfloor:actceil], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][t,P,r,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,v,P,actfloor:actceil], HD['Daaaa'][Dq,Dt,Ds,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,t,P,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                         -HD['HSaaaa'][t,P,r,s] * HD['Daa'][Dq,Dv] ,
                         +HD['HSaaaa'][v,P,r,s] * HD['Daa'][Dq,Dt] ))
    if q==t:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             -HD['HSaa'][P,s] * HD['Daa'][Dr,Dv] ,
                             +HD['HSaa'][P,r] * HD['Daa'][Ds,Dv] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             -HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             +HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,s,P,actfloor:actceil], HD['Daa'][Dv,:] ) ))
    if q==v:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dt,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dt,:] ) ,
                             +HD['HSaa'][P,s] * HD['Daa'][Dr,Dt] ,
                             -HD['HSaa'][P,r] * HD['Daa'][Ds,Dt] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dt,:] ) ,
                             -HD['Daa'][Dr,Dt] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             +HD['Daa'][Ds,Dt] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaa'][Dr,:,Dt,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaa'][Ds,:,Dt,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,P,actfloor:actceil], HD['Daa'][Dt,:] ) ))
    if r==v:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,:,Dq,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Ds,Dt,Dq,:] ) ,
                             +HD['HSaa'][P,s] * HD['Daa'][Dq,Dt] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Dt,Dq,:] ) ,
                             -HD['Daa'][Dq,Dt] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaa'][Dt,:,Dq,:] ) ))
    if s==v:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,:,Dq,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Dt,Dq,:] ) ,
                             -HD['HSaa'][P,r] * HD['Daa'][Dq,Dt] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Dt,Dq,:] ) ,
                             +HD['Daa'][Dq,Dt] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaa'][Dt,:,Dq,:] ) ))
    if r==t:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,:,Dq,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                             -HD['HSaa'][P,s] * HD['Daa'][Dq,Dv] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                             +HD['Daa'][Dq,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaa'][Dv,:,Dq,:] ) ))
    if s==t:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,:,Dq,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Dv,Dq,:] ) ,
                             +HD['HSaa'][P,r] * HD['Daa'][Dq,Dv] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Dv,Dq,:] ) ,
                             -HD['Daa'][Dq,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaa'][Dv,:,Dq,:] ) ))
    return result


def lucc_CAAA_CA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,v) = e
    Dq, Dr, Ds, Dv = q-actfloor, r-actfloor, s-actfloor, v-actfloor
    result = 0
    result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dq,:,:] )
    result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dv,Dr,Ds] ,
                         -HD['Daaaa'][Dq,Dv,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dq,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,s,actfloor:actceil], HD['Daaaa'][Dr,Dv,Dq,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][s,T,P,actfloor:actceil], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,r,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][v,P,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,T,P,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                         +HD['HSaaaa'][P,T,r,s] * HD['Daa'][Dq,Dv] ,
                         +HD['HSaaaa'][P,T,s,v] * HD['Daa'][Dq,Dr] ,
                         +HD['HSaaaa'][v,P,T,s] * HD['Daa'][Dq,Dr] ,
                         -HD['HSaaaa'][P,T,r,v] * HD['Daa'][Dq,Ds] ,
                         -HD['HSaaaa'][v,P,T,r] * HD['Daa'][Dq,Ds] ))
    if P==T:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,:,Dr,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,:,Ds,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][r,s,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dv,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][q,v,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                             -HD['HSaa'][r,v] * HD['Daa'][Dq,Ds] ,
                             +HD['HSaa'][s,v] * HD['Daa'][Dq,Dr] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                             +HD['Daa'][Dq,Ds] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,v] ) ,
                             -HD['Daa'][Dq,Dr] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,v] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dq,:,Ds,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dq,:,Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,v,actfloor:actceil], HD['Daa'][Dq,:] ) ))
        if r==v:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,:,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,:,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Ds,:] ) ))
        if s==v:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,:,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,:,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dr,:] ) ))
    if r==v:
        result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Dq,Ds] ,
                             +HD['Daa'][Dq,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaa'][Ds,:,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,T,P,actfloor:actceil], HD['Daa'][Dq,:] ) ))
    if s==v:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daa'][Dq,Dr] ,
                             -HD['Daa'][Dq,Dr] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,T,P,actfloor:actceil], HD['Daa'][Dq,:] ) ))
    if q==v:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,T,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,T,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                             -HD['HSaaaa'][P,T,r,s] ))
    return result


def lucc_CAAA_VA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,v) = e
    Dq, Dr, Ds, Dv = q-actfloor, r-actfloor, s-actfloor, v-actfloor
    result = 0
    result += +einsum( 'ij,ij->', HD['HSaaaa'][T,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dq,:,:] )
    result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dv,Dr,Ds] ,
                         +HD['Daaaa'][Dq,Dv,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,:,Dq,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,P,s,actfloor:actceil], HD['Daaaa'][Dr,Dv,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,s,P,actfloor:actceil], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,P,r,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,r,P,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                         -HD['HSaaaa'][T,P,r,s] * HD['Daa'][Dq,Dv] ))
    return result


def lucc_CAAA_VC(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,V) = e
    Dq, Dr, Ds = q-actfloor, r-actfloor, s-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,P,V,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dq,:] ) ,
                         -HD['HSaaaa'][T,P,V,s] * HD['Daa'][Dq,Dr] ,
                         -HD['HSaaaa'][P,V,s,T] * HD['Daa'][Dq,Dr] ,
                         +HD['HSaaaa'][T,P,V,r] * HD['Daa'][Dq,Ds] ,
                         +HD['HSaaaa'][P,V,r,T] * HD['Daa'][Dq,Ds] ))
    if P==V:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][T,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] )
        result += 2 * fsum(( -HD['HSaa'][s,T] * HD['Daa'][Dq,Dr] ,
                             +HD['HSaa'][r,T] * HD['Daa'][Dq,Ds] ,
                             +HD['Daa'][Dq,Dr] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,T] ) ,
                             -HD['Daa'][Dq,Ds] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Dq,:,Dr,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Dq,:,Ds,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,actfloor:actceil,T], HD['Daa'][Dq,:] ) ))
    return result


def lucc_CCAA_AA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(t,v) = e
    Dr, Ds, Dt, Dv = r-actfloor, s-actfloor, t-actfloor, v-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,t,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,Q,v,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dt,:] ) ,
                         -HD['HSaaaa'][P,Q,s,v] * HD['Daa'][Dr,Dt] ,
                         +HD['HSaaaa'][P,Q,s,t] * HD['Daa'][Dr,Dv] ,
                         +HD['HSaaaa'][P,Q,r,v] * HD['Daa'][Ds,Dt] ,
                         -HD['HSaaaa'][P,Q,r,t] * HD['Daa'][Ds,Dv] ))
    if r==v:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dt,:,:] )
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,s,actfloor:actceil], HD['Daa'][Dt,:] )
    if s==v:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dt,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,r,actfloor:actceil], HD['Daa'][Dt,:] )
    if r==t:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dv,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,s,actfloor:actceil], HD['Daa'][Dv,:] )
    if s==t:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dv,:,:] )
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,r,actfloor:actceil], HD['Daa'][Dv,:] )
    return result


def lucc_CCAA_CA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,v) = e
    Dr, Ds, Dv = r-actfloor, s-actfloor, v-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                         +HD['HSaaaa'][P,Q,T,r] * HD['Daa'][Ds,Dv] ,
                         -HD['HSaaaa'][P,Q,T,s] * HD['Daa'][Dr,Dv] ))
    if Q==T:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][v,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             -HD['HSaa'][P,s] * HD['Daa'][Dr,Dv] ,
                             +HD['HSaa'][P,r] * HD['Daa'][Ds,Dv] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             -HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             +HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,s,P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,P,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,P,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                             +HD['HSaaaa'][v,P,r,s] ))
        if r==v:
            result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,:,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +HD['HSaa'][P,s] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daa'][:,:] ) ))
        if s==v:
            result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,:,:,:] )
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -HD['HSaa'][P,r] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daa'][:,:] ) ))
    if P==T:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][v,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             +HD['HSaa'][Q,s] * HD['Daa'][Dr,Dv] ,
                             -HD['HSaa'][Q,r] * HD['Daa'][Ds,Dv] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             +HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,r] ) ,
                             -HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,s] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,r,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,s,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,Q,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,Q,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,Q,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                             -HD['HSaaaa'][v,Q,r,s] ))
        if r==v:
            result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,:,:,:] )
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -HD['HSaa'][Q,s] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,s] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,s,actfloor:actceil], HD['Daa'][:,:] ) ))
        if s==v:
            result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,:,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +HD['HSaa'][Q,r] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,r] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,r,actfloor:actceil], HD['Daa'][:,:] ) ))
    if r==v:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                             +HD['HSaaaa'][P,Q,T,s] ))
    if s==v:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                             -HD['HSaaaa'][P,Q,T,r] ))
    return result


def lucc_CCAA_VA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,v) = e
    Dr, Ds, Dv = r-actfloor, s-actfloor, v-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                         +HD['HSaaaa'][P,Q,s,T] * HD['Daa'][Dr,Dv] ,
                         -HD['HSaaaa'][P,Q,r,T] * HD['Daa'][Ds,Dv] ))
    return result


def lucc_CCAA_VC(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,V) = e
    Dr, Ds = r-actfloor, s-actfloor
    result = 0
    if Q==V:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][T,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,P,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,P,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                             -HD['HSaaaa'][T,P,r,s] ))
    if P==V:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][T,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,Q,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][T,Q,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                             +HD['HSaaaa'][T,Q,r,s] ))
    return result


def lucc_VAAA_AA(actfloor, actceil, HD, *e):
    (P,q,r,s),(t,v) = e
    Dq, Dr, Ds, Dt, Dv = q-actfloor, r-actfloor, s-actfloor, t-actfloor, v-actfloor
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][P,v,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dt,Dq,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][P,t,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dq,:,:] ) ))
    result += 2 * fsum(( -HD['HSaa'][P,t] * HD['Daaaa'][Dq,Dv,Dr,Ds] ,
                         +HD['HSaa'][P,v] * HD['Daaaa'][Dq,Dt,Dr,Ds] ,
                         +HD['Daaaa'][Dq,Dv,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,t] ) ,
                         -HD['Daaaa'][Dq,Dt,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,t,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dq,Dv,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dq,Dt,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,q,t,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,q,v,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dt,:] ) ))
    if r==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,:,Dq,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dt,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Ds,Dt,Dq,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Dt,Dq,:] ) ))
    if s==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,:,Dq,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dt,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Dt,Dq,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Dt,Dq,:] ) ))
    if r==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,:,Dq,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dv,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ))
    if s==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,:,Dq,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dv,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Dv,Dq,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Dv,Dq,:] ) ))
    if q==t:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ))
    if q==v:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dt,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dt,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dt,:] ) ))
    return result


def lucc_VAAA_CA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,v) = e
    Dq, Dr, Ds, Dv = q-actfloor, r-actfloor, s-actfloor, v-actfloor
    result = 0
    result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dq,:,:] )
    result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dv,Dr,Ds] ,
                         -HD['Daaaa'][Dq,Dv,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dq,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,q,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,v,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dq,:] ) ))
    if r==v:
        result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Dq,Ds] ,
                             +HD['Daa'][Dq,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Ds,:,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,q,T,actfloor:actceil], HD['Daa'][Ds,:] ) ))
    if s==v:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daa'][Dq,Dr] ,
                             -HD['Daa'][Dq,Dr] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,q,T,actfloor:actceil], HD['Daa'][Dr,:] ) ))
    if q==v:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] )
    return result


def lucc_VAAA_VA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,v) = e
    Dq, Dr, Ds, Dv = q-actfloor, r-actfloor, s-actfloor, v-actfloor
    result = 0
    result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dq,:,:] )
    result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dv,Dr,Ds] ,
                         +HD['Daaaa'][Dq,Dv,Dr,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,:,Dq,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,q,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dv,:] ) ))
    if P==T:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,:,Dr,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,:,Ds,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][r,s,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dv,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ))
    return result


def lucc_VAAA_VC(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,V) = e
    Dq, Dr, Ds = q-actfloor, r-actfloor, s-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dq,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,V,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dq,:] ) ))
    if P==T:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][q,V,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] )
        result += 2 * fsum(( -HD['HSaa'][s,V] * HD['Daa'][Dq,Dr] ,
                             +HD['HSaa'][r,V] * HD['Daa'][Dq,Ds] ,
                             -HD['Daa'][Dq,Dr] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,V,:actfloor] ) ,
                             +HD['Daa'][Dq,Ds] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,V,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,V,actfloor:actceil], HD['Daaaa'][Dq,:,Dr,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,V,actfloor:actceil], HD['Daaaa'][Dq,:,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,s,V,actfloor:actceil], HD['Daa'][Dq,:] ) ))
    return result


def lucc_VACA_AA(actfloor, actceil, HD, *e):
    (P,q,R,s),(t,v) = e
    Dq, Ds, Dt, Dv = q-actfloor, s-actfloor, t-actfloor, v-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][t,R,actfloor:actceil,P], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,v,R,actfloor:actceil], HD['Daaaa'][Ds,Dt,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][v,R,actfloor:actceil,P], HD['Daaaa'][Dq,Dt,Ds,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,t,R,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                         -HD['HSaaaa'][P,q,R,t] * HD['Daa'][Ds,Dv] ,
                         +HD['HSaaaa'][P,q,R,v] * HD['Daa'][Ds,Dt] ))
    if s==v:
        result += 2 * fsum(( +HD['HSaa'][P,R] * HD['Daa'][Dq,Dt] ,
                             +HD['Daa'][Dq,Dt] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Dt,:,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,q,R,actfloor:actceil], HD['Daa'][Dt,:] ) ))
    if s==t:
        result += 2 * fsum(( -HD['HSaa'][P,R] * HD['Daa'][Dq,Dv] ,
                             -HD['Daa'][Dq,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Dv,:,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,q,R,actfloor:actceil], HD['Daa'][Dv,:] ) ))
    if q==t:
        result += 2 * fsum(( -HD['HSaa'][P,R] * HD['Daa'][Ds,Dv] ,
                             -HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ))
    if q==v:
        result += 2 * fsum(( +HD['HSaa'][P,R] * HD['Daa'][Ds,Dt] ,
                             +HD['Daa'][Ds,Dt] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Ds,:,Dt,:] ) ))
    return result


def lucc_VACA_CA(actfloor, actceil, HD, *e):
    (P,q,R,s),(T,v) = e
    Dq, Ds, Dv = q-actfloor, s-actfloor, v-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][R,T,actfloor:actceil,P], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                         -HD['HSaaaa'][P,q,R,T] * HD['Daa'][Ds,Dv] ,
                         +HD['HSaaaa'][P,v,R,T] * HD['Daa'][Dq,Ds] ,
                         +HD['HSaaaa'][P,T,R,v] * HD['Daa'][Dq,Ds] ))
    if R==T:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,:,Dq,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dv,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                             +HD['HSaa'][P,v] * HD['Daa'][Dq,Ds] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                             -HD['Daa'][Dq,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Ds,:,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,q,v,actfloor:actceil], HD['Daa'][Ds,:] ) ))
        if q==v:
            result += +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,:,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Ds,:] ) ))
    if q==v:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daa'][Ds,:] )
    if s==v:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][R,T,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                             +HD['HSaaaa'][P,q,R,T] ))
    return result


def lucc_VACA_VA(actfloor, actceil, HD, *e):
    (P,q,R,s),(T,v) = e
    Dq, Ds, Dv = q-actfloor, s-actfloor, v-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,P], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                         -HD['HSaaaa'][P,q,R,T] * HD['Daa'][Ds,Dv] ))
    if P==T:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,:,Ds,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                             +HD['HSaa'][q,R] * HD['Daa'][Ds,Dv] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                             +HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,R,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ))
    return result


def lucc_VACA_VC(actfloor, actceil, HD, *e):
    (P,q,R,s),(T,V) = e
    Dq, Ds = q-actfloor, s-actfloor
    result = 0
    result += 2 * fsum(( -HD['HSaaaa'][P,T,R,V] * HD['Daa'][Dq,Ds] ,
                         -HD['HSaaaa'][P,V,R,T] * HD['Daa'][Dq,Ds] ))
    if R==V:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daa'][Dq,Ds] ,
                             +HD['Daa'][Dq,Ds] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Ds,:,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,q,actfloor:actceil,T], HD['Daa'][Ds,:] ) ))
        if P==T:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,:,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,:,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dq,:] ) ))
    if P==T:
        result += 2 * fsum(( +HD['HSaa'][R,V] * HD['Daa'][Dq,Ds] ,
                             +HD['Daa'][Dq,Ds] * einsum( 'II->', HD['HSaaaa'][R,:actfloor,V,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,V,actfloor:actceil], HD['Daaaa'][Dq,:,Ds,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][q,V,R,actfloor:actceil], HD['Daa'][Ds,:] ) ))
    return result


def lucc_VACC_AA(actfloor, actceil, HD, *e):
    (P,q,R,S),(t,v) = e
    Dq, Dt, Dv = q-actfloor, t-actfloor, v-actfloor
    result = 0
    result += 2 * fsum(( -HD['HSaaaa'][P,v,R,S] * HD['Daa'][Dq,Dt] ,
                         +HD['HSaaaa'][P,t,R,S] * HD['Daa'][Dq,Dv] ))
    if q==t:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][R,S,actfloor:actceil,P], HD['Daa'][Dv,:] )
    if q==v:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][R,S,actfloor:actceil,P], HD['Daa'][Dt,:] )
    return result


def lucc_VACC_CA(actfloor, actceil, HD, *e):
    (P,q,R,S),(T,v) = e
    Dq, Dv = q-actfloor, v-actfloor
    result = 0
    result += 2 * +HD['HSaaaa'][P,T,R,S] * HD['Daa'][Dq,Dv]
    if S==T:
        result += 2 * fsum(( -HD['HSaa'][P,R] * HD['Daa'][Dq,Dv] ,
                             -HD['Daa'][Dq,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Dv,:,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,q,R,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,R,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                             +HD['HSaaaa'][P,q,R,v] ))
        if q==v:
            result += 2 * fsum(( +HD['HSaa'][P,R] ,
                                 +einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daa'][:,:] ) ))
    if R==T:
        result += 2 * fsum(( +HD['HSaa'][P,S] * HD['Daa'][Dq,Dv] ,
                             +HD['Daa'][Dq,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,S,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,S,actfloor:actceil], HD['Daaaa'][Dv,:,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,q,S,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,S,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                             -HD['HSaaaa'][P,q,S,v] ))
        if q==v:
            result += 2 * fsum(( -HD['HSaa'][P,S] ,
                                 -einsum( 'II->', HD['HSaaaa'][P,:actfloor,S,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,S,actfloor:actceil], HD['Daa'][:,:] ) ))
    if q==v:
        result += 2 * -HD['HSaaaa'][P,T,R,S]
    return result


def lucc_VACC_VA(actfloor, actceil, HD, *e):
    (P,q,R,S),(T,v) = e
    Dq, Dv = q-actfloor, v-actfloor
    result = 0
    result += 2 * +HD['HSaaaa'][P,T,R,S] * HD['Daa'][Dq,Dv]
    if P==T:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][R,S,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dv,:,:] )
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][R,S,q,actfloor:actceil], HD['Daa'][Dv,:] )
    return result


def lucc_VACC_VC(actfloor, actceil, HD, *e):
    (P,q,R,S),(T,V) = e
    Dq = q-actfloor
    result = 0
    if P==T:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][R,S,V,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                             -HD['HSaaaa'][q,V,R,S] ))
        if S==V:
            result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,:,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +HD['HSaa'][q,R] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +einsum( 'II->', HD['HSaaaa'][q,:actfloor,R,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,R,actfloor:actceil], HD['Daa'][:,:] ) ))
        if R==V:
            result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,S,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,:,:,:] )
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][S,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -HD['HSaa'][q,S] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][S,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -einsum( 'II->', HD['HSaaaa'][q,:actfloor,S,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,S,actfloor:actceil], HD['Daa'][:,:] ) ))
    if S==V:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                             -HD['HSaaaa'][P,q,R,T] ))
    if R==V:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,S,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                             +HD['HSaaaa'][P,q,S,T] ))
    return result


def lucc_VVAA_AA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(t,v) = e
    Dr, Ds, Dt, Dv = r-actfloor, s-actfloor, t-actfloor, v-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,t,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,Q,v,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dt,:] ) ))
    if r==v:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dt,:,:] )
    if s==v:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dt,:,:] )
    if r==t:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dv,:,:] )
    if s==t:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dv,:,:] )
    return result


def lucc_VVAA_CA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,v) = e
    Dr, Ds, Dv = r-actfloor, s-actfloor, v-actfloor
    result = 0
    result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] )
    if r==v:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daa'][Ds,:] )
    if s==v:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daa'][Dr,:] )
    return result


def lucc_VVAA_VA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,v) = e
    Dr, Ds, Dv = r-actfloor, s-actfloor, v-actfloor
    result = 0
    result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dv,:] )
    if Q==T:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ))
    if P==T:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ))
    return result


def lucc_VVAA_VC(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,V) = e
    Dr, Ds = r-actfloor, s-actfloor
    result = 0
    if Q==T:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,V,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] )
    if P==T:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][Q,V,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Ds,:,:] )
    return result


def lucc_VVCA_AA(actfloor, actceil, HD, *e):
    (P,Q,R,s),(t,v) = e
    Ds, Dt, Dv = s-actfloor, t-actfloor, v-actfloor
    result = 0
    result += 2 * fsum(( -HD['HSaaaa'][P,Q,R,t] * HD['Daa'][Ds,Dv] ,
                         +HD['HSaaaa'][P,Q,R,v] * HD['Daa'][Ds,Dt] ))
    if s==v:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,R,actfloor:actceil], HD['Daa'][Dt,:] )
    if s==t:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,R,actfloor:actceil], HD['Daa'][Dv,:] )
    return result


def lucc_VVCA_CA(actfloor, actceil, HD, *e):
    (P,Q,R,s),(T,v) = e
    Ds, Dv = s-actfloor, v-actfloor
    result = 0
    result += 2 * -HD['HSaaaa'][P,Q,R,T] * HD['Daa'][Ds,Dv]
    if R==T:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dv,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,v,actfloor:actceil], HD['Daa'][Ds,:] )
    if s==v:
        result += 2 * +HD['HSaaaa'][P,Q,R,T]
    return result


def lucc_VVCA_VA(actfloor, actceil, HD, *e):
    (P,Q,R,s),(T,v) = e
    Ds, Dv = s-actfloor, v-actfloor
    result = 0
    result += 2 * -HD['HSaaaa'][P,Q,R,T] * HD['Daa'][Ds,Dv]
    if Q==T:
        result += 2 * fsum(( -HD['HSaa'][P,R] * HD['Daa'][Ds,Dv] ,
                             -HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ))
    if P==T:
        result += 2 * fsum(( +HD['HSaa'][Q,R] * HD['Daa'][Ds,Dv] ,
                             +HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,R,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ))
    return result


def lucc_VVCA_VC(actfloor, actceil, HD, *e):
    (P,Q,R,s),(T,V) = e
    Ds = s-actfloor
    result = 0
    if R==V:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,actfloor:actceil,T], HD['Daa'][Ds,:] )
        if Q==T:
            result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,:,:,:] )
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Ds,:] ) ))
        if P==T:
            result += +einsum( 'ijk,ijk->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,:,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Ds,:] ) ))
    if Q==T:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,V,R,actfloor:actceil], HD['Daa'][Ds,:] )
    if P==T:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][Q,V,R,actfloor:actceil], HD['Daa'][Ds,:] )
    return result


def lucc_VVCC_AA(actfloor, actceil, HD, *e):
    (P,Q,R,S),(t,v) = e
    Dt, Dv = t-actfloor, v-actfloor
    result = 0
    return result


def lucc_VVCC_CA(actfloor, actceil, HD, *e):
    (P,Q,R,S),(T,v) = e
    Dv = v-actfloor
    result = 0
    if S==T:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,R,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             +HD['HSaaaa'][P,Q,R,v] ))
    if R==T:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,S,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                             -HD['HSaaaa'][P,Q,S,v] ))
    return result


def lucc_VVCC_VA(actfloor, actceil, HD, *e):
    (P,Q,R,S),(T,v) = e
    Dv = v-actfloor
    result = 0
    if Q==T:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][R,S,actfloor:actceil,P], HD['Daa'][Dv,:] )
    if P==T:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][R,S,actfloor:actceil,Q], HD['Daa'][Dv,:] )
    return result


def lucc_VVCC_VC(actfloor, actceil, HD, *e):
    (P,Q,R,S),(T,V) = e
    result = 0
    if Q==T:
        result += 2 * +HD['HSaaaa'][P,V,R,S]
        if S==V:
            result += 2 * fsum(( -HD['HSaa'][P,R] ,
                                 -einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daa'][:,:] ) ))
        if R==V:
            result += 2 * fsum(( +HD['HSaa'][P,S] ,
                                 +einsum( 'II->', HD['HSaaaa'][P,:actfloor,S,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,S,actfloor:actceil], HD['Daa'][:,:] ) ))
    if S==V:
        result += 2 * -HD['HSaaaa'][P,Q,R,T]
        if P==T:
            result += 2 * fsum(( +HD['HSaa'][Q,R] ,
                                 +einsum( 'II->', HD['HSaaaa'][Q,:actfloor,R,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,R,actfloor:actceil], HD['Daa'][:,:] ) ))
    if R==V:
        result += 2 * +HD['HSaaaa'][P,Q,S,T]
        if P==T:
            result += 2 * fsum(( -HD['HSaa'][Q,S] ,
                                 -einsum( 'II->', HD['HSaaaa'][Q,:actfloor,S,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,S,actfloor:actceil], HD['Daa'][:,:] ) ))
    if P==T:
        result += 2 * -HD['HSaaaa'][Q,V,R,S]
    return result


def lucc_AAAA_CAAA(actfloor, actceil, HD, *e):
    (p,q,r,s),(T,u,v,w) = e
    Dp, Dq, Dr, Ds, Du, Dv, Dw = p-actfloor, q-actfloor, r-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][q,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dp,Du,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][s,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dp,Dq,Dv,Dw,Dr,Du,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][r,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dp,Dq,Dv,Dw,Ds,Du,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][p,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dq,Du,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][p,T] * HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] ,
                         -HD['HSaa'][q,T] * HD['Daaaaaa'][Dp,Dv,Dw,Dr,Ds,Du] ,
                         +HD['HSaa'][s,T] * HD['Daaaaaa'][Dp,Dq,Du,Dr,Dv,Dw] ,
                         -HD['HSaa'][r,T] * HD['Daaaaaa'][Dp,Dq,Du,Ds,Dv,Dw] ,
                         +HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                         -HD['Daaaaaa'][Dp,Dq,Du,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                         +HD['Daaaaaa'][Dp,Dq,Du,Dr,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,T,:actfloor] ) ,
                         -HD['Daaaaaa'][Dp,Dv,Dw,Dr,Ds,Du] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaaaa'][Dp,Dq,Du,:,Dr,Dv,Dw,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaaaa'][Dp,Dq,Du,:,Ds,Dv,Dw,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dq,Dv,Dw,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dp,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][s,T,w,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dv,Dr,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][s,T,v,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dw,Dr,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][p,q,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,s,T,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Du,Dv,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,w,T,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Du,Ds,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,v,T,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Du,Ds,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,T,w,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dv,Ds,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,T,v,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dw,Ds,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][p,w,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dq,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][p,v,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dq,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][p,T,w,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dq,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][p,T,v,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dq,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][q,w,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dp,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][q,v,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dp,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][q,T,w,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dp,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][q,T,v,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dp,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][s,w,T,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Du,Dr,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][s,v,T,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Du,Dr,Dw,:] ) ,
                         -HD['HSaaaa'][q,T,v,w] * HD['Daaaa'][Dp,Du,Dr,Ds] ,
                         +HD['HSaaaa'][p,T,v,w] * HD['Daaaa'][Dq,Du,Dr,Ds] ,
                         -HD['HSaaaa'][r,T,v,w] * HD['Daaaa'][Dp,Dq,Ds,Du] ,
                         +HD['HSaaaa'][s,T,v,w] * HD['Daaaa'][Dp,Dq,Dr,Du] ))
    if s==v:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][r,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dw,Du,:,:] )
        result += 2 * fsum(( -HD['HSaa'][p,T] * HD['Daaaa'][Dq,Dw,Dr,Du] ,
                             +HD['HSaa'][q,T] * HD['Daaaa'][Dp,Dw,Dr,Du] ,
                             -HD['Daaaa'][Dq,Dw,Dr,Du] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                             +HD['Daaaa'][Dp,Dw,Dr,Du] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dp,Dw,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dq,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][p,q,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][q,w,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dp,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,T,w,actfloor:actceil], HD['Daaaa'][Dp,Dq,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][p,w,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dq,:] ) ))
        if r==w:
            result += 2 * fsum(( +HD['HSaa'][p,T] * HD['Daa'][Dq,Du] ,
                                 -HD['HSaa'][q,T] * HD['Daa'][Dp,Du] ,
                                 -HD['Daa'][Dp,Du] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                                 +HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dp,:] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dq,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][p,q,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if r==w:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][s,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dv,Du,:,:] )
        result += 2 * fsum(( +HD['HSaa'][q,T] * HD['Daaaa'][Dp,Dv,Ds,Du] ,
                             -HD['HSaa'][p,T] * HD['Daaaa'][Dq,Dv,Ds,Du] ,
                             +HD['Daaaa'][Dp,Dv,Ds,Du] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                             -HD['Daaaa'][Dq,Dv,Ds,Du] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dp,Dv,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dq,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][p,q,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][q,v,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dp,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][p,v,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,T,v,actfloor:actceil], HD['Daaaa'][Dp,Dq,Du,:] ) ))
    if s==w:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][r,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dv,Du,:,:] )
        result += 2 * fsum(( +HD['HSaa'][p,T] * HD['Daaaa'][Dq,Dv,Dr,Du] ,
                             -HD['HSaa'][q,T] * HD['Daaaa'][Dp,Dv,Dr,Du] ,
                             +HD['Daaaa'][Dq,Dv,Dr,Du] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                             -HD['Daaaa'][Dp,Dv,Dr,Du] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dq,Dv,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dp,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][p,q,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][q,v,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dp,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,T,v,actfloor:actceil], HD['Daaaa'][Dp,Dq,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][p,v,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dq,:] ) ))
        if r==v:
            result += 2 * fsum(( -HD['HSaa'][p,T] * HD['Daa'][Dq,Du] ,
                                 +HD['HSaa'][q,T] * HD['Daa'][Dp,Du] ,
                                 +HD['Daa'][Dp,Du] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                                 -HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dp,:] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dq,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][p,q,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if r==v:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][s,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dw,Du,:,:] )
        result += 2 * fsum(( -HD['HSaa'][q,T] * HD['Daaaa'][Dp,Dw,Ds,Du] ,
                             +HD['HSaa'][p,T] * HD['Daaaa'][Dq,Dw,Ds,Du] ,
                             -HD['Daaaa'][Dp,Dw,Ds,Du] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                             +HD['Daaaa'][Dq,Dw,Ds,Du] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dp,Dw,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dq,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][p,q,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][q,w,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dp,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][p,w,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,T,w,actfloor:actceil], HD['Daaaa'][Dp,Dq,Du,:] ) ))
    if q==v:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][p,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Du,:,:] )
        result += 2 * fsum(( -HD['HSaa'][s,T] * HD['Daaaa'][Dp,Du,Dr,Dw] ,
                             +HD['HSaa'][r,T] * HD['Daaaa'][Dp,Du,Ds,Dw] ,
                             -HD['Daaaa'][Dp,Du,Dr,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,T,:actfloor] ) ,
                             +HD['Daaaa'][Dp,Du,Ds,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dp,Du,:,Ds,Dw,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dp,Du,:,Dr,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][p,T,w,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,T,actfloor:actceil], HD['Daaaa'][Dp,Du,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,w,T,actfloor:actceil], HD['Daaaa'][Dp,Du,Ds,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,w,T,actfloor:actceil], HD['Daaaa'][Dp,Du,Dr,:] ) ))
        if p==w:
            result += 2 * fsum(( -HD['HSaa'][r,T] * HD['Daa'][Ds,Du] ,
                                 +HD['HSaa'][s,T] * HD['Daa'][Dr,Du] ,
                                 -HD['Daa'][Ds,Du] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                                 +HD['Daa'][Dr,Du] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,T,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Ds,:] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dr,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,s,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if p==w:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][q,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Du,:,:] )
        result += 2 * fsum(( +HD['HSaa'][r,T] * HD['Daaaa'][Dq,Du,Ds,Dv] ,
                             -HD['HSaa'][s,T] * HD['Daaaa'][Dq,Du,Dr,Dv] ,
                             +HD['Daaaa'][Dq,Du,Ds,Dv] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                             -HD['Daaaa'][Dq,Du,Dr,Dv] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dq,Du,:,Ds,Dv,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dq,Du,:,Dr,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,T,actfloor:actceil], HD['Daaaa'][Dq,Du,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,v,T,actfloor:actceil], HD['Daaaa'][Dq,Du,Ds,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,v,T,actfloor:actceil], HD['Daaaa'][Dq,Du,Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][q,T,v,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ))
    if q==w:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][p,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Du,:,:] )
        result += 2 * fsum(( +HD['HSaa'][s,T] * HD['Daaaa'][Dp,Du,Dr,Dv] ,
                             -HD['HSaa'][r,T] * HD['Daaaa'][Dp,Du,Ds,Dv] ,
                             +HD['Daaaa'][Dp,Du,Dr,Dv] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,T,:actfloor] ) ,
                             -HD['Daaaa'][Dp,Du,Ds,Dv] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dp,Du,:,Ds,Dv,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dp,Du,:,Dr,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][p,T,v,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,s,T,actfloor:actceil], HD['Daaaa'][Dp,Du,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,v,T,actfloor:actceil], HD['Daaaa'][Dp,Du,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,v,T,actfloor:actceil], HD['Daaaa'][Dp,Du,Dr,:] ) ))
        if p==v:
            result += 2 * fsum(( +HD['HSaa'][r,T] * HD['Daa'][Ds,Du] ,
                                 -HD['HSaa'][s,T] * HD['Daa'][Dr,Du] ,
                                 +HD['Daa'][Ds,Du] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                                 -HD['Daa'][Dr,Du] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,T,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Ds,:] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][r,s,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if p==v:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][q,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Du,:,:] )
        result += 2 * fsum(( -HD['HSaa'][r,T] * HD['Daaaa'][Dq,Du,Ds,Dw] ,
                             +HD['HSaa'][s,T] * HD['Daaaa'][Dq,Du,Dr,Dw] ,
                             -HD['Daaaa'][Dq,Du,Ds,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,T,:actfloor] ) ,
                             +HD['Daaaa'][Dq,Du,Dr,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dq,Du,:,Ds,Dw,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dq,Du,:,Dr,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][q,T,w,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,s,T,actfloor:actceil], HD['Daaaa'][Dq,Du,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,w,T,actfloor:actceil], HD['Daaaa'][Dq,Du,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,w,T,actfloor:actceil], HD['Daaaa'][Dq,Du,Dr,:] ) ))
    return result


def lucc_AAAA_CCAA(actfloor, actceil, HD, *e):
    (p,q,r,s),(T,U,v,w) = e
    Dp, Dq, Dr, Ds, Dv, Dw = p-actfloor, q-actfloor, r-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,U,q,actfloor:actceil], HD['Daaaaaa'][Dp,Dv,Dw,Dr,Ds,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,U,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dp,Dq,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dp,Dq,:] ) ,
                         +HD['HSaaaa'][p,q,T,U] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                         -HD['HSaaaa'][r,s,T,U] * HD['Daaaa'][Dp,Dq,Dv,Dw] ,
                         -HD['HSaaaa'][q,w,T,U] * HD['Daaaa'][Dp,Dv,Dr,Ds] ,
                         +HD['HSaaaa'][q,v,T,U] * HD['Daaaa'][Dp,Dw,Dr,Ds] ,
                         -HD['HSaaaa'][r,w,T,U] * HD['Daaaa'][Dp,Dq,Ds,Dv] ,
                         +HD['HSaaaa'][r,v,T,U] * HD['Daaaa'][Dp,Dq,Ds,Dw] ,
                         +HD['HSaaaa'][s,w,T,U] * HD['Daaaa'][Dp,Dq,Dr,Dv] ,
                         -HD['HSaaaa'][s,v,T,U] * HD['Daaaa'][Dp,Dq,Dr,Dw] ,
                         +HD['HSaaaa'][p,w,T,U] * HD['Daaaa'][Dq,Dv,Dr,Ds] ,
                         -HD['HSaaaa'][p,v,T,U] * HD['Daaaa'][Dq,Dw,Dr,Ds] ))
    if s==v:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,q,actfloor:actceil], HD['Daaaa'][Dp,Dw,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daaaa'][Dq,Dw,Dr,:] ) ,
                             -HD['HSaaaa'][q,w,T,U] * HD['Daa'][Dp,Dr] ,
                             -HD['HSaaaa'][p,q,T,U] * HD['Daa'][Dr,Dw] ,
                             +HD['HSaaaa'][p,w,T,U] * HD['Daa'][Dq,Dr] ))
        if r==w:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,q,actfloor:actceil], HD['Daa'][Dp,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +HD['HSaaaa'][p,q,T,U] ))
    if r==w:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][T,U,q,actfloor:actceil], HD['Daaaa'][Dp,Dv,Ds,:] ) ,
                             -HD['HSaaaa'][p,q,T,U] * HD['Daa'][Ds,Dv] ,
                             +HD['HSaaaa'][p,v,T,U] * HD['Daa'][Dq,Ds] ,
                             -HD['HSaaaa'][q,v,T,U] * HD['Daa'][Dp,Ds] ))
    if s==w:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,U,q,actfloor:actceil], HD['Daaaa'][Dp,Dv,Dr,:] ) ,
                             +HD['HSaaaa'][p,q,T,U] * HD['Daa'][Dr,Dv] ,
                             -HD['HSaaaa'][p,v,T,U] * HD['Daa'][Dq,Dr] ,
                             +HD['HSaaaa'][q,v,T,U] * HD['Daa'][Dp,Dr] ))
        if r==v:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,q,actfloor:actceil], HD['Daa'][Dp,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -HD['HSaaaa'][p,q,T,U] ))
    if r==v:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daaaa'][Dq,Dw,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,U,q,actfloor:actceil], HD['Daaaa'][Dp,Dw,Ds,:] ) ,
                             +HD['HSaaaa'][p,q,T,U] * HD['Daa'][Ds,Dw] ,
                             -HD['HSaaaa'][p,w,T,U] * HD['Daa'][Dq,Ds] ,
                             +HD['HSaaaa'][q,w,T,U] * HD['Daa'][Dp,Ds] ))
    if q==v:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,s,actfloor:actceil], HD['Daaaa'][Dr,Dw,Dp,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daaaa'][Ds,Dw,Dp,:] ) ,
                             +HD['HSaaaa'][r,s,T,U] * HD['Daa'][Dp,Dw] ,
                             +HD['HSaaaa'][s,w,T,U] * HD['Daa'][Dp,Dr] ,
                             -HD['HSaaaa'][r,w,T,U] * HD['Daa'][Dp,Ds] ))
        if p==w:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -HD['HSaaaa'][r,s,T,U] ))
    if p==w:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,s,actfloor:actceil], HD['Daaaa'][Dr,Dv,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                             +HD['HSaaaa'][r,s,T,U] * HD['Daa'][Dq,Dv] ,
                             +HD['HSaaaa'][s,v,T,U] * HD['Daa'][Dq,Dr] ,
                             -HD['HSaaaa'][r,v,T,U] * HD['Daa'][Dq,Ds] ))
    if q==w:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,s,actfloor:actceil], HD['Daaaa'][Dr,Dv,Dp,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dp,:] ) ,
                             -HD['HSaaaa'][r,s,T,U] * HD['Daa'][Dp,Dv] ,
                             -HD['HSaaaa'][s,v,T,U] * HD['Daa'][Dp,Dr] ,
                             +HD['HSaaaa'][r,v,T,U] * HD['Daa'][Dp,Ds] ))
        if p==v:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][T,U,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +HD['HSaaaa'][r,s,T,U] ))
    if p==v:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,s,actfloor:actceil], HD['Daaaa'][Dr,Dw,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daaaa'][Ds,Dw,Dq,:] ) ,
                             -HD['HSaaaa'][r,s,T,U] * HD['Daa'][Dq,Dw] ,
                             -HD['HSaaaa'][s,w,T,U] * HD['Daa'][Dq,Dr] ,
                             +HD['HSaaaa'][r,w,T,U] * HD['Daa'][Dq,Ds] ))
    return result


def lucc_AAAA_VAAA(actfloor, actceil, HD, *e):
    (p,q,r,s),(T,u,v,w) = e
    Dp, Dq, Dr, Ds, Du, Dv, Dw = p-actfloor, q-actfloor, r-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['HSaaaa'][T,r,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dp,Dq,Dv,Dw,Ds,Du,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][T,p,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dq,Du,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][T,s,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dp,Dq,Dv,Dw,Dr,Du,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][T,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dp,Du,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][p,T] * HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] ,
                         -HD['HSaa'][q,T] * HD['Daaaaaa'][Dp,Dv,Dw,Dr,Ds,Du] ,
                         +HD['HSaa'][s,T] * HD['Daaaaaa'][Dp,Dq,Du,Dr,Dv,Dw] ,
                         -HD['HSaa'][r,T] * HD['Daaaaaa'][Dp,Dq,Du,Ds,Dv,Dw] ,
                         +HD['Daaaaaa'][Dp,Dq,Du,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                         -HD['Daaaaaa'][Dp,Dq,Du,Dr,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,T] ) ,
                         -HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,:actfloor,T] ) ,
                         +HD['Daaaaaa'][Dp,Dv,Dw,Dr,Ds,Du] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,T] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaaaa'][Dp,Dq,Du,:,Dr,Dv,Dw,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dq,Dv,Dw,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaaaa'][Dp,Dq,Du,:,Ds,Dv,Dw,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dp,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,u,p,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,u,q,actfloor:actceil], HD['Daaaaaa'][Dp,Dv,Dw,Dr,Ds,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,u,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dp,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,u,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dp,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][p,q,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,s,actfloor:actceil,T], HD['Daaaaaa'][Dp,Dq,Du,Dv,Dw,:] ) ,
                         +HD['HSaaaa'][p,q,u,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                         -HD['HSaaaa'][r,s,u,T] * HD['Daaaa'][Dp,Dq,Dv,Dw] ))
    if q==u:
        result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][T,s,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dv,Dw,Dr,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][T,r,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dp,Dv,Dw,Ds,:,:] ) ))
        result += 2 * fsum(( -HD['HSaa'][p,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][p,:actfloor,:actfloor,T] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][p,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
    if p==u:
        result += fsum(( -einsum( 'ij,ij->', HD['HSaaaa'][T,s,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][T,r,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,:,:] ) ))
        result += 2 * fsum(( +HD['HSaa'][q,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             -HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,T] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
    if r==u:
        result += fsum(( -einsum( 'ij,ij->', HD['HSaaaa'][T,p,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][T,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dp,:,:] ) ))
        result += 2 * fsum(( -HD['HSaa'][s,T] * HD['Daaaa'][Dp,Dq,Dv,Dw] ,
                             +HD['Daaaa'][Dp,Dq,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,T] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dp,Dq,:,Dv,Dw,:] ) ))
    if s==u:
        result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][T,p,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][T,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dp,:,:] ) ))
        result += 2 * fsum(( +HD['HSaa'][r,T] * HD['Daaaa'][Dp,Dq,Dv,Dw] ,
                             -HD['Daaaa'][Dp,Dq,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dp,Dq,:,Dv,Dw,:] ) ))
    return result


def lucc_AAAA_VACA(actfloor, actceil, HD, *e):
    (p,q,r,s),(T,u,V,w) = e
    Dp, Dq, Dr, Ds, Du, Dw = p-actfloor, q-actfloor, r-actfloor, s-actfloor, u-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][p,V,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,Du,Dq,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,p,V,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dq,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][q,V,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,Du,Dp,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,q,V,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dp,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][s,V,actfloor:actceil,T], HD['Daaaaaa'][Dp,Dq,Du,Dr,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,s,V,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dw,Dr,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,V,actfloor:actceil,T], HD['Daaaaaa'][Dp,Dq,Du,Ds,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,r,V,actfloor:actceil], HD['Daaaaaa'][Dp,Dq,Dw,Ds,Du,:] ) ,
                         +HD['HSaaaa'][p,V,u,T] * HD['Daaaa'][Dq,Dw,Dr,Ds] ,
                         -HD['HSaaaa'][r,V,u,T] * HD['Daaaa'][Dp,Dq,Ds,Dw] ,
                         -HD['HSaaaa'][q,V,u,T] * HD['Daaaa'][Dp,Dw,Dr,Ds] ,
                         +HD['HSaaaa'][s,V,u,T] * HD['Daaaa'][Dp,Dq,Dr,Dw] ))
    if q==u:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,s,V,actfloor:actceil], HD['Daaaa'][Dp,Dw,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][p,V,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][T,r,V,actfloor:actceil], HD['Daaaa'][Dp,Dw,Ds,:] ) ))
    if r==u:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,q,V,actfloor:actceil], HD['Daaaa'][Ds,Dw,Dp,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][T,p,V,actfloor:actceil], HD['Daaaa'][Ds,Dw,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,V,actfloor:actceil,T], HD['Daaaa'][Dp,Dq,Dw,:] ) ))
    if s==u:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,q,V,actfloor:actceil], HD['Daaaa'][Dr,Dw,Dp,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,V,actfloor:actceil,T], HD['Daaaa'][Dp,Dq,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,p,V,actfloor:actceil], HD['Daaaa'][Dr,Dw,Dq,:] ) ))
    if p==u:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][q,V,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,r,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Ds,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][T,s,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Dr,:] ) ))
    return result


def lucc_AAAA_VACC(actfloor, actceil, HD, *e):
    (p,q,r,s),(T,u,V,W) = e
    Dp, Dq, Dr, Ds, Du = p-actfloor, q-actfloor, r-actfloor, s-actfloor, u-actfloor
    result = 0
    result += 2 * fsum(( -HD['HSaaaa'][T,s,V,W] * HD['Daaaa'][Dp,Dq,Dr,Du] ,
                         -HD['HSaaaa'][T,p,V,W] * HD['Daaaa'][Dq,Du,Dr,Ds] ,
                         +HD['HSaaaa'][T,r,V,W] * HD['Daaaa'][Dp,Dq,Ds,Du] ,
                         +HD['HSaaaa'][T,q,V,W] * HD['Daaaa'][Dp,Du,Dr,Ds] ))
    if r==u:
        result += 2 * fsum(( +HD['HSaaaa'][T,p,V,W] * HD['Daa'][Dq,Ds] ,
                             -HD['HSaaaa'][T,q,V,W] * HD['Daa'][Dp,Ds] ))
    if s==u:
        result += 2 * fsum(( -HD['HSaaaa'][T,p,V,W] * HD['Daa'][Dq,Dr] ,
                             +HD['HSaaaa'][T,q,V,W] * HD['Daa'][Dp,Dr] ))
    if p==u:
        result += 2 * fsum(( -HD['HSaaaa'][T,r,V,W] * HD['Daa'][Dq,Ds] ,
                             +HD['HSaaaa'][T,s,V,W] * HD['Daa'][Dq,Dr] ))
    if q==u:
        result += 2 * fsum(( +HD['HSaaaa'][T,r,V,W] * HD['Daa'][Dp,Ds] ,
                             -HD['HSaaaa'][T,s,V,W] * HD['Daa'][Dp,Dr] ))
    return result


def lucc_AAAA_VVAA(actfloor, actceil, HD, *e):
    (p,q,r,s),(T,U,v,w) = e
    Dp, Dq, Dr, Ds, Dv, Dw = p-actfloor, q-actfloor, r-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,p,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,U,q,actfloor:actceil], HD['Daaaaaa'][Dp,Dv,Dw,Dr,Ds,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,U,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dp,Dq,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dp,Dq,:] ) ,
                         -HD['HSaaaa'][p,q,T,U] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                         +HD['HSaaaa'][r,s,T,U] * HD['Daaaa'][Dp,Dq,Dv,Dw] ))
    return result


def lucc_AAAA_VVCA(actfloor, actceil, HD, *e):
    (p,q,r,s),(T,U,V,w) = e
    Dp, Dq, Dr, Ds, Dw = p-actfloor, q-actfloor, r-actfloor, s-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -HD['HSaaaa'][p,V,T,U] * HD['Daaaa'][Dq,Dw,Dr,Ds] ,
                         +HD['HSaaaa'][q,V,T,U] * HD['Daaaa'][Dp,Dw,Dr,Ds] ,
                         -HD['HSaaaa'][s,V,T,U] * HD['Daaaa'][Dp,Dq,Dr,Dw] ,
                         +HD['HSaaaa'][r,V,T,U] * HD['Daaaa'][Dp,Dq,Ds,Dw] ))
    return result


def lucc_AAAA_VVCC(actfloor, actceil, HD, *e):
    (p,q,r,s),(T,U,V,W) = e
    Dp, Dq, Dr, Ds = p-actfloor, q-actfloor, r-actfloor, s-actfloor
    result = 0
    return result


def lucc_CAAA_AAAA(actfloor, actceil, HD, *e):
    (P,q,r,s),(t,u,v,w) = e
    Dq, Dr, Ds, Dt, Du, Dv, Dw = q-actfloor, r-actfloor, s-actfloor, t-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][t,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dq,Du,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][u,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dq,Dt,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][w,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dt,Du,Dq,Dv,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][v,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dt,Du,Dq,Dw,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][P,t] * HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] ,
                         -HD['HSaa'][P,u] * HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Dt] ,
                         +HD['HSaa'][P,w] * HD['Daaaaaa'][Dq,Dt,Du,Dr,Ds,Dv] ,
                         -HD['HSaa'][P,v] * HD['Daaaaaa'][Dq,Dt,Du,Dr,Ds,Dw] ,
                         -HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,t] ) ,
                         +HD['Daaaaaa'][Dq,Dt,Du,Dr,Ds,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                         +HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Dt] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,u] ) ,
                         -HD['Daaaaaa'][Dq,Dt,Du,Dr,Ds,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,u,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dt,:,Dq,Dv,Dw,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,v,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dw,:,Dq,Dt,Du,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,w,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,:,Dq,Dt,Du,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,t,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dq,Dv,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][s,v,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dt,Du,Dr,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][v,P,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][w,P,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][s,t,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][s,u,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Dt,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][t,P,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][u,P,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Dt,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][t,u,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][v,w,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dt,Du,Dr,Ds,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,w,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dt,Du,Ds,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,v,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dt,Du,Ds,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][w,P,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][v,P,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,t,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][r,u,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Dt,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][t,P,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][u,P,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Dt,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][s,w,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dt,Du,Dr,Dv,:] ) ,
                         -HD['HSaaaa'][t,P,r,s] * HD['Daaaa'][Dq,Du,Dv,Dw] ,
                         +HD['HSaaaa'][u,P,r,s] * HD['Daaaa'][Dq,Dt,Dv,Dw] ,
                         -HD['HSaaaa'][w,P,r,s] * HD['Daaaa'][Dq,Dv,Dt,Du] ,
                         +HD['HSaaaa'][v,P,r,s] * HD['Daaaa'][Dq,Dw,Dt,Du] ))
    if q==u:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dt,:,Dv,Dw,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dt,Dv,Dw,:] ) ,
                             -HD['HSaa'][P,t] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +HD['HSaa'][P,s] * HD['Daaaa'][Dr,Dt,Dv,Dw] ,
                             -HD['HSaa'][P,r] * HD['Daaaa'][Ds,Dt,Dv,Dw] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dt,Dv,Dw,:] ) ,
                             -HD['Daaaa'][Dr,Dt,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             +HD['Daaaa'][Ds,Dt,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,t] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,:,Dv,Dw,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,:,Dv,Dw,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,t,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,t,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,t,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,s,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] ) ))
    if q==t:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dv,Dw,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             -HD['HSaa'][P,s] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                             +HD['HSaa'][P,u] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +HD['HSaa'][P,r] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             -HD['Daaaa'][Ds,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             -HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,u] ) ,
                             +HD['Daaaa'][Dr,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dv,Dw,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,u,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dv,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,u,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,u,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
    if q==v:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dw,:,Dt,Du,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dt,Du,:] ) ,
                             -HD['HSaa'][P,r] * HD['Daaaa'][Ds,Dw,Dt,Du] ,
                             +HD['HSaa'][P,s] * HD['Daaaa'][Dr,Dw,Dt,Du] ,
                             -HD['HSaa'][P,w] * HD['Daaaa'][Dr,Ds,Dt,Du] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dt,Du,:] ) ,
                             +HD['Daaaa'][Dr,Ds,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                             +HD['Daaaa'][Ds,Dw,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             -HD['Daaaa'][Dr,Dw,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dw,:,Dt,Du,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dw,:,Dt,Du,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,w,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dt,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,w,P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,w,P,actfloor:actceil], HD['Daaaa'][Dt,Du,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,s,P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] ) ))
    if q==w:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,:,Dt,Du,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dt,Du,:] ) ,
                             +HD['HSaa'][P,r] * HD['Daaaa'][Ds,Dv,Dt,Du] ,
                             -HD['HSaa'][P,s] * HD['Daaaa'][Dr,Dv,Dt,Du] ,
                             +HD['HSaa'][P,v] * HD['Daaaa'][Dr,Ds,Dt,Du] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dt,Du,:] ) ,
                             +HD['Daaaa'][Dr,Dv,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             -HD['Daaaa'][Dr,Ds,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                             -HD['Daaaa'][Ds,Dv,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,:,Dt,Du,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,v,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dt,Du,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,:,Dt,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,v,P,actfloor:actceil], HD['Daaaa'][Dt,Du,Ds,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,v,P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dr,:] ) ))
    if s==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Dt,Du,:,Dq,Dw,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][w,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,Dw,:] ) ,
                             -HD['HSaa'][P,r] * HD['Daaaa'][Dq,Dw,Dt,Du] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,Dw,:] ) ,
                             +HD['Daaaa'][Dq,Dw,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dq,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][w,P,r,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ))
        if r==w:
            result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dq,:,:] )
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ))
    if r==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Ds,Dt,Du,:,Dq,Dv,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][v,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dv,:] ) ,
                             -HD['HSaa'][P,s] * HD['Daaaa'][Dq,Dv,Dt,Du] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dv,:] ) ,
                             +HD['Daaaa'][Dq,Dv,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dq,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,P,s,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ))
    if s==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Dt,Du,:,Dq,Dv,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][v,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,Dv,:] ) ,
                             +HD['HSaa'][P,r] * HD['Daaaa'][Dq,Dv,Dt,Du] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,Dv,:] ) ,
                             -HD['Daaaa'][Dq,Dv,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dq,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,P,r,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ))
        if r==v:
            result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dq,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ))
    if r==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Ds,Dt,Du,:,Dq,Dw,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][w,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dw,:] ) ,
                             +HD['HSaa'][P,s] * HD['Daaaa'][Dq,Dw,Dt,Du] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dw,:] ) ,
                             -HD['Daaaa'][Dq,Dw,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dq,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][w,P,s,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ))
    if s==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Dv,Dw,:,Dq,Dt,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][t,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Dt,:] ) ,
                             -HD['HSaa'][P,r] * HD['Daaaa'][Dq,Dt,Dv,Dw] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Dt,:] ) ,
                             +HD['Daaaa'][Dq,Dt,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,Dt,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][t,P,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ))
        if r==t:
            result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,:,:] )
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ))
    if r==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Ds,Dv,Dw,:,Dq,Du,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][u,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ,
                             -HD['HSaa'][P,s] * HD['Daaaa'][Dq,Du,Dv,Dw] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ,
                             +HD['Daaaa'][Dq,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][u,P,s,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ))
    if r==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Ds,Dv,Dw,:,Dq,Dt,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][t,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Dt,:] ) ,
                             +HD['HSaa'][P,s] * HD['Daaaa'][Dq,Dt,Dv,Dw] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Dt,:] ) ,
                             -HD['Daaaa'][Dq,Dt,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,Dt,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][t,P,s,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ))
        if s==t:
            result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ))
    if s==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Dv,Dw,:,Dq,Du,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][u,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Du,:] ) ,
                             +HD['HSaa'][P,r] * HD['Daaaa'][Dq,Du,Dv,Dw] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Du,:] ) ,
                             -HD['Daaaa'][Dq,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][u,P,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ))
    return result


def lucc_CAAA_CAAA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,u,v,w) = e
    Dq, Dr, Ds, Du, Dv, Dw = q-actfloor, r-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dq,Du,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] ,
                         +HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dq,Dv,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][r,T,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][w,P,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dq,Dv,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][v,P,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dq,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,w,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dq,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dq,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][s,T,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ,
                         -HD['HSaaaa'][v,P,T,r] * HD['Daaaa'][Dq,Dw,Ds,Du] ,
                         +HD['HSaaaa'][P,T,v,w] * HD['Daaaa'][Dq,Du,Dr,Ds] ,
                         +HD['HSaaaa'][P,T,r,s] * HD['Daaaa'][Dq,Du,Dv,Dw] ,
                         -HD['HSaaaa'][P,T,s,w] * HD['Daaaa'][Dq,Du,Dr,Dv] ,
                         +HD['HSaaaa'][P,T,s,v] * HD['Daaaa'][Dq,Du,Dr,Dw] ,
                         -HD['HSaaaa'][w,P,T,s] * HD['Daaaa'][Dq,Dv,Dr,Du] ,
                         +HD['HSaaaa'][v,P,T,s] * HD['Daaaa'][Dq,Dw,Dr,Du] ,
                         +HD['HSaaaa'][P,T,r,w] * HD['Daaaa'][Dq,Du,Ds,Dv] ,
                         -HD['HSaaaa'][P,T,r,v] * HD['Daaaa'][Dq,Du,Ds,Dw] ,
                         +HD['HSaaaa'][w,P,T,r] * HD['Daaaa'][Dq,Dv,Ds,Du] ))
    if P==T:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dv,Dw,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dq,Dv,Dw,:,Dr,Du,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dq,Dv,Dw,:,Ds,Du,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][r,s,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Du,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][q,w,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][q,v,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dw,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                             +HD['HSaa'][r,w] * HD['Daaaa'][Dq,Dv,Ds,Du] ,
                             -HD['HSaa'][r,v] * HD['Daaaa'][Dq,Dw,Ds,Du] ,
                             -HD['HSaa'][s,w] * HD['Daaaa'][Dq,Dv,Dr,Du] ,
                             +HD['HSaa'][s,v] * HD['Daaaa'][Dq,Dw,Dr,Du] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Du,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                             +HD['Daaaa'][Dq,Dv,Dr,Du] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,w] ) ,
                             -HD['Daaaa'][Dq,Dw,Dr,Du] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,v] ) ,
                             +HD['Daaaa'][Dq,Dw,Ds,Du] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,v] ) ,
                             -HD['Daaaa'][Dq,Dv,Ds,Du] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,w] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,v,actfloor:actceil], HD['Daaaaaa'][Dq,Dw,:,Dr,Du,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,v,actfloor:actceil], HD['Daaaaaa'][Dq,Dw,:,Ds,Du,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,w,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,:,Ds,Du,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,w,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,:,Dr,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,w,r,actfloor:actceil], HD['Daaaa'][Ds,Du,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,w,actfloor:actceil], HD['Daaaa'][Dq,Dv,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,s,v,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,w,s,actfloor:actceil], HD['Daaaa'][Dr,Du,Dq,:] ) ,
                             +HD['HSaaaa'][r,s,v,w] * HD['Daa'][Dq,Du] ))
        if s==w:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,:,Du,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dv,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][q,v,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Du,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaa'][Dq,Dv,Du,:] ) ,
                                 +HD['HSaa'][r,v] * HD['Daa'][Dq,Du] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dv,Du,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                                 -HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,v] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dq,:,Du,:] ) ))
            if r==v:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ))
        if s==v:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dw,:,Du,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dw,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][q,w,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Du,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                                 +einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                                 -HD['HSaa'][r,w] * HD['Daa'][Dq,Du] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                                 +HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,w] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,w,actfloor:actceil], HD['Daaaa'][Dq,:,Du,:] ) ))
            if r==w:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ))
        if r==w:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dv,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,:,Du,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][q,v,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaa'][Dq,Dv,Du,:] ) ,
                                 -HD['HSaa'][s,v] * HD['Daa'][Dq,Du] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dv,Du,:] ) ,
                                 +HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,v] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dq,:,Du,:] ) ))
        if r==v:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dw,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dw,:,Du,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][q,w,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                                 +HD['HSaa'][s,w] * HD['Daa'][Dq,Du] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                                 -HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,w] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,w,actfloor:actceil], HD['Daaaa'][Dq,:,Du,:] ) ))
    if s==v:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dw,Dr,Du] ,
                             -HD['Daaaa'][Dq,Dw,Dr,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dq,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,T,P,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][w,P,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dq,:] ) ,
                             -HD['HSaaaa'][w,P,T,r] * HD['Daa'][Dq,Du] ))
        if r==w:
            result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Dq,Du] ,
                                 +HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaa'][Du,:,Dq,:] ) ))
    if r==w:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dv,Ds,Du] ,
                             -HD['Daaaa'][Dq,Dv,Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dq,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,P,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,T,P,actfloor:actceil], HD['Daaaa'][Dq,Dv,Du,:] ) ,
                             -HD['HSaaaa'][v,P,T,s] * HD['Daa'][Dq,Du] ))
    if s==w:
        result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dv,Dr,Du] ,
                             +HD['Daaaa'][Dq,Dv,Dr,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dq,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,T,P,actfloor:actceil], HD['Daaaa'][Dq,Dv,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,P,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dq,:] ) ,
                             +HD['HSaaaa'][v,P,T,r] * HD['Daa'][Dq,Du] ))
        if r==v:
            result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daa'][Dq,Du] ,
                                 -HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaa'][Du,:,Dq,:] ) ))
    if r==v:
        result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dw,Ds,Du] ,
                             +HD['Daaaa'][Dq,Dw,Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dq,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][w,P,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,T,P,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                             +HD['HSaaaa'][w,P,T,s] * HD['Daa'][Dq,Du] ))
    if q==v:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Du,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,T,r,actfloor:actceil], HD['Daaaa'][Ds,Dw,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,T,w,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,T,s,actfloor:actceil], HD['Daaaa'][Dr,Dw,Du,:] ) ,
                             +HD['HSaaaa'][P,T,r,s] * HD['Daa'][Du,Dw] ,
                             -HD['HSaaaa'][P,T,r,w] * HD['Daa'][Ds,Du] ,
                             +HD['HSaaaa'][P,T,s,w] * HD['Daa'][Dr,Du] ))
    if q==w:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Du,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,T,r,actfloor:actceil], HD['Daaaa'][Ds,Dv,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,T,s,actfloor:actceil], HD['Daaaa'][Dr,Dv,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ,
                             -HD['HSaaaa'][P,T,r,s] * HD['Daa'][Du,Dv] ,
                             +HD['HSaaaa'][P,T,r,v] * HD['Daa'][Ds,Du] ,
                             -HD['HSaaaa'][P,T,s,v] * HD['Daa'][Dr,Du] ))
    return result


def lucc_CAAA_CCAA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,U,v,w) = e
    Dq, Dr, Ds, Dv, Dw = q-actfloor, r-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,:] ) ,
                         +HD['HSaaaa'][v,P,T,U] * HD['Daaaa'][Dq,Dw,Dr,Ds] ,
                         -HD['HSaaaa'][w,P,T,U] * HD['Daaaa'][Dq,Dv,Dr,Ds] ))
    if P==U:
        result += fsum(( -einsum( 'ij,ij->', HD['HSaaaa'][s,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][r,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,:,:] ) ))
        result += 2 * fsum(( -HD['HSaa'][q,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             -HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][q,w,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][q,v,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,T,w,actfloor:actceil], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,T,v,actfloor:actceil], HD['Daaaa'][Dq,Dw,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,T,w,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,T,v,actfloor:actceil], HD['Daaaa'][Dq,Dw,Ds,:] ) ,
                             -HD['HSaaaa'][r,T,v,w] * HD['Daa'][Dq,Ds] ,
                             +HD['HSaaaa'][s,T,v,w] * HD['Daa'][Dq,Dr] ))
        if s==v:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][r,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] )
            result += 2 * fsum(( +HD['HSaa'][q,T] * HD['Daa'][Dr,Dw] ,
                                 +HD['Daa'][Dr,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,T,w,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][q,w,T,actfloor:actceil], HD['Daa'][Dr,:] ) ))
            if r==w:
                result += 2 * fsum(( -HD['HSaa'][q,T] ,
                                     -einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daa'][:,:] ) ))
        if r==w:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][s,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dv,:,:] )
            result += 2 * fsum(( +HD['HSaa'][q,T] * HD['Daa'][Ds,Dv] ,
                                 +HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][q,v,T,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][s,T,v,actfloor:actceil], HD['Daa'][Dq,:] ) ))
        if s==w:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][r,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dv,:,:] )
            result += 2 * fsum(( -HD['HSaa'][q,T] * HD['Daa'][Dr,Dv] ,
                                 -HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][q,v,T,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][r,T,v,actfloor:actceil], HD['Daa'][Dq,:] ) ))
            if r==v:
                result += 2 * fsum(( +HD['HSaa'][q,T] ,
                                     +einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daa'][:,:] ) ))
        if r==v:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][s,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] )
            result += 2 * fsum(( -HD['HSaa'][q,T] * HD['Daa'][Ds,Dw] ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,T,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Ds,:,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][s,T,w,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][q,w,T,actfloor:actceil], HD['Daa'][Ds,:] ) ))
    if P==T:
        result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][s,U,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][r,U,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,:,:] ) ))
        result += 2 * fsum(( +HD['HSaa'][q,U] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,U,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,U,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,U,v,actfloor:actceil], HD['Daaaa'][Dq,Dw,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][q,w,U,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][q,v,U,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,U,w,actfloor:actceil], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,U,v,actfloor:actceil], HD['Daaaa'][Dq,Dw,Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,U,w,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                             +HD['HSaaaa'][r,U,v,w] * HD['Daa'][Dq,Ds] ,
                             -HD['HSaaaa'][s,U,v,w] * HD['Daa'][Dq,Dr] ))
        if s==v:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][r,U,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] )
            result += 2 * fsum(( -HD['HSaa'][q,U] * HD['Daa'][Dr,Dw] ,
                                 -HD['Daa'][Dr,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,U,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,U,actfloor:actceil], HD['Daaaa'][Dr,:,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][r,U,w,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][q,w,U,actfloor:actceil], HD['Daa'][Dr,:] ) ))
            if r==w:
                result += 2 * fsum(( +HD['HSaa'][q,U] ,
                                     +einsum( 'II->', HD['HSaaaa'][q,:actfloor,U,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,U,actfloor:actceil], HD['Daa'][:,:] ) ))
        if r==w:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][s,U,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dv,:,:] )
            result += 2 * fsum(( -HD['HSaa'][q,U] * HD['Daa'][Ds,Dv] ,
                                 -HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,U,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,U,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][q,v,U,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][s,U,v,actfloor:actceil], HD['Daa'][Dq,:] ) ))
        if s==w:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][r,U,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dv,:,:] )
            result += 2 * fsum(( +HD['HSaa'][q,U] * HD['Daa'][Dr,Dv] ,
                                 +HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,U,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,U,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][q,v,U,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,U,v,actfloor:actceil], HD['Daa'][Dq,:] ) ))
            if r==v:
                result += 2 * fsum(( -HD['HSaa'][q,U] ,
                                     -einsum( 'II->', HD['HSaaaa'][q,:actfloor,U,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,U,actfloor:actceil], HD['Daa'][:,:] ) ))
        if r==v:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][s,U,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] )
            result += 2 * fsum(( +HD['HSaa'][q,U] * HD['Daa'][Ds,Dw] ,
                                 +HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,U,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,U,actfloor:actceil], HD['Daaaa'][Ds,:,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][s,U,w,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][q,w,U,actfloor:actceil], HD['Daa'][Ds,:] ) ))
    if s==v:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daaaa'][Dq,Dw,Dr,:] ) ,
                             -HD['HSaaaa'][w,P,T,U] * HD['Daa'][Dq,Dr] ))
        if r==w:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daa'][Dq,:] )
    if r==w:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                             -HD['HSaaaa'][v,P,T,U] * HD['Daa'][Dq,Ds] ))
    if s==w:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                             +HD['HSaaaa'][v,P,T,U] * HD['Daa'][Dq,Dr] ))
        if r==v:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daa'][Dq,:] )
    if r==v:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daaaa'][Dq,Dw,Ds,:] ) ,
                             +HD['HSaaaa'][w,P,T,U] * HD['Daa'][Dq,Ds] ))
    return result


def lucc_CAAA_VAAA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,u,v,w) = e
    Dq, Dr, Ds, Du, Dv, Dw = q-actfloor, r-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += +einsum( 'ij,ij->', HD['HSaaaa'][T,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dq,Du,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] ,
                         -HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,T], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dq,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,u,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,P,s,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,s,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,P,r,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][T,r,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                         -HD['HSaaaa'][T,P,r,s] * HD['Daaaa'][Dq,Du,Dv,Dw] ))
    if q==u:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][T,s,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,r,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
    if r==u:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][T,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,:,:] )
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,P,s,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] )
    if s==u:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][T,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,P,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] )
    return result


def lucc_CAAA_VACA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,u,V,w) = e
    Dq, Dr, Ds, Du, Dw = q-actfloor, r-actfloor, s-actfloor, u-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,Du,Dq,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,P,V,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dq,Du,:] ) ,
                         +HD['HSaaaa'][P,V,r,T] * HD['Daaaa'][Dq,Dw,Ds,Du] ,
                         +HD['HSaaaa'][P,V,u,T] * HD['Daaaa'][Dq,Dw,Dr,Ds] ,
                         -HD['HSaaaa'][T,P,V,s] * HD['Daaaa'][Dq,Du,Dr,Dw] ,
                         -HD['HSaaaa'][P,V,s,T] * HD['Daaaa'][Dq,Dw,Dr,Du] ,
                         +HD['HSaaaa'][T,P,V,r] * HD['Daaaa'][Dq,Du,Ds,Dw] ))
    if P==V:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][T,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Du,:,:] )
        result += 2 * fsum(( -HD['HSaa'][s,T] * HD['Daaaa'][Dq,Du,Dr,Dw] ,
                             +HD['HSaa'][r,T] * HD['Daaaa'][Dq,Du,Ds,Dw] ,
                             -HD['Daaaa'][Dq,Du,Ds,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                             +HD['Daaaa'][Dq,Du,Dr,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,T] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dq,Du,:,Ds,Dw,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dq,Du,:,Dr,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,s,actfloor:actceil,T], HD['Daaaa'][Dq,Du,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][T,u,s,actfloor:actceil], HD['Daaaa'][Dr,Dw,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,u,r,actfloor:actceil], HD['Daaaa'][Ds,Dw,Dq,:] ) ,
                             +HD['HSaaaa'][r,s,u,T] * HD['Daa'][Dq,Dw] ))
        if r==u:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][T,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dw,:,:] )
            result += 2 * fsum(( +HD['HSaa'][s,T] * HD['Daa'][Dq,Dw] ,
                                 -HD['Daa'][Dq,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,T] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Dq,:,Dw,:] ) ))
        if s==u:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][T,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dw,:,:] )
            result += 2 * fsum(( -HD['HSaa'][r,T] * HD['Daa'][Dq,Dw] ,
                                 +HD['Daa'][Dq,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Dq,:,Dw,:] ) ))
    if r==u:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,P,V,actfloor:actceil], HD['Daaaa'][Ds,Dw,Dq,:] ) ,
                             +HD['HSaaaa'][T,P,V,s] * HD['Daa'][Dq,Dw] ))
    if s==u:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,P,V,actfloor:actceil], HD['Daaaa'][Dr,Dw,Dq,:] ) ,
                             -HD['HSaaaa'][T,P,V,r] * HD['Daa'][Dq,Dw] ))
    if q==u:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             +HD['HSaaaa'][P,V,s,T] * HD['Daa'][Dr,Dw] ,
                             -HD['HSaaaa'][P,V,r,T] * HD['Daa'][Ds,Dw] ))
    return result


def lucc_CAAA_VACC(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,u,V,W) = e
    Dq, Dr, Ds, Du = q-actfloor, r-actfloor, s-actfloor, u-actfloor
    result = 0
    result += 2 * -HD['HSaaaa'][T,P,V,W] * HD['Daaaa'][Dq,Du,Dr,Ds]
    if P==W:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,q,V,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,V,actfloor:actceil,T], HD['Daaaa'][Dq,Du,Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,V,actfloor:actceil,T], HD['Daaaa'][Dq,Du,Ds,:] ) ,
                             +HD['HSaaaa'][s,V,u,T] * HD['Daa'][Dq,Dr] ,
                             -HD['HSaaaa'][r,V,u,T] * HD['Daa'][Dq,Ds] ))
        if r==u:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][s,V,actfloor:actceil,T], HD['Daa'][Dq,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][T,q,V,actfloor:actceil], HD['Daa'][Ds,:] ) ))
        if s==u:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][r,V,actfloor:actceil,T], HD['Daa'][Dq,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][T,q,V,actfloor:actceil], HD['Daa'][Dr,:] ) ))
    if P==V:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,q,W,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,W,actfloor:actceil,T], HD['Daaaa'][Dq,Du,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,W,actfloor:actceil,T], HD['Daaaa'][Dq,Du,Ds,:] ) ,
                             -HD['HSaaaa'][s,W,u,T] * HD['Daa'][Dq,Dr] ,
                             +HD['HSaaaa'][r,W,u,T] * HD['Daa'][Dq,Ds] ))
        if s==u:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][r,W,actfloor:actceil,T], HD['Daa'][Dq,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][T,q,W,actfloor:actceil], HD['Daa'][Dr,:] ) ))
        if r==u:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,q,W,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][s,W,actfloor:actceil,T], HD['Daa'][Dq,:] ) ))
    if r==u:
        result += 2 * +HD['HSaaaa'][T,P,V,W] * HD['Daa'][Dq,Ds]
    if s==u:
        result += 2 * -HD['HSaaaa'][T,P,V,W] * HD['Daa'][Dq,Dr]
    return result


def lucc_CAAA_VVAA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,U,v,w) = e
    Dq, Dr, Ds, Dv, Dw = q-actfloor, r-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,U,P,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,:] )
    return result


def lucc_CAAA_VVCA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,U,V,w) = e
    Dq, Dr, Ds, Dw = q-actfloor, r-actfloor, s-actfloor, w-actfloor
    result = 0
    result += 2 * -HD['HSaaaa'][P,V,T,U] * HD['Daaaa'][Dq,Dw,Dr,Ds]
    if P==V:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,s,actfloor:actceil], HD['Daaaa'][Dr,Dw,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daaaa'][Ds,Dw,Dq,:] ) ,
                             -HD['HSaaaa'][r,s,T,U] * HD['Daa'][Dq,Dw] ))
    return result


def lucc_CAAA_VVCC(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,U,V,W) = e
    Dq, Dr, Ds = q-actfloor, r-actfloor, s-actfloor
    result = 0
    if P==W:
        result += 2 * fsum(( -HD['HSaaaa'][s,V,T,U] * HD['Daa'][Dq,Dr] ,
                             +HD['HSaaaa'][r,V,T,U] * HD['Daa'][Dq,Ds] ))
    if P==V:
        result += 2 * fsum(( +HD['HSaaaa'][s,W,T,U] * HD['Daa'][Dq,Dr] ,
                             -HD['HSaaaa'][r,W,T,U] * HD['Daa'][Dq,Ds] ))
    return result


def lucc_CCAA_AAAA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(t,u,v,w) = e
    Dr, Ds, Dt, Du, Dv, Dw = r-actfloor, s-actfloor, t-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,t,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,Q,u,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dt,Dv,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,Q,w,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dt,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,Q,v,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dt,Du,:] ) ,
                         +HD['HSaaaa'][P,Q,t,u] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                         -HD['HSaaaa'][P,Q,r,t] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                         +HD['HSaaaa'][P,Q,r,u] * HD['Daaaa'][Ds,Dt,Dv,Dw] ,
                         -HD['HSaaaa'][P,Q,v,w] * HD['Daaaa'][Dr,Ds,Dt,Du] ,
                         +HD['HSaaaa'][P,Q,s,w] * HD['Daaaa'][Dr,Dv,Dt,Du] ,
                         -HD['HSaaaa'][P,Q,s,v] * HD['Daaaa'][Dr,Dw,Dt,Du] ,
                         +HD['HSaaaa'][P,Q,s,t] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                         -HD['HSaaaa'][P,Q,s,u] * HD['Daaaa'][Dr,Dt,Dv,Dw] ,
                         -HD['HSaaaa'][P,Q,r,w] * HD['Daaaa'][Ds,Dv,Dt,Du] ,
                         +HD['HSaaaa'][P,Q,r,v] * HD['Daaaa'][Ds,Dw,Dt,Du] ))
    if s==v:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dw,:,:] )
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,r,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] )
        if r==w:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,Du,:,:] )
    if r==w:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dv,:,:] )
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,s,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] )
    if s==w:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dv,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,r,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] )
        if r==v:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,Du,:,:] )
    if r==v:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dw,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,s,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] )
    if s==u:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dt,:,:] )
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] )
        if r==t:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    if r==t:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Du,:,:] )
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,s,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] )
    if r==u:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dt,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,s,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] )
        if s==t:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    if s==t:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Du,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] )
    return result


def lucc_CCAA_CAAA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,u,v,w) = e
    Dr, Ds, Du, Dv, Dw = r-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                         +HD['HSaaaa'][P,Q,T,r] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                         -HD['HSaaaa'][P,Q,T,s] * HD['Daaaa'][Dr,Du,Dv,Dw] ))
    if Q==T:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dv,Dw,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][w,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][v,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dw,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             -HD['HSaa'][P,s] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                             +HD['HSaa'][P,r] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             +HD['Daaaa'][Dr,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                             -HD['Daaaa'][Ds,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dv,Dw,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dv,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,P,s,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][w,P,r,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][w,P,s,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,P,r,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                             -HD['HSaaaa'][v,P,r,s] * HD['Daa'][Du,Dw] ,
                             +HD['HSaaaa'][w,P,r,s] * HD['Daa'][Du,Dv] ))
        if s==v:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dw,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][w,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Du,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                                 +HD['HSaa'][P,r] * HD['Daa'][Du,Dw] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                                 -HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][w,P,r,actfloor:actceil], HD['Daa'][Du,:] ) ))
            if r==w:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ))
        if r==w:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dv,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][v,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] ) ,
                                 +HD['HSaa'][P,s] * HD['Daa'][Du,Dv] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] ) ,
                                 -HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaa'][Du,:,Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][v,P,s,actfloor:actceil], HD['Daa'][Du,:] ) ))
        if s==w:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dv,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][v,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Du,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                                 -HD['HSaa'][P,r] * HD['Daa'][Du,Dv] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                                 +HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,r] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,r,actfloor:actceil], HD['Daaaa'][Du,:,Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][v,P,r,actfloor:actceil], HD['Daa'][Du,:] ) ))
            if r==v:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ))
        if r==v:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dw,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][w,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 -HD['HSaa'][P,s] * HD['Daa'][Du,Dw] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,s] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,s,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][w,P,s,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if P==T:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dv,Dw,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][w,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][v,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dw,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             +HD['HSaa'][Q,s] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                             -HD['HSaa'][Q,r] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             -HD['Daaaa'][Dr,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,s] ) ,
                             +HD['Daaaa'][Ds,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,r] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,s,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dv,Dw,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,r,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dv,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][w,Q,r,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,s,Q,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][w,Q,s,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,Q,r,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,Q,s,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                             -HD['HSaaaa'][w,Q,r,s] * HD['Daa'][Du,Dv] ,
                             +HD['HSaaaa'][v,Q,r,s] * HD['Daa'][Du,Dw] ))
        if s==v:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dw,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][w,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Du,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                                 -HD['HSaa'][Q,r] * HD['Daa'][Du,Dw] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                                 +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,r] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,r,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][w,Q,r,actfloor:actceil], HD['Daa'][Du,:] ) ))
            if r==w:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ))
        if r==w:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dv,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][v,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] ) ,
                                 -HD['HSaa'][Q,s] * HD['Daa'][Du,Dv] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] ) ,
                                 +HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,s] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,s,actfloor:actceil], HD['Daaaa'][Du,:,Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][v,Q,s,actfloor:actceil], HD['Daa'][Du,:] ) ))
        if s==w:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dv,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][v,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Du,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                                 +HD['HSaa'][Q,r] * HD['Daa'][Du,Dv] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                                 -HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,r] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,r,actfloor:actceil], HD['Daaaa'][Du,:,Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][v,Q,r,actfloor:actceil], HD['Daa'][Du,:] ) ))
            if r==v:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ))
        if r==v:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dw,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][w,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 +HD['HSaa'][Q,s] * HD['Daa'][Du,Dw] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 -HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,s] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,s,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][w,Q,s,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if s==v:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                             +HD['HSaaaa'][P,Q,T,r] * HD['Daa'][Du,Dw] ))
        if r==w:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daa'][Du,:] )
    if r==w:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] ) ,
                             +HD['HSaaaa'][P,Q,T,s] * HD['Daa'][Du,Dv] ))
    if s==w:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                             -HD['HSaaaa'][P,Q,T,r] * HD['Daa'][Du,Dv] ))
        if r==v:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daa'][Du,:] )
    if r==v:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                             -HD['HSaaaa'][P,Q,T,s] * HD['Daa'][Du,Dw] ))
    return result


def lucc_CCAA_CCAA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,U,v,w) = e
    Dr, Ds, Dv, Dw = r-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * -HD['HSaaaa'][P,Q,T,U] * HD['Daaaa'][Dr,Ds,Dv,Dw]
    if Q==U:
        result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,T,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][w,P,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,T,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,P,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             +HD['HSaaaa'][w,P,T,r] * HD['Daa'][Ds,Dv] ,
                             +HD['HSaaaa'][v,P,T,s] * HD['Daa'][Dr,Dw] ,
                             -HD['HSaaaa'][w,P,T,s] * HD['Daa'][Dr,Dv] ,
                             -HD['HSaaaa'][v,P,T,r] * HD['Daa'][Ds,Dw] ))
        if P==T:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dr,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Ds,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,s,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                                 -HD['HSaa'][s,w] * HD['Daa'][Dr,Dv] ,
                                 +HD['HSaa'][s,v] * HD['Daa'][Dr,Dw] ,
                                 +HD['HSaa'][r,w] * HD['Daa'][Ds,Dv] ,
                                 -HD['HSaa'][r,v] * HD['Daa'][Ds,Dw] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                                 +HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,w] ) ,
                                 -HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,w] ) ,
                                 +HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,v] ) ,
                                 -HD['Daa'][Dr,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,v] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,w,actfloor:actceil], HD['Daaaa'][Dv,:,Ds,:] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dw,:,Ds,:] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dw,:,Dr,:] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,w,actfloor:actceil], HD['Daaaa'][Dv,:,Dr,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][v,w,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][r,s,w,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,s,v,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][v,w,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +HD['HSaaaa'][r,s,v,w] ))
            if s==v:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     -HD['HSaa'][r,w] ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     +einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,w] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,w,actfloor:actceil], HD['Daa'][:,:] ) ))
            if r==v:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     +HD['HSaa'][s,w] ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     -einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,w] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,w,actfloor:actceil], HD['Daa'][:,:] ) ))
            if r==w:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     -HD['HSaa'][s,v] ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     +einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,v] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,v,actfloor:actceil], HD['Daa'][:,:] ) ))
            if s==w:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     +HD['HSaa'][r,v] ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     -einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,v] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,v,actfloor:actceil], HD['Daa'][:,:] ) ))
        if s==v:
            result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daa'][Dr,Dw] ,
                                 -HD['Daa'][Dr,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][w,P,T,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][r,T,P,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 -HD['HSaaaa'][w,P,T,r] ))
            if r==w:
                result += 2 * fsum(( +HD['HSaa'][P,T] ,
                                     +einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daa'][:,:] ) ))
        if r==w:
            result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daa'][Ds,Dv] ,
                                 -HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][v,P,T,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][s,T,P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 -HD['HSaaaa'][v,P,T,s] ))
        if s==w:
            result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Dr,Dv] ,
                                 +HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,T,P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][v,P,T,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +HD['HSaaaa'][v,P,T,r] ))
            if r==v:
                result += 2 * fsum(( -HD['HSaa'][P,T] ,
                                     -einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daa'][:,:] ) ))
        if r==v:
            result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Ds,Dw] ,
                                 +HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,T,actfloor:actceil], HD['Daaaa'][Ds,:,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][s,T,P,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][w,P,T,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +HD['HSaaaa'][w,P,T,s] ))
    if Q==T:
        result += 2 * fsum(( -HD['HSaa'][P,U] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             -HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,U,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,U,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,P,U,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,U,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][w,P,U,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,U,P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                             -HD['HSaaaa'][v,P,U,s] * HD['Daa'][Dr,Dw] ,
                             -HD['HSaaaa'][w,P,U,r] * HD['Daa'][Ds,Dv] ,
                             +HD['HSaaaa'][v,P,U,r] * HD['Daa'][Ds,Dw] ,
                             +HD['HSaaaa'][w,P,U,s] * HD['Daa'][Dr,Dv] ))
        if P==U:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dr,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Ds,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,s,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                                 -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                                 +HD['HSaa'][s,w] * HD['Daa'][Dr,Dv] ,
                                 -HD['HSaa'][s,v] * HD['Daa'][Dr,Dw] ,
                                 -HD['HSaa'][r,w] * HD['Daa'][Ds,Dv] ,
                                 +HD['HSaa'][r,v] * HD['Daa'][Ds,Dw] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                                 -HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,w] ) ,
                                 +HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,w] ) ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,v] ) ,
                                 +HD['Daa'][Dr,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,v] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,w,actfloor:actceil], HD['Daaaa'][Dv,:,Ds,:] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dw,:,Ds,:] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dw,:,Dr,:] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,w,actfloor:actceil], HD['Daaaa'][Dv,:,Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][v,w,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,s,w,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][r,s,v,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][v,w,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -HD['HSaaaa'][r,s,v,w] ))
            if s==v:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     +HD['HSaa'][r,w] ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     -einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,w] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,w,actfloor:actceil], HD['Daa'][:,:] ) ))
            if r==v:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     -HD['HSaa'][s,w] ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     +einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,w] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,w,actfloor:actceil], HD['Daa'][:,:] ) ))
            if r==w:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     +HD['HSaa'][s,v] ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     -einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,v] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,v,actfloor:actceil], HD['Daa'][:,:] ) ))
            if s==w:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     -HD['HSaa'][r,v] ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     +einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,v] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,v,actfloor:actceil], HD['Daa'][:,:] ) ))
        if s==v:
            result += 2 * fsum(( +HD['HSaa'][P,U] * HD['Daa'][Dr,Dw] ,
                                 +HD['Daa'][Dr,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,U,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,U,actfloor:actceil], HD['Daaaa'][Dr,:,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,U,P,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][w,P,U,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +HD['HSaaaa'][w,P,U,r] ))
            if r==w:
                result += 2 * fsum(( -HD['HSaa'][P,U] ,
                                     -einsum( 'II->', HD['HSaaaa'][P,:actfloor,U,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,U,actfloor:actceil], HD['Daa'][:,:] ) ))
        if r==w:
            result += 2 * fsum(( +HD['HSaa'][P,U] * HD['Daa'][Ds,Dv] ,
                                 +HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,U,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,U,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][v,P,U,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][s,U,P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 +HD['HSaaaa'][v,P,U,s] ))
        if s==w:
            result += 2 * fsum(( -HD['HSaa'][P,U] * HD['Daa'][Dr,Dv] ,
                                 -HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,U,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,U,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][r,U,P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][v,P,U,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -HD['HSaaaa'][v,P,U,r] ))
            if r==v:
                result += 2 * fsum(( +HD['HSaa'][P,U] ,
                                     +einsum( 'II->', HD['HSaaaa'][P,:actfloor,U,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,U,actfloor:actceil], HD['Daa'][:,:] ) ))
        if r==v:
            result += 2 * fsum(( -HD['HSaa'][P,U] * HD['Daa'][Ds,Dw] ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,U,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,P,U,actfloor:actceil], HD['Daaaa'][Ds,:,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][s,U,P,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][w,P,U,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -HD['HSaaaa'][w,P,U,s] ))
    if P==U:
        result += 2 * fsum(( -HD['HSaa'][Q,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             -HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][w,Q,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,Q,T,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,T,Q,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][s,T,Q,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                             -HD['HSaaaa'][v,Q,T,s] * HD['Daa'][Dr,Dw] ,
                             -HD['HSaaaa'][w,Q,T,r] * HD['Daa'][Ds,Dv] ,
                             +HD['HSaaaa'][v,Q,T,r] * HD['Daa'][Ds,Dw] ,
                             +HD['HSaaaa'][w,Q,T,s] * HD['Daa'][Dr,Dv] ))
        if s==v:
            result += 2 * fsum(( +HD['HSaa'][Q,T] * HD['Daa'][Dr,Dw] ,
                                 +HD['Daa'][Dr,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,T,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,T,Q,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][w,Q,T,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +HD['HSaaaa'][w,Q,T,r] ))
            if r==w:
                result += 2 * fsum(( -HD['HSaa'][Q,T] ,
                                     -einsum( 'II->', HD['HSaaaa'][Q,:actfloor,T,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,T,actfloor:actceil], HD['Daa'][:,:] ) ))
        if r==v:
            result += 2 * fsum(( -HD['HSaa'][Q,T] * HD['Daa'][Ds,Dw] ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,T,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,T,actfloor:actceil], HD['Daaaa'][Ds,:,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][s,T,Q,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][w,Q,T,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -HD['HSaaaa'][w,Q,T,s] ))
            if s==w:
                result += 2 * fsum(( +HD['HSaa'][Q,T] ,
                                     +einsum( 'II->', HD['HSaaaa'][Q,:actfloor,T,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,T,actfloor:actceil], HD['Daa'][:,:] ) ))
        if s==w:
            result += 2 * fsum(( -HD['HSaa'][Q,T] * HD['Daa'][Dr,Dv] ,
                                 -HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,T,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,T,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][v,Q,T,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][r,T,Q,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 -HD['HSaaaa'][v,Q,T,r] ))
        if r==w:
            result += 2 * fsum(( +HD['HSaa'][Q,T] * HD['Daa'][Ds,Dv] ,
                                 +HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,T,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,T,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][v,Q,T,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][s,T,Q,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 +HD['HSaaaa'][v,Q,T,s] ))
    if P==T:
        result += 2 * fsum(( +HD['HSaa'][Q,U] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,U,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,U,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][w,Q,U,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,Q,U,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,U,Q,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,U,Q,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                             +HD['HSaaaa'][v,Q,U,s] * HD['Daa'][Dr,Dw] ,
                             +HD['HSaaaa'][w,Q,U,r] * HD['Daa'][Ds,Dv] ,
                             -HD['HSaaaa'][v,Q,U,r] * HD['Daa'][Ds,Dw] ,
                             -HD['HSaaaa'][w,Q,U,s] * HD['Daa'][Dr,Dv] ))
        if r==v:
            result += 2 * fsum(( +HD['HSaa'][Q,U] * HD['Daa'][Ds,Dw] ,
                                 +HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,U,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,U,actfloor:actceil], HD['Daaaa'][Ds,:,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][w,Q,U,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][s,U,Q,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 +HD['HSaaaa'][w,Q,U,s] ))
            if s==w:
                result += 2 * fsum(( -HD['HSaa'][Q,U] ,
                                     -einsum( 'II->', HD['HSaaaa'][Q,:actfloor,U,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,U,actfloor:actceil], HD['Daa'][:,:] ) ))
        if s==v:
            result += 2 * fsum(( -HD['HSaa'][Q,U] * HD['Daa'][Dr,Dw] ,
                                 -HD['Daa'][Dr,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,U,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,U,actfloor:actceil], HD['Daaaa'][Dr,:,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][r,U,Q,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][w,Q,U,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -HD['HSaaaa'][w,Q,U,r] ))
            if r==w:
                result += 2 * fsum(( +HD['HSaa'][Q,U] ,
                                     +einsum( 'II->', HD['HSaaaa'][Q,:actfloor,U,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,U,actfloor:actceil], HD['Daa'][:,:] ) ))
        if s==w:
            result += 2 * fsum(( +HD['HSaa'][Q,U] * HD['Daa'][Dr,Dv] ,
                                 +HD['Daa'][Dr,Dv] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,U,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,U,actfloor:actceil], HD['Daaaa'][Dr,:,Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][v,Q,U,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,U,Q,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 +HD['HSaaaa'][v,Q,U,r] ))
        if r==w:
            result += 2 * fsum(( -HD['HSaa'][Q,U] * HD['Daa'][Ds,Dv] ,
                                 -HD['Daa'][Ds,Dv] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,U,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,Q,U,actfloor:actceil], HD['Daaaa'][Ds,:,Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][v,Q,U,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][s,U,Q,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 -HD['HSaaaa'][v,Q,U,s] ))
    if r==w:
        result += 2 * +HD['HSaaaa'][P,Q,T,U] * HD['Daa'][Ds,Dv]
        if s==v:
            result += 2 * -HD['HSaaaa'][P,Q,T,U]
    if s==v:
        result += 2 * +HD['HSaaaa'][P,Q,T,U] * HD['Daa'][Dr,Dw]
    if s==w:
        result += 2 * -HD['HSaaaa'][P,Q,T,U] * HD['Daa'][Dr,Dv]
        if r==v:
            result += 2 * +HD['HSaaaa'][P,Q,T,U]
    if r==v:
        result += 2 * -HD['HSaaaa'][P,Q,T,U] * HD['Daa'][Ds,Dw]
    return result


def lucc_CCAA_VAAA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,u,v,w) = e
    Dr, Ds, Du, Dv, Dw = r-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                         -HD['HSaaaa'][P,Q,u,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                         +HD['HSaaaa'][P,Q,s,T] * HD['Daaaa'][Dr,Du,Dv,Dw] ,
                         -HD['HSaaaa'][P,Q,r,T] * HD['Daaaa'][Ds,Du,Dv,Dw] ))
    return result


def lucc_CCAA_VACA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,u,V,w) = e
    Dr, Ds, Du, Dw = r-actfloor, s-actfloor, u-actfloor, w-actfloor
    result = 0
    if Q==V:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][T,P,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Du,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,P,s,actfloor:actceil], HD['Daaaa'][Dr,Dw,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][T,P,r,actfloor:actceil], HD['Daaaa'][Ds,Dw,Du,:] ) ,
                             +HD['HSaaaa'][T,P,r,s] * HD['Daa'][Du,Dw] ))
        if r==u:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][T,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dw,:,:] )
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,P,s,actfloor:actceil], HD['Daa'][Dw,:] )
        if s==u:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][T,P,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dw,:,:] )
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,P,r,actfloor:actceil], HD['Daa'][Dw,:] )
    if P==V:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][T,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Du,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,Q,s,actfloor:actceil], HD['Daaaa'][Dr,Dw,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,Q,r,actfloor:actceil], HD['Daaaa'][Ds,Dw,Du,:] ) ,
                             -HD['HSaaaa'][T,Q,r,s] * HD['Daa'][Du,Dw] ))
        if r==u:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][T,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dw,:,:] )
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,Q,s,actfloor:actceil], HD['Daa'][Dw,:] )
        if s==u:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][T,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dr,Dw,:,:] )
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,Q,r,actfloor:actceil], HD['Daa'][Dw,:] )
    return result


def lucc_CCAA_VACC(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,u,V,W) = e
    Dr, Ds, Du = r-actfloor, s-actfloor, u-actfloor
    result = 0
    if Q==V:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,P,W,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ,
                             -HD['HSaaaa'][T,P,W,r] * HD['Daa'][Ds,Du] ,
                             +HD['HSaaaa'][T,P,W,s] * HD['Daa'][Dr,Du] ))
        if P==W:
            result += 2 * fsum(( +HD['HSaa'][s,T] * HD['Daa'][Dr,Du] ,
                                 -HD['HSaa'][r,T] * HD['Daa'][Ds,Du] ,
                                 +HD['Daa'][Ds,Du] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                                 -HD['Daa'][Dr,Du] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,T] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Du,:,Dr,:] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Du,:,Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][T,u,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][T,u,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][r,s,actfloor:actceil,T], HD['Daa'][Du,:] ) ,
                                 -HD['HSaaaa'][r,s,u,T] ))
            if r==u:
                result += 2 * fsum(( -HD['HSaa'][s,T] ,
                                     +einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,T] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
            if s==u:
                result += 2 * fsum(( +HD['HSaa'][r,T] ,
                                     -einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
        if r==u:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,P,W,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -HD['HSaaaa'][T,P,W,s] ))
        if s==u:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,P,W,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +HD['HSaaaa'][T,P,W,r] ))
    if P==W:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,Q,V,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ,
                             +HD['HSaaaa'][T,Q,V,s] * HD['Daa'][Dr,Du] ,
                             -HD['HSaaaa'][T,Q,V,r] * HD['Daa'][Ds,Du] ))
        if s==u:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,Q,V,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +HD['HSaaaa'][T,Q,V,r] ))
        if r==u:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,Q,V,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -HD['HSaaaa'][T,Q,V,s] ))
    if Q==W:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,P,V,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ,
                             +HD['HSaaaa'][T,P,V,r] * HD['Daa'][Ds,Du] ,
                             -HD['HSaaaa'][T,P,V,s] * HD['Daa'][Dr,Du] ))
        if P==V:
            result += 2 * fsum(( -HD['HSaa'][s,T] * HD['Daa'][Dr,Du] ,
                                 +HD['HSaa'][r,T] * HD['Daa'][Ds,Du] ,
                                 -HD['Daa'][Ds,Du] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                                 +HD['Daa'][Dr,Du] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,T] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Du,:,Dr,:] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Du,:,Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][T,u,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][T,u,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,s,actfloor:actceil,T], HD['Daa'][Du,:] ) ,
                                 +HD['HSaaaa'][r,s,u,T] ))
            if r==u:
                result += 2 * fsum(( +HD['HSaa'][s,T] ,
                                     -einsum( 'II->', HD['HSaaaa'][s,:actfloor,:actfloor,T] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
            if s==u:
                result += 2 * fsum(( -HD['HSaa'][r,T] ,
                                     +einsum( 'II->', HD['HSaaaa'][r,:actfloor,:actfloor,T] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
        if r==u:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,P,V,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +HD['HSaaaa'][T,P,V,s] ))
        if s==u:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,P,V,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -HD['HSaaaa'][T,P,V,r] ))
    if P==V:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,Q,W,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] ) ,
                             +HD['HSaaaa'][T,Q,W,r] * HD['Daa'][Ds,Du] ,
                             -HD['HSaaaa'][T,Q,W,s] * HD['Daa'][Dr,Du] ))
        if s==u:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,Q,W,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -HD['HSaaaa'][T,Q,W,r] ))
        if r==u:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,Q,W,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +HD['HSaaaa'][T,Q,W,s] ))
    return result


def lucc_CCAA_VVAA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,U,v,w) = e
    Dr, Ds, Dv, Dw = r-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * +HD['HSaaaa'][P,Q,T,U] * HD['Daaaa'][Dr,Ds,Dv,Dw]
    return result


def lucc_CCAA_VVCA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,U,V,w) = e
    Dr, Ds, Dw = r-actfloor, s-actfloor, w-actfloor
    result = 0
    return result


def lucc_CCAA_VVCC(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,U,V,W) = e
    Dr, Ds = r-actfloor, s-actfloor
    result = 0
    if Q==V:
        if P==W:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +HD['HSaaaa'][r,s,T,U] ))
    if Q==W:
        if P==V:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][T,U,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -HD['HSaaaa'][r,s,T,U] ))
    return result


def lucc_VAAA_AAAA(actfloor, actceil, HD, *e):
    (P,q,r,s),(t,u,v,w) = e
    Dq, Dr, Ds, Dt, Du, Dv, Dw = q-actfloor, r-actfloor, s-actfloor, t-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += fsum(( -einsum( 'ij,ij->', HD['HSaaaa'][P,t,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dq,Du,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][P,u,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dq,Dt,:,:] ) ,
                     -einsum( 'ij,ij->', HD['HSaaaa'][P,w,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dt,Du,Dq,Dv,:,:] ) ,
                     +einsum( 'ij,ij->', HD['HSaaaa'][P,v,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dt,Du,Dq,Dw,:,:] ) ))
    result += 2 * fsum(( +HD['HSaa'][P,t] * HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] ,
                         -HD['HSaa'][P,u] * HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Dt] ,
                         +HD['HSaa'][P,w] * HD['Daaaaaa'][Dq,Dt,Du,Dr,Ds,Dv] ,
                         -HD['HSaa'][P,v] * HD['Daaaaaa'][Dq,Dt,Du,Dr,Ds,Dw] ,
                         +HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Dt] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,u] ) ,
                         -HD['Daaaaaa'][Dq,Dt,Du,Dr,Ds,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                         +HD['Daaaaaa'][Dq,Dt,Du,Dr,Ds,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                         -HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,t] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,t,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dq,Dv,Dw,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,u,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dt,:,Dq,Dv,Dw,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,w,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,:,Dq,Dt,Du,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dw,:,Dq,Dt,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,q,t,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,q,u,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dt,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,q,w,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dt,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,q,v,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dt,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][t,u,actfloor:actceil,P], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][v,w,actfloor:actceil,P], HD['Daaaaaa'][Dq,Dt,Du,Dr,Ds,:] ) ,
                         -HD['HSaaaa'][P,q,t,u] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                         +HD['HSaaaa'][P,q,v,w] * HD['Daaaa'][Dr,Ds,Dt,Du] ))
    if s==v:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Dt,Du,:,Dq,Dw,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dw,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,w,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,Dw,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,Dw,:] ) ))
        if r==w:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dq,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,Du,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ))
    if r==w:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Ds,Dt,Du,:,Dq,Dv,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,v,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dv,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dv,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dv,:] ) ))
    if s==w:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Dt,Du,:,Dq,Dv,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,v,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dv,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,Dv,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dq,Dv,:] ) ))
        if r==v:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dq,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,Du,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ))
    if r==v:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Ds,Dt,Du,:,Dq,Dw,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,w,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dw,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dw,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dw,:] ) ))
    if s==u:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Dv,Dw,:,Dq,Dt,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dt,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,t,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Dt,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Dt,:] ) ))
        if r==t:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ))
    if r==t:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Ds,Dv,Dw,:,Dq,Du,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,u,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Du,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ))
    if r==u:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Ds,Dv,Dw,:,Dq,Dt,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,t,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dt,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Dt,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Dt,:] ) ))
        if s==t:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ))
    if s==t:
        result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Dv,Dw,:,Dq,Du,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Du,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,u,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Du,:] ) ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,Du,:] ) ))
    if q==u:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dt,:,Dv,Dw,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dt,Dv,Dw,:] ) ,
                             -HD['HSaa'][P,t] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dt,Dv,Dw,:] ) ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,t] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,t,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
    if q==t:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dv,Dw,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             +HD['HSaa'][P,u] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             -HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,u] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,u,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
    if q==v:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dw,:,Dt,Du,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dt,Du,:] ) ,
                             -HD['HSaa'][P,w] * HD['Daaaa'][Dr,Ds,Dt,Du] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dt,Du,:] ) ,
                             +HD['Daaaa'][Dr,Ds,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,w,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dt,Du,:] ) ))
    if q==w:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,:,Dt,Du,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dt,Du,:] ) ,
                             +HD['HSaa'][P,v] * HD['Daaaa'][Dr,Ds,Dt,Du] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dt,Du,:] ) ,
                             -HD['Daaaa'][Dr,Ds,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dt,Du,:] ) ))
    return result


def lucc_VAAA_CAAA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,u,v,w) = e
    Dq, Dr, Ds, Du, Dv, Dw = q-actfloor, r-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dq,Du,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] ,
                         +HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dq,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,q,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,w,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dq,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,v,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dq,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,w,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dq,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dq,Du,:] ) ,
                         +HD['HSaaaa'][P,T,v,w] * HD['Daaaa'][Dq,Du,Dr,Ds] ))
    if s==v:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dw,Dr,Du] ,
                             -HD['Daaaa'][Dq,Dw,Dr,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dq,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,q,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,w,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dq,:] ) ))
        if r==w:
            result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Dq,Du] ,
                                 +HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dq,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][P,q,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if r==w:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dv,Ds,Du] ,
                             -HD['Daaaa'][Dq,Dv,Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dq,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,q,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,v,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dq,:] ) ))
    if s==w:
        result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dv,Dr,Du] ,
                             +HD['Daaaa'][Dq,Dv,Dr,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Dr,Du,:,Dq,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,q,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,v,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dq,:] ) ))
        if r==v:
            result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daa'][Dq,Du] ,
                                 -HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaa'][Du,:,Dq,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][P,q,T,actfloor:actceil], HD['Daa'][Du,:] ) ))
    if r==v:
        result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dw,Ds,Du] ,
                             +HD['Daaaa'][Dq,Dw,Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,T,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,T,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dq,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,q,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,w,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dq,:] ) ))
    if q==v:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Du,:,:] )
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,T,w,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] )
    if q==w:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Du,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daaaa'][Dr,Ds,Du,:] )
    return result


def lucc_VAAA_CCAA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,U,v,w) = e
    Dq, Dr, Ds, Dv, Dw = q-actfloor, r-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,:] ) ,
                         +HD['HSaaaa'][P,q,T,U] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                         +HD['HSaaaa'][P,w,T,U] * HD['Daaaa'][Dq,Dv,Dr,Ds] ,
                         -HD['HSaaaa'][P,v,T,U] * HD['Daaaa'][Dq,Dw,Dr,Ds] ))
    if s==v:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daaaa'][Dq,Dw,Dr,:] ) ,
                             +HD['HSaaaa'][P,w,T,U] * HD['Daa'][Dq,Dr] ,
                             -HD['HSaaaa'][P,q,T,U] * HD['Daa'][Dr,Dw] ))
        if r==w:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                                 +HD['HSaaaa'][P,q,T,U] ))
    if r==w:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daaaa'][Dq,Dv,Ds,:] ) ,
                             +HD['HSaaaa'][P,v,T,U] * HD['Daa'][Dq,Ds] ,
                             -HD['HSaaaa'][P,q,T,U] * HD['Daa'][Ds,Dv] ))
    if s==w:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daaaa'][Dq,Dv,Dr,:] ) ,
                             +HD['HSaaaa'][P,q,T,U] * HD['Daa'][Dr,Dv] ,
                             -HD['HSaaaa'][P,v,T,U] * HD['Daa'][Dq,Dr] ))
        if r==v:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                                 -HD['HSaaaa'][P,q,T,U] ))
    if r==v:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daaaa'][Dq,Dw,Ds,:] ) ,
                             -HD['HSaaaa'][P,w,T,U] * HD['Daa'][Dq,Ds] ,
                             +HD['HSaaaa'][P,q,T,U] * HD['Daa'][Ds,Dw] ))
    return result


def lucc_VAAA_VAAA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,u,v,w) = e
    Dq, Dr, Ds, Du, Dv, Dw = q-actfloor, r-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Dv,Dw,Dq,Du,:,:] )
    result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] ,
                         -HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dq,Dv,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,q,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][T,u,actfloor:actceil,P], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,:] ) ,
                         +HD['HSaaaa'][P,q,u,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ))
    if P==T:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dv,Dw,:,:] ) ,
                         -einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dq,Dv,Dw,:,Dr,Du,:,:] ) ,
                         +einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dq,Dv,Dw,:,Ds,Du,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][r,s,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Du,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][s,u,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][r,u,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                             -HD['HSaa'][q,u] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Du,:] ) ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,u] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,u,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
        if q==u:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dr,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Ds,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,s,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                                 -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ))
    if q==u:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
    if r==u:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,:,:] )
    if s==u:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dq,:,:] )
    return result


def lucc_VAAA_VACA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,u,V,w) = e
    Dq, Dr, Ds, Du, Dw = q-actfloor, r-actfloor, s-actfloor, u-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,T,V,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dq,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,Du,Dq,Dw,:] ) ,
                         +HD['HSaaaa'][P,V,u,T] * HD['Daaaa'][Dq,Dw,Dr,Ds] ))
    if P==T:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][q,V,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dw,:,:] )
        result += 2 * fsum(( -HD['HSaa'][s,V] * HD['Daaaa'][Dq,Dw,Dr,Du] ,
                             +HD['HSaa'][r,V] * HD['Daaaa'][Dq,Dw,Ds,Du] ,
                             +HD['Daaaa'][Dq,Dw,Ds,Du] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,V,:actfloor] ) ,
                             -HD['Daaaa'][Dq,Dw,Dr,Du] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,V,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,V,actfloor:actceil], HD['Daaaaaa'][Dq,Dw,:,Ds,Du,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,V,actfloor:actceil], HD['Daaaaaa'][Dq,Dw,:,Dr,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][q,V,u,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][r,s,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][s,u,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][r,u,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Ds,:] ) ))
        if q==u:
            result += 2 * fsum(( +HD['HSaa'][s,V] * HD['Daa'][Dr,Dw] ,
                                 -HD['HSaa'][r,V] * HD['Daa'][Ds,Dw] ,
                                 +HD['Daa'][Dr,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,V,:actfloor] ) ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,V,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,V,actfloor:actceil], HD['Daaaa'][Dw,:,Dr,:] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,V,actfloor:actceil], HD['Daaaa'][Dw,:,Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,s,V,actfloor:actceil], HD['Daa'][Dw,:] ) ))
    if q==u:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dw,:] )
    if r==u:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,T,V,actfloor:actceil], HD['Daaaa'][Ds,Dw,Dq,:] )
    if s==u:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,T,V,actfloor:actceil], HD['Daaaa'][Dr,Dw,Dq,:] )
    return result


def lucc_VAAA_VACC(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,u,V,W) = e
    Dq, Dr, Ds, Du = q-actfloor, r-actfloor, s-actfloor, u-actfloor
    result = 0
    result += 2 * +HD['HSaaaa'][P,T,V,W] * HD['Daaaa'][Dq,Du,Dr,Ds]
    if P==T:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][V,W,s,actfloor:actceil], HD['Daaaa'][Dr,Du,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][V,W,r,actfloor:actceil], HD['Daaaa'][Ds,Du,Dq,:] ) ,
                             +HD['HSaaaa'][r,s,V,W] * HD['Daa'][Dq,Du] ,
                             +HD['HSaaaa'][s,u,V,W] * HD['Daa'][Dq,Dr] ,
                             -HD['HSaaaa'][r,u,V,W] * HD['Daa'][Dq,Ds] ))
        if q==u:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][V,W,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][V,W,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -HD['HSaaaa'][r,s,V,W] ))
    if r==u:
        result += 2 * -HD['HSaaaa'][P,T,V,W] * HD['Daa'][Dq,Ds]
    if s==u:
        result += 2 * +HD['HSaaaa'][P,T,V,W] * HD['Daa'][Dq,Dr]
    return result


def lucc_VAAA_VVAA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,U,v,w) = e
    Dq, Dr, Ds, Dv, Dw = q-actfloor, r-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daaaaaa'][Dq,Dv,Dw,Dr,Ds,:] ) ,
                         -HD['HSaaaa'][P,q,T,U] * HD['Daaaa'][Dr,Ds,Dv,Dw] ))
    if P==U:
        result += fsum(( -einsum( 'ij,ij->', HD['HSaaaa'][T,s,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][T,r,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,:,:] ) ))
        result += 2 * fsum(( +HD['HSaa'][q,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             -HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,T] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
    if P==T:
        result += fsum(( +einsum( 'ij,ij->', HD['HSaaaa'][U,s,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Dr,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][U,r,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,:,:] ) ))
        result += 2 * fsum(( -HD['HSaa'][q,U] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,U] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,U], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
    return result


def lucc_VAAA_VVCA(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,U,V,w) = e
    Dq, Dr, Ds, Dw = q-actfloor, r-actfloor, s-actfloor, w-actfloor
    result = 0
    result += 2 * -HD['HSaaaa'][P,V,T,U] * HD['Daaaa'][Dq,Dw,Dr,Ds]
    if P==U:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][q,V,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][T,s,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Dr,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,r,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Ds,:] ) ))
    if P==T:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][q,V,actfloor:actceil,U], HD['Daaaa'][Dr,Ds,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][U,s,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Dr,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][U,r,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Ds,:] ) ))
    return result


def lucc_VAAA_VVCC(actfloor, actceil, HD, *e):
    (P,q,r,s),(T,U,V,W) = e
    Dq, Dr, Ds = q-actfloor, r-actfloor, s-actfloor
    result = 0
    if P==T:
        result += 2 * fsum(( -HD['HSaaaa'][U,s,V,W] * HD['Daa'][Dq,Dr] ,
                             +HD['HSaaaa'][U,r,V,W] * HD['Daa'][Dq,Ds] ))
    if P==U:
        result += 2 * fsum(( +HD['HSaaaa'][T,s,V,W] * HD['Daa'][Dq,Dr] ,
                             -HD['HSaaaa'][T,r,V,W] * HD['Daa'][Dq,Ds] ))
    return result


def lucc_VACA_AAAA(actfloor, actceil, HD, *e):
    (P,q,R,s),(t,u,v,w) = e
    Dq, Ds, Dt, Du, Dv, Dw = q-actfloor, s-actfloor, t-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][w,R,actfloor:actceil,P], HD['Daaaaaa'][Dq,Dt,Du,Ds,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][v,R,actfloor:actceil,P], HD['Daaaaaa'][Dq,Dt,Du,Ds,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,t,R,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,u,R,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Dt,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][t,R,actfloor:actceil,P], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][u,R,actfloor:actceil,P], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Dt,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,w,R,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dv,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,v,R,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dq,Dw,:] ) ,
                         -HD['HSaaaa'][P,q,R,t] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                         +HD['HSaaaa'][P,q,R,u] * HD['Daaaa'][Ds,Dt,Dv,Dw] ,
                         -HD['HSaaaa'][P,q,R,w] * HD['Daaaa'][Ds,Dv,Dt,Du] ,
                         +HD['HSaaaa'][P,q,R,v] * HD['Daaaa'][Ds,Dw,Dt,Du] ))
    if s==w:
        result += 2 * fsum(( -HD['HSaa'][P,R] * HD['Daaaa'][Dq,Dv,Dt,Du] ,
                             -HD['Daaaa'][Dq,Dv,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dq,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,q,R,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,v,R,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ))
    if s==v:
        result += 2 * fsum(( +HD['HSaa'][P,R] * HD['Daaaa'][Dq,Dw,Dt,Du] ,
                             +HD['Daaaa'][Dq,Dw,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Dt,Du,:,Dq,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,q,R,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,w,R,actfloor:actceil], HD['Daaaa'][Dt,Du,Dq,:] ) ))
    if s==t:
        result += 2 * fsum(( -HD['HSaa'][P,R] * HD['Daaaa'][Dq,Du,Dv,Dw] ,
                             -HD['Daaaa'][Dq,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,u,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,q,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ))
    if s==u:
        result += 2 * fsum(( +HD['HSaa'][P,R] * HD['Daaaa'][Dq,Dt,Dv,Dw] ,
                             +HD['Daaaa'][Dq,Dt,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,Dt,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,t,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,q,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] ) ))
    if q==t:
        result += 2 * fsum(( -HD['HSaa'][P,R] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                             -HD['Daaaa'][Ds,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dv,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][u,R,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
    if q==u:
        result += 2 * fsum(( +HD['HSaa'][P,R] * HD['Daaaa'][Ds,Dt,Dv,Dw] ,
                             +HD['Daaaa'][Ds,Dt,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,:,Dv,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][t,R,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
    if q==w:
        result += 2 * fsum(( -HD['HSaa'][P,R] * HD['Daaaa'][Ds,Dv,Dt,Du] ,
                             -HD['Daaaa'][Ds,Dv,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,:,Dt,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,R,actfloor:actceil,P], HD['Daaaa'][Dt,Du,Ds,:] ) ))
    if q==v:
        result += 2 * fsum(( +HD['HSaa'][P,R] * HD['Daaaa'][Ds,Dw,Dt,Du] ,
                             +HD['Daaaa'][Ds,Dw,Dt,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Ds,Dw,:,Dt,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][w,R,actfloor:actceil,P], HD['Daaaa'][Dt,Du,Ds,:] ) ))
    return result


def lucc_VACA_CAAA(actfloor, actceil, HD, *e):
    (P,q,R,s),(T,u,v,w) = e
    Dq, Ds, Du, Dv, Dw = q-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][R,T,actfloor:actceil,P], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                         +HD['HSaaaa'][P,T,R,v] * HD['Daaaa'][Dq,Du,Ds,Dw] ,
                         -HD['HSaaaa'][P,q,R,T] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                         -HD['HSaaaa'][P,w,R,T] * HD['Daaaa'][Dq,Dv,Ds,Du] ,
                         +HD['HSaaaa'][P,v,R,T] * HD['Daaaa'][Dq,Dw,Ds,Du] ,
                         -HD['HSaaaa'][P,T,R,w] * HD['Daaaa'][Dq,Du,Ds,Dv] ))
    if R==T:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Ds,Dv,Dw,:,Dq,Du,:,:] ) ,
                         +einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Du,:,:] ) ))
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ,
                             -HD['HSaa'][P,w] * HD['Daaaa'][Dq,Du,Ds,Dv] ,
                             +HD['HSaa'][P,v] * HD['Daaaa'][Dq,Du,Ds,Dw] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ,
                             +HD['Daaaa'][Dq,Du,Ds,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                             -HD['Daaaa'][Dq,Du,Ds,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daaaaaa'][Ds,Dw,:,Dq,Du,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,w,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,:,Dq,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,q,v,actfloor:actceil], HD['Daaaa'][Ds,Dw,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,w,actfloor:actceil,P], HD['Daaaa'][Dq,Du,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,q,w,actfloor:actceil], HD['Daaaa'][Ds,Dv,Du,:] ) ,
                             -HD['HSaaaa'][P,q,v,w] * HD['Daa'][Ds,Du] ))
        if q==w:
            result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,:,Du,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Ds,Dv,Du,:] ) ,
                                 -HD['HSaa'][P,v] * HD['Daa'][Ds,Du] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Dv,Du,:] ) ,
                                 +HD['Daa'][Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Ds,:,Du,:] ) ))
        if q==v:
            result += +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dw,:,Du,:,:] )
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Ds,Dw,Du,:] ) ,
                                 +HD['HSaa'][P,w] * HD['Daa'][Ds,Du] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Dw,Du,:] ) ,
                                 -HD['Daa'][Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,w,actfloor:actceil], HD['Daaaa'][Ds,:,Du,:] ) ))
    if q==v:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daaaa'][Ds,Dw,Du,:] ) ,
                             +HD['HSaaaa'][P,T,R,w] * HD['Daa'][Ds,Du] ))
    if q==w:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daaaa'][Ds,Dv,Du,:] ) ,
                             -HD['HSaaaa'][P,T,R,v] * HD['Daa'][Ds,Du] ))
    if s==w:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][R,T,actfloor:actceil,P], HD['Daaaa'][Dq,Dv,Du,:] ) ,
                             -HD['HSaaaa'][P,v,R,T] * HD['Daa'][Dq,Du] ,
                             +HD['HSaaaa'][P,q,R,T] * HD['Daa'][Du,Dv] ))
    if s==v:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][R,T,actfloor:actceil,P], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                             -HD['HSaaaa'][P,q,R,T] * HD['Daa'][Du,Dw] ,
                             +HD['HSaaaa'][P,w,R,T] * HD['Daa'][Dq,Du] ))
    return result


def lucc_VACA_CCAA(actfloor, actceil, HD, *e):
    (P,q,R,s),(T,U,v,w) = e
    Dq, Ds, Dv, Dw = q-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    if R==T:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,U,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,U,v,actfloor:actceil], HD['Daaaa'][Ds,Dw,Dq,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,U,w,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                             -HD['HSaaaa'][P,U,v,w] * HD['Daa'][Dq,Ds] ))
        if q==w:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][P,U,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dv,:,:] )
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,U,v,actfloor:actceil], HD['Daa'][Ds,:] )
        if q==v:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][P,U,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dw,:,:] )
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,U,w,actfloor:actceil], HD['Daa'][Ds,:] )
    if R==U:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daaaa'][Ds,Dw,Dq,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,T,w,actfloor:actceil], HD['Daaaa'][Ds,Dv,Dq,:] ) ,
                             +HD['HSaaaa'][P,T,v,w] * HD['Daa'][Dq,Ds] ))
        if q==w:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dv,:,:] )
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,T,v,actfloor:actceil], HD['Daa'][Ds,:] )
        if q==v:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][P,T,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Dw,:,:] )
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,T,w,actfloor:actceil], HD['Daa'][Ds,:] )
    return result


def lucc_VACA_VAAA(actfloor, actceil, HD, *e):
    (P,q,R,s),(T,u,v,w) = e
    Dq, Ds, Du, Dv, Dw = q-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,P], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dq,Du,:] ) ,
                         -HD['HSaaaa'][P,q,R,T] * HD['Daaaa'][Ds,Du,Dv,Dw] ))
    if P==T:
        result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dq,Dv,Dw,:,Ds,Du,:,:] ) ,
                         -einsum( 'ij,ij->', HD['HSaaaa'][u,R,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,:,:] ) ))
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                             +HD['HSaa'][q,R] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,Du,:] ) ,
                             +HD['Daaaa'][Ds,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,R,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dv,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][u,R,q,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
        if q==u:
            result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Ds,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
    if q==u:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Ds,:] )
    if s==u:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] )
    return result


def lucc_VACA_VACA(actfloor, actceil, HD, *e):
    (P,q,R,s),(T,u,V,w) = e
    Dq, Ds, Du, Dw = q-actfloor, s-actfloor, u-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -HD['HSaaaa'][P,T,R,V] * HD['Daaaa'][Dq,Du,Ds,Dw] ,
                         -HD['HSaaaa'][P,V,R,T] * HD['Daaaa'][Dq,Dw,Ds,Du] ))
    if R==V:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dq,Dw,Ds,Du] ,
                             +HD['Daaaa'][Dq,Dw,Ds,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Ds,Du,:,Dq,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,q,actfloor:actceil,T], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][T,u,actfloor:actceil,P], HD['Daaaa'][Dq,Dw,Ds,:] ) ,
                             -HD['HSaaaa'][P,q,u,T] * HD['Daa'][Ds,Dw] ))
        if P==T:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dw,:,Du,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dw,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][s,u,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                                 +HD['HSaa'][q,u] * HD['Daa'][Ds,Dw] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,u] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,u,actfloor:actceil], HD['Daaaa'][Ds,:,Dw,:] ) ))
            if q==u:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ))
        if q==u:
            result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Ds,Dw] ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Ds,:,Dw,:] ) ))
    if P==T:
        result += 2 * fsum(( +HD['HSaa'][R,V] * HD['Daaaa'][Dq,Dw,Ds,Du] ,
                             +HD['Daaaa'][Dq,Dw,Ds,Du] * einsum( 'II->', HD['HSaaaa'][R,:actfloor,V,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,V,actfloor:actceil], HD['Daaaaaa'][Dq,Dw,:,Ds,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][u,R,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Ds,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][q,V,R,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                             +HD['HSaaaa'][q,V,R,u] * HD['Daa'][Ds,Dw] ))
        if q==u:
            result += 2 * fsum(( -HD['HSaa'][R,V] * HD['Daa'][Ds,Dw] ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][R,:actfloor,V,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,V,actfloor:actceil], HD['Daaaa'][Dw,:,Ds,:] ) ))
    if q==u:
        result += 2 * +HD['HSaaaa'][P,V,R,T] * HD['Daa'][Ds,Dw]
    if s==u:
        result += 2 * +HD['HSaaaa'][P,T,R,V] * HD['Daa'][Dq,Dw]
    return result


def lucc_VACA_VACC(actfloor, actceil, HD, *e):
    (P,q,R,s),(T,u,V,W) = e
    Dq, Ds, Du = q-actfloor, s-actfloor, u-actfloor
    result = 0
    if P==T:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][V,W,R,actfloor:actceil], HD['Daaaa'][Ds,Du,Dq,:] ) ,
                             +HD['HSaaaa'][u,R,V,W] * HD['Daa'][Dq,Ds] ))
        if R==W:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][q,V,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] )
            result += 2 * fsum(( -HD['HSaa'][s,V] * HD['Daa'][Dq,Du] ,
                                 -HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,V,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,V,actfloor:actceil], HD['Daaaa'][Dq,:,Du,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][q,V,u,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][s,u,V,actfloor:actceil], HD['Daa'][Dq,:] ) ))
            if q==u:
                result += 2 * fsum(( +HD['HSaa'][s,V] ,
                                     +einsum( 'II->', HD['HSaaaa'][s,:actfloor,V,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,V,actfloor:actceil], HD['Daa'][:,:] ) ))
        if R==V:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][q,W,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] )
            result += 2 * fsum(( +HD['HSaa'][s,W] * HD['Daa'][Dq,Du] ,
                                 +HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,W,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,W,actfloor:actceil], HD['Daaaa'][Dq,:,Du,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][q,W,u,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][s,u,W,actfloor:actceil], HD['Daa'][Dq,:] ) ))
            if q==u:
                result += 2 * fsum(( -HD['HSaa'][s,W] ,
                                     -einsum( 'II->', HD['HSaaaa'][s,:actfloor,W,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,W,actfloor:actceil], HD['Daa'][:,:] ) ))
        if q==u:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][V,W,R,actfloor:actceil], HD['Daa'][Ds,:] )
    if R==W:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daaaa'][Ds,Du,Dq,:] ) ,
                             +HD['HSaaaa'][P,V,u,T] * HD['Daa'][Dq,Ds] ))
        if q==u:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daa'][Ds,:] )
    if R==V:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,W,actfloor:actceil,T], HD['Daaaa'][Ds,Du,Dq,:] ) ,
                             -HD['HSaaaa'][P,W,u,T] * HD['Daa'][Dq,Ds] ))
        if q==u:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,W,actfloor:actceil,T], HD['Daa'][Ds,:] )
    return result


def lucc_VACA_VVAA(actfloor, actceil, HD, *e):
    (P,q,R,s),(T,U,v,w) = e
    Dq, Ds, Dv, Dw = q-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    if P==U:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][T,R,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,R,q,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] )
    if P==T:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][U,R,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Ds,:,:] )
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][U,R,q,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] )
    return result


def lucc_VACA_VVCA(actfloor, actceil, HD, *e):
    (P,q,R,s),(T,U,V,w) = e
    Dq, Ds, Dw = q-actfloor, s-actfloor, w-actfloor
    result = 0
    if R==V:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daaaa'][Dq,Dw,Ds,:] ) ,
                             +HD['HSaaaa'][P,q,T,U] * HD['Daa'][Ds,Dw] ))
        if P==U:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][T,s,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] )
            result += 2 * fsum(( -HD['HSaa'][q,T] * HD['Daa'][Ds,Dw] ,
                                 +HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,T] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Ds,:,Dw,:] ) ))
        if P==T:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][U,s,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] )
            result += 2 * fsum(( +HD['HSaa'][q,U] * HD['Daa'][Ds,Dw] ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,U] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,U], HD['Daaaa'][Ds,:,Dw,:] ) ))
    if P==U:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,R,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Ds,:] ) ,
                             -HD['HSaaaa'][q,V,R,T] * HD['Daa'][Ds,Dw] ))
    if P==T:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][U,R,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Ds,:] ) ,
                             +HD['HSaaaa'][q,V,R,U] * HD['Daa'][Ds,Dw] ))
    return result


def lucc_VACA_VVCC(actfloor, actceil, HD, *e):
    (P,q,R,s),(T,U,V,W) = e
    Dq, Ds = q-actfloor, s-actfloor
    result = 0
    if R==W:
        result += 2 * -HD['HSaaaa'][P,V,T,U] * HD['Daa'][Dq,Ds]
        if P==U:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,s,V,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][q,V,actfloor:actceil,T], HD['Daa'][Ds,:] ) ))
        if P==T:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][q,V,actfloor:actceil,U], HD['Daa'][Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][U,s,V,actfloor:actceil], HD['Daa'][Dq,:] ) ))
    if R==V:
        result += 2 * +HD['HSaaaa'][P,W,T,U] * HD['Daa'][Dq,Ds]
        if P==U:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,s,W,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][q,W,actfloor:actceil,T], HD['Daa'][Ds,:] ) ))
        if P==T:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][q,W,actfloor:actceil,U], HD['Daa'][Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][U,s,W,actfloor:actceil], HD['Daa'][Dq,:] ) ))
    if P==U:
        result += 2 * -HD['HSaaaa'][T,R,V,W] * HD['Daa'][Dq,Ds]
    if P==T:
        result += 2 * +HD['HSaaaa'][U,R,V,W] * HD['Daa'][Dq,Ds]
    return result


def lucc_VACC_AAAA(actfloor, actceil, HD, *e):
    (P,q,R,S),(t,u,v,w) = e
    Dq, Dt, Du, Dv, Dw = q-actfloor, t-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +HD['HSaaaa'][P,w,R,S] * HD['Daaaa'][Dq,Dv,Dt,Du] ,
                         -HD['HSaaaa'][P,v,R,S] * HD['Daaaa'][Dq,Dw,Dt,Du] ,
                         +HD['HSaaaa'][P,t,R,S] * HD['Daaaa'][Dq,Du,Dv,Dw] ,
                         -HD['HSaaaa'][P,u,R,S] * HD['Daaaa'][Dq,Dt,Dv,Dw] ))
    if q==t:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][R,S,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Du,:] )
    if q==u:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][R,S,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Dt,:] )
    if q==w:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][R,S,actfloor:actceil,P], HD['Daaaa'][Dt,Du,Dv,:] )
    if q==v:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][R,S,actfloor:actceil,P], HD['Daaaa'][Dt,Du,Dw,:] )
    return result


def lucc_VACC_CAAA(actfloor, actceil, HD, *e):
    (P,q,R,S),(T,u,v,w) = e
    Dq, Du, Dv, Dw = q-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * +HD['HSaaaa'][P,T,R,S] * HD['Daaaa'][Dq,Du,Dv,Dw]
    if S==T:
        result += 2 * fsum(( -HD['HSaa'][P,R] * HD['Daaaa'][Dq,Du,Dv,Dw] ,
                             -HD['Daaaa'][Dq,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][P,q,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][w,R,actfloor:actceil,P], HD['Daaaa'][Dq,Du,Dv,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][v,R,actfloor:actceil,P], HD['Daaaa'][Dq,Du,Dw,:] ) ,
                             +HD['HSaaaa'][P,q,R,w] * HD['Daa'][Du,Dv] ,
                             -HD['HSaaaa'][P,q,R,v] * HD['Daa'][Du,Dw] ))
        if q==w:
            result += 2 * fsum(( +HD['HSaa'][P,R] * HD['Daa'][Du,Dv] ,
                                 +HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Dv,:,Du,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][v,R,actfloor:actceil,P], HD['Daa'][Du,:] ) ))
        if q==v:
            result += 2 * fsum(( -HD['HSaa'][P,R] * HD['Daa'][Du,Dw] ,
                                 -HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Dw,:,Du,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][w,R,actfloor:actceil,P], HD['Daa'][Du,:] ) ))
    if R==T:
        result += 2 * fsum(( +HD['HSaa'][P,S] * HD['Daaaa'][Dq,Du,Dv,Dw] ,
                             +HD['Daaaa'][Dq,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,S,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,S,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,q,S,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][w,S,actfloor:actceil,P], HD['Daaaa'][Dq,Du,Dv,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][v,S,actfloor:actceil,P], HD['Daaaa'][Dq,Du,Dw,:] ) ,
                             -HD['HSaaaa'][P,q,S,w] * HD['Daa'][Du,Dv] ,
                             +HD['HSaaaa'][P,q,S,v] * HD['Daa'][Du,Dw] ))
        if q==w:
            result += 2 * fsum(( -HD['HSaa'][P,S] * HD['Daa'][Du,Dv] ,
                                 -HD['Daa'][Du,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,S,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,S,actfloor:actceil], HD['Daaaa'][Dv,:,Du,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][v,S,actfloor:actceil,P], HD['Daa'][Du,:] ) ))
        if q==v:
            result += 2 * fsum(( +HD['HSaa'][P,S] * HD['Daa'][Du,Dw] ,
                                 +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,S,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,S,actfloor:actceil], HD['Daaaa'][Dw,:,Du,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][w,S,actfloor:actceil,P], HD['Daa'][Du,:] ) ))
    if q==w:
        result += 2 * -HD['HSaaaa'][P,T,R,S] * HD['Daa'][Du,Dv]
    if q==v:
        result += 2 * +HD['HSaaaa'][P,T,R,S] * HD['Daa'][Du,Dw]
    return result


def lucc_VACC_CCAA(actfloor, actceil, HD, *e):
    (P,q,R,S),(T,U,v,w) = e
    Dq, Dv, Dw = q-actfloor, v-actfloor, w-actfloor
    result = 0
    if S==U:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                             -HD['HSaaaa'][P,T,R,w] * HD['Daa'][Dq,Dv] ,
                             +HD['HSaaaa'][P,T,R,v] * HD['Daa'][Dq,Dw] ))
        if R==T:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                                 -HD['HSaa'][P,w] * HD['Daa'][Dq,Dv] ,
                                 +HD['HSaa'][P,v] * HD['Daa'][Dq,Dw] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                                 -HD['Daa'][Dq,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                                 +HD['Daa'][Dq,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,w,actfloor:actceil], HD['Daaaa'][Dv,:,Dq,:] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dw,:,Dq,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][P,q,w,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][P,q,v,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][v,w,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                                 -HD['HSaaaa'][P,q,v,w] ))
            if q==v:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     +HD['HSaa'][P,w] ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     -einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,w,actfloor:actceil], HD['Daa'][:,:] ) ))
            if q==w:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     -HD['HSaa'][P,v] ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     +einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daa'][:,:] ) ))
        if q==v:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 +HD['HSaaaa'][P,T,R,w] ))
        if q==w:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,T,R,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 -HD['HSaaaa'][P,T,R,v] ))
    if R==T:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,U,S,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                             +HD['HSaaaa'][P,U,S,v] * HD['Daa'][Dq,Dw] ,
                             -HD['HSaaaa'][P,U,S,w] * HD['Daa'][Dq,Dv] ))
        if q==w:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,U,S,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 -HD['HSaaaa'][P,U,S,v] ))
        if q==v:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,U,S,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 +HD['HSaaaa'][P,U,S,w] ))
    if R==U:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,T,S,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                             -HD['HSaaaa'][P,T,S,v] * HD['Daa'][Dq,Dw] ,
                             +HD['HSaaaa'][P,T,S,w] * HD['Daa'][Dq,Dv] ))
        if S==T:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dq,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                                 +HD['HSaa'][P,w] * HD['Daa'][Dq,Dv] ,
                                 -HD['HSaa'][P,v] * HD['Daa'][Dq,Dw] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                                 +HD['Daa'][Dq,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                                 -HD['Daa'][Dq,Dv] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,w,actfloor:actceil], HD['Daaaa'][Dv,:,Dq,:] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daaaa'][Dw,:,Dq,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][P,q,w,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][P,q,v,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][v,w,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                                 +HD['HSaaaa'][P,q,v,w] ))
            if q==v:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     -HD['HSaa'][P,w] ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     +einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,w] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,w,actfloor:actceil], HD['Daa'][:,:] ) ))
            if q==w:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     +HD['HSaa'][P,v] ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                     -einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,v] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,v,actfloor:actceil], HD['Daa'][:,:] ) ))
        if q==v:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,T,S,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 -HD['HSaaaa'][P,T,S,w] ))
        if q==w:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,T,S,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 +HD['HSaaaa'][P,T,S,v] ))
    if S==T:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,U,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dq,:] ) ,
                             -HD['HSaaaa'][P,U,R,v] * HD['Daa'][Dq,Dw] ,
                             +HD['HSaaaa'][P,U,R,w] * HD['Daa'][Dq,Dv] ))
        if q==w:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,U,R,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 +HD['HSaaaa'][P,U,R,v] ))
        if q==v:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,U,R,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 -HD['HSaaaa'][P,U,R,w] ))
    return result


def lucc_VACC_VAAA(actfloor, actceil, HD, *e):
    (P,q,R,S),(T,u,v,w) = e
    Dq, Du, Dv, Dw = q-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * +HD['HSaaaa'][P,T,R,S] * HD['Daaaa'][Dq,Du,Dv,Dw]
    if P==T:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][R,S,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dv,Dw,Du,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][R,S,q,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] )
        if q==u:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][R,S,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    return result


def lucc_VACC_VACA(actfloor, actceil, HD, *e):
    (P,q,R,S),(T,u,V,w) = e
    Dq, Du, Dw = q-actfloor, u-actfloor, w-actfloor
    result = 0
    if P==T:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][R,S,V,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                             +HD['HSaaaa'][q,V,R,S] * HD['Daa'][Du,Dw] ))
        if S==V:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dw,:,Du,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][u,R,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                                 -HD['HSaa'][q,R] * HD['Daa'][Du,Dw] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                                 -HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,R,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][u,R,q,actfloor:actceil], HD['Daa'][Dw,:] ) ))
            if q==u:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ))
        if R==V:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,S,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dq,Dw,:,Du,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][u,S,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][S,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                                 +HD['HSaa'][q,S] * HD['Daa'][Du,Dw] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][S,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                                 +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][q,:actfloor,S,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,S,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][u,S,q,actfloor:actceil], HD['Daa'][Dw,:] ) ))
            if q==u:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,S,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][S,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][S,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ))
        if q==u:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][R,S,V,actfloor:actceil], HD['Daa'][Dw,:] )
    if S==V:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,P], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                             +HD['HSaaaa'][P,q,R,T] * HD['Daa'][Du,Dw] ))
        if q==u:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,P], HD['Daa'][Dw,:] )
    if R==V:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,S,actfloor:actceil,P], HD['Daaaa'][Dq,Dw,Du,:] ) ,
                             -HD['HSaaaa'][P,q,S,T] * HD['Daa'][Du,Dw] ))
        if q==u:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,S,actfloor:actceil,P], HD['Daa'][Dw,:] )
    return result


def lucc_VACC_VACC(actfloor, actceil, HD, *e):
    (P,q,R,S),(T,u,V,W) = e
    Dq, Du = q-actfloor, u-actfloor
    result = 0
    if P==T:
        result += 2 * -HD['HSaaaa'][R,S,V,W] * HD['Daa'][Dq,Du]
        if S==V:
            result += 2 * fsum(( -HD['HSaa'][R,W] * HD['Daa'][Dq,Du] ,
                                 -HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][R,:actfloor,W,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,W,actfloor:actceil], HD['Daaaa'][Dq,:,Du,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][u,R,W,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][q,W,R,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 -HD['HSaaaa'][q,W,R,u] ))
            if R==W:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     -HD['HSaa'][q,u] ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     +einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,u] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,u,actfloor:actceil], HD['Daa'][:,:] ) ))
            if q==u:
                result += 2 * fsum(( +HD['HSaa'][R,W] ,
                                     +einsum( 'II->', HD['HSaaaa'][R,:actfloor,W,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,W,actfloor:actceil], HD['Daa'][:,:] ) ))
        if R==W:
            result += 2 * fsum(( -HD['HSaa'][S,V] * HD['Daa'][Dq,Du] ,
                                 -HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][S,:actfloor,V,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,S,V,actfloor:actceil], HD['Daaaa'][Dq,:,Du,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][q,V,S,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][u,S,V,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -HD['HSaaaa'][q,V,S,u] ))
            if q==u:
                result += 2 * fsum(( +HD['HSaa'][S,V] ,
                                     +einsum( 'II->', HD['HSaaaa'][S,:actfloor,V,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,S,V,actfloor:actceil], HD['Daa'][:,:] ) ))
        if S==W:
            result += 2 * fsum(( +HD['HSaa'][R,V] * HD['Daa'][Dq,Du] ,
                                 +HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][R,:actfloor,V,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,V,actfloor:actceil], HD['Daaaa'][Dq,:,Du,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][u,R,V,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][q,V,R,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 +HD['HSaaaa'][q,V,R,u] ))
            if R==V:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][q,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     +HD['HSaa'][q,u] ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     -einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,u] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,u,actfloor:actceil], HD['Daa'][:,:] ) ))
            if q==u:
                result += 2 * fsum(( -HD['HSaa'][R,V] ,
                                     -einsum( 'II->', HD['HSaaaa'][R,:actfloor,V,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,V,actfloor:actceil], HD['Daa'][:,:] ) ))
        if R==V:
            result += 2 * fsum(( +HD['HSaa'][S,W] * HD['Daa'][Dq,Du] ,
                                 +HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][S,:actfloor,W,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,S,W,actfloor:actceil], HD['Daaaa'][Dq,:,Du,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][q,W,S,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][u,S,W,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +HD['HSaaaa'][q,W,S,u] ))
            if q==u:
                result += 2 * fsum(( -HD['HSaa'][S,W] ,
                                     -einsum( 'II->', HD['HSaaaa'][S,:actfloor,W,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,S,W,actfloor:actceil], HD['Daa'][:,:] ) ))
        if q==u:
            result += 2 * +HD['HSaaaa'][R,S,V,W]
    if S==V:
        result += 2 * +HD['HSaaaa'][P,W,R,T] * HD['Daa'][Dq,Du]
        if R==W:
            result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Dq,Du] ,
                                 -HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Du,:,Dq,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][P,q,actfloor:actceil,T], HD['Daa'][Du,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][T,u,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                                 +HD['HSaaaa'][P,q,u,T] ))
            if q==u:
                result += 2 * fsum(( -HD['HSaa'][P,T] ,
                                     +einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
        if q==u:
            result += 2 * -HD['HSaaaa'][P,W,R,T]
    if R==W:
        result += 2 * +HD['HSaaaa'][P,V,S,T] * HD['Daa'][Dq,Du]
        if q==u:
            result += 2 * -HD['HSaaaa'][P,V,S,T]
    if S==W:
        result += 2 * -HD['HSaaaa'][P,V,R,T] * HD['Daa'][Dq,Du]
        if R==V:
            result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daa'][Dq,Du] ,
                                 +HD['Daa'][Dq,Du] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Du,:,Dq,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][P,q,actfloor:actceil,T], HD['Daa'][Du,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][T,u,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                                 -HD['HSaaaa'][P,q,u,T] ))
            if q==u:
                result += 2 * fsum(( +HD['HSaa'][P,T] ,
                                     -einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
        if q==u:
            result += 2 * +HD['HSaaaa'][P,V,R,T]
    if R==V:
        result += 2 * -HD['HSaaaa'][P,W,S,T] * HD['Daa'][Dq,Du]
        if q==u:
            result += 2 * +HD['HSaaaa'][P,W,S,T]
    return result


def lucc_VACC_VVAA(actfloor, actceil, HD, *e):
    (P,q,R,S),(T,U,v,w) = e
    Dq, Dv, Dw = q-actfloor, v-actfloor, w-actfloor
    result = 0
    return result


def lucc_VACC_VVCA(actfloor, actceil, HD, *e):
    (P,q,R,S),(T,U,V,w) = e
    Dq, Dw = q-actfloor, w-actfloor
    result = 0
    if S==V:
        if P==U:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][T,R,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] )
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,R,q,actfloor:actceil], HD['Daa'][Dw,:] )
        if P==T:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][U,R,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] )
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][U,R,q,actfloor:actceil], HD['Daa'][Dw,:] )
    if P==U:
        if R==V:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][T,S,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] )
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,S,q,actfloor:actceil], HD['Daa'][Dw,:] )
    if R==V:
        if P==T:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][U,S,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dq,Dw,:,:] )
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][U,S,q,actfloor:actceil], HD['Daa'][Dw,:] )
    return result


def lucc_VACC_VVCC(actfloor, actceil, HD, *e):
    (P,q,R,S),(T,U,V,W) = e
    Dq = q-actfloor
    result = 0
    if P==U:
        if S==V:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,R,W,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +HD['HSaaaa'][q,W,R,T] ))
            if R==W:
                result += 2 * fsum(( +HD['HSaa'][q,T] ,
                                     -einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,T] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
        if R==W:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,S,V,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +HD['HSaaaa'][q,V,S,T] ))
        if S==W:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,R,V,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -HD['HSaaaa'][q,V,R,T] ))
            if R==V:
                result += 2 * fsum(( -HD['HSaa'][q,T] ,
                                     +einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,T] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
        if R==V:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,S,W,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -HD['HSaaaa'][q,W,S,T] ))
    if P==T:
        if S==V:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][U,R,W,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -HD['HSaaaa'][q,W,R,U] ))
            if R==W:
                result += 2 * fsum(( -HD['HSaa'][q,U] ,
                                     +einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,U] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,U], HD['Daa'][:,:] ) ))
        if R==W:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][U,S,V,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 -HD['HSaaaa'][q,V,S,U] ))
        if S==W:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][U,R,V,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +HD['HSaaaa'][q,V,R,U] ))
            if R==V:
                result += 2 * fsum(( +HD['HSaa'][q,U] ,
                                     -einsum( 'II->', HD['HSaaaa'][q,:actfloor,:actfloor,U] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][q,actfloor:actceil,actfloor:actceil,U], HD['Daa'][:,:] ) ))
        if R==V:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][U,S,W,actfloor:actceil], HD['Daa'][Dq,:] ) ,
                                 +HD['HSaaaa'][q,W,S,U] ))
    if S==V:
        if R==W:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                                 -HD['HSaaaa'][P,q,T,U] ))
    if S==W:
        if R==V:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][T,U,actfloor:actceil,P], HD['Daa'][Dq,:] ) ,
                                 +HD['HSaaaa'][P,q,T,U] ))
    return result


def lucc_VVAA_AAAA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(t,u,v,w) = e
    Dr, Ds, Dt, Du, Dv, Dw = r-actfloor, s-actfloor, t-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,t,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,Q,u,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dt,Dv,Dw,:] ) ,
                         +einsum( 'i,i->', HD['HSaaaa'][P,Q,w,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dv,Dt,Du,:] ) ,
                         -einsum( 'i,i->', HD['HSaaaa'][P,Q,v,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Dw,Dt,Du,:] ) ,
                         -HD['HSaaaa'][P,Q,t,u] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                         +HD['HSaaaa'][P,Q,v,w] * HD['Daaaa'][Dr,Ds,Dt,Du] ))
    if s==v:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dw,:,:] )
        if r==w:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,Du,:,:] )
    if r==w:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dv,:,:] )
    if s==w:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dt,Du,Dv,:,:] )
        if r==v:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dt,Du,:,:] )
    if r==v:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dt,Du,Dw,:,:] )
    if s==u:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Dt,:,:] )
        if r==t:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    if r==t:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Du,:,:] )
    if r==u:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Dt,:,:] )
        if s==t:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    if s==t:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Dv,Dw,Du,:,:] )
    return result


def lucc_VVAA_CAAA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,u,v,w) = e
    Dr, Ds, Du, Dv, Dw = r-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] )
    if s==v:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dw,:] )
        if r==w:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daa'][Du,:] )
    if r==w:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dv,:] )
    if s==w:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaa'][Dr,Du,Dv,:] )
        if r==v:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daa'][Du,:] )
    if r==v:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,T,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] )
    return result


def lucc_VVAA_CCAA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,U,v,w) = e
    Dr, Ds, Dv, Dw = r-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * +HD['HSaaaa'][P,Q,T,U] * HD['Daaaa'][Dr,Ds,Dv,Dw]
    if s==v:
        result += 2 * -HD['HSaaaa'][P,Q,T,U] * HD['Daa'][Dr,Dw]
        if r==w:
            result += 2 * +HD['HSaaaa'][P,Q,T,U]
    if r==w:
        result += 2 * -HD['HSaaaa'][P,Q,T,U] * HD['Daa'][Ds,Dv]
    if s==w:
        result += 2 * +HD['HSaaaa'][P,Q,T,U] * HD['Daa'][Dr,Dv]
        if r==v:
            result += 2 * -HD['HSaaaa'][P,Q,T,U]
    if r==v:
        result += 2 * +HD['HSaaaa'][P,Q,T,U] * HD['Daa'][Ds,Dw]
    return result


def lucc_VVAA_VAAA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,u,v,w) = e
    Dr, Ds, Du, Dv, Dw = r-actfloor, s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                         +HD['HSaaaa'][P,Q,u,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ))
    if Q==T:
        result += +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dv,Dw,:,:] )
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             +HD['HSaa'][P,u] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             -HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,u] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,u,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
    if P==T:
        result += -einsum( 'ijk,ijk->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaaaa'][Dr,Ds,Du,:,Dv,Dw,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             -HD['HSaa'][Q,u] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             -einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dv,Dw,:] ) ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,u] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,u,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
    return result


def lucc_VVAA_VACA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,u,V,w) = e
    Dr, Ds, Du, Dw = r-actfloor, s-actfloor, u-actfloor, w-actfloor
    result = 0
    if Q==T:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,V,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dw,:,:] )
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,V,u,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dw,:] )
    if P==T:
        result += -einsum( 'ij,ij->', HD['HSaaaa'][Q,V,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dr,Ds,Du,Dw,:,:] )
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][Q,V,u,actfloor:actceil], HD['Daaaa'][Dr,Ds,Dw,:] )
    return result


def lucc_VVAA_VACC(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,u,V,W) = e
    Dr, Ds, Du = r-actfloor, s-actfloor, u-actfloor
    result = 0
    return result


def lucc_VVAA_VVAA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,U,v,w) = e
    Dr, Ds, Dv, Dw = r-actfloor, s-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * -HD['HSaaaa'][P,Q,T,U] * HD['Daaaa'][Dr,Ds,Dv,Dw]
    if Q==U:
        result += 2 * fsum(( -HD['HSaa'][P,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
        if P==T:
            result += fsum(( +einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dr,:,:] ) ,
                             -einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Ds,:,:] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][r,s,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                                 -einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
    if Q==T:
        result += 2 * fsum(( +HD['HSaa'][P,U] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             -HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,U] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,U], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
        if P==U:
            result += fsum(( -einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Dr,:,:] ) ,
                             +einsum( 'ijk,ijk->', HD['HSaaaa'][r,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Ds,:,:] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][r,s,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] ) ))
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaa'][r,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dr,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][r,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
    if P==U:
        result += 2 * fsum(( +HD['HSaa'][Q,T] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             -HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,T] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,T], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
    if P==T:
        result += 2 * fsum(( -HD['HSaa'][Q,U] * HD['Daaaa'][Dr,Ds,Dv,Dw] ,
                             +HD['Daaaa'][Dr,Ds,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,U] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,U], HD['Daaaaaa'][Dr,Ds,:,Dv,Dw,:] ) ))
    return result


def lucc_VVAA_VVCA(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,U,V,w) = e
    Dr, Ds, Dw = r-actfloor, s-actfloor, w-actfloor
    result = 0
    if P==T:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][Q,V,actfloor:actceil,U], HD['Daaaa'][Dr,Ds,Dw,:] )
        if Q==U:
            result += 2 * fsum(( +HD['HSaa'][s,V] * HD['Daa'][Dr,Dw] ,
                                 -HD['HSaa'][r,V] * HD['Daa'][Ds,Dw] ,
                                 +HD['Daa'][Dr,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,V,:actfloor] ) ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,V,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,V,actfloor:actceil], HD['Daaaa'][Dw,:,Dr,:] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,V,actfloor:actceil], HD['Daaaa'][Dw,:,Ds,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][r,s,V,actfloor:actceil], HD['Daa'][Dw,:] ) ))
    if Q==U:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dw,:] )
    if Q==T:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,U], HD['Daaaa'][Dr,Ds,Dw,:] )
        if P==U:
            result += 2 * fsum(( -HD['HSaa'][s,V] * HD['Daa'][Dr,Dw] ,
                                 +HD['HSaa'][r,V] * HD['Daa'][Ds,Dw] ,
                                 -HD['Daa'][Dr,Dw] * einsum( 'II->', HD['HSaaaa'][s,:actfloor,V,:actfloor] ) ,
                                 +HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][r,:actfloor,V,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,V,actfloor:actceil], HD['Daaaa'][Dw,:,Dr,:] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][r,actfloor:actceil,V,actfloor:actceil], HD['Daaaa'][Dw,:,Ds,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][r,s,V,actfloor:actceil], HD['Daa'][Dw,:] ) ))
    if P==U:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][Q,V,actfloor:actceil,T], HD['Daaaa'][Dr,Ds,Dw,:] )
    return result


def lucc_VVAA_VVCC(actfloor, actceil, HD, *e):
    (P,Q,r,s),(T,U,V,W) = e
    Dr, Ds = r-actfloor, s-actfloor
    result = 0
    if P==T:
        if Q==U:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][V,W,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][V,W,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 -HD['HSaaaa'][r,s,V,W] ))
    if Q==T:
        if P==U:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][V,W,s,actfloor:actceil], HD['Daa'][Dr,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][V,W,r,actfloor:actceil], HD['Daa'][Ds,:] ) ,
                                 +HD['HSaaaa'][r,s,V,W] ))
    return result


def lucc_VVCA_AAAA(actfloor, actceil, HD, *e):
    (P,Q,R,s),(t,u,v,w) = e
    Ds, Dt, Du, Dv, Dw = s-actfloor, t-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * fsum(( -HD['HSaaaa'][P,Q,R,t] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                         +HD['HSaaaa'][P,Q,R,u] * HD['Daaaa'][Ds,Dt,Dv,Dw] ,
                         -HD['HSaaaa'][P,Q,R,w] * HD['Daaaa'][Ds,Dv,Dt,Du] ,
                         +HD['HSaaaa'][P,Q,R,v] * HD['Daaaa'][Ds,Dw,Dt,Du] ))
    if s==w:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,R,actfloor:actceil], HD['Daaaa'][Dt,Du,Dv,:] )
    if s==v:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,R,actfloor:actceil], HD['Daaaa'][Dt,Du,Dw,:] )
    if s==t:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,Q,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] )
    if s==u:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,Q,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Dt,:] )
    return result


def lucc_VVCA_CAAA(actfloor, actceil, HD, *e):
    (P,Q,R,s),(T,u,v,w) = e
    Ds, Du, Dv, Dw = s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * -HD['HSaaaa'][P,Q,R,T] * HD['Daaaa'][Ds,Du,Dv,Dw]
    if R==T:
        result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Dv,Dw,Du,:,:] )
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,w,actfloor:actceil], HD['Daaaa'][Ds,Dv,Du,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][P,Q,v,actfloor:actceil], HD['Daaaa'][Ds,Dw,Du,:] ) ,
                             -HD['HSaaaa'][P,Q,v,w] * HD['Daa'][Ds,Du] ))
    if s==w:
        result += 2 * +HD['HSaaaa'][P,Q,R,T] * HD['Daa'][Du,Dv]
    if s==v:
        result += 2 * -HD['HSaaaa'][P,Q,R,T] * HD['Daa'][Du,Dw]
    return result


def lucc_VVCA_CCAA(actfloor, actceil, HD, *e):
    (P,Q,R,s),(T,U,v,w) = e
    Ds, Dv, Dw = s-actfloor, v-actfloor, w-actfloor
    result = 0
    return result


def lucc_VVCA_VAAA(actfloor, actceil, HD, *e):
    (P,Q,R,s),(T,u,v,w) = e
    Ds, Du, Dv, Dw = s-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    result += 2 * -HD['HSaaaa'][P,Q,R,T] * HD['Daaaa'][Ds,Du,Dv,Dw]
    if Q==T:
        result += 2 * fsum(( -HD['HSaa'][P,R] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                             -HD['Daaaa'][Ds,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                             -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dv,Dw,:] ) ,
                             -einsum( 'i,i->', HD['HSaaaa'][u,R,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
    if P==T:
        result += 2 * fsum(( +HD['HSaa'][Q,R] * HD['Daaaa'][Ds,Du,Dv,Dw] ,
                             +HD['Daaaa'][Ds,Du,Dv,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,R,:actfloor] ) ,
                             +einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,R,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dv,Dw,:] ) ,
                             +einsum( 'i,i->', HD['HSaaaa'][u,R,actfloor:actceil,Q], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
    return result


def lucc_VVCA_VACA(actfloor, actceil, HD, *e):
    (P,Q,R,s),(T,u,V,w) = e
    Ds, Du, Dw = s-actfloor, u-actfloor, w-actfloor
    result = 0
    if R==V:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,actfloor:actceil,T], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                             -HD['HSaaaa'][P,Q,u,T] * HD['Daa'][Ds,Dw] ))
        if Q==T:
            result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dw,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 -HD['HSaa'][P,u] * HD['Daa'][Ds,Dw] ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 +HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,u] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,u,actfloor:actceil], HD['Daaaa'][Ds,:,Dw,:] ) ))
        if P==T:
            result += +einsum( 'ijk,ijk->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Ds,Du,:,Dw,:,:] )
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 +HD['HSaa'][Q,u] * HD['Daa'][Ds,Dw] ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,u] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,u,actfloor:actceil], HD['Daaaa'][Ds,:,Dw,:] ) ))
    if Q==T:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,V,R,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                             -HD['HSaaaa'][P,V,R,u] * HD['Daa'][Ds,Dw] ))
    if P==T:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][Q,V,R,actfloor:actceil], HD['Daaaa'][Ds,Du,Dw,:] ) ,
                             +HD['HSaaaa'][Q,V,R,u] * HD['Daa'][Ds,Dw] ))
    return result


def lucc_VVCA_VACC(actfloor, actceil, HD, *e):
    (P,Q,R,s),(T,u,V,W) = e
    Ds, Du = s-actfloor, u-actfloor
    result = 0
    if Q==T:
        if R==W:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][P,V,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] )
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,V,u,actfloor:actceil], HD['Daa'][Ds,:] )
        if R==V:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][P,W,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] )
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,W,u,actfloor:actceil], HD['Daa'][Ds,:] )
    if R==W:
        if P==T:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][Q,V,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] )
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][Q,V,u,actfloor:actceil], HD['Daa'][Ds,:] )
    if R==V:
        if P==T:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][Q,W,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Ds,Du,:,:] )
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][Q,W,u,actfloor:actceil], HD['Daa'][Ds,:] )
    return result


def lucc_VVCA_VVAA(actfloor, actceil, HD, *e):
    (P,Q,R,s),(T,U,v,w) = e
    Ds, Dv, Dw = s-actfloor, v-actfloor, w-actfloor
    result = 0
    if P==T:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][U,R,actfloor:actceil,Q], HD['Daaaa'][Dv,Dw,Ds,:] )
        if Q==U:
            result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Ds,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                                 +einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
    if Q==U:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Ds,:] )
    if Q==T:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][U,R,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Ds,:] )
        if P==U:
            result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaaaa'][Dv,Dw,:,Ds,:,:] )
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ,
                                 -einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daaaa'][Dv,Dw,Ds,:] ) ))
    if P==U:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,Q], HD['Daaaa'][Dv,Dw,Ds,:] )
    return result


def lucc_VVCA_VVCA(actfloor, actceil, HD, *e):
    (P,Q,R,s),(T,U,V,w) = e
    Ds, Dw = s-actfloor, w-actfloor
    result = 0
    if R==V:
        result += 2 * +HD['HSaaaa'][P,Q,T,U] * HD['Daa'][Ds,Dw]
        if Q==U:
            result += 2 * fsum(( +HD['HSaa'][P,T] * HD['Daa'][Ds,Dw] ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Ds,:,Dw,:] ) ))
            if P==T:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ))
        if Q==T:
            result += 2 * fsum(( -HD['HSaa'][P,U] * HD['Daa'][Ds,Dw] ,
                                 +HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,U] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,U], HD['Daaaa'][Ds,:,Dw,:] ) ))
            if P==U:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][s,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][s,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][s,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ))
        if P==U:
            result += 2 * fsum(( -HD['HSaa'][Q,T] * HD['Daa'][Ds,Dw] ,
                                 +HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,T] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,T], HD['Daaaa'][Ds,:,Dw,:] ) ))
        if P==T:
            result += 2 * fsum(( +HD['HSaa'][Q,U] * HD['Daa'][Ds,Dw] ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,U] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,U], HD['Daaaa'][Ds,:,Dw,:] ) ))
    if Q==U:
        result += 2 * +HD['HSaaaa'][P,V,R,T] * HD['Daa'][Ds,Dw]
        if P==T:
            result += 2 * fsum(( -HD['HSaa'][R,V] * HD['Daa'][Ds,Dw] ,
                                 -HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][R,:actfloor,V,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,V,actfloor:actceil], HD['Daaaa'][Dw,:,Ds,:] ) ))
    if Q==T:
        result += 2 * -HD['HSaaaa'][P,V,R,U] * HD['Daa'][Ds,Dw]
        if P==U:
            result += 2 * fsum(( +HD['HSaa'][R,V] * HD['Daa'][Ds,Dw] ,
                                 +HD['Daa'][Ds,Dw] * einsum( 'II->', HD['HSaaaa'][R,:actfloor,V,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,V,actfloor:actceil], HD['Daaaa'][Dw,:,Ds,:] ) ))
    if P==U:
        result += 2 * -HD['HSaaaa'][Q,V,R,T] * HD['Daa'][Ds,Dw]
    if P==T:
        result += 2 * +HD['HSaaaa'][Q,V,R,U] * HD['Daa'][Ds,Dw]
    return result


def lucc_VVCA_VVCC(actfloor, actceil, HD, *e):
    (P,Q,R,s),(T,U,V,W) = e
    Ds = s-actfloor
    result = 0
    if R==W:
        if P==T:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][Q,V,actfloor:actceil,U], HD['Daa'][Ds,:] )
            if Q==U:
                result += 2 * fsum(( +HD['HSaa'][s,V] ,
                                     +einsum( 'II->', HD['HSaaaa'][s,:actfloor,V,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,V,actfloor:actceil], HD['Daa'][:,:] ) ))
        if Q==U:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,T], HD['Daa'][Ds,:] )
        if Q==T:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,V,actfloor:actceil,U], HD['Daa'][Ds,:] )
            if P==U:
                result += 2 * fsum(( -HD['HSaa'][s,V] ,
                                     -einsum( 'II->', HD['HSaaaa'][s,:actfloor,V,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,V,actfloor:actceil], HD['Daa'][:,:] ) ))
        if P==U:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][Q,V,actfloor:actceil,T], HD['Daa'][Ds,:] )
    if R==V:
        if P==T:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][Q,W,actfloor:actceil,U], HD['Daa'][Ds,:] )
            if Q==U:
                result += 2 * fsum(( -HD['HSaa'][s,W] ,
                                     -einsum( 'II->', HD['HSaaaa'][s,:actfloor,W,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,W,actfloor:actceil], HD['Daa'][:,:] ) ))
        if Q==U:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][P,W,actfloor:actceil,T], HD['Daa'][Ds,:] )
        if Q==T:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][P,W,actfloor:actceil,U], HD['Daa'][Ds,:] )
            if P==U:
                result += 2 * fsum(( +HD['HSaa'][s,W] ,
                                     +einsum( 'II->', HD['HSaaaa'][s,:actfloor,W,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][s,actfloor:actceil,W,actfloor:actceil], HD['Daa'][:,:] ) ))
        if P==U:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][Q,W,actfloor:actceil,T], HD['Daa'][Ds,:] )
    if P==T:
        if Q==U:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][V,W,R,actfloor:actceil], HD['Daa'][Ds,:] )
    if Q==T:
        if P==U:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][V,W,R,actfloor:actceil], HD['Daa'][Ds,:] )
    return result


def lucc_VVCC_AAAA(actfloor, actceil, HD, *e):
    (P,Q,R,S),(t,u,v,w) = e
    Dt, Du, Dv, Dw = t-actfloor, u-actfloor, v-actfloor, w-actfloor
    result = 0
    return result


def lucc_VVCC_CAAA(actfloor, actceil, HD, *e):
    (P,Q,R,S),(T,u,v,w) = e
    Du, Dv, Dw = u-actfloor, v-actfloor, w-actfloor
    result = 0
    if S==T:
        result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,R,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             +HD['HSaaaa'][P,Q,R,w] * HD['Daa'][Du,Dv] ,
                             -HD['HSaaaa'][P,Q,R,v] * HD['Daa'][Du,Dw] ))
    if R==T:
        result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,S,actfloor:actceil], HD['Daaaa'][Dv,Dw,Du,:] ) ,
                             -HD['HSaaaa'][P,Q,S,w] * HD['Daa'][Du,Dv] ,
                             +HD['HSaaaa'][P,Q,S,v] * HD['Daa'][Du,Dw] ))
    return result


def lucc_VVCC_CCAA(actfloor, actceil, HD, *e):
    (P,Q,R,S),(T,U,v,w) = e
    Dv, Dw = v-actfloor, w-actfloor
    result = 0
    if S==U:
        if R==T:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,w,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][P,Q,v,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 -HD['HSaaaa'][P,Q,v,w] ))
    if R==U:
        if S==T:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][P,Q,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,w,actfloor:actceil], HD['Daa'][Dv,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][P,Q,v,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                 +HD['HSaaaa'][P,Q,v,w] ))
    return result


def lucc_VVCC_VAAA(actfloor, actceil, HD, *e):
    (P,Q,R,S),(T,u,v,w) = e
    Du, Dv, Dw = u-actfloor, v-actfloor, w-actfloor
    result = 0
    if Q==T:
        result += 2 * +einsum( 'i,i->', HD['HSaaaa'][R,S,actfloor:actceil,P], HD['Daaaa'][Dv,Dw,Du,:] )
    if P==T:
        result += 2 * -einsum( 'i,i->', HD['HSaaaa'][R,S,actfloor:actceil,Q], HD['Daaaa'][Dv,Dw,Du,:] )
    return result


def lucc_VVCC_VACA(actfloor, actceil, HD, *e):
    (P,Q,R,S),(T,u,V,w) = e
    Du, Dw = u-actfloor, w-actfloor
    result = 0
    if Q==T:
        result += 2 * -HD['HSaaaa'][P,V,R,S] * HD['Daa'][Du,Dw]
        if S==V:
            result += 2 * fsum(( +HD['HSaa'][P,R] * HD['Daa'][Du,Dw] ,
                                 +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,R,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][u,R,actfloor:actceil,P], HD['Daa'][Dw,:] ) ))
        if R==V:
            result += 2 * fsum(( -HD['HSaa'][P,S] * HD['Daa'][Du,Dw] ,
                                 -HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][P,:actfloor,S,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,S,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][u,S,actfloor:actceil,P], HD['Daa'][Dw,:] ) ))
    if S==V:
        result += 2 * +HD['HSaaaa'][P,Q,R,T] * HD['Daa'][Du,Dw]
        if P==T:
            result += 2 * fsum(( -HD['HSaa'][Q,R] * HD['Daa'][Du,Dw] ,
                                 -HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,R,:actfloor] ) ,
                                 +einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,R,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                                 -einsum( 'i,i->', HD['HSaaaa'][u,R,actfloor:actceil,Q], HD['Daa'][Dw,:] ) ))
    if R==V:
        result += 2 * -HD['HSaaaa'][P,Q,S,T] * HD['Daa'][Du,Dw]
        if P==T:
            result += 2 * fsum(( +HD['HSaa'][Q,S] * HD['Daa'][Du,Dw] ,
                                 +HD['Daa'][Du,Dw] * einsum( 'II->', HD['HSaaaa'][Q,:actfloor,S,:actfloor] ) ,
                                 -einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,S,actfloor:actceil], HD['Daaaa'][Du,:,Dw,:] ) ,
                                 +einsum( 'i,i->', HD['HSaaaa'][u,S,actfloor:actceil,Q], HD['Daa'][Dw,:] ) ))
    if P==T:
        result += 2 * +HD['HSaaaa'][Q,V,R,S] * HD['Daa'][Du,Dw]
    return result


def lucc_VVCC_VACC(actfloor, actceil, HD, *e):
    (P,Q,R,S),(T,u,V,W) = e
    Du = u-actfloor
    result = 0
    if Q==T:
        if S==V:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,W,R,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 +HD['HSaaaa'][P,W,R,u] ))
            if R==W:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     +HD['HSaa'][P,u] ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     -einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,u] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,u,actfloor:actceil], HD['Daa'][:,:] ) ))
        if R==W:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,V,S,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 +HD['HSaaaa'][P,V,S,u] ))
        if S==W:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,V,R,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 -HD['HSaaaa'][P,V,R,u] ))
            if R==V:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][P,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     -HD['HSaa'][P,u] ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][P,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     +einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,u] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,u,actfloor:actceil], HD['Daa'][:,:] ) ))
        if R==V:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,W,S,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 -HD['HSaaaa'][P,W,S,u] ))
    if P==T:
        if S==V:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][Q,W,R,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 -HD['HSaaaa'][Q,W,R,u] ))
            if R==W:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     -HD['HSaa'][Q,u] ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     +einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,u] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,u,actfloor:actceil], HD['Daa'][:,:] ) ))
        if R==W:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][Q,V,S,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 -HD['HSaaaa'][Q,V,S,u] ))
        if S==W:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][Q,V,R,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 +HD['HSaaaa'][Q,V,R,u] ))
            if R==V:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Du,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][Q,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     +HD['HSaa'][Q,u] ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][Q,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                     -einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,u] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,u,actfloor:actceil], HD['Daa'][:,:] ) ))
        if R==V:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][Q,W,S,actfloor:actceil], HD['Daa'][Du,:] ) ,
                                 +HD['HSaaaa'][Q,W,S,u] ))
    if S==V:
        if R==W:
            result += 2 * fsum(( -einsum( 'i,i->', HD['HSaaaa'][P,Q,actfloor:actceil,T], HD['Daa'][Du,:] ) ,
                                 +HD['HSaaaa'][P,Q,u,T] ))
    if S==W:
        if R==V:
            result += 2 * fsum(( +einsum( 'i,i->', HD['HSaaaa'][P,Q,actfloor:actceil,T], HD['Daa'][Du,:] ) ,
                                 -HD['HSaaaa'][P,Q,u,T] ))
    return result


def lucc_VVCC_VVAA(actfloor, actceil, HD, *e):
    (P,Q,R,S),(T,U,v,w) = e
    Dv, Dw = v-actfloor, w-actfloor
    result = 0
    if P==T:
        if Q==U:
            result += -einsum( 'ij,ij->', HD['HSaaaa'][R,S,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    if Q==T:
        if P==U:
            result += +einsum( 'ij,ij->', HD['HSaaaa'][R,S,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dv,Dw,:,:] )
    return result


def lucc_VVCC_VVCA(actfloor, actceil, HD, *e):
    (P,Q,R,S),(T,U,V,w) = e
    Dw = w-actfloor
    result = 0
    if R==V:
        if P==T:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][U,S,actfloor:actceil,Q], HD['Daa'][Dw,:] )
            if Q==U:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,S,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][S,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][S,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ))
        if Q==U:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,S,actfloor:actceil,P], HD['Daa'][Dw,:] )
        if Q==T:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][U,S,actfloor:actceil,P], HD['Daa'][Dw,:] )
            if P==U:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,S,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][S,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][S,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ))
        if P==U:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,S,actfloor:actceil,Q], HD['Daa'][Dw,:] )
    if S==V:
        if Q==U:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,P], HD['Daa'][Dw,:] )
            if P==T:
                result += +einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( +einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     -einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ))
        if P==T:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][U,R,actfloor:actceil,Q], HD['Daa'][Dw,:] )
        if Q==T:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][U,R,actfloor:actceil,P], HD['Daa'][Dw,:] )
            if P==U:
                result += -einsum( 'ijk,ijk->', HD['HSaaaa'][actfloor:actceil,R,actfloor:actceil,actfloor:actceil], HD['Daaaa'][Dw,:,:,:] )
                result += 2 * fsum(( -einsum( 'i,i->', HD['HSaa'][R,actfloor:actceil], HD['Daa'][Dw,:] ) ,
                                     +einsum( 'IIj,j->', HD['HSaaaa'][R,:actfloor,:actfloor,actfloor:actceil], HD['Daa'][Dw,:] ) ))
        if P==U:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][T,R,actfloor:actceil,Q], HD['Daa'][Dw,:] )
    if P==T:
        if Q==U:
            result += 2 * +einsum( 'i,i->', HD['HSaaaa'][R,S,V,actfloor:actceil], HD['Daa'][Dw,:] )
    if Q==T:
        if P==U:
            result += 2 * -einsum( 'i,i->', HD['HSaaaa'][R,S,V,actfloor:actceil], HD['Daa'][Dw,:] )
    return result


def lucc_VVCC_VVCC(actfloor, actceil, HD, *e):
    (P,Q,R,S),(T,U,V,W) = e
    result = 0
    if S==V:
        if R==W:
            result += 2 * -HD['HSaaaa'][P,Q,T,U]
            if Q==U:
                result += 2 * fsum(( -HD['HSaa'][P,T] ,
                                     +einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
            if Q==T:
                result += 2 * fsum(( +HD['HSaa'][P,U] ,
                                     -einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,U] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,U], HD['Daa'][:,:] ) ))
            if P==U:
                result += 2 * fsum(( +HD['HSaa'][Q,T] ,
                                     -einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,T] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
            if P==T:
                result += 2 * fsum(( -HD['HSaa'][Q,U] ,
                                     +einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,U] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,U], HD['Daa'][:,:] ) ))
        if Q==U:
            result += 2 * -HD['HSaaaa'][P,W,R,T]
            if P==T:
                result += 2 * fsum(( +HD['HSaa'][R,W] ,
                                     +einsum( 'II->', HD['HSaaaa'][R,:actfloor,W,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,W,actfloor:actceil], HD['Daa'][:,:] ) ))
        if Q==T:
            result += 2 * +HD['HSaaaa'][P,W,R,U]
            if P==U:
                result += 2 * fsum(( -HD['HSaa'][R,W] ,
                                     -einsum( 'II->', HD['HSaaaa'][R,:actfloor,W,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,W,actfloor:actceil], HD['Daa'][:,:] ) ))
        if P==U:
            result += 2 * +HD['HSaaaa'][Q,W,R,T]
        if P==T:
            result += 2 * -HD['HSaaaa'][Q,W,R,U]
    if Q==U:
        if P==T:
            result += 2 * +HD['HSaaaa'][R,S,V,W]
            if R==W:
                result += 2 * fsum(( +HD['HSaa'][S,V] ,
                                     +einsum( 'II->', HD['HSaaaa'][S,:actfloor,V,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,S,V,actfloor:actceil], HD['Daa'][:,:] ) ))
            if R==V:
                result += 2 * fsum(( -HD['HSaa'][S,W] ,
                                     -einsum( 'II->', HD['HSaaaa'][S,:actfloor,W,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,S,W,actfloor:actceil], HD['Daa'][:,:] ) ))
            if S==W:
                result += 2 * fsum(( -HD['HSaa'][R,V] ,
                                     -einsum( 'II->', HD['HSaaaa'][R,:actfloor,V,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,V,actfloor:actceil], HD['Daa'][:,:] ) ))
        if S==W:
            result += 2 * +HD['HSaaaa'][P,V,R,T]
            if R==V:
                result += 2 * fsum(( +HD['HSaa'][P,T] ,
                                     -einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,T] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
        if R==V:
            result += 2 * +HD['HSaaaa'][P,W,S,T]
        if R==W:
            result += 2 * -HD['HSaaaa'][P,V,S,T]
    if R==W:
        if Q==T:
            result += 2 * +HD['HSaaaa'][P,V,S,U]
            if P==U:
                result += 2 * fsum(( -HD['HSaa'][S,V] ,
                                     -einsum( 'II->', HD['HSaaaa'][S,:actfloor,V,:actfloor] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,S,V,actfloor:actceil], HD['Daa'][:,:] ) ))
        if P==U:
            result += 2 * +HD['HSaaaa'][Q,V,S,T]
        if P==T:
            result += 2 * -HD['HSaaaa'][Q,V,S,U]
    if S==W:
        if R==V:
            result += 2 * +HD['HSaaaa'][P,Q,T,U]
            if Q==T:
                result += 2 * fsum(( -HD['HSaa'][P,U] ,
                                     +einsum( 'II->', HD['HSaaaa'][P,:actfloor,:actfloor,U] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][P,actfloor:actceil,actfloor:actceil,U], HD['Daa'][:,:] ) ))
            if P==U:
                result += 2 * fsum(( -HD['HSaa'][Q,T] ,
                                     +einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,T] ) ,
                                     +einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,T], HD['Daa'][:,:] ) ))
            if P==T:
                result += 2 * fsum(( +HD['HSaa'][Q,U] ,
                                     -einsum( 'II->', HD['HSaaaa'][Q,:actfloor,:actfloor,U] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][Q,actfloor:actceil,actfloor:actceil,U], HD['Daa'][:,:] ) ))
        if Q==T:
            result += 2 * -HD['HSaaaa'][P,V,R,U]
            if P==U:
                result += 2 * fsum(( +HD['HSaa'][R,V] ,
                                     +einsum( 'II->', HD['HSaaaa'][R,:actfloor,V,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,R,V,actfloor:actceil], HD['Daa'][:,:] ) ))
        if P==U:
            result += 2 * -HD['HSaaaa'][Q,V,R,T]
        if P==T:
            result += 2 * +HD['HSaaaa'][Q,V,R,U]
    if R==V:
        if P==U:
            result += 2 * -HD['HSaaaa'][Q,W,S,T]
            if Q==T:
                result += 2 * fsum(( +HD['HSaa'][S,W] ,
                                     +einsum( 'II->', HD['HSaaaa'][S,:actfloor,W,:actfloor] ) ,
                                     -einsum( 'ij,ij->', HD['HSaaaa'][actfloor:actceil,S,W,actfloor:actceil], HD['Daa'][:,:] ) ))
        if Q==T:
            result += 2 * -HD['HSaaaa'][P,W,S,U]
        if P==T:
            result += 2 * +HD['HSaaaa'][Q,W,S,U]
    if Q==T:
        if P==U:
            result += 2 * -HD['HSaaaa'][R,S,V,W]
    return result



FUNCMAPPER = {
'AA': lucc_AA , 'AA_AA': lucc_AA_AA , 'AA_AAAA': lucc_AA_AAAA , 'AAAA': lucc_AAAA ,
'AAAA_AA': lucc_AAAA_AA , 'AAAA_AAAA': lucc_AAAA_AAAA , 'AA_CA': lucc_AA_CA , 'AA_VA': lucc_AA_VA ,
'AA_VC': lucc_AA_VC , 'CA_AA': lucc_CA_AA , 'CA_CA': lucc_CA_CA , 'CA_VA': lucc_CA_VA ,
'CA_VC': lucc_CA_VC , 'VA_AA': lucc_VA_AA , 'VA_CA': lucc_VA_CA , 'VA_VA': lucc_VA_VA ,
'VA_VC': lucc_VA_VC , 'VC_AA': lucc_VC_AA , 'VC_CA': lucc_VC_CA , 'VC_VA': lucc_VC_VA ,
'VC_VC': lucc_VC_VC , 'AA_CAAA': lucc_AA_CAAA , 'AA_CCAA': lucc_AA_CCAA , 'AA_VAAA': lucc_AA_VAAA ,
'AA_VACA': lucc_AA_VACA , 'AA_VACC': lucc_AA_VACC , 'AA_VVAA': lucc_AA_VVAA , 'AA_VVCA': lucc_AA_VVCA ,
'AA_VVCC': lucc_AA_VVCC , 'CA_AAAA': lucc_CA_AAAA , 'CA_CAAA': lucc_CA_CAAA , 'CA_CCAA': lucc_CA_CCAA ,
'CA_VAAA': lucc_CA_VAAA , 'CA_VACA': lucc_CA_VACA , 'CA_VACC': lucc_CA_VACC , 'CA_VVAA': lucc_CA_VVAA ,
'CA_VVCA': lucc_CA_VVCA , 'CA_VVCC': lucc_CA_VVCC , 'VA_AAAA': lucc_VA_AAAA , 'VA_CAAA': lucc_VA_CAAA ,
'VA_CCAA': lucc_VA_CCAA , 'VA_VAAA': lucc_VA_VAAA , 'VA_VACA': lucc_VA_VACA , 'VA_VACC': lucc_VA_VACC ,
'VA_VVAA': lucc_VA_VVAA , 'VA_VVCA': lucc_VA_VVCA , 'VA_VVCC': lucc_VA_VVCC , 'VC_AAAA': lucc_VC_AAAA ,
'VC_CAAA': lucc_VC_CAAA , 'VC_CCAA': lucc_VC_CCAA , 'VC_VAAA': lucc_VC_VAAA , 'VC_VACA': lucc_VC_VACA ,
'VC_VACC': lucc_VC_VACC , 'VC_VVAA': lucc_VC_VVAA , 'VC_VVCA': lucc_VC_VVCA , 'VC_VVCC': lucc_VC_VVCC ,
'AAAA_CA': lucc_AAAA_CA , 'AAAA_VA': lucc_AAAA_VA , 'AAAA_VC': lucc_AAAA_VC , 'CAAA_AA': lucc_CAAA_AA ,
'CAAA_CA': lucc_CAAA_CA , 'CAAA_VA': lucc_CAAA_VA , 'CAAA_VC': lucc_CAAA_VC , 'CCAA_AA': lucc_CCAA_AA ,
'CCAA_CA': lucc_CCAA_CA , 'CCAA_VA': lucc_CCAA_VA , 'CCAA_VC': lucc_CCAA_VC , 'VAAA_AA': lucc_VAAA_AA ,
'VAAA_CA': lucc_VAAA_CA , 'VAAA_VA': lucc_VAAA_VA , 'VAAA_VC': lucc_VAAA_VC , 'VACA_AA': lucc_VACA_AA ,
'VACA_CA': lucc_VACA_CA , 'VACA_VA': lucc_VACA_VA , 'VACA_VC': lucc_VACA_VC , 'VACC_AA': lucc_VACC_AA ,
'VACC_CA': lucc_VACC_CA , 'VACC_VA': lucc_VACC_VA , 'VACC_VC': lucc_VACC_VC , 'VVAA_AA': lucc_VVAA_AA ,
'VVAA_CA': lucc_VVAA_CA , 'VVAA_VA': lucc_VVAA_VA , 'VVAA_VC': lucc_VVAA_VC , 'VVCA_AA': lucc_VVCA_AA ,
'VVCA_CA': lucc_VVCA_CA , 'VVCA_VA': lucc_VVCA_VA , 'VVCA_VC': lucc_VVCA_VC , 'VVCC_AA': lucc_VVCC_AA ,
'VVCC_CA': lucc_VVCC_CA , 'VVCC_VA': lucc_VVCC_VA , 'VVCC_VC': lucc_VVCC_VC , 'AAAA_CAAA': lucc_AAAA_CAAA ,
'AAAA_CCAA': lucc_AAAA_CCAA , 'AAAA_VAAA': lucc_AAAA_VAAA , 'AAAA_VACA': lucc_AAAA_VACA , 'AAAA_VACC': lucc_AAAA_VACC ,
'AAAA_VVAA': lucc_AAAA_VVAA , 'AAAA_VVCA': lucc_AAAA_VVCA , 'AAAA_VVCC': lucc_AAAA_VVCC , 'CAAA_AAAA': lucc_CAAA_AAAA ,
'CAAA_CAAA': lucc_CAAA_CAAA , 'CAAA_CCAA': lucc_CAAA_CCAA , 'CAAA_VAAA': lucc_CAAA_VAAA , 'CAAA_VACA': lucc_CAAA_VACA ,
'CAAA_VACC': lucc_CAAA_VACC , 'CAAA_VVAA': lucc_CAAA_VVAA , 'CAAA_VVCA': lucc_CAAA_VVCA , 'CAAA_VVCC': lucc_CAAA_VVCC ,
'CCAA_AAAA': lucc_CCAA_AAAA , 'CCAA_CAAA': lucc_CCAA_CAAA , 'CCAA_CCAA': lucc_CCAA_CCAA , 'CCAA_VAAA': lucc_CCAA_VAAA ,
'CCAA_VACA': lucc_CCAA_VACA , 'CCAA_VACC': lucc_CCAA_VACC , 'CCAA_VVAA': lucc_CCAA_VVAA , 'CCAA_VVCA': lucc_CCAA_VVCA ,
'CCAA_VVCC': lucc_CCAA_VVCC , 'VAAA_AAAA': lucc_VAAA_AAAA , 'VAAA_CAAA': lucc_VAAA_CAAA , 'VAAA_CCAA': lucc_VAAA_CCAA ,
'VAAA_VAAA': lucc_VAAA_VAAA , 'VAAA_VACA': lucc_VAAA_VACA , 'VAAA_VACC': lucc_VAAA_VACC , 'VAAA_VVAA': lucc_VAAA_VVAA ,
'VAAA_VVCA': lucc_VAAA_VVCA , 'VAAA_VVCC': lucc_VAAA_VVCC , 'VACA_AAAA': lucc_VACA_AAAA , 'VACA_CAAA': lucc_VACA_CAAA ,
'VACA_CCAA': lucc_VACA_CCAA , 'VACA_VAAA': lucc_VACA_VAAA , 'VACA_VACA': lucc_VACA_VACA , 'VACA_VACC': lucc_VACA_VACC ,
'VACA_VVAA': lucc_VACA_VVAA , 'VACA_VVCA': lucc_VACA_VVCA , 'VACA_VVCC': lucc_VACA_VVCC , 'VACC_AAAA': lucc_VACC_AAAA ,
'VACC_CAAA': lucc_VACC_CAAA , 'VACC_CCAA': lucc_VACC_CCAA , 'VACC_VAAA': lucc_VACC_VAAA , 'VACC_VACA': lucc_VACC_VACA ,
'VACC_VACC': lucc_VACC_VACC , 'VACC_VVAA': lucc_VACC_VVAA , 'VACC_VVCA': lucc_VACC_VVCA , 'VACC_VVCC': lucc_VACC_VVCC ,
'VVAA_AAAA': lucc_VVAA_AAAA , 'VVAA_CAAA': lucc_VVAA_CAAA , 'VVAA_CCAA': lucc_VVAA_CCAA , 'VVAA_VAAA': lucc_VVAA_VAAA ,
'VVAA_VACA': lucc_VVAA_VACA , 'VVAA_VACC': lucc_VVAA_VACC , 'VVAA_VVAA': lucc_VVAA_VVAA , 'VVAA_VVCA': lucc_VVAA_VVCA ,
'VVAA_VVCC': lucc_VVAA_VVCC , 'VVCA_AAAA': lucc_VVCA_AAAA , 'VVCA_CAAA': lucc_VVCA_CAAA , 'VVCA_CCAA': lucc_VVCA_CCAA ,
'VVCA_VAAA': lucc_VVCA_VAAA , 'VVCA_VACA': lucc_VVCA_VACA , 'VVCA_VACC': lucc_VVCA_VACC , 'VVCA_VVAA': lucc_VVCA_VVAA ,
'VVCA_VVCA': lucc_VVCA_VVCA , 'VVCA_VVCC': lucc_VVCA_VVCC , 'VVCC_AAAA': lucc_VVCC_AAAA , 'VVCC_CAAA': lucc_VVCC_CAAA ,
'VVCC_CCAA': lucc_VVCC_CCAA , 'VVCC_VAAA': lucc_VVCC_VAAA , 'VVCC_VACA': lucc_VVCC_VACA , 'VVCC_VACC': lucc_VVCC_VACC ,
'VVCC_VVAA': lucc_VVCC_VVAA , 'VVCC_VVCA': lucc_VVCC_VVCA , 'VVCC_VVCC': lucc_VVCC_VVCC
}