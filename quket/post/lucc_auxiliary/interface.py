"""
#######################
#        quket        #
#######################
Interface of LUCC sub functions.
"""

from itertools import product, groupby, chain
from operator import itemgetter

from numpy import array, zeros

from quket.fileio.fileio import prints
from quket.post.rdm import get_1RDM_full, get_2RDM_full, get_3RDM_full, get_4RDM_full
from quket.orbital.misc import get_htilde


#                            Frozen || Core | Active || Secondary
#                              nf   ||  nc  |   na   ||    ns
#
#    integrals                 nf   ||  nc  |   na   ||    ns
#    integrals_active               ||      |   na   ||
#    H_tlide                   nf   ||  nc  |   na   ||    ns
#    1234RDM                        ||      |   na   ||
#    excitation_list                ||  nc  |   na   ||    ns


def interface(mode, Quket, excitations):

    from numpy import sum as npsum

    from quket.post.lucc_auxiliary.func_spinorbital import do_spinorbital
    from quket.post.lucc_auxiliary.func_spinorbital_extended import do_spinorbital_extended

    if mode == "special":
        prints('special_lucc_spinorbital')
        #=== Obtain the dataset for einsum abyss ==================================#
        HD = special_lucc_spinorbital_preprocessing(Quket)
    elif mode == 'general':
        prints('general_lucc_spinorbital')
        #=== Obtain the dataset for einsum abyss ==================================#
        HD = general_lucc_spinorbital_preprocessing(Quket)
    else:
        raise ValueError(f"Only have 'special' or 'genreal' -> {mode}")

    LAGACY = 0
    if LAGACY:
        from quket.post.lucc_auxiliary.func_fullspin import do_spinfree
        from quket.post.lucc_auxiliary.func_fullspin_extended import do_spinfree_extended

    if not Quket.spinfree:

        if mode == "special":
            return  do_spinfree(Quket, excitations) \
                    if LAGACY else \
                    do_spinorbital(Quket, HD, excitations, excitations)

        elif mode == 'general':
            return do_spinfree_extended(Quket, excitations) \
                    if LAGACY else \
                    do_spinorbital_extended(Quket, HD, excitations, excitations)

        else:
            raise ValueError(f"Only have 'special' or 'genreal' -> {mode}")

    else:
        ####  modifing spinfree excitation_list  #######################################
        excitations = exploit_spinfree_excitation(excitations)

        Amat_dimension = (len(excitations), len(excitations))
        Amat = zeros(Amat_dimension)

        for (i,tv), (j,pr) in product(enumerate(excitations), repeat=2):

            if not len(tv) or not len(pr):
                Amat[i,j] = 0
                continue

            if mode == "special":
                Amat[i,j] = npsum(do_spinorbital(Quket, HD, tv, pr))
            elif mode == 'general':
                Amat[i,j] = npsum(do_spinorbital_extended(Quket, HD, tv, pr))
            else:
                raise ValueError(f"Only have 'special' or 'genreal' -> {mode}")

        return Amat

def exploit_spinfree_excitation(excitations):
    """
    Assuming that terms inside excitations is given in space-orbital
    """

    new = list(excitations[:])
    for i,row in enumerate(new):

        if len(row) == 2:
            pa, qa = [x*2 for x in row]
            pb, qb = [x+1 for x in [pa,qa]]
            candidates = ((pa, qa),
                          (pb, qb))
            candidate = tuple(x for x in candidates
                              if x[0]!=x[1])
            new[i] = candidate

        elif len(row) == 4:
            pa, qa, ra, sa = [x*2 for x in row]
            pb, qb, rb, sb = [x+1 for x in [pa, qa, ra, sa]]
            candidates = ((pa, qa, ra, sa),
                          (pa, qb, rb, sa),
                          (pb, qa, ra, sb),
                          (pb, qb, rb, sb))

            candidates = tuple(x for x in candidates
                              if set(x[:2])!=set(x[2:])
                                and x[0]!=x[1]
                                and x[2]!=x[3])
            new[i] = candidates

        else:
            raise NotImplementedError

    return new


def group_excitation_by_spin(L, debug=False):
    """Assuming full-spin index (index=qubit)"""
    GEBS = dict()

    for i,ext in enumerate(L):
        if len(ext)==0:
            continue
        is_even = [ x%2 for x in ext ]
        is_even_str = list(map(str, is_even))

        alphabet = ''.join([ 'b' if x else 'a' for x in is_even ])
        binary = ''.join(is_even_str)
        decimal = int(''.join(binary), base=2)

        key = alphabet, binary

        if alphabet not in GEBS:
            GEBS[alphabet] = []

        GEBS[alphabet].append(tuple([i]+ext))

        if debug: print(alphabet, binary, decimal)

    if debug:
        for k,v in GEBS.items():
            print(k, v, '\n')

    return GEBS


def sort_by_space_spin(ActFloor, ActCeil, e, debug=0):
    """Assuming full-spin index (index=qubit)

    Determination of CoreSpace or VirSpace through
    >>> spacetype = [1 if x<ActFloor else 2 if x>ActCeil else 0 for x in e]
    ActFloor and ActCeil in there is relative to the scope of excitation_list

    > excitation_list : [ frozen || core* | act* || vir* ]
    > counting 0 from the core

    So to say a excitation is from active space,
    the index need to be larger than 2*Quket.n_core_orbitals

    ActFloor = 2*Quket.n_core_orbitals
    ActCeil  = 2* (Quket.n_core_orbitals + 2*Quket.n_active_orbitals)
    """

    SPACE = {0: 'A', 1: 'C', 2: 'V'}
    spacetype = [1 if x<ActFloor else 2 if x>=ActCeil else 0 for x in e]
    spintype = [x%2 for x in e]

    if debug:
        print(f"\t\t    {e}\t{''.join(SPACE[x] for x in spacetype)}\t"
                f"{''.join('b' if x else 'a' for x in spintype)}")

    #--- sort : CCAV -> CCVA
    sort_this = tuple(zip(e, spacetype, spintype))

    mid = len(e)//2
    left, right = sort_this[:mid], sort_this[mid:]

    left, parity = paritymergesort(left, needswap=needswap_space_spin)
    right, parity = paritymergesort(right, parity, needswap=needswap_space_spin)

    #--- transpose : CCVA -> VACC
    if max(tuple(map(itemgetter(1), left))) < max(tuple(map(itemgetter(1), right))):

        left, right = right, left
        parity ^= 1

    e, spacetype, spintype = zip(*left+right)
    spacetype = ''.join(SPACE[x] for x in spacetype)
    spintype = ''.join('b' if x else 'a' for x in spintype)

    if debug:
        print(f"\t\t>>> {e}\t{spacetype}\t{spintype}\t{parity}")
        print()
    return e, spacetype, spintype, parity


def needswap_space_spin(a, b):
    # a = (index, space, spin)
    if a[1] < b[1]:
        return True
    elif a[1] == b[1]:
        if a[2] < b[2]:
            return True
        elif a[2] == b[2]:
            if a[0] < b[0]:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def paritymergesort(A, parity=0, needswap=needswap_space_spin):
    '''
    return:
        sorted list
        parity
    '''

    mid = len(A)//2
    left, right = A[:mid], A[mid:]
    lenLeft, lenRight = len(left), len(right)
    left_i = right_i = 0

    if lenLeft>1:
        left, parity = paritymergesort(left, parity, needswap)

    if lenRight>1:
        right, parity = paritymergesort(right, parity, needswap)

    newA = []
    put = newA.append

    while lenLeft and lenRight:

        while lenLeft and not needswap(left[left_i], right[right_i]):
            put(left[left_i])
            left_i += 1
            lenLeft -= 1

        if lenLeft:
            if lenLeft%2:
                while lenRight and needswap(left[left_i], right[right_i]):
                    put(right[right_i])
                    parity ^= 1
                    right_i += 1
                    lenRight -= 1

            else:
                while lenRight and needswap(left[left_i], right[right_i]):
                    put(right[right_i])
                    right_i += 1
                    lenRight -= 1

    if lenLeft: newA += left[left_i:]
    if lenRight: newA += right[right_i:]

    return newA, parity


def spintypes_from_fullspin_excitation_list(excitation_list, mode=str):
    if mode == str:
        return [ ''.join('b' if y%2 else 'a' for y in x)
                for x in excitation_list ]
    elif mode == int:
        return [ [y%2 for y in x]
                for x in excitation_list ]
    else:
        raise NotImplementedError


def spacetypes_from_fullspin_excitation_list(excitation_list, ActFloor, ActCeil, mode=str):
    if mode == str:
        return [ ''.join('V' if y>=ActCeil else 'V' if y<ActFloor else 'A'
                        for y in x)
                        for x in excitation_list]
    elif mode == int:
        return [ [2 if y>ActCeil else 0 if y<ActFloor else 1
                            for y in x]
                            for x in excitation_list]
    else:
        raise NotImplementedError


def fullspin_to_spaceorbital(excitation_list):
    return [[ y//2
            for y in x]
            for x in excitation_list ]

def shift_index_by_n(some_list, n):
    return [ x+n for x in some_list ]


##################################################
### Decomposistion of H1, H2, D1, D2, ... , D4 ###
##################################################


def get_bound_for_active_space(Quket):
    # No need to multiply by 2
    ActFloor = Quket.n_frozen_orbitals + Quket.n_core_orbitals
    ActCeil = ActFloor + Quket.n_active_orbitals
    return ActFloor, ActCeil


def get_bound_for_extended_space(Quket):
    CoreFloor = Quket.n_frozen_orbitals
    ActCeil = CoreFloor + Quket.n_core_orbitals + Quket.n_active_orbitals
    return CoreFloor, ActCeil


def get_spinorbital_H1(Quket, xfloor, xceil):
    ## Swap to this commented out code if spinorbital code is broken due to Htilde
    #if Quket.nc:
    #    HSaa = HSbb = Quket.one_body_integrals[xfloor:xceil, xfloor:xceil]
    #else:
    #    HSaa = HSbb = get_htilde(Quket)[1][xfloor:xceil, xfloor:xceil]
    HSaa = HSbb = get_htilde(Quket)[1][xfloor:xceil, xfloor:xceil]
    return HSaa, HSbb


def get_spinorbital_H2(Quket, xfloor, xceil):
    HSbaba = H2S = Quket.two_body_integrals.transpose(0,2,1,3)[xfloor:xceil, xfloor:xceil, xfloor:xceil, xfloor:xceil]
    HSaaaa = HSbbbb = H2S - H2S.transpose(0,1,3,2)
    return HSaaaa, HSbaba, HSbbbb


def decompose_one_body_intergrals_active(Quket):
    H1 = Quket.one_body_integrals_active
    n_orbitals = int(Quket.n_active_orbitals)
    Haa = zeros((n_orbitals,n_orbitals))
    Hbb = Haa.copy()
    for p,r in product(range(n_orbitals), repeat=2):
        pa = p*2
        pb = pa+1
        ra = r*2
        rb = ra+1#
        Haa[p,r] = H1[pa, ra]
        Hbb[p,r] = H1[pb, rb]
    return Haa, Hbb


def decompose_two_body_intergrals_active(Quket):
    H2 = Quket.two_body_integrals_active
    H2 = 2*(H2.copy().transpose(0,1,3,2) - H2.copy())

    n_orbitals = int(Quket.n_active_orbitals)
    Haaaa = zeros((n_orbitals,n_orbitals,n_orbitals,n_orbitals))
    Hbaba = Haaaa.copy()
    Hbbbb = Haaaa.copy()
    for p,q,r,s in product(range(n_orbitals), repeat=4):
        pa = p*2
        pb = pa+1
        qa = q*2
        qb = qa+1
        ra = r*2
        rb = ra+1
        sa = s*2
        sb = sa+1
        Haaaa[p,q,r,s] = H2[pa,qa,ra,sa]
        Hbaba[p,q,r,s] = H2[pb,qa,rb,sa]
        Hbbbb[p,q,r,s] = H2[pb,qb,rb,sb]
    return Haaaa, Hbaba, Hbbbb


def decompose_1RDM(Quket):
    D1 = get_1RDM_full(Quket.state)
    n_orbitals = int(Quket.n_active_orbitals)
    Daa = zeros((n_orbitals,n_orbitals))
    Dbb = Daa.copy()
    for p,r in product(range(n_orbitals), repeat=2):
        pa = p*2
        pb = pa+1
        ra = r*2
        rb = ra+1
        Daa[p,r] = D1[pa, ra]
        Dbb[p,r] = D1[pb, rb]
    return Daa, Dbb


def decompose_2RDM(Quket):
    D2 = get_2RDM_full(Quket.state)
    n_orbitals = int(Quket.n_active_orbitals)
    Daaaa = zeros((n_orbitals,n_orbitals,n_orbitals,n_orbitals))
    Dbaab = Daaaa.copy()
    Dbbbb = Daaaa.copy()
    for p,q,r,s in product(range(n_orbitals), repeat=4):
        pa = p*2
        pb = pa+1
        qa = q*2
        qb = qa+1
        ra = r*2
        rb = ra+1
        sa = s*2
        sb = sa+1
        Daaaa[p,q,r,s] = D2[pa,qa,ra,sa]
        Dbaab[p,q,r,s] = D2[pb,qa,ra,sb]
        Dbbbb[p,q,r,s] = D2[pb,qb,rb,sb]
    return Daaaa, Dbaab, Dbbbb


def decompose_3RDM(Quket):
    D3 = get_3RDM_full(Quket.state)
    n_orbitals = int(Quket.n_active_orbitals)
    Daaaaaa = zeros((n_orbitals,n_orbitals,n_orbitals,n_orbitals,n_orbitals,n_orbitals))
    Dbaaaab = Daaaaaa.copy()
    Dbbaabb = Daaaaaa.copy()
    Dbbbbbb = Daaaaaa.copy()
    for (p,q,r,t,u,v) in product( range(n_orbitals), repeat=6):
        pa = p*2
        pb = p*2+1
        qa = q*2
        qb = q*2+1
        ra = r*2
        rb = r*2+1
        ta = t*2
        tb = t*2+1
        ua = u*2
        ub = u*2+1
        va = v*2
        vb = v*2+1
        Daaaaaa[p,q,r,t,u,v] = D3[pa,qa,ra,ta,ua,va]
        Dbaaaab[p,q,r,t,u,v] = D3[pb,qa,ra,ta,ua,vb]
        Dbbaabb[p,q,r,t,u,v] = D3[pb,qb,ra,ta,ub,vb]
        Dbbbbbb[p,q,r,t,u,v] = D3[pb,qb,rb,tb,ub,vb]
    return Daaaaaa, Dbaaaab, Dbbaabb, Dbbbbbb


def decompose_4RDM(Quket):
    D4 = get_4RDM_full(Quket.state)
    n_orbitals = int(Quket.n_active_orbitals)
    Daaaaaaaa = zeros((n_orbitals,n_orbitals,n_orbitals,n_orbitals,n_orbitals,n_orbitals,n_orbitals,n_orbitals))
    Dbaaaaaab = Daaaaaaaa.copy()
    Dbbaaaabb = Daaaaaaaa.copy()
    Dbbbaabbb = Daaaaaaaa.copy()
    Dbbbbbbbb = Daaaaaaaa.copy()
    for (p,q,r,s,t,u,v,w) in product( range(n_orbitals), repeat=8):
        pa = p*2
        pb = p*2+1
        qa = q*2
        qb = q*2+1
        ra = r*2
        rb = r*2+1
        sa = s*2
        sb = s*2+1
        ta = t*2
        tb = t*2+1
        ua = u*2
        ub = u*2+1
        va = v*2
        vb = v*2+1
        wa = w*2
        wb = w*2+1
        Daaaaaaaa[p,q,r,s,t,u,v,w] = D4[pa,qa,ra,sa,ta,ua,va,wa]
        Dbaaaaaab[p,q,r,s,t,u,v,w] = D4[pb,qa,ra,sa,ta,ua,va,wb]
        Dbbaaaabb[p,q,r,s,t,u,v,w] = D4[pb,qb,ra,sa,ta,ua,vb,wb]
        Dbbbaabbb[p,q,r,s,t,u,v,w] = D4[pb,qb,rb,sa,ta,ub,vb,wb]
        Dbbbbbbbb[p,q,r,s,t,u,v,w] = D4[pb,qb,rb,sb,tb,ub,vb,wb]
    return Daaaaaaaa, Dbaaaaaab, Dbbaaaabb, Dbbbaabbb, Dbbbbbbbb


####  pre-processing  ##########################################################
####  pre-processing  ##########################################################
####  pre-processing  ##########################################################


def lucc_spinorbital_RDM_preprocessing(Quket):
    import time
    Daa, \
    Dbb = decompose_1RDM(Quket)

    t0 = time.time()
    Daaaa, \
    Dbaab, \
    Dbbbb = decompose_2RDM(Quket)
    t1 = time.time()
    prints('2RDM', t1-t0)

    t0 = time.time()
    Daaaaaa, \
    Dbaaaab, \
    Dbbaabb, \
    Dbbbbbb = decompose_3RDM(Quket)
    t1 = time.time()
    prints('3RDM', t1-t0)

    t0 = time.time()
    Daaaaaaaa, \
    Dbaaaaaab, \
    Dbbaaaabb, \
    Dbbbaabbb, \
    Dbbbbbbbb = decompose_4RDM(Quket)
    t1 = time.time()
    prints('4RDM', t1-t0)

    HD = {
        'Daa':  Daa ,
        'Dbb':  Dbb ,

        'Daaaa':  Daaaa ,
        'Dbaab':  Dbaab ,
        'Dbbbb':  Dbbbb ,
        'Dbaba': -Dbaab.transpose(0,1,3,2) ,
        'Dabab': -Dbaab.transpose(1,0,2,3) ,
        'Dabba':  Dbaab.transpose(1,0,3,2) ,

        'Daaaaaa':  Daaaaaa ,
        'Dbaaaab':  Dbaaaab ,
        'Dbbaabb':  Dbbaabb ,
        'Dbbbbbb':  Dbbbbbb ,

        'Dbaaaba': -Dbaaaab.transpose(0,1,2,3,5,4) ,
        'Dbaabaa':  Dbaaaab.transpose(0,1,2,5,3,4) ,
        'Daabaab':  Dbaaaab.transpose(1,2,0,3,4,5) ,
        'Daababa': -Dbaaaab.transpose(1,2,0,3,5,4)  ,

        'Dbbabab': -Dbbaabb.transpose(0,1,2,4,3,5) ,
        'Dbbabba':  Dbbaabb.transpose(0,1,2,4,5,3) ,
        'Dbababb': -Dbbaabb.transpose(0,2,1,3,4,5) ,
        'Dbabbab':  Dbbaabb.transpose(0,2,1,4,3,5) ,
        'Dbabbba': -Dbbaabb.transpose(0,2,1,4,5,3) ,

        'Daaaaaaaa':  Daaaaaaaa  ,
        'Dbaaaaaab':  Dbaaaaaab  ,
        'Dbbaaaabb':  Dbbaaaabb  ,
        'Dbbbaabbb':  Dbbbaabbb  ,
        'Dbbbbbbbb':  Dbbbbbbbb  ,
        'Dbaaaaaba': -Dbaaaaaab.transpose(0,1,2,3,4,5,7,6) ,
        'Dbaaaabaa':  Dbaaaaaab.transpose(0,1,2,3,4,7,5,6) ,
        'Daaabaaab': -Dbaaaaaab.transpose(1,2,3,0,4,5,6,7) ,
        'Daaabaaba':  Dbaaaaaab.transpose(1,2,3,0,4,5,7,6) ,
        'Dbbaaabab': -Dbbaaaabb.transpose(0,1,2,3,4,6,5,7) ,
        'Dbbaaabba':  Dbbaaaabb.transpose(0,1,2,3,4,6,7,5) ,
        'Dbbaabbaa':  Dbbaaaabb.transpose(0,1,2,3,6,7,4,5) ,
        'Dbaabaabb':  Dbbaaaabb.transpose(0,2,3,1,4,5,6,7) ,
        'Dbaababab': -Dbbaaaabb.transpose(0,2,3,1,4,6,5,7) ,
        'Dbaababba':  Dbbaaaabb.transpose(0,2,3,1,4,6,7,5) ,
        'Dbbbabbab':  Dbbbaabbb.transpose(0,1,2,3,5,6,4,7) ,
        'Dbbbabbba': -Dbbbaabbb.transpose(0,1,2,3,5,6,7,4) ,
        'Dbbababbb': -Dbbbaabbb.transpose(0,1,3,2,4,5,6,7) ,
        'Dbbabbbab': -Dbbbaabbb.transpose(0,1,3,2,5,6,4,7) ,
        'Dbbabbbba':  Dbbbaabbb.transpose(0,1,3,2,5,6,7,4) ,
        
        }

    return HD


def special_lucc_spinorbital_preprocessing(Quket):

    ActFloor, ActCeil = get_bound_for_active_space(Quket)

    # Space: Active_Active => [ froz || cor | act* || vir ]
    Haa, \
    Hbb = get_spinorbital_H1(Quket, ActFloor, ActCeil)

    Haaaa, \
    Hbaba, \
    Hbbbb = get_spinorbital_H2(Quket, ActFloor, ActCeil)

    from numpy import einsum
    import time
    t0 = time.time()
    HD = lucc_spinorbital_RDM_preprocessing(Quket)
    t1 = time.time()
    prints('lucc_spinorbital_RDM_preprocessing time: ',t1-t0)

    HD.update({
            'Haa'  : Haa  ,
            'Hbb'  : Haa ,
            'Haaaa': Haaaa  ,
            'Hbaba': Hbaba  ,
            'Hbbbb': Haaaa  ,
            'Habab': Hbaba.transpose(1,0,3,2) ,
            'Habba': -Hbaba.transpose(1,0,2,3) ,
            'Hbaab': -Hbaba.transpose(0,1,3,2)  ,

        })
    return HD


def general_lucc_spinorbital_preprocessing(Quket):

    CoreFloor, ActCeil = get_bound_for_extended_space(Quket)

    # Space: Active_Core + Active_Active => [ froz || cor* | act* || vir ]
    HSaa, \
    HSbb = get_spinorbital_H1(Quket, CoreFloor, ActCeil)

    HSaaaa, \
    HSbaba, \
    HSbbbb = get_spinorbital_H2(Quket, CoreFloor, ActCeil)

    # For pure Active Space excitations,
    # it must use Haa/Haaaa that is decomposed from integrals_active
    Haa, \
    Hbb = decompose_one_body_intergrals_active(Quket)

    Haaaa, \
    Hbaba, \
    Hbbbb = decompose_two_body_intergrals_active(Quket)

    HD = lucc_spinorbital_RDM_preprocessing(Quket)

    HD.update({
            'Haa'  : Haa  ,
            'Hbb'  : Haa ,
            'Haaaa': Haaaa  ,
            'Hbaba': Hbaba  ,
            'Hbbbb': Haaaa  ,
            'Habab': Hbaba.transpose(1,0,3,2) ,
            'Habba': -Hbaba.transpose(1,0,2,3) ,
            'Hbaab': -Hbaba.transpose(0,1,3,2)  ,


            'HSaa'  : HSaa  ,
            'HSbb'  : HSaa  ,
            'HSaaaa': HSaaaa  ,
            'HSbaba': HSbaba  ,
            'HSbbbb': HSaaaa  ,
            'HSabab': HSbaba.transpose(1,0,3,2)  ,
            'HSabba': -HSbaba.transpose(1,0,2,3)  ,
            'HSbaab': -HSbaba.transpose(0,1,3,2)  ,

        })

    return HD
