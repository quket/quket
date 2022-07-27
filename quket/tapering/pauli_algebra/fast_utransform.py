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
"""
"""
Enhancement of Unitary Transformation for the openfermion.QubitOperator class.
Zero dependency except for openfermion.QubitOperator (because of return value
is in openfermion.QubitOperator format.)

The point is U H U, since U is the same thing,
there must be a chance for mirror image,
i.e xxy yxx
some mirror image will add up
but somw might cancel out and return zero
aaa class   add up 2
            XXX
            YYY
            ZZZ
aba class	add up -2
            XYX
            XZX
            YXY
            YZY
            ZXZ
            ZYZ
aab class	add up 2
            XXY	YXX
            XXZ	ZXX
            XYY	YYX
            YYZ	ZYY
            ZZX	XZZ
            ZZY	YZZ
abc class	dependent on parity
            YZX	XZY
            ZXY	YXZ
            ZYX	XYZ
Say U is [0, 1, 2, 3]
Primarily we need product(U, repeat)
but due to symmetry, we don't need all
Each cycle we run, we truncate leftmost element
For Example
0 X [0, 1, 2, 3]
=> 00 01 02 03
1 X [1, 2, 3]
=> 11 12 13
2 X [2, 3]
=> 22 23
3 X [3]
=> 33
Here we successfully prevent symmetric outcomes
May implemented like this.
U = deque([0, 1, 2, 3])
while len(U):
    lead = U.popleft()
    #print((lead, lead), end=' ')
    for u in U:
        #print((lead, u), end=' ')
(0, 0) (0, 1) (0, 2) (0, 3) (1, 1) (1, 2) (1, 3) (2, 2) (2, 3) (3, 3)
Or reversely
U = [0, 1, 2, 3]
while len(U):
    lead = U.pop()
    #print((lead, lead), end=' ')
    for u in U:
        #print((lead, u), end=' ')
(3, 3) (3, 0) (3, 1) (3, 2) (2, 2) (2, 0) (2, 1) (1, 1) (1, 0) (0, 0)
Or use itertools.combinations() and compute diagonal terms seperately
The ordering of this set of caetesian product should not affect the result
The integrity is protected by commutativity of summation
Analysis of complexity
n = len(U)
Old = O(n^2)
Unique terms of this approach is
(n^2 + n) /2
New = O(0.5 n^2)
But in real world test, we got 20X speed up

Author of this module: TsangSiuChung
"""

from operator import itemgetter
from itertools import product, chain, combinations, repeat
from collections import defaultdict

from openfermion.ops import QubitOperator


qlookup = {  'XXX': ('X', 1),
             'XXY': ('Y', 1),
             'XXZ': ('Z', 1),
             'XYX': ('Y', -1),
             'XYY': ('X', 1),
             'XYZ': ( '', 1j),
             'XZX': ('Z', -1),
             'XZY': ( '', -1j),
             'XZZ': ('X', 1),
             'YXX': ('Y', 1),
             'YXY': ('X', -1),
             'YXZ': ( '', -1j),
             'YYX': ('X', 1),
             'YYY': ('Y', 1),
             'YYZ': ('Z', 1),
             'YZX': ( '', 1j),
             'YZY': ('Z', -1),
             'YZZ': ('Y', 1),
             'ZXX': ('Z', 1),
             'ZXY': ( '', 1j),
             'ZXZ': ('X', -1),
             'ZYX': ( '', -1j),
             'ZYY': ('Z', 1),
             'ZYZ': ('Y', -1),
             'ZZX': ('X', 1),
             'ZZY': ('Y', 1),
             'ZZZ': ('Z', 1),
              'XX': ( '', 1),
              'XY': ('Z', 1j),
              'XZ': ('Y', -1j),
              'YX': ('Z', -1j),
              'YY': ( '', 1),
              'YZ': ('X', 1j),
              'ZX': ('Y', 1j),
              'ZY': ('X', -1j),
              'ZZ': ( '', 1),
               'X': ('X', 1),
               'Y': ('Y', 1),
               'Z': ('Z', 1),
                '': ( '', 1)}


qlookup_xyz = {  'XXX': 'X',
                 'XXY': 'Y',
                 'XXZ': 'Z',
                 'XYX': 'Y',
                 'XYY': 'X',
                 'XYZ': '',
                 'XZX': 'Z',
                 'XZY': '',
                 'XZZ': 'X',
                 'YXX': 'Y',
                 'YXY': 'X',
                 'YXZ': '',
                 'YYX': 'X',
                 'YYY': 'Y',
                 'YYZ': 'Z',
                 'YZX': '',
                 'YZY': 'Z',
                 'YZZ': 'Y',
                 'ZXX': 'Z',
                 'ZXY': '',
                 'ZXZ': 'X',
                 'ZYX': '',
                 'ZYY': 'Z',
                 'ZYZ': 'Y',
                 'ZZX': 'X',
                 'ZZY': 'Y',
                 'ZZZ': 'Z',
                  'XX': '',
                  'XY': 'Z',
                  'XZ': 'Y',
                  'YX': 'Z',
                  'YY': '',
                  'YZ': 'X',
                  'ZX': 'Y',
                  'ZY': 'X',
                  'ZZ': '',
                   'X': 'X',
                   'Y': 'Y',
                   'Z': 'Z',
                    '': '' }


qlookup_cof = {  'XXX': 1,
                 'XXY': 1,
                 'XXZ': 1,
                 'XYX': -1,
                 'XYY': 1,
                 'XYZ': 1j,
                 'XZX': -1,
                 'XZY': -1j,
                 'XZZ': 1,
                 'YXX': 1,
                 'YXY': -1,
                 'YXZ': -1j,
                 'YYX': 1,
                 'YYY': 1,
                 'YYZ': 1,
                 'YZX': 1j,
                 'YZY': -1,
                 'YZZ': 1,
                 'ZXX': 1,
                 'ZXY': 1j,
                 'ZXZ': -1,
                 'ZYX': -1j,
                 'ZYY': 1,
                 'ZYZ': -1,
                 'ZZX': 1,
                 'ZZY': 1,
                 'ZZZ': 1,
                  'XY': 1j,
                  'YX': -1j,
                  'XZ': -1j,
                  'ZX': 1j,
                  'YZ': 1j,
                  'ZY': -1j,
                   'X': 1,
                   'Y': 1,
                   'Z': 1,
                    '': 1 }


qlookup_aclw = {  'XXX' : None,
                  'XXY' : None,
                  'XXZ' : None,
                  'XYX' :   -1,
                  'XYY' : None,
                  'XYZ' :    0,
                  'XZX' :   -1,
                  'XZY' :    1,
                  'XZZ' : None,
                  'YXX' : None,
                  'YXY' :   -1,
                  'YXZ' :    1,
                  'YYX' : None,
                  'YYY' : None,
                  'YYZ' : None,
                  'YZX' :    0,
                  'YZY' :   -1,
                  'YZZ' : None,
                  'ZXX' : None,
                  'ZXY' :    0,
                  'ZXZ' :   -1,
                  'ZYX' :    1,
                  'ZYY' : None,
                  'ZYZ' :   -1,
                  'ZZX' : None,
                  'ZZY' : None,
                  'ZZZ' : None,
                   'XX' : None,
                   'XY' :    0,
                   'XZ' :    1,
                   'YX' :    1,
                   'YY' : None,
                   'YZ' :    0,
                   'ZX' :    0,
                   'ZY' :    1,
                   'ZZ' : None,
                    'X' : None,
                    'Y' : None,
                    'Z' : None,
                     '' : None }


def q3lookup_gen():

    qlookup = {'XY':('Z',1j),
                'YX':('Z',-1j),
                'XZ':('Y',-1j),
                'ZX':('Y',1j),
                'YZ':('X',1j),
                'ZY':('X',-1j),
                'XX':('',1),
                'YY':('',1),
                'ZZ':('',1)
                }

    qlookup_xyz = {'XY':'Z',
                    'YX':'Z',
                    'XZ':'Y',
                    'ZX':'Y',
                    'YZ':'X',
                    'ZY':'X',
                    'X':'X',
                    'Y':'Y',
                    'Z':'Z'
                    }

    qlookup_cof = {'XY':1j,
                    'YX':-1j,
                    'XZ':-1j,
                    'ZX':1j,
                    'YZ':1j,
                    'ZY':-1j,
                    'X':1,
                    'Y':1,
                    'Z':1
                    }


    new_ref = []
    new_ref_xyz = []
    new_ref_cof = []

    for a,b,c in product(('X','Y','Z'), repeat=3):

        if a==c:
            newkey = b
            if a==b:
                cof = 1
            else:
                cof = -1

        elif a==b:
            newkey = c
            cof = 1

        elif b==c:
            newkey = a
            cof = 1

        else:
            newkey, cof = qlookup[f"{a}{b}"]

            if newkey != c:
                c = qlookup_xyz[f"{newkey}{c}"]
                cof *= qlookup_cof[f"{newkey}{c}"]
            else:
                newkey = ''

        old_key = ''.join((a,b,c))

        new_ref.append( (old_key,(newkey, cof)) )
        new_ref_xyz.append(  (old_key, newkey)  )
        new_ref_cof.append( (old_key, cof )  )

    return dict(new_ref), dict(new_ref_xyz),  dict(new_ref_cof)


def how_to_compute_real_parity(n=4, verbose=True):
    from numpy import array, zero, where
    from itertools import product

    hollow = zeros((n,n)).tolist()

    parity_set = set()
    parity_set3 = set()

    print(' ______________________________________________________________________________________________')
    print(f"|  clw  aclw   original equation   a=clw%2+aclw%2   b=(clw+aclw)%4   (a+b)%4  finalresult  |")
    print('|______________________________________________________________________________________________|')
    for clw,aclw in product(range(n), repeat=2):

        parity = ((-1)**((aclw)%2) * 1j**((clw + aclw)%4) + (-1)**((clw)%2) * 1j**((clw + aclw)%4))

        parity_set.add((  (aclw%2^clw%2), parity)  )

        if not aclw%2^clw%2:

            ai = (-1)**(aclw%2) + (-1)**(clw%2)
            bi = 1j**(clw%4+aclw%4)

            a2i =  aclw%2 + clw%2
            b2i = (clw + aclw)%4

            ci = ((aclw%2 + clw%2) + (clw + aclw)%4)//2%2

            parity_set3.add((ci, parity))
            hollow[clw][aclw] = parity

            if verbose:
                idx = f"{clw:>3} {aclw:3}"


                if not aclw%2-clw%2:
                    if not parity: print(f"this is strange.")
                    print(f"|{idx:>8} {ai:>8} * {bi:>8} {a2i:>10} {b2i:>18} {ci:>18} {parity:>12}    |")
    print('|______________________________________________________________________________________________|')
    print('\n                                aclw%2^clw%2 = ', parity_set)
    print('((aclw%2 + clw%2) + (clw + aclw)%4)//2%2 = ', parity_set3)

    return array(hollow)


def fast_utransform_special(H:QubitOperator, U:QubitOperator) -> QubitOperator:
    """
    Unitary transformation for special case which
    U * H * U
    U contains only one single term.
    Args:
        H (QubitOperator): Hamiltonian.
        U (QubitOperator): Unitary operators to do UHU.
    Return:
        newH (QubitOperator): Result of UHU.
    """
    U_key, U_cof = tuple(U.terms.items())[0]
    U_key = dict(U_key)
    U_cof *= U_cof  # U coefficient of UHU

    H_working = H.terms

    H_constant = 0
    if () in H_working:
        H_constant += H_working.pop(())

    for hkey, hcof in H_working.items():

        h_working = dict(hkey)

        intersect = set(h_working.keys()).intersection(U_key)
        if intersect:
            parity = 0
            for i in intersect:
                if h_working[i] != U_key[i]:
                    parity ^= 1
            if parity:
                hcof *= -1

        H_working[hkey] = hcof * U_cof

    if H_constant:
        H_working[()] = H_constant

    QO = QubitOperator
    new = QO()
    for x in H_working.items():
        new += QO(*x)

    return new


def fast_utransform_general(H:QubitOperator, U:QubitOperator, debug=0) -> QubitOperator:
    """
    Unitary transformation for general case which
    U * H * U
    U contains multiple terms.
    Args:
        H (QubitOperator): Hamiltonian.
        U (QubitOperator): Unitary operators to do UHU.
    Return:
        newH (QubitOperator): Result of UHU.
    Carry out unitary transformation directly upon XYZ string representation.
    Will be slower.
    Using parity check.
    Psuedocode:
        1. get one U from the U_list
        2. loop for each element in H
        3. get u_left from U
        4. compute UH, store computed information for lateral reuse
        5. get another u_right from U
        6. if u_right == u_left, we can reuse the previous UH result
        7. compute UHU
        8. update new_operator
        9. repeat from 2. until end of all combinations of UHU
        10. repeat from 1. until end of all U_list
    """
    Qaclw, Qxyz = qlookup_aclw, qlookup_xyz

    U_item = [ (i,dict(k), set(map(itemgetter(0), k)), v) \
              for i,(k,v) in enumerate(U.terms.items()) ]

    new = defaultdict(float)

    for hkey, hcof in H.terms.items():
        hkey, hidx, intersectBook = dict(hkey), frozenset(map(itemgetter(0),hkey)), dict()

        limit = 0
        while limit < len(U_item):
            #=== For Diagonal Terms U_leff == U_righ ===#
            i, lkey, lidx, lcof = U_item[limit]
            newcof = uh_cof = hcof*lcof
            basekey = hkey.copy()
            basekey.update(lkey)

            if i in intersectBook: uh_intersect = intersectBook[i]
            else: intersectBook[i] = uh_intersect = hidx & lidx

            for bit in uh_intersect:
                if lkey[bit] != hkey[bit]: newcof = -newcof

            if tuple(hkey.items()) in new:
                new[tuple(hkey.items())] += newcof*lcof
            else:
                new[tuple(hkey.items())] = newcof*lcof

            if debug:
                print('\n', (i,i))
                zeros = [' ' for x in range(14)]
                build = [zeros[:], zeros[:], zeros[:]]
                for iii, D in enumerate((lkey.items(), hkey.items(), lkey.items())):
                    for k,v in D:
                        build[iii][k] = v
                print(array(build))
                print(tuple(hkey.items()), new.get(tuple(hkey.items()), 0))

            #=== For OFF Diagonal Terms U_leff != U_righ ===#
            limit += 1
            for j, rkey, ridx, rcof in U_item[limit:]:
                newcof = 2*uh_cof*rcof

                if j in intersectBook: hu_intersect = intersectBook[j]
                else: intersectBook[j] = hu_intersect = hidx & set(rkey.keys())

                uu_intersect, uhu_intersect = lidx&ridx, uh_intersect&hu_intersect

                newkey = basekey.copy()
                newkey.update(rkey)
                clw = aclw = 0

                for bit in uh_intersect^uhu_intersect:  # uh
                    if lkey[bit] != hkey[bit]:
                        xyz = f"{lkey[bit]}{hkey[bit]}"
                        newkey[bit] = Qxyz[xyz]
                        if Qaclw[xyz]:
                            aclw += 1
                        else: clw += 1
                    else: newkey[bit] = None

                for bit in hu_intersect^uhu_intersect:  # hu
                    if hkey[bit] != rkey[bit]:
                        xyz = f"{hkey[bit]}{rkey[bit]}"
                        newkey[bit] = Qxyz[xyz]
                        if Qaclw[xyz]:
                            aclw += 1
                        else: clw += 1
                    else: newkey[bit] = None

                for bit in uu_intersect-hidx:  # uu
                    if lkey[bit] != rkey[bit]:
                        xyz = f"{lkey[bit]}{rkey[bit]}"
                        newkey[bit] = Qxyz[xyz]
                        if Qaclw[xyz]:
                            aclw += 1
                        else: clw += 1
                    else: newkey[bit] = None

                for bit in uhu_intersect:
                    ab, bc, ac = lkey[bit]==hkey[bit], hkey[bit]==rkey[bit], lkey[bit]==rkey[bit]
                    if ab:
                        if bc:
                            newkey[bit] = hkey[bit]  # aaa
                        else: newkey[bit] = rkey[bit]  # aab
                    elif bc: newkey[bit] = lkey[bit]  # baa
                    elif ac: newkey[bit] = hkey[bit]; newcof = -newcof  # aba
                    else:  # abc
                        if Qaclw[f"{hkey[bit]}{rkey[bit]}"]:
                            aclw += 1
                        else: clw += 1
                        newkey[bit] = None

                if not aclw%2 ^ clw%2:
                    if ((aclw%2 + clw%2) + (clw + aclw)%4)//2%2:
                        newkey = tuple(sorted( ((bit,xyz) for bit,xyz in newkey.items() if xyz) ,
                                                key=itemgetter(0) ))
                        new[newkey] -= newcof
                    else:
                        newkey = tuple(sorted( ((bit,xyz) for bit,xyz in newkey.items() if xyz) ,
                                                key=itemgetter(0) ))
                        new[newkey] += newcof

                    if debug and newkey == ((1,'X'),(2,'Y'),(3,'Y'),(5,'Z'),(6,'Z'),(9,'Z'),(10,'Z'),(13,'Z')):
                        print('\n', (i,j))
                        zeros = [' ' for x in range(14)]
                        build = [zeros[:], zeros[:], zeros[:]]
                        for iii, D in enumerate((lkey.items(), hkey.items(), rkey.items())):
                            for k,v in D:
                                build[iii][k] = v
                        print(array(build))

                        print('HERE',  clw, aclw, ((aclw + clw)%2 + (clw + aclw)%4)//2%2)
                        print(newkey, newcof, new.get(newkey, 0))

    QO = QubitOperator
    new_qo = QO()
    for k,v in new.items():
        if v: new_qo += QO(k,v)
    new_qo.compress()
    return new_qo


def binary_pauli_utransform(H):
    """
    Bidirectional QubitOperator Binarylizer
    (QubitOperator) -> Generator
    (Dict) -> QubitOperator
    """
    if isinstance(H, QubitOperator):
        return (tuple((dict( (b,1) if x=='X' else \
                              (b,2) if x=='Y' else \
                              (b,3) if x=='Z' else x for b,x in k ),v)
                for k,v in H.terms.items()))


    elif isinstance(H, dict):
        QO = QubitOperator
        new = QO()
        if () in H:
            new += QO((), H[()])
            del H[()]

        for k,v in H.items():
            new += QO(
                      tuple(
                             (b,'X') if x==1 else \
                             (b,'Y') if x==2 else \
                             (b,'Z') if x==3 else x for b,x in k
                            ), v)
        new.compress()
        return new


def fast_utransform_general_from_iterable(H:QubitOperator, Us:[QubitOperator], debug=0) -> QubitOperator:
    """
    Serial Unitary transformation for general case which
    [U] * H * [U]
    [U] is a list and each element contains multiple terms.
    Args:
        H (QubitOperator): Hamiltonian.
        Us (1dlist QubitOperator): List of Unitary operators to do UHU.
    Return:
        newH (QubitOperator): Result of UHU.
    First convert the all Operator into binary representation,
    then carry out unitariy transformation.
    Finally convert back to ordinary XYZ string representation.
    Using parity check.
    Using 123cyclic index clock system i.e. L%3+1 == R
    Psuedocode:
        1. get one U from the U_list
        2. loop for each element in H
        3. get u_left from U
        4. compute UH, store computed information for lateral reuse
        5. get another u_right from U
        6. if u_right == u_left, we can reuse the previous UH result
        7. compute UHU
        8. update new_operator
        9. repeat from 2. until end of all combinations of UHU
        10. repeat from 1. until end of all U_list
    """
    if debug:
        from numpy import array, zero

    count = 0
    H = binary_pauli_utransform(H)

    for U in Us:
        U = tuple(enumerate(binary_pauli_utransform(U)))
        new = defaultdict(float)

        for hkey, hcof in H:
            hidx, intersectBook = set(hkey.keys()), dict()

            limit = 0
            while limit < len(U):
                #=== For Diagonal Terms U_leff == U_righ ===#
                i,(lkey, lcof) = U[limit]

                basekey = hkey.copy()
                basekey.update(lkey)
                newcof = uh_cof = hcof*lcof

                lidx = set(lkey.keys())
                if i in intersectBook: uh_intersect = intersectBook[i]
                else: intersectBook[i] = uh_intersect = hidx & lidx

                for bit in uh_intersect:
                    if lkey[bit] != hkey[bit]: newcof = -newcof

                if tuple(hkey.items()) in new:
                    new[tuple(hkey.items())] += newcof*lcof
                else:
                    new[tuple(hkey.items())] = newcof*lcof

                if debug:
                    print('\n', (i,i))
                    zeros = [' ' for x in range(14)]
                    build = [zeros[:], zeros[:], zeros[:]]
                    for iii, D in enumerate((lkey.items(), hkey.items(), lkey.items())):
                        for k,v in D:
                            build[iii][k] = v
                    print(array(build))
                    print(tuple(hkey.items()), new.get(tuple(hkey.items()), 0))

                #=== For OFF Diagonal Terms U_leff != U_righ  ===#
                limit += 1
                for j,(rkey, rcof) in U[limit:]:
                    newcof = 2*uh_cof*rcof
                    newkey = basekey.copy()
                    newkey.update(rkey)
                    ridx = set(rkey.keys())

                    if j in intersectBook: hu_intersect = intersectBook[j]
                    else: intersectBook[j] = hu_intersect = hidx & ridx

                    uu_intersect, uhu_intersect = lidx&ridx, uh_intersect&hu_intersect
                    clw = aclw = 0

                    for bit in uh_intersect^uhu_intersect:  # uh
                        l, r = lkey[bit], hkey[bit]
                        if l != r:
                            newkey[bit] = l ^ r
                            if l%3+1 == r:
                                clw += 1
                            else: aclw += 1
                        else: newkey[bit] = None

                    for bit in hu_intersect^uhu_intersect:  # hu
                        l, r = hkey[bit], rkey[bit]
                        if l != r:
                            newkey[bit] = l ^ r
                            if l%3+1 == r:
                                clw += 1
                            else: aclw += 1
                        else: newkey[bit] = None

                    for bit in uu_intersect-uhu_intersect:  # uu
                        l, r = lkey[bit], rkey[bit]
                        if l != r:
                            newkey[bit] = l ^ r
                            if l%3+1 == r:
                                clw += 1
                            else: aclw += 1
                        else: newkey[bit] = None

                    for bit in uhu_intersect:  # uhu
                        ab, bc, ac = lkey[bit]==hkey[bit], hkey[bit]==rkey[bit], lkey[bit]==rkey[bit]
                        if ab: # if ab
                            if bc: newkey[bit] = hkey[bit]  # aaa
                            else: newkey[bit] = rkey[bit]  # aab
                        elif bc: newkey[bit] = lkey[bit]  # baa
                        elif ac: newkey[bit] = hkey[bit]; newcof = -newcof  # aba
                        else:  # abc
                            if hkey[bit]%3+1 == rkey[bit]:
                                clw += 1
                            else: aclw += 1
                            newkey[bit] = None

                    if not aclw%2 ^ clw%2:
                        if ((aclw%2 + clw%2) + (clw + aclw)%4)//2%2:  # Caution! (aclw%2+clw%2) != (aclw+clw)%2
                            newkey = tuple(sorted( ((bit,xyz) for bit,xyz in newkey.items() if xyz) ,
                                            key=itemgetter(0) ))
                            new[newkey] -= newcof
                        else:
                            newkey = tuple(sorted( ((bit,xyz) for bit,xyz in newkey.items() if xyz) ,
                                            key=itemgetter(0) ))
                            new[newkey] += newcof

                        #=== Note for debug ===#
                        # Compare UHU with this function
                        # Do UHU - fast_utransform_general
                        # If there is any remainder, pick one
                        # Print info when newkey == picked remainder
                        # Maybe do the uhu by hand to check whether it produce the right coefficient
                        # If differ by sign:
                        #     first check if the condition is exactly this
                        #         ((aclw%2 + clw%2) + (clw + aclw)%4)//2%2 -> negative
                        #     see how_to_compute_real_parity() for the reason
                        #     then check whether {acw, aclw} is computed right
                        #         XY or XYZ or likewise ->clw,clockwise
                        #         XZ or XYZ or likewise ->aclw,anticlockwise
                        #     make sure it is -> L%3+1 == R <- for clockwise
                        #     make sure you don't miss -> newcof = -newcof <- when XYX or ZYZ etc.
                        # If product of pauli XYZ is not right:
                        #     like  XY = X which is not
                        #     check if -> newkey[bit] = lkey[bit] <- etc. is doing right
                        #     check if newkey is throwing away -> newkey[bit] = None <- terms
                        # If coefficient is not right:
                        #     check if -> newcof = 2*uh_cof*rcof <- etc. is doing right
                        if debug and newkey == ((1,1),(2,2),(3,2),(5,3),(6,3),(9,3),(10,3),(13,3)):
                            print('\n', (i,j))
                            zeros = [' ' for x in range(14)]
                            build = [zeros[:], zeros[:], zeros[:]]
                            for iii, D in enumerate((lkey.items(), hkey.items(), rkey.items())):
                                for k,v in D:
                                    build[iii][k] = v
                            print(array(build))

                            print('HERE',  clw, aclw, ((aclw + clw)%2 + (clw + aclw)%4)//2%2)
                            print(newkey, newcof, new.get(newkey, 0))

        if count < len(Us):
            H = ( (dict(k),v) for k,v in new.items() if v )
            count += 1

    return binary_pauli_utransform({ k:v for k,v in new.items() if v})