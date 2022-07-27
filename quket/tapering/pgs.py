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

pgs.py

Handling Point Group Symmetry.

"""
from operator import itemgetter
from itertools import compress, zip_longest, filterfalse, starmap
from copy import deepcopy
from math import log2, ceil, floor
from statistics import stdev, mean, median
from collections.abc import Iterable
from collections import deque
import time

import numpy as np

from openfermion.utils import commutator

from quket import config as cf
from quket.fileio import prints


### Point Group Symmetry
def get_pointgroup_character_table(Mol, groupname, irrep_name, symm_orb, mo_coeff):
    '''Function
    Find the set of Eigenvalues of molecular symmetry operations based on 
    Point Group Symmetry(PGS) for Z2 symmetry qubit tapering.
    Harnessing the module Pyscf.symm, Pyscf is executed in this function 
    in order to extract information about PGS.
    Args:
        geometry (str): Molecule geometry in Pyscf formalism for construction of the molecule.
        subgroup (str): Subgroup of point group to be assigned to the molecule, default is True.
    Returns:
    if Abelian subgroup:
        (in tuple)
        operations_names (str 1darray): List of symmetry operations of the molecule.
        characters_list (int ndarray): Array of Eigenvalues corresponds to each 
                                       symmetry operation for symmetry-adapted orbital.
    else:
        None, None
        
    Author(s): TsangSiuChung, Takashi Tsuchimochi
    Example:
    h2o = 'O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0'
    get_pointgroup_character_table(h2o)
    >>> Point group = C2v
        Subgroup = C2v(Abelian)
        converged SCF energy = -74.9629466565383
        converged SCF energy = -74.9629466565383
        RHF = -74.9629466565383
        Eigenvalues of each irreducible representation of the symmetry adapted orbital corresponding to each symmetry operation:
        C2v | SymmOP \ IRREP    A1   A1   A1   A1   B1   B1   A1   A1   B2   B2   A1   A1   B1   B1
                C2x               1    1    1    1   -1   -1    1    1   -1   -1    1    1   -1   -1
                Sigv^xz           1    1    1    1    1    1    1    1   -1   -1    1    1    1    1
                Sigv^yz           1    1    1    1   -1   -1    1    1    1    1    1    1   -1   -1
    '''
    from pyscf import symm
    # Excape function if it is in non abelian subgroups
    # Acknoledgement: http://openmopac.net/manual/point_group_identification.html
    abelian_groups =['C1', 'C2', 'Ci', 'Cs', 'C2v', 'C2h', 'D2', 'D2h']
    if groupname in abelian_groups:
        pass
    elif groupname == 'C1':
        prints('''No available solutions for tapering non Z2 symmetric group.
                PGS tapering abandoned.''')
        return None, None,None
    else:
        prints("(non Abelian)")
        prints('''No available solutions for tapering non Z2 symmetric group.
                PGS tapering abandoned.''')
        return None,None,None

    # Excape function if we don't have this subgroup in the dictionary
    if groupname not in point_group_character_dict:
        prints('''Lacking information about this subgroup.
                PGS tapering abandoned.''')
        return None,None,None
    
    # Execute function only if we have this subgroup in the dictionary
    else:
        # Labeling each symmetry adapted orbital by
        # its irreducible representation of point-group(IRREP) 
        #   when arg#2 = mol.irrep_name, returns irrep_name
        #   when arg#2 = mol.irrep_id, returns irrep_id       
        IRREP_name_list = symm.label_orb_symm(Mol, irrep_name, symm_orb, mo_coeff)

        #symm.eigh( , molecular_orbital_symmetry) # Maybe useful but need investagation

        # Extract point group symmetry information of current molucule
        # from the dictionary "point_group" 
        target_point_group = point_group_character_dict[groupname] 

        # Construct the list of eigenvalues of each symmetry operation of the point group
        # Separate characters and their corresponding symmetry operation names
        
        operations_names = []
        characters_list = []
        for (k, symm_operation) in enumerate(target_point_group["symmetryOperation"][1:]): # Omitting the A1 irrep since it is useless
            temp = []
            for element in IRREP_name_list:
                op_id = k+1 # k+1 for skipping identity operation since totally useless
                character = target_point_group[element][op_id]
                # Append character to temporary list
                # Using for loop accounts for spin=2
                temp.append(character) 
                temp.append(character)         
            operations_names.append(symm_operation)        
            characters_list.append(temp)

        # Repeat twice acounts for spin=2 (# of orbital is doubled!)
        temp = []
        for i,name in enumerate(IRREP_name_list):
            temp.append(name)
            temp.append(name)
        IRREP_name_list = temp
        return operations_names, IRREP_name_list, characters_list


def get_total_symmetric_excite(excite_list, irrep_list):
    '''Function
    Extract simmetry preserving excitations from the provide excite list.

    Args:
        excite_list (int 2darray): List of excitation.
        irrep_list (str 1darray): Irrep name for the orbitals.

    Returns:
        total_symmeric_V (int 2darray): List of total symmetric excitation.

    Author(s): Takashi Tsuchimochi, TsangSiuChung
    '''
    # See the independent items in irrep_list
    irrep_set = sorted(list(set(irrep_list)))
    # Sort the orbital indices by irrep
    nirreps = len(irrep_list)
    index_irrep = [(i,o) for i,o in enumerate(irrep_list)]
    index_irrep.sort(key=itemgetter(1))
    # A dictionary for checking set relations
    # {irrep : orbital indices}
    irrep2index_dict = dict( (k, frozenset(map(itemgetter(0), g))) 
            for k,g in groupby(index_irrep, itemgetter(1)) )
    # Create another dict for quick reference
    # {orbital indices : irrep}
    index2irrep_dict = dict(index_irrep)

    #prints(index_irrep)  ## For debug
    #prints(irrep2index_dict)  ## For debug
    #prints(index2irrep_dict)  ## For debug

    total_symmeric_excite = []
    for i,row in enumerate(excite_list):
        ncols = len(row)
        # Prepare things for set relation checking
        if ncols == 2:
            crea_anni = [row]
        elif ncols > 2 and ncols%2==0:
            mid_point = int(ncols/2)
            # Slice the excite tuple in half and contain in a list
            crea_anni = [row[0: mid_point], row[mid_point: ]]

        jumped_out = 0
        for x in crea_anni:
            # Get irrep of the first element
            irrep = index2irrep_dict[x[0]]
            # See if the whole thing is under same irrep,
            # otherwise skip current row
            if not irrep2index_dict[irrep].issuperset(x):
                jumped_out = 1
                break
        if jumped_out:
            continue
        total_symmeric_excite.append(tuple(row))
    return total_symmeric_excite
    
    
### Dictionary ###
point_group_character_dict = {
    "Cs":{
        "symmetryOperation":('E','Sigh'),
        "h":2,
        "A\'":(1,1),
        "A\"":(1,-1)},
    "Ci":{
        "symmetryOperation":('E','i'),
        "h":2,
        "Ag":(1,1),
        "Au":(1,-1)},
    "C1":{
        "symmetryOperation":('E'),
        "h":1,
        "A":(1)},
    "C2":{
        "symmetryOperation":('E','C2^z'),
        "h":1,
        "A":(1,1),
        "B":(1,-1)},      
    "C2v":{
        "symmetryOperation":('E','C2x','Sigv^xz','Sigv^yz'),
        "h":4,
        "A1":(1,1,1,1),
        "A2":(1,1,-1,-1),
        "B1":(1,-1,1,-1),
        "B2":(1,-1,-1,1)},
    "C3v":{
        "symmetryOperation":('E','2C3','3Sigv^'),
        "h":6,
        "A1":(1,1,1),
        "A2":(1,1,-1),
        "E":(2,-1,0)},
    "C4v":{
        "symmetryOperation":('E','C2','2C4','2Sigv^','2Sigvd^'),
        "h":8,
        "A1":(1,1,1,1,1),
        "A2":(1,1,1,-1,-1),
        "B1":(1,1,-1,1,-1),
        "B2":(1,1,-1,-1,1),
        "E":(2,-2,0,0,0)},
    "D2":{
        "symmetryOperation":('E','C2^z','C2^y','C2^x'),
        "h":4,
        "A":(1,1,1,1),
        "B1":(1,1,-1,-1),
        "B2":(1,-1,1,-1),
        "B3":(1,-1,-1,1)},
    "D3":{
        "symmetryOperation":('E','2C3','3C2'),
        "h":6,
        "A1":(1,1,1),
        "A2":(1,1,-1),
        "E":(2,-1,0)},
    "D4":{
        "symmetryOperation":('E','C2','2C4','2C2\'','2C2\'\''),
        "h":8,
        "A1":(1,1,1,1,1),
        "A2":(1,1,1,-1,-1),
        "B1":(1,1,-1,1,-1),
        "B2":(1,1,-1,-1,1),
        "E":(2,-1,0,0,0)},
    "C2h":{
        "symmetryOperation":('E','C2^z','i','Sigh'),
        "h":4,
        "Ag":(1,1,1,1),
        "Bg":(1,-1,1,-1),
        "Au":(1,1,-1,-1),
        "Bu":(1,-1,-1,1)},
    "D2h":{
        "symmetryOperation":('E','C2^z','C2^y','C2^x','i','Sig^xy','Sig^xz','Sig^yz'),
        "h":8,
        "Ag":(1,1,1,1,1,1,1,1),
        "B1g":(1,1,-1,-1,1,1,-1,-1),
        "B2g":(1,-1,1,-1,1,-1,1,-1),
        "B3g":(1,-1,-1,1,1,-1,-1,1),
        "Au":(1,1,1,1,-1,-1,-1,-1),
        "B1u":(1,1,-1,-1,-1,-1,1,1),
        "B2u":(1,-1,1,-1,-1,1,-1,1),
        "B3u":(1,-1,-1,1,-1,1,1,-1)},
    "D2d":{
        "symmetryOperation":('E','C2','2C2\'','2S4','2Sigd'),
        "h":8,
        "A1":(1,1,1,1,1),
        "A2":(1,1,-1,1,-1),
        "B1":(1,1,1,-1,-1),
        "B2":(1,1,-1,-1,1),
        "E":(2,-2,0,0,0)}
    }
