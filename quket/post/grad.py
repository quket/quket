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

grad.py

Main driver of nuclear gradient.
Essentially using pyscf implementation.

"""
import numpy as np
from functools import reduce
from pyscf.grad import rhf as grad
from pyscf import lib, scf
from pyscf.grad.mp2 import _shell_prange
from quket.mpilib import mpilib as mpi
from quket import config as cf
from quket.utils import to_pyscf_geom, chk_energy
from quket.fileio import prints, printmat, print_geom, print_grad
from quket.fileio import set_config

def nuclear_grad(Quket):
    """Function
    Compute nuclear gradients.

    Author(s): Takashi Tsuchimochi
    """
    ### Get relaxed 1RDM  
    if not chk_energy(Quket):
        Quket.get_1RDM()
    de = None
    if mpi.main_rank:
        mol = Quket.pyscf
        mo_coeff = Quket.mo_coeff
        nao, nmo = mo_coeff.shape
        ncore = Quket.n_frozen_orbitals
        nact = Quket.n_active_orbitals
        atmlst = range(mol.natm)
        aoslices = mol.aoslice_by_atom()

        # Prepare gradient matrix
        de_1 = np.zeros((len(atmlst),3))   # One-body h1(x)
        de_2 = np.zeros_like(de_1)         # Two-body h2(x)
        de_w = np.zeros_like(de_1)         # Overlap  S(x)

        ### HF dm in AO basis 
        hf_dm1 = mo_coeff[:,:ncore+Quket.noa] @ mo_coeff[:,:ncore+Quket.noa].T \
               + mo_coeff[:,:ncore+Quket.nob] @ mo_coeff[:,:ncore+Quket.nob].T 
        ### Core dm in AO basis
        core_dm1 = 2* mo_coeff[:,:ncore] @ mo_coeff[:,:ncore].T
        ### dm1 in AO basis
        dm1 = Quket.DA + Quket.DB 
        dm1 = mo_coeff @ dm1 @ mo_coeff.T
        dm1_rel = Quket.RelDA + Quket.RelDB
        dm1_rel = mo_coeff @ dm1_rel @ mo_coeff.T
        dm1act = dm1 - core_dm1
        ### Response correction
        zeta = dm1_rel - dm1
        ### dm2 in AO basis (active)
        dm2 = Quket.Daaaa + Quket.Dbbbb + Quket.Dbaab + Quket.Dbaab.transpose(1,0,3,2)
        dm2 = np.einsum('pi,ijkl->pjkl', mo_coeff[:,ncore:ncore+nact], dm2)
        dm2 = np.einsum('pj,ijkl->ipkl', mo_coeff[:,ncore:ncore+nact], dm2)
        dm2 = np.einsum('pk,ijkl->ijpl', mo_coeff[:,ncore:ncore+nact], dm2)
        dm2 = np.einsum('pl,ijkl->ijkp', mo_coeff[:,ncore:ncore+nact], dm2)
        ### We have transformed
        ### dm2[p,q,r,s] = <p^ q^ r s>
        ### to AO. We need frozen-core orbitals. 
        for mu in range(nao):
            for nu in range(nao):
                for lam in range(nao):
                    for sig in range(nao):
                        dm2[mu, nu, lam, sig] += + dm1act[mu,sig] * core_dm1[nu, lam]  \
                                                 - dm1act[mu,lam] * core_dm1[nu, sig] /2 \
                                                 - dm1act[nu,sig] * core_dm1[mu, lam] /2 \
                                                 + dm1act[nu,lam] * core_dm1[mu, sig]  \
                                                 + core_dm1[mu,sig] * core_dm1[nu, lam] \
                                                 - core_dm1[mu,lam] * core_dm1[nu, sig] /2 
        ### Now dm2 contains the unrelaxed part of 2PDM (without zeta contributions)

        ### Change
        ### dm2[p,q,r,s] = <p^ q^ r s>
        ### to 
        ### dm2[p,s,q,r] = <p^ q^ r s>
        ### to comply with eri[p,r,q,s] = (pr|qs) = <pq|rs>
        dm2 = dm2.transpose(0,3,1,2).copy()
        dm2_ = np.zeros((nao, nao, nao*(nao+1)//2))
        ### eri[p,r,q,s] = (pr|qs) 
        ### dm2[p,r,q,s] = <p^ q^ s r>
        pq = 0
        for p in range(nao):
            for q in range(p+1):
                dm2_[:,:,pq] = 0.5*(dm2[:,:,p,q] + dm2[:,:,q,p])
                pq += 1

        h1 = scf.hf.get_hcore(mol)
        ### Compute energy in AO ###
        #e1 = np.einsum('pq,pq',h1,dm1) 
        #eri0 = mol.intor('int2e', aosym='s1')
        #e2 = np.einsum('pqrs,pqrs',eri0,dm2) 
        #print('e1 ',e1)
        #print('e2 ',e2/2)
        #print('Enuc ',Quket.nuclear_repulsion)

        offsetdic = mol.offset_nr_by_atom()
        diagidx = np.arange(nao)
        diagidx = diagidx*(diagidx+1)//2 + diagidx
        edm = np.zeros((nao,nao))
        vhf1 = np.zeros((len(atmlst),3,nao,nao))
        max_memory = 4000
        blksize = max(1, int(max_memory*.9e6/8/(nao**3*2.5)))
        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = offsetdic[ia]
            ip1 = p0
            vhf = np.zeros((3,nao,nao))
            for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
                ip0, ip1 = ip1, ip1 + nf
                shls_slice = (b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas)
                ### Regular (i,j|kl)
                h2 = mol.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
                ### Derivative (i[x],j|kl)
                h2x = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                                 shls_slice=shls_slice).reshape(3,nf,nao,-1)
                dm2buf = dm2_[ip0:ip1,:,:] 
                dm2buf[:,:,diagidx] *= .5
                edm += lib.einsum('ipx,iqx->pq', h2.reshape(nf,nao,-1), dm2buf)
                de_2[k] -= np.einsum('xijk,ijk->x', h2x, dm2buf) * 4 
                h2 = None
                dm2buf = None

                # HF part
                for i in range(3):
                    h2x_tmp = lib.unpack_tril(h2x[i].reshape(nf*nao,-1))
                    h2x_tmp = h2x_tmp.reshape(nf,nao,nao,nao)
                    vhf[i] += np.einsum('ijkl,ij->kl', h2x_tmp, hf_dm1[ip0:ip1])
                    vhf[i] -= np.einsum('ijkl,il->kj', h2x_tmp, hf_dm1[ip0:ip1]) * .5
                    vhf[i,ip0:ip1] += np.einsum('ijkl,kl->ij', h2x_tmp, hf_dm1)
                    vhf[i,ip0:ip1] -= np.einsum('ijkl,jk->il', h2x_tmp, hf_dm1) * .5
                h2x = h2x_tmp = None
            vhf1[k] = vhf

        # energy-weighted density matrix

        edm += 0.5*(h1 @ dm1_rel)
        edm = 2* mo_coeff @ mo_coeff.T @ edm
        edm += mo_coeff @ mo_coeff.T @ scf.hf.get_veff(mol,zeta)  @ hf_dm1
        edm += mo_coeff @ mo_coeff.T @ scf.hf.get_veff(mol,hf_dm1) @ zeta

        h1x = grad.get_hcore(mol)
        s1 = grad.get_ovlp(mol)
        def hcore_deriv(atm_id):
            shl0, shl1, p0, p1 = aoslices[atm_id]
            with mol.with_rinv_at_nucleus(atm_id):
                vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                vrinv *= -mol.atom_charge(atm_id)
            vrinv[:,p0:p1] += h1x[:,p0:p1]
            return vrinv + vrinv.transpose(0,2,1)
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices [ia,2:]
            h1xao = hcore_deriv(ia)
            de_1[k] += np.einsum('xij,ij->x', h1xao, dm1_rel)
            # Additional term for (p(x)q|rs) due to the response correction part
            # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
            de_2[k] -= np.einsum('xij,ij->x', vhf1[k], zeta) * 2
            # Handling Energy-weighted density matrix and S(x)
            de_w[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], edm[p0:p1] )
            de_w[k] -= np.einsum('xji,ij->x', s1[:,p0:p1], edm[:,p0:p1])

        de_nuc = grad.grad_nuc(mol, atmlst=atmlst)
        if cf.debug:
            printmat(de_1, name='One-body part (h1[x] D1rel')
            printmat(de_2, name='Two-body part (1/2 h2[x] D2rel')
            printmat(de_w, name='Energy-weighted density matrix W (S[x] W)')
        de = de_1+de_2+de_w+de_nuc

    de = mpi.bcast(de, root=0)
    print_grad(Quket.geometry, de)
    return de



def geomopt(Quket,init_dict,kwds):
    from pyscf.geomopt.berny_solver import to_berny_geom, _geom_to_atom
    from pyscf.geomopt.addons import symmetrize
    from quket.vqe import VQE_driver, vqe
    from quket.quket_data import QuketData
    try:
        from berny import Berny, geomlib

        if mpi.main_rank:
            mol = Quket.pyscf
            geom = to_berny_geom(mol)
        else:
            geom = None
        pyscf_geom = kwds['geometry']
        geom = mpi.bcast(geom, root=0)
        optimizer = Berny(geom)
        optimizer.send((Quket.energy, Quket.nuclear_grad))
        cycle = 1
        for cycle, geom in enumerate(optimizer):
            prints(f'"""""""""""""""""""')
            prints(f'     Cycle {cycle}')
            prints(f'"""""""""""""""""""')
            Quket = QuketData(**init_dict)
            set_config(kwds, Quket)
            if mpi.main_rank:
                if mol.symmetry:
                    geom.coords = symmetrize(mol, geom.coords)
                geometry = _geom_to_atom(mol, geom, False) * cf._bohr_to_angstrom
            else:
                geometry = None
            geometry = mpi.bcast(geometry, root=0)
            pyscf_geom = kwds['geometry']
            kwds['geometry'] = to_pyscf_geom(geometry, pyscf_geom)
            Quket.initialize(**kwds)
            Quket.openfermion_to_qulacs()
            Quket.set_projection()
            Quket.get_pauli_list()
            ## initialize theta_list so that it is wrongly read as initial guess in VQE
            Quket.theta_list = None

            if Quket.cf.do_taper_off:
                Quket.tapering.run(mapping=Quket.cf.mapping)
                ### Create excitation-pauli list, and transform relevant stuff by unitary
                Quket.transform_all(reduce=True)
            if Quket.pauli_list is not None:
                Quket.vqe()
            else:
                VQE_driver(Quket,
                   Quket.cf.kappa_guess,
                   Quket.cf.theta_guess,
                   Quket.cf.mix_level,
                   Quket.cf.opt_method,
                   Quket.cf.opt_options,
                   Quket.cf.print_level,
                   Quket.maxiter,
                   Quket.cf.Kappa_to_T1)
            if Quket.cf.do_taper_off:
                Quket.transform_all(backtransform=True, reduce=True)
            Quket.nuclear_grad = nuclear_grad(Quket)
            optimizer.send((Quket.energy, Quket.nuclear_grad))

        if mpi.main_rank:
            geometry = _geom_to_atom(mol, geom, False) * cf._bohr_to_angstrom
            final_geom = to_pyscf_geom(geometry, pyscf_geom)
            if optimizer.converged:
                prints(f"Geometry Optimization Converged in {cycle} cycles")
                print_geom(final_geom)
            else:
                prints(f"Geometry Optimization Failed after {cycle} cycles")
         
    except ImportError:
        prints('No pyberny installed.\nSkipping geometry opt...')
        prints('Install pyberny: \n\n\n  pip install -U pyberny\n\n')
