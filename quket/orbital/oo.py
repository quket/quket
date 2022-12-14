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
import numpy as np
import scipy as sp
from time import time

from pyscf import scf
from openfermion.hamiltonians import generate_hamiltonian
from quket.fileio import prints, printmat
from quket.mpilib import mpilib as mpi
from quket import config as cf
from quket.linalg import skew, vectorize_skew, expAexpB
from pyscf.lib import davidson
from .hess_diag import Hess_diag

def transform_integrals(Quket, kappa_list=None,  mo_coeff=None):
    """
    Transform one and two electron integrals either by kappa_list
    or mo_coeff. 
    If kappa_list is given, which is the upper triangular part of 
    the skew matrix kappa[p,q] in the rotation operator

      K = \sum_p>q  kappa[p,q] (p^ q - q^ p)

    this function performs the integral transformation 

      h1_new = exp(-K)  h1  exp(K)
      h2_new = exp(-K)  h2  exp(K)

    If mo_coeff is given, simply use it. If not given, Quket.mo_coeff0
    is used.

    If both kappa_list and mo_coeff is given, use mo_coeff @ exp[kappa]
    for new orbitals. 
    
    If neither is given, nothing is to be done.

    Args:
        Quket (QuketData): 
        kappa_list (1darray, optional): kappa_list 
        mo_coeff (2darray, optional): MO coefficients 

    Returns:
        mo_coeff (2darray): New MO coefficients 
        h1_new (2darray): New hpq
        h2_new (2darray): New (pq|rs)

    Author(s): Takashi Tsuchimochi
    """
    norbs = Quket.n_orbitals    

    if kappa_list is not None:
        kappa = np.zeros((norbs, norbs), dtype=float)
        pq = 0
        for p in range(norbs):
            for q in range(p):
                #if Quket.irrep_list[2*p] == Quket.irrep_list[2*q]:
                kappa[p,q] = kappa_list[pq]
                kappa[q,p] = -kappa[p,q] 
                pq += 1       
        U = sp.linalg.expm(kappa)

        # Update Molecular orbital
        if mo_coeff is not None:
            mo_coeff = mo_coeff @ U
        else:
            mo_coeff = Quket.mo_coeff0 @ U    
    elif mo_coeff is None: 
        prints('No transformation is done')
        return Quket.mo_coeff, Quket.one_body_integrals, Quket.two_body_integrals 
        
    # Now get new integrals with new orbitals
    nf = Quket.n_frozen_orbitals
    nc = Quket.n_core_orbitals
    na = Quket.n_active_orbitals
    

    hpq = scf.hf.get_hcore(Quket.pyscf)
    h1_new = mo_coeff.T @ hpq @ mo_coeff
    if mpi.main_rank:
        h2_new = Quket.pyscf.ao2mo(mo_coeff, compact=False)
    else:
        h2_new = None
    h2_new = mpi.bcast(h2_new)
    h2_new = h2_new.reshape((norbs,norbs,norbs,norbs)) 
    return mo_coeff, h1_new, h2_new
    
def get_energy_by_RDM(Quket, h1, h2, DA, DB, Daaaa, Dbbbb, Dbaab):
    from .misc import get_htilde
    ncore = Quket.n_frozen_orbitals + Quket.n_core_orbitals
    nact = Quket.n_active_orbitals
    norbs = Quket.n_orbitals

    # Active Space 1RDM
    Daa  = DA[ncore:ncore+nact,ncore:ncore+nact]
    Dbb  = DB[ncore:ncore+nact,ncore:ncore+nact]

    ### Get htilde
    Ecore, htilde = get_htilde(Quket, h1, h2) 
    h_xy = htilde[ncore:ncore+nact, ncore:ncore+nact]
    h_wxyz = h2[ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact,ncore:ncore+nact]
    E1 = np.einsum('pq,pq->',h_xy, Daa+Dbb)
    E2 = 0.5*np.einsum('prqs, pqsr->',h_wxyz, Daaaa+2*Dbaab+Dbbbb)
    return Ecore+E1+E2
    
    
    

def orbital_rotation(Quket, kappa_list=None, mo_coeff=None, mbe=False):
    """
    Rotate Hamiltonian. Either kappa_list or mo_coeff is required.
    If only kappa_list is given, which is the lower-triangular part of the following kappa matrix

      K = \sum_p>q  kappa[p,q] (p^ q - q^ p)

    then perform the rotation with the rotation operator exp(K)

      Hnew = exp(-K)  H  exp(K)

    If only mo_coeff is given, Hnew is simply generated by mo_coeff.
    If both kappa_list and mo_coeff is given, use mo_coeff @ exp[kappa]
    for new orbitals. 
    
    All attributes of Quket are updated to comply with the new transformed Hamiltonian.
    MO coefficients are also updated.
    Restricted spin-orbital is assumed, and kappa_list is a vectorized form
    of kappa[p,q] for p>q (a length of norbs*(norbs-1)/2, where norbs is
    the total number of orbitals).

    Author(s): Takashi Tsuchimochi
    """
    from .misc import get_htilde
   
    # Transform integrals and update Quket attributes
    Quket.mo_coeff, Quket.one_body_integrals, Quket.two_body_integrals \
        = transform_integrals(Quket, kappa_list=kappa_list, mo_coeff=mo_coeff)
    nf = Quket.n_frozen_orbitals
    nc = Quket.n_core_orbitals
    na = Quket.n_active_orbitals
    ns = Quket.n_secondary_orbitals
    if mbe:
        na += nc
        nc = 0
    
    # Get active space Hcore
    h0, htilde = get_htilde(Quket)
    h1 = htilde[nf+nc:nf+nc+na+ns, nf+nc:nf+nc+na+ns]
    h2 = Quket.two_body_integrals[nf+nc:nf+nc+na+ns, 
                                  nf+nc:nf+nc+na+ns, 
                                  nf+nc:nf+nc+na+ns, 
                                  nf+nc:nf+nc+na+ns]
     
    # Construct new Hamiltonian
    Quket.operators.Hamiltonian = generate_hamiltonian(h1, h2.transpose(0,3,2,1), h0)
    if not mbe:
        if Quket.cf.mapping == "jordan_wigner":
            Quket.operators.jordan_wigner()
        elif Quket.cf.mapping == "bravyi_kitaev":
            Quket.operators.bravyi_kitaev(Quket.n_qubits)
        if Quket.tapered['operators']:
            Quket.tapered['operators'] = False
            Quket.transform_operators()
        Quket.openfermion_to_qulacs()

def wrap_energy(alpha, delta_kappa, kappa, Quket):
    """
    Wrapper for energy estimate with new Hamiltonian transformed by
                Exp[kappa + alpha * delta_kappa]
    
        Args:
            alpha (float): Scaling factor
            delta_kappa (1darray): Update for kappa to be scaled
            kappa (1darray): Current kappa
            Quket (QuketData): QuketData instance

        Returns:
            Total Energy (float): Total Energy
    """
    #kappa_ = kappa + alpha * delta_kappa
    kappa_ = vectorize_skew(expAexpB(skew(kappa),  skew(delta_kappa)*alpha))
    mo_coeff, h1, h2 = transform_integrals(Quket, kappa_)
    return get_energy_by_RDM(Quket, h1, h2, Quket.DA, Quket.DB, Quket.Daaaa, Quket.Dbbbb, Quket.Dbaab)

def line_search(delta_kappa, kappa, Quket, precision=1e-3):    
    """
    Wrapper for line search to determine scaling factor alpha for next iteration.
            next_kappa = log[ exp[kappa] exp[alpha * delta_kappa] ]
        
        Args:
            delta_kappa (1darray): Update for kappa to be scaled
            kappa (1darray): Current kappa
            Quket (QuketData): QuketData instance
            precision (float, optional): tolerance for convergence in scipy's minimizer

        Returns:
            alpha (float): Scaling factor
    """
    precision = min(precision, 1e-3)
    cost = lambda alpha: wrap_energy(alpha, delta_kappa, kappa, Quket)
    alpha = 0
    #opt = sp.optimize.minimize(
    #           cost,
    #           alpha,
    #           method='L-BFGS-B',
    #            options={'disp':True, 'gtol':precision, 'ftol':precision}
    #           )
    #alpha = opt.x[0]
    if alpha < 1e-8:
        ### Scipy minimizer Failed
        ### use goldenratiosearch
        def goldenRatioSearch(function, range, tolerance):
            gamma = (-1+np.sqrt(5))/2
            a = range[0]
            b = range[1]
            p = b-gamma*(b-a)
            q = a+gamma*(b-a)
            Fp = function(p)
            Fq = function(q)
            width = 1e8
            icyc = 0
            while abs(Fp-Fq) > tolerance:
                if Fp <= Fq:
                    b = q
                    q = p
                    Fq = Fp
                    p = b-gamma*(b-a)
                    Fp = function(p)
                else:
                    a = p
                    p = q
                    Fp = Fq
                    q = a+gamma*(b-a)
                    Fq = function(q)
                width = abs(b-a)/2
                icyc += 1
            alpha = (a+b)/2
            prints(f'Golden Section done in {icyc}')
            return alpha
        alpha = goldenRatioSearch(cost, [0,1.5], precision) 

    return alpha 


def oo(Q, maxiter=None, gtol=None, ftol=None):
    """
    Main driver of Orbital optimization.
    Update mo coeffcients C by a skew matrix k as 
              
                C[k] = C0 @ exp[k] 

    where C0 is the original mo coefficients (usually HF).
    Each of these is stored as QuketData attributes, 
    
              C[k] : QuketData.mo_coeff
              C0   : QuketData.mo_coeff0
              k    : QuketData.kappa_list (only lower-triangular part)
    
    At each cycle of orbital optimization steps, most of attributes 
    related to Hamiltonian, such as QuketData.operators.Hamiltonian, will be overwritten.
    However, one can easily restore the original basis by performing 
    
            QuketData.orbital_rotation(mo_coeff=QuketData.mo_coeff0)

        Args:
            Q (QuketData): QuketData instance
            maxiter (int, optional): maximum number of iterations. Defaulted to Q.oo_maxiter
            gtol (float, optional): gradient convergence threshold. Defaulted to Q.oo_gtol
            ftol (float, optional): energy convergence threshold. Defaulted to Q.oo_ftol
            SA (bool, optional)
    
        Returns:
            None (QuketData is updated)
    """
    from quket.post import get_1RDM, get_2RDM
    from .grad import get_orbital_gradient
    from .hess_whole import get_orbital_hessian

    Q.cf.theta_guess = 'prev'
    method = 'AH'
    if not Q.converge:
        if Q.cf.do_taper_off:
            Q.taper_off()
        Q.run()
        if Q.cf.do_taper_off:
            Q.transform_all(backtransform=True) 
    else:
        if Q.cf.do_taper_off and Q.tapered["states"]:
            Q.transform_all(backtransform=True) 
    if maxiter is None:
        maxiter = Q.oo_maxiter
    if gtol is None:
        gtol = Q.oo_gtol
    if ftol is None:
        ftol = Q.oo_ftol
    deltaE = 1
    norbs = Q.n_orbitals
    nact  = Q.n_active_orbitals
    if Q.kappa_list is None:
        kappa_list = np.zeros((norbs*(norbs-1)//2),dtype=float)
    else:
        kappa_list = Q.kappa_list
    if Q.energy is None:
        Eold = 0
    else:
        Eold = Q.energy
    update_VQE = True
    cycle = 0    
    macro_cycle = 0
    AH_thres = 0.3
    while macro_cycle < maxiter: 
        cycle += 1
        prints(f'\n\n""""""""""""""""""""""""""""""""""""""')
        prints(f'     Orbital-Optimization Cycle {cycle}')
        prints(f'""""""""""""""""""""""""""""""""""""""\n')

        # Get Orbital Gradient and Hessian
        ### For state-average calculations, get averaged RDMs 
        if Q.multi.nstates!=0:
            DA = np.zeros((norbs, norbs), float)
            DB= np.zeros((norbs, norbs), float)
            Daaaa = np.zeros((nact, nact, nact, nact), float)
            Dbbbb = np.zeros((nact, nact, nact, nact), float)
            Dbaab = np.zeros((nact, nact, nact, nact), float)
            for istate in range(Q.multi.nstates):
                da, db = get_1RDM(Q,Q.multi.states[istate])
                daaaa, dbaab, dbbbb = get_2RDM(Q,Q.multi.states[istate])
                DA += da * Q.multi.weights[istate]
                DB += db * Q.multi.weights[istate]
                Daaaa += daaaa * Q.multi.weights[istate]
                Dbbbb += dbbbb * Q.multi.weights[istate]
                Dbaab += dbaab * Q.multi.weights[istate]
            Q.DA = DA/sum(Q.multi.weights)
            Q.DB = DB/sum(Q.multi.weights)
            Q.Daaaa = Daaaa/sum(Q.multi.weights)
            Q.Dbbbb = Dbbbb/sum(Q.multi.weights)
            Q.Dbaab = Dbaab/sum(Q.multi.weights)
        else:
            Q.DA, Q.DB = get_1RDM(Q, state=None)
            Q.Daaaa, Q.Dbaab, Q.Dbbbb = get_2RDM(Q, state=None)
        ### 
        ja,jb=get_orbital_gradient(Q)
        nott = norbs*(norbs-1)//2
        # symmetrize
        #g = np.block([ja,jb])
        #g_ = g.reshape(nott*2, 1)
        g = ja + jb
        g_ = g.reshape(nott, 1)

        if method == 'AH':
            ## Davidson
            from .hess import AHx
            aop = lambda x: AHx(x, Q, g_)[0]
            
            t_ini = time()
            x0 = np.zeros(nott+1)
            x0[0] = 1
            diag = np.hstack([0, Hess_diag(Q)[0]])
            prints('Running Davidson to get update by AH')
            eig, c = davidson(aop, x0, diag, nroots=1, follow_state =True)
            v_lowest = np.array(c).T
            t_dav = time()

            # For test purpose...will be removed 
            #if cf.debug:
            #    eig = np.array([eig])
            #    printmat(v_lowest, eig=eig, name='Davidson')
            #    Hwhole = get_orbital_hessian(Q)
            #    HAA = Hwhole[:nott, :nott]
            #    HBA = Hwhole[nott:, :nott]
            #    HAB = Hwhole[:nott, nott:]
            #    HBB = Hwhole[nott:, nott:]
            #    H = (HAA + HBA + HAB + HBB)
            #    
            #    # Diagonalize Augmented Hessian
            #    AH = np.block([[0,g_.T], [g_,H]])
            #    e,v = np.linalg.eigh(AH)
            #    printmat(v[:,:10], eig=e[:10], name='eigh')
            #    ### Check plausible vector ###
            #    k = 0
            #    while True:
            #        v_lowest = v.T[k]
            #        if abs(v_lowest[0]) < 0.1:
            #            k += 1
            #        else:
            #            break
            #    eig = e[k]
            if abs(v_lowest[0]) < AH_thres:
                prints(f'Warning: Too small C0 in AH : {v_lowest[0]}')
                prints(f'This usually means the Hessian is not positive semi-definite and there is a (maybe symmetry-breaking) lower state.')
            #    v_lowest[0] = 1
            #    prints(f'Warning: Too small C0 in AH : {v_lowest[0]}')
            #    prints(f'Run Davidson again')
                for kk in range(2,10):
                    x1 = np.zeros(nott+1)
                    x1[0] = 1
                    x0 = np.vstack([c, x1])
                    eig, v = davidson(aop, x0, diag, nroots=kk, follow_state =True)
                    c = np.array(v)
                    #printmat(c.T, eig=eig)
                    k = 0
                    k = np.argmax(abs(c[:,0]))
                    if abs(c[k,0]) > 0.1:
                        v_lowest = c[k,:]
                        eig = eig[k]
                        break
                    else:
                        continue
                    if kk == 9:
                        raise Exception(f'No reasonable eigenvector found for AH after 10 cycles.\n'
                                        f'You should check if the system is really reasonable.')
            
            prints(f'Target lowest eigenvalue of augmented Hessian:  {eig}  (C0 = {v_lowest[0]})')
            if cf.debug:
                printmat(v_lowest, eig=np.array([eig]), name='Davidson')
            direction = v_lowest[1:] / v_lowest[0]
            delta_kappa_list = direction
            alpha = line_search(delta_kappa_list, kappa_list, Q, precision=abs(deltaE))
            t_lin = time()

        elif method == 'NR':
            Hwhole = get_orbital_hessian(Q)
            HAA = Hwhole[:nott, :nott]
            HBA = Hwhole[nott:, :nott]
            HAB = Hwhole[:nott, nott:]
            HBB = Hwhole[nott:, nott:]
            H = (HAA + HBA + HAB + HBB)
            delta_kappa_list = (- np.linalg.pinv(H) @ g_) [:,0]
            alpha = 1
        elif method == 'SD':
            delta_kappa_list = np.zeros((norbs*(norbs-1)//2),dtype=float)
            for p in range(nott):
                if abs(diag[1+p]) > 1e-4:
                    delta_kappa_list[p] = - g_[p] / diag[1+p]

        gnorm = np.linalg.norm(g_)
        ### MPI broadcasting 
        alpha = mpi.bcast(alpha)
        kappa_list = mpi.bcast(kappa_list)
        delta_kappa_list = mpi.bcast(delta_kappa_list)
        deltaE = mpi.bcast(deltaE)
        gnorm = mpi.bcast(gnorm)

        if cf.debug:
            printmat(g_.T, name='Gradient')
            printmat(delta_kappa_list, name='Dkappa',n=1,m=len(delta_kappa_list))
        
        if (abs(deltaE) < ftol or gnorm < gtol) and update_VQE:
            prints(f'\n\n"""""""""""""""""""""""""""""""""""""""""""""""""""""""')
            prints(f'     Convergence of orbital optimizer with {macro_cycle} cycles')
            prints(f'"""""""""""""""""""""""""""""""""""""""""""""""""""""""')
            prints(f'  Final: E[oo-{Q.ansatz}] = {Q.energy} ')
            return
        prints('\n\n')
        if cycle > 1:
            prints(f'deltaE = {deltaE}    ', end='')
        prints(f'||g|| = {gnorm} ', end='') 
        prints(f'---> Updated with scaling factor {alpha}\n\n')
        Enext = wrap_energy(alpha, delta_kappa_list, kappa_list, Q)
        prints(f'Next energy estimation:  {Enext}') 
        if Enext > Eold and method == 'AH':
            raise ValueError(f'Augmented Hessian confused: alpha = {alpha},  Enext = {Enext},  Eold = {Eold}')

        # Update kappa
        if Q.cf.do_taper_off or Q.symmetry:
            # Force symmetry and do not allow symmetry breaking
            pq = 0
            for p in range(Q.n_orbitals):
                for q in range(p):
                    if Q.irrep_list[2*p] != Q.irrep_list[2*q]:
                        if abs(delta_kappa_list[pq]) > 1e-4:
                            print(f"Warning! Orbitals p={p} and q={q} have different symmetries but delta kappa is {delta_kappa_list[pq]}")
                        delta_kappa_list[pq] = 0    
                    pq += 1

        kappa_list = vectorize_skew(expAexpB(skew(kappa_list),  skew(delta_kappa_list)*alpha))

        Q.kappa_list = kappa_list
        # Orbital rotation of Quket objects (qubit Hamiltonian)
        orbital_rotation(Q, kappa_list)
        t_rot = time()

        # We shall perform VQE parameter optimization if orbital gradient is small enough with the current micro cycles 
        #update_VQE = -deltaE < ftol or gnorm < gtol
        
        #if update_VQE:
        #    # Perform VQE
        #    if Q.run_qubitfci:
        #        Q.fci2qubit()
        #    if Q.cf.taper_off:
        #        Q.transform_all()
        #    Q.run()
        #    Q.energy = Q.get_E()
        #    if Q.cf.taper_off:
        #        Q.transform_all(backtransform=True)
        #    macro_cycle += 1
        #else:
        #    Q.energy = Enext

        # Perform VQE
        if Q.run_qubitfci:
            Q.fci2qubit()
        t0 = time()
        if Q.cf.do_taper_off:
            Q.transform_all()
        #Q.theta_list *= 0
        Q.run()
        macro_cycle += 1
        t0 = time()
        prints()
        if Q.cf.do_taper_off:
            Q.transform_all(backtransform=True)
        
        deltaE = Q.energy - Eold
        Eold = Q.energy

#if __name__ == '__main__':
#    geometry = [
#        ['H', (0,0.0861418,0.7845969)],
#        ['O', (0., -0.4722836 ,0)],
#        ['H', (0,0.0861418,-0.7845969)]] 
#
#    Q = quket.create(basis="sto-6g",
#                     ansatz="uccd",
#                     maxiter=100,
#                     n_orbitals =5,
#                     n_electrons=6,
#                     geometry = geometry, taper_off='True', debug='False'
#                    )
#    
#    from quket.src.orbital_optimize import *
#    norbs = Q.n_orbitals
#    kappa_list = np.zeros((norbs*(norbs-1)//2),dtype=float)
#    from scipy.optimize import minimize
#    #cost_wrap = lambda kappa_list: evaluate_Ek(kappa_list, Q)
#    #jac = lambda kappa_list: wrapper_orbital_gradient(Q)
#    #
#    #opt = minimize(
#    #           cost_wrap,
#    #           kappa_list,
#    #           jac=jac,
#    #           method='l-bfgs-b',
#    #           options={'maxcor':15, 'disp':True}
#    #           )
def num_grad(Quket):
    norbs = Quket.n_orbitals
    ncore = Quket.n_frozen_orbitals
    nact = Quket.n_active_orbitals
    nott = norbs*(norbs-1)//2
    kappa = np.zeros(nott,dtype=float)
    delta_kappa = np.zeros(nott,dtype=float)
    grad = np.zeros(nott, dtype=float)
    Hess = np.zeros((nott, nott), dtype=float)
    E0 = wrap_energy(0, delta_kappa, kappa, Quket)
    pqrs = 0
    pq = 0
    stepsize = 1e-6
    for p in range(norbs):
        for q in range(p):
            # +
            delta_kappa[pq] += stepsize 
            Ep = wrap_energy(1, delta_kappa, kappa, Quket)
            delta_kappa *= 0
            # -
            delta_kappa[pq] -= stepsize 
            Em = wrap_energy(1, delta_kappa, kappa, Quket)
            delta_kappa *= 0
            grad[pq] = (Ep - Em)/(2*stepsize)
            pq += 1
    return grad 
        
def num_hess(Quket):
    norbs = Quket.n_orbitals
    ncore = Quket.n_frozen_orbitals
    nact = Quket.n_active_orbitals
    nott = norbs*(norbs-1)//2
    kappa = np.zeros(nott,dtype=float)
    delta_kappa = np.zeros(nott,dtype=float)
    grad = np.zeros(nott, dtype=float)
    Hess = np.zeros((nott, nott), dtype=float)
    E0 = wrap_energy(0, delta_kappa, kappa, Quket)
    pqrs = 0
    pq = 0
    stepsize = 5e-4
    for p in tqdm(range(norbs)):
        for q in range(p):
            prints(f'pq = {p} {q}  ({pq} / {nott})')
            rs = 0
            for r in range(norbs):
                for s in range(r): 
                    if pq < rs:
                        rs += 1
                        continue
                    if (p < ncore and q < ncore) or (r < ncore and s < ncore)\
                        or (p >= ncore+nact and q >= ncore+nact) or (r >= ncore+nact and s>= ncore+nact):
                        rs += 1
                        continue
                    if pqrs % mpi.nprocs == mpi.rank: 
                        # +, +
                        delta_kappa[pq] += stepsize 
                        delta_kappa[rs] += stepsize 
                        Epp = wrap_energy(1, delta_kappa, kappa, Quket)
                        delta_kappa *= 0
                        # +, -
                        delta_kappa[pq] += stepsize 
                        delta_kappa[rs] -= stepsize 
                        Epm = wrap_energy(1, delta_kappa, kappa, Quket)
                        delta_kappa *= 0
                        # -, +
                        delta_kappa[pq] -= stepsize 
                        delta_kappa[rs] += stepsize 
                        Emp = wrap_energy(1, delta_kappa, kappa, Quket)
                        delta_kappa *= 0
                        # -, -
                        delta_kappa[pq] -= stepsize 
                        delta_kappa[rs] -= stepsize 
                        Emm = wrap_energy(1, delta_kappa, kappa, Quket)
                        delta_kappa *= 0

                        Hess[pq,rs] = ( Epp - Epm - Emp + Emm ) / (4*stepsize**2)
                        Hess[rs,pq] = Hess[pq,rs] 
                    rs += 1
                    pqrs += 1
            pq += 1
    Hess = mpi.allreduce(Hess, mpi.MPI.SUM)
    return Hess        
    

