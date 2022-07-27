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
from scipy import linalg
from quket.fileio import prints

def vectorize_symm(A):
    """
    Given the nxn square matrix A (symmetric),
    vectrize its lower-triangular part (diagonals+off-diagonals), 
    return it as a vector. 

         |  0   1   3   6  10  ... |
         |  1   2   4   7  11  ... |
    A =  |  3   4   5   8  12  ... |   -->  B = [1,2,3,4,5,...]
         |  6   7   8   9  13  ... |
         | 10  11  12  13  14  ... |
         | ..  ..  ..  ..  ..  ... |

    Args:
        A (2darray): Symmetric matrix
    Returns:
        B (1darray): Lower-triangular of A

    Author(s): Takashi Tsuchimochi
    """
    N = A.shape[0]
    if A.shape[1] != N:
        raise ValueError(f'Matrix is not square in vectorize_symm ({A.shape[0], A.shape[1]})')
    NTT = N*(N+1)//2
    B = np.zeros((NTT))

    ij = 0
    for i in range(0, N):
        for j in range(i+1):
            B[ij] = A[i,j]
            ij += 1
    return B

def symm(B):
    """
    Given a vector B with the dimension n*(n+1)//2,
    form a symmetric matrix,

                                       |  0   1   3   6  10  ... |
                                       |  1   2   4   7  11  ... |
    B = [1,2,3,4,5,...]   -->     A =  |  3   4   5   8  12  ... |
                                       |  6   7   8   9  13  ... |
                                       | 10  11  12  13  14  ... |
                                       | ..  ..  ..  ..  ..  ... |
                              
    Args:
        B (1darray): Lower-triangular of A
    Returns:
        A (2darray): Symmetric matrix of (n x n)

    Author(s): Takashi Tsuchimochi
    """
    lenB = len(B)
    n = (-1+np.sqrt(1+8*lenB))//2
    if not n.is_integer():
        raise ValueError(f'Vector is not [N*(N+1)] in symm (lenB = {lenB})')
    n = int(n)
    A = np.zeros((n,n))

    ij = 0
    for i in range(n):
        for j in range(i+1):
            A[i,j] = A[j,i] = B[ij]
            ij += 1
    return A

def vectorize_skew(A):
    """
    Given the nxn square matrix A (skew),
    vectorize its lower-triangular part (off-diagonals), 
    return it as a vector. 

         |  0  -1  -2  -4  -7  ... |
         |  1   0  -3  -5  -8  ... |
    A =  |  2   3  -0  -6  -9  ... |   -->  B = [1,2,3,4,5,...]
         |  4   5   6   0 -10  ... |
         |  7   8   9  10   0  ... |
         | ..  ..  ..  ..  ..  ... |

    Args:
        A (2darray): Skew matrix
    Returns:
        B (1darray): Lower-triangular of A

    Author(s): Takashi Tsuchimochi
    """

    N = A.shape[0]
    if A.shape[1] != N:
        raise ValueError(f'Matrix is not square in vectorize_skew ({A.shape[0], A.shape[1]})')

    NTT = N*(N-1)//2
    B = np.zeros((NTT))

    ij = 0
    for i in range(1, N):
        for j in range(i):
            B[ij] = A[i,j]
            ij += 1
    return B
 
def skew(B):
    """
    Given a vector B with the dimension n*(n-1)//2,
    form a skew matrix,

                                       |  0  -1  -2  -4  -7  ... |  
                                       |  1   0  -3  -5  -8  ... |
    B = [1,2,3,4,5,...]   -->     A =  |  2   3  -0  -6  -9  ... |
                                       |  4   5   6   0 -10  ... |
                                       |  7   8   9  10   0  ... |
                                       | ..  ..  ..  ..  ..  ... |
                              
    Args:
        B (1darray): Lower-triangular of A
    Returns:
        A (2darray): Skew matrix of (n x n)

    Author(s): Takashi Tsuchimochi
    """
    lenB = len(B)
    n = (1+np.sqrt(1+8*lenB))//2
    if not n.is_integer():
        raise ValueError(f'Vector is not [N*(N-1)] in skew (lenB = {lenB})')
    n = int(n)
    A = np.zeros((n,n))
     

    ij = n*(n-1)//2 -1
    for i in range(n-1, 0, -1):
        for j in range(i, 0, -1):
            A[i, j-1] = B[ij] 
            ij -= 1 
    for i in range(n):
        for j in range(i):
           A[j,i] = - A[i,j] 
    return A

def skew3(B):
    lenB = len(B)
    if lenB == 0: 
        raise ValueError(f'length of B in skew3 is 0.')
    n = int(np.ceil((lenB*6)**(1/3))) + 1
    if lenB != n*(n-1)*(n-2)//6:    
        raise ValueError(f'length of B in skew3 is inconsistent. lenB = {lenB} and n = {n}')
    A = np.zeros((n,n,n))
    ijk = n*(n-1)*(n-2)//6 -1
    ijk = 0
    for i in range(n):
        for j in range(i):
            for k in range(j):
                A[i, j, k] = B[ijk] 
                A[i, k, j] =-B[ijk] 
                A[j, i, k] =-B[ijk] 
                A[j, k, i] = B[ijk] 
                A[k, i, j] = B[ijk] 
                A[k, j, i] =-B[ijk] 
                ijk += 1
    return A

def skew4(B):
    lenB = len(B)
    if lenB == 0: 
        raise ValueError(f'length of B in skew4 is 0.')
    n = int(np.ceil((lenB*24)**(1/4))) + 1
    if lenB != n*(n-1)*(n-2)*(n-3)//24:    
        raise ValueError(f'length of B in skew4 is inconsistent.')
    A = np.zeros((n,n,n,n))
    ijkl = 0
    for i in range(n):
        for j in range(i):
            for k in range(j):
                for l in range(k):
                    A[i, j, k, l] = B[ijkl] 
                    A[i, j, l, k] =-B[ijkl] 
                    A[i, k, j, l] =-B[ijkl] 
                    A[i, k, l, j] = B[ijkl] 
                    A[i, l, k, j] =-B[ijkl] 
                    A[i, l, j, k] = B[ijkl] 

                    A[j, i, k, l] =-B[ijkl] 
                    A[j, i, l, k] = B[ijkl] 
                    A[j, k, i, l] = B[ijkl] 
                    A[j, k, l, i] =-B[ijkl] 
                    A[j, l, k, i] = B[ijkl] 
                    A[j, l, i, k] =-B[ijkl] 

                    A[k, i, j, l] = B[ijkl] 
                    A[k, i, l, j] =-B[ijkl] 
                    A[k, j, i, l] =-B[ijkl] 
                    A[k, j, l, i] = B[ijkl] 
                    A[k, l, j, i] =-B[ijkl] 
                    A[k, l, i, j] = B[ijkl] 

                    A[l, i, j, k] =-B[ijkl] 
                    A[l, i, k, j] = B[ijkl] 
                    A[l, j, i, k] = B[ijkl] 
                    A[l, j, k, i] =-B[ijkl] 
                    A[l, k, j, i] = B[ijkl] 
                    A[l, k, i, j] =-B[ijkl] 

                    ijkl += 1
    return A


def lstsq(a, b,
          cond=None, overwrite_a=False, overwrite_b=False,
          check_finite=True, lapack_driver=None, regularization=None):
    """Function
    Wrapper for scipy.linalg.lstsq, which is known to have some bug
    related to 'SVD failure'.
    This wrapper simply tries lstsq some times until it succeeds...
    """
    if regularization is None or regularization == 0:
        for i in range(5):
            try:
                x, res, rnk, s = sp.linalg.lstsq(a, b, cond=cond,
                                                 overwrite_a=overwrite_a,
                                                 overwrite_b=overwrite_b,
                                                 check_finite=check_finite,
                                                 lapack_driver=lapack_driver)
                break
            except:
                pass
        else:
            # Come if not break
            #prints('ERROR')
            def cost_fun(vct):
                return np.linalg.norm(a@vct - b)

            x0 = np.zeros(len(b))
            opt_options = {"gtol": 1e-12,
                           "ftol": 1e-12}

            x = sp.optimize.minimize(cost_fun,
                                        x0=x0,
                                        method="L-BFGS-B",
                                        options=opt_options).x
            res = rnk = s = None
    else:
        x, istop, itn, r1norm, r2norm, anorm, acond, arnom, xnorm, var = sp.sparse.linalg.lsqr(a, b, damp=regularization)
        res = rnk = s = None
    return x, res, rnk, s

def root_inv(A, eps=1e-8):
    """Function:
    Get canonical (non-symmetric) A^-1/2. 
    Dimensions may be reduced.

    Author(s): Takashi Tsuchimochi
    """

    #u, s, vh = np.linalg.svd(A, hermitian=True)
    s,u = np.linalg.eigh(A)
    mask = s >= eps
    red_u = sp.compress(mask, u, axis=1)
    # Diagonal matrix of s**-1/2
    sinv2 = np.diag([1/np.sqrt(i) for i in s if i > eps])
    Sinv2 = red_u@sinv2
    return Sinv2

def nullspace(A, eps=1e-8):
    """Function:
    Get the nullspace and range of A. 

    Author(s): Takashi Tsuchimochi
    """

    s,u = np.linalg.eigh(A)
    mask = s >= eps
    Range = sp.compress(mask, u, axis=1)
    mask = s < eps
    Null = sp.compress(mask, u, axis=1)
    return Null, Range
def Lowdin_orthonormalization(S):
    """Function:
    Get symmetric A^-1/2 based on Lowdin's orthonormalization. 
    Dimensions may be reduced.

    Author(s): Takashi Tsuchimochi
    """
    eig,u = np.linalg.eigh(S)
    #s^(-1/2)
    eig_2 = np.diag(eig)
    eig_2 = np.linalg.pinv(eig_2)
    eig_2 = sp.linalg.sqrtm(eig_2)
    return u@eig_2@np.conjugate(u.T)

def T1vec2mat(noa, nob, nva, nvb, kappa_list):
    """Function:
    Expand kappa_list to ta and tb
    [in]  kappa_list: occ-vir matrices of alpha and beta
    [out] (occ+vir)-(occ+vir) matrices of alpha and beta
          (zeroes substituted in occ-occ and vir-vir)

    Author(s): Takashi Tsuchimochi
    """
    ta = np.zeros((noa+nva, noa+nva))
    tb = np.zeros((noa+nva, noa+nva))
    ia = 0
    for a in range(nva):
        for i in range(noa):
            ta[a+noa, i] = kappa_list[ia]
            ta[i, a+noa] = -kappa_list[ia]
            ia += 1
    for a in range(nvb):
        for i in range(nob):
            tb[a+nob, i] = kappa_list[ia]
            tb[i, a+nob] = -kappa_list[ia]
            ia += 1
    return ta, tb


def T1mat2vec(noa, nob, nva, nvb, ta, tb):
    """Function:
    Extract occ-vir block of ta and tb to make kappa_list
    [in]  (occ+vir)-(occ+vir) matrices of alpha and beta
          (zeroes assumed in occ-occ and vir-vir)
    [out] kappa_list: occ-vir matrices of alpha and beta

    Author(s): Takashi Tsuchimochi
    """
    kappa_list = np.zeros(noa*nva + nob*nvb)
    ia = 0
    for a in range(nva):
        for i in range(noa):
            kappa_list[ia] = ta[a+noa, i]
            ia += 1
    for a in range(nvb):
        for i in range(nob):
            kappa_list[ia] = tb[a+nob, i]
            ia += 1
    return kappa_list


def expAexpB(A, B):
    """Function:
    Given n-by-n matrices A and B, do log(exp(A).exp(B))

    Author(s): Takashi Tsuchimochi
    """
    if A.shape != B.shape:
        raise ValueError(f'Wrong dimensions in expAexpB: A={A.shape}  B={B.shape}')
    return sp.linalg.logm(sp.linalg.expm(A) @ sp.linalg.expm(B))


def T1mult(noa, nob, nva, nvb, kappa1, kappa2):
    """Function:
    Given two kappa's, approximately combine them.

    Author(s): Takashi Tsuchimochi
    """
    t1a, t1b = T1vec2mat(noa, nob, nva, nvb, kappa1)
    t2a, t2b = T1vec2mat(noa, nob, nva, nvb, kappa2)
    t12a = expAexpB(noa+nva, t1a, t2a)
    t12b = expAexpB(noa+nva, t1b, t2b)
    kappa12 = T1mat2vec(noa, nob, nva, nvb, t12a, t12b)
    return kappa12

def Binomial(n, r):
    """Function:
    Given integers n and r, compute nCr

    Args:
       n (int): n of nCr
       r (int): r of nCr

    Returns:
       nCr
    """
# scipy.special.combの方が基本高速みたいです
    return sp.special.comb(n, r, exact=True)

def Lowdin_deriv_d(Seff, H):
    s, c = np.linalg.eigh(Seff)
    s12 = np.diag(1/np.sqrt(s))
    s32 = np.diag(1/(np.sqrt(s) * s))
    H1 = c.T @ H @ c
    dim = H.shape[0]
    eps = 0.00001
    alpha = np.array([[H1[i, j]/(s[j] - s[i]) if abs(s[j] - s[i]) > eps else 0 for j in range(dim) ] for i in range(dim)])
    H1d = np.diag(np.diagonal(H1))
    c1 = c @ alpha
    return c1 @ s12 @ c.T - c @ s32 @ H1d @ c.T / 2 + c @ s12 @ c1.T

def tikhonov(A, eps=1e-6):
    """
    Perform Tikhonov-regularized pseudo-inverse of A.
    """
    u, a, v = np.linalg.svd(A)
    a_reg = np.zeros(len(a), float)
    for k in range(a.shape[0]):
        a_reg[k] = a[k]/(a[k]**2 + eps) 
    return v.T @ np.diag(a_reg) @ u.T


def _Decompose_expK_to_Givens_rotations(K, eps=1e-12):
    """
    Given a skew matrix K, the unitary as defined by its exponential
    U = exp(K) is decomposed to M Givens rotations:
    
       U = uM(thetaM) ... u2(theta2) u1(theta1)
    
    where |theta1| > |theta2| > ... > |thetaM|
    This subroutine is by no means efficient but works.
    
    A list that contains the Givens indices i,j and
    rotation angle theta is returned.

    Arg(s):
        K (np.ndarray): squared, skew matrix
        eps (float): threshold to terminate 

    Return(s):
        u_list (list): A list of i, j, theta

    Author(s): Takashi Tsuchimochi
    """
    ndim = K.shape[0]
    if ndim != K.shape[1]:
        raise ValueError(f"The K matrix has to be square and skew.")
    if np.linalg.norm(K+np.conjugate(K.T)) > 1e-12:
        raise ValueError(f"The K matrix has to be square and skew.")
    expK = sp.linalg.expm(K)
    abs_tril_expK = abs(np.tril(expK,-1))
    theta = 1
    u_list = []
    # Performing Givens rotations to determine decomposed u1, u2, ...
    while np.linalg.norm(abs_tril_expK) > eps:
        abs_tril_expK = abs(np.tril(expK,-1))
        j = np.argmax(abs_tril_expK)//ndim
        i = np.argmax(abs_tril_expK)%ndim
        aji = expK[j,i]
        ajj = expK[j,j]
        theta = np.arctan(-aji/ajj)
        u = np.eye(ndim)
        # This is actually u_dagger
        u[i,i] = np.cos(theta)
        u[j,i] = np.sin(theta)
        u[i,j] = -np.sin(theta)
        u[j,j] = np.cos(theta)
        expK = expK @ u
        if abs(theta) > eps:
            u_list.append([i,j,theta])
        else:
            break
    return u_list

def Decompose_expK_to_Givens_rotations(K, eps=1e-12):
    """
    Given a skew matrix K, the unitary as defined by its exponential
    U = exp(K) is decomposed to M Givens rotations:
    
       U = uM(thetaM) ... u2(theta2) u1(theta1)
    
    where |theta1| > |theta2| > ... > |thetaM|
    This subroutine is by no means efficient but works.
    
    A list of actual u1, u2, ... matrices is returned. 

    Arg(s):
        K (np.ndarray): squared, skew matrix
        eps (float): threshold to terminate 

    Return(s):
        u_list (list): A list of i, j, theta

    Author(s): Takashi Tsuchimochi
    """
    ndim = K.shape[0]
    u_list = _Decompose_expK_to_Givens_rotations(K, eps=eps)
    umat_list = []
    for i,j,theta in u_list:
        u = np.eye(ndim)
        u[i,i] = np.cos(theta)
        u[j,i] = np.sin(theta)
        u[i,j] = -np.sin(theta)
        u[j,j] = np.cos(theta)
        umat_list.append(u)
    return umat_list

    
