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

optimizer.py

Functions for VQE optimizer.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import OptimizeResult

from quket.fileio import prints, printmat

def SR1(cost_wrap_mpi, jac_wrap_mpi, theta_list, ndim, cost_callback):

    #Symmetric Rank-1 method

    # initial state
    f = cost_wrap_mpi
    x_pre = theta_list
    x = x_pre
    prints("x_init = ",x)
    prints("f_init = ",f(x))
    dir = None
    alpha0 = 1.0
    alpha = alpha0
    B0 = np.eye(ndim)
    prints("B0 = ",B0)
    B = B0
    # for callback
    f_list = []
    x_list = []
    j_list = []
    nit = 0

    # 直線探索でステップサイズ更新
    # line_search = lambda alpha_k: f(x - alpha_k * dir)

    for i in range(100):
        prints("nit = ",nit)
        prints("f(x) = ",f(x))
        gradient = jac_wrap_mpi(x)
        if(np.linalg.norm(gradient) <= 1e-4):
            break
        B_inv = np.linalg.inv(B)
        dir = B_inv @ gradient
        # line_search = lambda alpha_k: f(x - alpha_k * dir)
        #res = minimize_scalar(line_search)
        #alpha = res.x
        #x_pre = x
        #x = x - alpha * dir
        x_pre = x
        x = x - dir
        # セカント条件
        s = x - x_pre
        gradient_new = jac_wrap_mpi(x)
        y = gradient_new - gradient
        # 近似ヘッセ行列の更新
        Bs = B @ s
        y_Bs_t = np.transpose(y - Bs)
        inner = (y - Bs) @ y_Bs_t
        B = B + inner / (y_Bs_t @ s)
        nit += 1
        #prints("B = ",B)
        printmat(B, name='B')
    prints("x_final = ",x)
    prints("f_final = ",f(x))
    ### print out final results ###
    final_param_list = x #estimated x

    return final_param_list

def LSR1(cost_wrap_mpi, jac_wrap_mpi, theta_list, ndim, cost_callback):

    # L-SR1
    # initital state
    f = cost_wrap_mpi
    x_pre =  theta_list
    x = x_pre
    prints("x_init = ",x_pre)
    prints("f_init = ",f(x_pre))
    grad_pre = jac_wrap_mpi(x_pre)
    grad = grad_pre
    #B_init = np.eye(ndim)
    #B_init[0][0] = -1
    #printmat(B_init, name='B_init')
    # B0はPreconditioner
    B0 = None
    B = None
    dir = None
    m = 200
    j = []
    r = []
    pmax = 0.20
    # for callback
    nit = 0
    nit_pmax = 0

    # line search
    line_search = lambda alpha_k: f(x - alpha_k * dir)

    # Preconditioner(へシアンの対角近似)
    def preconditioner(fk, xk, h=1e-5):
        dim = len(xk)
        F0 = 2*fk(xk)
        dx = 4*h*h
        H = np.zeros((dim, dim))
        B_diag = np.zeros((ndim), dtype=float)
        for i in range(dim):
            xk[i] += 2*h
            Fp = fk(xk)
            xk[i] -= 4*h
            Fb = fk(xk)
            F = Fp - F0 + Fb
            H[i][i] = F / dx
            if abs(H[i][i])< 1e-6:
                pass
            else:
                H[i][i] = 1 / H[i][i]
                B_diag[i] = dx / F
            xk[i] += 2*h
        return H, B_diag

    B0_list = []
    B0 = preconditioner(f, x)
    # while not (np.linalg.norm(grad) <= 1e-4):
    B0_list.append(preconditioner(f, x)[1])
    for i in range(100):
        if(np.linalg.norm(grad) <= 1e-4):
            break
        prints("gradient_norm = ",np.linalg.norm(grad))
        prints("f(x) = ",f(x))
        prints("nit = ",nit)
        if nit % m == 0:
            B0 = preconditioner(f, x)[0]
        #B0 = np.diag(B0_list[0])
        #B0_list.append(preconditioner(f, x)[1])
        #printmat(B0_list[i], name=f'{i}th preconditioner')
        B_list = 0
        if nit == 0:
            B = np.diag(B0_list[0])
            B = B0
        elif nit < m:
            B0 = np.diag(B0_list[0])
            prints(B0.shape)
            for i in range(nit):
                j_t = j[i].reshape(-1, 1)
                j[i] = j[i].reshape(1, ndim)
                #printmat(j_t, name=f'{i} j_t')
                #printmat(j[i], name=f'{i} j[i]')
                #prints(f'{i} r[i]={r[i]}')
                #if abs(r[i]) < 1e-10:
                #    r[i] = 1e-10
                B_list += (j_t @ j[i]) / r[i]
            B = B0 + B_list
        else:
            prints(f'nit-m={nit-m}')
            B0 = np.diag(B0_list[nit-m])
            for i in range(m):
                j_t = j[i].reshape(-1, 1)
                j[i] = j[i].reshape(1, ndim)
                #printmat(j_t, name=f'{i} j_t')
                #printmat(j[i], name=f'{i} j[i]')
                #prints(f'{i} r[i]={r[i]}')
                #if abs(r[i]) < 1e-10:
                #    r[i] = 1e-10
                B_list += (j_t @ j[i]) / r[i]
            B = B0 + B_list
        dir = B @ grad
        printmat(grad, name='grad')
        printmat(B, name='B')
        dir_norm = np.linalg.norm(dir)
        printmat(dir, name=f'dir,  norm={dir_norm}')
        if dir_norm > pmax:
            dir *= (pmax / dir_norm)
            nit_pmax += 1
        x_pre = x
        # res = minimize_scalar(line_search)
        # alpha = res.x
        # x = x - alpha * dir
        x = x - dir
        if nit > m:
            j.pop(0)
            r.pop(0)
        # セカント条件
        s = x - x_pre
        grad_pre = grad.copy()
        grad = jac_wrap_mpi(x)
        y = grad - grad_pre
        j_k = B @ y
        j_k = s - j_k
        # rはスカラー
        r_k = j_k @ y
        j.append(j_k)
        r.append(r_k)
        nit += 1
    prints("nit_pmax = ",nit_pmax)
    prints("x_final = ",x)
    prints("f_final = ",f(x))
    ### print out final results ###
    final_param_list = x #estimated x
    return final_param_list
