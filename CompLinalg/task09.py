# Задача 9
#
# Вам необходимо реализовать алгоритм тридиагонализации так, чтобы он работал за O(n^3).
#
# Input: Симметричная матрица A \in M_n(\R).
# Output: A' - тридиагонализация матрицы A, ортогональная матрица Q такая, что Q^TAQ = A'.

import numpy as np
from numpy import linalg as la

# Быстрое вычисление произведения Hv \cdot A, где Hv - соответствующая вектору v матрица Хаусхолдера
def householderTransformationL(v, A):
    n = A.shape[0]
    v = v.reshape((n, 1))
    H = A - (2 * v).dot((v.T).dot(A))
    return H

# Быстрое вычисление произведения A \cdot Hv, где Hv - соответствующая вектору v матрица Хаусхолдера
def householderTransformationR(v, A):
    n = A.shape[0]
    v = v.reshape((n, 1))
    H = A - 2 * ((A.dot(v).reshape((n, 1))).dot(v.T))
    return H

# Нахождение тридиагонализации матрицы A и соответствующей ортогональной матрицы Q
def threediagonalization(A):
    eps = 0.0000001
    n = A.shape[0]
    Q = np.eye(n)
    tdA = np.copy(A)
    for i in range(n - 2):
        v = tdA[(i + 1):, i]
        if la.norm(v) < eps:
            continue
        u = v / la.norm(v)
        u[0] -= 1
        if la.norm(u) < eps:
            continue
        u = u / la.norm(u)
        v = np.fromiter([0 if j <= i else u[j - i - 1] for j in range(n)], dtype=float)
        Q = householderTransformationR(v, Q)
        tdA = householderTransformationL(v, tdA)
        tdA = householderTransformationR(v, tdA)
    return tdA, Q

# ========== Пример работы: ==========
#
#
# A = np.array([[1, 2, 3, 4],
#               [2, 3, 4, 1],
#               [3, 4, 1, 2],
#               [4, 1, 2, 3]],
#             dtype=float)
#
# tdA, Q = threediagonalization(A)
#
# print(tdA)
#
# > [[ 1.00000000e+00  5.38516481e+00 -8.79671992e-16  4.60709581e-16]
#    [ 5.38516481e+00  6.24137931e+00  1.99404583e+00  0.00000000e+00]
#    [-8.79671992e-16  1.99404583e+00  2.31364461e+00  1.77787738e+00]
#    [ 4.60709581e-16  2.22044605e-16  1.77787738e+00 -1.55502392e+00]]
#
# print(Q)
#
# > [[ 1.          0.          0.          0.        ]
#    [ 0.          0.37139068  0.88629224 -0.27668579]
#    [ 0.          0.55708601  0.02568963  0.83005736]
#    [ 0.          0.74278135 -0.46241334 -0.48420012]]