# Задача 6
#
# Реализуйте QR-разложение с помощью матриц Хаусхолдера. Применяйте домножение на матрицы Хаусхолдера к
# подматрицам нужного размера.
#
# Input: A \in M_n(\R).
# Output: Q и R такие, что A = QR, где Q - ортогональная, а R - верхнетреугольная.

import numpy as np
from numpy import linalg as la

# Домножение матрицы A на матрицу Хаусхолдера, соответствующую вектору v.
def householderTransformation(v, A):
    n = A.shape[0]
    v = v.reshape((n, 1))
    H = A - (2 * v).dot((v.T).dot(A))
    return H

# QR-разложение с помощью матриц Хаусхолдера
def qrDecomposition(M):
    n = M.shape[0]
    Q = np.eye(n)
    A = np.copy(M)
    for i in range(n-1):
        v = A[i:, i]
        u = v / la.norm(v)
        u[0] -= 1
        u = u / la.norm(u)
        v = np.fromiter([0 if j < i else u[j - i] for j in range(n)], dtype=float)
        Q = householderTransformation(v, Q)
        A = householderTransformation(v, A)
    return Q.T, A


# ========== Пример работы: ==========
#
#
# A = np.array([[1,2,4],
#               [3,3,2],
#               [4,1,3]],
#             dtype=float)
#
# Q, R = qrDecomposition(A)
#
# print(Q)
#
# > [[ 0.19611614  0.61547108  0.76337004]
#    [ 0.58834841  0.54893366 -0.59373225]
#    [ 0.78446454 -0.56556802  0.25445668]]
#
# print(R)
#
# > [[ 5.09901951e+00  2.94174203e+00  4.31455497e+00]
#    [-1.39368474e-15  2.31217513e+00  1.86304759e+00]
#    [-1.72859003e-15 -2.91433544e-16  2.62938568e+00]]