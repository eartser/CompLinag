# Задача 5
#
# Реализуйте быстрое домножение на матрицу Хаусхолдера, отвечающую вектору v.
#
# Input: A \in M_n(\R), v \in \R^n.
# Output: H_v \cdot A.

import numpy as np

def householderTransformation(v, A):
    n = A.shape[0]
    v = v.reshape((n, 1))
    H = A - (2 * v).dot((v.T).dot(A))
    return H
    

# ========== Пример работы: ==========
#
#
# A = np.array([[1,2,3],
#               [4,5,6],
#               [7,8,9]],
#             dtype=float)
#
# v = np.array([0, 1, 0], dtype=float)
#
# print(householderTransformation(v, A))
#
# > [[ 1.  2.  3.]
#    [-4. -5. -6.]
#    [ 7.  8.  9.]]