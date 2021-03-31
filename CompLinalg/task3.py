# Задача 3
#
# Вычислите за O(n) произведение G(i, j, phi)A.
#
# Input: A \in M_n(\R), i,j \in {1, .., n}, коэффициенты c,s \in \R такие, что c^2 + s^2 = 1.
# Output: G(i, j, phi)A для соответствующего phi.

import numpy as np

# Домножение матрицы A на матрицу вращения Гивенса
def givensRotation(A, i, j, c, s):
    u = c * A[i, ...] + s * A[j, ...]
    v = -s * A[i, ...] + c * A[j, ...]
    A[i, ...] = u
    A[j, ...] = v
    return None

# ========== Пример работы: ==========
#
#
# A = np.array([[2, 4],
#               [6, 13]],
#              dtype=float)
#
# givensRotation(A, 0, 1, 0, 1)
#
# print(A)
#
# > [[ 6. 13.]
#    [-2. -4.]]