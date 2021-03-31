# Задача 7
#
# Вам необходимо реализовать итерационный алгоритм.
#
# Input: A \in M_n(\R), v \in \R^n, eps \in \R - точность.
# Output: lam - приближение к максимальному собственному числу A, вектор v \in \C^n такой, что ||v|| = 1 и ||Av - lam*v|| < eps,
#         или 0, если таких lam и v не удается найти за разумное время.

import numpy as np
from numpy import linalg as la

MAX_ITER = 100000

# Ищет максимальное собственное число и соответствующий ему собственный вектор матрицы A с помощью итерационного алгоритма
# с заданным приближенным собственным вектором x0 и заданной точностью eps
def iterationAlgorithm(A, x0, eps):
    n = A.shape[0]
    x0 = x0.reshape((n, 1))
    x = x0 / la.norm(x0)
    for q in range(MAX_ITER):
        v = A.dot(x)
        lam = (x.T).dot(v)
        if la.norm(v - lam * x) < eps:
            return lam.tolist()[0][0], x
        x = v
        x = x / la.norm(x)
    return 0
    

# ========== Пример работы: ==========
#
#
# A = np.array([[1,2,4],
#               [3,3,2],
#               [4,1,3]],
#             dtype=float)
#
# x0 = np.array([1, 1, 1],
#             dtype=float)
#
# l, v = iterationAlgorithm(A, x0, 0.000001)
#
# print(l)
#
# > 7.646808567075823
#
# print(v)
#
# > [[0.53698744]
#    [0.60132877]
#    [0.59164872]]