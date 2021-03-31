# Задача 2
#
# Необходимо реализовать метод Гаусса-Зейделя так, чтобы каждый его шаг работал за O(n^2).
#
# Input: A \in M_n(\R), b \in \R^n, точность eps > 0
# Output: x такое, что ||Ax - b|| < eps, или 0, если на диагонали матрицы A стоят нули или на протяжении
#         20 итераций подряд ||x_s|| >= ||x_{s-1}|| + 1.

import numpy as np
from numpy import linalg as la
import random

MAX_ITER = 10000

# Строит по матрице A размера n \times n нижнетреугольную матрицу L и верхнетреугольную с нулями на диагонали матрицу U
# такие, что A = L + U
def getLU(n, A):
    L = np.array([[A[i][j] if i >= j else 0 for j in range(n)] for i in range(n)]).reshape((n, n))
    U = A - L
    return L, U

# Проверяет, стоят ли на диагонали матрицы A размера n \times n нули
def checkDiagonal(n, A):
    for i in range(n):
        if (A[i, i] == 0):
            return False
    return True

# Метод Гаусса-Зейделя для решения СЛУ вида Ax = b с точностью eps
def gaussZeidelMethod(A, b, eps):
    n = A.shape[0]
    A = A.reshape((n, n))
    if not checkDiagonal(n, A):
        return 0
    L, U = getLU(n, A)
    b = b.reshape((n, 1))
    x = np.array([random.random() for i in range(n)]).reshape((n, 1))
    cnt = 0
    for q in range(MAX_ITER):
        if la.norm(A.dot(x) - b, 2) < eps:
            return x
        c = -U.dot(x) + b
        xn = np.zeros((n, 1))
        for i in range(n):
            xn[i, 0] = (c[i] - L[i, ...].dot(xn)) / A[i, i]
        if la.norm(xn) >= la.norm(x) + 1:
            cnt += 1
        else:
            cnt = 0
        if cnt == 20:
            break
        x = xn
    return 0

# ========== Пример работы: ==========
#
#
# A = np.array([[5, -1],
#               [2, 3]],
#              dtype=float)
#
# b = np.array([3, 25],
#              dtype=float)
#
# eps = 0.0000001
#
# print(gaussZeidelMethod(A, b, eps))
#
# > [[2.00000002]
#    [6.99999999]]