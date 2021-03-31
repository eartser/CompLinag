# Задача 1
#
# Необходимо реализовать метод простых итераций так, чтобы каждый его шаг работал за O(n^2).
#
# Input: A \in M_n(\R), b \in \R^n, точность eps > 0
# Output: x такое, что ||x - Ax - b|| < eps, или 0, если круги Герршгорина не лежат в единичном круге с центрром в 0 и 
#         на протяжении 20 итераций подряд ||x_s|| >= ||x_{s-1}|| + 1.

import numpy as np
from numpy import linalg as la
import random

MAX_ITER = 10000

# Находит все круги Гершгорина матрицы A размера n \times n
def getGershgorinCircles(n, A):
    circles = []
    for i in range(n):
        r = 0
        for j in range(n):
            if j != i:
                r += abs(A[i, j])
        circles.append((A[i, i], r))
    return circles


# Проверяет, лежат ли все круги Гершгорина матрицы A размера n \times n в единичном круге с центром в 0
def checkGershgorinCircles(n, A):
    circles = getGershgorinCircles(n, A)
    for (a, r) in circles:
        if abs(a) + r > 1:
            return False
    return True

# Метод простых итераций для решения СЛУ вида x = Ax + b с точностью eps
def iterationMethod(A, b, eps):
    n = A.shape[0]
    A = A.reshape((n, n))
    b = b.reshape((n, 1))
    x = np.array([random.random() for i in range(n)]).reshape((n, 1))
    boundedEigenvalues = checkGershgorinCircles(n, A)
    cnt = 0
    for q in range(MAX_ITER):
        if la.norm(x - A.dot(x) - b, 2) < eps:
            return x
        xn = A.dot(x) + b
        if not boundedEigenvalues:
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
# A = np.array([[0.1, 0.2, 0.3], 
#               [0.4, 0.5, 0.4], 
#               [0.3, 0.2, 0.1]], 
#              dtype=float)
#
# b = np.array([1, 1, 1], 
#              dtype=float)
#
# eps = 0.0001
#
# print(iterationMethod(A, b, eps))
#
# > [[4.99977512]
#    [9.99949053]
#    [4.99977512]]