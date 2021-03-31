# Задача 8
#
# Вам необходимо реализовать QR-алгоритм.
#
# Input: Симметричная матрица A \in M_n(\R), eps \in \R - точность.
# Output: Диагональные эдементы матрицы A_k и матрица Q^{(k)} для такого k, что радиусы кругов Гершгорина матрицы A_k меньше eps.

import numpy as np

MAX_ITER = 100000

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

# Проверяет, что все круги Гершгорина матрицы A размера n \times n имеют радиус не больший eps
def checkGershgorinCircles(n, A, eps):
    circles = getGershgorinCircles(n, A)
    for (a, r) in circles:
        if r > eps:
            return False
    return True

# Домножение матрицы A на матрицу вращения Гивенса
def givensRotation(A, i, j, c, s):
    u = c * A[i, ...] + s * A[j, ...]
    v = -s * A[i, ...] + c * A[j, ...]
    A[i, ...] = u
    A[j, ...] = v
    return None

# QR-разложение с помощью матриц Гивенса
def qrDecomposition(M):
    n = M.shape[0]
    A = np.copy(M)
    Q = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s = A[j, i] / ((A[i, i] ** 2 + A[j, i] ** 2) ** 0.5)
            c = A[i, i] / ((A[i, i] ** 2 + A[j, i] ** 2) ** 0.5)
            givensRotation(Q, i, j, c, s)
            givensRotation(A, i, j, c, s)
    return Q.T, A

def qrAlgorithm(A, eps):
    n = A.shape[0]
    Qk = np.eye(n)
    for q in range(MAX_ITER):
        if checkGershgorinCircles(n, A, eps):
            return [A[i, i] for i in range(n)], Qk
        Q, R = qrDecomposition(A)
        A = R.dot(Q)
        Qk = Qk.dot(Q)
    return [A[i, i] for i in range(n)], Qk


# ========== Пример работы: ==========
#
#
# A = np.array([[2, 1, 5],
#               [1, 3, 6],
#               [5, 6, 4]],
#             dtype=float)
#
# vals, Q = qrAlgorithm(A, 0.00001)
#
# print(vals)
#
# > [11.604679223333097, -4.035400459438676, 1.4307212361055794]
#
# print(Q)
#
# > [[ 0.4301042  -0.49007201  0.75818191]
#    [ 0.54946414 -0.52425819 -0.65057091]
#    [ 0.71630967  0.69640705  0.04379134]]