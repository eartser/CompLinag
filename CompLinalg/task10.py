# Задача 10
#
# Вам необходимо реализовать QR-алгоритм для трехдиагональных матриц так, чтобы каждый шаг требовал O(n^2) операций.
#
# Input: Трехдиагональная матрица A \in M_n(\R), eps \in \R - точность.
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

# Расширенное QR-разложение с помощью матриц Гивенса
def decompose(M):
    n = M.shape[0]
    A = np.copy(M)
    Q = np.eye(n)
    decomposition = []
    for i in range(n):
        for j in range(i + 1, n):
            s = A[j, i] / ((A[i, i] ** 2 + A[j, i] ** 2) ** 0.5)
            c = A[i, i] / ((A[i, i] ** 2 + A[j, i] ** 2) ** 0.5)
            givensRotation(Q, i, j, c, s)
            givensRotation(A, i, j, c, s)
            decomposition.append((i, j, c, s))
    return Q.T, A, decomposition

# Умножение с использованием результатов QR-разложения с помощью матриц Гивенса
# Для трехдиагональных матриц такое умножение работает за O(n^2)
def fastMultiplication(M, decomposition):
    A = np.copy(M).T
    for (i, j, c, s) in decomposition:
        givensRotation(A, i, j, c, s)
    return A.T

# QR-алгоритм с оптимизацией для трехдиагональных матриц
def qrAlgorithm(A, eps):
    n = A.shape[0]
    Qk = np.eye(n)
    for q in range(MAX_ITER):
        if checkGershgorinCircles(n, A, eps):
            return [A[i, i] for i in range(n)], Qk
        Q, R, decomposition = decompose(A)
        Qk = fastMultiplication(Qk, decomposition)
        A = fastMultiplication(R, decomposition)
    return [A[i, i] for i in range(n)], Qk


# ========== Пример работы: ==========
#
#
# A = np.array([[2, 1, 0],
#               [1, 3, 6],
#               [0, 6, 4]],
#             dtype=float)
#
# vals, Q = qrAlgorithm(A, 0.00001)
#
# print(vals)
#
# > [9.581628435649646, -2.6384993226022324, 2.056870886952592]
#
# print(Q)
#
# > [[ 0.08947776 -0.15793521 -0.98338711]
#    [ 0.6783871   0.73257289 -0.05592761]
#    [ 0.72923568 -0.66211285  0.17269018]]