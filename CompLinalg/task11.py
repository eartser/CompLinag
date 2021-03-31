# Задача 11
#
# Вам необходимо реализовать QR-алгоритм с использованием сдвига Уилкинсона так, чтобы каждый шаг требовал O(n^2) операций.
#
# Input: Трехдиагональная матрица A \in M_n(\R), eps \in \R - точность.
# Output: Диагональные эдементы матрицы A_k и матрица Q^{(k)} для такого k, что радиусы кругов Гершгорина матрицы A_k меньше eps.

import numpy as np

MAX_ITER = 100000

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

# Ищет собственные числа матрицы размера 2 \times 2
def findEigenvalues(A):
    d = (A[0, 0] - A[1, 1]) ** 2 + 4 * A[0, 1] * A[1, 0]
    l1 = (A[0, 0] + A[1, 1] + d ** 0.5) / 2
    l2 = (A[0, 0] + A[1, 1] - d ** 0.5) / 2
    return l1, l2

# Проверка кругов Гершгорина для сдвига Уилкинсона
def checkGershgorinCircle(A, eps):
    n = A.shape[0]
    a, b = 0, 0
    for i in range(n - 1):
        a += abs(A[-1][i])
        b += abs(A[i][-1])
    if a < eps and b < eps:
        return True
    return False

# QR-алгоритм с использованием сдвига Уилкинсона
def qrAlgorithm(M, eps, epsGC=0.00001):
    A = np.copy(M)
    n = A.shape[0]
    Qk = np.eye(n)
    vals = []
    while A.shape[0] > 1:
        m = A.shape[0]
        l1, l2 = findEigenvalues(np.array([A[i, -2:] for i in range(m - 2, m)], dtype=float))
        l = 0
        if abs(l1 - A[m-1, m-1]) < abs(l2 - A[m-1, m-1]):
            l = l1
        else:
            l = l2
        E = np.eye(m)
        Q, R, decomposition = decompose(A - l * E)
        Qk = fastMultiplication(Qk, decomposition)
        A = fastMultiplication(R, decomposition)
        A += l * E
        if checkGershgorinCircle(A, epsGC):
            vals.append(l)
            A = np.array([A[i][:-1] for i in range(m - 1)], dtype=float)
    vals.append(A[0, 0])
    return list(reversed(vals)), Qk


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
# > [2.0568708869722503, -2.6384993226218905, 9.581624465011934]
#
# print(Q)
#
# > [[ 0.98338743 -0.15793319  0.08947776]
#    [ 0.05592612  0.73257301  0.6783871 ]
#    [-0.17268883 -0.6621132   0.72923568]]