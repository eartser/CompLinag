# Задача 12
#
# Реализуйте тест графов на неизоморфность.
#
# Input: Матрицы смежности A(G_1) и A(G_2).
# Output: 0 - если графы не изоморфны, 1 - если это неясно.

import numpy as np
from numpy import linalg as la

MAX_ITER = 100000

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


def checkNotIsomorphic(A, B):
    # Проверка на количество вершин в графах
    def checkSizes(A, B):
        return A.shape == B.shape

    # Проверка на степени вершин в графах
    def checkDegrees(A, B):
        n = A.shape[0]
        degA, degB = [], []
        for i in range(n):
            degA.append(sum([A[i, j] for j in range(n)]))
            degB.append(sum([B[i, j] for j in range(n)]))
        return sorted(degA) == sorted(degB)

    # Проверка на спектры графов
    def checkSpectrum(A, B):
        tdA, QA = threediagonalization(A)
        tdB, QB = threediagonalization(B)
        valsA, QA = qrAlgorithm(tdA, 0.00001)
        valsB, QB = qrAlgorithm(tdB, 0.00001)
        return list(sorted(valsA)) == list(sorted(valsB))

    if not checkSizes(A, B):
        return 0
    if not checkDegrees(A, B):
        return 0
    if not checkSpectrum(A, B):
        return 0
    return 1


# ========== Пример работы: ==========
#
#
# A = np.array([[0, 1, 0],
#               [1, 0, 6],
#               [0, 6, 0]],
#             dtype=float)
#
# B = np.array([[0, 1, 1],
#               [1, 0, 6],
#               [1, 6, 0]],
#             dtype=float)
#
# print(checkNotIsomorphic(A, B))
#
# > 0