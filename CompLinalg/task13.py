# Задача 13
#
# Нужно реализовать две программы, которые считают оптимальное alpha для данных типов графов.
#
# Input: n - в первом случае, p - во втором случае.
# Output: alpha = max(|lam_2|, |lam_n|) / d.

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
def qrAlgorithm(M, eps):
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
        if checkGershgorinCircle(A, eps):
            vals.append(l)
            A = np.array([A[i][:-1] for i in range(m - 1)], dtype=float)
    vals.append(A[0, 0])
    return list(reversed(vals)), Qk

# Подсчитывает индекс вершины по паре чисел x, y
def getIndex(n, x, y):
    x %= n
    y %= n
    return x + n * y

# Находит оптимальное для заданного deg-регулярного графа alpha
def getAlpha(A, deg):
    eps = 0.00001
    n = A.shape[0]
    tdA, QA = threediagonalization(A)
    vals, QA = qrAlgorithm(tdA, eps)
    vals = list(sorted(vals, reverse=True))
    if n == 1:
        return abs(vals[0]) / deg
    return max(abs(vals[1]), abs(vals[n - 1])) / deg

# Решает задачу для n
def alphaN(n):
    A = np.zeros((n * n, n * n))
    for x in range(n):
        for y in range(n):
            ind = getIndex(n, x, y)
            A[ind, getIndex(n, x + 2 * y, y)] += 1
            A[ind, getIndex(n, x - 2 * y, y)] += 1
            A[ind, getIndex(n, x + (2 * y + 1), y)] += 1
            A[ind, getIndex(n, x - (2 * y + 1), y)] += 1
            A[ind, getIndex(n, x, y + 2 * x)] += 1
            A[ind, getIndex(n, x, y - 2 * x)] += 1
            A[ind, getIndex(n, x, y + (2 * x + 1))] += 1
            A[ind, getIndex(n, x, y - (2 * x + 1))] += 1
    return getAlpha(A, 8)

# Решает задачу для p
def alphaP(p):
    A = np.zeros((p + 1, p + 1))
    for i in range(p):
        A[i, (i + 1) % p] += 1
        A[i, (i - 1) % p] += 1
        if i != 0:
            A[i, pow(i, p - 2, p)] += 1
        else:
            A[i, p] += 1
    A[p, p] += 2
    A[p, 0] += 1
    return getAlpha(A, 3)


# ========== Пример работы: ==========
#
#
# print(alphaN(2))
#
# > 0.5
#
# print(alphaP(2))
#
# > 0.5773502691896262