"""Кодировки решений для ML-суррогатов.

Цель — представить решение задачи в виде вектора признаков, по которому
стоимость целевой функции является (приблизительно) линейной.
Это позволяет использовать простую регрессию (Ridge) вместо MLP и
получать высокий R² при малом числе обучающих примеров.
"""

from __future__ import annotations

import numpy as np


def encode_route_adjacency(route: list[int], n: int) -> np.ndarray:
    """Кодировать TSP-маршрут как бинарный вектор смежности.

    Длина вектора n*(n-1)/2 — каждая пара (i, j), i < j, кодируется одним
    битом: 1, если ребро присутствует в маршруте (включая возврат в старт),
    иначе 0. Длина маршрута линейна по этому вектору, что делает Ridge
    почти идеальным аппроксиматором.
    """
    size = n * (n - 1) // 2
    vec = np.zeros(size, dtype=np.float32)
    if not route:
        return vec
    m = len(route)
    for k in range(m):
        a = route[k]
        b = route[(k + 1) % m]
        i, j = (a, b) if a < b else (b, a)
        # индекс пары (i, j) в верхнетреугольной развёртке
        idx = i * (2 * n - i - 1) // 2 + (j - i - 1)
        vec[idx] = 1.0
    return vec


def encode_perm_assignment(perm: list[int], n: int) -> np.ndarray:
    """Кодировать перестановку assignment как плоскую n×n бинарную матрицу.

    perm[i] = j означает, что работник i назначен задаче j.
    Стоимость sum(cost_matrix[i, perm[i]]) линейна по этой кодировке.
    """
    mat = np.zeros((n, n), dtype=np.float32)
    for i, j in enumerate(perm):
        if 0 <= j < n:
            mat[i, j] = 1.0
    return mat.ravel()
