"""MLP-суррогатная модель на базе scikit-learn.

Многослойный перцептрон (MLP) обучается предсказывать значение целевой
функции по кодировке решения. Используется как быстрый аппроксиматор
в генетическом алгоритме — часть популяции оценивается суррогатом
вместо точного вычисления.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from app.ml.base import BaseSurrogateModel


class MLPSurrogateModel(BaseSurrogateModel):
    """Суррогатная модель на основе многослойного перцептрона (MLP).

    Архитектура настраивается через параметры конструктора.
    По умолчанию используется сеть с двумя скрытыми слоями (64, 32).

    Attributes:
        hidden_layers: Кортеж с размерами скрытых слоёв.
        max_iter: Максимальное число эпох обучения.
        _model: Экземпляр MLPRegressor.
        _scaler_X: Нормализатор входных признаков.
        _scaler_y: Нормализатор целевой переменной.
        _trained: Флаг обученности модели.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64, 32),
        max_iter: int = 200,
        learning_rate_init: float = 0.001,
        activation: str = "relu",
        random_state: int | None = 42,
    ):
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter

        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            activation=activation,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            warm_start=True,  # инкрементальное дообучение
        )
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        self._trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обучить / дообучить модель на новых данных."""
        if len(X) < 5:
            return

        X_scaled = self._scaler_X.fit_transform(X)
        y_scaled = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        self._model.fit(X_scaled, y_scaled)
        self._trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказать значения целевой функции."""
        if not self._trained:
            raise RuntimeError("Модель не обучена. Вызовите fit() перед predict().")

        X_scaled = self._scaler_X.transform(X)
        y_scaled = self._model.predict(X_scaled)
        return self._scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def is_ready(self) -> bool:
        """Модель готова к предсказаниям."""
        return self._trained

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² на переданных данных."""
        if not self._trained or len(X) < 2:
            return 0.0

        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)
