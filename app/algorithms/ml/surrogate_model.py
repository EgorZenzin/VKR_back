"""
Суррогатная модель на основе нейронной сети (MLPRegressor).

Обучается на уже вычисленных решениях и предсказывает качество
новых кандидатов, позволяя отсеивать неперспективные варианты
до полного вычисления целевой функции.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class SurrogateModel:
    """Суррогатная нейросеть для оценки качества решений."""

    def __init__(
        self,
        hidden_layers: tuple = (64, 32),
        warmup_size: int = 50,
        retrain_every: int = 20,
    ):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            warm_start=True,
        )
        self.scaler = StandardScaler()
        self.warmup_size = warmup_size
        self.retrain_every = retrain_every

        self.X_data: list[np.ndarray] = []
        self.y_data: list[float] = []
        self.is_trained = False
        self._samples_since_train = 0

    def add_sample(self, features: np.ndarray, fitness: float):
        """Добавить пару (признаки, значение целевой функции)."""
        self.X_data.append(features)
        self.y_data.append(fitness)
        self._samples_since_train += 1

    def add_batch(self, features_batch: list[np.ndarray], fitness_batch: list[float]):
        """Добавить пакет данных."""
        self.X_data.extend(features_batch)
        self.y_data.extend(fitness_batch)
        self._samples_since_train += len(features_batch)

    def should_train(self) -> bool:
        """Проверить, пора ли обучать/переобучать модель."""
        if len(self.X_data) < self.warmup_size:
            return False
        if not self.is_trained:
            return True
        return self._samples_since_train >= self.retrain_every

    def train(self):
        """Обучить нейросеть на накопленных данных."""
        if len(self.X_data) < self.warmup_size:
            return

        X = np.array(self.X_data)
        y = np.array(self.y_data)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.model.fit(X_scaled, y)
        self.is_trained = True
        self._samples_since_train = 0

    def predict(self, features: np.ndarray) -> float:
        """Предсказать качество одного решения."""
        if not self.is_trained:
            return 0.0
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])

    def predict_batch(self, features_batch: list[np.ndarray]) -> np.ndarray:
        """Предсказать качество пакета решений."""
        if not self.is_trained:
            return np.zeros(len(features_batch))
        X = np.array(features_batch)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def select_top_k(
        self, candidates: list, features_batch: list[np.ndarray], k: int
    ) -> list:
        """Отобрать top-k перспективных кандидатов по предсказанию НС."""
        if not self.is_trained or len(candidates) <= k:
            return candidates

        predictions = self.predict_batch(features_batch)
        top_indices = np.argsort(predictions)[:k]
        return [candidates[i] for i in top_indices]

    def get_stats(self) -> dict:
        """Статистика модели."""
        return {
            "total_samples": len(self.X_data),
            "is_trained": self.is_trained,
            "hidden_layers": list(self.model.hidden_layer_sizes),
        }
