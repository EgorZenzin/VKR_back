"""Вспомогательные утилиты."""

from typing import Any
import numpy as np


def convert_numpy(obj: Any) -> Any:
    """Рекурсивно конвертировать numpy-типы в нативные Python-типы для JSON-сериализации."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(x) for x in obj]
    return obj
