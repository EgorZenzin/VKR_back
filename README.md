# VKR Back — Комбинаторная оптимизация API

REST API для решения задач комбинаторной оптимизации с использованием различных алгоритмов, сравнением результатов и визуализацией.

## Задачи и алгоритмы

| Задача | Алгоритмы |
|--------|-----------|
| **TSP (Коммивояжёр)** | Ближайший сосед, Генетический, Муравьиный (ACO), Имитация отжига |
| **Назначения** | Венгерский, Жадный, Генетический |
| **Рюкзак** | Динамическое программирование, Жадный, Генетический |
| **Раскраска графа** | Жадный, Генетический, Имитация отжига |
| **Макс. поток** | Форда-Фалкерсона, Эдмондса-Карпа, Диница |

## Установка и запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Создание БД PostgreSQL
# Убедитесь, что PostgreSQL запущен и база vkr_optimizer создана

# Запуск
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Переменные окружения

Создайте `.env` файл:
```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vkr_optimizer
```

## API Эндпоинты

| Метод | URL | Описание |
|-------|-----|----------|
| GET | `/problems` | Список задач и алгоритмов |
| POST | `/solve` | Запуск алгоритма |
| POST | `/compare` | Сравнение алгоритмов |
| GET | `/history/` | История результатов |
| GET | `/history/{id}` | Детали результата |
| GET | `/chart/route/{id}` | График маршрута TSP |
| GET | `/chart/convergence/{id}` | График сходимости |
| GET | `/chart/comparison/{id}` | Сравнительная диаграмма |
| GET | `/chart/graph/{id}` | Визуализация графа |

## Примеры запросов

### Решить TSP
```json
POST /solve
{
  "problem_type": "tsp",
  "algorithm": "genetic",
  "input_data": {
    "cities": [
      {"x": 0, "y": 0},
      {"x": 1, "y": 5},
      {"x": 5, "y": 2},
      {"x": 3, "y": 7}
    ]
  },
  "params": {"generations": 200, "population_size": 50}
}
```

### Сравнить алгоритмы
```json
POST /compare
{
  "problem_type": "tsp",
  "algorithms": ["nearest_neighbor", "genetic", "simulated_annealing"],
  "input_data": {
    "cities": [
      {"x": 0, "y": 0},
      {"x": 1, "y": 5},
      {"x": 5, "y": 2}
    ]
  }
}
```

Swagger UI: http://localhost:8000/docs