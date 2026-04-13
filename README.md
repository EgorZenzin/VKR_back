# Combinatorial Optimization API

REST API для решения задач комбинаторной оптимизации **на основе машинного обучения** с возможностью выбора задачи, выбора алгоритма и сравнения нескольких алгоритмов между собой.

## Поддерживаемые задачи

| Задача | Идентификатор | Оптимизация |
|--------|--------------|-------------|
| Задача коммивояжёра | `tsp` | минимизация |
| Задача о рюкзаке | `knapsack` | максимизация |
| Задача назначения | `assignment` | минимизация |

## Алгоритмы

| Алгоритм | TSP | Рюкзак | Назначение |
|----------|-----|--------|------------|
| Жадный | ✅ | ✅ | ✅ |
| Полный перебор | ✅ (n ≤ 12) | ✅ (n ≤ 20) | ✅ (n ≤ 10) |
| Генетический | ✅ | ✅ | ✅ |
| **Генетический + ML** 🧠 | ✅ | ✅ | ✅ |
| Имитация отжига | ✅ | — | — |
| Динамическое программирование | — | ✅ | — |
| Венгерский | — | — | ✅ |

## ML-суррогатная модель

Ключевая особенность проекта — интеграция **суррогатной MLP-модели** (многослойного перцептрона) в генетические алгоритмы для ускорения оценки качества решений.

### Принцип работы

1. **Фаза прогрева** (`warmup_generations`): все особи оцениваются точной целевой функцией, накапливаются обучающие данные.
2. **Обучение суррогата**: MLP обучается на парах (кодировка решения → значение ЦФ).
3. **ML-фаза**: часть популяции (`surrogate_ratio`) оценивается суррогатом вместо точного вычисления.
4. **Дообучение**: суррогат периодически дообучается на новых точных оценках (`retrain_every`).

### ML-метрики в ответе API

Алгоритмы с ML возвращают поле `ml_metrics`:

```json
{
  "ml_used": true,
  "surrogate_model": "MLPRegressor",
  "exact_evaluations": 2750,
  "surrogate_evaluations": 2250,
  "surrogate_accuracy_r2": 0.85,
  "warmup_generations": 10,
  "surrogate_ratio": 0.5,
  "training_samples": 2750
}
```

### Параметры ML-алгоритмов

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `warmup_generations` | 15-20 | Число поколений чисто точной оценки |
| `surrogate_ratio` | 0.5 | Доля популяции, оцениваемой суррогатом |
| `retrain_every` | 10 | Периодичность дообучения суррогата |
| `hidden_layers` | (64, 32) | Архитектура скрытых слоёв MLP |

### Архитектура ML-модуля

Модуль `app/ml/` спроектирован для **замены модели без изменения алгоритмов**:

- `base.py` — абстрактный интерфейс `BaseSurrogateModel` (fit, predict, score, is_ready)
- `surrogate.py` — конкретная реализация `MLPSurrogateModel` на scikit-learn

Для замены модели достаточно создать новый класс, наследующий `BaseSurrogateModel`.

## Установка и запуск

```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Активация (Linux/Mac)
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск сервера
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

После запуска доступны:
- API: http://localhost:8000
- Документация Swagger: http://localhost:8000/docs
- Документация ReDoc: http://localhost:8000/redoc

## API Endpoints

### `GET /health`
Проверка работоспособности сервиса.

### `GET /tasks`
Список всех поддерживаемых задач с доступными алгоритмами.

### `GET /tasks/{task_name}/algorithms`
Список алгоритмов для конкретной задачи.

### `POST /solve`
Решить задачу одним алгоритмом.

### `POST /compare`
Сравнить несколько алгоритмов на одной задаче.

## Примеры запросов

### Решение задачи коммивояжёра (генетический алгоритм)

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "tsp",
    "algorithm": "genetic",
    "input_data": {
      "cities": [
        {"x": 0, "y": 0, "name": "A"},
        {"x": 1, "y": 5, "name": "B"},
        {"x": 5, "y": 2, "name": "C"},
        {"x": 6, "y": 6, "name": "D"},
        {"x": 8, "y": 3, "name": "E"}
      ]
    },
    "params": {
      "population_size": 50,
      "generations": 200
    }
  }'
```

### Решение задачи о рюкзаке (динамическое программирование)

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "knapsack",
    "algorithm": "dynamic_programming",
    "input_data": {
      "items": [
        {"weight": 2, "value": 6, "name": "Книга"},
        {"weight": 3, "value": 5, "name": "Ноутбук"},
        {"weight": 4, "value": 8, "name": "Планшет"},
        {"weight": 5, "value": 10, "name": "Камера"}
      ],
      "capacity": 8
    }
  }'
```

### Решение задачи назначения (венгерский алгоритм)

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "assignment",
    "algorithm": "hungarian",
    "input_data": {
      "cost_matrix": [
        [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4]
      ]
    }
  }'
```

### Сравнение алгоритмов для TSP

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "tsp",
    "algorithms": ["greedy", "genetic", "genetic_ml", "simulated_annealing"],
    "input_data": {
      "cities": [
        {"x": 0, "y": 0},
        {"x": 1, "y": 5},
        {"x": 5, "y": 2},
        {"x": 6, "y": 6},
        {"x": 8, "y": 3}
      ]
    },
    "params": {
      "genetic": {"population_size": 50, "generations": 100},
      "genetic_ml": {"population_size": 50, "generations": 100, "warmup_generations": 10},
      "simulated_annealing": {"initial_temp": 5000, "cooling_rate": 0.999}
    }
  }'
```

### Сравнение ГА с ML и без ML (основной сценарий)

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "knapsack",
    "algorithms": ["genetic", "genetic_ml"],
    "input_data": {
      "items": [
        {"weight": 2, "value": 10},
        {"weight": 5, "value": 20},
        {"weight": 3, "value": 15},
        {"weight": 7, "value": 25},
        {"weight": 1, "value": 5},
        {"weight": 4, "value": 18}
      ],
      "capacity": 15
    },
    "params": {
      "genetic": {"generations": 200, "population_size": 100},
      "genetic_ml": {"generations": 200, "population_size": 100, "warmup_generations": 15, "surrogate_ratio": 0.5}
    }
  }'
```

## Примеры ответов

### POST /solve (TSP, жадный алгоритм)

```json
{
  "task_name": "tsp",
  "task_display_name": "Задача коммивояжёра",
  "algorithm_name": "greedy",
  "display_name": "Жадный алгоритм (ближайший сосед)",
  "input_data": {
    "cities": [
      {"x": 0, "y": 0, "name": "A"},
      {"x": 1, "y": 5, "name": "B"},
      {"x": 5, "y": 2, "name": "C"},
      {"x": 6, "y": 6, "name": "D"},
      {"x": 8, "y": 3, "name": "E"}
    ]
  },
  "solution": [0, 2, 4, 3, 1],
  "objective_value": 20.57,
  "execution_time": 0.000123,
  "iterations": null,
  "convergence_history": [20.57],
  "visualization_data": {
    "cities": [...],
    "route_order": [0, 2, 4, 3, 1],
    "route_coordinates": [
      {"x": 0, "y": 0}, {"x": 5, "y": 2}, {"x": 8, "y": 3},
      {"x": 6, "y": 6}, {"x": 1, "y": 5}, {"x": 0, "y": 0}
    ]
  }
}
```

### POST /compare

```json
{
  "task_name": "tsp",
  "task_display_name": "Задача коммивояжёра",
  "input_data": { "..." : "..." },
  "algorithms": ["greedy", "genetic", "simulated_annealing"],
  "results": [
    {
      "algorithm_name": "greedy",
      "display_name": "Жадный алгоритм (ближайший сосед)",
      "solution": [0, 2, 4, 3, 1],
      "objective_value": 20.57,
      "execution_time": 0.000123,
      "iterations": null,
      "convergence_history": [20.57],
      "visualization_data": { "..." : "..." }
    }
  ],
  "comparison_charts": {
    "convergence": [
      {
        "algorithm_name": "greedy",
        "display_name": "Жадный алгоритм (ближайший сосед)",
        "data": [{"iteration": 0, "value": 20.57}]
      },
      {
        "algorithm_name": "genetic",
        "display_name": "Генетический алгоритм",
        "data": [{"iteration": 0, "value": 25.3}, {"iteration": 1, "value": 22.1}, "..."]
      }
    ],
    "time_comparison": {
      "labels": ["Жадный алгоритм", "Генетический алгоритм", "Имитация отжига"],
      "values": [0.000123, 0.145, 0.089]
    },
    "quality_comparison": {
      "labels": ["Жадный алгоритм", "Генетический алгоритм", "Имитация отжига"],
      "values": [20.57, 18.92, 19.15]
    }
  }
}
```

## Архитектура проекта

```
app/
├── main.py                    # Точка входа FastAPI
├── config.py                  # Конфигурация
├── ml/                        # ML-модуль (суррогатные модели)
│   ├── base.py                # Абстрактный интерфейс BaseSurrogateModel
│   └── surrogate.py           # MLP-суррогат (MLPSurrogateModel)
├── algorithms/                # Алгоритмы решения
│   ├── base.py                # Базовый класс BaseAlgorithm + AlgorithmResult
│   ├── registry.py            # Центральный реестр задач и алгоритмов
│   ├── tsp/                   # Алгоритмы для задачи коммивояжёра
│   │   ├── genetic.py         # Генетический алгоритм
│   │   └── genetic_ml.py      # Генетический алгоритм + ML-суррогат
│   ├── knapsack/              # Алгоритмы для задачи о рюкзаке
│   │   ├── genetic.py
│   │   └── genetic_ml.py
│   └── assignment/            # Алгоритмы для задачи назначения
│       ├── genetic.py
│       └── genetic_ml.py
├── schemas/                   # Pydantic-схемы входных/выходных данных
│   ├── common.py              # Общие схемы (запросы, ответы, графики)
│   ├── tsp.py                 # Схемы для TSP
│   ├── knapsack.py            # Схемы для рюкзака
│   └── assignment.py          # Схемы для назначения
├── routers/                   # Маршруты API
│   ├── tasks.py               # GET /tasks, GET /tasks/{task_name}/algorithms
│   ├── solver.py              # POST /solve
│   └── compare.py             # POST /compare
├── services/                  # Бизнес-логика
│   ├── solver_service.py      # Оркестрация решения
│   └── compare_service.py     # Оркестрация сравнения
├── comparison/                # Формирование данных для графиков сравнения
│   └── comparator.py
├── visualization_data/        # Подготовка данных для визуализации
│   └── chart_data.py
└── utils/                     # Вспомогательные утилиты
    └── helpers.py
```

### Добавление новой задачи

1. Создайте директорию `app/algorithms/<task_name>/`.
2. Реализуйте алгоритмы, унаследовав `BaseAlgorithm`.
3. Зарегистрируйте задачу и алгоритмы в `app/algorithms/registry.py`.
4. Добавьте Pydantic-схему в `app/schemas/`.
5. Добавьте визуализацию в `app/visualization_data/chart_data.py`.

### Добавление нового алгоритма

1. Создайте файл в `app/algorithms/<task_name>/<algorithm>.py`.
2. Наследуйте `BaseAlgorithm`, реализуйте `solve()`.
3. Добавьте экземпляр в `TASK_REGISTRY` для нужных задач.
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