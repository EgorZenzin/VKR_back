import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_route_chart(cities: list[dict], route: list[int]) -> bytes:
    """Генерация графика маршрута TSP."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    xs = [c["x"] for c in cities]
    ys = [c["y"] for c in cities]

    # Маршрут
    route_x = [xs[i] for i in route] + [xs[route[0]]]
    route_y = [ys[i] for i in route] + [ys[route[0]]]

    ax.plot(route_x, route_y, "b-o", markersize=8, linewidth=1.5, zorder=2)
    ax.plot(xs[route[0]], ys[route[0]], "r*", markersize=15, zorder=3, label="Старт")

    for i, (x, y) in enumerate(zip(xs, ys)):
        label = cities[i].get("name", str(i))
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_title("Маршрут TSP", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_convergence_chart(convergence_history: list[float], algorithm_name: str = "") -> bytes:
    """Генерация графика сходимости."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(convergence_history, "b-", linewidth=1.5)
    ax.set_title(f"Сходимость {algorithm_name}".strip(), fontsize=14)
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Лучшая стоимость")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_comparison_chart(results: list[dict]) -> bytes:
    """Генерация столбчатой диаграммы сравнения алгоритмов."""
    valid = [r for r in results if r.get("cost") is not None and r.get("error") is None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = [r["algorithm"] for r in valid]
    costs = [r["cost"] for r in valid]
    times = [r["execution_time"] for r in valid]

    colors = plt.cm.Set2(np.linspace(0, 1, len(valid)))

    # Стоимость
    axes[0].bar(names, costs, color=colors)
    axes[0].set_title("Стоимость решения", fontsize=13)
    axes[0].set_ylabel("Стоимость")
    axes[0].tick_params(axis="x", rotation=30)

    # Время
    axes[1].bar(names, times, color=colors)
    axes[1].set_title("Время выполнения", fontsize=13)
    axes[1].set_ylabel("Секунды")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_graph_visualization(
    n_vertices: int, edges: list[dict], coloring: dict | None = None,
    flow_edges: list[dict] | None = None
) -> bytes:
    """Визуализация графа (раскраска или поток)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Расположение вершин по кругу
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    pos_x = np.cos(angles)
    pos_y = np.sin(angles)

    # Рёбра
    for e in edges:
        u, v = e["u"], e["v"]
        ax.plot([pos_x[u], pos_x[v]], [pos_y[u], pos_y[v]], "gray", linewidth=1, alpha=0.5)
        if flow_edges:
            for fe in flow_edges:
                if fe["u"] == u and fe["v"] == v:
                    mid_x = (pos_x[u] + pos_x[v]) / 2
                    mid_y = (pos_y[u] + pos_y[v]) / 2
                    ax.annotate(
                        f"{fe['flow']:.0f}/{fe['capacity']:.0f}",
                        (mid_x, mid_y), fontsize=8, ha="center",
                    )

    # Вершины
    if coloring:
        colors_map = plt.cm.Set3(np.linspace(0, 1, max(coloring.values()) + 2))
        node_colors = [colors_map[coloring.get(str(i), 0)] for i in range(n_vertices)]
    else:
        node_colors = ["lightblue"] * n_vertices

    for i in range(n_vertices):
        ax.plot(pos_x[i], pos_y[i], "o", color=node_colors[i], markersize=20, zorder=3)
        ax.annotate(str(i), (pos_x[i], pos_y[i]), ha="center", va="center", fontsize=10, zorder=4)

    ax.set_aspect("equal")
    ax.set_title("Визуализация графа", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
