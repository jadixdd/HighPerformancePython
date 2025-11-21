import time
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.typing import NDArray
from typing import Union

# Параметры тестирования
H_VALUES: List[float] = [0.1, 0.01, 0.005]
EPSILON_VALUES: List[float] = [0.1, 0.01, 0.001]
DEFAULT_PLOT_EPS: float = 0.001  

Numeric = Union[float, NDArray[np.float64]]

# Граничные условия 

def f1(y: Numeric) -> Numeric:
    """u(0, y) - Левая граница"""
    return 5 * y - y**2

def f2(y: Numeric) -> Numeric:
    """u(1, y) - Правая граница"""
    return 4 - y**2 + 5 * y

def f3(x: Numeric) -> Numeric:
    """u(x, 0) - Нижняя граница"""
    return x**2 + 3 * x

def f4(x: Numeric) -> Numeric:
    """u(x, 1) - Верхняя граница"""
    return x**2 + 3 * x + 4


def initialize_grid(n: int) -> np.ndarray:
    """
    Инициализирует сетку (N x N) с граничными условиями.
    N = n + 1 (количество узлов)
    U[j, i] соответствует u(x_i, y_j)
    """
    N = n + 1
    U = np.zeros((N, N), dtype=float)
    

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    # Применение граничных условий
    U[:, 0] = f1(y)       # u(0, y) - Левая граница (i=0)
    U[:, N-1] = f2(y)     # u(1, y) - Правая граница (i=n)
    U[0, :] = f3(x)       # u(x, 0) - Нижняя граница (j=0)
    U[N-1, :] = f4(x)     # u(x, 1) - Верхняя граница (j=n)

    return U

def solve_gauss_seidel_python(
    n: int, epsilon: float
) -> Tuple[np.ndarray, int, float]:
    """
    Решает уравнение Лапласа методом Гаусса-Зейделя
    с использованием стандартных циклов Python.
    """
    U = initialize_grid(n)
    
    start_time = time.perf_counter()
    iterations = 0

    while True:
        U_old = U.copy()
        max_diff = 0.0

        # Обновление внутренних
        for j in range(1, n):
            for i in range(1, n):
                U[j, i] = (U[j, i+1] + U[j, i-1] +
                           U[j+1, i] + U[j-1, i]) / 4.0

        # Проверка сходимости
        for j in range(1, n):
            for i in range(1, n):
                diff = abs(U[j, i] - U_old[j, i])
                if diff > max_diff:
                    max_diff = diff
        
        iterations += 1
        
        if max_diff < epsilon:
            break
            
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    return U, iterations, elapsed_time

def solve_gauss_seidel_numpy(
    n: int, epsilon: float
) -> Tuple[np.ndarray, int, float]:
    """
    Решает уравнение Лапласа методом Гаусса-Зейделя.
    Использует циклы Python для последовательного обновления (необходимо для Г-З)
    и NumPy для операций с массивами (копирование, проверка сходимости).
    """
    U = initialize_grid(n)
    
    start_time = time.perf_counter()
    iterations = 0

    while True:
        U_old = U.copy()
        

        for j in range(1, n):
            for i in range(1, n):

                U[j, i] = (U[j, i+1] + U[j, i-1] +
                           U[j+1, i] + U[j-1, i]) / 4.0

        max_diff = np.max(np.abs(U[1:n, 1:n] - U_old[1:n, 1:n]))
        
        iterations += 1
        
        if max_diff < epsilon:
            break
            
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    return U, iterations, elapsed_time

def plot_solution(U: np.ndarray, h: float, eps: float, filename: str):
    """
    Строит 3D-график решения и сохраняет его в файл.
    """
    N = U.shape[0]
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # U[j, i] -> u(x_i, y_j)
    # X[j, i] = x_i
    # Y[j, i] = y_j
    # plot_surface ожидает Z[j, i], что совпадает с U
    ax.plot_surface(X, Y, U, cmap=cm.get_cmap("viridis"))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('U(x, y)')
    ax.set_title(f"Решение (h={h}, eps={eps})")
    
    plt.savefig(filename)
    print(f"График сохранен в {filename}")
    plt.close(fig)


def main():
    """
    Главная функция: запускает симуляции и генерирует отчет.
    """
    results_py = [["" for _ in EPSILON_VALUES] for _ in H_VALUES]
    results_np = [["" for _ in EPSILON_VALUES] for _ in H_VALUES]
    plot_files = []

    print("Запуск симуляций... Это может занять некоторое время.")

    for i, h in enumerate(H_VALUES):
        n = int(1.0 / h)
        print(f"\n--- Тестирование h = {h} (n = {n}) ---")
        
        for j, eps in enumerate(EPSILON_VALUES):
            print(f"  Запуск Python (G-S) для eps = {eps}...")
            U_py, iter_py, time_py = solve_gauss_seidel_python(n, eps)
            results_py[i][j] = f"{time_py:.4f}c ({iter_py} итер.)"
            
            print(f"  Запуск NumPy (Jacobi) для eps = {eps}...")
            U_np, iter_np, time_np = solve_gauss_seidel_numpy(n, eps)
            results_np[i][j] = f"{time_np:.4f}c ({iter_np} итер.)"

            if eps == DEFAULT_PLOT_EPS:
                filename = f"solution_h_{str(h).replace('.', '_')}.png"
                plot_files.append((h, filename))
                plot_solution(U_np, h, eps, filename)

    print("\nСимуляции завершены. Генерация отчета 'report.md'...")

    # --- Генерация markdown-отчета ---
    
    report = "# Отчет: Лабораторная работа 1 (Вариант 12)\n\n"
    report += "Решение задачи Дирихле для уравнения Лапласа методом конечных разностей.\n\n"
    
    # Таблица 1: Python
    report += "## 1. Реализация на Python (Циклы, Метод Гаусса-Зейделя)\n\n"
    report += f"| h | eps = {EPSILON_VALUES[0]} | eps = {EPSILON_VALUES[1]} | eps = {EPSILON_VALUES[2]} |\n"
    report += "|---|---|---|---|\n"
    for i, h in enumerate(H_VALUES):
        report += f"| {h} | {results_py[i][0]} | {results_py[i][1]} | {results_py[i][2]} |\n"
        
    # Таблица 2: NumPy
    report += "\n## 2. Реализация на NumPy (Векторизация, Метод Гаусса-Зейделя)\n\n"
    report += f"| h | eps = {EPSILON_VALUES[0]} | eps = {EPSILON_VALUES[1]} | eps = {EPSILON_VALUES[2]} |\n"
    report += "|---|---|---|---|\n"
    for i, h in enumerate(H_VALUES):
        report += f"| {h} | {results_np[i][0]} | {results_np[i][1]} | {results_np[i][2]} |\n"
        
    # Графики
    report += "\n## 3. Графики решений (NumPy)\n\n"
    report += f"Графики построены для `eps = {DEFAULT_PLOT_EPS}`.\n\n"
    for h, filename in plot_files:
        report += f"### Решение для h = {h}\n"
        report += f"![Решение h={h}]({filename})\n\n"
    
    # Сохранение отчета
    try:
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("Отчет 'report.md' успешно сгенерирован.")
    except IOError as e:
        print(f"Ошибка при записи файла отчета: {e}")

if __name__ == "__main__":
    main()