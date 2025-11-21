import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit # type: ignore

try:
    import cpp_laplace # type: ignore
except ImportError:
    print("Модуль cpp_laplace не найден. Убедитесь, что вы установили его через pip install .")
    cpp_laplace = None



@njit
def f1(y: float) -> float:
    """Левая граница u(0, y)."""
    return 5 * y - y**2

@njit
def f2(y: float) -> float:
    """Правая граница u(1, y)."""
    return 4 - y**2 + 5 * y

@njit
def f3(x: float) -> float:
    """Нижняя граница u(x, 0)."""
    return x**2 + 3 * x

@njit
def f4(x: float) -> float:
    """Верхняя граница u(x, 1)."""
    return x**2 + 3 * x + 4


def initialize_grid(n: int) -> np.ndarray:
    """
    Инициализация сетки с граничными условиями.
    Размер сетки (n+1) x (n+1).
    """
    size = n + 1
    h = 1.0 / n
    grid = np.zeros((size, size), dtype=np.float64)

    # Заполнение границ
    for i in range(size):
        x = i * h
        grid[i, 0] = f3(x)          # Низ (y=0, j=0)
        grid[i, size - 1] = f4(x)   # Верх (y=1, j=n)

    for j in range(size):
        y = j * h
        grid[0, j] = f1(y)          # Лево (x=0, i=0)
        grid[size - 1, j] = f2(y)   # Право (x=1, i=n)

    return grid



@njit
def solve_numba(grid: np.ndarray, eps: float, max_iter: int = 100000) -> np.ndarray:
    """
    Решение задачи Дирихле методом Гаусса-Зейделя (Numba).
    """
    rows, cols = grid.shape
    
    for k in range(max_iter):
        max_diff = 0.0
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                old_val = grid[i, j]
                new_val = 0.25 * (grid[i + 1, j] + grid[i - 1, j] +
                                  grid[i, j + 1] + grid[i, j - 1])
                grid[i, j] = new_val
                diff = abs(new_val - old_val)
                if diff > max_diff:
                    max_diff = diff
        
        if max_diff < eps:
            break
            
    return grid


def run_experiments() -> None:
    h_values = [0.1, 0.01, 0.005]
    eps_values = [0.1, 0.01, 0.001]
    
    results = []

    print(f"{'h':<10} {'eps':<10} {'Impl':<10} {'Time (s)':<15}")
    print("-" * 50)

    for h in h_values:
        n = int(1.0 / h)
        for eps in eps_values:
            grid_numba = initialize_grid(n)
            
            # Numba 
            if h == h_values[0] and eps == eps_values[0]:
                 solve_numba(grid_numba.copy(), eps)

            start_time = time.time()
            solve_numba(grid_numba, eps)
            numba_time = time.time() - start_time
            
            print(f"{h:<10} {eps:<10} {'Numba':<10} {numba_time:<15.6f}")
            results.append(('Numba', h, eps, numba_time))

            # PyBind11
            if cpp_laplace:
                grid_cpp = initialize_grid(n)
                start_time = time.time()

                cpp_laplace.solve(n, eps, 100000, grid_cpp)
                cpp_time = time.time() - start_time
                
                print(f"{h:<10} {eps:<10} {'C++':<10} {cpp_time:<15.6f}")
                results.append(('C++', h, eps, cpp_time))
    

    plot_results(n)

def plot_results(n: int) -> None:
    """Визуализация результата."""
    grid = initialize_grid(n)
    solve_numba(grid, 0.001)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(grid.T, origin='lower', extent=(0, 1, 0, 1), cmap='hot')
    plt.colorbar(label='u(x, y)')
    plt.title('Решение уравнения Лапласа (Вариант 12)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('solution_heatmap.png')
    print("\nГрафик сохранен как solution_heatmap.png")

if __name__ == "__main__":
    run_experiments()