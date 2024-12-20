import time
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from tabulate import tabulate

# Количество потоков в блоке (Thread Per Block) для GPU
TPB = 16

# Количество итераций для усреднения времени выполнения
ITER = 70

# Небольшая константа для избежания деления на ноль
EPS = 10e-8

# Декоратор, указывающий, что функция будет выполняться на GPU с использованием Numba
@cuda.jit
def gpu_vec_sum(vec, res):
    tx = cuda.threadIdx.x  # Индекс потока в блоке
    bx = cuda.blockIdx.x   # Индекс блока
    idx = tx + bx * TPB    # Индекс элемента в массиве

    if idx < vec.shape[0]:
        cuda.atomic.add(res, 0, vec[idx])

# Функция для проведения измерений на CPU и GPU
def calculation():
    rows = []
    vec_size_min = 50000
    vec_size_max = 1000000
    vec_size_interval = 50000

    for vec_size in range(vec_size_min, vec_size_max + 1, vec_size_interval):
        cpu_time_sum = 0.
        gpu_time_sum = 0.

        for _ in range(ITER):
            vec = np.ones(vec_size)
            res = np.zeros(1, dtype=np.int32)

            d_vec = cuda.to_device(vec)  # Отправляем вектор на GPU
            d_res = cuda.to_device(res)  # Создаем массив для результата на GPU

            start = time.time()

            # Вызываем функцию на GPU для суммирования
            gpu_vec_sum[int((vec_size + TPB) / TPB), TPB](d_vec, d_res)

            gpu_time_sum += time.time() - start

            res = d_res.copy_to_host()  # Копируем результат с GPU на CPU

            start = time.time()
            real_res = np.sum(vec)  # Выполняем суммирование на CPU
            cpu_time = time.time() - start
            cpu_time_sum += cpu_time

        row = [vec_size,  cpu_time_sum / ITER, gpu_time_sum / ITER]
        rows.append(row)

    # Выводим результаты в виде таблицы
    print(tabulate(rows, headers=['vector size', 'cpu, ms', 'gpu, ms']))
    return rows

# Функция для построения графиков
def plots(vec_array, cpu_time_array, gpu_time_array, acceleration_array):
    # График времени работы программы на CPU
    plt.figure()
    plt.title("CPU")
    plt.plot(vec_array, cpu_time_array)
    plt.xlabel("размер вектора")
    plt.ylabel("время, мс")
    plt.grid()

    # График времени работы программы на GPU
    plt.figure()
    plt.title("GPU")
    plt.plot(vec_array, gpu_time_array)
    plt.xlabel("размер вектора")
    plt.ylabel("время, мс")
    plt.grid()

    # График ускорения вычислений на GPU относительно CPU
    plt.figure()
    plt.title("Ускорение")
    plt.plot(vec_array, acceleration_array)
    plt.xlabel("размер вектора")
    plt.grid()
    plt.show()

# Выполняем измерения и сохраняем результаты
output_data = calculation()

# Данные для построения графиков
vec_array = list(map(lambda x: x[0], output_data))
cpu_time_array = list(map(lambda x: x[1], output_data))
gpu_time_array = list(map(lambda x: x[2], output_data))
acceleration_array = list(map(lambda x: x[1] / (x[2] if x[2] > EPS else EPS), output_data))

# Строим графики
plots(vec_array, cpu_time_array, gpu_time_array, acceleration_array)
