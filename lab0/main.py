import numpy as np
import time
from numba import cuda
import math
from matplotlib import pyplot as plt

matrix_size = 100

# Инициализация матриц для CPU
cpu_matrix1 = np.random.randint(0, 10, (matrix_size, matrix_size))
cpu_matrix2 = np.random.randint(0, 10, (matrix_size, matrix_size))
cpu_matrix_res = np.zeros((matrix_size, matrix_size), dtype=int)

# Инициализация матриц для GPU
gpu_matrix1 = cuda.to_device(cpu_matrix1)
gpu_matrix2 = cuda.to_device(cpu_matrix2)
gpu_matrix_res = cuda.device_array((len(cpu_matrix1), len(cpu_matrix2)))

# Функция матричного умножения на CPU
def cpu_mat_mul(A, B, C):
    for i in range(matrix_size):
        for j in range(matrix_size):
            res = 0
            for k in range(matrix_size):
                res += A[i, k] * B[k, j]
            C[i, j] = res

def cpu_calc():
    print("CPU начинает работу.")
    start_time = time.time()
    cpu_mat_mul(cpu_matrix1, cpu_matrix2, cpu_matrix_res)
    cpu_time = time.time() - start_time
    print(f"{cpu_time} секунд - время вычисления на CPU")
    return cpu_time

@cuda.jit
def gpu_mat_mul(A, B, C):
    for i in range(matrix_size):
        for j in range(matrix_size):
            rez = 0
            for z in range(matrix_size):
                rez += A[i, z] * B[z, j]
            C[i, j] = rez

def gpu_calc():
    # Параметры ядра
    threadsperblock = (32, 32)
    blockspergrid_x = int(math.ceil(cpu_matrix1.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(cpu_matrix2.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print(f"Размер сетки = {blockspergrid}, {threadsperblock}")
    print("Конец работы CPU!\n")

    print("GPU начинает свою работу...")
    start_time = time.time()
    gpu_mat_mul[blockspergrid, threadsperblock](gpu_matrix1, gpu_matrix2, gpu_matrix_res)
    gpu_time = time.time() - start_time
    print(f"{gpu_time} секунд - время вычисления на GPU")
    print("Конец работы GPU!\n")
    return gpu_time

if __name__ == "__main__":
    cpu_time = cpu_calc()
    gpu_time = gpu_calc()
    
    x = [0.20007705688476562, 1.621795415878296, 13.972699165344238, 105.11987590789795, 956.7778902053833,
         1989.2391362190247]
    y = [0.15534392356872559, 0.16351268768310547, 0.1880006504058838, 0.2145817470550537, 0.25744752883911133,
         0.2705642986297607]
    z = [100, 200, 400, 800, 1600, 2000]

    # Создание субплотов
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # График времени работы CPU
    axs[0].plot(x, z)
    axs[0].set_title("Время работы CPU")
    axs[0].set_xlabel("Размер блоков")
    axs[0].set_ylabel("Время работы в секундах")
    axs[0].grid()

    # График времени работы GPU
    axs[1].plot(z, y)
    axs[1].set_title("Время работы GPU")
    axs[1].set_xlabel("Размер блоков")
    axs[1].set_ylabel("Время работы в секундах")
    axs[1].grid()

    # График ускорения
    vsp = [x[i] / y[i] for i in range(len(x))]
    axs[2].plot(z, vsp)
    axs[2].set_title("Ускорение")
    axs[2].set_xlabel("Размер блоков")
    axs[2].grid()

    plt.tight_layout()
    plt.savefig("performance_graphs.png")  # Сохранение граф