import cv2
import numpy as np
import cupy as cp
import time


def harris_corner_detection(image, threshold):
    # Преобразуем изображение в тип float32
    image = np.float32(image)

    # Вычисляем градиенты
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Вычисляем элементы матрицы A
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # Применяем гауссово размытие
    Ixx = cv2.GaussianBlur(Ixx, (5, 5), sigmaX=1)
    Ixy = cv2.GaussianBlur(Ixy, (5, 5), sigmaX=1)
    Iyy = cv2.GaussianBlur(Iyy, (5, 5), sigmaX=1)

    # Формируем матрицу A и вычисляем R
    det_A = Ixx * Iyy - Ixy * Ixy
    trace_A = Ixx + Iyy
    R = det_A - 0.04 * (trace_A ** 2)

    # Нахождение углов
    corners = np.zeros_like(R)
    corners[R > threshold] = 255

    return corners.astype(np.uint8)

def main(source_image_path, threshold, output_image_path):
    # Чтение изображения
    image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)

    # Запуск алгоритма без GPU
    start_time = time.time()
    corners_cpu = harris_corner_detection(image, threshold)
    cpu_time = time.time() - start_time

    # Отметка углов на исходном изображении
    output_image_cpu = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    output_image_cpu[corners_cpu > 0] = [0, 0, 255]  # Красные углы

    # Сохранение результата
    cv2.imwrite(output_image_path + "_cpu.png", output_image_cpu)

    # Запуск алгоритма с GPU
    image_gpu = cp.asarray(image)
    start_time = time.time()
    corners_gpu = harris_corner_detection(cp.asnumpy(image_gpu), threshold)
    gpu_time = time.time() - start_time

    # Отметка углов на исходном изображении
    output_image_gpu = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    output_image_gpu[corners_gpu > 0] = [0, 0, 255]  # Красные углы

    # Сохранение результата
    cv2.imwrite(output_image_path + "_gpu.png", output_image_gpu)

    # Сравнение результатов
    result_match = np.array_equal(corners_cpu, corners_gpu)

    # Вывод результатов
    print(f"Execution time without GPU: {cpu_time:.4f} seconds")
    print(f"Execution time with GPU: {gpu_time:.4f} seconds")
    print(f"Results match: {result_match}")

if __name__ == "__main__":
    # Ввод данных от пользователя
    source_image_path = input("Введите путь к исходному изображению: ")
    threshold = float(input("Введите значение порога для обнаружения углов: "))
    output_image_path = input("Введите путь для сохранения выходного изображения: ")

    main(source_image_path, threshold*100, output_image_path)