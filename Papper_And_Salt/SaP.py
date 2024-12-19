import numpy as np
import cv2
from numba import cuda
import numba
import time
import os

# Функция для добавления шума "соль и перец"
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Добавляем соль
    num_salt = np.ceil(salt_prob * total_pixels).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255  # Соль - белые пиксели

    # Добавляем перец
    num_pepper = np.ceil(pepper_prob * total_pixels).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0  # Перец - черные пиксели

    return noisy_image

# CUDA ядро для медианного фильтра с 9 точками
@cuda.jit
def median_filter_kernel(input_image, output_image, width, height):
    # Определяем текстурную память
    tex = cuda.texture.Array(input_image.shape[0], input_image.shape[1], numba.cuda.dtype.uint8)

    # Получаем позицию потока
    x, y = cuda.grid(2)

    # Проверяем границы
    if x < width and y < height:
        # Создаем окно для медианного фильтра
        window = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ix = min(max(x + dx, 0), width - 1)  # Ограничиваем индекс по x
                iy = min(max(y + dy, 0), height - 1)  # Ограничиваем индекс по y
                window.append(cuda.texture.read(tex, (ix, iy)))  # Читаем значение из текстуры

        # Сортируем окно, чтобы найти медиану
        window.sort()
        output_image[y, x] = window[4]  # Медиана - это 5-й элемент

def apply_median_filter(input_image):
    # Получаем размеры изображения
    height, width = input_image.shape

    # Выделяем память для выходного изображения
    output_image = np.zeros_like(input_image)

    # Копируем входное изображение на устройство
    input_image_device = cuda.to_device(input_image)
    output_image_device = cuda.to_device(output_image)

    # Определяем размеры блоков и сетки
    block_size = (16, 16)
    grid_size = (int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1])))

    # Запускаем таймер
    start_time = time.time()

    # Запускаем ядро
    median_filter_kernel[grid_size, block_size](input_image_device, output_image_device, width, height)

    # Копируем выходное изображение обратно на хост
    output_image_device.copy_to_host(output_image)

    # Вычисляем время обработки
    processing_time = time.time() - start_time

    return output_image, processing_time

def main():
    # Запрашиваем у пользователя путь к входному изображению
    input_image_path = input("Введите путь к входному изображению: ")

    # Загружаем входное изображение
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Проверяем, было ли загружено изображение
    if input_image is None:
        print("Ошибка: Не удалось загрузить изображение.")
        return

    # Добавляем шум "соль и перец"
    noisy_image = add_salt_and_pepper_noise(input_image, salt_prob=0.02, pepper_prob=0.02)

    # Сохраняем зашумленное изображение
    noisy_image_path = os.path.splitext(input_image_path)[0] + "_noised.bmp"
    cv2.imwrite(noisy_image_path, noisy_image)
    print(f"Зашумленное изображение сохранено по пути: {noisy_image_path}")

    # Применяем медианный фильтр к зашумленному изображению
    output_image, processing_time = apply_median_filter(noisy_image)

    # Определяем путь для выходного изображения
    output_image_path = os.path.join(os.path.dirname (input_image_path), "output_image.bmp")

    # Сохраняем выходное изображение
    cv2.imwrite(output_image_path, output_image)

    # Выводим время обработки
    print(f"Время обработки: {processing_time:.4f} секунд")
    print(f"Выходное изображение сохранено по пути: {output_image_path}")

if __name__ == "__main__":
    main()