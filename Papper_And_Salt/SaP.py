from numba import cuda
import numpy as np
import cv2
import time
import math

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """Добавляет шум соли и перца к изображению."""
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

@cuda.jit
def mf_kernel(input_image, output_image, width, height):
    """CUDA ядро для медианного фильтра 9-точек."""
    # Определяем позицию потока в сетке
    x, y = cuda.grid(2)

    if x < 1 or y < 1 or x >= width - 1 or y >= height - 1:
        # Пропускаем граничные пиксели (нельзя применить 3x3 фильтр)
        return

    # Создаем список для хранения 9 пикселей из области 3x3
    window = cuda.local.array(9, dtype=np.float32)

    idx = 0
    for j in range(-1, 2):
        for i in range(-1, 2):
            window[idx] = input_image[y + j, x + i]
            idx += 1

    # Сортировка пузырьком для нахождения медианы
    for i in range(9):
        for j in range(i + 1, 9):
            if window[i] > window[j]:
                temp = window[i]
                window[i] = window[j]
                window[j] = temp

    # Присваиваем медианное значение выходному изображению
    output_image[y, x] = window[4]

def mf_gpu(input_image):
    """Применяет медианный фильтр 9-точек с использованием CUDA."""
    # Преобразуем входное изображение в float32 для совместимости
    input_image = input_image.astype(np.float32)

    # Получаем размеры изображения
    height, width = input_image.shape

    # Выделяем память для выходного изображения
    output_image = np.zeros((height, width))

    # Копируем данные на устройство
    cuda_input = cuda.to_device(input_image)
    cuda_output = cuda.to_device(output_image)

    # Определяем размеры блока и сетки
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Запускаем ядро
    mf_kernel[blocks_per_grid, threads_per_block](cuda_input, cuda_output, width, height)

    # Копируем результат обратно на хост
    output_image = cuda_output.copy_to_host()

    return output_image

def save_image(image, filename):
    """Сохраняет изображение в файл."""
    # Преобразуем изображение в формат uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, image)

# Запрос пути к файлу у пользователя
file_path = input("путь к изображению: ")

# Запрос у пользователя, хочет ли он зашумить изображение
choice = input("Введите 1 для зашумления изображения или 0 для выбора зашумленного файла: ")

if choice == '1':
    # Загружаем изображение
    src_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Проверка на успешную загрузку изображения
    if src_image is None:
        print(f"Ошибка: Не удалось загрузить изображение из файла '{file_path}'. Проверьте путь и формат файла.")
    else:
        # Добавляем шум
        noisy_image = add_salt_and_pepper_noise(src_image, salt_prob=0.05, pepper_prob=0.05)
        
        # Сохраняем зашумленное изображение
        save_image(noisy_image, 'noisy_image.bmp')

elif choice == '0':
    # Загружаем зашумленное изображение
    noisy_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # Применяем медианный фильтр
    start_gpu = time.time()
    filtered_image = mf_gpu(noisy_image)
    time_gpu = time.time() - start_gpu

    # Сохраняем результирующее изображение
    save_image(filtered_image, 'filtered_image.bmp')

    # Время
    print("Время (GPU): ", time_gpu)
    print("Отфильтрованное изображение сохранено как 'filtered_image.bmp'.")
