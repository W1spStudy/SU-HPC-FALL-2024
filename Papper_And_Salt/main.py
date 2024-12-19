import cv2

def convert_jpg_to_bmp(input_jpg_path, output_bmp_path):
    # Загружаем изображение в формате JPG
    image = cv2.imread(input_jpg_path)

    # Проверяем, было ли загружено изображение
    if image is None:
        print("Ошибка: Не удалось загрузить изображение. Проверьте путь к файлу.")
        return

    # Сохраняем изображение в формате BMP
    cv2.imwrite(output_bmp_path, image)
    print(f"Изображение сохранено в формате BMP по пути: {output_bmp_path}")

# Пример использования функции
if __name__ == "__main__":
    input_jpg_path = input("Введите путь к изображению в формате JPG: ")
    output_bmp_path = input("Введите путь для сохранения изображения в формате BMP: ")
    convert_jpg_to_bmp(input_jpg_path, output_bmp_path)