import tensorflow as tf
import numpy as np
import os

if not os.path.exists('bag_classifier.keras'):
    print("Модель не найдена! Сначала обучи её.")
    exit()

IMG_HEIGHT = 150
IMG_WIDTH = 150

# 1. Загружаем модель
model = tf.keras.models.load_model('bag_classifier.keras')

# 2. Загружаем названия классов
class_names = np.load('class_names.npy', allow_pickle=True)

print("Модель загружена")
print("Классы:", class_names)


def predict_image(img_path):
    # загрузка изображения
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)

    # нормализация (ВАЖНО — как при обучении)
    #img_array = img_array / 255.0

    # добавляем batch dimension
    img_array = tf.expand_dims(img_array, 0)

    # предсказание
    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions[0])
    confidence = 100 * np.max(predictions[0])

    predicted_class_name = class_names[predicted_class_index]

    print(f"На фото: {predicted_class_name} ({confidence:.2f}%)")

    return predicted_class_name


# ===== пример =====
if __name__ == "__main__":
    path = r'C:\Users\PC\Downloads\images.jpg'
    predict_image(path)