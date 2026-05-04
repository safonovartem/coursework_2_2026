import keras
from tensorflow.keras.models import load_model
import tensorflow as tf

model = load_model('my_model.h5') # Загрузка Keras модели

BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150
EPOCHS = 15
class_names = train_ds.class_names

def predict_new_image(img_path):
    # Загружаем изображение и меняем его размер
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    # Преобразуем в массив чисел
    img_array = tf.keras.utils.img_to_array(img)
    # Добавляем дополнительное измерение (сеть ожидает массив изображений, а не одно)
    img_array = tf.expand_dims(img_array, 0)

    # Делаем предсказание
    predictions = model.predict(img_array)

    # Получаем индекс самого вероятного класса и процент уверенности
    predicted_class_index = np.argmax(predictions[0])
    confidence = 100 * np.max(predictions[0])

    predicted_class_name = class_names[predicted_class_index]

    print(f"На фото: {predicted_class_name} с вероятностью {confidence:.2f}%")
    return predicted_class_name

# Пример использования функции предсказания:
the_way = r'C:\Users\PC\Downloads\Telegram Desktop\photo_2026-04-29_13-19-11.jpg'
predict_new_image(the_way)