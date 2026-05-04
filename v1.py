import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
import numpy as np
import os
import warnings
# ==========================================
# 0. Настройка CUDA и GPU
# ==========================================
# Проверяем наличие доступных видеокарт
physical_devices = tf.config.list_physical_devices('GPU')

if len(physical_devices) > 0:
    print(f"✅ Обнаружено GPU: {len(physical_devices)} шт.")
    for gpu in physical_devices:
        print(f"   - {gpu.name}")
    try:
        # Включаем динамическое выделение памяти для каждой видеокарты.
        # Это позволяет TensorFlow брать память по мере необходимости,
        # а не резервировать все 100% VRAM сразу.
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Динамическое выделение памяти GPU включено.\n")
    except RuntimeError as e:
        # Ошибка может возникнуть, если виртуальные устройства уже были инициализированы
        print(e)
else:
    print("❌ GPU не обнаружено. Вычисления будут производиться на медленном CPU.")
    print("Убедитесь, что у вас установлены NVIDIA Drivers, CUDA Toolkit и cuDNN.\n")

# ==========================================
# 1. Основные параметры
# ==========================================
BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150
EPOCHS = 15
DATASET_DIR = "dataset"  # Путь к вашей папке с данными

print("Загрузка данных...")

# 2. Загрузка обучающей выборки (80% данных)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

# 3. Загрузка валидационной выборки (20% данных)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(f"Обнаружены классы: {class_names}")

# Оптимизация загрузки данных для GPU
# AUTOTUNE позволяет TensorFlow самому решать, сколько потоков процессора
# выделить на подготовку картинок, чтобы видеокарта никогда не простаивала без данных.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. Аугментация данных
data_augmentation = Sequential([
    layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# 5. Построение архитектуры нейронной сети
model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# 6. Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7. Обучение сети
print("Начало обучения на GPU...")
# Именно здесь, внутри model.fit, TensorFlow автоматически отправит
# все матричные вычисления на ядра CUDA вашей видеокарты NVIDIA.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

model.save('bag_classifier.keras')
print("Модель обучена и сохранена как bag_classifier.keras")


# ==========================================
# 8. Функция для проверки новых фотографий
# ==========================================
def predict_new_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions[0])
    confidence = 100 * np.max(predictions[0])
    predicted_class_name = class_names[predicted_class_index]

    print(f"На фото: {predicted_class_name} с вероятностью {confidence:.2f}%")
    return predicted_class_name