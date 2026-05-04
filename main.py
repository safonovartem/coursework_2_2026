import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
import numpy as np
import os
import warnings
import torch
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Основные параметры
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

# Оптимизация загрузки данных в память
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. Аугментация данных (искусственное расширение датасета)
data_augmentation = Sequential([
    layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# 5. Построение архитектуры нейронной сети
model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255),  # Нормализация пикселей от 0 до 1

    # Первый сверточный блок
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Второй сверточный блок
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Третий сверточный блок
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Переход к полносвязным слоям
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Защита от переобучения
    layers.Dense(len(class_names), activation='softmax')  # Выходной слой (3 класса)
])

# 6. Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7. Обучение сети
print("Начало обучения...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Сохранение обученной модели
# после обучения модели
model.save('bag_classifier.keras')

# сохраняем имена классов
np.save('class_names.npy', class_names)

print("Модель и классы сохранены")

