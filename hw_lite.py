# Последовательная модель НС
from tensorflow.keras.models import Sequential

# Основные слои
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization

# Утилиты для to_categorical()
from tensorflow.keras import utils

# Алгоритмы оптимизации для обучения модели
from tensorflow.keras.optimizers import Adam, Adadelta

# Библиотека для работы с массивами
import numpy as np

# Библиотека для работы с таблицами
import pandas as pd

# Отрисовка графиков
import matplotlib.pyplot as plt

# Связь с google-диском
from google.colab import files

# Предварительная обработка данных
from sklearn import preprocessing

# Разделение данных на выборки
from sklearn.model_selection import train_test_split

# Для загрузки датасета
from keras.datasets import fashion_mnist

# Отрисовывать изображения в ноутбуке, а не в консоль или файл
%matplotlib inline


# Задание
#
# Используя шаблон ноутбука для распознавания видов одежды и аксессуаров из набора fashion_mnist,
# выполните следующие действия:
#
#     Создайте 9 моделей нейронной сети с различными архитектурами и сравните в них значения точности
#     на проверочной выборке (на последней эпохе) и на тестовой выборке.
#     Используйте следующее деление: обучающая выборка - 50000 примеров, проверочная выборка - 10000 примеров,
#
#     тестовая выборка - 10000 примеров.
#
#     Заполните сравнительную таблицу в конце ноутбука, напишите свои выводы по результатам проведенных тестов.
#
# Шаблон ноутбука
# Импорт библиотек
# Описание базы
# База: одежда, обувь и аксессуары
#
#     Датасет состоит из набора изображений одежды, обуви, аксессуаров и их классов.
#     Изображения одного вида хранятся в numpy-массиве (28, 28) - x_train, x_test.
#     База содержит 10 классов: (Футболка, Брюки, Пуловер, Платье, Пальто, Сандалии/Босоножки, Рубашка, Кроссовки, Сумочка, Ботильоны) - y_train, y_test.
#     Примеров: train - 60000, test - 10000.

# Загрузка датасета
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Вывод размерностей выборок

print('Размер x_train:',x_train.shape)
print('Размер y_train:',y_train.shape)
print('Размер x_test:',x_test.shape)
print('Размер y_test:',y_test.shape)

# Выбор 1 изображения каждого класса
imgs = np.array([x_train[y_train==i][0] for i in range(10)])

# Соединение изображения в одну линию
imgs = np.concatenate(imgs, axis=1)

# Создание поля для изображения
plt.figure(figsize=(30, 6))

# Отрисовка итогового изображения
plt.imshow(imgs, cmap='Greys_r')

# Без сетки
plt.grid(False)

# Без осей
plt.axis('off')

# Вывод результата
plt.show()

# Ваше решение
CLASS_COUNT = 10
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
# x_train = x_train.reshape(x_train.shape[0], 784)
# x_test = x_test.reshape(x_test.shape[0], 784)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = utils.to_categorical(y_train, CLASS_COUNT)
y_test = utils.to_categorical(y_test, CLASS_COUNT)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#1
model = Sequential()
model.add(Dense(500, input_dim = 784, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(CLASS_COUNT, activation = 'softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
history_1 = model.fit(x_train,
                    y_train,
                    batch_size=100,
                    validation_split=0.1,
                    epochs=100,
                    verbose=1)

plt.plot(history_1.history['loss'],
         label='Ошибка на обучающем наборе')

plt.plot(history_1.history['val_loss'],
         label='Ошибка на проверочном наборе')

plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')

plt.legend()

plt.show()

#2
model = Sequential()
model.add(Dense(500, input_dim = 784, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(CLASS_COUNT, activation = 'softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
history_2 = model.fit(x_train,
                    y_train,
                    batch_size=100,
                    validation_split=0.1,
                    epochs=100,
                    verbose=1)

plt.plot(history_2.history['loss'],
         label='Ошибка на обучающем наборе')

plt.plot(history_2.history['val_loss'],
         label='Ошибка на проверочном наборе')

plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')

plt.legend()

plt.show()

#3
model = Sequential()
model.add(Dense(5000, input_dim = 784, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(CLASS_COUNT, activation = 'softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
history_3 = model.fit(x_train,
                    y_train,
                    batch_size=100,
                    validation_split=0.1,
                    epochs=100,
                    verbose=1)

plt.plot(history_3.history['loss'],
         label='Ошибка на обучающем наборе')

plt.plot(history_3.history['val_loss'],
         label='Ошибка на проверочном наборе')

plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')

plt.legend()

plt.show()



