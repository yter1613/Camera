Система подсчета входящих людей с использованием нескольких камер
Этот проект предоставляет решение для обнаружения и подсчета количества людей, входящих в определенную область (например, дверь) с использованием нескольких видеопотоков. В реализации используется модель YOLO (You Only Look Once) для обнаружения объектов в каждом кадре видеопотоков.

Возможности
Обнаружение людей с использованием модели YOLOv3-tiny.
Отслеживание и подсчет количества людей, входящих через определенную дверь.
Поддержка нескольких видеопотоков одновременно.
Отображение количества вошедших людей на кадре в реальном времени.
Требования
Python 3.x
OpenCV
NumPy

Скачайте файлы модели YOLOv3-tiny:

yolov3-tiny.weights
yolov3-tiny.cfg
Поместите эти файлы в директорию проекта.

Добавьте ваши видеофайлы:
Поместите ваши видеофайлы (например, camera1.mp4, camera2.mp4, camera3.mp4, camera4.mp4) в директорию проекта.


Обновите список видеопотоков:
Измените список video_streams в скрипте, чтобы включить пути к вашим видеофайлам и координаты для области дверей в каждом видео


Решение проблем
Видео файл не найден: Убедитесь, что пути к видеофайлам указаны правильно и файлы существуют в указанной директории.
Отсутствуют файлы модели: Скачайте файлы модели YOLOv3-tiny и поместите их в директорию проекта.
Не установлены зависимости: Убедитесь, что установлены все необходимые пакеты Python.
