import cv2
import numpy as np
import threading

# Функция для обнаружения людей на кадре
def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personIdx and confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            results.append((x, y, x + w, y + h))

    return results

# Функция для обработки видеопотока с одной камеры
def process_video_stream(video_path, net, ln, door_coords, camera_id):
    vs = cv2.VideoCapture(video_path)

    # Проверка успешного открытия видеопотока
    if not vs.isOpened():
        print(f"Ошибка: не удается открыть видеофайл {video_path}")
        return

    door_x1, door_y1, door_x2, door_y2 = door_coords

    # Счетчики
    people_entered = 0

    # Словарь для хранения координат предыдущих кадров для каждого объекта
    object_previous_positions = {}

    # Цикл по каждому кадру в видеопотоке
    while True:
        # Считываем кадр из видеопотока
        ret, frame = vs.read()

        # Проверяем, получен ли кадр
        if not ret:
            break

        # Обнаруживаем людей на кадре
        boxes = detect_people(frame, net, ln)

        # Рисуем bounding box'ы вокруг обнаруженных людей на кадре
        current_positions = []
        for (startX, startY, endX, endY) in boxes:
            # Находим центр объекта
            centerX = int((startX + endX) / 2)
            centerY = int((startY + endY) / 2)

            # Добавляем текущую позицию в список
            current_positions.append((centerX, centerY))

            # Проверка пересечения линии дверей
            if (door_x1 < centerX < door_x2) and (door_y1 < centerY < door_y2):
                if (centerX, centerY) in object_previous_positions:
                    prevX, prevY = object_previous_positions[(centerX, centerY)]
                    if prevY < door_y1 and centerY >= door_y1:
                        people_entered += 1

            # Обновление предыдущей позиции для текущего объекта
            object_previous_positions[(centerX, centerY)] = (centerX, centerY)

        # Удаление старых позиций объектов, которые больше не видны
        object_previous_positions = {key: value for key, value in object_previous_positions.items() if key in current_positions}

        # Рисуем область дверей
        cv2.rectangle(frame, (door_x1, door_y1), (door_x2, door_y2), (0, 0, 255), 2)

        # Отображение количества вошедших людей
        cv2.putText(frame, f'Camera {camera_id} Entered: {people_entered}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Показываем результат
        cv2.imshow(f"Camera {camera_id}", frame)
        key = cv2.waitKey(1) & 0xFF

        # Выход из цикла по нажатию клавиши 'q'
        if key == ord('q'):
            break

    # Очищаем ресурсы
    vs.release()
    cv2.destroyAllWindows()

# Загружаем модель для обнаружения объектов (в данном случае - для обнаружения людей)
weightsPath = "yolov3-tiny.weights"  # Путь к весам модели
configPath = "yolov3-tiny.cfg"  # Путь к конфигурационному файлу модели
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Получаем названия слоев в нейронной сети
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Список путей к видеопотокам и координаты дверей для каждой камеры
video_streams = [
    ("camera1.mp4", (200, 100, 400, 400)),
    ("camera2.mp4", (150, 50, 350, 350)),
    ("camera3.mp4", (100, 150, 300, 300)),
    ("camera4.mp4", (120, 180, 320, 320))
]

# Запуск обработки видеопотоков в отдельных потоках
threads = []
for i, (video_path, door_coords) in enumerate(video_streams):
    t = threading.Thread(target=process_video_stream, args=(video_path, net, ln, door_coords, i + 1))
    t.start()
    threads.append(t)

# Ожидание завершения всех потоков
for t in threads:
    t.join()
