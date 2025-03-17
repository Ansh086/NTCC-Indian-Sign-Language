import cv2
import mediapipe as mp
import numpy as np
import math
import time
import keras.models as models

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

np.set_printoptions(suppress=True)

model = models.load_model("DataMP\!!Model\keras_model.h5")
sign_labels = open("DataMP\!!Model\labels.txt", "r").readlines()

capture = cv2.VideoCapture(0)

offset = 20
img_size = 300

while capture.isOpened():
    success, image = capture.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_white = np.ones((480, 640, 3), np.uint8) * 255
    results = hands.process(image_rgb)

    imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255

    if results.multi_hand_landmarks:
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
            mp_drawing.draw_landmarks(img_white, hand_landmarks, mp_hands.HAND_CONNECTIONS, None,
                                      mp_drawing.DrawingSpec((0, 0, 0), 6, 5))
        x_min -= offset
        y_min -= offset
        x_max += offset
        y_max += offset

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)

        imgCrop = img_white[y_min:y_max, x_min:x_max]
        imgCropShape = imgCrop.shape

        aspectRatio = imgCropShape[0] / imgCropShape[1]

        if aspectRatio > 1:
            k = img_size / imgCropShape[0]
            wCal = math.ceil(k * imgCropShape[1])
            imgResize = cv2.resize(imgCrop, (wCal, img_size))
            wGap = math.ceil((img_size - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = img_size / imgCropShape[1]
            hCal = math.ceil(k * imgCropShape[0])
            imgResize = cv2.resize(imgCrop, (img_size, hCal))
            hGap = math.ceil((img_size - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        imgWhite_resized = cv2.resize(imgWhite, (224, 224))
        imgWhite_normalized = imgWhite_resized / 255.0

        imgWhite_input = np.expand_dims(imgWhite_normalized, axis=0)

        prediction = model.predict(imgWhite_input)
        index = np.argmax(prediction)
        class_name = sign_labels[index].strip()
        confidence_score = prediction[0][index]

        cv2.putText(image, f'{class_name} ({confidence_score * 100:.2f}%)', (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 10)

    cv2.imshow("Image", image)
    cv2.imshow("ImageWhite", imgWhite)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

capture.release()
cv2.destroyAllWindows()
