from collections.abc import Iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

# Initialization
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.vl.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5  # Confidence threshold

print("[INFO] Loading face detector...")
prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] Loading face embedder...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

print("[INFO] Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)

# 🔹 Initialize name collector for attendance
detected_names = []

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0),
                                             swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            if proba > 0.7:  # Only consider strong predictions
                print(f"{name} : {proba * 100:.2f}%")
                detected_names.append(name)

            # Draw box and label
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.putText(frame, "Press 'q' or 'Esc' to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

# 🔹 Attendance Processing
all_students = []
with open("student.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        all_students.append(row[0].strip())

present_students = list(set(detected_names))

with open("Attendance.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Attendance"])
    for student in all_students:
        if student in present_students:
            writer.writerow([student, "Present"])
        else:
            writer.writerow([student, "Absent"])
