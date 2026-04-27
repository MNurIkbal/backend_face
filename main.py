import cv2
import os
import numpy as np
import time
import random
import mediapipe as mp
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model_face.keras")
IMG_SIZE = 100
CONFIDENCE_THRESHOLD = 0.7

# ==============================
# MEDIAPIPE INIT
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# ==============================
# BLINK DETECTION
# ==============================
def detect_blink(landmarks):
    left_eye = [33, 160, 158, 133, 153, 144]

    def ear(eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (A + B) / (2.0 * C)

    return ear([landmarks[i] for i in left_eye]) < 0.20

# ==============================
# 1. AMBIL DATA
# ==============================
def ambil_data(nama):
    path = os.path.join(DATASET_PATH, nama)
    os.makedirs(path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    count = 0

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            cv2.imwrite(os.path.join(path, f"{count}.jpg"), face)
            count += 1

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.putText(frame, f"Count: {count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Ambil Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

# ==============================
# 2. LOAD DATASET
# ==============================
def load_dataset():
    data = []
    label = []
    label_dict = {}

    kelas = sorted(os.listdir(DATASET_PATH))  # FIX URUTAN STABIL

    for i, nama in enumerate(kelas):
        folder = os.path.join(DATASET_PATH, nama)

        if not os.path.isdir(folder):
            continue

        label_dict[i] = nama

        for file in os.listdir(folder):
            if not file.endswith(".jpg"):
                continue

            img = cv2.imread(os.path.join(folder, file), 0)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            data.append(img)
            label.append(i)

    data = np.array(data, dtype=np.float32) / 255.0
    data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    label = np.array(label)

    print("[INFO] Kelas terbaca:", label_dict)

    return data, label, label_dict
# ==============================
# 3. MODEL CNN
# ==============================
def buat_model(jumlah_kelas):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(jumlah_kelas, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# ==============================
# 4. TRAIN MODEL
# ==============================
def train_model():
    data, label, label_dict = load_dataset()

    model = buat_model(len(label_dict))
    model.fit(data, label, epochs=30, shuffle=True)

    model.save(MODEL_PATH)
    print("[INFO] Model disimpan!")

# ==============================
# 5. REALTIME (ANTI FOTO + VIDEO)
# ==============================
def realtime(label_dict):
    model = load_model(MODEL_PATH)
    cam = cv2.VideoCapture(0)

    challenge = random.choice(["BLINK", "LOOK_LEFT", "LOOK_RIGHT"])
    verified = False
    prev_nose = None

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            cv2.putText(frame, "NO FACE DETECTED", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("ANTI SPOOF", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for face_landmarks in results.multi_face_landmarks:

            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # =========================
            # LIVENESS CHECK (IMPROVED)
            # =========================
            nose = landmarks[1]

            if challenge == "BLINK":
                if detect_blink(landmarks):
                    verified = True

            elif challenge == "LOOK_LEFT":
                if nose[0] < w // 2 - 60:
                    verified = True

            elif challenge == "LOOK_RIGHT":
                if nose[0] > w // 2 + 60:
                    verified = True

            # movement check (lebih ketat)
            if prev_nose is not None:
                move = np.linalg.norm(np.array(nose) - np.array(prev_nose))
                if move < 1.5:
                    verified = False  # kemungkinan foto

            prev_nose = nose

            # =========================
            # FACE BOUNDING BOX FIX
            # =========================
            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]

            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # =========================
            # ANTI PHOTO / SCREEN DETECTION
            # =========================
            texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if texture_score < 40:  # threshold spoof
                cv2.putText(frame, "SPOOF DETECTED (PHOTO/SCREEN)", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                continue

            # =========================
            # FACE PREPROCESS
            # =========================
            face = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            # =========================
            # PREDICTION
            # =========================
            pred = model.predict(face, verbose=0)[0]

            confidence = np.max(pred)
            label_index = np.argmax(pred)

            # entropy check (ANTI OVERCONFIDENT FALSE MATCH)
            entropy = -np.sum(pred * np.log(pred + 1e-10))

            # =========================
            # FINAL DECISION FIX
            # =========================
            if (confidence > 0.80) and (entropy < 1.5) and verified:
                nama = label_dict[label_index]
                color = (0, 255, 0)
            else:
                nama = "TIDAK DIKETAHUI"
                color = (0, 0, 255)

            cv2.putText(frame, nama, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.putText(frame, f"Challenge: {challenge}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("ANTI SPOOF FACE RECOGNITION", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
# ==============================
# MAIN MENU
# ==============================
if __name__ == "__main__":
    print("1. Ambil Data")
    print("2. Train Model")
    print("3. Jalankan")

    pilih = input("Pilih: ")

    if pilih == "1":
        nama = input("Nama: ")
        ambil_data(nama)

    elif pilih == "2":
        train_model()

    elif pilih == "3":
        _, _, label_dict = load_dataset()
        realtime(label_dict)