import cv2
import os
import numpy as np
import random
import mediapipe as mp
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model_face.keras")
ABSENSI_FILE = os.path.join(BASE_DIR, "absensi.csv")

IMG_SIZE = 100

# ==============================
# MEDIAPIPE
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# ==============================
# ABSENSI FUNCTION
# ==============================
def simpan_absensi(nama):
    now = datetime.now()
    tanggal = now.strftime("%Y-%m-%d")
    jam = now.strftime("%H:%M:%S")

    if not os.path.exists(ABSENSI_FILE):
        df = pd.DataFrame(columns=["nama", "tanggal", "jam"])
        df.to_csv(ABSENSI_FILE, index=False)

    df = pd.read_csv(ABSENSI_FILE)

    # cek sudah absen hari ini
    if len(df[(df["nama"] == nama) & (df["tanggal"] == tanggal)]) > 0:
        return False

    new_data = pd.DataFrame([[nama, tanggal, jam]],
                            columns=["nama", "tanggal", "jam"])

    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(ABSENSI_FILE, index=False)

    return True

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
# LOAD DATASET LABEL
# ==============================
def load_labels():
    label_dict = {}
    kelas = sorted(os.listdir(DATASET_PATH))

    for i, nama in enumerate(kelas):
        if os.path.isdir(os.path.join(DATASET_PATH, nama)):
            label_dict[i] = nama

    return label_dict

# ==============================
# REALTIME ABSENSI
# ==============================
def realtime():
    model = load_model(MODEL_PATH)
    label_dict = load_labels()

    cam = cv2.VideoCapture(0)

    challenge = random.choice(["BLINK", "LOOK_LEFT", "LOOK_RIGHT"])
    verified = False
    prev_nose = None
    counter_ok = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            cv2.putText(frame, "NO FACE DETECTED", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("ABSENSI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for face_landmarks in results.multi_face_landmarks:

            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # =========================
            # LIVENESS CHECK
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

            # movement check
            if prev_nose is not None:
                move = np.linalg.norm(np.array(nose) - np.array(prev_nose))
                if move < 1.5:
                    verified = False

            prev_nose = nose

            # =========================
            # FACE AREA
            # =========================
            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]

            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # anti spoof texture
            texture = cv2.Laplacian(gray, cv2.CV_64F).var()
            if texture < 40:
                cv2.putText(frame, "SPOOF DETECTED", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                continue

            # =========================
            # PREPROCESS
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

            entropy = -np.sum(pred * np.log(pred + 1e-10))

            # =========================
            # STABILITY CHECK
            # =========================
            if confidence > 0.85 and entropy < 1.5 and verified:
                counter_ok += 1
            else:
                counter_ok = 0

            # =========================
            # FINAL ABSENSI
            # =========================
            if counter_ok >= 5:
                nama = label_dict[label_index]

                sukses = simpan_absensi(nama)

                if sukses:
                    text = f"{nama} - ABSENSI BERHASIL"
                    color = (0, 255, 0)
                else:
                    text = f"{nama} - SUDAH ABSEN"
                    color = (0, 255, 255)

                cv2.putText(frame, text, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                cv2.putText(frame, "VERIFIKASI WAJAH...", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            cv2.putText(frame, f"Challenge: {challenge}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        cv2.imshow("SISTEM ABSENSI FACE AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    realtime()