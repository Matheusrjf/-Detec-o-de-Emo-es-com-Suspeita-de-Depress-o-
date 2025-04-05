import cv2
from tensorflow.keras.models import load_model
import numpy as np
from collections import Counter, deque
import time

# Carregar modelo de detecção de expressões faciais
model = load_model('modelo_emocoes.h5')

# Dicionário de emoções
emotion_labels = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

# Fila para registrar últimas emoções
emotion_window = deque(maxlen=100)

# Função de detecção
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    predictions = model.predict(face, verbose=0)
    return np.argmax(predictions[0])

# Função para analisar sinais de depressão
def check_depression(emotion_history):
    counter = Counter(emotion_history)
    total = sum(counter.values())
    sad_ratio = counter.get(4, 0) / total  # 4 = Sad
    neutral_ratio = counter.get(6, 0) / total  # 6 = Neutral

    if total >= 50:  # análise após 50 quadros (~10 segundos)
        if sad_ratio > 0.4 and neutral_ratio > 0.4:
            return True
    return False

# Captura de vídeo
cap = cv2.VideoCapture(0)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emotion_index = detect_emotion(frame)
    emotion = emotion_labels[emotion_index]
    emotion_window.append(emotion_index)

    # Mostrar emoção no vídeo
    cv2.putText(frame, f"Emotion: {emotion}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Checar possível depressão
    if check_depression(emotion_window):
        cv2.putText(frame, "SUSPEITA DE DEPRESSÃO", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Mostrar tempo de execução
    elapsed = int(time.time() - start_time)
    cv2.putText(frame, f"Tempo: {elapsed}s", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # Exibir vídeo
    cv2.imshow("Monitor de Humor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerrar
cap.release()
cv2.destroyAllWindows()
