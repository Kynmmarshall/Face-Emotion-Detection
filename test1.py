import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained emotion recognition model
model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)

# Emotion labels
emotion_dict = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (64, 64))         # Resize to 64x64
        roi_normalized = roi_resized / 255.0            # Normalize
        roi_reshaped = np.reshape(roi_normalized, (1, 64, 64, 1))  # Reshape properly
        
        prediction = model.predict(roi_reshaped)
        emotion_label = emotion_dict[np.argmax(prediction)]
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    
    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
