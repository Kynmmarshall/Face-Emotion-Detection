import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("face_model.h5", compile=False)
print(f"Model input shape: {model.input_shape}")

# Emotion labels with corresponding colors
emotion_dict = {
    'Angry': (0, 0, 255),       # Red
    'Disgust': (0, 102, 0),      # Dark Green
    'Fear': (153, 0, 153),       # Purple
    'Happy': (0, 255, 255),      # Yellow
    'Neutral': (200, 200, 200),  # Light Gray
    'Sad': (255, 0, 0),          # Blue
    'Surprise': (0, 165, 255)    # Orange
}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Load face detector - using more accurate DNN-based model
prototxt_path = "dat.prototxt"
model_path = "caffe.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Get model's expected input size
input_height, input_width = 48, 48  # From model summary

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Mirror the frame for more natural interaction
    frame = cv2.flip(frame, 1)
    (h, w) = frame.shape[:2]
    
    # Create blob from image for face detection
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 
        1.0, 
        (300, 300), 
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    
    # Process detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter weak detections
        if confidence > 0.5:
            # Compute face coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure coordinates stay within frame boundaries
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            
            # Extract face ROI
            face_roi = frame[startY:endY, startX:endX]
            
            # Skip if face region is too small
            if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                continue
                
            try:
                # Convert to grayscale and resize for model
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray_face, (input_width, input_height))
                
                # Normalize and prepare for model
                normalized = resized.astype('float32') / 255.0
                input_tensor = normalized.reshape(1, input_height, input_width, 1)

                # Predict emotion
                predictions = model.predict(input_tensor, verbose=0)
                emotion_idx = np.argmax(predictions)
                emotion_name = list(emotion_dict.keys())[emotion_idx]
                confidence = np.max(predictions)
                
                # Get color for this emotion
                color = emotion_dict[emotion_name]
                
                # Draw face rectangle and emotion label
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                # Create background for text
                text_bg = np.zeros((40, endX - startX, 3), dtype=np.uint8)
                text_bg[:, :] = color
                
                # Place text background above face
                if startY - 40 >= 0:
                    frame[startY-40:startY, startX:endX] = text_bg
                
                # Display emotion text
                text = f"{emotion_name} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (startX + 5, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Processing error: {e}")
                continue

    # Display instructions
    cv2.putText(frame, "Press 'Q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow("Emotion Detector", frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()