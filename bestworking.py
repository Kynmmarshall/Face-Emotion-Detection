import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load emotion recognition model
model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)
print(f"Model input shape: {model.input_shape}")

# Emotion labels with corresponding colors
emotion_dict = {
    'Angry': (0, 0, 255),       # Red
    'Disgust': (0, 102, 0),      # Dark Green
    'Fear': (153, 0, 153),       # Purple
    'Happy': (0, 255, 25),      # green
    'Suprise': (0, 165, 255),    # Orange
    'Sad': (255, 0, 0),          # Blue
    'Neutral': (200, 200, 200),  # Light Gray
}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set camera resolution for better face detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Load face detector - using more accurate Face Detection Model
prototxt_path = "dat.prototxt"
model_path = "caffe.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Get model's expected input size
input_height, input_width = 64, 64  # Updated to match XCEPTION input

# Performance tracking
frame_count = 0
start_time = time.time()
fps = 0

# Preprocessing parameters •	Enhances image contrast especially in low-light or over-exposed areas.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    if frame_count % 10 == 0:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()
    
    # Mirror the frame for more natural interaction (like selfie view)
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
        
        # Filter weak detections with higher threshold
        if confidence > 0.7:  # Increased confidence threshold
            # Compute face coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Expand face region for better context
            expand_w = int((endX - startX) * 0.1)
            expand_h = int((endY - startY) * 0.1)
            startX, startY = max(0, startX - expand_w), max(0, startY - expand_h)
            endX, endY = min(w - 1, endX + expand_w), min(h - 1, endY + expand_h)
            
            # Skip if face region is too small
            if (endY - startY) < 5 or (endX - startX) < 5:
                continue
                
            try:
                # Extract face ROI (region of interest) 
                face_roi = frame[startY:endY, startX:endX]
                
                # Convert to grayscale and apply histogram equalization
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                enhanced_face = clahe.apply(gray_face)
                
                # Resize for model with better interpolation
                resized = cv2.resize(enhanced_face, (input_width, input_height), 
                                     interpolation=cv2.INTER_CUBIC)
                
                # Normalize and prepare for model
                normalized = resized.astype('float32') / 255.0
                input_tensor = normalized.reshape(1, input_height, input_width, 1)

                # Predict emotion
                predictions = model.predict(input_tensor, verbose=0)
                emotion_idx = np.argmax(predictions)
                emotion_name = list(emotion_dict.keys())[emotion_idx]
                confidence = np.max(predictions)
                
                # Only display high-confidence results
                if confidence > 0.3:  # Confidence threshold for display
                    # Get color for this emotion
                    color = emotion_dict[emotion_name]
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                    
                    # Create background for text
                    text_size = cv2.getTextSize(emotion_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, 
                                (startX, startY - 40),
                                (startX + text_size[0] + 20, startY),
                                color, -1)
                    
                    # Display emotion text
                    text = f"{emotion_name} {confidence*100:.1f}%"
                    cv2.putText(frame, text, (startX + 10, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
            except Exception as e:
                # Skip frame if any processing error occurs
                continue

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),2)
    
    # Display instructions
    cv2.putText(frame, "Press 'Q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow("Emotion Detector", frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()