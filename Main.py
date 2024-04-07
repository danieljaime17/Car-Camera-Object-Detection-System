import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions


# Load pre-trained object detection model
model = MobileNetV2(weights='imagenet')

# Function to process frames and detect objects
def detect_objects(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)
    predictions = model.predict(frame)
    return decode_predictions(predictions)

# Video processing
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Detect objects in the current frame
    predictions = detect_objects(frame)
    
    # Draw bounding boxes and labels for detected objects
    for pred in predictions[0]:
        class_id = int(pred[0])
        class_name = decode_predictions(np.array([class_id]))[0][0][1]
        confidence = pred[2]
        left, top, right, bottom = int(pred[3][0]), int(pred[3][1]), int(pred[3][2]), int(pred[3][3])
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name}: {confidence:.2f}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show processed frame
    cv2.imshow('Detected Objects', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
