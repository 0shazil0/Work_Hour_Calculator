import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
import csv

# Load the saved model
model = load_model('actions_w_w_3.h5')

# Load the MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

actions = ['working', 'not_working'] 
colors = [(0, 255, 0), (0, 0, 255)]  

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sequence = []
sentence = []
predictions = []
threshold = 0.5
current_action = 'not_working'  # Initialize with a default action
action_start_time = time.time()
csv_file = 'activity_log.csv'

# MediaPipe detection function
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
    results = model.process(image)  # Make detections
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert the image back to BGR
    return image, results

# Function to draw landmarks
def draw_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    features = np.concatenate([pose, face, lh, rh])
    
    if features.shape[0] < 1704:
        features = np.pad(features, (0, 1704 - features.shape[0]), 'constant')
    
    return features

# Define function to pad sequence
def pad_sequence(sequence, max_sequence_length, expected_dimensionality):
    # Ensure the sequence is the correct length and shape
    padded_sequence = np.zeros((max_sequence_length, expected_dimensionality))
    padded_sequence[:sequence.shape[0], :sequence.shape[1]] = sequence
    return padded_sequence

# Function to visualize probabilities (dummy implementation)
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    return output_frame

# Function to log activity to CSV file
def log_activity(action, duration):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([action, duration])

# Function to detect multiple persons using MobileNet-SSD
def detect_persons(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    persons = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Class label for person
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                persons.append((startX, startY, endX, endY))
    return persons

cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons
        persons = detect_persons(frame)

        for (startX, startY, endX, endY) in persons:
            person_image = frame[startY:endY, startX:endX]
            # Make detections
            person_image, results = mediapipe_detection(person_image, holistic)
            print(results)

            # Draw landmarks
            draw_landmarks(person_image, results)

            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                # Convert sequence to a numpy array
                sequence_array = np.array(sequence)
                # Ensure the sequence has the same shape as the padded sequences
                padded_sequence = pad_sequence(sequence_array, max_sequence_length=30, expected_dimensionality=1704)
                # Perform prediction
                res = model.predict(np.expand_dims(padded_sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                # Determine the current action
                action = actions[np.argmax(res)]

                # If the action changes, log the previous action's duration
                if current_action != action:
                    if current_action is not None:
                        duration = (time.time() - action_start_time) / 60  # Duration in minutes
                        log_activity(current_action, duration)
                    current_action = action
                    action_start_time = time.time()

                # Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                person_image = prob_viz(res, actions, person_image, colors)

            # Draw timer and action status
            elapsed_time = time.time() - action_start_time
            minutes, seconds = divmod(elapsed_time, 60)
            timer_text = f"{current_action}: {int(minutes):02}:{int(seconds):02}"
            color = colors[actions.index(current_action)] if current_action in actions else (0, 255, 255)  # Default to yellow if action is invalid

            cv2.rectangle(person_image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(person_image, ' '.join(sentence), (3, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(person_image, timer_text, (10, person_image.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Place the person image back into the original frame
            frame[startY:endY, startX:endX] = person_image

        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            # Log the final action duration before exiting
            if current_action is not None:
                duration = (time.time() - action_start_time) / 60  # Duration in minutes
                log_activity(current_action, duration)
            break

cap.release()
cv2.destroyAllWindows()
