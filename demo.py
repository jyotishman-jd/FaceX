import cv2
import numpy as np
import joblib

# Load the trained SVM model and label encoder
svm_model = joblib.load('output_model/svm_model.joblib')
label_encoder = joblib.load('output_model/label_encoder.joblib')

# Function to calculate LBP for a single pixel
def calculate_lbp_pixel(img, x, y):
    center = img[x, y]
    code = 0
    power_val = 0

    for i in range(8):
        new_x = x + int(np.round(np.cos(i * 2 * np.pi / 8)))
        new_y = y - int(np.round(np.sin(i * 2 * np.pi / 8)))

        if img[new_x, new_y] >= center:
            code += 2 ** power_val
        power_val += 1

    return code

# Function to compute LBP for the entire image
def calculate_lbp_image(img):
    lbp_img = np.zeros_like(img)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            lbp_img[i, j] = calculate_lbp_pixel(img, i, j)

    return lbp_img.flatten()




def recognize_face(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection using a pre-trained Haarcascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray_frame[y:y + h, x:x + w]

        # Perform LBP feature extraction
        lbp_features = calculate_lbp_image(face_roi).flatten()

        # Reshape features for prediction
        lbp_features = lbp_features[:500].reshape(1, -1)

        # Make prediction using the loaded SVM model
        prediction = svm_model.predict(lbp_features)

        # Convert numerical label back to original string label
        predicted_label = label_encoder.inverse_transform(prediction)

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Predicted: {predicted_label[0]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Resize the frame to a smaller resolution
    # frame = cv2.resize(frame, (1280, 720))

    # Perform real-time face recognition
    predicted_label = recognize_face(frame)

    # Display the frame with recognized label
    cv2.putText(frame, f"Predicted: {predicted_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-time Face Recognition", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
