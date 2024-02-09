# app.py

from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import os
import webbrowser
import threading
import subprocess

app = Flask(__name__)
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        id = request.form['id']
        name = request.form['name']

        # Create a directory to store images if it doesn't exist
        output_folder = "dataSet"
        os.makedirs(output_folder, exist_ok=True)

        # Call the face capture function from datacollect.py
        capture_faces(id, output_folder)

        return redirect(url_for('capture_complete', message="Capture is complete!"))

    return render_template('form.html')

def capture_faces(id, output_folder):
    cam = cv2.VideoCapture(0)
    sampleNum = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleNum += 1
            img_path = os.path.join(output_folder, f"User.{id}.{sampleNum}.jpg")
            cv2.imwrite(img_path, gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.waitKey(1000)

        cv2.imshow("Face", img)
        cv2.waitKey(1)

        if sampleNum > 20:
            break

    cam.release()
    cv2.destroyAllWindows()

@app.route('/capture_complete')
def capture_complete():
    return render_template('capture_complete.html', message=request.args.get('message', ''))

@app.route('/start_training', methods=['GET', 'POST'])
def start_training():
    try:
        # Run recognition2.py using subprocess
        subprocess.run(['python', 'recognition2.py'], check=True)
        message = "Training completed successfully!"
    except subprocess.CalledProcessError:
        message = "Error occurred during training."

    return render_template('training_complete.html', message=message)

@app.route('/check_face', methods=['POST'])
def check_face():
    if request.method == 'POST':
        try:
            # Run detector.py using subprocess
            subprocess.run(['python', 'detector.py'], check=True)
            return "Face checked successfully!"
        except subprocess.CalledProcessError as e:
            return f"Error: {e}"
    else:
        # Handle GET requests if needed
        return "This route only accepts POST requests."

if __name__ == '__main__':
    app_thread = threading.Thread(target=app.run, kwargs={'debug': True, 'host': '127.0.0.1', 'port': 5000, 'use_reloader': False})
    app_thread.start()

    webbrowser.open('http://127.0.0.1:5000/')

    app_thread.join()
