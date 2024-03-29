import cv2

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainingData.yml")

# Font configuration for displaying text
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 0, 0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        # Displaying the recognized ID on the image
        cv2.putText(img, f"ID: {id}", (x, y + h + 30), fontFace, fontScale, fontColor)

    cv2.imshow("Face", img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()