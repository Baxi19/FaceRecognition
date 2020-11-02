from FaceRecognizer.LBPRecognizer import *

# Orden de las personas en el dataset
face_person = {1: "Leo", 2: "Fito", 3: "Danilo", 4: "Fabian Zamora", 5: "Randald"}

# Imagen a probar
img = cv2.imread("./partners/subject04.normal")

# Convierte a escala de grices
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Imprime un rectángulo por cada cara encontrada
faces = face_cascade.detectMultiScale(gray, 1.05, 8)
for face in faces:
    x, y, w, h = face
    face_region = gray[y:y + h, x:x + w]
    person_number, confidence = face_recognizer.predict(face_region)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(img, face_person[person_number], (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))


cv2.imshow("Face Recognized", img)
cv2.waitKey(0)
cv2.destroyAllWindows()