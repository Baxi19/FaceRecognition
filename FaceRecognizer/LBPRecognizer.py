import os
import cv2
import numpy as np

# Función para entrenar el algoritmo
def get_training_data(face_cascade, data_dir):
    images = [] # Lista para almacenar las caras
    labels = [] # Lista para almacenar las etiquetas de las caras
    # Se leen los archivos, menos los .wink para hacer pruebas más adelante
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if not f.endswith('.wink')]
    for image_file in image_files:
        img = cv2.imread(image_file, 0)  # Lee la imagen en formato de grises
        img = np.array(img)              # Se obtiene la matriz asociada a la imagen
        filename = os.path.split(image_file)[1]
        true_person_number = int(filename.split(".")[0].replace("subject", "")) #

        # Se realiza la detección de caras
        faces = face_cascade.detectMultiScale(img, 1.05, 6)
        for face in faces:
            x, y, w, h = face
            face_region = img[y:y+h, x:x+w]
            images.append(face_region)
            labels.append(true_person_number)

    return images, labels


def evaluate(face_recognizer, face_cascade, data_dir):
    # Se obtienen las imágenes que no se procesaron anteriormente para evaluarlas
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wink')]
    num_correct = 0
    for image_file in image_files:
        img = cv2.imread(image_file, 0)   # Lee la imagen en formato de grises
        img = np.array(img)               # Se obtiene la matriz asociada a la imagen
        filename = os.path.split(image_file)[1]
        true_person_number = int(filename.split(".")[0].replace("subject", "")) # Obtiene el número de la persona

        # Se realiza la detección de caras
        faces = face_cascade.detectMultiScale(img, 1.05, 6)
        for face in faces:
            x, y, w, h = face
            face_region = img[y:y + h, x:x + w]
            person_number, confidence = face_recognizer.predict(face_region)
            if person_number == true_person_number:
                num_correct += 1
                print("Correctly identified person {} with confidence {}".format(true_person_number, confidence))
            else:
                print("Incorrectly identified real person {} to false person {}".format(true_person_number, person_number))
    accuracy = (num_correct / len(image_files)) * 100
    print(f'Precisión: {accuracy}%')


# Carga el modelo de cascada
face_cascade = cv2.CascadeClassifier('.\CascadeXML\haarcascade_frontalface_default.xml')
# Se utiliza el patrón binario local (LBP) para realizar la detección de caras
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
images, labels = get_training_data(face_cascade, 'partners') # Se carga el dataset de imagenes de los compañeros
face_recognizer.train(images, np.array(labels)) # Se pasan los datos al algoritmo
evaluate(face_recognizer, face_cascade, 'partners')







