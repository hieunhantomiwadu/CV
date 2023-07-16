import cv2
import numpy as np
from PIL import Image
import os

#Directory path name where the face images are stored.
path = 'images'
recognizer = cv2.face.LBPHFaceRecognizer_create()
#Haar cascade file
detector = cv2.CascadeClassifier("cascade.xml");

def getImagesAndLabels(path):
    faces = []
    labels = {}
    ids = []
    num_name = 0
    for fold in os.listdir(path):
        labels[num_name] = fold
        for file in os.listdir(os.path.join(path, fold)):
            link = os.path.join(path, fold, file)
            img = cv2.imread(link, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            ids.append(num_name)
        num_name

    return ids, faces, labels

print ("\n[INFO] Training faces...")
ids,faces, labels = getImagesAndLabels(path)
ids = np.array(ids, dtype=np.int32)
#labels = cv2.UMat(labels)
recognizer.train(faces, ids)
# Save the model into the current directory.
recognizer.write('trainer.yml')
print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

def trained_names():
    global labels
    return labels