import cv2
import glob
import logging as Log
import shutil
import os
import dlib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np

Log.basicConfig(level=Log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ROIPath = ('/Users/Filip/Desktop/Faks/BIOMETRIJA PROJEKT/UTKface_inthewild/Detected/')
files = glob.glob('./UTKface_inthewild/Training/*.jpg')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


Log.info("Started programme")

try:
    shutil.rmtree(ROIPath)
    os.mkdir(ROIPath)
except OSError as e:
    Log.info(e)

counter = 0
age = []
gender = []
landmarks = []
for file in files:
    filename = os.path.basename(file)
    Log.info("Processing image: " + filename)
    img = cv2.imread(file)
    parts = filename.split("_")
    newFileName = parts[0]+ "_" + parts[1]+"_"
    FeatureCoordinates = []
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    found_first_face = False
    for idx, (x, y, w, h) in enumerate(face):
        if not found_first_face:
            cropped_face = gray_img[y:y + h, x:x + w]
            cropped_face = cv2.resize(cropped_face, (100,100))
            x_abs, y_abs = 0, 0
            x_rel, y_rel = 0, 0

            landmarksfrompredictor = landmark_predictor(cropped_face, dlib.rectangle(x_abs + x_rel, y_abs + y_rel, x_abs + w + x_rel, y_abs + h + y_rel))
            if len(landmarksfrompredictor.parts()) == 68:
                for n in range(0, 68):
                    x = landmarksfrompredictor.part(n).x
                    y = landmarksfrompredictor.part(n).y
                    FeatureCoordinates.append([x,y])

                landmarks.append(FeatureCoordinates)
                age.append(parts[0])
                gender.append(parts[1])
                if cv2.imwrite(ROIPath+newFileName+str(counter)+".jpg", cropped_face):
                    Log.info("Succesfully wrote file: " + newFileName + str(counter) + ".jpg")
                    found_first_face=True
                else:
                    Log.info("Failed to write file: " + newFileName + str(counter) + ".jpg")
                    found_first_face=True  
            else:
                Log.warning("Landmarks not found not writing down file")
                found_first_face=True

    counter+=1

Log.warning("age length: " + str(len(age)))
Log.warning("gender length: " + str(len(gender)))
Log.warning("landmarks length: " + str(len(landmarks)))
data = {
    'age': age,
    'gender': gender,
    'landmarks' : landmarks 
}
npage = np.array(age)
npgender = np.array(gender)
nplandmarks = np.array(landmarks)
y = []
X = []

trainingImages = glob.glob('./UTKface_inthewild/Detected/*.jpg')

for imagePath in trainingImages:
    img = cv2.imread(imagePath)
    img = img / 255.0
    X.append(img)

X = np.array(X)
n_samples = X.shape[0]
feature_vector_size = X.shape[1] * X.shape[2] * X.shape[3]
X_reshaped = X.reshape(n_samples, feature_vector_size)
y = np.column_stack((npage.flatten(),npgender.flatten()))
y = np.column_stack((y,nplandmarks.reshape(nplandmarks.shape[0],-1)))
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.1)
base_estimator = RandomForestClassifier(n_estimators=50, n_jobs=-1)
multi_output_clf = MultiOutputClassifier(base_estimator)
Log.warning("Training started")
sample_weights = [0.5,0.5,0]
multi_output_clf.fit(X_train, y_train, sample_weight=sample_weights.append(np.zeros(17)))
Log.warning("Prediction started")
predictions = multi_output_clf.predict(X_test)
Log.info(y_test[0])
Log.info(predictions[0])
list_ages_guessed = []
list_gender_guessed = []
list_ages = []
list_gender = []
accuracies_age = 0
accuracies_gender = []
for i in range(len(y_test)):
    if np.abs(int(y_test[i, 0]) - int(predictions[i, 0])) <6:
        accuracies_age +=1
        list_ages_guessed.append(predictions[i,0])
        list_ages.append(y_test[i,0])
    else:
        accuracies_age +=0
        list_ages_guessed.append(predictions[i,0])
        list_ages.append(y_test[i,0])
acc_age = accuracy_score(y_test[:, 0], predictions[:, 0])
acc_gender = accuracy_score(y_test[:, 1], predictions[:, 1])
for i in range(len(y_test)):
    list_gender_guessed.append(predictions[i,1])
    list_gender.append(y_test[i,1])
sum_age = 0
age_counter = 0
for i in range(len(y_test)):
    abs_diff = np.abs(int(y_test[i, 0]) - int(predictions[i, 0]))
    sum_age += abs_diff
    age_counter +=1

accuracies_gender.append(acc_gender)

overall_accuracy_age = accuracies_age / age_counter
overall_accuracy_gender = sum(accuracies_gender) / len(accuracies_gender) 
overall_missed_age = sum_age/ age_counter
Log.info("Algorithm age accuracy: " + str(overall_accuracy_age))
Log.info("Algorithm average age discrepancy: " + str(overall_missed_age))
Log.info("Algorithm gender accuracy: " + str(overall_accuracy_gender))
dataGuessing = {
    'AgesGuessed': list_ages_guessed,
    'AgesActual': list_ages,
    'GendersGuessed': list_gender_guessed,
    'GendersActual': list_gender
}
df = pd.DataFrame(dataGuessing)
df.to_csv('GuessingData.csv', index=False)
Log.info("Guessing data dumped into CSV...")