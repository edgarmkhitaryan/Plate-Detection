import glob
import numpy as np
from os.path import splitext, basename, sep
from keras_preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.ensemble import RandomForestClassifier
import pickle


# TODO: RUN WHEN NEED TO TRAIN NEW MODEL FOR CHARACTER RECOGNITION
def save_character_recognition_model():
    dataset_paths = glob.glob("Resources/dataset_characters/**/*.jpg")

    # Arrange input data and corresponding labels
    X = []
    labels = []

    for image_path in dataset_paths:
        label = image_path.split(sep)[-2]
        image = load_img(image_path, target_size=(80, 80))
        image = img_to_array(image)

        X.append(image)
        labels.append(label)

    X = np.array(X, dtype="float16")
    X = X.reshape(X.shape[0], 19200)
    y = np.array(labels)

    (train_X, test_X, train_Y, test_Y) = train_test_split(X, y, test_size=0.05, stratify=y, random_state=42)

    rand_forest = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42)
    rand_forest.fit(train_X, train_Y)

    with open("Resources/character_recognition_model.pkl", 'wb') as file:
        pickle.dump(rand_forest, file)

    print("Accuracy on training set : {:.3f}".format(rand_forest.score(train_X, train_Y)))
    print("Accuracy on test set : {:.3f}".format(rand_forest.score(test_X, test_Y)))

    print("[INFO] Find {:d} images with {:d} classes".format(len(X), len(set(labels))))
