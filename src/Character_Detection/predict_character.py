import cv2
import numpy as np
import pickle


def predict_character(img):
    img = cv2.resize(img, (80, 80))
    img = np.stack((img,) * 3, axis=-1)

    with open("Resources/character_recognition_model.pkl", 'rb') as file:
        model = pickle.load(file)

    result = model.predict(img.reshape(1, 19200))
    return result
