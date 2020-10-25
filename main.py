from src.Character_Detection.train_characters import  save_character_recognition_model
from src.Plate_Detection.get_plate import get_plate
from src.Character_Detection.image_processing import get_characters
import glob
from src.Character_Detection.predict_character import predict_character


# TODO: IMPORTANT uncomment this function when running for the first time than you should comment it
#save_character_recognition_model()

# TODO: Change image if you wish...
image_paths = glob.glob("Plate_examples/*.jpg")
test_image = image_paths[13]

try:
    print("[INFO] getting plate")
    plates, _ = get_plate(test_image)

    for plate in plates:
        print("[INFO] processing image")
        res = get_characters(plate)

        print("[INFO] printing prediction")
        for r in res:
            print(predict_character(r))
        print("________________________")
except FileNotFoundError as e:
    print("Please uncomment save_character_recognition_model() function in main.py")
except AssertionError as e:
    print("[ERROR] CAR Plate was not found")
