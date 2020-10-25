import numpy as np
import cv2
from src.Plate_Detection.plate_detection_load_model import wpod_net
from src.Character_Detection.image_processing import preprocess_image
import matplotlib.pyplot as plt
from src.Plate_Detection.local_utils import detect_lp
from os.path import splitext, basename


def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _, plate, _, cors = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    print("Detect %i plate(s) in" % len(plate), splitext(basename(image_path))[0])
    # print("Coordinate of plate(s) in image: \n", cor)

    # Visualize our result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.axis(False)
    plt.imshow(preprocess_image(image_path))
    plt.subplot(1, 2, 2)
    plt.axis(False)
    plt.imshow(plate[0])
    plt.show()

    plates_detected_image = preprocess_image(image_path)

    for cor in cors:
        plates_detected_image = draw_box(plates_detected_image, cor)

    plt.figure(figsize=(8, 8))
    plt.axis(False)
    plt.imshow(plates_detected_image)
    plt.show()

    return plate, cors


# test_image = image_paths[11]
# LpImg, cor = get_plate(test_image)


def draw_box(image, cor, thickness=3):
    pts = []
    x_coordinates = cor[0]
    y_coordinates = cor[1]
    # store the top-left, top-right, bottom-left, bottom-right
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]), int(y_coordinates[i])])

    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    vehicle_image = image

    cv2.polylines(vehicle_image, [pts], True, (0, 255, 0), thickness)
    return vehicle_image
