import cv2
import matplotlib.pyplot as plt


def preprocess_image(image_path, resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


def get_characters(plate):
    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(plate, alpha=255.0)

    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Applied inversed thresh_binary
    binary = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ## Applied dilation
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

    plt.imshow(thre_mor)
    plt.show()

    characters = find_contours(plate_image, binary, thre_mor)

    return characters


def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def find_contours(plate_image, binary, thre_mor):
    cont, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()

    # Initialize a list which will be used to append charater image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60

    coord_last_cont = (0, 0, 0, 0)

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
            if h / plate_image.shape[0] >= 0.3:  # Select contour which has the height larger than 50% of the plate
                check_overlap = x >= coord_last_cont[0] and y >= coord_last_cont[1] and (x+w) <= coord_last_cont[2] and (y+h) <= coord_last_cont[3]

                # Separate number and gibe prediction
                if not check_overlap:
                    # Draw bounding box aground digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    curr_num = thre_mor[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)
                    coord_last_cont = (x, y, x+w, y+h)

    plt.imshow(test_roi)
    plt.show()

    # for plate_char in crop_characters:
    #     plt.imshow(plate_char)
    #     plt.show()

    print("Detect {} letters...".format(len(crop_characters)))

    return crop_characters

