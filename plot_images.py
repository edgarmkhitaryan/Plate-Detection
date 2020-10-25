import matplotlib.pyplot as plt
from os.path import splitext, basename
import glob
from src.Character_Detection.image_processing import preprocess_image

# Create a list of image paths
image_paths = glob.glob("Plate_examples/*.jpg")
print("Found %i images..."%(len(image_paths)))

# Visualize data in subplot
fig = plt.figure(figsize=(12, 8))
cols = 5
rows = 4
fig_list = []
for i in range(cols*rows):
    fig_list.append(fig.add_subplot(rows, cols, i+1))
    title = splitext(basename(image_paths[i]))[0]
    fig_list[-1].set_title(title)
    img = preprocess_image(image_paths[i], True)
    plt.axis(False)
    plt.imshow(img)

plt.tight_layout(True)
plt.show()