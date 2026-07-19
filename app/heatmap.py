import matplotlib.pyplot as plt
import numpy as np


def generate_heatmap(image1, image2):

    # Resize both images to the same size
    image1 = image1.resize((224, 224))
    image2 = image2.resize((224, 224))

    img1 = np.array(image1).astype(np.float32)
    img2 = np.array(image2).astype(np.float32)

    # Absolute pixel difference
    difference = np.abs(img1 - img2)

    # Convert to grayscale
    difference = np.mean(difference, axis=2)

    plt.figure(figsize=(5, 5))
    plt.imshow(difference, cmap="hot")
    plt.title("Change Heatmap")
    plt.axis("off")

    return plt