import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def show_images(*images, titles = None, cmap=cm.get_cmap("Greys")):
    rows = int(np.sqrt(len(images)))
    cols = int(np.ceil(len(images) / rows))

    if not titles:
        titles = [
            "image " + str(i)
            for i in range(len(images))
        ]

    f, axs = plt.subplots(rows, cols)
    for i in range(len(images)):
        axs[i].imshow(images[i], cmap=cmap)
        axs[i].set_title(titles[i])

    plt.show()