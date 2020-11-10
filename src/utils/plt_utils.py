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

    print("Plotting", rows, "rows and", cols, "columns")
    
    f, axs = plt.subplots(rows, cols)
    axs = axs.flatten()
    for i in range(len(images)):
        axs[i].imshow(images[i], cmap=cmap)
        axs[i].set_title(titles[i])

def plt_histograms(*hists, titles=None):
    rows = int(np.sqrt(len(hists)))
    cols = int(np.ceil(len(hists) / rows))

    if not titles:
        titles = [
            "image " + str(i)
            for i in range(len(hists))
        ]

    print("Plotting", rows, "rows and", cols, "columns")
    
    f, axs = plt.subplots(rows, cols)
    axs = axs.flatten()
    for i in range(len(hists)):
        bins_num = int(np.log2(len(hists[i])) + 1)
        axs[i].hist(hists[i], bins=bins_num, histtype='step')
        axs[i].set_title(titles[i])