import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def show_image(image, title=None, cmap=cm.get_cmap("gray")):
    plt.figure()
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)


def show_images(*images, titles = None, cmap=cm.get_cmap("gray"), sup_title = None):
    rows = int(np.sqrt(len(images)))
    cols = int(np.ceil(len(images) / rows))

    if not titles:
        titles = [
            "image " + str(i)
            for i in range(len(images))
        ]

    print("Plotting", rows, "rows and", cols, "columns")
    
    f, axs = plt.subplots(rows, cols)
    if sup_title:
        f.suptitle(sup_title)
    
    axs = axs.flatten()
    for i in range(len(images)):
        axs[i].imshow(images[i], cmap=cmap)
        axs[i].set_title(titles[i])

def plt_histograms(*hists, titles=None, sup_title = None):
    rows = int(np.sqrt(len(hists)))
    cols = int(np.ceil(len(hists) / rows))

    if not titles:
        titles = [
            "hist " + str(i)
            for i in range(len(hists))
        ]

    print("Plotting", rows, "rows and", cols, "columns")
    
    f, axs = plt.subplots(rows, cols)
    if sup_title:
        f.suptitle(sup_title)
    
    axs = axs.flatten()
    for i in range(len(hists)):
        bins_num = int(max(hists[i]) - min(hists[i]))
        axs[i].hist(hists[i], bins=bins_num, histtype='step')
        axs[i].set_title(titles[i])
        
def plot_corner_points(images, corners, titles = None, cmap=cm.get_cmap("gray"), sup_title = None):
    rows = int(np.sqrt(len(images)))
    cols = int(np.ceil(len(images) / rows))

    if not titles:
        titles = [
            "image " + str(i)
            for i in range(len(images))
        ]

    print("Plotting", rows, "rows and", cols, "columns")
    
    f, axs = plt.subplots(rows, cols)
    if sup_title:
        f.suptitle(sup_title)
    
    axs = axs.flatten()
    for i in range(len(images)):
        axs[i].imshow(images[i], cmap=cmap)
        if len(corners[i]) == 0:
            continue
        x = np.where(corners[i])[1]
        y = np.where(corners[i])[0]
        axs[i].scatter(x, y)
        axs[i].set_title(titles[i])

def plot_image_points(images, image_points, titles = None, cmap=cm.get_cmap("gray"), sup_title = None, same_colour=True):
    rows = int(np.sqrt(len(images)))
    cols = int(np.ceil(len(images) / rows))

    if not titles:
        titles = [
            "image " + str(i)
            for i in range(len(images))
        ]

    print("Plotting", rows, "rows and", cols, "columns")
    
    f, axs = plt.subplots(rows, cols)
    if sup_title:
        f.suptitle(sup_title)
    
    axs = axs.flatten()
    for i in range(len(images)):
        axs[i].imshow(images[i], cmap=cmap)

        if same_colour:
            x = [
                point[0] 
                for point in image_points[i]
            ]
            y = [
                point[1] 
                for point in image_points[i]
            ]

            axs[i].scatter(x, y)
        else:
            colours = cm.rainbow(np.linspace(0, 1, len(image_points[i])))
            for (point, colour) in zip(image_points[i], colours):
                if point is not None:
                    axs[i].scatter(point[0], point[1], color=colour)

        axs[i].set_title(titles[i])

def plot_point_path(images, point_masks, points, titles = None, cmap=cm.get_cmap("gray"), sup_title = None):
    rows = int(np.sqrt(len(images)))
    cols = int(np.ceil(len(images) / rows))

    if not titles:
        titles = [
            "image " + str(i)
            for i in range(len(images))
        ]

    print("Plotting", rows, "rows and", cols, "columns")
    
    f, axs = plt.subplots(rows, cols)
    if sup_title:
        f.suptitle(sup_title)
    
    axs = axs.flatten()
    for i in range(len(images)):
        axs[i].imshow(images[i], cmap=cmap)
        x = np.where(point_masks[i])[1]
        y = np.where(point_masks[i])[0]
        axs[i].scatter(x, y)
        x = [point[0] for point in points[i]]
        y = [point[1] for point in points[i]]

        segments = len(x)-1
        colors_arr = np.linspace(0,1,segments)**2
        for j in range(segments):
            axs[i].plot(x[j:j+2], y[j:j+2], color=(colors_arr[j]/2, 0.5, colors_arr[j]))
            
        axs[i].set_title(titles[i])