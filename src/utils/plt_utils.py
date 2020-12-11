import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.pyplot import show

def show_image(image, title=None, cmap=cm.get_cmap("gray")):
    '''
    @brief  Methods to display an image with a given title and colourmap

    @param image    The image to be displayed
    @param title    The title of the images. No title if None
    @param cmap     The colourmap used to display the image
    '''
    plt.figure()
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)


def show_images(*images, titles = None, cmap=cm.get_cmap("gray"), sup_title = None):
    '''
    @brief  Methods to display multiple images at once. Subplot orientation 
            adaptively configured

    @param *images      The image to be displayed
    @param title        A list of titles corresponding to each subplots
    @param cmap         The colourmap used to display the images
    @param sup_title    The super title of the entire figure (not super title 
                        if None)
    '''
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
    '''
    @brief  Methods to display multiple histograms at once. Subplot 
            orientation adaptively configured

    @param *hists       The distributions to be displayed
    @param title        A list of titles corresponding to each subplot
    @param sup_title    The super title of the entire figure (not super title 
                        if None)
    '''
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

def plot_image_points(images, image_points, titles = None, cmap=cm.get_cmap("gray"), sup_title = None, same_colour=True):
    '''
    @brief  Methods to display multiple images at once, Each image 
            superimposed by a scatter plot of points. This is used to display 
            points of interest in the image.

    @param images       The image to be displayed
    @param image_points A list with the same length as the image list. Each 
                        item contains a list of points to be superimposed on 
                        the corresponding image
    @param title        A list of titles corresponding to each subplots
    @param cmap         The colourmap used to display the images
    @param sup_title    The super title of the entire figure (not super title 
                        if None)
    @param same_colour  If True, all points are plotted with the same colour. 
                        If False, the points are coloured by the rainbow 
                        spectrum based on their order, starting at purple
    '''
    rows = int(np.sqrt(len(images)))
    cols = int(np.ceil(len(images) / rows))

    if not titles:
        titles = [
            "image " + str(i)
            for i in range(len(images))
        ]

    print("Plotting", rows, "rows and", cols, "columns")
    
    f, axs = plt.subplots(rows, cols, figsize=(15,5))
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