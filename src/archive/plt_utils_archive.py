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