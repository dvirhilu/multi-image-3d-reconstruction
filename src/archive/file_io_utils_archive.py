def load_calib_images(calib_name):
    filenames = glob.glob("./images/calibration/" + calib_name + "/*")

    return [
        cv2.imread(file)
        for file in filenames
    ]
