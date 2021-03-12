import os
import cv2


# Get list of all files in all subdirectories
def list_files(folder):
    file_list = []
    subdirs = [x[0] for x in os.walk(folder)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if len(files) > 0:
            for file in files:
                file_list.append(os.path.join(subdir, file))
    return file_list


# Get list of all image files from file list
def load_images_from_folder(folder):
    images = []
    files = list_files(folder)
    for image_path in files:
        img = cv2.imread(image_path)

        if img is not None:
            images.append(image_path)

    return images
