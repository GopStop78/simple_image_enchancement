import cv2
import os
import file_routines as frut
import processing_routines as proc


# Simple routine for batch image processing
def process_image_list(img_list, use_simple_filtering):
    for img in img_list:
        cv_img = cv2.imread(img)

        cv2.imshow('Source', cv_img)

        result = process_image_cycle(cv_img, use_simple_filtering)

        if result is not None:
            res_path = os.path.splitext(img)[0] + '_adjusted.png'
            cv2.imwrite(res_path, result)

        cv2.destroyAllWindows()


# Simple routine for runtime image processing and saving
def process_image_cycle(cv_image, use_simple_filtering):
    processing_window = 'Current result'

    # cv2.createTrackbar('gamma', processing_window, 0, slider_max, on_trackbar)
    while True:
        result = proc.adjust_image_params(cv_image, use_simple_filtering)

        cv2.imshow(processing_window, result)

        key = cv2.waitKey(5)

        if key == 27:
            return

        if key == ord('s'):
            return result


if __name__ == '__main__':
    simple_filtering = True

    image_path_atom = 'd:/Work/atl_test/for_test/atom'
    image_path_silar = 'd:/Work/atl_test/for_test/silar'

    # Load image pack
    image_list = frut.load_images_from_folder(image_path_silar)

    # Process image pack
    process_image_list(image_list, simple_filtering)
