import cv2

import numpy as np
import math


# Adjusts image gamma automatically using common method
def adjust_gamma_1(image):
    # convert img to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid * 255) / math.log(mean)

    # do gamma correction
    img_gamma = np.power(image, gamma).clip(0, 255).astype(np.uint8)
    return img_gamma


# Adjusts image gamma automatically using HSV space
def adjust_gamma_2(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(val)
    gamma = math.log(mid * 255) / math.log(mean)

    # do gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)

    # combine new value channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

    return img_gamma


# Adjusts image contrast using adaptive CLAHE algorithm
def adjust_contrast_adaptive(src):
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe = cv2.createCLAHE(clipLimit=40)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return result


# Adjusts image contrast using mean-std normalization and weighted scaling
def adjust_simple(gray, scaling_mean, shifting_result, scaling_result):
    # convert img to gray
    float_gray = np.float32(gray)

    mean = scaling_mean * np.mean(float_gray)
    std = np.std(float_gray)

    # do gamma correction
    float_gray -= mean
    float_gray *= (1.0 / std)

    max_val = np.abs(float_gray.max())

    float_gray *= scaling_result * (255.0 / max_val)
    float_gray += shifting_result

    return float_gray.clip(0, 255).astype(np.uint8)


# Performs morphological preprocessing for noise suppression and white contour closing
def preprocess_morphological(image):
    kernel_w_median = 3
    kernel_size_blur = (3, 3)
    kernel_size_morph = (7, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size_morph)

    # blurred1 = cv2.GaussianBlur(image, kernel_size, 0)
    # cv2.imshow('Gauss', blurred1)

    medianned = cv2.medianBlur(image, kernel_w_median)
    cv2.imshow('median', medianned)

    blurred = cv2.blur(medianned, kernel_size_blur)
    cv2.imshow('blur', blurred)

    # result = cv2.dilate(image, kernel, iterations=1)
    result = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    return result


# Tries to detect object mask using adaptive binarization
def detect_object_mask(preprocessed_image):
    result_binary = cv2.adaptiveThreshold(preprocessed_image, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -1.5)
    rows, cols = result_binary.shape
    image_mask = np.zeros([rows, cols], np.uint8)
    # image_mask[12:rows - 12, 22:cols - 22] = result_binary[12:rows - 12, 22:cols - 22]
    image_mask[12:rows - 12, 25:cols - 35] = result_binary[12:rows - 12, 25:cols - 35]

    preprocessed_second = preprocess_morphological(image_mask)
    result_binary = cv2.morphologyEx(preprocessed_second, cv2.MORPH_CLOSE, (13, 13))
    binary_mask = cv2.medianBlur(result_binary, 7)

    return binary_mask


# Enchances object contrast using weighted masking
def apply_object_mask(image, mask):
    float_result_binary = cv2.addWeighted(image, 0.85, mask, 0.25, 0)
    result_masked = float_result_binary.clip(0, 255).astype(np.uint8)
    # result_binary = cv2.bitwise_and(preprocessed_first, preprocessed_first, mask=binary_mask)

    return result_masked


# Adjusts image contrast using selected method
def adjust_image_params(cv_image, use_simple_filtering):
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    preprocessed_first = preprocess_morphological(gray)

    if use_simple_filtering:
        result = adjust_simple(preprocessed_first, 1.1, 0.0, 1.0)
    else:
        mask = detect_object_mask(preprocessed_first)
        result = apply_object_mask(preprocessed_first, mask)

    final = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return final
