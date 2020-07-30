import cv2
import numpy as np


def clear_noise(img):
    return cv2.bilateralFilter(img, 9, 10, 75)


def improve_contrast(img):
    if len(img.shape) > 2:
        raise NameError('expect for gray-scale input')
    return cv2.equalizeHist(img)


def change_color_space(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def hair_removal(img):
    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (9, 9))

    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret, thresh2 = cv2.threshold(blackhat, 5, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(img, thresh2, 1, cv2.INPAINT_TELEA)
    return dst


def HZ_preprocess(img, hair=False):
    img = np.array(img)
    if hair:
        return clear_noise(improve_contrast(change_color_space(hair_removal(img))))
    else:
        return clear_noise(improve_contrast(change_color_space(img)))

