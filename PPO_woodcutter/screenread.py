import pygetwindow as gw
from PIL import ImageGrab
import cv2
import time
from skimage.metrics import structural_similarity as ssim

import numpy as np

score_value=0

def getss(program_name):
    try:
        # Get the window with the specified program name
        window = gw.getWindowsWithTitle(program_name)[0]

        # Get the coordinates of the window
        left, top, right, bottom = window.left, window.top, window.right, window.bottom

        # Capture the contents of the window using ImageGrab
        screenshot = ImageGrab.grab(bbox=(left, top+500, right-45, bottom-110))

        # Save or process the screenshot as needed
        screenshot.save("screenshot.png")
        img = cv2.imread("screenshot.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (5,5), 0)
        if(img[-1][-1] == 167):
            img = cv2.Canny(img, 200 , 300)
        elif(img [-1][-1]== 97):
            img = cv2.Canny(img, 200 , 300)
        else:
            img = cv2.Canny(img, 125 , 175)

        return img

    except IndexError:
        print(f"No window found with the title '{program_name}'.")
        return None

def getscoreimg(program_name):
    try:
        # Get the window with the specified program name
        window = gw.getWindowsWithTitle(program_name)[0]


        # Get the coordinates of the window
        left, top, right, bottom = window.left, window.top, window.right, window.bottom

        sc_left, sc_top, sc_right, sc_bottom = left+260, top + 170, right - 300, bottom - 812
        score = ImageGrab.grab(bbox=(sc_left, sc_top, sc_right, sc_bottom))

        score.save("score.png")
        img = cv2.imread("score.png", cv2.IMREAD_GRAYSCALE)
        
        return img

    except IndexError:
        print(f"No window found with the title '{program_name}'.")
        return None
    
def images_are_different(img1, img2, threshold=0.95):
    # return ImageChops.difference(img1, img2).getbbox() is not None
    similarity, _ = ssim(img1, img2, full=True)
    return similarity < threshold
