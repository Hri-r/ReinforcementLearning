import cv2 as cv
from screenread import getss

img = getss("LDPlayer")

# img = cv.imread("screenshot.png", cv.IMREAD_GRAYSCALE)
print(img)
img = cv.GaussianBlur(img, (5,5), 0)
if(img[-1][-1] == 167):
    img = cv.Canny(img, 200 , 300)
elif(img [-1][-1]== 97):
    img = cv.Canny(img, 200 , 300)
else:
    img = cv.Canny(img, 125 , 175)
cv.imshow('img', img)

# cv.imshow('img', img)
cv.waitKey(0)