from paddleocr import PaddleOCR,draw_ocr
import sys
import math
import csv

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = '1483.png'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='arial_narrow_7.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')

print(len(result))
with open("1483.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file_name', 'chart_title', 'x_title_Bounding_Box', 'x_title', 'y_title_Bounding_Box', 'y_title', 'chart_type_Bounding_Box', 'chart_type', 'x_Bounding_Box', 'x', 'y_Bounding_Box', 'y'])

    row = []
    x_values = []
    y_values = []
    for i in range(len(result)):
        for lines in result:
            if (i==0):
                row.append(img_path)
                row.append(lines[1][0])
                break
            if(i==1):
                row.append(lines[0])
                break
            if(i==2):
                row.append(lines[1][0])
                break
            if(i==3):
                row.append(lines[0])
                break
            if(i==4):
                row.append(lines[1][0])
                break
            if(i==5):
                row.append(lines[0])
                break
    
    writer.writerow(row)
            
    

    
'''
#openCV
import cv2
import numpy as np

# read the input image
img = cv2.imread('2.png')

# convert the input image to grayscale image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Apply a threshold to the image to convert it to black and white
#_, black = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# find the edges using Canny edge detector
edges = cv2.Canny(gray,50,0,apertureSize = 3)
minLineLength = 10
maxLineGap = 5

# apply probabilistic Hough transform
lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
for line in lines:
   for x1,y1,x2,y2 in line:
      cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imshow('houghlines.jpg',img)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
