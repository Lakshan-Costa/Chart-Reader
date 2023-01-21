from paddleocr import PaddleOCR,draw_ocr
import sys
import math
import csv
import cv2
import numpy as np
from os import walk
from PIL import Image

def dataOCR(left_line, bottom_line):
    x1, y1, x2, y2 = left_line
    x3, y3, x4, y4 = bottom_line
    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
    # to switch the language model in order.
    ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
    img_path = 'C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/Images/Vertical_Bar_Chart/VerticalBar18.jpg'
    result = ocr.ocr(img_path, cls=True)
    # Get the boxes, texts, and scores for the lines on the left of the line
    top_txts = []
    top_boxes = []
    top_scores = []
    min_y = y1
    left_boxes = []
    left_txts = []
    left_scores = []
    # Get the boxes, texts, and scores for the lines on the bottom of the line
    bottom_boxes = []
    bottom_txts = []
    bottom_scores = []

    title_boxes = []
    title_txts = []
    title_scores = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            box = line[0]
            if box[0][0] < x1:
                left_boxes.append(box)
                left_txts.append(line[1][0])
                left_scores.append(line[1][1])
            elif box[1][1] > y4:
                bottom_boxes.append(box)
                bottom_txts.append(line[1][0])
                bottom_scores.append(line[1][1])
            else:
                title_boxes.append(box)
                title_txts.append(line[1][0])
                title_scores.append(line[1][1])
                
    # draw result for left line
    
    image = Image.open(img_path).convert('RGB')
    im_show = draw_ocr(image, left_boxes, left_txts, left_scores, font_path='arial_narrow_7.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('resultleft.jpg')
    # draw result for bottom line
    image = Image.open(img_path).convert('RGB')
    im_show = draw_ocr(image, bottom_boxes, bottom_txts, bottom_scores, font_path='arial_narrow_7.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('resultbottom.jpg')
    # draw result for top
    image = Image.open(img_path).convert('RGB')
    im_show = draw_ocr(image, title_boxes, title_txts, title_scores, font_path='arial_narrow_7.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('resulttop.jpg')





def axisLines(file_Name, i):
    img = cv2.imread(file_Name)  
    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  
    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
  
    # Initialize empty lists to store the left and bottom lines
    left_line = []
    bottom_line = []

    # Iterate over the detected lines
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        if abs(theta) < np.pi / 4:
            # Check if line is leftmost
            if (not left_line) or (x1 < left_line[0] and x2 < left_line[2]):
                left_line = (x1, y1, x2, y2)
        else:
            # Check if line is bottommost
            if (not bottom_line) or (y1 > bottom_line[1] and y2 > bottom_line[3]):
                bottom_line = (x1, y1, x2, y2)

    #calculating the slope of the lines
    slope1 = (bottom_line[3]-bottom_line[1])/(bottom_line[2]-bottom_line[0])
    slope2 = (left_line[3]-left_line[1])/(left_line[2]-left_line[0]) if (left_line[2]-left_line[0])!=0 else 0.00001
    
    #if slope of the lines is close to infinity or close to 0, that means the lines are almost vertical or horizontal
    if abs(slope1) < 0.1  or abs(slope1) > 10:
        #width = img.shape[1]
    
        #bottom_line = (0, bottom_line[1], width, bottom_line[3])
        cv2.line(img, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (0, 0, 255), 2)
    if abs(slope2) < 0.1 or abs(slope2) > 10:
        cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 2)
    cv2.imwrite("axis/"+i+"result.png", img)
    print(i)
    
    #dataOCR(left_line, bottom_line)
    
    #INCREASE ACCURACY OF THE AXIS DETECTION
    #Get the lines which are above the text and on the right so if there is a border there wouldn't be an issue
    #Get the lines if the line postion is after more than 50% of the image


   

def fileNames():
    folder = "C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/Images/Vertical_Bar_Chart"
    filenames = next(walk(folder), (None, None, []))[2]  # [] if no file
    for i in filenames:
        axisLines(f"{folder}/{i}", i)
fileNames()
#axisLines()
