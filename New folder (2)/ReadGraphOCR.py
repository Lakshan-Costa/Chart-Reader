from paddleocr import PaddleOCR,draw_ocr
import sys
import math
import csv
import cv2
import numpy as np
from os import walk
from PIL import Image

def VerticalBarChart():
    print("Vertical Bar chart")

def HorizontalBarChart():
    print("Horizontal Bar chart")

def dataOCR(left_line, bottom_line, top_border_line):
    img_path = 'C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/Images/Vertical_Bar_chart/VBC68.jpg'

    x1, y1, x2, y2 = left_line
    x3, y3, x4, y4 = bottom_line
    x5, y5, x6, y6 = top_border_line
    
    
    
    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
    # to switch the language model in order.
    ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
    result = ocr.ocr(img_path, cls=True)
    # Get the boxes, texts, and scores for the lines on the left of the line
    all_boxes = []
    all_txts = []
    all_scores = []
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
            print(line)
            box = line[0]
            all_boxes.append(box)
            all_txts.append(line[1][0])
            all_scores.append(line[1][1])
            
            if box[0][0] < x1:
                left_boxes.append(box)
                left_txts.append(line[1][0])
                left_scores.append(line[1][1])
            elif box[1][1] > y4:
                bottom_boxes.append(box)
                bottom_txts.append(line[1][0])
                bottom_scores.append(line[1][1])
            elif box[1][1] > y5:
                title_boxes.append(box)
                title_txts.append(line[1][0])
                title_scores.append(line[1][1])
            
                
    # draw result for left line
    image = Image.open(img_path).convert('RGB')
    im_show = draw_ocr(image, all_boxes, all_txts, all_scores, font_path='arial_narrow_7.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('resultOCR.jpg')
    
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

'''
def drawLine():
    img = cv2.imread("result.png") 
    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  
    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
  
    # Initialize empty lists to store the left and bottom lines
    left_line = []
    bottom_line = []

    offset = 10

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
        cv2.line(img, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (0, 0, 255), 2)
    if abs(slope2) < 0.1 or abs(slope2) > 10:
        cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 2)
    cv2.imwrite("result2.png", img)
'''

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
    width, height = Image.open(file_Name).size
    top_border_line = 0, 0, width, 0
    
    offset = 10
    mid = height/2
    # Iterate over the detected lines
    for line in lines:
        rho, theta = line[0]            
        a = np.cos(theta)
        b = np.sin(theta)
        aa = np.cos(90 - theta)
        bb = np.sin(90 -theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        y3 = int(y0 + 2000 * (a))
        y4 = int(y0 + 2000 * (a))
            
        if abs(theta) < np.pi / 4:
                # Check if line is leftmost
            if (not left_line) or (x1 < left_line[0] and x2 < left_line[2]):
                left_line = (x1, y1, x2, y2)
        elif (y2>mid):
               # Check if line is bottommost
            if (not bottom_line):
                #print(theta, rho)
                bottom_line = (x1, y1, x2, y2)
                #print(str(mid) +" "+ str(y1)+ file_Name)
        else:
            if(not bottom_line):            #These are the ones that doesn't have a bottom line
                bottom_line = (x1, y3, x2, y4)
                #print(theta,rho)
                #print(str(mid) +" "+ str(y1)+ file_Name)
        
    
   #calculating the slope of the lines
    slope1 = (bottom_line[3]-bottom_line[1])/(bottom_line[2]-bottom_line[0])
    slope2 = (left_line[3]-left_line[1])/(left_line[2]-left_line[0]) if (left_line[2]-left_line[0])!=0 else 0.00001
    
    #if slope of the lines is close to infinity or close to 0, that means the lines are almost vertical or horizontal
    if abs(slope1) < 0.1  or abs(slope1) > 10:
        cv2.line(img, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (0, 0, 255), 2)
    if abs(slope2) < 0.1 or abs(slope2) > 10:
        cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 2)
    cv2.imwrite(r"result.png", img) #can remove the extention
    print(i)

    #Create a dataset using the bottomline
    #bottom_line_file = open("position.txt", 'a')
    #bottom_line_file.writelines(file_Name+","+ str(bottom_line)+"\n")
    #bottom_line_file.close()
    
    #draw rectangles
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] #RETR_TREE, RETR_CCOMP, RETR_LIST  https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    count = 0
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:2] #Gives the largest contour
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect = cv2.boxPoints(cv2.minAreaRect(c))
        
        cv2.drawContours(img, [c], 0, (0,255,255), 2)
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -1)
        print([(rect[i][0], rect[i][1]) for i in range(4)])

        # Detect the corners using Harris corner detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
    
        # Check if the Harris corner points are inside the bounding rectangle
        for i in range(dst.shape[0]):
            for j in range(dst.shape[1]):
                if dst[i, j] > 0.01 * dst.max():
                    if x <= j <= x + w and y <= i <= y + h:
                        img[i, j] = [0, 0, 255]
            
        res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
        cv2.imwrite("result.png", res_final)
        #cv2.imwrite("cropped_image_without_contours.png", cropped_image_without_contours)
        #if(rect[2][1] < bottom_line[3] and rect[3][1] < bottom_line[1] and w*h> 100):
                #print([(rect[i][0], rect[i][1]) for i in range(4)])
                #cv2.line(img, (0, int(rect[0][1])), (int(rect[0][0])+1000, int(rect[0][1])), (255, 0, 255), 1)


        #cv2.imshow("boxes", mask)
        #cv2.imshow("final image", res_final)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()



        

        '''
        x, y, w, h = cv2.boundingRect(c)
        rect = cv2.boxPoints(cv2.minAreaRect(c))
        
        #print(bottom_line)
        
        print([(rect[i][0], rect[i][1]) for i in range(4)])

        #if(rect[2][1] < bottom_line[3] and rect[3][1] < bottom_line[1] and w*h> 100):
                #print([(rect[i][0], rect[i][1]) for i in range(4)])
                #cv2.line(img, (0, int(rect[0][1])), (int(rect[0][0])+1000, int(rect[0][1])), (255, 0, 255), 1)

        #print(rect[0][0])
        #print(left_line[0])
        
            #for i in range(2):
                #for j in range (2):
                    #if(rect[i][j] == bottom_line[1] or bottom_line[3]): #if there is an equality with the bottom_line
                        #print("Yes")                                                                        # That means it's one of the bars
                
        #x, y = rect[i][0], rect[i][1]
        #print("X position: ", x, "Y position: ", y)
        if cv2.contourArea(c) > 1000:
            
            
        #if w*h>1000: #Check if this displays the ledgend for all images
            cv2.drawContours(img,[c], 0, (0,255,0), 1)
            cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -1)
            
                
            '''
                #cv2.line(img, (0, 100), (500, 100), (255, 0, 255), 5)
            
                #print("X position: ", x, "Y position: ", y)
                

                    
            
            
    #res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    #cv2.imwrite("result.png", res_final)
    #cv2.imwrite("cropped_image_without_contours.png", cropped_image_without_contours)


    #cv2.imshow("boxes", mask)
    #cv2.imshow("final image", res_final)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    '''
    cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        if c[0][0][1] < bottom_line[3] and c[0][0][0] > left_line[0]: 
            cv2.drawContours(img,[c], 0, (0,200,0), 1)
            print("x: ", c[0][0][0], "y: ", c[0][0][1])

    cv2.imshow("result", img)
    cv2.waitKey(0)
    '''
    dataOCR(left_line, bottom_line, top_border_line)
    
    #INCREASE ACCURACY OF THE AXIS DETECTION
    #Get the lines which are above the text and on the right so if there is a border there wouldn't be an issue
    #Get the lines if the line postion is after more than 50% of the image
   

def fileNames():
    folder = "C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/Images/Vertical_Bar_chart"
    filenames = next(walk(folder), (None, None, []))[2]  # [] if no file
    for i in filenames:
        axisLines(f"{folder}/{i}", i)
#fileNames()
axisLines("C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/Images/Vertical_Bar_chart/VBC68.jpg", 1)
#axisLines("C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/New folder(2)/result.png", 1)
