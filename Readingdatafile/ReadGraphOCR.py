from paddleocr import PaddleOCR,draw_ocr
import sys
import math
import csv
import cv2
import numpy as np
from os import walk
from PIL import Image
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

global yvalue_list
yvalue_list = []

def VerticalBarChart():
    print("Vertical Bar chart")

def HorizontalBarChart():
    print("Horizontal Bar chart")


def createTable(line_start_point, yvalue_list):

    bottomy = sorted(bottom_boxes, key=lambda x: x[1][1])
    bottomx = sorted(bottom_boxes, key=lambda x: x[1][0])

    secondbottomx = bottomx[-2]

    #print(str(bottomx))

    maxbottomy = bottomy[-1]
    maxbottomx = bottomx[-1] #Remove this and add 2nd to last

    bottomy.remove(maxbottomx)
    bottomy.remove(maxbottomy)

    #print(bottomy)
    bottomy = sorted(bottomy, key=lambda x: x[1][0])
    bottomy.append(maxbottomx)
    bottomy.append(maxbottomy) # THIS IS THE TITLE OF THE X AXIS

    sorted_bottom_txts = [bottom_txts[bottomy.index(box)] for box in bottom_boxes]



    #Try to get using the same method used in the x axis 
    for i, element in enumerate(left_txts):
        try:
            left_txts[i] = float(element)
        except ValueError:
            pass

    for item in left_txts:
        if isinstance(item, str):
            ytitle = item
            break

    titley = sorted(title_boxes, key=lambda x: x[1][1])
    sorted_title_txts = [title_txts[titley.index(box)] for box in title_boxes]

    
    table = pd.DataFrame(
        {"x axis": sorted_bottom_txts[:-1],
         "y axis": yvalue_list
        
        })

    table2Titles = ["", "Chart Name", "Chart Type", "x Axis Name", "y Axis Name", "Legend"]
    table2Descriptions = ["", sorted_title_txts[0], chartType, sorted_bottom_txts[-1], ytitle, sorted_title_txts[1]]

    table2 = pd.DataFrame({
        "x axis": table2Titles,
        "y axis": table2Descriptions
        })

    table = table.append(table2, ignore_index = True)
    
    print(table)
    

    #print(bottomy)

    #print(bottomx)
    

    #print(bottomx)
    # Output: [[273.0, 497.0], [337.0, 497.0], [337.0, 511.0], [273.0, 511.0]]
    print()


def calculatingYValue(line_start_point, xintersection, yintersection, left_boxes, left_txts):
    sorted_lines = sorted(left_boxes, key=lambda x: (x[0][1] + x[2][1])/2)
    maxleftbox = ((sorted_lines[0][1][1] +sorted_lines[0][2][1])/2)
    minleftbox = (sorted_lines[-1][1][1] + sorted_lines[-1][2][1])/2

    maxlefttext = float(left_txts[0])
    minlefttext = float(left_txts[-1])

    #print(minlefttext, maxlefttext) #Minimum text value and Maximum text value

    #print(yintersection, maxleftbox) #The y axis intersection and maximum value of the highest text box
    #print(line_start_point[1]) #The bar chart y value

    yvalue = ((line_start_point[1] - yintersection) * (maxlefttext - minlefttext))/(maxleftbox - yintersection) # This is the Y value test it
    #print(yvalue, 1)


    yvalue_list.append(yvalue)

    if (len(yvalue_list) == len(bottom_boxes) - 1):
        createTable(line_start_point, yvalue_list)

def dataOCR(left_line, bottom_line, top_border_line):
    img_path = 'C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/Images/Vertical_Bar_chart/VBC223.jpg'

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

    global left_boxes
    global left_txts
    left_boxes = []
    left_txts = []
    left_scores = []
    # Get the boxes, texts, and scores for the lines on the bottom of the line

    global bottom_boxes
    global bottom_txts
    bottom_boxes = []
    bottom_txts = []
    bottom_scores = []

    global title_boxes
    global title_txts
    title_boxes = []
    title_txts = []
    title_scores = []
    for idx in range(len(result)):
        global res
        res = result[idx]
        for line in res:
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

    dataOCR(left_line, bottom_line, top_border_line)
    
    #draw rectangles
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] #RETR_TREE, RETR_CCOMP, RETR_LIST  https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    count = 0
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:2] #Gives the largest contour
    marked_y = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect = cv2.boxPoints(cv2.minAreaRect(c))
        
        cv2.drawContours(img, [c], 0, (0,255,255), 1)
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -1)
        print([(rect[i][0], rect[i][1]) for i in range(4)])

        # Detect the corners using Harris corner detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
    
        # initialize a set to store the marked positions
        marked_positions = set()

        yintersection = bottom_line[1]
        xintersection = left_line[0]
                            
        # loop through the dst array
        for i in range(dst.shape[0]):
            for j in range(dst.shape[1]):
                # check if the current position is a Harris corner
                if dst[i, j] > 0.01 * dst.max():
                    # check if the current position is inside the bounding rectangle
                    if x <= j <= x + w and y <= i <= y + h:
                        # check if the current position has not been marked before
                        
                            
                        inside_box = False
                        for l in range(len(res)):   #What's the difference?? for box in res
                                #print(j, i)
                            
                            x_min, y_min = res[l][0][0][0], res[l][0][0][1]
                            x_max, y_max = res[l][0][2][0], res[l][0][2][1]
                            if x_min <= j <= x_max and y_min <= i <= y_max:
                                inside_box = True
                                break

                        if not inside_box:
                                
                            line_start_point = (j, i)
                            #cv2.line(img, line_start_point, (j-2000, i), (255, 0, 255), 1) #Draw the horizontal pink line
                            if i != marked_y:  # only print if y position is different
                                marked_y = i

                                if (j, i) not in marked_positions:
                            
                                    # mark the position with a red dot
                                    img[i, j] = [0, 0, 255]
                                    
                                    # add the position to the set of marked positions
                                    #marking the positions so that the red dot is only one pixel
                                    marked_positions.add((j+1, i+1))
                                    marked_positions.add((j, i+2))
                                    marked_positions.add((j+3, i+2))
                                    marked_positions.add((j+1, i+3))
                                    marked_positions.add((j, i+3))
                                    marked_positions.add((j, i+1))
                     
                                    #Draw the horizontal pink line
                                    cv2.line(img, (j, i), (j-2000, i), (255, 0, 255), 1) 

                            # Check if the starting point of the line is not inside the bounding boxes

                                    

                                    calculatingYValue(line_start_point, xintersection, yintersection, left_boxes, left_txts)

                            
                            
        res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
        cv2.imwrite("result.png", res_final)       
        sorted_lst = sorted(left_boxes, key=lambda x: min([y[1] for y in x]), reverse=True)

        highest_value1 = sorted_lst[-1][0][1]
        highest_value2 = sorted_lst[-1][2][1]
        
        lowest_value1 = sorted_lst[0][0][1]
        lowest_value2 = sorted_lst[0][2][1]

        highest_value_pixel = (highest_value1+highest_value2)/2
        lowest_value_pixel = (lowest_value1+lowest_value2)/2
        
        #print(highest_value_pixel)
        #print(lowest_value_pixel)             
            
            
    #res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    #cv2.imwrite("result.png", res_final)
    #cv2.imwrite("cropped_image_without_contours.png", cropped_image_without_contours)


    #cv2.imshow("boxes", mask)
    #cv2.imshow("final image", res_final)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    
    #INCREASE ACCURACY OF THE AXIS DETECTION
    #Get the lines which are above the text and on the right so if there is a border there wouldn't be an issue
    #Get the lines if the line postion is after more than 50% of the image
   

def fileNames():
    folder = "C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/Images/Vertical_Bar_chart"
    filenames = next(walk(folder), (None, None, []))[2]  # [] if no file
    for i in filenames:
        axisLines(f"{folder}/{i}", i)

def checkChartType(fileName):
    model = keras.models.load_model("C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/Chart_Classification_79.h5")

    train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.1)
    train_generator = train_datagen.flow_from_directory(
        'C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/Images',
        target_size = (300, 300),
        batch_size = 256,
        class_mode = 'categorical',
        subset = 'training'
    )

    # Load and preprocess the image
    img = Image.open("C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/test/1381.png")
    img = img.resize((300, 300))
    x = np.array(img)
    x = x[:, :, :3] # Remove the 4th channel
    x = x / 255. # Scale pixel values to [0, 1]
    x = np.expand_dims(x, axis=0) # Add an extra dimension

    # Pass the image through the model
    preds = model.predict(x)

    # Get the class with the highest predicted probability
    class_idx = np.argmax(preds[0])

    # Get the class labels from the generator
    class_labels = train_generator.class_indices
    class_labels = {v: k for k, v in class_labels.items()}

    # Display the predicted class
    global chartType
    chartType = f"{class_labels[class_idx]}"

    axisLines(fileName, 1)
    
    
#fileNames()
#axisLines("C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/Images/Vertical_Bar_chart/VBC25.jpg", 1)
checkChartType("C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/Images/Vertical_Bar_chart/VBC223.jpg")

#axisLines("C:/Users/Lakshan/OneDrive/Documents/GitHub/Chart-Reader/New folder(2)/result.png", 1)
