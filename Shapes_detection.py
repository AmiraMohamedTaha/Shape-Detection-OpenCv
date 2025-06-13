import cv2
import numpy as np

def shape_determination(contour):

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    num_of_vertices = len(approx)
    area = cv2.contourArea(contour)
    
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return 'unknown'
    aspect_ratio = w / h
    
    if num_of_vertices == 3:
        return 'triangle'
    elif num_of_vertices == 4:
        if 0.9 <= aspect_ratio <= 1.1:
            return 'square'
        else:
            return 'rectangle'
    else:
        if perimeter == 0:
            return 'unknown'  
        circle = (4 * np.pi * area) / (perimeter**2)
        if circle >= 0.85:
            return 'circle'
        else:
            return 'unknown'

def detect_shapes(img):
   
    img = cv2.imread("myvenv/Shapes.jpg")
    
    
    img_to_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_to_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    square_contours = []
    circle_contours = []
    triangle_contours = []
    rectangle_contours = []
    
    
    
    min_area_filtered = 77  
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_filtered:
            continue
        
        shape = shape_determination(contour)
        if shape == 'square':
            square_contours.append(contour)
        elif shape == 'circle':
            circle_contours.append(contour)
        elif shape == 'triangle':
            triangle_contours.append(contour)
        elif shape == 'rectangle':
            rectangle_contours.append(contour)
    
    if square_contours:
        cv2.drawContours(img, square_contours, -1, (0 , 0, 255), 3)
        print(f"{len(square_contours)} Squares")
    if circle_contours:
        cv2.drawContours(img, circle_contours, -1, (255, 0 , 0), 3)
        print(f"{len(circle_contours)} Circles")
    if triangle_contours:
        cv2.drawContours(img, triangle_contours, -1, (0, 255 , 0), 3)
        print(f"{len(triangle_contours)} Triangles")
    if rectangle_contours:
        cv2.drawContours(img, rectangle_contours, -1, (0 , 255 , 255), 3)
        print(f"{len(rectangle_contours)} Rectangles")
  
    cv2.imshow('Shapes contoured', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_shapes("myvenv/Shapes.jpg")  