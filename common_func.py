import cv2
import numpy as np
import math

def retrieve_image(image, lower_color_boundary, upper_color_boundary):
    
    frame = cv2.imread(image)

    # Convert BGR to HSV color scheme
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of wound red color in HSV

    # Thresulted_imagehold the HSV image to get only red colors that match with wound colors
    mask = cv2.inRange(hsv, lower_color_boundary, upper_color_boundary)

    # Bitwise-AND mask and original image
    resulted_image = cv2.bitwise_and(frame,frame, mask= mask)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Scan the wound in the frame: ',(0,50), font, 1, (99,74,154), 3, cv2.LINE_AA)

    # Calculating percentage area
    try: 
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_TC89_KCOS)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)

        areacnt = cv2.contourArea(cnt)
        arearatio=((areacnt)/208154)*100
        
        boxes = []
        for c in cnt:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x,y, x+w,y+h])

        boxes = np.asarray(boxes)
        # need an extra "min/max" for contours outside the frame
        left = np.min(boxes[:,0])
        top = np.min(boxes[:,1])
        right = np.max(boxes[:,2])
        bottom = np.max(boxes[:,3])
        
        cv2.rectangle(frame, (left,top), (right,bottom), (255, 0, 0), 2)
    
    except:
        pass

    cv2.imwrite('frame.jpg', frame)
    cv2.imwrite('mask.jpg', mask)
    cv2.imwrite('resulted_image.jpg', resulted_image)

    print("The area of the wound is: ", arearatio * 0.6615, "cm squared.")
    print("The area of the Custom-Aid is: ", (right - left)*(bottom - top)*2.989*pow(10, -4), "cm squared.")
    print("The length is equal to: ", (right - left) / 95.23, "cm.")
    print("The width is equal to: ", (bottom - top) / 95.23, "cm.")

def eliminate_inner_contour(image):

    # Read in the image as grayscale - Note the 0 flag
    im = cv2.imread(image, 0)

    # Run findContours - Note the RETR_EXTERNAL flag
    # Also, we want to find the best contour possible with CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create an output of all zeroes that has the same shape as the input
    # image
    out = np.zeros_like(im)

    # On this output, draw all of the contours that we have detected
    # in white, and set the thickness to be 3 pixels
    cv2.drawContours(out, contours, -1, 255, 1)

    # Spawn new windows that shows us the donut
    # (in grayscale) and the detected contour
    cv2.imwrite(f'eliminate_inner_contour_{image}', out)

def draw_outer_contour(image):
    input = cv2.imread(image)
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

    #find all contours
    img = cv2.pyrDown(gray)
    _, threshed = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    #find maximum contour and draw   
    cmax = max(contours, key = cv2.contourArea) 
    epsilon = 0.001 * cv2.arcLength(cmax, True)
    approx = cv2.approxPolyDP(cmax, epsilon, True)
    cv2.drawContours(input, [approx], -1, (0, 255, 0), 3)

    cv2.imshow("Contour", input)

    width, height = gray.shape

    #fill maximum contour and draw   
    img = np.zeros( [width, height, 1],dtype=np.uint8 )
    cv2.fillPoly(img, pts =[cmax], color=(255,255,255))

    cv2.imwrite(f"outer_contour_{image}", img)

if __name__ == "__main__":
    
    lower_color_boundary = np.array([0,120,0])
    upper_color_boundary = np.array([22,255,255])

    retrieve_image("test4.jpg", lower_color_boundary, upper_color_boundary)
    eliminate_inner_contour("resulted_image.jpg")
    draw_outer_contour("eliminate_inner_contour_resulted_image.jpg")
