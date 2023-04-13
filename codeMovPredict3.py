import cv2
import numpy as np

cap = cv2.VideoCapture('C:/Users/Vespertino/Documents/predictMove/tester3.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
frame_num = 0
prev_contours = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    fgmask = fgbg.apply(frame)
    kernel = np.ones((3,3),np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    curr_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        
        (x, y, w, h) = cv2.boundingRect(contour)
        curr_contours.append((x, y, w, h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if prev_contours:
        for i, curr_contour in enumerate(curr_contours):
            min_distance = float('inf')
            closest_prev_contour = None
            
            for j, prev_contour in enumerate(prev_contours):
                distance = ((curr_contour[0] + curr_contour[2]/2) - (prev_contour[0] + prev_contour[2]/2))**2 + ((curr_contour[1] + curr_contour[3]/2) - (prev_contour[1] + prev_contour[3]/2))**2
                if distance < min_distance:
                    min_distance = distance
                    closest_prev_contour = j
            
            if closest_prev_contour is not None:
                direction = ''
                x_diff = curr_contour[0] - prev_contours[closest_prev_contour][0]
                y_diff = curr_contour[1] - prev_contours[closest_prev_contour][1]
                if x_diff > 0:
                    direction += 'right'
                elif x_diff < 0:
                    direction += 'left'
                if y_diff > 0:
                    direction += 'down'
                elif y_diff < 0:
                    direction += 'up'
                
                if direction:
                    prev_contours.pop(closest_prev_contour)
                    cv2.putText(frame, direction, (curr_contour[0], curr_contour[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.imshow('frame', frame)
                    cv2.waitKey(0)
                    break
            else:
                prev_contours.append(curr_contour)
    
    prev_contours = curr_contours
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
