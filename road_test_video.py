import numpy as np
import cv2
import math
from numpy import ones,vstack
from statistics import mean

SCREEN_HEIGHT = 720
SCREEN_WIDTH = 1280

def roi(img, vertices):

    mask = np.zeros_like(img)

    cv2.fillPoly(mask, vertices, 255)

    masked = cv2.bitwise_and(img, mask)
    return masked

def line_slope(line):
    delta_y = line[3] - line[1]
    delta_x = line[2] - line[0]
    return delta_y / delta_x

def average_lane(lines):
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    for coords in lines:
        x1s.append(coords[0])
        y1s.append(coords[1])
        x2s.append(coords[2])
        y2s.append(coords[3])
    return [int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s))]

def process_img(image):

    processed_img =  cv2.Canny(image, threshold1 = 250, threshold2=400) #def: 350, 400

    processed_img = cv2.GaussianBlur(processed_img,(5,5),1)

    vertices = np.array([[0,SCREEN_HEIGHT],[320,390],[920,390],[SCREEN_WIDTH,SCREEN_HEIGHT],
                         ], np.int32)
    processed_img = roi(processed_img, [vertices])

    #                                     rho   theta   thresh  min length, max gap:
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 300,      15,       50)           #def: 300, 15, 150

    left_group = []
    right_group = []

    if lines is not None:
        for current in lines:
            coords = current[0]
            current_slope = line_slope(coords)

            if current_slope < 0:
                left_group.append(coords)
            else:
                right_group.append(coords)

    right_slope = 0
    left_slope = 0
    target_slope = 0
    left_angle = 0
    right_angle = 0
    x = 0
    y = 0

    if right_group:
        average_right = average_lane(right_group)
        cv2.line(processed_img, (average_right[0], average_right[1]), (average_right[2], average_right[3]), [255,0,0], 3)
        right_slope = line_slope(average_right)
        right_angle = math.atan(right_slope)

    if left_group:
        average_left = average_lane(left_group)
        cv2.line(processed_img, (average_left[0], average_left[1]), (average_left[2], average_left[3]), [255,0,0], 3)
        left_slope = line_slope(average_left)
        left_angle = math.pi - math.atan(abs(left_slope))

    if right_group and left_group:
        print('Left angle: ' + str(left_angle))
        print('Right angle: ' + str(right_angle))

        target_angle = abs((left_angle + right_angle)/2)
        target_angle = -target_angle
        target_slope = math.tan(target_angle)

        print('Target angle: ' + str(target_angle))
        y = 300
        x = int((y-SCREEN_HEIGHT)/-target_slope + SCREEN_WIDTH/2)
        cv2.line(processed_img, (int(SCREEN_WIDTH/2), SCREEN_HEIGHT), (x, y), [255,0,0], 3)

    print('Left slope: ' + str(left_slope))
    print('Right slope: ' + str(right_slope))

    print('Target slope: ' + str(target_slope))
    print('(X: ' + str(x) + ' )(Y: ' + str(y) + ' )')

    return processed_img

cap = cv2.VideoCapture('skylines.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    new_screen = process_img(frame)
    cv2.imshow('window', new_screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()