import numpy as np
import cv2
import math
from numpy import ones,vstack
from statistics import mean

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

    processed_img =  cv2.Canny(image, threshold1 = 250, threshold2=400) #def: 250, 400

    processed_img = cv2.GaussianBlur(processed_img,(5,5),3)

    vertices = np.array([[0,480],[0,260],[260,160],[380,160],[640,260],[640,480],
                         ], np.int32)
    processed_img = roi(processed_img, [vertices])

    #                                     rho   theta   thresh  min length, max gap:
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 300,      15,       150)           #def: 300, 15, 150

    left_group = []
    right_group = []

    if lines.any():
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
        right_angle = math.pi - math.atan(abs(right_slope))

    if left_group:
        average_left = average_lane(left_group)
        cv2.line(processed_img, (average_left[0], average_left[1]), (average_left[2], average_left[3]), [255,0,0], 3)
        left_slope = line_slope(average_left)
        left_angle = math.atan(abs(left_slope))

    if right_group and left_group:
        print('Left angle: ' + str(left_angle))
        print('Right angle: ' + str(right_angle))

        target_angle = (left_angle + right_angle)/2
        target_angle = -target_angle
        target_slope = math.tan(target_angle)

        print('Target angle: ' + str(target_angle))
        y = 300
        x = int((y-480)/-target_slope + 320)
        cv2.line(processed_img, (320, 480), (x, y), [255,0,0], 3)

    print('Left slope: ' + str(left_slope))
    print('Right slope: ' + str(right_slope))

    print('Target slope: ' + str(target_slope))
    print('(X: ' + str(x) + ' )(Y: ' + str(y) + ' )')

    return processed_img

# road_simple      - success
# road_simple_2    - success
# road_simple_high - success
# road_hard        - 60%
# offroad          - impossible
# road_left        - 80%
# road_left_2      - 70%
# road_left_3      - success
# road_right       - 60%
# road_right_2     - 80%
# road_obst        - success
# road_obst_2      - fail
# road_border      - success
# road_corner      - success
# road_real

cv2.setUseOptimized(True);
cv2.setNumThreads(4);

screen =  np.array(cv2.imread('road_simple.jpg'))
screen =  cv2.Canny(screen, threshold1 = 250, threshold2=400) #def: 250, 400
screen = cv2.GaussianBlur(screen,(5,5),3)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(screen)
ss.switchToSelectiveSearchQuality()
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))

# number of region proposals to show
numShowRects = 100
    # increment to increase/decrease total number
    # of reason proposals to be shown
increment = 20

while True:
        # create a copy of original image
    imOut = screen.copy()

        # itereate over all the region proposals
    for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect
            cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break

        # show output
    cv2.imshow("Output", imOut)

        # record key press
    k = cv2.waitKey(0) & 0xFF

        # m is pressed
    if k == 109:
            # increase total number of rectangles to show by increment
        numShowRects += increment
        # l is pressed
    elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
        numShowRects -= increment
        # q is pressed
    elif k == 113:
        break

#new_screen = process_img(screen)
#cv2.imshow('window', new_screen)

#cv2.waitKey(0)
cv2.destroyAllWindows()
