import cv
import cv2
import math
import numpy as np

cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)

img1 = cv2.imread('setcards.jpg')
img2 = np.zeros(img1.shape)

frame = cv2.Canny(img1, 10, 100, 3)
# lines = cv2.HoughLines(frame, 1, math.pi/180, 200)

# COPIED:
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 404, 156, apertureSize=3)

lines = cv2.HoughLinesP(edges,1,np.pi/180,50, minLineLength=20, maxLineGap=2)

for x1,y1,x2,y2 in lines[0]:
    cv2.line(img1,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imshow("w1", edges)

var = {
    'threshold1': 6910,
    'threshold2': 2500,
    'apertureSize': 5,
    'houghParam1': 1,
    'houghParam2': np.pi/180,
    'houghParam3': 50,
    'minLineLength': 20,
    'maxLineGap': 2,
}
curvar = 'threshold1'
amt = 2.0
toShow = 'edges'

cv2.imshow("w1", img1)

lasterr = None
while True:
    try:
        threshold1 = var['threshold1']
        threshold2 = var['threshold2']
        apertureSize = int(var['apertureSize'])
        edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=apertureSize)

        houghParam1 = var['houghParam1']
        houghParam2 = var['houghParam2']
        houghParam3 = var['houghParam3']
        minLineLength = var['minLineLength']
        maxLineGap = var['maxLineGap']
        lines = cv2.HoughLinesP(edges, houghParam1, houghParam2, houghParam3, minLineLength=minLineLength, maxLineGap=maxLineGap)

        if toShow != 'edges':
            img1 = cv2.imread('setcards.jpg')
            img2 = np.zeros(img1.shape)
            if lines is not None and lines[0] is not None:
                for x1,y1,x2,y2 in lines[0]:
                    cv2.line(img1, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.line(img2, (x1,y1), (x2,y2), (0,0,255), 2)

        if toShow == 'edges':
            cv2.imshow("w1", edges)
        elif toShow == 'img1':
            cv2.imshow("w1", img1)
        elif toShow == 'img2':
            cv2.imshow("w1", img2)

        lasterr = None
    except Exception as e:
        if str(lasterr) != str(e):
            lasterr = e
            print e
        cv2.imshow("w1", np.zeros(img1.shape))

    c = cv.WaitKey(10)
    if c == ord('='):
        var[curvar] += amt
    elif c == ord('-'):
        if var[curvar] - amt > 0:
            var[curvar] -= amt
    elif c == ord('1'):
        curvar = 'threshold1'
    elif c == ord('2'):
        curvar = 'threshold2'
    elif c == ord('3'):
        curvar = 'apertureSize'
    elif c == ord('4'):
        curvar = 'houghParam1'
    elif c == ord('5'):
        curvar = 'houghParam2'
    elif c == ord('6'):
        curvar = 'houghParam3'
    elif c == ord('7'):
        curvar = 'minLineLength'
    elif c == ord('8'):
        curvar = 'maxLineGap'
    elif c == ord('['):
        amt /= 2
    elif c == ord(']'):
        amt *= 2
    elif c == ord('p'):
        print var,amt,toShow
    elif c == ord('z'):
        toShow = 'img1'
    elif c == ord('x'):
        toShow = 'img2'
    elif c == ord('c'):
        toShow = 'edges'
    elif c == ord('q'):
        exit(0)

raw_input()
