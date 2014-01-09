import cv
import cv2
import math
import numpy as np
import scipy.spatial

cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)

cards = cv2.imread('setcards.jpg')
blank = np.zeros(cards.shape)
CARD_DIMENSIONS = (50, 50)


# Playing card
# corners = np.float32([
#     (63, 74),
#     (161, 32),
#     (271, 143),
#     (151, 218)
# ])

# Triple diamond
# corners = np.float32([
#     (171, 231),
#     (297, 229),
#     (299, 420),
#     (177, 421)
# ])

# Purple squiggle
corners = np.float32([
    (185, 437),
    (306, 436),
    (306, 626),
    (184, 626)
])

target_corners = np.float32([
    (0, 0),
    (CARD_DIMENSIONS[0], 0),
    (CARD_DIMENSIONS[0], CARD_DIMENSIONS[1]),
    (0, CARD_DIMENSIONS[1])
])

m, mask = cv2.findHomography(corners, target_corners)

skew = cv2.warpPerspective(cards, m, CARD_DIMENSIONS)

y,x,depth = skew.shape
print y,x

BLACK = [0,0,0]
WHITE = [255,255,255]
CARDWHITE = [201,194,178]
RED = [208,38,31]
GREEN = [0,137,66]
PURPLE = [69,24,70]
colors = [BLACK, WHITE, CARDWHITE, RED, GREEN, PURPLE]
color_counts = [0] * len(colors) # black,white,cardwhite, orange, green, purple
for i in xrange(y):
    for j in xrange(x):
        color = list(reversed(skew[i,j,:]))
        dists = scipy.spatial.distance.cdist([color], colors)
        if np.min(dists) < 50:
            color_counts[np.argmin(dists)] += 1

print color, color_counts



# cv2.imshow("w1", skew)

# raw_input()
