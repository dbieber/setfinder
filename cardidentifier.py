import cv
import cv2
import math
import numpy as np

class CardIdentifier():
    def __init__(self):
        pass

    def predict(self, image):
        return []

    def rectify(self, image, pts):
        CARD_DIMENSIONS = (250, 350)
        target_corners = np.float32([
            (0, 0),
            (CARD_DIMENSIONS[0], 0),
            (CARD_DIMENSIONS[0], CARD_DIMENSIONS[1]),
            (0, CARD_DIMENSIONS[1])
        ])

        corners = np.float32(pts)
        print corners

        m, mask = cv2.findHomography(corners, target_corners)

        skew = cv2.warpPerspective(image, m, CARD_DIMENSIONS)

        return skew
