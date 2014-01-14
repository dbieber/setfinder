import cv2
import numpy as np
import math
import card_finder
from sklearn.svm import SVC
from memoize import memoized

NUMBER, SHADING, COLOR, SHAPE = 0, 1, 2, 3

class Card():
    def __init__(self, image):
        self.image = image  # already rectified

        self.number = None
        self.shading = None
        self.color = None
        self.shape = None

        self.grayimage = None
        self.edgesimage = None
        self.hsvimage = None
        self.centerpt = None
        self.distlist = None

        self.shape_pred = None

    def opencv_show(self):
        cv2.destroyWindow('card_window')
        cv2.imshow('card_window', self.image)

    def opencv_show_canny(self):
        cv2.destroyWindow('canny_window')
        cv2.imshow('canny_window', self.edgesimage)
        cv2.moveWindow('canny_window', 400, 0)

    def gray(self):
        if self.grayimage is None:
            self.grayimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.grayimage

    def edges(self):
        if self.edgesimage is None:
            self.edgesimage = cv2.Canny(self.gray(), 404/8, 156/8, apertureSize=3)
        return self.edgesimage

    def hsv(self):
        if self.hsvimage is None:
            self.hsvimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        return self.hsvimage

    def center(self):
        if self.centerpt is None:
            number = 2 if self.predict_number() == "two" else 1
            self.centerpt = find_center(self.edges(), number)
        return self.centerpt

    def fail(self):
        if 'fail' in self.labels():
            return True

    def labels(self):
        return [self.predict_number(), self.predict_shading(), self.predict_color(), self.predict_shape()]

    def dists(self):
        if self.distlist is None:
            canny = self.edges()
            center = self.center()
            dists = dists_to_edges(canny, center)
            for i in range(len(dists)):
                dists[i] = dists[i]**3
            denom = np.linalg.norm(dists)
            if denom != 0:
                self.distlist = dists / denom
            else:
                self.distlist = dists
        return self.distlist

    def number_features(self):


    def predict_number(self):
        if self.number is None:
            edges = self.edges()
            height, width = edges.shape
            x = width / 2
            count = 0
            waiting = 0
            for y in range(5, height-5):
                if edges[y][x] > 0 and waiting <= 0:
                    waiting = 10
                    count = count + 1
                waiting = waiting - 1

            count = count / 2
            if count == 1:
                self.number = "one"
            elif count == 2:
                self.number = "two"
            elif count == 3:
                self.number = "three"
            else:
                self.number = "fail"
        return self.number

    # <codecell>

    def predict_shading(self):
        if self.shading is None:
            gray = self.gray()
            edges = cv2.Canny(gray, 404/4, 156/4, apertureSize=3)
            number = self.number
            height, width = gray.shape

            tc = (10,50) # topcenter row,col
            white = np.mean(gray[tc[0]-2:tc[0]+2,tc[1]-2:tc[1]+2])

            center = self.center()

            ws = 2 # window size
            window = gray[center[1]-ws:center[1]+ws, center[0]-ws:center[0]+ws]
            avg_color = np.mean(window)

            ratio = avg_color / white
            if ratio < .6:
                self.shading = 'solid'
            elif ratio < .95:
                self.shading = 'striped'
            else:
                self.shading = 'empty'
        return self.shading

    # <codecell>

    def predict_color(self, flag=False):
        if self.color is None:
            gray = self.gray()
            edges = self.edges() #cv2.Canny(gray, 404/4, 156/4, apertureSize=3)
            image = self.hsv()

            # Assumes rectified image
            height, width = edges.shape
            x = width / 2

            # weird HSV

            RED = np.array([2, 224, 189])
            GREEN = np.array([70, 240, 105])
            PURPLE = np.array([166, 207, 79])

            # HSV
            """
            RED = np.array([5, 88, 74])
            GREEN = np.array([141, 94, 41])
            PURPLE = np.array([293, 81, 31])
            """
            colors = [RED, GREEN, PURPLE]
            color_names = ["red", "green", "purple"]
            color_counts = np.array([0, 0, 0])
            for x in range(width/2-10, width/2+10, 4):

                inside = False
                ever_switch = False
                beginning = True
                beg_count = 0
                beg_color = np.array([0, 0, 0])
                color_counts_before = color_counts.copy()
                for y in range(5, height-5):
                    if edges[y, x] > 0:
                        if beginning and beg_count != 0:
                            beg_color /= beg_count
                        if flag:
                            print 'beg_color', beg_color
                        beginning = False
                        inside = not inside
                        ever_switch = True

                    if beginning:
                        beg_count += 1
                        beg_color += image[y,x]


                    if inside:
                        color = np.array(image[y,x])
                        if flag:
                            print color

                        if np.linalg.norm(color-beg_color, ord=1) < 60:
                            continue
                        """
                        if color[0] <= 20 or 165 <= color[0]:
                            color_counts[0] += 1
                        if 30 <= color[0] <= 90:
                            color_counts[1] += 1
                        if 100 <= color[0] <= 165:
                            color_counts[2] += 1
                        """
                        dists = []
                        for c in colors:
                            d1 = abs(c[0]-color[0])
                            d2 = abs(c[0]+180-color[0])
                            dists.append(min(d1,d2))
                        if np.min(dists) < 25:
                            color_counts[np.argmin(dists)] += 1
                if not ever_switch or inside:
                    color_counts = color_counts_before

            self.color = color_names[np.argmax(color_counts)]
        return self.color

    # <codecell>

    def predict_shape(self):
        if self.shape is None:
            dists = self.dists()

            shape_distlists = [c.dists() for c in [diamondcard, squigglecard, rectanglecard]]
            sims = [similarity(dists, distlist) for distlist in shape_distlists]

            if np.max(sims) < .5:
                self.shape = 'fail'
            else:
                shapes = ["diamond", "squiggle", "rounded-rectangle"]
                self.shape = shapes[np.argmax(sims)]
                self.shape_pred = sims

        return self.shape


def main():

    vc = cv2.VideoCapture(0)
    cv2.namedWindow('win', cv2.CV_WINDOW_AUTOSIZE)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    looking_at = 0
    while True:
        rval, frame = vc.read()

        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break

        if frame is None:
            continue

        frame = cv2.resize(frame, (640,480))
        quads = []
        for i in range(0, 3):
            quads.extend(card_finder.find_cards_with_parameter_setting(frame, i))

        quads = card_finder.reduce_quads(quads)
        cards = []
        for q in quads:
            card = Card(rectify(frame, q))
            if not card.fail():
                cards.append(card)

        print set(' '.join(c.labels()) for c in cards)

        image = mark_quads(frame, quads)
        cv2.imshow('win', image)

        rval, frame = vc.read()
        looking_at = 0

        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            break
        if len(cards) > 0:
            key = cv2.waitKey(0)
            while key != ord('t'):
                if key == ord('p'):
                    looking_at -= 1
                    if looking_at < 0:
                        looking_at = len(cards) - 1
                    cards[looking_at].opencv_show()
                    cards[looking_at].opencv_show_canny()
                    print cards[looking_at].shape_pred
                    print ' '.join(cards[looking_at].labels())

                if key == ord('n'):
                    looking_at += 1
                    if looking_at >= len(cards):
                        looking_at = 0
                    cards[looking_at].opencv_show()
                    cards[looking_at].opencv_show_canny()
                    print cards[looking_at].shape_pred
                    print ' '.join(cards[looking_at].labels())

                if key == 27: # exit on ESC
                    sys.exit(0)

                key = cv2.waitKey(0)

"""
 A card_list should contain at least four entries for every card.
 Entry 1) Color
 Entry 2) Shape
 Entry 3) Number
 Entry 4) Shading
 There can be more entries, in fact we encourage using the fifth
 entry to be the index into the quad array
"""
def calc_sets(card_list):
    for i in range(len(card_list)):
        for j in range(i+1, len(card_list)):
            for k in range(j+1, len(card_list)):
                for l in range(4):
                    if ((card_list[i][l] == card_list[j][l] == card_list[k][l]) or
                        (card_list[i][l] != card_list[j][l] and
                        card_list[j][l] != card_list[k][l] and
                        card_list[k][l] != card_list[i][l])):
                        return i,j,k
    return None

"""
cards is an array of three indicies into the quads array
"""
def mark_set(image, cards, quads):
    for c in cards:
        arr = [np.array(quads[c],'int32')]
        cv2.polylines(image, arr, True, (0,0,100), thickness=3)

    return image

def mark_quads(image, quads):
    for q in quads:
        arr = [np.array(q,'int32')]
        cv2.fillPoly(image,arr,(0,0,100))
    return image

if __name__ == '__main__':
    main()

