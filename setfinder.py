# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#%pylab inline
#import cv
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as pl
import math
import random
import card_finder
from memoize import memoized
import threading
import time
import subprocess

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# <codecell>

NUMBER, SHADING, COLOR, SHAPE = 0, 1, 2, 3
CARD_DIMENSIONS = (100, 140)  # x,y

# <headingcell level=1>

# Classifier

# <codecell>

# <headingcell level=1>

# Data

# <codecell>
class Classifier():
    def __init__(self):
        self.number_clf = SVC()
        self.shading_clf = RandomForestClassifier(50)
        self.color_clf = RandomForestClassifier(50)
        self.shape_clf = RandomForestClassifier(50)

    def fit(self, X, Y):
        for card, labels in zip(X, Y):
            card.known_number = labels[NUMBER]

        number_features = [card.number_features() for card in X]
        shading_features = [card.shading_features() for card in X]
        color_features = [card.color_features() for card in X]
        shape_features = [card.shape_features() for card in X]

        self.number_clf.fit(number_features, Y[:,NUMBER])
        self.shading_clf.fit(shading_features, Y[:,SHADING])
        self.color_clf.fit(color_features, Y[:,COLOR])
        self.shape_clf.fit(shape_features, Y[:,SHAPE])

class Card():
    def __init__(self, image):
        self.image = image  # already rectified
        self.params = [0.0625, .125, .3, .5, .75, 1]
        self.shading_params = [(50,6), (20, 6), (80, 6)]
        self.known_number = None

    def opencv_show_detailed_shading(self):

        image = self.image.copy()

        for tc in self.shading_params: # topcenter row,col
            cv2.rectangle(image, (tc[1]-2, tc[0]-2), (tc[1]+2, tc[0]+2), (200, 100, 0))

        center = self.center()

        ws = 2 # window size
        cv2.rectangle(image, (center[0]-ws, center[1]-ws), (center[0]+ws, center[1]+ws), (200, 100, 0))

        name = 'image'
        center = self.center()
        cv2.destroyWindow(name)
        cv2.imshow(name, image)
        cv2.moveWindow(name, 0, 0)



    def opencv_show_detailed(self):
        pos = 0

        image = self.image.copy()
        name = 'image'
        cv2.destroyWindow(name)
        cv2.imshow(name, image)
        cv2.moveWindow(name, pos, 0)
        pos += 150

        name = 'edges: %d %d %d' % cannyrows(self.edges())
        edgesimage = self.edges().copy()
        cv2.circle(edgesimage, self.center(), 1, 255)
        cv2.destroyWindow(name)
        cv2.imshow(name, edgesimage)
        cv2.moveWindow(name, pos, 0)
        pos += 150

        blur = cv2.blur(self.gray(), ksize=(4,4))
        name = 'blur'
        cv2.destroyWindow(name)
        cv2.imshow(name, blur)
        cv2.moveWindow(name, pos, 0)

        blur2 = cv2.blur(self.gray(), ksize=(6,6))
        name = 'blur2'
        cv2.destroyWindow(name)
        cv2.imshow(name, blur)
        cv2.moveWindow(name, pos, 200)
        pos += 150

        for p in self.params:
            canny = cv2.Canny(blur, 404*p, 156*p, apertureSize=3)
            r1,r2,r3 = cannyrows(canny)
            name = 'bc %.3f %d %d %d' % (p, r1,r2,r3)
            cv2.destroyWindow(name)
            cv2.imshow(name, canny)
            cv2.moveWindow(name, pos, 0)

            canny = cv2.Canny(blur2, 404*p, 156*p, apertureSize=3)
            r1,r2,r3 = cannyrows(canny)
            name = 'bc2 %.3f %d %d %d' % (p, r1,r2,r3)
            cv2.destroyWindow(name)
            cv2.imshow(name, cv2.Canny(blur2, 404*p, 156*p, apertureSize=3))
            cv2.moveWindow(name, pos, 200)

            canny = cv2.Canny(self.gray(), 404*p, 156*p, apertureSize=3)
            r1,r2,r3 = cannyrows(canny)
            name = 'c %.3f %d %d %d' % (p, r1,r2,r3)
            cv2.destroyWindow(name)
            cv2.imshow(name, cv2.Canny(self.gray(), 404*p, 156*p, apertureSize=3))
            cv2.moveWindow(name, pos, 400)
            pos += 150

    def opencv_show(self):
        cv2.destroyWindow('card_window')
        cv2.imshow('card_window', self.image)

    def opencv_show_canny(self):
        cv2.destroyWindow('canny_window')
        cv2.imshow('canny_window', self.edges())
        cv2.moveWindow('canny_window', 400, 0)

    @memoized
    def gray(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    @memoized
    def edges(self):
        return self.best_canny()
        return self.blurred_canny()
        return cv2.Canny(self.gray(), 404/4, 156/4, apertureSize=3)

    @memoized
    def hsv(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    @memoized
    def center(self):
        number = 2 if self.predict_number() == "two" else 1
        return find_center(self.edges(), number)

    @memoized
    def blurred_canny(self):
        blur = cv2.blur(self.gray(), ksize=(4,4))
        canny = cv2.Canny(blur, 404/8, 156/8, apertureSize=3)
        return canny

    @memoized
    def best_canny(self):

        blur = cv2.blur(self.gray(), ksize=(4,4))
        # prevgood, prevbad = 0, 1000
        for i, p in enumerate(self.params):
            canny = cv2.Canny(blur, 404*p, 156*p, apertureSize=3)
            number = self.predict_number_old(canny)
            if number == "fail":
                # print self.predict_number_old(canny, verbose=True)
                continue
            else:
                # print number
                break

            # good_rows, bad_rows, maxconsecgoodrows = cannyrows(canny)

            # if (good_rows < prevgood and bad_rows >= prevbad) or (good_rows == 0 and prevgood != 0):
            #     # if not getting better, return the previous image
            #     p = self.params[i-1]
            #     return cv2.Canny(blur, 404*p, 156*p, apertureSize=3)

            # # if  good_rows < prevgood * .85 and (bad_rows == 0 or bad_rows >= .85 * prevbad):
            # #     # if we lose a lot of good rows and not many bad rows, maybe we lost a whole shape!
            # #     p = self.params[i-1]
            # #     return cv2.Canny(blur, 404*p, 156*p, apertureSize=3)

            # if bad_rows == 0:
            #     break


            # prevgood, prevbad = good_rows, bad_rows

            # TODO(Bieber): Try with and without this
            # if good_rows > 10 and bad_rows < 5:
            #     break

        return canny


    def fail(self):
        return 'fail' in self.labels()

    def labels(self):
        return [self.predict_number(), self.predict_shading(), self.predict_color(), self.predict_shape()]

    @memoized
    def dists(self, canny = None):
        if canny is None:
            canny = self.edges()
        center = self.center()
        return dists_to_edges(canny, center)

    @memoized
    def number_features2(self):
        edges = self.edges()
        height, width = edges.shape

        num_rows = []
        for y in range(5, height-5):
            num_rows.append(1 if sum(edges[y,5:width-5]) > 0 else 0)

        return num_rows

    @memoized
    def number_features(self):
        edges = self.edges()
        height, width = edges.shape
        xs = [width / 2 + 5]
        features = []
        for x in xs:
            count = 0
            waiting = 0
            for y in range(5, height-5):
                if edges[y,x] > 0 and waiting <= 0:
                    waiting = 10
                    count = count + 1
                waiting = waiting - 1

            features.append(count)

        return features

    @memoized
    def predict_number(self):
        return self.known_number or CLASSIFIER.number_clf.predict(self.number_features())[0]

    def predict_number_old(self, edges, verbose=False):
        height, width = edges.shape
        xs = [width / 2 - 10, width / 2 - 5, width / 2, width / 2 + 5, width / 2 + 10]
        answers = []
        for x in xs:
            count = 0
            waiting = 0
            for y in range(5, height-5):
                if edges[y,x] > 0 and waiting <= 0:
                    waiting = 10
                    count = count + 1
                waiting = waiting - 1
            answers.append(count)

        if verbose:
            print answers

        if all(a == answers[0] for a in answers[1:]):
            if answers[0] in [2, 4, 6]:
                return answers[0] / 2

        return "fail"

    # <codecell>

    @memoized
    def shading_features(self):
        gray = self.gray()
        edges = self.edges()
        # edges = self.blurred_canny()
        # edges = cv2.Canny(gray, 404/4, 156/4, apertureSize=3)
        height, width = gray.shape
        image = self.hsv()

        ratios = []

        center = self.center()
        window = []
        ws = 5 # window size
        window.append(.001 + np.mean(image[center[1]-ws:center[1]+ws, center[0]-ws:center[0]+ws, 0]))
        window.append(.001 + np.mean(image[center[1]-ws:center[1]+ws, center[0]-ws:center[0]+ws, 1]))
        window.append(.001 + np.mean(gray[center[1]-ws:center[1]+ws, center[0]-ws:center[0]+ws]))

        for tc in self.shading_params: # topcenter row,col
            white = []

            white.append(.001 + np.mean(image[tc[0]-2:tc[0]+2,tc[1]-2:tc[1]+2,0]))
            white.append(.001 + np.mean(image[tc[0]-2:tc[0]+2,tc[1]-2:tc[1]+2,1]))
            white.append(.001 + np.mean(gray[tc[0]-2:tc[0]+2,tc[1]-2:tc[1]+2]))

            ratio = np.array(white)/np.array(window)

            ratios.append(ratio[1])
            ratios.append(ratio[2])
        return ratios

    @memoized
    def new_shading_features(self):
        gray = self.gray()
        edges = self.edges()
        # edges = self.blurred_canny()
        # edges = cv2.Canny(gray, 404/4, 156/4, apertureSize=3)
        height, width = gray.shape
        image = self.hsv()

        ratios = []
        white = []
        window = []
        for tc in [(10,50), (10, 20), (10, 80)]: # topcenter row,col
            h1 = cv2.calcHist( [gray[5:20, 5:width-5]], [0], None, [16], [0, 256] )
            #white.append(np.mean(image[tc[0]-2:tc[0]+2,tc[1]-2:tc[1]+2,0]))
            #white.append(np.mean(image[tc[0]-2:tc[0]+2,tc[1]-2:tc[1]+2,1]))
            #white.append(np.mean(gray[tc[0]-2:tc[0]+2,tc[1]-2:tc[1]+2]))

            center = self.center()
            wy = 4
            wx = 10

            h2 = cv2.calcHist( [gray[center[1]-wy:center[1]+wy, center[0]-wx:center[0]+wx]], [0], None, [160], [0, 256] )

            ws = 2 # window size
            #window.append(np.mean(image[center[1]-ws:center[1]+ws, center[0]-ws:center[0]+ws, 0]))
            #window.append(np.mean(image[center[1]-ws:center[1]+ws, center[0]-ws:center[0]+ws, 1]))
            #window.append(np.mean(gray[center[1]-ws:center[1]+ws, center[0]-ws:center[0]+ws]))

            #ratio = np.array(window) / np.array(white)

            #ratios.append(ratio[1])
            #ratios.append(ratio[2])


        return np.reshape(h2, np.prod(h2.shape)) #[np.mean(ratios)]


    @memoized
    def predict_shading(self):
        return CLASSIFIER.shading_clf.predict(self.shading_features())[0]

    @memoized
    def predict_shading_old(self):
        gray = self.gray()
        edges = cv2.Canny(gray, 404/4, 156/4, apertureSize=3)
        number = self.predict_number()
        height, width = gray.shape

        tc = (10,10) # topcenter row,col
        white = np.mean(gray[tc[0]-2:tc[0]+2,tc[1]-2:tc[1]+2])

        center = self.center()

        ws = 2 # window size
        window = gray[center[1]-ws:center[1]+ws, center[0]-ws:center[0]+ws]
        avg_color = np.mean(window)

        ratio = avg_color / white
        if ratio < .6:
            return 'solid'
        elif ratio < .95:
            return 'striped'
        else:
            return 'empty'

    # <codecell>

    @memoized
    def color_features(self, flag=False):
        gray = self.gray()
        edges = self.edges() #cv2.Canny(gray, 404/4, 156/4, apertureSize=3)
        image = self.hsv()

        # Assumes rectified image
        height, width = edges.shape
        x = width / 2

        # ch0 = cv2.calcHist([image], [0], None, [256], [0,256])[:,0]
        # ch0 = cv2.calcHist([image], [0], None, [256], [0,256])[:,0]
        # ch0 = cv2.calcHist([image], [0], None, [256], [0,256])[:,0]
        # h = cv2.calcHist( [image], [0, 1, 2], None, [180, 256, 180], [0, 180, 0, 256, 0, 180] )
        h = cv2.calcHist( [image], [0, 1, 2], None, [32, 32, 32], [0, 180, 0, 256, 0, 256] )
        return np.reshape(h, np.prod(h.shape))
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
        avg_colors = []
        avg_sats = []
        final_avg = 0
        final_total = 0
        for x in range(width/2-20, width/2+20, 8):
            avg_color = 0
            avg_sat = 0
            avg_sat_count = 0
            avg_count = 0
            inside = False
            ever_switch = False
            beginning = True
            beg_count = 0
            beg_color = np.array([0, 0, 0])
            color_counts_before = color_counts.copy()
            for y in range(5, height-5):
                color = np.array(image[y,x])
                if color[1] > 50:
                    if color[0] < 25:
                        avg_color += 180
                    avg_color += color[0]

                    avg_count += 1
                avg_sat += color[2]
                avg_sat_count += 1

            if avg_count < 0:

                avg_color = 0
                avg_sat = 0
                avg_sat_count = 0
                avg_count = 0
                for y in range(5, height-5):
                    color = np.array(image[y,x])
                    avg_color += color[0]

                    avg_count += 1
                    avg_sat += color[2]
                    avg_sat_count += 1
                """
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

                color = np.array(image[y,x])
                if not beginning and np.linalg.norm(color-beg_color, ord=1) > 10:

                    avg_color += color
                    avg_count += 1
                """
                """
                if inside:
                    color = np.array(image[y,x])
                    if flag:
                        print color

                    if np.linalg.norm(color-beg_color, ord=1) < 50:
                        continue
                """
                """
                    if color[0] <= 20 or 165 <= color[0]:
                        color_counts[0] += 1
                    if 30 <= color[0] <= 90:
                        color_counts[1] += 1
                    if 100 <= color[0] <= 165:
                        color_counts[2] += 1
                """

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
            """

            if avg_count == 0:
                avg_colors.append(avg_color)

            else:
                avg_colors.append(avg_color/avg_count)

            avg_sats.append(avg_color/avg_sat_count)



        final_total = 0
        total = 0
        for a in avg_colors:
            final_total += a
            total += 1

        if total != 0:
            final_total /= total

        #avg_colors.extend(avg_sats)
        avg_colors.append(final_total)
        return [final_total]  #np.average(avg_colors)


    @memoized
    def predict_color(self):
        return CLASSIFIER.color_clf.predict(self.color_features())[0]

    @memoized
    def predict_color_old(self, flag=False):
        gray = self.gray()
        image = self.hsv()
        edges = cv2.Canny(image[:,:,1], 404/2, 156/2, apertureSize=3)

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

                    if abs(color[1]-beg_color[1]) < 60:
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

        return color_names[np.argmax(color_counts)]

    # <codecell>

    @memoized
    def shape_features(self):
        # sobel = cv2.Sobel(self.gray(), -1, dx=1, dy=1)
        # cv2.imshow('sobel', sobel)
        # h = cv2.calcHist([sobel], [0], None, [256], [0,256])[:,0]
        # # print h
        # return h
        return self.dists()


    @memoized
    def predict_shape(self):
        return CLASSIFIER.shape_clf.predict(self.shape_features())[0]

    @memoized
    def predict_shape_old(self):
        dists = self.dists()

        shape_distlists = [c.dists() for c in [diamondcard, squigglecard, rectanglecard]]
        sims = [similarity(dists, distlist) for distlist in shape_distlists]

        if np.max(sims) < .5:
            return 'fail'
        else:
            shapes = ["diamond", "squiggle", "rounded-rectangle"]
            shape = shapes[np.argmax(sims)]
            return shape

def cannyrows(canny):
    height, width = canny.shape
    features = []
    bad_rows = 0
    good_rows = 0
    maxconsecgoodrows = 0
    numrowchunks = 0
    consec_good_rows = 0
    scanheight = 3
    for y in np.arange(5, height-5-scanheight, scanheight):
        x = 5
        found_edges = 0
        while x < width - 5:
            if sum(canny[y:y+scanheight,x]) > 0:
                found_edges += 1
                while x < width - 5 and sum(canny[y:y+scanheight,x]) > 0:
                    x += 1
            x += 1
        if found_edges > 4:
            bad_rows += 1

        if found_edges == 2 or found_edges == 4:
            good_rows += 1

            if consec_good_rows == 0:
                numrowchunks += 1
            consec_good_rows += 1
            maxconsecgoodrows = max(maxconsecgoodrows, consec_good_rows)
        elif consec_good_rows > 0:
            consec_good_rows = 0
    return good_rows, bad_rows, numrowchunks

# <codecell>

def find_center(canny, number=1):
    center = (CARD_DIMENSIONS[0] / 2, CARD_DIMENSIONS[1] / 2)    # x,y
    if number == 2:
        center = (CARD_DIMENSIONS[0] / 2, CARD_DIMENSIONS[1] / 3)

    for i in xrange(2):
        distRight = distance_to_edge(canny, center, 0)
        distLeft = distance_to_edge(canny, center, np.pi)
        center = (center[0] + (distRight - distLeft) / 2, center[1])

        distUp = distance_to_edge(canny, center, np.pi/2)
        distDown = distance_to_edge(canny, center, 3*np.pi/2)
        center = (center[0], center[1] + (distUp - distDown) / 2)

    center = (ir(center[0]), ir(center[1]))
    return center

# <codecell>
def ir(x): #int round
    return int(round(x))

def distance_to_edge(canny, pt, direction):
    dist = 0.0
    point = pt  # x,y
    dx = np.cos(direction)
    dy = np.sin(direction)
    while dist < 50 and not is_edge_at(canny, point):
        point = (ir(pt[0] + dist * dx), ir(pt[1] + dist * dy))
        dist += 1
    return dist


def color_at_edge(canny, hsv, pt, direction):
    dist = 0.0
    point = pt  # x,y
    dx = np.cos(direction)
    dy = np.sin(direction)
    while dist < 40 and not is_edge_at(canny, point):
        point = (int(pt[0] + dist * dx), int(pt[1] + dist * dy))
        dist += 1
    return np.mean(hsv[point[1]-1:point[1]+1,point[0]-1:point[0]+1,0])

# <codecell>

def dists_to_edges(canny, point):
    dists = []
    for direction in np.arange(0,2*np.pi,np.pi/24):
        dist = distance_to_edge(canny, point, direction)
        if dist == 50 and len(dists) > 0:
            dist = dists[-1]
        dists.append(dist)
    return np.array(dists)

def colors_at_edges(canny, hsv, point):
    colors = []
    for direction in np.arange(0,2*np.pi,np.pi/12):
        color = color_at_edge(canny, hsv, point, direction)
        if not (0 < color and color < 255):
            color = colors[-1]
        colors.append(color)
    return np.array(colors)

# <codecell>

def is_edge_at(canny, point):
    # point is x,y
    x,y = ir(point[0]), ir(point[1])
    window = canny[y-1:y+1,x-1:x+1]
    return np.sum(window) > 0

# <codecell>

def similarity(a1,a2):
    return 1 - np.sqrt(sum(a1-a2)**2)

# <codecell>

def groundtruth(filename="data/ground_truth.txt"):
    groundtruthfile = open(filename)

    for line in groundtruthfile:
        tokens = [x.strip() for x in line.strip().split(":")]
        filename = tokens[0]
        cards = None if len(tokens) == 1 else [x.strip() for x in tokens[1].split(";")]
        yield (filename, cards)

    groundtruthfile.close()

# <codecell>

def attrs_from_card(card):
    tokens = card.split(" ")
    return tokens[:4]

def points_from_card(card):
    tokens = card.split(" ")
    if len(tokens) > 4:
        points = eval("[%s]" % ''.join(tokens[4:]))
        return points

# <codecell>

def rectify(image, pts):
    corners = np.float32(pts)

    target_corners = np.float32([
        (0, 0),
        (CARD_DIMENSIONS[0], 0),
        (CARD_DIMENSIONS[0], CARD_DIMENSIONS[1]),
        (0, CARD_DIMENSIONS[1])
    ])

    m, mask = cv2.findHomography(corners, target_corners)
    skew = cv2.warpPerspective(image, m, CARD_DIMENSIONS)
    return skew

# <codecell>

def gen_data(filename):
    X = []  # cards
    Y = []  # labels
    for i, (filename, cards) in enumerate(groundtruth(filename)):
        image = cv2.imread(filename)
        for j, card in enumerate(cards):
            pts = points_from_card(card)
            if not pts:
                continue

            cardimage = rectify(image, pts)

            X.append(Card(cardimage))
            Y.append(attrs_from_card(card))

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# <headingcell level=1>

# Train

# <codecell>

# <codecell>

# TODO(Bieber): Index to speed up finding particular cards
def exampleofcard(feature=SHADING, value="empty"):
    ANY = "*"
    desiredValues = ["one","empty",ANY,"rounded-rectangle"]
    desiredValues[feature] = value

    for card, labels in zip(trainX, trainY):
        ok = True
        for attr in xrange(4):
            if desiredValues[attr] != ANY and labels[attr] != desiredValues[attr]:
                ok = False
                break
        if ok:
            return card

# <codecell>

# <headingcell level=1>

# Test

# <codecell>

def confusion_matrices(testX, testY):
    pred = [card.predict_number() for card in testX]
    labels = ["one", "two", "three"]
    cm = confusion_matrix(testY[:,NUMBER], pred, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    pl.title('Number Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()

    pred = [card.predict_shading() for card in testX]
    labels = ["empty", "striped", "solid"]
    cm = confusion_matrix(testY[:,SHADING], pred, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    pl.title('Shading Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()

    pred = [card.predict_shape() for card in testX]
    labels = ["rounded-rectangle", "squiggle", "diamond"]
    cm = confusion_matrix(testY[:,SHAPE], pred, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    pl.title('Shape Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()

    pred = [card.predict_color() for card in testX]
    labels = ["red", "green", "purple"]
    cm = confusion_matrix(testY[:,COLOR], pred, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    pl.title('Color Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()


def train(X,Y):
    print "Training on %d images..." % len(X)
    CLASSIFIER.fit(X, Y)
    print "Done training."

def test(X,Y):
    counts = [0] * 5
    print "Testing on %d images..." % len(X)

    for card, labels in zip(X, Y):
        numberok = card.predict_number() == labels[NUMBER]
        shadingok = card.predict_shading() == labels[SHADING]
        colorok = card.predict_color() == labels[COLOR]
        shapeok = card.predict_shape() == labels[SHAPE]

        if not shadingok:
            image = card.image.copy()
            # card.opencv_show_canny()
            card.opencv_show_detailed_shading()
            # print card.number_features()
            # print card.number_features2()
            print "Predict: ", ' '.join(str(x) for x in card.labels())
            print "Actual:  ", ' '.join(labels)
            print
            cv2.waitKey(0)

        counts[NUMBER] += numberok
        counts[SHADING] += shadingok
        counts[COLOR] += colorok
        counts[SHAPE] += shapeok
        counts[4] += numberok and shadingok and colorok and shapeok
    # #     print "Predict: ", ' '.join(str(x) for x in card.labels())
    # #     print "Actual:  ", ' '.join(labels)
    # #     print

    print counts, len(X)

def test_cards_main():
    X, Y = gen_data("data/total_data.txt")

    perm = range(len(X))
    random.shuffle(perm)
    N = 100
    trainX = X[perm[:N]]
    trainY = Y[perm[:N]]
    testX =  X[perm[N:]]
    testY =  Y[perm[N:]]

    train(trainX, trainY)

    # for card, label in zip(X, Y):
    #     card.opencv_show_detailed()
    # print "Moving on..."

    test(testX, testY)

# <codecell>

#def show(image):
#    plt.imshow(image, cmap=cm.Greys_r)
#    plt.show()

# <codecell>

def main():
    def display(frame):
        looking_at = 0

        frame = cv2.resize(frame, (640,480))
        framecopy = frame.copy()
        #quads = []
        #for i in range(0, 7):
        #    quads.extend(card_finder.find_cards_with_parameter_setting(frame, i))

        quads = card_finder.new_card_finder(frame)
        #quads = card_finder.reduce_quads(quads)
        quads = card_finder.rotate_quads(quads)
        cards = []
        for q in quads:
            card = Card(rectify(frame, q))
            if not card.fail():
                cards.append(card)

        print ', '.join(' '.join(card.labels()) for card in cards)

        sets = calc_sets(cards)
        if sets is not None:
            image = mark_set(frame, sets, quads)
        else:
            image = mark_quads(frame, quads)
        cv2.imshow('win', image)
        gray = cv2.cvtColor(framecopy,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 404/2, 156/2, apertureSize=3)
        cv2.imshow('win2', edges)
        cv2.moveWindow('win2', 500,0)

        looking_at = 0

        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            sys.exit(0)
        key = cv2.waitKey(0)
        while key != ord('t'):
            if key == ord('p'):
                looking_at -= 1
                if looking_at < 0:
                    looking_at = len(cards) - 1
                cards[looking_at].opencv_show()
                cards[looking_at].opencv_show_canny()
                print ' '.join(cards[looking_at].labels())

            if key == ord('n'):
                looking_at += 1
                if looking_at >= len(cards):
                    looking_at = 0
                cards[looking_at].opencv_show()
                cards[looking_at].opencv_show_canny()
                print ' '.join(cards[looking_at].labels())

            if key == 27: # exit on ESC
                sys.exit(0)

            key = cv2.waitKey(0)

    if len(sys.argv) > 1:
        use_camera = False
        image = cv2.imread(sys.argv[1])
        display(image)
    else:

        vc = cv2.VideoCapture(0)
        cv2.namedWindow('win', cv2.CV_WINDOW_AUTOSIZE)

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False

        while True:
            rval, frame = vc.read()
            key = cv2.waitKey(10)
            if frame is None:
                continue
            else:
                display(frame)

            if key == 27: # exit on ESC
                break

class PredictionThread(threading.Thread):
    def __init__(self, image):
        threading.Thread.__init__(self)
        self.image = image.copy()
        self.done = False
        self.cards = None
        self.sets = None
        self.quads = None

    def run(self):
        frame = self.image
        quads = card_finder.new_card_finder(frame)
        quads = card_finder.rotate_quads(quads)
        self.quads = quads
        self.cards = []
        for q in quads:
            card = Card(rectify(frame, q))
            if not card.fail():
                self.cards.append(card)

        self.sets = calc_sets(self.cards)
        self.done = True

class VideoDemo():
    def __init__(self):
        self.vc = cv2.VideoCapture(0)
        cv2.namedWindow('win', cv2.CV_WINDOW_AUTOSIZE)

        self.PredictionThread = None

        if self.vc.isOpened(): # try to get the first frame
            self.vc.read()

    def run(self):
        readyForNext = True

        lastcards = None
        lastset = None # indices into lastcards
        lastquads = None

        while True:
            rval, frame = self.vc.read()
            key = cv2.waitKey(10)
            time.sleep(.1)
            if frame is None:
                continue
            else:
                framecopy = frame.copy()
                framecopy = cv2.resize(framecopy, (framecopy.shape[1]/2, framecopy.shape[0]/2))

                if readyForNext:
                    readyForNext = False
                    self.PredictionThread = PredictionThread(framecopy)
                    self.PredictionThread.start()

                if self.PredictionThread and self.PredictionThread.done:
                    if self.PredictionThread.sets:
                        newset = []
                        keepold = False
                        if lastset:
                            keepold = True
                            for ci in lastset:
                                carddesired = lastcards[ci]
                                cardfound = False
                                for i, card in enumerate(self.PredictionThread.cards):
                                    if carddesired.labels() == card.labels():
                                        cardfound = True
                                        newset.append(i)
                                        break
                                if not cardfound:
                                    keepold = False
                                    break
                        if keepold and False:
                            lastset = newset
                        else:
                            lastset = self.PredictionThread.sets
                            subprocess.call(["say", "set"])

                        lastcards = self.PredictionThread.cards
                        lastquads = self.PredictionThread.quads

                    readyForNext = True

                if lastset:
                    mark_set(framecopy, lastset, lastquads)

                cv2.imshow('win', framecopy)

            if key == 27: # exit on ESC
                sys.exit(0)

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
    for i, cardi in enumerate(card_list):
        for j, cardj in enumerate(card_list):
            if j == i:
                continue
            for k, cardk in enumerate(card_list):
                if k == i or k == j:
                    continue
                is_ok = True
                all_same = True
                for l in range(4):
                    if not (cardi.labels()[l] == cardj.labels()[l] == cardk.labels()[l]):
                        all_same = False
                    if not ((cardi.labels()[l] == cardj.labels()[l] == cardk.labels()[l]) or
                        (cardi.labels()[l] != cardj.labels()[l] and
                        cardj.labels()[l] != cardk.labels()[l] and
                        cardk.labels()[l] != cardi.labels()[l])):
                        is_ok = False
                if is_ok and not all_same:
                    return i,j,k
    return None

"""
cards is an array of three indicies into the quads array
"""
def mark_set(image, cards, quads):
    for c in cards:
        arr = [np.array(quads[c],'int32')]
        cv2.polylines(image, arr, True, (0, 143, 255), thickness=3)

    return image

def mark_quads(image, quads):
    for q in quads:
        arr = [np.array(q,'int32')]
        cv2.fillPoly(image,arr,(0,0,100))
    return image

# diamondcard = exampleofcard(SHAPE, "diamond")
# rectanglecard = exampleofcard(SHAPE, "rounded-rectangle")
# squigglecard = exampleofcard(SHAPE, "squiggle")

def showcards(X,Y):
    cv2.namedWindow('win')
    for card, labels in zip(X,Y):
        cv2.destroyWindow('win')
        cv2.imshow('win', card.image)
        raw_input()

if __name__ == '__main__':
    # Global
    CLASSIFIER = Classifier()

    run_tests = False

    if run_tests:
        test_cards_main()
    else:
        #showcards(trainX, trainY)
        trainX, trainY = gen_data("data/total_data.txt")

        perm = range(len(trainX))
        random.shuffle(perm)
        N = 5000
        trainX = trainX[perm[:N]]
        trainY = trainY[perm[:N]]

        train(trainX, trainY)

        demo = VideoDemo()
        demo.run()
