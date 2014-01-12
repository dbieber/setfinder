# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#%pylab inline
import cv
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import math
import card_finder

# <codecell>

NUMBER, SHADING, COLOR, SHAPE = 0, 1, 2, 3

# <headingcell level=1>

# Classifier

# <codecell>

# <headingcell level=1>

# Data

# <codecell>

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

    def opencv_show(self):
        cv2.destroyWindow('card_window')
        cv2.imshow('card_window', self.image)
        
    def gray(self):
        if self.grayimage is None:
            self.grayimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.grayimage
    
    def edges(self):
        if self.edgesimage is None:
            self.edgesimage = cv2.Canny(self.gray(), 404/3, 156/3, apertureSize=3)
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
            self.distlist = dists / np.linalg.norm(dists)
        return self.distlist

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
            edges = cv2.Canny(gray, 404/2, 156/2, apertureSize=3)
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
            edges = cv2.Canny(gray, 404/4, 156/4, apertureSize=3)
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

            shapes = ["diamond", "squiggle", "rounded-rectangle"]
            self.shape = shapes[np.argmax(sims)]
        return self.shape



# <codecell>

def find_center(canny, number=1):
    # x,y
    center = (50,70)
    if number == 2:
        center = (50,48)
        
    for i in xrange(2):
        distRight = distance_to_edge(canny, center, 0)
        distLeft = distance_to_edge(canny, center, np.pi)
        center = (center[0] + (distRight - distLeft) / 2, center[1])
        
        distUp = distance_to_edge(canny, center, np.pi/2)
        distDown = distance_to_edge(canny, center, 3*np.pi/2)
        center = (center[0], center[1] + (distUp - distDown) / 2)
        
    center = (int(center[0]), int(center[1]))
    return center

# <codecell>

def distance_to_edge(canny, pt, direction):
    dist = 0.0
    point = pt  # x,y
    while dist < 50 and not is_edge_at(canny, point):
        point = (int(pt[0] + dist * np.cos(direction)), int(pt[1] + dist * np.sin(direction)))
        dist += 1
    return dist

# <codecell>

def dists_to_edges(canny, point):
    dists = []
    for direction in np.arange(0,2*np.pi,np.pi/24):
        dist = distance_to_edge(canny, point, direction)
        if dist == 50 and len(dists) > 0:
            dist = dists[-1]
        dists.append(dist)
    return np.array(dists)

# <codecell>

def is_edge_at(canny, point):
    # point is x,y
    window = canny[point[1]-1:point[1]+1,point[0]-1:point[0]+1]
    return np.sum(window) > 0

# <codecell>

def similarity(a1,a2):
    return 1 - np.sqrt(sum(a1-a2)**2)

# <codecell>

CARD_DIMENSIONS = (100, 140)

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


def test_cards():
    X, Y = gen_data("data/training.txt")

    # <codecell>

    counts = [0] * 5
    for card, labels in zip(X, Y):
        card.predict_number()
        card.predict_shading()
        card.predict_color()
        card.predict_shape()
        
        numberok = card.number == labels[NUMBER]
        shadingok = card.shading == labels[SHADING]
        colorok = card.color == labels[COLOR]
        shapeok = card.shape == labels[SHAPE]
        
        if not shadingok:
            gray = card.gray().copy()
            cv2.circle(gray, card.center(), 1, 255)
            #show(gray)
        
        counts[NUMBER] += numberok
        counts[SHADING] += shadingok
        counts[COLOR] += colorok
        counts[SHAPE] += shapeok
        counts[4] += numberok and shadingok and colorok and shapeok
    #     print "Predict: ", ' '.join(str(x) for x in card.labels())
    #     print "Actual:  ", ' '.join(labels)
    #     print
        
    print counts, len(X)

# <codecell>

#def show(image):
#    plt.imshow(image, cmap=cm.Greys_r)
#    plt.show()

# <codecell>

def main():

    print card_finder
    print 'hit any key to take a picture'
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    looking_at = 0
    while rval:
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
                    print ' '.join(cards[looking_at].labels())

                if key == ord('n'):
                    looking_at += 1
                    if looking_at >= len(cards):
                        looking_at = 0
                    cards[looking_at].opencv_show()
                    print ' '.join(cards[looking_at].labels())

                if key == 27: # exit on ESC
                    break
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
    

trainX, trainY = gen_data("data/training.txt")
diamondcard = exampleofcard(SHAPE, "diamond")
rectanglecard = exampleofcard(SHAPE, "rounded-rectangle")
squigglecard = exampleofcard(SHAPE, "squiggle")

main()


