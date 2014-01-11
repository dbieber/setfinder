import cv
import cv2
import math
import numpy as np
import random
import scipy.spatial
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class CardIdentifier():
    def __init__(self):
        self.n_clusters = 40
        self.patch_size = 25
        self.patches_per_image = 100

        # The classifiers used for identification
        self.patch_clusterer = None
        self.shape_clf = None

    def patches_from(self, image):
        y,x,depth = image.shape
        patches = []

        ps = self.patch_size
        # for i in xrange(self.patches_per_image):
        #     ry = random.randint(0, y - ps - 1)
        #     rx = random.randint(0, x - ps - 1)
        for ry in np.arange(0,1,1.0/10):
            for rx in np.arange(0,1,1.0/10):
                sy = ry * (y-ps)
                sx = rx * (x-ps)
                patch = image[sy:sy+ps,sx:sx+ps,:]
                patch = np.reshape(patch, np.prod(patch.shape))
                patches.append(patch)

        return patches

    def bag_from(self, image):
        patches = self.patches_from(image)
        bag = [0] * self.n_clusters
        for cluster in self.patch_clusterer.predict(patches):
            bag[cluster] += 1
        return bag

    def fit(self, X, Y):
        patches = []
        print "Creating patches..."
        for image, card in zip(X, Y):
            patches.extend(self.patches_from(image))

        print "Clustering patches..."
        self.patch_clusterer = KMeans(init='k-means++', n_clusters=self.n_clusters)
        self.patch_clusterer.fit(patches)

        bags = []
        print "Creating bags..."
        for image, card in zip(X, Y):
            bags.append(self.bag_from(image))

        SHAPE = 3
        print "Training on shapes..."
        self.shape_clf = SVC()
        self.shape_clf.fit(bags, Y[:,SHAPE])

    def predict_shape(self, image):
        bag = self.bag_from(image)
        return self.shape_clf.predict(bag)

    def predict_color(self, image):
        # Assumes rectified image
        y,x,depth = image.shape


        BLACK = [0,0,0]
        WHITE = [255,255,255]
        CARDWHITE = [201,194,178]
        RED = [208,38,31]
        GREEN = [0,137,66]
        PURPLE = [69,24,70]
        colors = [BLACK, WHITE, CARDWHITE, RED, GREEN, PURPLE]
        color_names = ["BLACK", "WHITE", "CARDWHITE", "RED", "GREEN", "PURPLE"]
        color_counts = [0] * len(colors) # black,white,cardwhite, orange, green, purple
        for i in xrange(y):
            for j in xrange(x):
                color = list(reversed(image[i,j,:]))
                dists = scipy.spatial.distance.cdist([color], colors)
                if np.min(dists) < 50:
                    color_counts[np.argmin(dists)] += 1

        color_counts[0] = 0
        color_counts[1] = 0
        color_counts[2] = 0
        print color_counts
        return color_names[np.argmax(color_counts)]

    def predict(self, image):
        # skew = self.rectify(image, pts)
        return [self.predict_color(image)]

    def rectify(self, image, pts):
        corners = np.float32(pts)

        CARD_DIMENSIONS = (100, 140)
        target_corners = np.float32([
            (0, 0),
            (CARD_DIMENSIONS[0], 0),
            (CARD_DIMENSIONS[0], CARD_DIMENSIONS[1]),
            (0, CARD_DIMENSIONS[1])
        ])

        m, mask = cv2.findHomography(corners, target_corners)
        skew = cv2.warpPerspective(image, m, CARD_DIMENSIONS)
        return skew
