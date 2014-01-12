import cv
import cv2
import math
import numpy as np
import random
#import scipy.spatial
#from sklearn.cluster import KMeans
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC

SIFT_extractor = cv2.DescriptorExtractor_create("SIFT")

class CardIdentifier():
    def __init__(self):
        self.n_clusters = 40
        self.patch_size = 9
        self.patches_per_image = 100

        # The classifiers used for identification
        self.patch_clusterer = None
        self.shape_clf = None
        self.number_clf = None
        self.shading_clf = None

    # Black and white patches
    def patches_from(self, image):
        image = cv2.cvtColor(image, cv.CV_RGB2GRAY)
        y,x = image.shape
        image = image[20:120,20:80]
        y,x = image.shape
        patches = []



        ps = self.patch_size
        # for i in xrange(self.patches_per_image):
        #     ry = random.randint(0, y - ps - 1)
        #     rx = random.randint(0, x - ps - 1)
        for ry in np.arange(0,1,1.0/20):
            for rx in np.arange(0,1,1.0/20):
                sy = ry * (y-ps)
                sx = rx * (x-ps)



                patch = image[sy:sy+ps,sx:sx+ps]
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

        print "Clustering %d patches..." % len(patches)
        self.patch_clusterer = KMeans(init='k-means++', n_clusters=self.n_clusters)
        self.patch_clusterer.fit(patches)
        print "self.patch_clusterer.get_params()"
        print self.patch_clusterer.get_params()

        bags = []
        print "Creating bags..."
        for image, card in zip(X, Y):
            bags.append(self.bag_from(image))

        NUMBER = 0
        SHADING = 1
        COLOR = 2
        SHAPE = 3
        print "Training on shapes, numbers, and shadings..."
        self.number_clf = SVC()
        self.number_clf.fit(bags, Y[:,NUMBER])
        self.shading_clf = SVC()
        self.shading_clf.fit(bags, Y[:,SHADING])
        self.shape_clf = SVC()
        self.shape_clf.fit(bags, Y[:,SHAPE])

    def predict_number(self, image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 404/3, 156/3, apertureSize=3)
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
            return "one"
        if count == 2:
            return "two"
        if count == 3:
            return "three"
        return "fail"

    def predict_shading(self, image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #edges = cv2.Canny(gray, 404/4, 156/4, apertureSize=3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.blur(image, (3,3))

        # Assumes rectified image
        height, width, depth = image.shape
        x = width / 2

        # weird HSV
        shading_names = ["solid", "empty", "striped"]
        shading_counts = np.array([0, 0, 0])
        for x in range(width/2-10, width/2+10, 4):

            inside = False
            ever_switch = False
            beginning = True

            beg_count = 0
            beg_color = np.array([0, 0, 0])

            card_count = 0
            card_color = np.array([0, 0, 0])

            avg_color = np.array([0, 0, 0])
            countdown = 5
            switches = 0

            inside_long = 0

            first_sat = int(image[5, x, 1])
            second_sat = 0
            second_hue = 0
            countdown = 0
            threshold1 = False

            for y in range(5, height-5):
                sat = int(image[y, x, 1])
                if not threshold1 and abs(sat - first_sat) > 5:
                    threshold1 = True
                    countdown = 10

                if sat > second_sat:
                    second_hue = int(image[y,x,0])
                    second_sat = sat

                if threshold1:
                    print first_sat, second_sat, sat
                    countdown -= 1
                    if countdown == 0:
                        if abs(sat - second_sat) < 20:
                            shading_counts[0] += 1
                        elif abs(int(image[y,x,0]) - second_hue) < 25:
                            shading_counts[2] += 1
                        else:
                            shading_counts[1] += 1
                        break
                        

                """
                if edges[y, x] > 0:
                    switches += 1
                    if beginning and beg_count != 0:
                        beg_color /= beg_count
                    beginning = False
                    inside = not inside
                    ever_switch = True

                if switches > 3 or (switches == 2 and inside_long > 10):
                    break

                if inside:
                    inside_long += 1

                color = np.array(image[y,x])

                if beginning:
                    beg_count += 1
                    beg_color += image[y,x]
                else:
                    if countdown == 4:
                        #card_color = color
                        mixed_color = beg_color.copy()
                        mixed_color[0] = card_color[0]
                    elif countdown <= 0:

                        dists = []
                        colors = [card_color, beg_color, mixed_color]
                        for c in colors:
                            dists.append(np.linalg.norm(c - color, ord=1))
                            
                        if np.min(dists) < 40:
                            shading_counts[np.argmin(dists)] += 1
                    #print color

                    countdown -= 1
                """

        #print 'card_color', card_color, 'beg_color', beg_color
        #min_count = min(shading_counts)
        #if min_count > 0 and max(shading_counts) / min_count  < 2:
        #    return shading_names[2], shading_counts

        return shading_names[np.argmax(shading_counts)], shading_counts

    def predict_shape(self, image):
        bag = self.bag_from(image)
        return self.shape_clf.predict(bag)

    def predict_color(self, image, flag = False):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 404/4, 156/4, apertureSize=3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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

        return color_names[np.argmax(color_counts)], color_counts

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
