import cv
import cv2
import json
import numpy as np
from cardidentifier import CardIdentifier

def groundtruth(filename="data/ground_truth.txt"):
    groundtruthfile = open(filename)

    for line in groundtruthfile:
        tokens = [x.strip() for x in line.strip().split(":")]
        filename = tokens[0]
        cards = None if len(tokens) == 1 else [x.strip() for x in tokens[1].split(";")]
        yield (filename, cards)

    groundtruthfile.close()

def generate_groundtruth(filename="data/_ground_truth.txt"):
    output = open(filename, 'w')

    current_image = None
    current_points = []
    def on_mouse(event, x, y, flag, param):
        if event == 1:  # MouseDown
            current_points.append((x, y))
            cv.Circle(current_image, (x,y), 15, (0,0,255), thickness=3)
            cv.ShowImage("window", current_image)

    cv.NamedWindow("window")
    cv.SetMouseCallback("window", on_mouse)

    for i, (filename, cards) in enumerate(groundtruth()):
        current_image = cv.LoadImage(filename)
        cv.ShowImage("window", current_image)

        card_outputs = []

        for i, card in enumerate(cards):
            attrs = attrs_from_card(card)
            attrs_str = ' '.join(attrs)
            print "Click the corners of the %s" % attrs_str

            already_points = points_from_card(card)
            if already_points:
                current_points = already_points
                print "Already at %s" % str(current_points)

            while True:
                if len(current_points) == 4:
                    # Finished with that image!
                    points_str = ", ".join(str(x) for x in current_points)
                    print points_str
                    card_outputs.append("%s %s" % (attrs_str, points_str))

                    current_points = []
                    break
                if cv.WaitKey(10) == 27:
                    card_outputs.append("%s" % attrs_str)
                    current_points = []
                    break

        line = "%s: %s" % (filename, "; ".join(card_outputs))
        output.write("%s\n" % line)

    output.close()

def attrs_from_card(card):
    tokens = card.split(" ")
    return tokens[:4]

def points_from_card(card):
    tokens = card.split(" ")
    if len(tokens) > 4:
        points = eval("[%s]" % ''.join(tokens[4:]))
        return points

def main():
    c = CardIdentifier()

    """
    # Train
    X = []
    Y = []
    for i, (filename, cards) in enumerate(groundtruth("data/ground_truth.txt")):
        image = cv2.imread(filename)
        for j, card in enumerate(cards):
            if j == 10:
                break

            pts = points_from_card(card)
            if not pts:
                continue

            cardimage = c.rectify(image, pts)

            X.append(cardimage)
            Y.append(attrs_from_card(card))

    Y = np.array(Y)
    c.fit(X, Y)
    """

    cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
    bad_color_count = 0
    total = 0
    for i, (filename, cards) in enumerate(groundtruth("data/training.txt")):
        for card in cards:
            pts = points_from_card(card)
            image = cv2.imread(filename)
            if not pts:
                continue

            image = c.rectify(image, pts)

            cv2.destroyWindow("w1")
            cv2.imshow("w1", image)
            #print card
            #print c.predict_number(image),
            color = c.predict_color(image)
            #print color
            #print c.predict_shading(image)[0],
            #print c.predict_shape(image)[0]
            if card.find(color[0]) == -1:
                print card
                print color
                color = c.predict_color(image, True)
                cv2.waitKey(0)
                bad_color_count = bad_color_count + 1
            total = total + 1
    print bad_color_count, total

if __name__ == "__main__":
    main()
    # generate_groundtruth()
