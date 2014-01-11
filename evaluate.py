import cv
import cv2
import json
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
            print "Click the corners of the %s" % card

            while True:
                if len(current_points) == 4:
                    # Finished with that image!
                    print current_points

                    card_outputs.append("%s %s" % (card, ", ".join(str(x) for x in current_points)))

                    current_points = []
                    break
                if cv.WaitKey(10) == 27:
                    break

        line = "%s: %s" % (filename, ";".join(card_outputs))
        output.write("%s\n" % line)

    output.close()

def points_from_card(card):
    tokens = card.split(" ")
    points = eval("[%s]" % ''.join(tokens[4:]))
    return points

def main():
    c = CardIdentifier()

    cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
    for i, (filename, cards) in enumerate(groundtruth("data/ground_truth2.txt")):
        for card in cards:
            print card
            pts = points_from_card(card)
            image = cv2.imread(filename)
            image = c.rectify(image, pts)
            cv2.destroyWindow("w1")
            cv2.imshow("w1", image)
            raw_input()

if __name__ == "__main__":
    main()
    # generate_groundtruth()
