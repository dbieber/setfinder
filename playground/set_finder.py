import cv
import cv2
import numpy as np
import card_finder 

def main():

    print card_finder
    print 'hit any key to take a picture'
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        frame = cv2.resize(frame, (640,480))
        quads = []
        for i in range(0, 3):
            quads.extend(card_finder.find_cards_with_parameter_setting(frame, i))

        quads = card_finder.reduce_quads(quads)
        print len(quads)

        image = mark_quads(frame, quads)
        cv2.imshow('win', image)

        rval, frame = vc.read()
        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            break



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
    
main()
