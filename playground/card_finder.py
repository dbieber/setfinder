import cv
import cv2
import math
import numpy as np
from sets import Set
import copy
import sys

cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)

card_file = '../data/input_images/'+str(sys.argv[1])+'.jpg'
img1 = cv2.imread(card_file)


def make_segments(rho, theta, diag):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + diag*(-b))
    y1 = int(y0 + diag*(a))
    x2 = int(x0 - diag*(-b))
    y2 = int(y0 - diag*(a))
    return (x1, y1, x2, y2)

def get_segments(lines, edges, sobelx, sobely, probabilistic):

    border = 2
    inc = 30

    segments = []

    width, height = edges.shape
    diag = np.sqrt(width**2 + height**2)

    if not probabilistic:
        if lines is not None and lines[0] is not None:
            for rho, theta in lines[0]:
                segments.append(make_segments(rho, theta, diag))
            short_seg = []
            for x1,y1,x2,y2 in segments:
                in_line = False
                dx = (x2-x1)/float(diag)
                dy = (y2-y1)/float(diag)
                if dx == 0:
                    seqx = np.repeat(x2, int(diag)/inc)
                else:
                    seqx = np.arange(x1, x2, dx*inc)
                    
                if dy == 0:
                    seqy = np.repeat(y2, int(diag)/inc)
                else:
                    seqy = np.arange(y1,y2,dy*inc)

                for (x,y) in zip(seqx, seqy):
                    x_i = round(x)
                    y_i = round(y)
                    flag = 0
                    if border <= y_i < width-border and border <= x_i < height-border:
                        for i in range(-border, border):
                            for j in range(-border, border):
                                if edges[y_i+j][x_i+i] > 0: 
                                    #edges2 = copy.deepcopy(edges)
                                    #cv2.circle(edges2, (int(x_i), int(y_i)), 5, (255,255,255), -1)
                                    #print x_i, y_i
                                    #cv2.imshow("w1", edges2)
                                    #if (cv.waitkey(0) == ord('q')):
                                    #    exit(0)

                                    flag = flag + 1
                        if flag > 0 and not in_line:
                            start_x = x_i
                            start_y = y_i
                            in_line = True
                        if flag == 0 and in_line:
                            small_x = start_x
                            large_x = x_i
                            if x_i < small_x:
                                small_x = x_i
                                large_x = start_x
                            small_y = start_y
                            large_y = y_i
                            if y_i < small_y:
                                small_y = y_i
                                large_y = start_y
                            short_seg.append((int(small_x-inc*dx*2), 
                                              int(small_y-inc*dy*2), 
                                              int(large_x+inc*dx*2), 
                                              int(large_y+inc*dy*2)))
                            in_line = False
                if in_line:
                    small_x = start_x
                    large_x = x_i
                    if x_i < small_x:
                        small_x = x_i
                        large_x = start_x
                    small_y = start_y
                    large_y = y_i
                    if y_i < small_y:
                        small_y = y_i
                        large_y = start_y
                    short_seg.append((int(small_x-inc*dx*2), 
                                      int(small_y-inc*dy*2), 
                                      int(large_x+inc*dx*2), 
                                      int(large_y+inc*dy*2)))


            segments = short_seg
    else:
        line_using = []
        for i in range(width):
            line_using.append([])
            for j in range(height):
                line_using[i].append([])

        if lines is not None and lines[0] is not None:
            for x1,y1,x2,y2 in lines[0]:
                segments.append((x1,y1,x2,y2))
            short_seg = []
            keep_seg = []
            index = 0
            for x1,y1,x2,y2 in segments:
                keep_seg.append(True)

                dx = float(x2-x1)
                dy = float(y2-y1)
                length = math.sqrt(dx**2+dy**2)
                dy = dy/length
                dx = dx/length

                def extend_line(x, y, dx, dy):
                    close = True
                    possible_x, possible_y = x, y
                    possible_seg = -1
                    start_x, start_y = x, y
                    while close:
                        x = x + dx*inc
                        y = y + dy*inc
                        x_i = round(x)
                        y_i = round(y)
                        flag = 0
                        if border <= y_i < width-border and border <= x_i < height-border:
                            for i in range(-border, border):
                                for j in range(-border, border):
                                    grad = np.array([sobelx[y_i+j][x_i+i], sobely[y_i+j][x_i+i]])
                                    if (edges[y_i+j][x_i+i] > 0 and
                                        abs(np.dot(grad, np.array([dx, dy])))/np.linalg.norm(grad) < .7): 
                                        already_added = False
                                        for l in line_using[int(y_i+j)][int(x_i+i)]:
                                            if l != index and keep_seg[l]:
                                                x_s1, y_s1, x_s2, y_s2 = short_seg[l]
                                                v = np.array([x_s2-x_s1, y_s2-y_s1])
                                                u = np.array([dx, dy])
                                                v = v / float(np.linalg.norm(v))
                                                u = u / float(np.linalg.norm(u))
                                                dot = np.dot(u, v)
                                                if abs(dot) > .95:
                                                    line_using[int(y_i+j)][int(x_i+i)].append(index)
                                                    possible_seg = l
                                                    if dot > 0:
                                                        possible_x, possible_y = x_s2, y_s2
                                                    else:
                                                        possible_x, possible_y = x_s1, y_s1
                                            else:
                                                already_added = True
                                        if not already_added:
                                            line_using[int(y_i+j)][int(x_i+i)].append(index)
                                            
                                        flag = flag + 1

                            if flag == 0:
                                frac = .5
                                num_loop = int(np.log2(inc))
                                x = x - dx*inc*frac
                                y = y - dy*inc*frac
                                for h in range(num_loop):
                                    x_i = round(x)
                                    y_i = round(y)
                                    direction = -1
                                    for i in range(-border, border):
                                        for j in range(-border, border):

                                            grad = np.array([sobelx[y_i+j][x_i+i], sobely[y_i+j][x_i+i]])
                                            if (edges[y_i+j][x_i+i] > 0 and
                                                abs(np.dot(grad, np.array([dx, dy])))/np.linalg.norm(grad) < .7): 
                                                direction = 1
                                                break
                                        if direction == 1:
                                            break
                                    x = x + dx*inc*frac*direction
                                    y = y + dy*inc*frac*direction
                                    frac = frac/2.0

                                close = False
                        else:
                            close = False
                    dist1 = (possible_x - start_x)**2+(possible_y - start_y)**2
                    dist2 = (x - start_x)**2+(y - start_y)**2
                    if possible_seg == -1:
                        return x, y

                    if dist1 > dist2:
                        return possible_x, possible_y
                    else:
                        return x, y

                if (dx > 0 and x1 > x2) or (dx < 0 and x1 < x2):
                    x_end, y_end = extend_line(x1, y1, dx, dy)
                    x_start, y_start = extend_line(x2, y2, -dx, -dy)
                else:
                    x_start, y_start = extend_line(x1, y1, dx, dy)
                    x_end, y_end = extend_line(x2, y2, -dx, -dy)
                start_change = 0
                end_change = 0
                if dx > 0:
                    if x1 > x2:
                        end_change = 1
                        start_change = -1
                    else:
                        end_change = -1
                        start_change = 1
                else:
                    if x1 < x2:
                        end_change = 1
                        start_change = -1
                    else:
                        end_change = -1
                        start_change = 1


                scale = math.sqrt((x_end-x_start)**2+(y_end-y_start)**2)/20.0
                x_start = x_start + start_change*dx*scale
                y_start = y_start + start_change*dy*scale
                x_end = x_end + end_change*dx*scale
                y_end = y_end + end_change*dy*scale

                short_seg.append((int(x_start), 
                                  int(y_start), 
                                  int(x_end), 
                                  int(y_end)))

                index = index + 1

            segments = []
            for i in range(len(short_seg)):
                if keep_seg[i]:
                    segments.append(short_seg[i])
    return segments

def get_intersections(segments):
    intersections = []
    for i in range(len(segments)):
        intersections.append([])

    for i in range(len(segments)):
        these_intersections = []
        x1_i, y1_i, x2_i, y2_i = segments[i]
        p = np.array([x1_i, y1_i])
        r = np.array([x2_i - x1_i, y2_i - y1_i])
        for j in range(i, len(segments)):
            x1_j, y1_j, x2_j, y2_j = segments[j]
            q = np.array([x1_j, y1_j])
            s = np.array([x2_j - x1_j, y2_j - y1_j])
            denom = float(np.cross(r, s))
            if abs(denom) > .001:
                v = q - p
                t = np.cross(v,s)/denom
                u = np.cross(v,r)/denom
                if 0 <= t <= 1 and 0 <= u <= 1:
                    x, y = p+t*r
                    x2, y2 = q+u*s
                    if abs(np.dot(r,s)/np.linalg.norm(r)/np.linalg.norm(s)) < .8:
                    #if 0 <= x <= width and 0 <= y <= height:
                        these_intersections.append([x,y,i,j])
                        intersections[j].append([x,y,i,j])
        intersections[i].extend(these_intersections)

    for i in range(len(intersections)):
        intersections[i] = sorted(intersections[i], key = lambda x: x[1])
        intersections[i] = sorted(intersections[i], key = lambda x: x[0])

    return intersections

def get_quads(intersections):
    quads = Set([])

    def find_intersection_index(intersection, i):
        for k in range(len(intersection)):
            if (intersection[k][2] == i or
                intersection[k][3] == i):
                return k
        return -1

    def get_other_index(intersection, i):
        x,y,a,b = intersection
        other = a
        if other == i:
            other = b
        return other
         

    def make_quad(intersections):
        q = []
        for inter in intersections:
            q.append((int(inter[0]),int(inter[1])))
        return (q[0], q[1], q[2], q[3])

    search_range = 3

    for i in range(len(intersections)):
        for j in range(len(intersections[i])):
            other = get_other_index(intersections[i][j], i)
            
            index = find_intersection_index(intersections[other], i)

            for m in range(search_range):
                m_sum = j+m+1
                if m_sum < len(intersections[i]):
                    intersection_south = intersections[i][m_sum]
                    #print intersections[i][j], intersection_south
                    other_index_south = get_other_index(intersection_south, i)
                    #print i,other, other_index_south
                    for n in range(search_range):
                        n_sum = index+n+1
                        if n_sum < len(intersections[other]):
                            intersection_east = intersections[other][n_sum]
                            other_index_east = get_other_index(intersection_east, other)

                            #print intersections[other_index_east], other_index_south
                            found_index = find_intersection_index(intersections[other_index_east],other_index_south)
                            #print other,i,j,other_index_south, other_index_east
                            if found_index != -1:
                                q = make_quad([intersections[i][j],
                                                        intersection_east,
                                                        intersections[other_index_east][found_index],
                                                        intersection_south])
                                length = []
                                flag = False
                                for o in range(len(q)):
                                    p = (o + 1) % len(q)
                                    x1, y1 = q[o]
                                    x2, y2 = q[p]
                                    l = (x1-x2)**2+(y1-y2)**2
                                    length.append(l)
                                    if l <= 400:
                                        flag = True
                                if not flag:
                                    for o in range(len(length)):
                                        p = (o + 1) % len(q)
                                        l1 = length[o]
                                        l2 = float(length[p])
                                        if not (.25 < l1/l2 < 4):
                                            flag = True
                                
                                if not flag:
                                    quads.add(q)

    return quads

def reduce_quads(quads):
    def sort_quad(q):
        avg = np.average(np.array(q), axis=0)
        angles = []
        i = 0
        new_quad = np.array(q)
        for x,y in q:
            angles.append([math.atan2(y-avg[1], x-avg[0]), i])
            i = i + 1

        angles = sorted(angles, key = lambda x: x[0])
        i = 0
        for a in angles:
            new_quad[i] = q[angles[1]]
            i = i + 1
        return new_quad

    new_quads = []
    for q in quads:
        new_quads.append(sort_quad(q))
    return new_quads


def find_cards(img1):

    # Do initial openCV processing

    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)

    edges = cv2.Canny(gray, 404, 156, apertureSize=3)
    lines = cv2.HoughLinesP(edges,1,np.pi/360,20, minLineLength=10, maxLineGap=2)
    print len(lines[0])

    intersections = []


    segments = get_segments(lines, edges, sobelx, sobely, True)
    intersections = get_intersections(segments)
    quads = get_quads(intersections)

    return reduce_quads(quads)

card_file = '../data/input_images/'+str(sys.argv[1])+'.jpg'
img1 = cv2.imread(card_file)
quads = find_cards(img1)


for q in quads:
    arr = [np.array(q,'int32')]
    cv2.fillPoly(img1,arr,(0,0,100))

cv2.imshow("w1", img1)
cv.WaitKey(0)
