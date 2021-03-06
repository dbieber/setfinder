#import cv
import cv2
import math
import numpy as np
from sets import Set
import copy
import sys
import setfinder
import networkx as nx

"""
cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)

card_file = '../data/input_images/'+str(sys.argv[1])+'.jpg'
img1 = cv2.imread(card_file)
"""


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
                                    if (edges[y_i+j][x_i+i] > 0):# and
                                        #abs(np.dot(grad, np.array([dx, dy])))/np.linalg.norm(grad) < .7):
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
                                            if (edges[y_i+j][x_i+i] > 0):# and
                                                #abs(np.dot(grad, np.array([dx, dy])))/np.linalg.norm(grad) < .7):
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
                    if abs(np.dot(r,s)/np.linalg.norm(r)/np.linalg.norm(s)) < .4:
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
            new_quad[i] = q[a[1]]
            i = i + 1
        return new_quad

    new_quads = []
    for q in quads:
        new_quads.append(sort_quad(q))

    keep_quads = []
    for q1 in new_quads:
        keep_quads.append(True)

    for i in range(len(new_quads)):
        q1 = new_quads[i]
        if keep_quads[i]:
            for j in range(i+1, len(new_quads)):
                if keep_quads[j]:
                    q2 = new_quads[j]
                    keep_quads[j] = False
                    for k in range(4):
                        p1 = q1[k]
                        p2 = q2[k]
                        if (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 > 16:
                            keep_quads[j] = True
    final_quads = []
    for q,k in zip(new_quads, keep_quads):
        if k:
            final_quads.append(q)
    return final_quads

def rotate_quads(quads):
    index = 0
    for q in quads:
        if np.linalg.norm(q[0] - q[1]) > np.linalg.norm(q[1] - q[2]):
            q2 = []
            for p in q[1:]:
                q2.append(p)
            q2.append(q[0])
            quads[index] = np.array(q2)
        index += 1
    return quads

def find_cards(img1, rho, theta, threshold, minLineLength, maxLineGap):

    # Do initial openCV processing

    #img1 = cv2.fastNlMeansDenoisingColored(img1,None,10,10,7,21)
    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #gray = clahe.apply(gray)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)

    #gray = cv2.blur(gray, (3, 3))
    edges = cv2.Canny(gray, 404/2, 156/2, apertureSize=3)
    cv2.imshow('edges', edges)
    lines = cv2.HoughLinesP(edges,rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is None:
        return []

    intersections = []


    segments = get_segments(lines, edges, sobelx, sobely, True)
    intersections = get_intersections(segments)
    quads = get_quads(intersections)
    quads = reduce_quads(quads)

    return quads

parameters = [[1, 90, 50, 10, 2],
              [1, 180, 20, 10, 2],
              [.5, 180, 30, 10, 2],
              [.75, 720, 40, 10, 2],
              [1, 720, 50, 10, 2],
              [2, 90, 40, 10, 2],
              [1, 360, 50, 10, 2]]


def find_cards_with_parameter_setting(img1, i):

    p = parameters[i]
    return find_cards(img1, p[0], np.pi/p[1], p[2], p[3], p[4])

def new_card_finder(image):

    quads = []

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #gray = clahe.apply(gray)

    #gray = cv2.blur(gray, (3, 3))
    edges = cv2.Canny(gray, 404/2, 156/2, apertureSize=3)
    #cv2.imshow('edges', edges)

    height, width = edges.shape
    def convert_to_flatened_index(i, j):
        return i * width + j

    def convert_from_flatened_index(index):
        return (index / width, index % width)

    def make_graph(edges):
        G = nx.Graph()
        index = 0
        for i in range(height):
            for j in range(width):
                if edges[i, j] > 1:
                    flag = 0
                    bounds = 1
                    while flag < 2 and bounds < 3:
                        for k in range(-bounds, bounds+1):
                            for l in range(-bounds, bounds+1):
                                if abs(k) == bounds or abs(l) == bounds:
                                    if 0 <= i + k < height and 0 <= j + l < width and edges[i+k, j+l] > 0:
                                        index2 = convert_to_flatened_index(i+k, j+l)
                                        flag += 1
                                        G.add_edge(index, index2)
                        bounds = bounds + 1

                index = index + 1
        return G

    connected_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    G = make_graph(edges)
    comps = nx.connected_component_subgraphs(G)


    # TODO Remove if first and last node aren't adjacent
    for g in comps:
        if g.number_of_nodes() > 200:
            """
            for index in g.nodes_iter():
                (i, j) = convert_from_flatened_index(index)
                connected_edges[i, j] = [100, 50, 0]
            """


            nodes = g.nodes()
            use_node = None
            for try_index in range(0, 20, 5):
                if try_index < len(nodes):
                    node1 = g.nodes()[try_index]
                    neighbors = g.neighbors(node1)
                    if len(neighbors) == 2:
                        use_node = neighbors[0]
                        g.remove_edge(node1, use_node)
                        break

            #tree = nx.dfs_tree(g, g.nodes()[0])
            #nodes = nx.topological_sort(tree)
            if use_node is None:
                continue
            try:
                nodes = nx.shortest_path(g, node1, use_node)
            except nx.NetworkXNoPath:
                continue


            search_length = 5
            length = len(nodes)


            def find_nearby_dot(i):
                y, x = convert_from_flatened_index(nodes[i])
                index1 = (i + search_length) % length
                index2 = (i - search_length + length) % length
                y1, x1 = convert_from_flatened_index(nodes[index1])
                y2, x2 = convert_from_flatened_index(nodes[index2])
                v1 = np.array([x - x1, y - y1])
                v2 = np.array([x2 - x, y2 - y])
                return abs(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

            corners = []
            i = 0
            dot_threshold = .9
            while i < length:
                mindot = find_nearby_dot(i)
                if 0 < mindot < dot_threshold:
                    minindex = i
                    vote = 0
                    for j in range(-search_length, search_length):
                        index = (i + j + length) % length
                        dot = find_nearby_dot(index)
                        if 0 < dot < dot_threshold:
                            vote += 1
                        if dot < mindot:
                            mindot = dot
                            minindex = index
                    if vote > search_length/4:
                        if i < 20:
                            length -= (20-i)
                        y, x = convert_from_flatened_index(nodes[minindex])
                        corners.append((x, y))
                        i += 20
                i += 1
            if len(corners) == 4:
                quads.append(corners)
                """
                for c in corners:
                    cv2.circle(connected_edges, c, 3, (250, 50, 0), thickness=-1)
                """

        """
        cv2.imshow('connected_edges', connected_edges)
        cv2.waitKey(0)
        """

    def keep_rectangles(quads):
        final_quads = quads
        # for q in quads:
        #     for i in range(len(q)):
        #         q1 = np.array(q[i])
        #         q2 = np.array(q[(i+1)%len(q)])
        #         q3 = np.array(q[(i+2)%len(q)])
        #         v1 = q2-q1
        #         v2 = q2-q3
        #         if abs(np.dot(v1, v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)) < .5:
        #             final_quads.append(q)

        def inside_quad(q, p):
            neg = True
            pos = True
            p = np.array(p)
            for i in range(len(q)):
                q1 = np.array(q[i])
                q2 = np.array(q[(i+1)%len(q)])

                v1 = q2-q1
                v2 = p-q1
                cross = np.cross(v1, v2)
                if cross < 0:
                    pos = False
                if cross > 0:
                    neg = False

            return neg or pos

        real_final_quads = []
        for i, q in enumerate(final_quads):
            in_any_quad = False
            for j, q2 in enumerate(final_quads):
                if i == j:
                    continue

                all_inside_q2 = True
                for p in q:
                    if not inside_quad(q2, p):
                        all_inside_q2 = False
                        break
                if all_inside_q2:
                    in_any_quad = True
                    break
            if not in_any_quad:
                real_final_quads.append(q)

        return real_final_quads

    quads = reduce_quads(quads)
    return keep_rectangles(quads)

def test_card_finder():
    def run_test(image):
        image = cv2.resize(image, (640,480))
        quads = []
        quads.extend(new_card_finder(image))
        #for i in range(0, 1):
        #    quads.extend(find_cards_with_parameter_setting(image, i))

        quads = reduce_quads(quads)
        quads = rotate_quads(quads)
        cards = []
        """
        for q in quads:
            card = Card(rectify(image, q))
            if not card.fail():
                cards.append(card)

        print set(' '.join(c.labels()) for c in cards)
        """

        image = setfinder.mark_quads(image, quads)
        cv2.imshow('win', image)
        cv2.waitKey(0)

    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
        run_test(image)
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
                run_test(frame)

            if key == 27: # exit on ESC
                break


if __name__ == '__main__':
    test_card_finder()

"""
card_file = '../data/input_images/'+str(sys.argv[1])+'.jpg'

rho = 1
theta = np.pi/360
threshold = 30
minLineLength = 10
maxLineGap = 2
img1 = cv2.imread(card_file)
quads = []
for i in range(0, len(parameters)):
    quads.extend(find_cards_with_parameter_setting(img1, i))

quads = reduce_quads(quads)

for q in quads:
    arr = [np.array(q,'int32')]
    cv2.fillPoly(img1,arr,(0,0,100))

cv2.imwrite('../data/output_images/'+str(sys.argv[1])+'.jpg', img1)
"""

"""
for rho in [.125, .25, .5, .75, 1, 2, 3, 4]:
    for theta in [np.pi/720, np.pi/360, np.pi/180, np.pi/90, np.pi/45]:
        for threshold in [10, 20, 30, 40, 50]:

            img1 = cv2.imread(card_file)



            quads = find_cards(img1, rho, theta, threshold, minLineLength, maxLineGap)


            for q in quads:
                arr = [np.array(q,'int32')]
                cv2.fillPoly(img1,arr,(0,0,100))

            cv2.imwrite('../data/output_images/'+str(sys.argv[1])+'_'+str(rho)+'_'+str(theta)+
                                                                  '_'+str(threshold)+'_'+str(minLineLength)+
                                                                  '_'+str(maxLineGap)+'.jpg', img1)
"""
