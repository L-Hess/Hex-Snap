import pkg_resources
import numpy as np
import logging
import cv2
import os
import pandas as pd
from moviepy.editor import VideoFileClip
import networkx as nx


# Euclidean distance
def distance(x1, y1, x2, y2):
    """Euclidean distance"""
    r = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return r


def make_directory(pathname, path_str):
    path = pkg_resources.resource_filename(pathname, path_str)
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)


def ls_dist(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    px = x2 - x1
    py = y2 - y1

    norm = px * px + py * py

    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    sqr_dist = dx * dx + dy * dy

    return sqr_dist, (x, y)


# For time alignment of both videos on basis of the LED light flickering
class TimeCorrect:
    def __init__(self, pathname, sources):

        # Loading in the position log files as created by the tracker
        self.data_path_1 = pkg_resources.resource_filename(pathname, '/data/interim/Position_log_files/{}/pos_log_file_0.csv'.format(sources[0][len(sources[0])-29:len(sources[0])-10]))
        self.data_path_2 = pkg_resources.resource_filename(pathname, '/data/interim/Position_log_files/{}/pos_log_file_1.csv'.format(sources[0][len(sources[0])-29:len(sources[0])-10]))

        self.dat_0_load = np.genfromtxt(self.data_path_1, delimiter=',', skip_header=False)
        self.dat_1_load = np.genfromtxt(self.data_path_2, delimiter=',', skip_header=False)

        self.dat_0 = np.random.rand(len(self.dat_0_load), 5)
        self.dat_1 = np.random.rand(len(self.dat_1_load), 5)

        self.dat_0[:, :-1] = self.dat_0_load
        self.dat_1[:, :-1] = self.dat_1_load

        self.dat_0f = None
        self.dat_1f = None

    # Time alignment of the position log files
    def correction(self):
        """"Time alignment based on on- and offsets of LED light"""

        # Takes the LED-states of both log files, where 1 is on and 0 is off
        led_0 = self.dat_0[:, 3]
        led_1 = self.dat_1[:, 3]

        # Calculates where in the log-files the LED-state goes from off to on (0 to 1)
        led_0_peaks = np.diff(led_0)
        led_1_peaks = np.diff(led_1)
        i_0 = np.sort(np.append(np.where(led_0_peaks == 1)[0], np.where(led_0_peaks == -1)[0]))
        i_1 = np.sort(np.append(np.where(led_1_peaks == 1)[0], np.where(led_1_peaks == -1)[0]))
        peak_diff_0 = np.diff(i_0)
        peak_diff_1 = np.diff(i_1)

        # Calculates where in the log-files the LED-state goes from off to on (0 to 1)
        led_0_peaks = np.diff(led_0)
        led_1_peaks = np.diff(led_1)
        i_0 = np.sort(np.append(np.where(led_0_peaks == 1)[0], np.where(led_0_peaks == -1)[0]))
        i_1 = np.sort(np.append(np.where(led_1_peaks == 1)[0], np.where(led_1_peaks == -1)[0]))
        peak_diff_0 = np.diff(i_0)
        peak_diff_1 = np.diff(i_1)

        min_it = np.min([len(peak_diff_0), len(peak_diff_1)])
        dat_0f = np.full((1, 5), np.nan)
        dat_1f = np.full((1, 5), np.nan)

        time = (peak_diff_0[0] - 1) * 0.5
        for i in range(min_it):
            self.dat_0[:, 4] = time
            self.dat_1[:, 4] = time
            if peak_diff_0[i] > peak_diff_1[i]:
                dat_0f = np.concatenate((dat_0f, self.dat_0[i_0[i]:i_0[i + 1]]), axis=0)
                dat_1f = np.concatenate((dat_1f, self.dat_1[i_1[i]:i_1[i + 1]]), axis=0)
                if not i == min_it - 1:
                    diff = peak_diff_0[i] - peak_diff_1[i]
                    add = np.full((diff, 5), np.nan)
                    add[:, 4] = time
                    add[:, 3] = dat_0f[len(dat_1f) - 1, 3]
                    dat_1f = np.concatenate((dat_1f, add), axis=0)

            elif peak_diff_1[i] > peak_diff_0[i]:
                dat_1f = np.concatenate((dat_1f, self.dat_1[i_1[i]:i_1[i + 1]]), axis=0)
                dat_0f = np.concatenate((dat_0f, self.dat_0[i_0[i]:i_0[i + 1]]), axis=0)
                if not i == min_it - 1:
                    diff = peak_diff_1[i] - peak_diff_0[i]
                    add = np.full((diff, 5), np.nan)
                    add[:, 4] = time
                    add[:, 3] = dat_0f[len(dat_0f) - 1, 3]
                    dat_0f = np.concatenate((dat_0f, add), axis=0)

            elif peak_diff_0[i] == peak_diff_1[i]:
                dat_0f = np.concatenate((dat_0f, self.dat_0[i_0[i]:i_0[i + 1]]), axis=0)
                dat_1f = np.concatenate((dat_1f, self.dat_1[i_1[i]:i_1[i + 1]]), axis=0)

            time += 0.5

        # Create pandas dataframe for interpolation
        df0 = pd.DataFrame()
        df0['x'], df0['y'], df0['frame_n'], df0['LED_state'], df0['time'] = dat_0f[:, 0], dat_0f[:, 1], dat_0f[:,
                                                                                                        2], dat_0f[:,
                                                                                                            3], dat_0f[
                                                                                                                :, 4]
        df1 = pd.DataFrame()
        df1['x'], df1['y'], df1['frame_n'], df1['LED_state'], df1['time'] = dat_1f[:, 0], dat_1f[:, 1], dat_1f[:,
                                                                                                        2], dat_1f[:,
                                                                                                            3], dat_1f[
                                                                                                                :, 4]
        # Interpolate missing x and y values using linear interpolation
        df0['x'] = df0['x'].interpolate(method='linear', limit=3)
        df0['y'] = df0['y'].interpolate(method='linear', limit=3)
        df1['x'] = df1['x'].interpolate(method='linear', limit=3)
        df1['y'] = df1['y'].interpolate(method='linear', limit=3)

        # Convert back to numpy arrays
        dat_0f = df0.values
        dat_1f = df1.values

        return dat_0f, dat_1f


class Linearization:
    def __init__(self, pathname, dat_0, dat_1, sources):

        # Load in the time aligned log files and corrected node positions
        # Ghost nodes are used for nodes that are off of the video
        # Dwell nodes are used for nodes that are off of the video
        self.dat_0 = dat_0
        self.dat_1 = dat_1

        make_directory(pathname, "/data/interim/linearized_position_log_files/{}".format(sources[0][len(sources[0])-29:len(sources[0])-10]))

        self.nodes_top_path = pkg_resources.resource_filename(pathname, '/src/Resources/aligned/corr_node_pos_top.csv')
        self.nodes_bot_path = pkg_resources.resource_filename(pathname, '/src/Resources/aligned/corr_node_pos_bot.csv')

        self.nodes_top = np.genfromtxt(self.nodes_top_path, delimiter=',', skip_header=True)
        self.nodes_bot = np.genfromtxt(self.nodes_bot_path, delimiter=',', skip_header=True)

        self.pathname = pathname
        self.sources = sources

    # Linearization of the original paths of the log files on basis of the node positions
    def lin(self):
        """"Linearization of the mouse position based on node positions"""

        n = 0
        path_0, path_1 = None, None

        for dat in [self.dat_0, self.dat_1]:
            path = pkg_resources.resource_filename(self.pathname, '/data/interim/linearized_Position_log_files/{}/pos_log_file_lin_{}.csv'.format(self.sources[0][len(self.sources[0])-29:len(self.sources[0])-10], n))
            pos_log_file = open(path, 'w')
            pos_log_file.write('x, y, frame_n, LED state, time, rel_pos, closest node, second closest node\n')

            if n == 0:
                flower_graph = {1: [2, 6],
                                2: [1, 3],
                                3: [2, 4, 7],
                                4: [3, 5],
                                5: [4, 8],
                                6: [1, 9, 17],
                                7: [3, 9, 10],
                                8: [5, 10, 11],
                                9: [6, 7, 12],
                                10: [7, 8, 13],
                                11: [8, 14],
                                12: [9, 15, 19],
                                13: [10, 15, 16],
                                14: [11, 16],
                                15: [12, 13],
                                16: [13, 14],
                                17: [6, 18],
                                18: [17, 19],
                                19: [12, 18], }

                node_positions = {}
                path_nodes = pkg_resources.resource_filename(self.pathname, 'src/Resources/aligned/corr_node_pos_top.csv')
                with open(path_nodes, 'r') as npf:
                    next(npf)
                    for line in npf:
                        x, y, nn = map(str.strip, line.split(','))
                        node_positions[float(nn)] = (float(x), float(y))

                mg = nx.Graph(flower_graph)

            if n == 1:
                flower_graph = {6: [9, 17],
                                7: [9, 10],
                                8: [10, 11],
                                9: [6, 7, 12],
                                10: [7, 8, 13],
                                11: [8, 14],
                                12: [9, 15, 19],
                                13: [10, 15, 16],
                                14: [11, 16],
                                15: [12, 13, 22],
                                16: [13, 14, 24],
                                17: [6, 18],
                                18: [17, 19],
                                19: [12, 18, 20],
                                20: [19, 21],
                                21: [20, 22],
                                22: [15, 21, 23],
                                23: [22, 24],
                                24: [16, 23]}

                node_positions = {}
                path_nodes = pkg_resources.resource_filename(self.pathname,
                                                             'src/Resources/aligned/corr_node_pos_bot.csv')
                with open(path_nodes, 'r') as npf:
                    next(npf)
                    for line in npf:
                        x, y, nn = map(str.strip, line.split(','))
                        node_positions[float(nn)] = (float(x), float(y))

                mg = nx.Graph(flower_graph)

            edge_list = list(mg.edges)
            vertex_list = list(node_positions.keys())
            distances = np.zeros(len(edge_list))

            for [x, y, z, l, time] in dat:
                rel_pos = np.nan
                closest_vertex = None
                second_vertex = None

                if not np.isnan(x):
                    # Find closest edge with projection
                    for i, edge in enumerate(edge_list):
                        n1 = node_positions[edge[0]]
                        n2 = node_positions[edge[1]]
                        d, p4 = ls_dist(n1, n2, (x,y))
                        distances[i] = d
                    closest_idx = distances.argmin()
                    closest_edge = edge_list[closest_idx]

                    # Find closest vertex from projection
                    v1 = closest_edge[0]
                    v2 = closest_edge[1]
                    d1 = distance(node_positions[v1][0], node_positions[v1][1], x, y)
                    d2 = distance(node_positions[v2][0], node_positions[v2][1], x, y)
                    vertex_dist = distance(node_positions[v1][0], node_positions[v1][1], node_positions[v2][0], node_positions[v2][1])
                    if d1 > d2:
                        closest_vertex = v2
                        second_vertex = v1
                        dist = d2
                    else:
                        closest_vertex = v1
                        second_vertex = v2
                        dist = d1

                    rel_pos = dist/vertex_dist

                    # Correct for cases in which the mouse is 'behind' the node, fix these positions to being on top of
                    # the node itself
                    if rel_pos > 1:
                        rel_pos = 1
                    if rel_pos < 0:
                        rel_pos = 0

                # Add the relative position and two closest nodes to the log file
                pos_log_file.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(x, y, z, l, time, rel_pos, closest_vertex, second_vertex))

            pos_log_file.close()

            if n == 0:
                path_0 = path
            elif n == 1:
                path_1 = path

            n += 1

        return path_0, path_1


class GroundTruth:
    def __init__(self, pathname, path_0, path_1, sources):
        self.dat_0 = np.genfromtxt(path_0, delimiter=',', skip_header=True)
        self.dat_1 = np.genfromtxt(path_1, delimiter=',', skip_header=True)
        self.ref_nodes = np.genfromtxt(pkg_resources.resource_filename(pathname, "/src/Resources/default/ref_nodes.csv"), delimiter=',', skip_header=True)
        self.sources = sources
        self.pathname = pathname

        make_directory(pathname, "/data/interim/pos_log_files_gt")
        make_directory(pathname, "/data/interim/pos_log_files_gt/{}".format(sources[0][len(sources[0])-29:len(sources[0])-10]))
        make_directory(pathname, "/data/interim/pos_log_files_stitched/{}".format(sources[0][len(sources[0])-29:len(sources[0])-10]))

    def gt_mapping(self):
        """"Mapping of linearized mouse positions to a ground truth map based on relative position (for both
         sources separately)"""

        n = 0

        for dat in [self.dat_0, self.dat_1]:
            path = pkg_resources.resource_filename(self.pathname, '/data/interim/pos_log_files_gt/{}/pos_log_file_gt_{}.csv'.format(self.sources[0][len(self.sources[0])-29:len(self.sources[0])-10], n))
            pos_log_file = open(path, 'w')
            pos_log_file.write('x, y, frame_n, LED state, time, rel_pos, closest node, second closest node, gt_x, gt_y\n')

            for k in range(len(dat)):
                x4, y4 = np.nan, np.nan
                if not np.isnan(dat[k, 5]):
                    x1 = self.ref_nodes[int(dat[k, 6]) - 1, 0]
                    y1 = self.ref_nodes[int(dat[k, 6]) - 1, 1]
                    x2 = self.ref_nodes[int(dat[k, 7]) - 1, 0]
                    y2 = self.ref_nodes[int(dat[k, 7]) - 1, 1]
                    d = dat[k, 5] * distance(x1, y1, x2, y2)

                    if x1 != x2:
                        slope = (y2 - y1) / (x2 - x1)

                        a = 1 + slope ** 2
                        b = (-2 * x1 - 2 * slope ** 2 * x1)
                        c = (x1 ** 2 + slope ** 2 * x1 ** 2 - d ** 2)
                        if x1 >= x2:
                            x4 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                        elif x1 <= x2:
                            x4 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                        y4 = slope * (x4 - x1) + y1

                    elif x1 == x2:
                        x4 = x1
                        if y1 > y2:
                            y4 = y1 - d
                        elif y1 < y2:
                            y4 = y1 + d

                pos_log_file.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(dat[k, 0], dat[k, 1], dat[k, 2], dat[k, 3], dat[k, 4], dat[k, 5], dat[k, 6], dat[k, 7], x4, y4))

            if n == 0:
                self.path_0 = path
            if n == 1:
                self.path_1 = path

            n += 1

        return self.path_0, self.path_1

    def gt_stitch(self):
        """"Stitching together the ground truth positions of both sources to one set of positions"""

        path = pkg_resources.resource_filename(self.pathname,
                                               '/data/interim/pos_log_files_stitched/{}/pos_log_file_stitched.csv'.format(
                                                   self.sources[0][
                                                   len(self.sources[0]) - 29:len(self.sources[0]) - 10]))
        pos_log_file = open(path, 'w')
        pos_log_file.write('frame_n, x, y, time\n')

        dat_0 = np.genfromtxt(self.path_0, delimiter=',', skip_header=True)
        dat_1 = np.genfromtxt(self.path_1, delimiter=',', skip_header=True)

        x_log = []
        y_log = []

        for i in range(len(dat_0)):

            dist = np.nan

            if not np.isnan(dat_0[i, 0]) and not np.isnan(dat_1[i, 0]):
                dist = distance(dat_0[i, 8], dat_0[i, 9], dat_1[i, 8], dat_1[i, 9])

                if dist <= 400:
                    x = (dat_0[i, 8] + dat_1[i, 8]) / 2
                    y = (dat_0[i, 9] + dat_1[i, 9]) / 2

                if x_log != []:
                    dist_0 = distance(dat_0[i, 8], dat_0[i, 9], x_log[len(x_log) - 1], y_log[len(y_log) - 1])
                    dist_1 = distance(dat_1[i, 8], dat_1[i, 9], x_log[len(x_log) - 1], y_log[len(y_log) - 1])

                    if dist_0 < dist_1:
                        x = dat_0[i, 8]
                        y = dat_0[i, 9]

                    elif dist_0 > dist_1:
                        x = dat_1[i, 8]
                        y = dat_1[i, 9]

                    else:
                        x = (dat_0[i, 8] + dat_1[i, 8]) / 2
                        y = (dat_0[i, 9] + dat_1[i, 9]) / 2

                else:
                    x = (dat_0[i, 8] + dat_1[i, 8]) / 2
                    y = (dat_0[i, 9] + dat_1[i, 9]) / 2

            elif not np.isnan(dat_0[i, 0]) and np.isnan(dat_1[i, 0]):
                x = dat_0[i, 8]
                y = dat_0[i, 9]

            elif np.isnan(dat_0[i, 0]) and not np.isnan(dat_1[i, 0]):
                x = dat_1[i, 8]
                y = dat_1[i, 9]

            else:
                x = np.nan
                y = np.nan

            if not np.isnan(x):
                x_log.append(x)
                y_log.append(y)

            time = dat_0[i, 4]

            pos_log_file.write("{}, {}, {}, {}\n".format(i, x, y, time))

        return path


# Calculation of the Homography matrix and remapping of node and LED position for the new videos
class Homography:
    def __init__(self, pathname, sources):
        self.sources = sources

        self.pathname = pathname

        make_directory(self.pathname, "/data/raw/{}".format(sources[0][len(sources[0])-29:len(sources[0])-10]))

        # Create a copy of the first frame and the automatically generated mask, for example to be used to create
        #  a new hand-made mask if automatically generated mask is deemed insufficient
        for n, source in enumerate(self.sources):
            make_directory(self.pathname, "/data/raw/{}/frame_images".format(
                sources[0][len(sources[0]) - 29:len(sources[0]) - 10], n))
            make_directory(self.pathname, "/data/raw/{}/masks".format(
                sources[0][len(sources[0]) - 29:len(sources[0]) - 10]))

            cap = cv2.VideoCapture(source)
            cap.set(1, 1)
            rt, frame = cap.read()
            output = pkg_resources.resource_filename(self.pathname, "/data/raw/{}/frame_images/frame_{}.png".format(sources[0][len(sources[0])-29:len(sources[0])-10], n))
            cv2.imwrite(output, frame)
            cap.release()
            cv2.destroyAllWindows()

        make_directory(pathname, "/src/resources/aligned")

        # Load in all node and LED boundary positions
        self.stand_0_path = pkg_resources.resource_filename(pathname, '/src/resources/default/stand_0.png')
        self.stand_1_path = pkg_resources.resource_filename(pathname, '/src/resources/default/stand_1.png')

        self.stand_nodes_top_path = pkg_resources.resource_filename(pathname, '/src/resources/default/node_pos_top.csv')
        self.stand_nodes_bot_path = pkg_resources.resource_filename(pathname, '/src/resources/default/node_pos_bottom.csv')

        self.im_0_path = pkg_resources.resource_filename(pathname, "/data/raw/{}/frame_images/frame_0.png".format(self.sources[0][len(self.sources[0])-29:len(self.sources[0])-10]))
        self.im_1_path = pkg_resources.resource_filename(pathname, "/data/raw/{}/frame_images/frame_1.png".format(self.sources[0][len(self.sources[0])-29:len(self.sources[0])-10]))

        self.stand_LED_top_path = pkg_resources.resource_filename(pathname, '/src/resources/default/LED_top.csv')
        self.stand_LED_bot_path = pkg_resources.resource_filename(pathname, '/src/resources/default/LED_bot.csv')

        self.stand_nodes_top = np.genfromtxt(self.stand_nodes_top_path, delimiter=',', skip_header=True)
        self.stand_nodes_bot = np.genfromtxt(self.stand_nodes_bot_path, delimiter=',', skip_header=True)

        self.stand_LED_top = np.genfromtxt(self.stand_LED_top_path, delimiter=',', skip_header=True)
        self.stand_LED_bot = np.genfromtxt(self.stand_LED_bot_path, delimiter=',', skip_header=True)

        self.stand_0 = cv2.imread(self.stand_0_path, 0)
        self.stand_1 = cv2.imread(self.stand_1_path, 0)

        self.im_0 = cv2.imread(self.im_0_path, 0)
        self.im_1 = cv2.imread(self.im_1_path, 0)

        # The minimum amount of matches for feature matching in order to continue
        self.MIN_MATCH_COUNT = 50

        # Initiate SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()

        # Set up parameters for feature matching
        self.FLANN_INDEX_KDTREE = 0
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=20000)

        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

        self.M_0, self.M_1 = None, None
        self.dst_0, self.dst_1 = None, None
        self.pts_0, self.pts_1 = None, None
        self.LED_top, self.LED_bot = None,  None

        self.LED_pts_0, self.LED_dst_0 = None, None

        # Initiate position files for homography-corrected node and LED crop positions
        corr_top = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_node_pos_top.csv')
        corr_bot = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_node_pos_bot.csv')
        corr_LED_top = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_LED_top.csv')
        corr_LED_bot = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_LED_bot.csv')
        self.corr_top = open(corr_top, 'w')
        self.corr_bot = open(corr_bot, 'w')
        self.corr_LED_top = open(corr_LED_top, 'w')
        self.corr_LED_bot = open(corr_LED_bot, 'w')
        self.corr_top.write('x, y, node\n')
        self.corr_bot.write('x, y, node\n')
        self.corr_LED_top.write('x, y\n')
        self.corr_LED_bot.write('x, y\n')

        # Emtpy lists for multiple purposes
        self.stdvs1, self.stdvs2 = [], []
        self.log_onsets, self.log_offsets = [], []
        self.LED_pts_0, self.LED_pts_1 = [], []

    def homography_calc(self):
        """"Calculation of the homography matrices of the top and bottom video sources"""

        # find the key points and descriptors with SIFT
        kp1, des1 = self.sift.detectAndCompute(self.stand_0, None)
        kp2, des2 = self.sift.detectAndCompute(self.im_0, None)
        kp3, des3 = self.sift.detectAndCompute(self.stand_1, None)
        kp4, des4 = self.sift.detectAndCompute(self.im_1, None)

        matches_0 = self.flann.knnMatch(des1, des2, k=2)
        matches_1 = self.flann.knnMatch(des3, des4, k=2)

        # store all the good matches as per Lowe's ratio test.
        good_0 = []
        for m, n in matches_0:
            if m.distance < 0.7 * n.distance:
                good_0.append(m)

        good_1 = []
        for m, n in matches_1:
            if m.distance < 0.7 * n.distance:
                good_1.append(m)

        # If enough good matches have been found, calculate new node locations
        if len(good_0) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_0]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_0]).reshape(-1, 1, 2)

            # Calculate the homography matrix
            self.M_0, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Calculate updated points
            self.pts_0 = np.float32(self.stand_nodes_top[:, 0:2]).reshape(-1, 1, 2)
            self.dst_0 = cv2.perspectiveTransform(self.pts_0, self.M_0)

            self.LED_pts_0 = np.float32(self.stand_LED_top.reshape(-1, 1, 2))
            self.LED_dst_0 = cv2.perspectiveTransform(self.LED_pts_0, self.M_0)

            # Save all new points to new csv files
            for i in range(len(self.stand_LED_top[:, 0])):
                self.corr_LED_top.write(
                    '{}, {}\n'.format(int(self.LED_dst_0[i, 0, 0]), int(self.LED_dst_0[i, 0, 1])))
            for i in range(len(self.stand_nodes_top[:, 0])):
                self.corr_top.write(
                    '{}, {}, {}\n'.format(self.dst_0[i, 0, 0], self.dst_0[i, 0, 1], self.stand_nodes_top[i, 2]))
        else:
            logging.debug('Error: Not enough matches found')

        self.corr_top.close()
        self.corr_LED_top.close()

        # If enough good matches have been found, calculate new node locations
        if len(good_1) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp3[m.queryIdx].pt for m in good_1]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp4[m.trainIdx].pt for m in good_1]).reshape(-1, 1, 2)

            # Calculate the homography matrix
            self.M_1, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Calculate updated points
            self.pts_1 = np.float32(self.stand_nodes_bot[:, 0:2]).reshape(-1, 1, 2)
            self.dst_1 = cv2.perspectiveTransform(self.pts_1, self.M_1)

            self.LED_pts_1 = np.float32(self.stand_LED_bot.reshape(-1, 1, 2))
            self.LED_dst_1 = cv2.perspectiveTransform(self.LED_pts_1, self.M_1)

            # Save all new points to new csv files
            for i in range(len(self.stand_LED_bot[:, 0])):
                self.corr_LED_bot.write(
                    '{}, {}\n'.format(int(self.LED_dst_1[i, 0, 0]), int(self.LED_dst_1[i, 0, 1])))
            for i in range(len(self.stand_nodes_bot[:, 0])):
                self.corr_bot.write(
                    '{}, {}, {}\n'.format(self.dst_1[i, 0, 0], self.dst_1[i, 0, 1], self.stand_nodes_bot[i, 2]))

        else:
            logging.debug('Error: Not enough matches found')

        # Close the new position files for future usage in pipeline
        self.corr_bot.close()
        self.corr_LED_bot.close()

    # Finds the position of the LED light on basis of the Stddev of the first few frames of the video
    def LEDfind(self, sources, iterations=100):
        """"Find the position of the LED light in both frames using the standard deviation of a
        cut-out part of the frames of both sources"""

        self.LED_top = self.LED_dst_0
        self.LED_bot = self.LED_dst_1

        # Find the borders of the new LED crop points
        minx1, miny1 = int(np.min(self.LED_top[:, 0, 0])), int(np.min(self.LED_top[:, 0, 1]))
        maxx1, maxy1 = int(np.max(self.LED_top[:, 0, 0])), int(np.max(self.LED_top[:, 0, 1]))

        minx2, miny2 = int(np.min(self.LED_bot[:, 0, 0])), int(np.min(self.LED_bot[:, 0, 1]))
        maxx2, maxy2 = int(np.max(self.LED_bot[:, 0, 0])), int(np.max(self.LED_bot[:, 0, 1]))

        # Start the video capture of the two videos
        cap_0 = cv2.VideoCapture(sources[0])
        cap_1 = cv2.VideoCapture(sources[1])

        # Calculate the size of the cropped videos
        width1, height1 = maxx1 - minx1, maxy1 - miny1
        width2, height2 = maxx2 - minx2, maxy2 - miny2

        # Initiate matrices to save pixel data in
        val1 = np.zeros((height1, width1, iterations))
        val2 = np.zeros((height2, width2, iterations))
        i = 0

        # Calculate the value of all pixels in the cropped video files and save them in the initiated matrices
        while i < iterations-1:
            rt, frame_0 = cap_0.read()
            frame_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)
            frame_0 = frame_0[miny1:maxy1, minx1:maxx1]
            for k in range(height1):
                for j in range(width1):
                    val1[k, j, i] = frame_0[k, j]
            rt, frame_1 = cap_1.read()
            frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
            frame_1 = frame_1[miny2:maxy2, minx2:maxx2]
            for k in range(height2):
                for j in range(width2):
                    val2[k, j, i] = frame_1[k, j]
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i += 1
        cap_0.release()
        cv2.destroyAllWindows()

        # Calculate the standard deviation of all pixels over the amount of frames (iterations)
        for i in range(height1):
            for j in range(width1):
                stdv1 = np.std(val1[i, j, :])
                self.stdvs1.append(stdv1)

        # Find the position of the pixel in the cropped frame for which the standard deviation is the highest, this
        # pixel should be the pixel in which the LED-light is located
        n1 = np.argmax(self.stdvs1)
        y1 = np.floor(n1 / width1)
        x1 = n1 - y1 * width1

        # Calculate the standard deviation of all pixels over the amount of frames (iterations)
        for i in range(height2):
            for j in range(width2):
                stdv2 = np.std(val2[i, j, :])
                self.stdvs2.append(stdv2)

        # Find the position of the pixel in the cropped frame for which the standard deviation is the highest, this
        # pixel should be the pixel in which the LED-light is located
        n2 = np.argmax(self.stdvs2)
        y2 = np.floor(n2 / width2)
        x2 = n2 - y2 * width2

        # Map the LED light location from the cropped frame to the original video frame
        x1 += minx1
        y1 += miny1
        x2 += minx2
        y2 += miny2

        return [x1, y1, x2, y2]

    def LED_thresh(self, sources, iterations, LED_pos):
        """"Find the thresholds of the LED light for on- and offsets in both video sources"""

        led_values_0 = []
        led_values_1 = []

        # Start the video capture of the two videos
        cap_0 = cv2.VideoCapture(sources[0])
        cap_1 = cv2.VideoCapture(sources[1])

        i = 0
        # Calculate the value of all pixels in the cropped video files and save them in the initiated matrices
        while i < iterations:
            _, frame_0 = cap_0.read()
            _, frame_1 = cap_1.read()

            led_frame_0 = frame_0[int((LED_pos[1]-2)):int((LED_pos[1]+2)),
                             int((LED_pos[0]-2)):int((LED_pos[0]+2))]
            led_frame_1 = frame_1[int((LED_pos[3]-2)):int((LED_pos[3]+2)),
                             int((LED_pos[2]-2)):int((LED_pos[2]+2))]

            led_val_0 = np.mean(led_frame_0)
            led_val_1 = np.mean(led_frame_1)
            led_values_0.append(led_val_0)
            led_values_1.append(led_val_1)

            i += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap_0.release()
        cv2.destroyAllWindows()

        min_0, max_0 = np.min(led_values_0), np.max(led_values_0)
        min_1, max_1 = np.min(led_values_1), np.max(led_values_1)
        thresh_0, thresh_1 = (min_0 + max_0)/2, (min_1 + max_1)/2

        return [thresh_0, thresh_1]


class TrialCut:
    def __init__(self, paths, data, sources):

        # Load in all data and the excel file containing the log data of the experiment
        self.path_vid_0 = paths[0]
        self.path_vid_1 = paths[1]
        self.log_path = paths[2]
        self.dat_0 = np.genfromtxt(data[0], delimiter=',', skip_header=False)
        self.dat_1 = np.genfromtxt(data[1], delimiter=',', skip_header=False)
        self.dat = np.genfromtxt(data[2], delimiter=',', skip_header=True)

        self.df = pd.read_excel(self.log_path, dtype='str')
        self.array = self.df.values

        self.log_onsets, self.log_offsets = [], []
        self.onsets, self.offsets = [], []

        self.vid = VideoFileClip(sources[0])
        self.duration = self.vid.duration*15
        self.vid.reader.close()

    def log_data(self):
        """"Find the specific timings of the on- and offsets of trials in the time-aligned log files"""

        # Standard video timing (to correct for the normal start of the video)
        # Example: video timestamp = 14:00, onset time = 14:20 --> time in video = 00:20
        vid_t = np.floor((3600 * int(self.path_vid_0[len(self.path_vid_0)-18:len(self.path_vid_0)-16]) + 60 * int(self.path_vid_0[len(self.path_vid_0)-15:len(self.path_vid_0)-13]) + int(
            self.path_vid_0[len(self.path_vid_0)-12:len(self.path_vid_0)-10])))

        # Load in the on- and offsets of trials from the log file
        onsets = self.array[:, 2][:]
        offsets = self.array[:, 3][:]

        for i in range(len(onsets)):
            self.onsets.append(onsets[i][11:19])

        for i in range(len(onsets)):
            self.offsets.append(offsets[i][11:19])

        # Find timstamps in log files
        for i in range(len(self.onsets)):
            on_t = np.floor((3600 * int(self.onsets[i][0:2]) + 60 * int(self.onsets[i][3:5]) + int(self.onsets[i][6:8])) - vid_t)
            off_t = np.floor((3600 * int(self.offsets[i][0:2]) + 60 * int(self.offsets[i][3:5]) + int(self.offsets[i][6:8])) - vid_t)+1.5

            on_tf = np.argwhere(self.dat_0[:, 4] == on_t)
            if on_t in self.dat_0[:, 4]:
                on_tf = on_tf[0][0]
            else:
                on_tf = np.nan
            self.log_onsets.append(on_tf)

            off_tf = np.argwhere(self.dat_0[:, 4] == off_t)
            if off_t in self.dat_0[:, 4]:
                off_tf = off_tf[len(off_tf)-1][0]
            else:
                off_tf = np.nan
            self.log_offsets.append(off_tf)

    def cut(self, pathname):
        """"Cut away the relevant data from the log file for each trial and save it in two separate files"""

        make_directory(pathname, "/data/processed/{}".format(self.path_vid_0[len(self.path_vid_0)-29:len(self.path_vid_0)-10]))

        # Loop through all trials
        n = 1
        for i in range(len(self.log_onsets)):

            if not np.isnan(self.log_onsets[i]) and not np.isnan(self.log_offsets[i]):

                # Make directories for saving data files of trials
                make_directory(pathname, "/data/processed/{}/{}".format(
                    self.path_vid_0[len(self.path_vid_0) - 29:len(self.path_vid_0) - 10], "trial_{}".format(n)))

                make_directory(pathname, "/data/processed/{}/{}/position_log_files".format(
                    self.path_vid_0[len(self.path_vid_0) - 29:len(self.path_vid_0) - 10], "trial_{}".format(n)))

                path = pkg_resources.resource_filename(pathname, "/data/processed/{}/{}/position_log_files".format(
                    self.path_vid_0[len(self.path_vid_0) - 29:len(self.path_vid_0) - 10], "trial_{}".format(n)))

                if os.path.exists(path):
                    self.path_0 = path + '/pos_log_file_0.csv'
                    self.path_1 = path + '/pos_log_file_1.csv'

                    self.dat_0f = self.dat_0[self.log_onsets[i]:self.log_offsets[i]]
                    self.dat_1f = self.dat_1[self.log_onsets[i]:self.log_offsets[i]]

                    # Sometimes experimenters double click the physical clicker by mistake, these 'trials'
                    # get cut out immediately; it is assumed no trial will take less than 5 (time aligned) frames
                    if not self.log_offsets[i]-self.log_onsets[i] <= 5:

                        np.savetxt(self.path_0, self.dat_0f, delimiter=",", header="x,y,frame_n,LED_state, rel_pos, first node, second node, gt_x, gt_y", comments='')
                        np.savetxt(self.path_1, self.dat_1f, delimiter=",", header="x,y,frame_n,LED_state, rel_pos, first node, second node, gt_x, gt_y", comments='')

                        n += 1

    def cut_stitch(self, pathname):
        """"Cut away the relevant data from the log file for each trial and save it in one file"""

        make_directory(pathname, "/data/processed/{}".format(
            self.path_vid_0[len(self.path_vid_0) - 29:len(self.path_vid_0) - 10]))

        # Loop through all trials
        n = 1
        for i in range(len(self.log_onsets)):

            if not np.isnan(self.log_onsets[i]) and not np.isnan(self.log_offsets[i]):

                # Make directories for saving data files of trials
                make_directory(pathname, "/data/processed/{}/{}".format(
                    self.path_vid_0[len(self.path_vid_0) - 29:len(self.path_vid_0) - 10], "trial_{}".format(n)))

                make_directory(pathname, "/data/processed/{}/{}/position_log_files".format(
                    self.path_vid_0[len(self.path_vid_0) - 29:len(self.path_vid_0) - 10], "trial_{}".format(n)))

                path = pkg_resources.resource_filename(pathname, "/data/processed/{}/{}/position_log_files".format(
                    self.path_vid_0[len(self.path_vid_0) - 29:len(self.path_vid_0) - 10], "trial_{}".format(n)))

                if os.path.exists(path):
                    self.path = path + '/pos_log_file.csv'

                    self.dat_f = self.dat[self.log_onsets[i]:self.log_offsets[i]]

                    # Sometimes experimenters double click the physical clicker by mistake, these 'trials'
                    # get cut out immediately; it is assumed no trial will take less than 5 (time aligned) frames
                    if not self.log_offsets[i] - self.log_onsets[i] <= 5:
                        np.savetxt(self.path, self.dat_f, delimiter=",", header="frame_n, x, y", comments='')

                        n += 1
