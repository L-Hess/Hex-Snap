import pkg_resources
import numpy as np
import logging
import cv2


# Euclidean distance
def distance(x1, y1, x2, y2):
    """Euclidean distance"""
    r = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return r


# Obtaining the linearized position using simple algebra
# Maps mouse position to the closest point on the line between the two closest nodes as seen from mouse position
def lin_pos(x1, y1, x2, y2, x3, y3, z):
    """Linearized position on basis of the two closest nodes"""
    x = None
    y = None
    if x1 is not None and x2 is not None and y1 is not None and y2 is not None:
        if x1 == x2:
            x = x1
            y = y3
        else:
            slope = (y2-y1)/(x2-x1)
            x = (y3+(1/slope)*x3-y1+slope*x1)/(slope+(1/slope))
            y = slope*(x-x1)+y1
        x, y = int(x), int(y)
    return x, y, z


# For time alignment of both videos on basis of the LED light flickering
class Timecorrect:
    def __init__(self, pathname, sources):

        # Loading in the position log files as created by the tracker
        self.data_path_1 = pkg_resources.resource_filename(pathname, '/data/Position_log_files/pos_log_file_0.csv')
        self.data_path_2 = pkg_resources.resource_filename(pathname, '/output/Position_log_files/pos_log_file_1.csv')

        self.dat_0 = np.genfromtxt(self.data_path_1, delimiter=',', skip_header=False)
        self.dat_1 = np.genfromtxt(self.data_path_2, delimiter=',', skip_header=False)

        # Initiation of new time aligned position log files
        path_0 = pkg_resources.resource_filename(pathname, '/output/Time_corrected_log_files/pos_log_file_tcorr_0.csv')
        path_1 = pkg_resources.resource_filename(pathname, '/output/Time_corrected_log_files/pos_log_file_tcorr_1.csv')
        self.pos_log_file_0 = open(path_0, 'w')
        self.pos_log_file_1 = open(path_1, 'w')

        self.dat_0f = None
        self.dat_1f = None

    # Time alignment of the position log files
    def correction(self):

        # Takes the LED-states of both log files, where 1 is on and 0 is off
        led_0 = self.dat_0[:, 3]
        led_1 = self.dat_1[:, 3]

        # Calculates where in the log-files the LED-state goes from off to on (0 to 1)
        led_0_peaks = np.diff(led_0) > 0
        led_1_peaks = np.diff(led_1) > 0
        i_0 = np.where(led_0_peaks == 1)[0]
        i_1 = np.where(led_1_peaks == 1)[0]

        # Check which position log file is longer (in general this is the log file of the bottom video)
        # Then creates a new log file for the other video and maps positions of LED-states of the shorter video
        # to the positions of LED-states of the longer video
        # This creates a new log file for the shorter video in which onsets of LED light are aligned between both
        #  log-files
        # Note that this can only be done if one of the videos is longer than the other, which will be the case in most
        # cases for 30 min long videos (as are used for the HexMaze)
        if len(led_0) > len(led_1):
            self.dat_0f = self.dat_0
            self.dat_1f = np.full_like(self.dat_0f, np.nan)
            for i in range(len(i_1) - 1):
                self.dat_0f[i_1[i]:i_1[i] + (i_0[i + 1] - i_0[i])] = self.dat_0[i_0[i]:i_0[i + 1]]
        elif len(led_0) < len(led_1):
            self.dat_1f = self.dat_1
            self.dat_0f = np.full_like(self.dat_1f, np.nan)
            for i in range(len(i_0) - 1):
                self.dat_0f[i_1[i]:i_1[i] + (i_0[i + 1] - i_0[i])] = self.dat_0[i_0[i]:i_0[i + 1]]
        elif len(led_0) == len(led_1):
            pass

        # Save the new log files
        if not len(led_0) == len(led_1):
            for i in range(len(self.dat_0f)):
                self.pos_log_file_0.write('{}, {}, {}, {}\n'.format(self.dat_0f[i, 0], self.dat_0f[i, 1], self.dat_0f[i, 2], self.dat_0f[i, 3]))

            for i in range(len(self.dat_1f)):
                self.pos_log_file_1.write('{}, {}, {}, {}\n'.format(self.dat_1f[i, 0], self.dat_1f[i, 1], self.dat_1f[i, 2], self.dat_1f[i, 3]))


class Linearization:
    def __init__(self, pathname):

        # Load in the time aligned log files and corrected node positions
        # Ghost nodes are used for nodes that are off of the video
        # Dwell nodes are used for nodes that are off of the video
        self.data_path_1 = pkg_resources.resource_filename(pathname, '/output/Time_corrected_log_files/pos_log_file_tcorr_0.csv')
        self.data_path_2 = pkg_resources.resource_filename(pathname, '/output/Time_corrected_log_files/pos_log_file_tcorr_1.csv')

        self.dat_0 = np.genfromtxt(self.data_path_1, delimiter=',', skip_header=False)
        self.dat_1 = np.genfromtxt(self.data_path_2, delimiter=',', skip_header=False)

        self.path_0 = pkg_resources.resource_filename(pathname, '/output/Linearized_Position_log_files/pos_log_file_lin_0.csv')
        self.path_1 = pkg_resources.resource_filename(pathname, '/output/Linearized_Position_log_files/pos_log_file_lin_1.csv')
        self.pos_log_file_lin_0 = open(self.path_0, 'w')
        self.pos_log_file_lin_1 = open(self.path_1, 'w')

        self.nodes_top_path = pkg_resources.resource_filename(pathname, '/src/resources/node_pos_top.csv')
        self.nodes_bot_path = pkg_resources.resource_filename(pathname, '/src/resources/node_pos_bottom.csv')
        self.ghost_nodes_top_path = pkg_resources.resource_filename(pathname, '/src/resources/ghost_node_top.csv')
        self.ghost_nodes_bot_path = pkg_resources.resource_filename(pathname, '/src/resources/ghost_node_bottom.csv')
        self.dwell_nodes_top_path = pkg_resources.resource_filename(pathname, '/src/resources/ghost_node_top.csv')
        self.dwell_nodes_bot_path = pkg_resources.resource_filename(pathname, '/src/resources/ghost_node_bottom.csv')
        self.im_0_path = pkg_resources.resource_filename(pathname, '/output/Video_images/im_0.png')
        self.im_1_path = pkg_resources.resource_filename(pathname, '/output/Video_images/im_1.png')

        self.nodes_top = np.genfromtxt(self.nodes_top_path, delimiter=',', skip_header=True)
        self.nodes_bot = np.genfromtxt(self.nodes_bot_path, delimiter=',', skip_header=True)
        self.ghost_nodes_top = np.genfromtxt(self.ghost_nodes_top_path, delimiter=',', skip_header=True)
        self.ghost_nodes_bot = np.genfromtxt(self.ghost_nodes_bot_path, delimiter=',', skip_header=True)
        self.dwell_nodes_top = np.genfromtxt(self.dwell_nodes_top_path, delimiter=',', skip_header=True)
        self.dwell_nodes_bot = np.genfromtxt(self.dwell_nodes_bot_path, delimiter=',', skip_header=True)

    # Linearization of the original paths of the log files on basis of the node positions
    def lin(self):
        for [x, y, z, l] in self.dat_0:

            # Calculate the distance of each mouse position to all nodes
            dist = distance(x, y, self.nodes_top[:, 0], self.nodes_top[:, 1])

            # Finds the closest node and saves its position and node number
            dist1, node1 = np.min(dist), self.nodes_top[np.argmin(dist), 2]
            x_node_1, y_node_1 = self.nodes_top[np.argmin(dist), 0], self.nodes_top[np.argmin(dist), 1]
            if np.isnan(dist1):
                node1 = None
                x_node_1, y_node_1 = None, None

            # Find the second closest node and save its position and node number
            dist[np.argmin(dist)] = 1e12
            dist2, node2 = np.min(dist), self.nodes_top[np.argmin(dist), 2]
            x_node_2, y_node_2 = self.nodes_top[np.argmin(dist), 0], self.nodes_top[np.argmin(dist), 1]
            if np.isnan(dist2):
                node2 = None
                x_node_2, y_node_2 = None, None

            # Find the relative position between both nodes on basis of the linearized position
            # (see the function lin_pos as earlier defined)
            rel_pos = np.nan
            if not np.isnan(dist1):
                nodes_dist = distance(x_node_1, y_node_1, x_node_2, y_node_2)
                x_lin, y_lin, z_lin = lin_pos(x_node_1, y_node_1, x_node_2, y_node_2, x, y, z)
                lin_dist_1 = distance(x_lin, y_lin, x_node_1, y_node_1)
                lin_dist_2 = distance(x_lin, y_lin, x_node_2, y_node_2)
                if lin_dist_1 + lin_dist_2 > nodes_dist + 0.2:
                    rel_pos = -lin_dist_1 / nodes_dist
                else:
                    rel_pos = lin_dist_1 / nodes_dist

            # Add the relative position and two closest nodes to the log file
            self.pos_log_file_lin_0.write('{}, {}, {}, {}, {}, {}, {}\n'.format(x, y, z, l, rel_pos, node1, node2))

        for [x, y, z, l] in self.dat_1:

            # Calculate the distance of each mouse position to all nodes
            dist = distance(x, y, self.nodes_bot[:, 0], self.nodes_bot[:, 1])

            # Finds the closest node and saves its position and node number
            dist1, node1 = np.min(dist), self.nodes_bot[np.argmin(dist), 2]
            x_node_1, y_node_1 = self.nodes_bot[np.argmin(dist), 0], self.nodes_bot[np.argmin(dist), 1]
            if np.isnan(dist1):
                node1 = None
                x_node_1, y_node_1 = None, None

            # Find the second closest node and save its position and node number
            dist[np.argmin(dist)] = 1e12
            dist2, node2 = np.min(dist), self.nodes_bot[np.argmin(dist), 2]
            x_node_2, y_node_2 = self.nodes_bot[np.argmin(dist), 0], self.nodes_bot[np.argmin(dist), 1]
            if np.isnan(dist2):
                node2 = None
                x_node_2, y_node_2 = None, None

            # Find the relative position between both nodes on basis of the linearized position
            # (see the function lin_pos as earlier defined)
            rel_pos = np.nan
            if not np.isnan(dist1):
                nodes_dist = distance(x_node_1, y_node_1, x_node_2, y_node_2)
                x_lin, y_lin, z_lin = lin_pos(x_node_1, y_node_1, x_node_2, y_node_2, x, y, z)
                lin_dist_1 = distance(x_lin, y_lin, x_node_1, y_node_1)
                lin_dist_2 = distance(x_lin, y_lin, x_node_2, y_node_2)
                if lin_dist_1 + lin_dist_2 > nodes_dist + 0.2:
                    rel_pos = -lin_dist_1 / nodes_dist
                else:
                    rel_pos = lin_dist_1 / nodes_dist

            # Add the relative position and two closest nodes to the log file
            self.pos_log_file_lin_1.write('{}, {}, {}, {}, {}, {}, {}\n'.format(x, y, z, l, rel_pos, node1, node2))


# Calculation of the Homography matrix and remapping of node and LED position for the new videos
class Homography:
    def __init__(self, pathname, sources):
        self.sources = sources

        self.pathname = pathname

        for n, source in enumerate(self.sources):
            cap = cv2.VideoCapture(source)
            cap.set(1,1)
            rt, frame = cap.read()
            output = pkg_resources.resource_filename(self.pathname, '/data/raw/frame_images/im_{}.png'.format(n))
            cv2.imwrite(output, frame)

        self.stand_0_path = pkg_resources.resource_filename(pathname, '/src/resources/default/stand_0.png')
        self.stand_1_path = pkg_resources.resource_filename(pathname, '/src/resources/default/stand_1.png')

        self.stand_nodes_top_path = pkg_resources.resource_filename(pathname, '/src/resources/default/node_pos_top.csv')
        self.stand_nodes_bot_path = pkg_resources.resource_filename(pathname, '/src/resources/default/node_pos_bottom.csv')

        self.stand_ghost_nodes_top_path = pkg_resources.resource_filename(pathname, '/src/resources/default/ghost_node_top.csv')
        self.stand_ghost_nodes_bot_path = pkg_resources.resource_filename(pathname, '/src/resources/default/ghost_node_bottom.csv')

        self.stand_dwell_nodes_top_path = pkg_resources.resource_filename(pathname, '/src/resources/default/dwell_nodes_top.csv')
        self.stand_dwell_nodes_bot_path = pkg_resources.resource_filename(pathname, '/src/resources/default/dwell_nodes_bot.csv')

        self.im_0_path = pkg_resources.resource_filename(pathname, '/data/raw/frame_images/im_0.png')
        self.im_1_path = pkg_resources.resource_filename(pathname, '/data/raw/frame_images/im_1.png')

        self.stand_LED_top_path = pkg_resources.resource_filename(pathname, '/src/resources/default/LED_top.csv')
        self.stand_LED_bot_path = pkg_resources.resource_filename(pathname, '/src/resources/default/LED_bot.csv')

        self.stand_nodes_top = np.genfromtxt(self.stand_nodes_top_path, delimiter=',', skip_header=True)
        self.stand_nodes_bot = np.genfromtxt(self.stand_nodes_bot_path, delimiter=',', skip_header=True)

        self.stand_ghost_nodes_top = np.genfromtxt(self.stand_ghost_nodes_top_path, delimiter=',', skip_header=True)
        self.stand_ghost_nodes_bot = np.genfromtxt(self.stand_ghost_nodes_bot_path, delimiter=',', skip_header=True)

        self.stand_dwell_nodes_top = np.genfromtxt(self.stand_dwell_nodes_top_path, delimiter=',', skip_header=True)
        self.stand_dwell_nodes_bot = np.genfromtxt(self.stand_dwell_nodes_bot_path, delimiter=',', skip_header=True)

        self.stand_LED_top = np.genfromtxt(self.stand_LED_top_path, delimiter=',', skip_header=True)
        self.stand_LED_bot = np.genfromtxt(self.stand_LED_bot_path, delimiter=',', skip_header=True)

        self.stand_0 = cv2.imread(self.stand_0_path, 0)
        self.stand_1 = cv2.imread(self.stand_1_path, 0)

        self.im_0 = cv2.imread(self.im_0_path, 0)
        self.im_1 = cv2.imread(self.im_1_path, 0)

        self.MIN_MATCH_COUNT = 50

        # Initiate SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()

        # Set up parameters for feature matching
        self.FLANN_INDEX_KDTREE = 0
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=50)

        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

        self.M_0, self.M_1 = None, None
        self.dst_0, self.dst_1 = None, None
        self.pts_0, self.pts_1 = None, None
        self.LED_top, self.LED_bot =  None,  None

        # Initiate position files for homography-corrected node and LED crop positions
        corr_top = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_node_pos_top.csv')
        corr_bot = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_node_pos_bot.csv')
        corr_ghost_top = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_ghost_node_pos_top.csv')
        corr_ghost_bot = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_ghost_node_pos_bot.csv')
        corr_dwell_top = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_dwell_node_pos_top.csv')
        corr_dwell_bot = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_dwell_node_pos_bot.csv')
        corr_LED_top = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_LED_top.csv')
        corr_LED_bot = pkg_resources.resource_filename(pathname, '/src/resources/aligned/corr_LED_bot.csv')
        self.corr_top = open(corr_top, 'w')
        self.corr_bot = open(corr_bot, 'w')
        self.corr_ghost_top = open(corr_ghost_top, 'w')
        self.corr_ghost_bot = open(corr_ghost_bot, 'w')
        self.corr_dwell_top = open(corr_dwell_top, 'w')
        self.corr_dwell_bot = open(corr_dwell_bot, 'w')
        self.corr_LED_top = open(corr_LED_top, 'w')
        self.corr_LED_bot = open(corr_LED_bot, 'w')
        self.corr_top.write('x, y, node\n')
        self.corr_bot.write('x, y, node\n')
        self.corr_ghost_top.write('x, y, node\n')
        self.corr_ghost_bot.write('x, y, node\n')
        self.corr_dwell_top.write('x, y, node\n')
        self.corr_dwell_bot.write('x, y, node\n')
        self.corr_LED_top.write('x, y\n')
        self.corr_LED_bot.write('x, y\n')

        self.stdvs1, self.stdvs2 = [], []

    def homography_calc(self):
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

            self.ghost_pts_0 = np.float32(self.stand_ghost_nodes_top[:, 0:2]).reshape(-1, 1, 2)
            self.ghost_dst_0 = cv2.perspectiveTransform(self.ghost_pts_0, self.M_0)

            self.dwell_pts_0 = np.float32(self.stand_dwell_nodes_top[:, 0:2]).reshape(-1, 1, 2)
            self.dwell_dst_0 = cv2.perspectiveTransform(self.dwell_pts_0, self.M_0)

            self.LED_pts_0 = np.float32(self.stand_LED_top.reshape(-1, 1, 2))
            self.LED_dst_0 = cv2.perspectiveTransform(self.LED_pts_0, self.M_0)

            # Save all new points to new csv files
            for i in range(len(self.stand_LED_top[:, 0])):
                self.corr_LED_top.write(
                    '{}, {}\n'.format(int(self.LED_dst_0[i, 0, 0]), int(self.LED_dst_0[i, 0, 1])))
            for i in range(len(self.stand_nodes_top[:, 0])):
                self.corr_top.write(
                    '{}, {}, {}\n'.format(self.dst_0[i, 0, 0], self.dst_0[i, 0, 1], self.stand_nodes_top[i, 2]))
            for i in range(len(self.stand_ghost_nodes_top[:, 0])):
                self.corr_ghost_top.write(
                    '{}, {}, {}\n'.format(self.ghost_dst_0[i, 0, 0], self.ghost_dst_0[i, 0, 1], self.stand_ghost_nodes_top[i, 2]))
            for i in range(len(self.stand_dwell_nodes_top[:, 0])):
                self.corr_dwell_top.write(
                    '{}, {}, {}\n'.format(self.dwell_dst_0[i, 0, 0], self.dwell_dst_0[i, 0, 1], self.stand_dwell_nodes_top[i, 2]))
        else:
            logging.debug('Error: Not enough matches found')

        self.corr_top.close()
        self.corr_ghost_top.close()
        self.corr_dwell_top.close()
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

            self.ghost_pts_1 = np.float32(self.stand_ghost_nodes_bot[:, 0:2]).reshape(-1, 1, 2)
            self.ghost_dst_1 = cv2.perspectiveTransform(self.ghost_pts_1, self.M_1)

            self.dwell_pts_1 = np.float32(self.stand_dwell_nodes_bot[:, 0:2]).reshape(-1, 1, 2)
            self.dwell_dst_1 = cv2.perspectiveTransform(self.dwell_pts_1, self.M_1)

            self.LED_pts_1 = np.float32(self.stand_LED_bot.reshape(-1, 1, 2))
            self.LED_dst_1 = cv2.perspectiveTransform(self.LED_pts_1, self.M_1)

            # Save all new points to new csv files
            for i in range(len(self.stand_LED_bot[:, 0])):
                self.corr_LED_bot.write(
                    '{}, {}\n'.format(int(self.LED_dst_1[i, 0, 0]), int(self.LED_dst_1[i, 0, 1])))
            for i in range(len(self.stand_nodes_bot[:, 0])):
                self.corr_bot.write(
                    '{}, {}, {}\n'.format(self.dst_1[i, 0, 0], self.dst_1[i, 0, 1], self.stand_nodes_bot[i, 2]))
            for i in range(len(self.stand_ghost_nodes_bot[:, 0])):
                self.corr_ghost_bot.write(
                    '{}, {}, {}\n'.format(self.ghost_dst_1[i, 0, 0], self.ghost_dst_1[i, 0, 1], self.stand_ghost_nodes_bot[i, 2]))
            for i in range(len(self.stand_dwell_nodes_bot[:, 0])):
                self.corr_dwell_bot.write(
                    '{}, {}, {}\n'.format(self.dwell_dst_1[i, 0, 0], self.dwell_dst_1[i, 0, 1], self.stand_dwell_nodes_bot[i, 2]))

        else:
            logging.debug('Error: Not enough matches found')

        self.corr_bot.close()
        self.corr_ghost_bot.close()
        self.corr_dwell_bot.close()
        self.corr_LED_bot.close()

    # Finds the position of the LED light on basis of the Stdev of the first few frames of the video
    def LEDfind(self, sources, iterations=100):
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
            i += 1
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
