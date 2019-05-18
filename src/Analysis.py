import numpy as np
import matplotlib.pyplot as plt
import pkg_resources
import logging


# Euclidean distance
def distance(x1, y1, x2, y2):
    r = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return r


# Obtaining the linearized position
def lin_pos(x1, y1, x2, y2, x3, y3, z):
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


class Display:
    def __init__(self, pathname):
        self.data_path_1 = pkg_resources.resource_filename(pathname, '/output/Position_log_files/pos_log_file_0.csv')
        self.data_path_2 = pkg_resources.resource_filename(pathname, '/output/Position_log_files/pos_log_file_1.csv')
        self.nodes_top_path = pkg_resources.resource_filename(pathname, '/output/Corr_node_pos/corr_node_pos_top.csv')
        self.nodes_bot_path = pkg_resources.resource_filename(pathname, '/output/Corr_node_pos/corr_node_pos_bot.csv')
        self.ghost_nodes_top_path = pkg_resources.resource_filename(pathname, '/output/Corr_node_pos/corr_ghost_node_pos_top.csv')
        self.ghost_nodes_bot_path = pkg_resources.resource_filename(pathname, '/output/Corr_node_pos/corr_node_pos_bot.csv')
        self.dwell_nodes_top_path = pkg_resources.resource_filename(pathname, '/output/Corr_node_pos/corr_dwell_node_pos_top.csv')
        self.dwell_nodes_bot_path = pkg_resources.resource_filename(pathname, '/output/Corr_node_pos/corr_dwell_node_pos_bot.csv')
        self.im_0_path = pkg_resources.resource_filename(pathname, '/output/Video_images/im_0.png')
        self.im_1_path = pkg_resources.resource_filename(pathname, '/output/Video_images/im_1.png')
        self.ref_nodes_path = pkg_resources.resource_filename(pathname, '/lib/resources/ref_nodes.csv')
        self.pos_log_file_lin_0_path = pkg_resources.resource_filename(pathname, '/output/Linearized_Position_log_files'
                                                                                 '/pos_log_file_lin_0.csv')
        self.pos_log_file_lin_1_path = pkg_resources.resource_filename(pathname, '/output/Linearized_Position_log_files'
                                                                                 '/pos_log_file_lin_1.csv')
        self.LED_top_path = pkg_resources.resource_filename(pathname, '/output/Corr_node_pos/corr_LED_top.csv')
        self.LED_bot_path = pkg_resources.resource_filename(pathname, '/output/Corr_node_pos/corr_LED_bot.csv')

        self.pathname = pathname

        self.dat_1 = np.genfromtxt(self.data_path_1, delimiter=',', skip_header=False)
        self.dat_2 = np.genfromtxt(self.data_path_2, delimiter=',', skip_header=False)

        self.nodes_top = np.genfromtxt(self.nodes_top_path, delimiter=',', skip_header=False)
        self.nodes_bottom = np.genfromtxt(self.nodes_bot_path, delimiter=',', skip_header=False)
        self.ghost_nodes_top = np.genfromtxt(self.ghost_nodes_top_path, delimiter=',', skip_header=True)
        self.ghost_nodes_bottom = np.genfromtxt(self.ghost_nodes_bot_path, delimiter=',', skip_header=True)
        self.dwell_nodes_top = np.genfromtxt(self.dwell_nodes_top_path, delimiter=',', skip_header=True)
        self.dwell_nodes_bottom = np.genfromtxt(self.dwell_nodes_bot_path, delimiter=',', skip_header=True)
        self.LED_top = np.genfromtxt(self.LED_top_path, delimiter=',', skip_header=True)
        self.LED_bot = np.genfromtxt(self.LED_bot_path, delimiter=',', skip_header=True)
        self.im_0 = plt.imread(self.im_0_path)
        self.im_1 = plt.imread(self.im_1_path)
        self.ref_nodes = np.genfromtxt(self.ref_nodes_path, delimiter=',', skip_header=True)
        self.pos_log_file_lin_0 = np.genfromtxt(self.pos_log_file_lin_0_path, delimiter=',', skip_header=False)
        self.pos_log_file_lin_1 = np.genfromtxt(self.pos_log_file_lin_1_path, delimiter=',', skip_header=False)

        self.x_top, self.y_top, self.z_top = [], [], []
        self.x_bot, self.y_bot, self.z_bot = [], [], []

        self.nodelist_bot = []
        self.nodelist_top = []

        self.z_0, self.z_1 = [], []
        self.l_0, self.l_1 = [], []

        self.xp_0, self.yp_0 = [], []
        self.xp_1, self.yp_1 = [], []

    def path_display(self):
        logging.debug('Saving raw path images initiated')
        plt.figure('Top')
        plt.imshow(self.im_0, cmap='gray')
        plt.plot(self.dat_1[:, 0], self.dat_1[:, 1], 'o', color='k')
        plt.plot(self.nodes_top[:15, 0], self.nodes_top[:15, 1], 'X', markersize=20, color='blue')
        plt.plot(self.ghost_nodes_top[:, 0], self.ghost_nodes_top[:, 1], 'o', markersize=20, color='orange')
        path = pkg_resources.resource_filename(self.pathname, '/output/Original_path/Original_path_top.png')
        plt.savefig(path)

        plt.figure('Bottom')
        plt.imshow(self.im_1, cmap='gray')
        plt.plot(self.dat_2[:, 0], self.dat_2[:, 1], 'o', color='k')
        plt.plot(self.nodes_bottom[:16, 0], self.nodes_bottom[:16, 1], 'X', markersize=20, color='blue')
        plt.plot(self.ghost_nodes_bottom[:, 0], self.ghost_nodes_bottom[:, 1], 'o', markersize=20, color='orange')
        path = pkg_resources.resource_filename(self.pathname, '/output/Original_path/Original_path_bot.png')
        plt.savefig(path)

        logging.debug('Saving raw path images done')

    def lin_path_display(self):
        logging.debug('Saving linear path images initiated')
        # Find x and y positions of the two closest nodes for all mouse positions
        for [x, y, z, l] in self.dat_1:
            dist = distance(x, y, self.nodes_top[:, 0], self.nodes_top[:, 1])
            dist1, node1 = np.min(dist), self.nodes_top[np.argmin(dist), 2]
            x_node_1, y_node_1 = self.nodes_top[np.argmin(dist), 0], self.nodes_top[np.argmin(dist), 1]
            if np.isnan(dist1):
                node1 = None
                x_node_1, y_node_1 = None, None
            dist[np.argmin(dist)] = 1e12
            dist2, node2 = np.min(dist), self.nodes_top[np.argmin(dist), 2]
            x_node_2, y_node_2 = self.nodes_top[np.argmin(dist), 0], self.nodes_top[np.argmin(dist), 1]
            if np.isnan(dist2):
                node2 = None
                x_node_2, y_node_2 = None, None

            # Create an array of linearized positions based on the nodes for all mouse positions including frame number
            x_, y_, z_ = lin_pos(x_node_1, y_node_1, x_node_2, y_node_2, x, y, z)
            self.x_top.append(x_), self.y_top.append(y_), self.z_top.append(z_)

        for [x, y, z, l] in self.dat_2:
            dist = distance(x, y, self.nodes_bottom[:, 0], self.nodes_bottom[:, 1])
            dist1, node1 = np.min(dist), self.nodes_bottom[np.argmin(dist), 2]
            x_node_1, y_node_1 = self.nodes_bottom[np.argmin(dist), 0], self.nodes_bottom[np.argmin(dist), 1]
            if np.isnan(dist1):
                node1 = None
                x_node_1, y_node_1 = None, None
            dist[np.argmin(dist)] = 1e12
            dist2, node2 = np.min(dist), self.nodes_bottom[np.argmin(dist), 2]
            x_node_2, y_node_2 = self.nodes_bottom[np.argmin(dist), 0], self.nodes_bottom[np.argmin(dist), 1]
            if np.isnan(dist2):
                node2 = None
                x_node_2, y_node_2 = None, None

            # Create an array of linearized positions based on the nodes for all mouse positions including frame number
            x_, y_, z_ = lin_pos(x_node_1, y_node_1, x_node_2, y_node_2, x, y, z)
            self.x_bot.append(x_), self.y_bot.append(y_), self.z_bot.append(z_)

        plt.figure(3)
        plt.imshow(self.im_0, cmap='gray')
        plt.plot(self.x_top, self.y_top, 'o', color='k')
        plt.plot(self.nodes_top[0:15, 0], self.nodes_top[0:15, 1], 'X', markersize=20, color='blue')
        plt.plot(self.ghost_nodes_top[:, 0], self.ghost_nodes_top[:, 1], 'o', markersize=20, color='orange')
        path = pkg_resources.resource_filename(self.pathname, '/output/Linearized_path/Linearized_path_top.png')
        plt.savefig(path)

        plt.figure(4)
        plt.imshow(self.im_1, cmap='gray')
        plt.plot(self.x_bot, self.y_bot, 'o', color='k')
        plt.plot(self.nodes_bottom[0:16, 0], self.nodes_bottom[0:16, 1], 'X', markersize=20, color='blue')
        plt.plot(self.ghost_nodes_bottom[:, 0], self.ghost_nodes_bottom[:, 1], 'o', markersize=20, color='orange')
        path = pkg_resources.resource_filename(self.pathname, '/output/Linearized_path/Linearized_path_bot.png')
        plt.savefig(path)

        logging.debug('Saving linear path images done')

    def dwell_time(self):
        logging.debug('Saving dwell time plot initiated')
        lin_dat_top = np.zeros((len(self.x_top), 2))
        lin_dat_top[:, 0], lin_dat_top[:, 1] = self.x_top, self.y_top
        lin_dat_bot = np.zeros((len(self.x_bot), 2))
        lin_dat_bot[:, 0], lin_dat_bot[:, 1] = self.x_bot, self.y_bot

        for x, y in lin_dat_top:
            dist = distance(x, y, self.nodes_top[:, 0], self.nodes_top[:, 1])
            dist, node = np.min(dist), self.nodes_top[np.argmin(dist), 2]
            if np.isnan(dist):
                node = None
            if node == 16 and x <= 625:
                node = None
            if node == 16 and y >= 226 and x <= 47:
                node = None
            self.nodelist_top.append(node)

        self.nodelist_top = [x for x in self.nodelist_top if x is not None]

        for x, y in lin_dat_bot:
            dist = distance(x, y, self.nodes_bottom[:, 0], self.nodes_bottom[:, 1])
            dist, node = np.min(dist), self.nodes_bottom[np.argmin(dist), 2]
            if np.isnan(dist):
                node = None
            if node == 16 and x >= 740 and y <= 310:
                node = None
            self.nodelist_bot.append(node)

        self.nodelist_bot = [x for x in self.nodelist_bot if x is not None]

        bins = [self.nodelist_top.count(1)/15, self.nodelist_top.count(2)/15, self.nodelist_top.count(3)/15, self.nodelist_top.count(4)/15,
                self.nodelist_top.count(5)/15, self.nodelist_top.count(6)/15 + self.nodelist_bot.count(6)/15, self.nodelist_top.count(7)/15,
                self.nodelist_top.count(8)/15, self.nodelist_top.count(9)/15, self.nodelist_top.count(10)/15, self.nodelist_top.count(11)/15,
                self.nodelist_bot.count(12)/15, self.nodelist_bot.count(13)/15, self.nodelist_top.count(14)/15, self.nodelist_bot.count(15)/15,
                self.nodelist_top.count(16)/15 + self.nodelist_bot.count(16)/15, self.nodelist_bot.count(17)/15, self.nodelist_bot.count(18)/15,
                self.nodelist_bot.count(19)/15, self.nodelist_bot.count(20)/15, self.nodelist_bot.count(21)/15, self.nodelist_bot.count(22)/15,
                self.nodelist_bot.count(23)/15, self.nodelist_bot.count(24)/15]
        y_pos = np.arange(len(bins))
        bin_names = ('node 1', 'node 2', 'node 3', 'node 4', 'node 5', 'node 6', 'node 7', 'node 8', 'node 9', 'node 10', 'node 11','node 12', 'node 13', 'node 14', 'node 15', 'node 16', 'node 17', 'node 18', 'node 19', 'node 20', 'node 21','node 22', 'node 23', 'node 24')

        fig, ax = plt.subplots(1, 1, figsize=(22, 22))
        ax.bar(y_pos, bins, width=0.8)
        ax.set_xticks(y_pos)
        ax.set_xticklabels(bin_names, fontdict=None, minor=False)
        ax.set_title("Dwell time")
        ax.set_ylabel('Seconds spent at node')
        path = pkg_resources.resource_filename(self.pathname, '/output/Dwell_time/Dwell_time.png')
        plt.savefig(path)

        logging.debug('Saving dwell time plot done')

    def ground_truth(self):
        dat_0 = self.pos_log_file_lin_0
        dat_1 = self.pos_log_file_lin_1

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.scatter(self.ref_nodes[:, 0], self.ref_nodes[:, 1])
        lines = [(self.ref_nodes[0, 0], self.ref_nodes[1, 0], self.ref_nodes[2, 0], self.ref_nodes[3, 0], self.ref_nodes[4, 0], self.ref_nodes[7, 0],
                  self.ref_nodes[9, 0], self.ref_nodes[6, 0], self.ref_nodes[8, 0], self.ref_nodes[5, 0], self.ref_nodes[16, 0],
                  self.ref_nodes[17, 0], self.ref_nodes[18, 0], self.ref_nodes[11, 0], self.ref_nodes[14, 0], self.ref_nodes[12, 0],
                  self.ref_nodes[15, 0], self.ref_nodes[23, 0], self.ref_nodes[22, 0], self.ref_nodes[21, 0], self.ref_nodes[20, 0],
                  self.ref_nodes[19, 0], self.ref_nodes[18, 0], self.ref_nodes[11, 0], self.ref_nodes[8, 0], self.ref_nodes[6, 0],
                  self.ref_nodes[9, 0], self.ref_nodes[12, 0], self.ref_nodes[14, 0], self.ref_nodes[21, 0], self.ref_nodes[22, 0],
                  self.ref_nodes[23, 0], self.ref_nodes[15, 0], self.ref_nodes[13, 0], self.ref_nodes[10, 0], self.ref_nodes[7, 0],
                  self.ref_nodes[9, 0], self.ref_nodes[6, 0], self.ref_nodes[2, 0], self.ref_nodes[1, 0], self.ref_nodes[0, 0], self.ref_nodes[5, 0]),
                 (self.ref_nodes[0, 1], self.ref_nodes[1, 1], self.ref_nodes[2, 1], self.ref_nodes[3, 1], self.ref_nodes[4, 1], self.ref_nodes[7, 1],
                  self.ref_nodes[9, 1], self.ref_nodes[6, 1], self.ref_nodes[8, 1], self.ref_nodes[5, 1], self.ref_nodes[16, 1],
                  self.ref_nodes[17, 1], self.ref_nodes[18, 1], self.ref_nodes[11, 1], self.ref_nodes[14, 1], self.ref_nodes[12, 1],
                  self.ref_nodes[15, 1], self.ref_nodes[23, 1], self.ref_nodes[22, 1], self.ref_nodes[21, 1], self.ref_nodes[20, 1],
                  self.ref_nodes[19, 1], self.ref_nodes[18, 1], self.ref_nodes[11, 1], self.ref_nodes[8, 1], self.ref_nodes[6, 1],
                  self.ref_nodes[9, 1], self.ref_nodes[12, 1], self.ref_nodes[14, 1], self.ref_nodes[21, 1], self.ref_nodes[22, 1],
                  self.ref_nodes[23, 1], self.ref_nodes[15, 1], self.ref_nodes[13, 1], self.ref_nodes[10, 1], self.ref_nodes[7, 1],
                  self.ref_nodes[9, 1], self.ref_nodes[6, 1], self.ref_nodes[2, 1], self.ref_nodes[1, 1], self.ref_nodes[0, 1], self.ref_nodes[5, 1])]
        plt.plot(*lines, 'blue')

        for i, txt in enumerate(self.ref_nodes[:, 2]):
            ax.annotate(txt, (self.ref_nodes[i, 0], self.ref_nodes[i, 1]))

        for k in range(len(dat_0)):

            if not np.isnan(dat_0[k, 5]):
                x1 = self.ref_nodes[int(dat_0[k, 5]) - 1, 0]
                y1 = self.ref_nodes[int(dat_0[k, 5]) - 1, 1]
                x2 = self.ref_nodes[int(dat_0[k, 6]) - 1, 0]
                y2 = self.ref_nodes[int(dat_0[k, 6]) - 1, 1]
                d = dat_0[k, 4] * distance(x1, y1, x2, y2)

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

                self.xp_0.append(x4)
                self.yp_0.append(y4)

        for k in range(len(dat_1)):

            if not np.isnan(dat_1[k, 5]):
                x1 = self.ref_nodes[int(dat_1[k, 5]) - 1, 0]
                y1 = self.ref_nodes[int(dat_1[k, 5]) - 1, 1]
                x2 = self.ref_nodes[int(dat_1[k, 6]) - 1, 0]
                y2 = self.ref_nodes[int(dat_1[k, 6]) - 1, 1]
                d = dat_1[k, 4] * distance(x1, y1, x2, y2)

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

                self.xp_1.append(x4)
                self.yp_1.append(y4)

        plt.plot(self.xp_0, self.yp_0, 'o', markersize='12', color='green')
        plt.plot(self.xp_1, self.yp_1, 'o', markersize='12', color='red')

        path = pkg_resources.resource_filename(self.pathname, '/output/Ground_truth_path/ground_truth_path.png')
        plt.savefig(path)
