import numpy as np
from matplotlib import pyplot as plt
import pkg_resources
import os
import io
import base64
import urllib
import networkx as nx

from src.validation import Validate


# Euclidean distance
def distance(x1, y1, x2, y2):
    r = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return r


def fig2html(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    string = base64.b64encode(buf.read())

    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "{}"/>'.format(uri)

    return html


class TrialDisplay:
    def __init__(self, pathname, paths):
        self.path_vid_0 = paths[0]
        self.path_vid_1 = paths[1]
        self.log_path = paths[2]
        self.pathname = pathname

        self.ref_nodes_path = pkg_resources.resource_filename(pathname, '/src/Resources/default/ref_nodes.csv')
        self.ref_nodes = np.genfromtxt(self.ref_nodes_path, delimiter=',', skip_header=True)

        max_dist_log, mean_dist_log = [], []

        flower_graph, node_positions = TrialDisplay.gt_map(self)

        path = pkg_resources.resource_filename(self.pathname, "/data/processed/{}".format(self.path_vid_0[len(self.path_vid_0)-29:len(self.path_vid_0)-10]))
        if os.path.exists(path):
            for dirs, subdirs, files in os.walk(path):
                for dir in subdirs:
                    if not dir.endswith("position_log_files") and not dir.endswith("ground_truth"):
                        try:
                            summary_log
                        except UnboundLocalError:
                            summary_log = np.zeros((2, len(subdirs)))
                        data_path = os.path.join(path, dir, 'position_log_files', 'pos_log_file.csv')
                        self.data = np.genfromtxt(data_path, delimiter=',', skip_header=True)
                        savepath = os.path.join(path, dir)

                        path_log, path_length, shortest_path_length = TrialDisplay.path_metrics(self, flower_graph, node_positions)

                        n = int(dir.replace("trial_", ""))

                        TrialDisplay.make_html(self, savepath, flower_graph, node_positions, n, path_log, path_length, shortest_path_length)

    def make_html(self, savepath, flower_graph, node_positions, n, path_log, path_length, shortest_path_length):

        report = ''
        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

        report += '<B> Trial {} <B>'.format(n) + '<br>'

        mg = nx.Graph(flower_graph)
        fig_1 = nx.draw_networkx(mg, pos=node_positions, nodecolor='r', edge_color='b', alpha=1, font_size=10)

        plt.scatter(self.data[:, 1], self.data[:, 2], color='red')

        report += '<B> Ground truth <B>' + '<br>'
        report += fig2html(fig_1) + '<br>'
        report += '<br>'
        report += '<i> Path taken: {} <i>'.format(path_log) + '<br>'
        report += '<i> Path length: {} <i>'.format(path_length) + '<br>'
        report += '<i> Shortest possible path length: {} <i>'.format(shortest_path_length) + '<br>'
        report += '<br>'

        plt.clf()

        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

    def gt_map(self):
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
        with open(self.ref_nodes_path, 'r') as npf:
            next(npf)
            for line in npf:
                x, y, nn = map(str.strip, line.split(','))
                node_positions[int(nn)] = (int(float(x)), int(float(y)))

        return flower_graph, node_positions

    def path_metrics(self, flower_graph, node_positions):

        closest_nodes = []

        for _, x, y in self.data:

            # Calculate the distance of each mouse position to all nodes
            dist = distance(x, y, self.ref_nodes[:, 0],self.ref_nodes[:, 1])

            # Finds the closest node and saves its position and node number
            dist1, closest_node = np.min(dist), self.ref_nodes[np.argmin(dist), 2]
            if np.isnan(dist1):
                closest_node = np.nan

            # Find the second closest node and save its position and node number
            dist[np.argmin(dist)] = 1e12
            dist2, second_closest_node = np.min(dist), self.ref_nodes[np.argmin(dist), 2]
            if np.isnan(dist2):
                second_closest_node = np.nan

            closest_nodes.append(closest_node)

        closest_nodes = [x for x in closest_nodes if str(x) != 'nan']

        path_log = []
        for i in range(len(closest_nodes)):
            if i == 0:
                path_log.append(closest_nodes[i])
            else:

                if path_log[len(path_log)-1] != closest_nodes[i]:
                    path_log.append(int(closest_nodes[i]))

        mg = nx.Graph(flower_graph)
        nx.spring_layout(mg, pos=node_positions)

        path_length = len(path_log)
        shortest_path_length = len(nx.shortest_path(mg, path_log[0], path_log[len(path_log)-1]))

        return path_log, path_length, shortest_path_length

