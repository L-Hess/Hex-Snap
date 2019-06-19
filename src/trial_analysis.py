import numpy as np
from matplotlib import pyplot as plt
import pkg_resources
import os
import io
import base64
import urllib
import codecs
import networkx as nx
from scipy.signal import savgol_filter

from src.validation import Validate
from scipy.signal import savgol_filter


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


def smooth(array):

    N = int(len(array)/10)

    for k in range(N):
        new_array = np.zeros_like(array)
        for i in range(len(array)):
            if i == 0:
                new_array[i] = np.mean(array[i:i+1])
            elif i == len(array):
                new_array[i] = np.mean(array[i-1:i])
            else:
                new_array[i] = np.mean(array[i-1:i+1])
        array = new_array

    return array




class TrialDisplay:
    def __init__(self, pathname, paths):
        self.path_vid_0 = paths[0]
        self.path_vid_1 = paths[1]
        self.log_path = paths[2]
        self.pathname = pathname

        self.ref_nodes_path = pkg_resources.resource_filename(pathname, '/src/Resources/default/ref_nodes.csv')
        self.ref_nodes = np.genfromtxt(self.ref_nodes_path, delimiter=',', skip_header=True)

        flower_graph, node_positions = TrialDisplay.gt_map(self)

        max_dist_log, mean_dist_log = [], []

        path = pkg_resources.resource_filename(self.pathname, "/data/processed/{}".format(self.path_vid_0[len(self.path_vid_0)-29:len(self.path_vid_0)-10]))
        if os.path.exists(path):
            for dirs, subdirs, files in os.walk(path):
                for dir in subdirs:
                    if not dir.endswith("position_log_files") and not dir.endswith("ground_truth"):
                        try:
                            summary_log
                        except UnboundLocalError:
                            summary_log = np.zeros((2, len(subdirs)))
                        path_0 = os.path.join(path, dir, 'position_log_files', 'pos_log_file_0.csv')
                        path_1 = os.path.join(path, dir, 'position_log_files', 'pos_log_file_1.csv')
                        savepath = os.path.join(path, dir)
                        validate = Validate(path_0, path_1)
                        time_diff = validate.time_alignment_check()
                        dist_diff = validate.gt_distance_check()
                        cleaned_dist = [x for x in dist_diff if str(x) != 'nan']
                        max_dist, mean_dist = np.nan, np.nan
                        if cleaned_dist:
                            max_dist = max(cleaned_dist)
                            mean_dist = np.mean(cleaned_dist)
                            max_dist_log.append(max_dist)
                            mean_dist_log.append(mean_dist)

                        data_path = os.path.join(path, dir, 'position_log_files', 'pos_log_file.csv')
                        self.data = np.genfromtxt(data_path, delimiter=',', skip_header=True)

                        path_log, path_length, shortest_path_length = TrialDisplay.path_metrics(self, flower_graph,
                                                                                                node_positions)
                        dwell_data = TrialDisplay.dwell_times(self)

                        velocities = TrialDisplay.velocity(self)

                        n = int(dir.replace("trial_", ""))

                        TrialDisplay.make_html_tracking(self, savepath, time_diff, dist_diff, max_dist, mean_dist, n)
                        TrialDisplay.make_html_analysis(self, savepath, flower_graph, node_positions, n, path_log,
                                                        path_length, shortest_path_length, dwell_data, velocities)

        TrialDisplay.make_summary_html_tracking(self, pathname, path, max_dist_log, mean_dist_log)


    def make_html_tracking(self, savepath, time_align, gt_dist, max_dist, mean_dist, n):

        report = ''
        with open('{}/TRACKING_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

        max_time = max(time_align)
        mean_time = np.mean(time_align)

        report += '<B> Trial {} <B>'.format(n) + '<br>'

        fig_1 = plt.plot(time_align)
        report += '<B> Time alignment <B>' + '<br>'
        report += fig2html(fig_1) + '<br>'
        report += '<i> Maximum time dilation between sources = {} <i>'.format(max_time) + '<br>'
        report += '<i> Average time dilation between sources = {} <i>'.format(mean_time) + '<br>'

        plt.clf()

        if not np.isnan(max_dist):
            fig_2 = plt.plot(gt_dist, 'o')
            plt.axis([0, len(gt_dist), 0, max_dist+int(max_dist/5)])
            report += '<B> Ground truth distance between sources <B>' + '<br>'
            report += fig2html(fig_2) + '<br>'
            report += '<i> Maximum distance between sources = {} <i>'.format(max_dist) + '<br>'
            report += '<i> Average distance between sources = {} <i>'.format(mean_dist) + '<br>'

            plt.clf()

        else:
            report += '<B> Ground truth distance between sources <B>' + '<br>' + '<br>'
            report += '<i> No frame with mouse present in both sources <i>' + '<br>'

        with open('{}/TRACKING_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

    def make_summary_html_tracking(self, pathname, path, max_dist_log, mean_dist_log):

        max_dist = max(max_dist_log)
        mean_dist = np.mean(mean_dist_log)

        origin_path = pkg_resources.resource_filename(pathname, "/data/interim/pos_log_files_gt/{}".format(self.path_vid_0[len(self.path_vid_0)-29:len(self.path_vid_0)-10]))
        val = Validate(origin_path + '/pos_log_file_gt_0.csv', origin_path +'/pos_log_file_gt_1.csv')
        dist_series = val.gt_distance_check()

        cleaned_dist = [x for x in dist_series if str(x) != 'nan']
        max_dist_series, mean_dist_series = np.nan, np.nan
        if cleaned_dist:
            max_dist_series = max(cleaned_dist)
            mean_dist_series = np.mean(cleaned_dist)

        report = ''
        with open('{}/TRACKING_REPORT_SUMMARY.html'.format(path), 'w') as rf:
            rf.write(report)
            report = ''

        if not np.isnan(max_dist):
            fig_2 = plt.plot(dist_series, 'o')
            plt.axis([0, len(dist_series), 0, max_dist+int(max_dist_series/5)])
            report += '<B> Ground truth distance between sources <B>' + '<br>'
            report += fig2html(fig_2) + '<br>'
            report += '<br>'
            report += '<i> Maximum distance between sources across whole video = {} <i>'.format(max_dist_series) + '<br>'
            report += '<i> Average distance between sources across whole video = {} <i>'.format(mean_dist_series) + '<br>'
            report += '<br>'
            report += '<i> Maximum distance between sources across all trials = {} <i>'.format(max_dist) + '<br>'
            report += '<i> Average distance between sources across all trials = {} <i>'.format(mean_dist) + '<br>'
            report += '<br>'

            plt.clf()

        if os.path.exists(path):
            for dirs, subdirs, files in os.walk(path):
                for file in files:
                        if file.endswith('TRACKING_REPORT.html'):
                            filepath = os.path.join(path, dirs, file)
                            if os.path.exists(filepath):
                                with open(filepath) as f:
                                    html = f.readlines()[0]
                                    report += html
                                    report += '<br>'

        with open('{}/TRACKING_REPORT_SUMMARY.html'.format(path), 'w') as rf:
            rf.write(report)
            report = ''

    def make_html_analysis(self, savepath, flower_graph, node_positions, n, path_log, path_length, shortest_path_length, dwell_data, velocities):

        report = ''
        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

        report += '<B> Trial {} <B>'.format(n) + '<br>'

        mg = nx.Graph(flower_graph)
        fig_1 = nx.draw_networkx(mg, pos=node_positions, nodecolor='r', edge_color='b', alpha=1, font_size=10)

        plt.scatter(self.data[:, 1], self.data[:, 2], color='blue')

        report += '<B> Ground truth <B>' + '<br>'
        report += fig2html(fig_1) + '<br>'
        report += '<br>'
        report += '<i> Path taken: {} <i>'.format(path_log) + '<br>'
        report += '<i> Path length: {} <i>'.format(path_length) + '<br>'
        report += '<i> Shortest possible path length: {} <i>'.format(shortest_path_length) + '<br>'
        report += '<br>'

        plt.clf()

        fig_2 = plt.plot(velocities)

        report += '<B> Velocities <B>' + '<br>'
        report += fig2html(fig_2) + '<br>'
        report += '<br>'

        plt.clf()

        report += '<B> Dwell times (frames) <B>' + '<br>'

        nodes = dwell_data[0]
        times = dwell_data[1]

        for i in range(len(nodes)):
            report += '<br>'
            report += 'Node: {}, Dwell time: {} frames'.format(nodes[i], times[i]) + '<br>'
            report += '<br>'

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
                path_log.append(int(closest_nodes[i]))
            else:

                if path_log[len(path_log)-1] != closest_nodes[i]:
                    path_log.append(int(closest_nodes[i]))

        mg = nx.Graph(flower_graph)
        nx.spring_layout(mg, pos=node_positions)

        path_length = len(path_log)
        shortest_path_length = len(nx.shortest_path(mg, path_log[0], path_log[len(path_log)-1]))

        return path_log, path_length, shortest_path_length

    def dwell_times(self):

        closest_nodes = []

        for _, x, y in self.data:

            # Calculate the distance of each mouse position to all nodes
            dist = distance(x, y, self.ref_nodes[:, 0],self.ref_nodes[:, 1])

            # Finds the closest node and saves its position and node number
            dist1, closest_node = np.min(dist), self.ref_nodes[np.argmin(dist), 2]
            if np.isnan(dist1):
                closest_node = np.nan

            closest_nodes.append(closest_node)

        closest_nodes = [x for x in closest_nodes if str(x) != 'nan']

        array, counts = np.unique(closest_nodes, return_counts=True)

        return [array, counts]

    def velocity(self):

        velocities = []

        for i in range(1, len(self.data)):

            x, y = self.data[i, 1], self.data[i, 2]
            x_prev, y_prev = self.data[i-1, 1], self.data[i-1, 2]

            if x and y and x_prev and y_prev:
                d = distance(x, y, x_prev, y_prev)

                velocities.append(d)

            else:
                velocities.append(0)

        velocities = smooth(velocities)

        return velocities
