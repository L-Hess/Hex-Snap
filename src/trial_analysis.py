import numpy as np
from matplotlib import pyplot as plt
import pkg_resources
import os
import io
import base64
import urllib
import networkx as nx
import pandas as pd
from src.validation import Validate


# Euclidean distance
def distance(x1, y1, x2, y2):
    """Euclidean distance."""

    r = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return r


def fig2html(fig):
    """Convert a figure to html format"""

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    string = base64.b64encode(buf.read())

    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "{}"/>'.format(uri)

    return html


def smooth(array):
    """Smoothing of an array"""

    n = int(len(array)/10)

    for k in range(n):
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


class TrialAnalysis:
    def __init__(self, pathname, paths):

        # Load in all data and pathnames for the relevant files
        self.path_vid_0 = paths[0]
        self.path_vid_1 = paths[1]
        self.log_path = paths[2]
        self.pathname = pathname

        # Empty lists to store trial data in for later conversion to excel file
        self.paths = []
        self.path_lengths = []
        self.shortest_path_lengths = []
        self.velocities = []
        self.correct_path = []
        self.number = []

        self.data = None
        self.fps = None

        # Load in the reference node positions of the ground truth map
        self.ref_nodes_path = pkg_resources.resource_filename(pathname, '/src/Resources/default/ref_nodes.csv')
        self.ref_nodes = np.genfromtxt(self.ref_nodes_path, delimiter=',', skip_header=True)

        # Create the ground truth map that can be used as input for a networkx graph
        flower_graph, node_positions = TrialAnalysis.gt_map(self)

        # Emtpy lists to store the maximum and minimum distances between the two video sources of trials in
        max_dist_log, mean_dist_log = [], []

        # Root directory of the trial data
        path = pkg_resources.resource_filename(self.pathname, "/data/processed/{}"
                                               .format(self.path_vid_0[len(self.path_vid_0)-29:
                                                                       len(self.path_vid_0)-10]))

        # Finding the relevant part of the input log file for the experiment
        self.df = pd.read_excel(self.log_path)
        start_time = 3600*int(path[len(path)-8:len(path)-6]) + 60*int(path[len(path)-5:
                                                                           len(path)-3]) + int(path[len(path)-2:])
        trial_starts = self.df['timestamp_start_trial'].astype('str')
        trial_starts = np.array([3600*int(x[11:13]) + 60*int(x[14:16]) + int(x[17:19]) - start_time for x in
                                 trial_starts])

        stamps = np.nonzero(trial_starts > 0)

        self.drop_number_top = stamps[0][0]
        self.df.drop(self.df.head(self.drop_number_top).index, inplace=True)

        # Loops through all trial data and appends relevant information to before-mentioned lists
        def dir_loop(dir_count, dirs, path):
            """Loops through data of all trials and appends analysis data to relevant list"""

            if 'trial_{}'.format(dir_count) in dirs:
                try:
                    path_0 = os.path.join(path, 'trial_{}'.format(dir_count), 'position_log_files', 'pos_log_file_0.csv')
                    path_1 = os.path.join(path, 'trial_{}'.format(dir_count), 'position_log_files', 'pos_log_file_1.csv')
                    savepath = os.path.join(path, 'trial_{}'.format(dir_count))
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

                    data_path = os.path.join(path, 'trial_{}'.format(dir_count), 'position_log_files', 'pos_log_file.csv')
                    self.data = np.genfromtxt(data_path, delimiter=',', skip_header=True)

                    # Trial number
                    n = int(dir_count)

                    # List indication number
                    number = n - 1

                    self.fps = None

                    trial_time = self.df['timestamp_end_trial'][number + self.drop_number_top] \
                                 - self.df['timestamp_start_trial'][number + self.drop_number_top]
                    trial_time_s = trial_time.total_seconds()

                    try:
                        self.fps = len(self.data) / trial_time_s

                    except ZeroDivisionError:
                        print('Error: The total time of trial {} of video {} was 0 seconds, as this is not possible,'
                              ' please check the trial times in the log file.\n'
                              ' Right now dwell time analysis and velocity analysis have not been performed correctly'
                              .format(dir_count, self.path_vid_0[len(self.path_vid_0)-29:len(self.path_vid_0)-10]))
                        pass

                    # Calculate all path metrics for a trial
                    path_log, path_length, shortest_path_length, correct_path = \
                        TrialAnalysis.path_metrics(self, flower_graph, node_positions, number)
                    # Calculate dwell times for all nodes for a trial
                    array, counts, counts_string = TrialAnalysis.dwell_times(self)

                    # Calculate the velocities of all frames of a trial
                    velocities, velocity_average = TrialAnalysis.velocity(self)

                    # Append all calculated data of a trial to the relevant list
                    self.paths.append(path_log)
                    self.path_lengths.append(path_length)
                    self.shortest_path_lengths.append(shortest_path_length)
                    self.correct_path.append(correct_path)
                    self.velocities.append(velocity_average)

                    if dir_count == 1:
                        self.dwell_data = counts
                    else:
                        self.dwell_data = np.vstack((self.dwell_data, counts))

                    # TrialAnalysis.make_html_tracking(self, savepath, time_diff, dist_diff, max_dist, mean_dist, n)
                    TrialAnalysis.make_html_analysis(self, savepath, flower_graph, node_positions, n, path_log,
                                                     path_length, shortest_path_length, counts, array, velocities)

                    dir_count += 1

                    dir_loop(dir_count, dirs, path)

                # If an OSError is present, the trial does not exist anymore, thus only loops until gone through all
                #  data
                except OSError:
                    pass

        # Begin at trial 1
        dir_count = 1
        dirs = os.listdir(path)

        # Loop through all trials
        dir_loop(dir_count, dirs, path)

        # Create a new excel file containing all analysis data
        TrialAnalysis.data_log(self, path)

    def make_html_tracking(self, savepath, time_align, gt_dist, max_dist, mean_dist, n):
        """Create a HTML file of the tracking data of a trial"""

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

    def make_summary_html_tracking(self, pathname, path, max_dist_log, mean_dist_log):
        """Creates a HTML file containing a summary overview of all trials"""

        max_dist = max(max_dist_log)
        mean_dist = np.mean(mean_dist_log)

        origin_path = pkg_resources.resource_filename(pathname, "/data/interim/pos_log_files_gt/{}"
                                                      .format(self.path_vid_0[len(self.path_vid_0)-29:
                                                                              len(self.path_vid_0)-10]))
        val = Validate(origin_path + '/pos_log_file_gt_0.csv', origin_path + '/pos_log_file_gt_1.csv')
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
            report += '<i> Maximum distance between sources across whole video = {} <i>'.format(max_dist_series)\
                      + '<br>'
            report += '<i> Average distance between sources across whole video = {} <i>'.format(mean_dist_series)\
                      + '<br>'
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

    def make_html_analysis(self, savepath, flower_graph, node_positions, n, path_log, path_length,
                           shortest_path_length, dwell_data, dwell_array, velocities):
        """Create a HTML file containing the relevant analysis information of a trial"""

        report = ''
        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

        report += '<B> Trial {} <B>'. format(n)
        report += '<br>'

        fig_1, ax = plt.subplots(1,1, figsize=(5, 5))
        # Create the graph of the ground truth map, together with the logged positions
        mg = nx.Graph(flower_graph)
        nx.draw_networkx(mg, pos=node_positions, nodecolor='r', edge_color='b', alpha=1, font_size=10)
        plt.scatter(self.data[:, 1], self.data[:, 2], color='red')
        plt.title('Ground truth path')

        report += fig2html(fig_1)
        report += '<br>'

        plt.clf()

        # Create a matplotlib subplots object to save velocity and dwell time plots in
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # Create the velocity plot
        ax[0].plot(velocities)
        ax[0].set_title('Velocity plot')
        ax[0].set_xlabel('Frame number')
        ax[0].set_ylabel('Velocity (m/s)')

        # Create the dwell time plot
        ax[1].bar(dwell_array, dwell_data)
        ax[1].set_xticks(dwell_array)
        ax[1].set_xlabel('Node')
        ax[1].set_ylabel('Dwell time (s)')

        report += fig2html(fig)
        report += '<br>'

        plt.clf()

        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)

    def gt_map(self):
        """Creates a standard flower graph of the Hex-Maze (ground truth) and find the positions of all nodes"""

        # Standard flower graph of the Hex-Maze containing all nodes and connections between them
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

        # Create a list with node positions for all nodes of the flower graph
        node_positions = {}
        with open(self.ref_nodes_path, 'r') as npf:
            next(npf)
            for line in npf:
                x, y, nn = map(str.strip, line.split(','))
                node_positions[int(nn)] = (int(float(x)), int(float(y)))

        return flower_graph, node_positions

    def path_metrics(self, flower_graph, node_positions, n):
        """Calculate the path taken (tracked), path length, shortest path length and gives the path as filed
         by the experimenter together with if this is in correspondence with the tracked path of a trial"""

        start = self.df['start_location'].iloc[int(n)]
        goal = self.df['goal_location'].iloc[int(n)]
        gt_path = self.df['Path'].iloc[int(n)].split(',')

        closest_nodes = [int(start)]

        for _, x, y, _ in self.data:

            # Calculate the distance of each mouse position to all nodes
            dist = distance(x, y, self.ref_nodes[:, 0], self.ref_nodes[:, 1])

            # Finds the closest node and saves its position and node number
            dist1, closest_node = np.min(dist), self.ref_nodes[np.argmin(dist), 2]
            if np.isnan(dist1):
                pass
            elif dist1 < 100:
                if closest_nodes == [int(start)] and int(closest_node) == int(goal):
                    pass
                else:
                    # print(closest_node, start)
                    closest_nodes.append(str(int(closest_node)))

        closest_nodes = [x for x in closest_nodes if str(x) != 'nan']

        # Find the path taken by the mouse during the trial
        path_log = []
        path_log_str = ''
        for i in range(len(closest_nodes)):
            if i == 0:
                path_log.append(str(closest_nodes[i]))
                path_log_str += str(closest_nodes[i])+','
            else:
                if path_log[len(path_log)-1] != closest_nodes[i]:
                    path_log.append(str(int(closest_nodes[i])))
                    path_log_str += str(int(closest_nodes[i]))+','

        path_log_str = path_log_str[:len(path_log_str)-1]

        # Find the shortest path possible during the trial and the path length of the path taken
        mg = nx.Graph(flower_graph)
        nx.spring_layout(mg, pos=node_positions)

        path_length = len(gt_path)
        shortest_path_length = len(nx.shortest_path(mg, start, goal))

        df = pd.read_excel(self.log_path)
        path = df['Path'].iloc[int(n)]

        # Checks if the tracker finds the same path taken as the experimenters data
        if path == path_log_str:
            correct_path = True
        else:
            correct_path = False

        return path_log_str, path_length, shortest_path_length, correct_path

    def dwell_times(self):
        """Calculates the dwell times in seconds for all node positions of a trial"""

        closest_nodes = []

        for _, x, y, _ in self.data:

            # Calculate the distance of each mouse position to all nodes
            dist = distance(x, y, self.ref_nodes[:, 0],self.ref_nodes[:, 1])

            # Finds the closest node and saves its position and node number
            dist1, closest_node = np.min(dist), self.ref_nodes[np.argmin(dist), 2]
            if np.isnan(dist1):
                closest_node = np.nan

            closest_nodes.append(closest_node)

        closest_nodes = [x for x in closest_nodes if str(x) != 'nan']

        # In order to get full overview of counts for all nodes, otherwise nodes not visited will not show up
        closest_nodes = np.append(closest_nodes, np.arange(1, 25, 1))

        array, counts = np.unique(closest_nodes, return_counts=True)
        # Correct for added nodes
        counts -= 1

        if self.fps:
            counts = counts/self.fps

        # Convert counts to string for saving to excel sheet
        counts_string = ','.join(map(str, counts))

        return array, counts, counts_string

    def velocity(self):
        """Calculates the velocities for each frame (in m/s) of a trial"""

        velocity_average = None
        velocities_log = []
        velocities = []

        for i in range(1, len(self.data)):

            x, y = self.data[i, 1], self.data[i, 2]
            x_prev, y_prev = self.data[i-1, 1], self.data[i-1, 2]

            if not np.isnan(x) and not np.isnan(y) and not np.isnan(x_prev) and not np.isnan(y_prev):
                d = distance(x, y, x_prev, y_prev)
                velocity = d

                if self.fps:
                    velocity = d*self.fps/1000

                velocities.append(velocity)
                velocities_log.append(velocity)

            else:
                velocities_log.append(0)

        velocity_average = np.mean(velocities)
        velocities = smooth(velocities_log)

        return velocities, velocity_average

    def data_log(self, path):
        """Creates an excel file with all relevant data for the entire experiment"""

        savepath = os.path.join(path, 'trial_data_{}.xlsx'.format(path[len(path)-19:]))

        rows, _ = self.df.shape
        n = rows-len(self.paths)
        self.df.drop(self.df.tail(n).index, inplace=True)

        # Save all analysis data in the new excel file
        self.df['tracked_path'] = self.paths
        self.df['Tracked path correct?'] = self.correct_path
        self.df['path length'] = self.path_lengths
        self.df['shortest_path_length'] = self.shortest_path_lengths
        self.df['average velocity (m/s)'] = self.velocities

        df_dwell = pd.DataFrame()
        df_dwell['Trial'] = np.arange(0, len(self.df['tracked_path']), 1)
        for i in range(24):
            df_dwell['Node {} (s)'.format(i+1)] = self.dwell_data[:, i]

        writer = pd.ExcelWriter(savepath, engine='xlsxwriter')
        self.df.to_excel(writer, sheet_name='Analysis')
        df_dwell.to_excel(writer, sheet_name='Dwell_times')
        writer.save()
