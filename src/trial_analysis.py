import numpy as np
from matplotlib import pyplot as plt
import pkg_resources
import os
import io
import base64
import urllib
import codecs

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
                        savepath = os.path.join(path, dir)

                        lines = TrialDisplay.gt_map(self)

                        n = int(dir.replace("trial_", ""))

                        TrialDisplay.make_html(self, savepath, data_path, lines, n)

    def make_html(self, savepath, data_path, lines, n):
        data = np.genfromtxt(data_path, delimiter=',', skip_header=True)

        report = ''
        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

        report += '<B> Trial {} <B>'.format(n) + '<br>'

        fig_1 = plt.scatter(self.ref_nodes[:, 0], self.ref_nodes[:, 1])
        plt.plot(*lines, 'blue')
        plt.plot(data[:, 1], data[:, 2], 'o', color='red')
        report += '<B> Ground truth <B>' + '<br>'
        report += fig2html(fig_1) + '<br>'
        report += '<br>'

        plt.clf()

        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

    def gt_map(self):
        lines = [(self.ref_nodes[0, 0], self.ref_nodes[1, 0], self.ref_nodes[2, 0], self.ref_nodes[3, 0],
                      self.ref_nodes[4, 0], self.ref_nodes[7, 0],
                      self.ref_nodes[9, 0], self.ref_nodes[6, 0], self.ref_nodes[8, 0], self.ref_nodes[5, 0],
                      self.ref_nodes[16, 0],
                      self.ref_nodes[17, 0], self.ref_nodes[18, 0], self.ref_nodes[11, 0], self.ref_nodes[14, 0],
                      self.ref_nodes[12, 0],
                      self.ref_nodes[15, 0], self.ref_nodes[23, 0], self.ref_nodes[22, 0], self.ref_nodes[21, 0],
                      self.ref_nodes[20, 0],
                      self.ref_nodes[19, 0], self.ref_nodes[18, 0], self.ref_nodes[11, 0], self.ref_nodes[8, 0],
                      self.ref_nodes[6, 0],
                      self.ref_nodes[9, 0], self.ref_nodes[12, 0], self.ref_nodes[14, 0], self.ref_nodes[21, 0],
                      self.ref_nodes[22, 0],
                      self.ref_nodes[23, 0], self.ref_nodes[15, 0], self.ref_nodes[13, 0], self.ref_nodes[10, 0],
                      self.ref_nodes[7, 0],
                      self.ref_nodes[9, 0], self.ref_nodes[6, 0], self.ref_nodes[2, 0], self.ref_nodes[1, 0],
                      self.ref_nodes[0, 0], self.ref_nodes[5, 0]),
                     (self.ref_nodes[0, 1], self.ref_nodes[1, 1], self.ref_nodes[2, 1], self.ref_nodes[3, 1],
                      self.ref_nodes[4, 1], self.ref_nodes[7, 1],
                      self.ref_nodes[9, 1], self.ref_nodes[6, 1], self.ref_nodes[8, 1], self.ref_nodes[5, 1],
                      self.ref_nodes[16, 1],
                      self.ref_nodes[17, 1], self.ref_nodes[18, 1], self.ref_nodes[11, 1], self.ref_nodes[14, 1],
                      self.ref_nodes[12, 1],
                      self.ref_nodes[15, 1], self.ref_nodes[23, 1], self.ref_nodes[22, 1], self.ref_nodes[21, 1],
                      self.ref_nodes[20, 1],
                      self.ref_nodes[19, 1], self.ref_nodes[18, 1], self.ref_nodes[11, 1], self.ref_nodes[8, 1],
                      self.ref_nodes[6, 1],
                      self.ref_nodes[9, 1], self.ref_nodes[12, 1], self.ref_nodes[14, 1], self.ref_nodes[21, 1],
                      self.ref_nodes[22, 1],
                      self.ref_nodes[23, 1], self.ref_nodes[15, 1], self.ref_nodes[13, 1], self.ref_nodes[10, 1],
                      self.ref_nodes[7, 1],
                      self.ref_nodes[9, 1], self.ref_nodes[6, 1], self.ref_nodes[2, 1], self.ref_nodes[1, 1],
                      self.ref_nodes[0, 1], self.ref_nodes[5, 1])]

        return lines


