import numpy as np
from matplotlib import pyplot as plt
import pkg_resources
import os
import io
import base64
import urllib


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

        path = pkg_resources.resource_filename(self.pathname, "/data/processed/{}".format(self.path_vid_0[len(self.path_vid_0)-29:len(self.path_vid_0)-10]))
        if os.path.exists(path):
            for dirs, subdirs, files in os.walk(path):
                for dir in subdirs:
                    if not dir.endswith("position_log_files") and not dir.endswith("ground_truth"):
                        path_0 = os.path.join(path, dir, 'position_log_files', 'pos_log_file_0.csv')
                        path_1 = os.path.join(path, dir, 'position_log_files', 'pos_log_file_1.csv')
                        savepath = os.path.join(path,dir)
                        gt = TrialDisplay.ground_truth_path(self, path_0=path_0, path_1=path_1, savepath=savepath)
                        path_length, shortest_path_length = TrialDisplay.path_data(self, path_0=path_0, path_1=path_1, savepath=savepath)
                        TrialDisplay.make_html(self, savepath, gt)

    def ground_truth_path(self, path_0, path_1, savepath):
        dat_0 = np.genfromtxt(path_0, delimiter=',', skip_header=True)
        dat_1 = np.genfromtxt(path_1, delimiter=',', skip_header=True)

        ref_nodes = np.genfromtxt(pkg_resources.resource_filename(self.pathname, "/src/Resources/default/ref_nodes.csv"), delimiter=',', skip_header=True)

        fig, ax = plt.subplots(figsize = (5,5))
        ax.scatter(ref_nodes[:,0], ref_nodes[:,1])
        lines = [(ref_nodes[0,0],ref_nodes[1,0], ref_nodes[2,0],ref_nodes[3,0],ref_nodes[4,0],ref_nodes[7,0],ref_nodes[9,0],ref_nodes[6,0],ref_nodes[8,0],ref_nodes[5,0],ref_nodes[16,0],ref_nodes[17,0],ref_nodes[18,0],ref_nodes[11,0],ref_nodes[14,0],ref_nodes[12,0],ref_nodes[15,0],ref_nodes[23,0],ref_nodes[22,0],ref_nodes[21,0],ref_nodes[20,0],ref_nodes[19,0],ref_nodes[18,0],ref_nodes[11,0],ref_nodes[8,0],ref_nodes[6,0],ref_nodes[9,0],ref_nodes[12,0],ref_nodes[14,0],ref_nodes[21,0],ref_nodes[22,0],ref_nodes[23,0],ref_nodes[15,0],ref_nodes[13,0],ref_nodes[10,0],ref_nodes[7,0],ref_nodes[9,0],ref_nodes[6,0],ref_nodes[2,0],ref_nodes[1,0],ref_nodes[0,0],ref_nodes[5,0]), (ref_nodes[0,1], ref_nodes[1,1], ref_nodes[2,1],ref_nodes[3,1],ref_nodes[4,1],ref_nodes[7,1],ref_nodes[9,1],ref_nodes[6,1],ref_nodes[8,1],ref_nodes[5,1],ref_nodes[16,1],ref_nodes[17,1],ref_nodes[18,1],ref_nodes[11,1],ref_nodes[14,1],ref_nodes[12,1],ref_nodes[15,1],ref_nodes[23,1],ref_nodes[22,1],ref_nodes[21,1],ref_nodes[20,1],ref_nodes[19,1],ref_nodes[18,1],ref_nodes[11,1],ref_nodes[8,1],ref_nodes[6,1],ref_nodes[9,1],ref_nodes[12,1],ref_nodes[14,1],ref_nodes[21,1],ref_nodes[22,1],ref_nodes[23,1],ref_nodes[15,1],ref_nodes[13,1],ref_nodes[10,1],ref_nodes[7,1],ref_nodes[9,1],ref_nodes[6,1],ref_nodes[2,1],ref_nodes[1,1],ref_nodes[0,1],ref_nodes[5,1])]
        plt.plot(*lines, 'blue')

        for i, txt in enumerate(ref_nodes[:, 2]):
            ax.annotate(txt, (ref_nodes[i, 0], ref_nodes[i, 1]))

        for k in range(len(dat_0)):

            if not np.isnan(dat_0[k, 5]):
                x1 = ref_nodes[int(dat_0[k, 5]) - 1, 0]
                y1 = ref_nodes[int(dat_0[k, 5]) - 1, 1]
                x2 = ref_nodes[int(dat_0[k, 6]) - 1, 0]
                y2 = ref_nodes[int(dat_0[k, 6]) - 1, 1]
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

                plt.plot(x4, y4, 'o', color='green', markersize='5')

        for k in range(len(dat_1)):

            if not np.isnan(dat_1[k, 5]):
                x1 = ref_nodes[int(dat_1[k, 5]) - 1, 0]
                y1 = ref_nodes[int(dat_1[k, 5]) - 1, 1]
                x2 = ref_nodes[int(dat_1[k, 6]) - 1, 0]
                y2 = ref_nodes[int(dat_1[k, 6]) - 1, 1]
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

                plt.plot(x4, y4, 'o', color='red', markersize='5')
        return fig

    def path_data(self, path_0, path_1, savepath):
        dat_0 = np.genfromtxt(path_0, delimiter=',', skip_header=True)
        dat_1 = np.genfromtxt(path_1, delimiter=',', skip_header=True)

        path_length = None
        shortest_path_length = None

        nodes_dat_0 = dat_0[:, 5]
        nodes_dat_1 = dat_1[:, 5]

        k=0
        l=0
        for i in range(len(nodes_dat_0)):
            if nodes_dat_0[i] == nodes_dat_1[i]:
                k+=1
            elif np.isnan(nodes_dat_0[i]) and not np.isnan(nodes_dat_1[i]):
                k+=1
            elif np.isnan(nodes_dat_1[i]) and not np.isnan(nodes_dat_0[i]):
                k+=1
            if np.isnan(nodes_dat_1[i]) and np.isnan(nodes_dat_0[i]):
                l +=1

        return path_length, shortest_path_length

    def make_html(self, savepath, ground_truth):
        report = ''
        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

        fig_1 = ground_truth
        report += '<B> Test <B>' + '<br>'
        report += fig2html(fig_1)

        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''
