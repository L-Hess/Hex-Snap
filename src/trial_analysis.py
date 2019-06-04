import numpy as np
from matplotlib import pyplot as plt
import pkg_resources
import os
import io
import base64
import urllib

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

                        n = int(dir.replace("trial_", ""))

                        TrialDisplay.make_html(self, savepath, time_diff, dist_diff)
                        TrialDisplay.make_summary_html(self, path, n, time_diff, dist_diff)


    def make_html(self, savepath, time_align, gt_dist):
        report = ''
        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

        fig_1 = plt.plot(time_align)
        report += '<B> Time alignment <B>' + '<br>'
        report += fig2html(fig_1) + '<br>'

        plt.clf()

        cleaned_dist = [x for x in gt_dist if str(x) != 'nan']
        if cleaned_dist:
            max_dist = max(cleaned_dist)
            mean_dist = np.mean(cleaned_dist)

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

        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

    def make_summary_html(self, savepath, trial_n, time_align, gt_dist):
        report = ''
        with open('{}/SUMMARY_ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''




