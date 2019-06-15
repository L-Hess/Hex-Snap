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

                        n = int(dir.replace("trial_", ""))

                        # TrialDisplay.make_html(self, savepath, time_diff, dist_diff, max_dist, mean_dist, n)

        self.counts = TrialDisplay.make_summary_html(self, pathname, path, max_dist_log, mean_dist_log)

    def counting(self):

        counts = self.counts

        return counts

    def make_html(self, savepath, time_align, gt_dist, max_dist, mean_dist, n):

        report = ''
        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

        max_time = max(time_align)
        mean_time = np.mean(time_align)

        report += '<B> Trial {} <B>'.format(n) + '<br>'

        fig_1 = plt.plot(time_align)
        plt.title('Time dilation after time alignment across whole video')
        plt.xlabel('Index of LED-onset')
        plt.ylabel('Frame difference between top and bottom sources')
        report += '<B> Time alignment <B>' + '<br>'
        report += fig2html(fig_1) + '<br>'
        report += '<i> Maximum time dilation between sources = {} <i>'.format(max_time) + '<br>'
        report += '<i> Average time dilation between sources = {} <i>'.format(mean_time) + '<br>'

        plt.clf()

        fig_1 = plt.plot(time_align)
        plt.title('Time dilation after time alignment across whole video')
        plt.xlabel('Index of LED-onset')
        plt.ylabel('Frame difference between top and bottom sources')
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

        with open('{}/ANALYSIS_REPORT.html'.format(savepath), 'w') as rf:
            rf.write(report)
            report = ''

    def make_summary_html(self, pathname, path, max_dist_log, mean_dist_log):

        max_dist = max(max_dist_log)
        mean_dist = np.mean(mean_dist_log)

        origin_path = pkg_resources.resource_filename(pathname, "/data/interim/pos_log_files_gt/{}".format(self.path_vid_0[len(self.path_vid_0)-29:len(self.path_vid_0)-10]))
        val = Validate(origin_path + '/pos_log_file_gt_0.csv', origin_path +'/pos_log_file_gt_1.csv')
        time_series = val.time_alignment_check()
        dist_series = val.gt_distance_check()

        max_time_series = max(time_series)
        mean_time_series = np.mean(time_series)
        _, counts = np.unique([0,1,2,3,1,1,1,1,1], return_counts=True)

        cleaned_dist = [x for x in dist_series if str(x) != 'nan']
        max_dist_series, mean_dist_series = np.nan, np.nan
        if cleaned_dist:
            max_dist_series = max(cleaned_dist)
            mean_dist_series = np.mean(cleaned_dist)

        report = ''
        with open('{}/ANALYSIS_REPORT_SUMMARY.html'.format(path), 'w') as rf:
            rf.write(report)
            report = ''

        report += '<B> Summary <B>' + '<br>'

        fig_1, ax = plt.subplots(1, 2, figsize=(20, 5))

        ## Entire video
        p = (0)
        ax[p].plot(time_series, color='k')
        ax[p].set_title('Time dilation after alignment on LED offsets across entire video')
        ax[p].set_xlabel('Index of LED-onset')
        ax[p].set_ylabel('Frame difference between top and bottom sources')
        ax[p].set_ylim(-3, 3)
        ax[p].set_xlim(0, len(time_series))

        ## Zoom in
        p = (1)
        ax[p].plot(time_series[:300], color='k')
        ax[p].set_title('Time dilation after alignment on LED offsets for first 300 LED onsets')
        ax[p].set_xlabel('Index of LED-onset')
        ax[p].set_ylabel('Frame difference between top and bottom sources')
        ax[p].set_ylim(-3, 3)
        ax[p].set_xlim(0, 300)
        report += '<B> Time alignment whole video <B>' + '<br>'
        report += fig2html(fig_1) + '<br>'
        report += '<br>'
        report += '<i> Maximum time dilation between sources = {} <i>'.format(max_time_series) + '<br>'
        report += '<i> Average time dilation between sources = {} <i>'.format(mean_time_series) + '<br>'
        report += '<br>'

        plt.clf()

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
                        if file.endswith('ANALYSIS_REPORT.html'):
                            filepath = os.path.join(path, dirs, file)
                            if os.path.exists(filepath):
                                with open(filepath) as f:
                                    html = f.readlines()[0]
                                    report += html
                                    report += '<br>'

        with open('{}/ANALYSIS_REPORT_SUMMARY.html'.format(path), 'w') as rf:
            rf.write(report)
            report = ''

        return counts



