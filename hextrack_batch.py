import cv2
import time
import yaml
import argparse
import logging
import pkg_resources
from pathlib import Path

import os
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import numpy as np
import glob
from matplotlib import pyplot as plt

from src.tracker import Tracker
from src.preprocessing import TimeCorrect
from src.preprocessing import Linearization
from src.preprocessing import GroundTruth
from src.preprocessing import Homography
from src.preprocessing import TrialCut
from src.trial_analysis import TrialAnalysis

# If true, no tracking is performed, can only be used if pos_log_files are already available in the system
ONLY_ANALYSIS = True

# If True, mask is checked, handy to put on False for quick processing and mask does not matter much
Mask_check = False


def find_nearest(array, value):
    """"Finds the nearest log file in time (only if timestamp before video timestamp) among all log files in a list"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    val = array[idx]
    if val > value:
        array[idx] = 0
        val = find_nearest(array, value)
    return val


# Grab frames and return captured frame
class Grabber:
    def __init__(self, src):
        """"Grabs new frames and passes them on"""
        self.src = src
        self.capture = cv2.VideoCapture(src)

    def next(self):
        rt, frame = self.capture.read()
        return frame


# Loops through frames, capturing them and applying tracking
class OfflineHextrack:
    def __init__(self, cfg, src, n, LED_pos, LED_thresholds, sources):

        # Video frame sources (top and bottom)
        self.sources = sources

        self.cfg = cfg
        self.frame_idx = 0
        self.n = n
        self.mask_init = True
        self.made_mask = None

        # Create path to csv log file for tracking mouse position and LED-light state
        path = pkg_resources.resource_filename(__name__, "/data/interim/position_log_files/{}".format(src[len(src)-29:
                                                                                                          len(src)-10]))
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed, this path probably already exists" % path)
        self.path = pkg_resources.resource_filename(__name__, '/data/interim/Position_log_files/{}/pos_log_file_{}.csv'
                                                    .format(src[len(src)-29:len(src)-10], n))

        # Initiation of the Grabbers and Trackers and creation of csv log file
        self.grabber = Grabber(src)
        self.tracker = Tracker(cfg, pos_log_file=open(self.path, 'w'), name=__name__, LED_pos=LED_pos,
                               LED_thresholds=LED_thresholds)

        logging.debug('HexTrack initialization done!')

        # Video reader used to infer amount of frames
        self.vid = VideoFileClip(src)
        self.duration = self.vid.duration*15
        self.src = src

    # Loops through grabbing and tracking each frame of the video file
    def loop(self):
        """"Loop through all frames in video and track mouse positions"""

        # tqdm package used to monitor tracking progress
        pbar = tqdm(range(int(self.duration)))
        for i in pbar:

            # Grab next frame, stops loop if no new frame is present (happens when all frames in video tracked)
            frame = self.grabber.next()
            if frame is None:
                break

            # Checks if the frame has a mask already, if not, it creates a new mask
            if self.mask_init:
                self.tracker.apply(frame, self.frame_idx, n=self.n, src=self.src)
            elif not self.mask_init:
                self.tracker.apply(frame, self.frame_idx, mask_frame=self.made_mask, n=self.n, src=self.src)

            if Mask_check:

                # At the second frame, show computer-generated mask
                # If not sufficient, gives possibility to input user-generated mask
                if self.frame_idx == 0:
                    path = pkg_resources.resource_filename(__name__, "/data/raw/{}/Masks/mask_{}.png"
                                                           .format(self.sources[0][len(self.sources[0])-29:
                                                                                   len(self.sources[0])-10], n))
                    mask = cv2.imread(path)
                    plt.figure('Mask check')
                    plt.imshow(mask)
                    plt.show()
                    mask_check = input("If the mask is sufficient, enter y: ")
                    if mask_check != 'y':
                        input('Please upload custom mask under the name new_mask.png to the output folder'
                              ' and press enter')
                        mask_path = pkg_resources.resource_filename(__name__, "/Input_mask/new_mask.png")
                        self.made_mask = cv2.imread(mask_path, 0)
                        self.mask_init = False

            self.frame_idx += 1

        # Close down tracker position log file, tqdm progress bar and video reader
        self.tracker.close()
        pbar.close()
        self.vid.reader.close()

    def stop(self):
        """Closes the position log files for following steps to be used"""

        self.tracker.close()
        cv2.destroyAllWindows()
        raise SystemExit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', nargs='*', help='Map containing sources')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    parser.add_argument('-c', '--config', help='Configuration file')
    parser.add_argument('-n', '--nodes', help='Node location file')

    cli_args = parser.parse_args()

    logfile = Path.home() / "Videos/hextrack/{}_hextrack_log".format(
        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())))

    # Construct the shared array to fit all frames
    cfg_path = pkg_resources.resource_filename(__name__, '/src/resources/default/default_config_batch.yml')
    if cli_args.config is not None:
        cfg_path = Path(cli_args.config)
        if not cfg_path.exists():
            raise FileNotFoundError('Config file not found!')

    with open(cfg_path, 'r') as cfg_f:
        cfg = yaml.load(cfg_f, Loader=yaml.FullLoader)

    if cli_args.source is not None:
        cfg['video_map'] = cli_args.source

    rootdir = cfg['video_map'][0]
    log = None

    # Find videos in map and tracks them
    for _, _, files in os.walk(rootdir):
        for file in files:
            if file.endswith("0.avi"):
                logs = []
                logs_time = []
                path_0 = os.path.join(rootdir, file)

                # Finds the path towards the top video source and bottom video source
                # Note: if bottom video source 1 s late, failsafe in place
                path_1 = path_0[:len(path_0) - 5] + "1.avi"
                if not os.path.exists(path_1):
                    path_1 = path_0[:len(path_0)-11] + "{}_cam_1.avi".format(int(path_0[len(path_0)-11])+1)
                time = 31536000 * int(path_0[len(path_0) - 29:len(path_0) - 25]) + 2592000 * \
                       int(path_0[len(path_0) - 24:len(path_0) - 22]) + 86400 * int(
                    path_0[len(path_0) - 21:len(path_0) - 19]) + \
                       3600 * int(path_0[len(path_0) - 18:len(path_0) - 16]) + 60 * int(
                    path_0[len(path_0) - 15:len(path_0) - 13]) \
                       + int(path_0[len(path_0) - 12:len(path_0) - 10])
                sources = [path_0, path_1]

                # Scans through files and finds correct log file within map for each video
                # (also works for multiple log files present, always finds log file closest before video time)
                for file in files:
                    if file.endswith('log.xlsx'):
                        logs.append(file)
                        try:
                            logs_time.append(
                                31536000 * int(file[0:4]) + 2592000 * int(file[5:7]) + 86400 * int(file[8:10]) + 3600
                                * int(file[11:13]) + 60 * int(file[14:16]) + int(file[17:19]))
                        except ValueError:
                            print('Error: This file was read in: {}, if starts with ~$, this is a temporary file,'
                                  ' close the excel file next time before starting the pipeline.\n'
                                  ' Analysis will proceed as normal.'.format(file))

                try:
                    log_time = find_nearest(logs_time, time)
                    log_time_y = int(np.floor(log_time / 31536000))
                    log_time_month = int(np.floor((log_time - log_time_y * 31536000) / 2592000))
                    log_time_d = int(np.floor((log_time - log_time_y * 31536000 - log_time_month * 2592000) / 86400))
                    log_time_h = int(np.floor((log_time - log_time_y * 31536000 - log_time_month * 2592000 - log_time_d
                                               * 86400) / 3600))
                    log_time_m = int(np.floor((log_time - log_time_y * 31536000 - log_time_month * 2592000 - log_time_d
                                               * 86400 - log_time_h * 3600) / 60))
                    log_time_s = int(np.floor((log_time - log_time_y * 31536000 - log_time_month * 2592000 - log_time_d
                                               * 86400 - log_time_h * 3600 - log_time_m * 60)))
                    for name in glob.glob(
                            '{}/{}_{}*{}*{}*log.xlsx'.format(rootdir, path_0[len(path_0) - 29:len(path_0) - 19],
                                                             format(log_time_h, '02d'), format(log_time_m, '02d'),
                                                             format(log_time_s, '02d'))):
                        log = name
                except ValueError:
                    print('Error:Log file is probably not present in designated folder')

                # Save video paths and log path
                paths = [path_0, path_1, log]

                try:
                    # Initiate calculation of the homography matrix, directly corrects all node and LED positions
                    homography = Homography(__name__, sources=sources)
                    homography.homography_calc()
                    # Initiates OfflineHextrack to track mouse positions and save position log files
                    for n, src in enumerate(sources):
                        print('Source {} @ {} starting'.format(n, src))

                        if not ONLY_ANALYSIS:
                            # Calculate LED position and LED threshold
                            LED_pos = homography.LEDfind(sources=sources, iterations=200)
                            LED_thresholds = homography.LED_thresh(sources=sources, iterations=50, LED_pos=LED_pos)

                            # Track mouse position for entire video
                            ht = OfflineHextrack(cfg=cfg, src=src, n=n, LED_pos=LED_pos, LED_thresholds=LED_thresholds,
                                                 sources=sources)
                            ht.loop()

                            logging.debug('Position files acquired')

                    # Time alignment
                    tcorrect = TimeCorrect(__name__, sources=sources)
                    dat_0, dat_1 = tcorrect.correction()

                    # Linearization
                    linearization = Linearization(__name__, dat_0, dat_1, sources=sources)
                    lin_path_0, lin_path_1 = linearization.lin()

                    # Map to ground truth (relative position)
                    groundtruth = GroundTruth(__name__, lin_path_0, lin_path_1, sources=sources)
                    gt_path_0, gt_path_1 = groundtruth.gt_mapping()
                    gt_path = groundtruth.gt_stitch()

                    # Trial selection and cutout
                    trialcut = TrialCut(paths, [gt_path_0, gt_path_1, gt_path], sources)
                    trialcut.log_data()
                    trialcut.cut(__name__)
                    trialcut.cut_stitch(__name__)

                    # Trial analysis
                    TrialAnalysis(__name__, paths)

                except cv2.error or OSError:
                    print('Error: Something is wrong with the video file; process is continued without analysis of'
                          ' this particular video')
