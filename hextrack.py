import cv2
import time
import yaml
import argparse
import logging
import pkg_resources
from pathlib import Path
import threading
import os

from src.tracker import Tracker
from src.Analysis import Display
from src.preprocessing import Timecorrect
from src.preprocessing import Linearization
from src.preprocessing import Homography

# If true, no tracking is performed, can only be used if pos_log_files are already available in the system
ONLY_ANALYSIS = True


# Grab frames and return captured frame
class Grabber:
    def __init__(self, src):
        self.src = src
        self.capture = cv2.VideoCapture(src)

    def next(self):
        rt, frame = self.capture.read()
        return frame


# Loops through frames, capturing them and applying tracking
class OfflineHextrack:
    def __init__(self, cfg, src, n, LED_pos):
        threading.current_thread().name = 'HexTrack'

        self.cfg = cfg
        self.frame_idx = 0
        self.n = n
        self.mask_init = True
        self.made_mask = None

        # Create path to csv log file for tracking mouse position and LED-light state
        if n == 0:
            path = pkg_resources.resource_filename(__name__, "/data/interim/position_log_files/{}".format(src[57:76]))
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed, this path probably already exists" % path)
        self.path = pkg_resources.resource_filename(__name__, '/data/interim/Position_log_files/{}/pos_log_file_{}.csv'
                                                    .format(src[57:76], n))

        # Initiation of the Grabbers and Trackers and creation of csv log file
        self.grabber = Grabber(src)
        self.tracker = Tracker(cfg, pos_log_file=open(self.path, 'w'), name=__name__, LED_pos=LED_pos)

        logging.debug('HexTrack initialization done!')

    # Loops through grabbing and tracking each frame of the video file
    def loop(self):
        # i = 0
        # while i < 2000:
        #     i += 1
        while True:
            frame = self.grabber.next()
            if frame is None:
                break

            # Checks if the frame has a mask already, if not, it creates a new mask
            if self.mask_init:
                self.tracker.apply(frame, self.frame_idx, n=self.n)
            elif not self.mask_init:
                self.tracker.apply(frame, self.frame_idx, mask_frame=self.made_mask, n=self.n)

            # At the second frame, show computer-generated mask
            # If not sufficient, gives possibility to input user-generated mask
            # if self.frame_idx == 1:
            #     path = pkg_resources.resource_filename(__name__, '/output/Masks/mask_{}.png'.format(n))
            #     mask = cv2.imread(path)
            #     plt.figure('Mask check')
            #     plt.imshow(mask)
            #     plt.show()
            #     mask_check = input("If the mask is sufficient, enter y: ")
            #     if mask_check != 'y':
            #         input('Please upload custom mask under the name new_mask.png to the output folder and press enter')
            #         self.made_mask = cv2.imread('new_mask.png', 0)
            #         self.mask_init = False
            self.frame_idx += 1
        self.tracker.close()

    # Redundant, might be deleted later
    def process_events(self, display=False):
        if not display:
            return

        # Event loop call
        key = cv2.waitKey(1)

        # Process Keypress Events
        if key == ord('q'):
            self.stop()

    def stop(self):
        self.tracker.close()
        cv2.destroyAllWindows()
        raise SystemExit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sources', nargs='*', help='List of sources to read from')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    parser.add_argument('-c', '--config', help='Configuration file')
    parser.add_argument('-n', '--nodes', help='Node location file')

    cli_args = parser.parse_args()

    logfile = Path.home() / "Videos/hextrack/{}_hextrack_log".format(
        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())))

    if cli_args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - (%(threadName)-9s) %(message)s')

    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - (%(threadName)-9s) %(message)s')

    fh = logging.FileHandler(str(logfile))
    fhf = logging.Formatter('%(asctime)s : %(levelname)s : [%(threadName)-9s] - %(message)s')
    fh.setFormatter(fhf)
    logging.getLogger('').addHandler(fh)

    # Construct the shared array to fit all frames
    cfg_path = pkg_resources.resource_filename(__name__, '/src/resources/default/default_config.yml')
    if cli_args.config is not None:
        cfg_path = Path(cli_args.config)
        if not cfg_path.exists():
            raise FileNotFoundError('Config file not found!')

    with open(cfg_path, 'r') as cfg_f:
        cfg = yaml.load(cfg_f)

    if cli_args.sources is not None:
        cfg['frame_sources'] = cli_args.sources

    # Initiate calculation of the homography matrix, directly corrects all node and LED positions
    homography = Homography(__name__, sources=cfg['frame_sources'])
    homography.homography_calc()
    LED_pos = homography.LEDfind(sources=cfg['frame_sources'], iterations=200)

    # Initiates OfflineHextrack to track mouse positions and save position log files
    for n, src in enumerate(cfg['frame_sources']):
        print('Source {} @ {} starting'.format(n, src))

        if not ONLY_ANALYSIS:
            ht = OfflineHextrack(cfg=cfg, src=src, n=n, LED_pos=LED_pos)
            ht.loop()

    logging.debug('Position files acquired')

    # Option to skip time correction and linearization
    key = input('Enter y for time correction and linearization: ')
    if key == 'y':
        tcorrect = Timecorrect(__name__, sources=cfg['frame_sources'])
        tcorrect.correction()
        linearization = Linearization(__name__, sources=cfg['frame_sources'])
        linearization.lin()
    else:
        pass
    #
    # # Option to skip analysis
    # key = input('Enter y for analysis: ')
    #
    # if key == 'y':
    #     display = Display(__name__)
    #     display.path_display()
    #     display.lin_path_display()
    #     display.dwell_time()
    #     display.ground_truth()
    # if key != 'y':
    #     pass
