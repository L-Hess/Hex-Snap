import cv2
import math
import numpy as np
from collections import deque
import pkg_resources
from src.kalman import KalmanFilter
from matplotlib import pyplot as plt

# Kalman filter cannot be older than 15 frames, otherwise deemed insufficiently informed to predict mouse position
KF_REGISTRATION_AGE = 15

# Window size for the search window around the last recorded mouse position
SEARCH_WINDOW_SIZE = 60

# Kernel for Kalman filtering
KERNEL_3 = np.ones((3, 3), np.uint8)

# Saves past positions predicted by the Kalman filter
TRAIL_LENGTH = 10


# Simple calculation of the centroid of a contour
def centroid(cnt):
    """Centroid of an OpenCV contour"""
    m = cv2.moments(cnt)
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    return cx, cy


# Euclidean distance
def distance(x1, y1, x2, y2):
    """Euclidean distance."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Simple tracking of the mouse position on basis of filtering out background and finding the biggest blob in the filter
# (which would be the mouse)
class Tracker:
    def __init__(self, cfg, LED_pos, LED_thresholds, idx=0, thresh_mask=100, thresh_detect=35, pos_log_file=None,
                 name=__name__):
        super().__init__()
        self.id = idx
        self.cfg = cfg

        self.name = name

        self.thresh_detect = 255 - thresh_detect
        self.thresh_led_0 = LED_thresholds[0]
        self.thresh_led_1 = LED_thresholds[1]

        self.n_frames = 0
        self.frame = None
        self.img = None
        self.thresh_mask = thresh_mask
        self.thresh_detect = 255 - thresh_detect

        self.width = cfg['frame_width']
        self.height = cfg['frame_height']
        self.colors = cfg['frame_colors']

        self.search_point = None
        self.search_window_size = SEARCH_WINDOW_SIZE

        self.foi_frame = None
        self.mask_frame = np.zeros((self.height, self.width), np.uint8)
        self.has_mask = False

        self.pos_log_file = pos_log_file

        self.kf = KalmanFilter()
        self.kf_results = deque(maxlen=TRAIL_LENGTH)
        self.last_kf_pos = (-100, -100)
        self.kf_age = 0
        self.results = deque(maxlen=TRAIL_LENGTH)

        self.width = cfg['frame_width']
        self.height = cfg['frame_height']
        self.colors = cfg['frame_colors']

        self.cfg = cfg

        self.contours = None
        self.largest_contour = None

        self.search_point = None
        self.search_window_size = SEARCH_WINDOW_SIZE

        self.positions = []
        self.threshold = None

        self.led_frame = None
        self.led_state = None

        self.n = None
        self.id_ = None

        self.led_count = 0

        self.LED_pos = LED_pos

        self.largest_areas = np.full(20, 100)

        self.masks = np.arange(15, 10000, 15)

    # Making a mask on basis of the input frame
    def make_mask(self, frame, global_threshold=70):
        """Create a new mask on basis of the input frame"""
        _, mask = cv2.threshold(frame, global_threshold, 255, cv2.THRESH_BINARY)
        self.mask_frame = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_3)
        self.has_mask = True

    # Find a window in which the mouse is expected to be next, this is done in order to speed up the tracking process
    def get_search_window(self):
        """Calculate the bounding box around a search point, i.e. last animal position or
         Kalman filter prediction."""
        x1 = min(max(0, self.search_point[0] - self.search_window_size // 2), self.width)
        y1 = min(max(0, self.search_point[1] - self.search_window_size // 2), self.height)

        x2 = min(max(0, self.search_point[0] + self.search_window_size // 2), self.width)
        y2 = min(max(0, self.search_point[1] + self.search_window_size // 2), self.height)
        return (x1, y1), (x2, y2)

    # Find the mouse in the frame (if present) and locate its position
    def apply(self, frame, idx, n, src, mask_frame=None):
        """Tracking of the mouse position on basis of masking and a kalman filter"""

        self.id_ = idx
        mask_check = mask_frame
        self.n = n

        cx, cy = None, None

        f_start = self.id * self.height
        f_end = (self.id + 1) * self.height
        self.frame = frame[f_start:f_end, :]

        # On the first frame, save mask
        if self.id_ == 0:
            path = pkg_resources.resource_filename(self.name, "/data/raw/{}/frame_images/frame_{}.png".
                                                   format(src[len(src)-29:len(src)-10], n))
            cv2.imwrite(path, self.frame)

        # Check if a mask is already present, if not, create a new mask
        if not self.has_mask:
            foi = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
            if np.mean(foi) > 15:
                self.make_mask(cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY), global_threshold=self.thresh_mask)

        # if self.id_ in self.masks:
        #     self.has_mask = False

        if mask_check is not None:
            self.mask_frame = mask_check
            self.has_mask = True

        # cut out search window from image and from mask, if needed
        if self.search_point is None:
            foi = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
            mask = self.mask_frame // 255
            foi_ofs = (0, 0)
        else:
            p1, p2 = self.get_search_window()
            foi = cv2.cvtColor(self.frame[p1[1]:p2[1], p1[0]:p2[0]], cv2.COLOR_RGB2GRAY)
            foi_ofs = p1
            mask = self.mask_frame[p1[1]:p2[1], p1[0]:p2[0]] // 255

        # On the first frame, save mask
        if self.id_ == 0:
            path = pkg_resources.resource_filename(self.name, "/data/raw/{}/masks/mask_{}.png"
                                                   .format(src[len(src)-29:len(src)-10], n))
            cv2.imwrite(path, self.mask_frame)

        # Apply mask to frame
        masked = cv2.bitwise_not(foi) * mask
        masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, KERNEL_3)

        _, thresh = cv2.threshold(masked, self.thresh_detect, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, KERNEL_3)
        self.threshold = self.mask_frame

        # Find the largest contour in the frame on basis of an earlier defined threshold
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # minimum of the mouse area, if largest area is below, it is not tracked
        # minimum mouse area is adaptive to the mouse areas already logged in frames before, this greatly
        # enhances tracking when mouse is behind wall or cue because in those cases the kalman filter takes over
        min_mouse_area = np.mean(self.largest_areas[len(self.largest_areas)-10:])/2

        largest_cnt, largest_area = None, 0
        sum_area = 0
        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area > min_mouse_area:
                sum_area += area

                # Only uses the largest area if larger than the minimum mouse area
                # (filters out other stuff being tracked)
                if area > largest_area:
                    largest_area = area
                    largest_cnt = cnt

                    # Only save the largest area if not larger than 2 times the current minimum mouse area
                    # Prevents weird tracking from screwing up the minimum mouse area
                    # (for example the experimenter being tracked)
                    if largest_area < 2*min_mouse_area:
                        self.largest_areas = np.append(self.largest_areas, largest_area)

        # Correct coordinates for search window location
        if largest_cnt is not None:
            largest_cnt[:, :, 0] = largest_cnt[:, :, 0] + foi_ofs[0]
            largest_cnt[:, :, 1] = largest_cnt[:, :, 1] + foi_ofs[1]

        self.contours = contours
        self.largest_contour = largest_cnt

        # If a largest contour is present (mouse is in the maze), find its centroid
        if largest_cnt is None:
            self.kf_age += 1
            self.results.appendleft(None)
            self.search_window_size = min(int(self.search_window_size * 1.5), max(self.width, self.height * 2))

        else:
            # center coordinates of contour
            self.search_window_size = max(SEARCH_WINDOW_SIZE, int(self.search_window_size * .75))
            cx, cy = centroid(largest_cnt)

            # Save centroid location in results
            self.results.appendleft((cx, cy))
            self.kf.correct(cx, cy)
            self.kf_age = 0

            self.positions.append((cx, cy))

        # Kalman filter of position
        # Only predict position if the age of the last measurement is low enough
        # Else assume KF has no useful information about mouse position either.
        if self.kf_age < KF_REGISTRATION_AGE:
            kf_res = self.kf.predict()
            kfx = min(max(0, int(kf_res[0])), self.width)
            kfy = min(max(0, int(kf_res[1])), self.height)

            self.kf_results.appendleft((kfx, kfy))

            self.last_kf_pos = (kfx, kfy)
            self.search_point = self.last_kf_pos

        else:
            self.search_point = None
            self.last_kf_pos = None
            self.kf_results.appendleft(None)

            kfx = None
            kfy = None

        # Use LED position as earlier calculated using homography and stddev filtering to monitor its state (on or off)
        if self.n == 0:
            self.led_frame = self.frame[int((self.LED_pos[1]-2)):
                                        int((self.LED_pos[1]+2)), int((self.LED_pos[0]-2)):int((self.LED_pos[0]+2)), 0]
            self.led_state = np.mean(self.led_frame) > self.thresh_led_0

        if self.n == 1:
            self.led_frame = self.frame[int((self.LED_pos[3]-2)):
                                        int((self.LED_pos[3]+2)), int((self.LED_pos[2]-2)):int((self.LED_pos[2]+2)), 0]
            self.led_state = np.mean(self.led_frame) > self.thresh_led_1

        # Logs LED state to be on if mean is larger than threshold
        if self.led_state:
            self.led_state = 1
        else:
            self.led_state = 0

        # If no largest area is found, take the positions the kalman filter spits out
        if not cx and not cy and kfx and kfy:
            cx = kfx
            cy = kfy

        # If the positions fall outside of the frame (due to kalman filter) ignore these
        if cx and cy:
            if cx <= 0 or cx >= 800 or cy <= 0 or cy >= 600:
                cx = None
                cy = None

        # Save mouse position and LED state in log file
        if largest_cnt is None:
            if self.pos_log_file is not None:
                self.pos_log_file.write('None, None, {}, {}\n'.format(self.id_, self.led_state))
        else:
            if self.pos_log_file is not None:
                self.pos_log_file.write('{}, {}, {}, {}\n'.format(cx, cy, self.id_, self.led_state))

    def close(self):
        """Close the position log file, must be performed in order to use file for further processing"""
        self.pos_log_file.close()
