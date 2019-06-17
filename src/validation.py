import numpy as np
import matplotlib.pyplot as plt


def distance(x1, y1, x2, y2):
    """Euclidean distance"""
    r = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return r


class Validate:
    def __init__(self, path_0, path_1):
        self.dat_0 = np.genfromtxt(path_0, delimiter=',', skip_header=True)
        self.dat_1 = np.genfromtxt(path_1, delimiter=',', skip_header=True)

    def time_alignment_check(self):

        led_0 = self.dat_0[:, 3]
        led_1 = self.dat_1[:, 3]

        led_0_peaks = np.diff(led_0) < 0
        led_1_peaks = np.diff(led_1) < 0

        np.savetxt(r'D:\led_0.txt', led_0)
        np.savetxt(r'D:\led_1.txt', led_1)
        i_0 = np.where(led_0_peaks == 1)[0]
        i_1 = np.where(led_1_peaks == 1)[0]
        i_diff = i_0 - i_1

        return i_diff

    def gt_distance_check(self):
        dist = []
        for k in range(len(self.dat_0)):
            if not np.isnan(self.dat_0[k, 0]) and not np.isnan(self.dat_1[k, 0]):
                dist.append(distance(self.dat_0[k, 7], self.dat_0[k, 8], self.dat_1[k, 7], self.dat_1[k, 8]))
            else:
                dist.append(np.nan)

        return dist




