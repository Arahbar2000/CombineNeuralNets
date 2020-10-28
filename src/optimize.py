import numpy as np
import math
import statistics
import cv2

class Optimize:
    def __init__(self, segmentations, ensembles):
        self.segmentations = segmentations
        self.ensembles = ensembles

    def optimize(self):
        keys = list(self.ensembles.keys())
        i = np.argmax([self.anmi(self.ensembles[ensemble]['mask']) for ensemble in keys])
        return keys[i]

    def anmi(self, s):
        return statistics.mean([self.nmi(s, self.segmentations[segmentation]['mask'])
                                for segmentation in self.segmentations.keys()])

    def nmi(self, s1, s2):
        s1_labels, s1_counts = np.unique(s1, return_counts=True)
        s2_labels, s2_counts = np.unique(s2, return_counts=True)
        num = 0
        n = s1.shape[0] * s1.shape[1]
        for h, r_h in zip(s1_labels, s1_counts):
            s1_img = np.where(s1 == h, 1, 0)
            for l, r_l in zip(s2_labels, s2_counts):
                s2_img = np.where(s2 == l, 1, 0)
                r_hl = np.count_nonzero(s1_img * s2_img)
                if(r_hl == 0):
                    num += 0
                else:
                    num += r_hl * math.log(n * r_hl / r_h / r_l)
        den = np.sum([r_h * math.log(r_h / n) for r_h in s1_counts]) \
              * np.sum([r_l * math.log(r_l / n) for r_l in s2_counts])
        if den == 0:
            return 0
        return num / math.sqrt(den)
