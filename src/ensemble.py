import numpy as np
from skimage.measure import label
from skimage.morphology import erosion
from random_walker import random_walker
class Ensemble:
    def __init__(self, segmentations, erosions=6, beta=30):
        self.segmentations = segmentations
        self.erosions = erosions
        self.beta = beta

    def ensemble(self):
        seeds = self._erode_images()
        seed_union = np.zeros_like(seeds[0])
        orig_intersection = np.zeros_like(self.segmentations[0])
        for i in range(len(self.segmentations)):
            seed_union = seed_union + self.seeds[i]
            orig_intersection = orig_intersection + self.segmentations[i]
        labels = np.where(seed_union > 0, 1, 0)
        labels[orig_intersection == 0] = 2
        data = np.dstack([segmentation for segmentation in self.segmentations])
        final = random_walker(data, labels, beta=self.beta, mode='bf')
        final[final == 2] = 0
        return label(final)

    def _erode_images(self):
        seeds = [x.copy() for x in self.segmentations]
        for i in range(self.erosions):
            for j in range(len(self.segmentations)):
                seeds[j] = erosion(seeds[j])
        return seeds
