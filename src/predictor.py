import skimage.io
import skimage.morphology
import numpy as np
def predict(class_image_data, borders=0, connectivity=0, passes=2):
    # Class image has 3 planes for background, interior, and boundaries. Each plan show the
    # probability that the pixel belongs to the class

    # Each pixel in 'pred' gets the plane index (0, 1, of 2) for which the plane value (prob) is max
    # Hence, pred is an image whose pixels have values 0 (background), 1 (interior) or 2 (boundary)
    pred = np.argmax(class_image_data, -1)

    cell_min_size = 25
    cell_label = 1  # This value corresponds to the interior class

    # 'cell' is an image that contains only interior
    cell = (pred == cell_label)

    # Remove small holes and small objects from 'cell'
    cell = skimage.morphology.remove_small_holes(cell, area_threshold=cell_min_size)
    cell = skimage.morphology.remove_small_objects(cell, min_size=cell_min_size)

    # Convert the 'cell' image into an image that contains labels (one label for each segmented nucleus)
    [interior, _] = skimage.morphology.label(cell, return_num=True, background=0)
    interior[np.where(interior != 0)] += 1
    boundary = (pred == 2)
    final = interior.copy()
    flag = True
    counter = 0
    while flag:
        counter += 1
        if passes is not None:
            if counter == passes + 1:
                break
        changes = np.zeros((256, 256))
        flag = False
        # goes through every pixel in image
        for y in range(0, 256):
            for x in range(0, 256):
                # only processes contour pixels
                if boundary[y][x]:
                    # stores different labels in neighborhood
                    labels = set()
                    # determines type of connectivity to use
                    if connectivity:
                        neighbors = neighbors_edges(y, x)
                    else:
                        neighbors = neighbors_vertices(y, x)
                    # adds each label to labels set
                    for pixel in neighbors_vertices(y, x):
                        if final[pixel] > 0:
                            flag = True
                            labels.add(final[pixel])
                    # if pixel neighbor does not have any labeled image regions, continue to next pixel
                    if not labels:
                        continue
                    # make sure we do not process this pixel again
                    boundary[y, x] = False
                    labels = list(labels)
                    # determines whether to separate borders
                    if len(labels) > 1:
                        if borders:
                            # picks a random label from the list of labels
                            changes[y, x] = labels[random.randint(0, len(labels) - 1)]
                        else:
                            changes[y, x] = 0
                    else:
                        changes[y, x] = labels[0]
        # changed pixels are stored in a temporary array
        # at the end of each pass, the we update the final image with the changes
        mask = (changes > 0)
        final[mask] = changes[mask]

    return final


def neighbors_vertices(y, x):
    calculate_neighbors = lambda y, x: [(i, j) for i in range(y - 1, y + 2)
                      for j in range(x - 1, x + 2)
                      if (0 <= x < 256 and
                          0 <= y < 256 and
                          (i != y or j != x)
                          and (0 <= i < 256)
                          and (0 <= j < 256))]
    all_neighbors = calculate_neighbors(y, x)
    all_neighbors.append((y, x))
    return all_neighbors


def neighbors_edges(y, x):
    n = lambda y, x: [(i, j) for i in range(y - 1, y + 2)
                      for j in range(x - 1, x + 2)
                      if (0 <= x < 256 and
                          0 <= y < 256 and
                          (i != y or j != x) and
                          (i != y-1 or j != x-1) and
                          (i != y+1 or j != x+1) and
                          (i != y-1 or j != x+1) and
                          (i != y+1 or j != x-1)
                          and (0 <= i < 256)
                          and (0 <= j < 256))]
    all_neighbors = n(y, x)
    all_neighbors.append((y, x))
    return all_neighbors