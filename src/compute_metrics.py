import numpy as np


def get_per_image_metrics(ground_truth, prediction, verbose):
    """ Compute performance metrics using a ground truth image for segmented nuclei
    and the corresponding predicted image. Both ground-truth and predicted images
    represent labeled segments. That is, each nuclei segment has a numeric label,
    which is an integer number greater than 1. The background is always set to 0.

    Returns:
        A dictionary with the Average F1, Jaccard index, split rate, and merge rate.
        The dictionary uses keys af1, jac, spr, and mgr to define these values.
     """

    # Compute IoU
    iou_array = intersection_over_union(ground_truth=ground_truth, prediction=prediction, verbose=verbose)

    # Check if M (number of objects in ground truth is non-zero. If yes, computes the Jaccard index
    if iou_array.shape[0] > 0 and iou_array.shape[1] > 0:
        jaccard = np.amax(iou_array, axis=0).mean()
    else:
        jaccard = 0.0

    # Calculate F1 score at all thresholds
    th_values = np.arange(0.5, 0.95, 0.05)
    average_f1 = 0
    for t in th_values:
        f1, tp, fp, fn = measures_at(t, iou_array)
        average_f1 += f1
        if verbose:
            print("     th: {:.2f}, f1: {:.6f}, tp: {}, fp: {}, fn: {}".format(t, f1, tp, fp, fn))

    average_f1 = average_f1 / len(th_values)

    sm_dict = count_splits_and_merges(iou_array)

    if verbose:
        print("AF1: {}   JAC: {}   SPR: {}  MGR: {}".format(average_f1, jaccard,
                                                            sm_dict["split_rate"], sm_dict["merge_rate"]))

    return {
        "jac": jaccard,
        "af1": average_f1,
        "split_rate": sm_dict["split_rate"],
        "merge_rate": sm_dict["merge_rate"]
    }


def measures_at(threshold, iou_array):
    """Compute the F1 value, the number of true positives, the number of false positives,
    and the number of false negatives. It uses the IoU array and a threshold that defines
    hits and misses.

    Returns:
        A list of values corresponding to the F1 coefficient, the true positives, the
        false positives, and the true negatives.
    """
    matches = iou_array > threshold

    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects

    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))

    true_pos, false_pos, false_neg = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-9)

    return f1, true_pos, false_pos, false_neg


def intersection_over_union(ground_truth, prediction, verbose=False):
    """ Compute the IoU matrix using the ground truth and the predicted image.

    Returns:
        A numpy array of size MxN where M is the number of nuclei in the ground truth and N the number
    of nuclei in the predicted image. The values of the array represent IoU values on a per object basis.
    """

    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))

    if verbose:
        print("ground truth objs: {}, predicted objs: {}".format(true_objects, pred_objects))

    # Compute intersection
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects, pred_objects))
    intersection = h[0]

    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]

    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]

    # Compute Intersection over Union
    union[union == 0] = 1e-9
    iou_array = intersection / union

    if verbose:
        print("Shape of IoU array {}".format(iou_array.shape))
    return iou_array


def count_splits_and_merges(iou_array):
    """ Count the number of splits and merges detected in the IoU array.

    The IoU array is a matrix of size MxN where M is the number of annotated nuclei and
    N is the number of predicted nuclei.

    Returns:
        A dictionary with the number of splits and merges
    """
    num_annot_nuclei, num_pred_nuclei = iou_array.shape

    if num_annot_nuclei == 0 or num_pred_nuclei == 0:
        return {
            "merge_rate": 0,
            "split_rate": 0
        }

    match_threshold = 0.1

    matches = iou_array > match_threshold
    merges = np.sum(matches, axis=0) > 1
    splits = np.sum(matches, axis=1) > 1

    return {
        "merge_rate": np.sum(merges) / num_pred_nuclei,
        "split_rate": np.sum(splits) / num_annot_nuclei
    }