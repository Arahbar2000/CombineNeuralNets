import os
import argparse
import numpy as np
import pandas as pd
import skimage.io
import skimage.segmentation
import skimage.measure
import skimage.morphology
import compute_metrics as comp


class Config:
    def __init__(self, unet, mrcnn, annot, file):
        self.unet = unet
        self.mrcnn = mrcnn
        self.annot = annot
        f = open(file)
        self.files = f.readlines()
        f.close()


def parse_arguments():
    description = 'Gets the root directory'
    parser = argparse.ArgumentParser(description=description)
    required = parser.add_argument_group("required arguments")

    required.add_argument('-u', '--unet', type=str, required=True)
    required.add_argument('-m', '--mrcnn', type=str, required=True)
    required.add_argument('-a', '--annot', type=str, required=True)
    required.add_argument('-f', '--file', type=str, required=True)
    args = parser.parse_args()

    return {
        'root': args.root
    }


def process_all_images(unet_dir, mrcnn_dir, annot_dir, filenames):
    filenames = sorted(filenames)
    jaccard_results = pd.DataFrame(columns=['unet', 'mrcnn', 'ensemble'], copy=True)
    f1_results = pd.DataFrame(columns=['unet', 'mrcnn', 'ensemble'], copy=True)
    split_results = pd.DataFrame(columns=['unet', 'mrcnn', 'ensemble'], copy=True)
    merge_results = pd.DataFrame(columns=['unet', 'mrcnn', 'ensemble'], copy=True)
    for file in filenames:
        unet_path = os.path.join(unet_dir, file)
        mrcnn_path = os.path.join(mrcnn_dir, file)
        annot_path = os.path.join(annot_dir, file)
        unet = skimage.io.imread(unet, as_gray=True)
        mrcnn = skimage.io.imread(mrcnn, as_gray=True)
        annot = skimage.io.imread(annot_path, as_gray=True)
        ensemble = ensemble_segmentations(unet_path, mrcnn_path)
        unet_results = comp.get_per_image_metric(annot, unet)
        mrcnn_results = comp.get_per_image_metric(annot, mrcnn)
        ensemble_results = comp.get_per_image_metric(annot, ensemble)
        jaccard_results = jaccard_results.append({'unet': unet_results['jac'], 'mrcnn': mrcnn_results['jac'],
                                                  'ensemble': ensemble_results['jac']})
        f1_results = f1_results.append({'unet': unet_results['af1'], 'mrcnn': mrcnn_results['af1'],
                                        'ensemble': ensemble_results['af1']})
        split_results = split_results.append({'unet': unet_results['split_rate'], 'mrcnn': mrcnn_results['split_rate'],
                                              'ensemble': ensemble_results['split_rate']})
        merge_results = merge_results.append({'unet': unet_results['merge_rate'], 'mrcnn': mrcnn_results['merge_rate'],
                                              'ensemble': ensemble_results['merge_rate']})
        tables = {'jac': jaccard_results, 'f1': f1_results, 'split': split_results, 'merge': merge_results}
        output_multiple_tables(tables)

def ensemble_segmentations(unet, mrcnn):
    unet_seed, mrcnn_seed = erode_images(unet, mrcnn, 4)
    ensemble = np.where(unet + mrcnn > 0, 1, 0)
    ensemble_seed = np.where(unet_seed + mrcnn_seed > 0, 1, 0)
    labels = np.where(ensemble > 0, 0, 2)
    labels[ensemble_seed > 0] = 1
    final = skimage.segmentation.random_walker(ensemble_seed, labels)
    final[final == 1] = 1
    final[final == 2] = 0
    final = skimage.measure.label(final)
    return final


def erode_images(img1, img2, num):
    img1_seed = img1.copy()
    img2_seed = img2.copy()
    for i in range(num):
        img1_seed = skimage.morphology.erosion(img1_seed)
        img2_seed = skimage.morphology.erosion(img2_seed)
    return img1_seed, img2_seed

def output_multiple_tables(tables):
    f = open('results.txt', 'w')
    f.write('Jaccard Scores')
    f.write(tables['jac'].to_string(index=True))
    f.write('\n\n')
    f.write('F1 Scores\n')
    f.write(tables['f1'].to_string(header=True, name=True))
    f.write('\n\n')
    f.write('Split Rates\n')
    f.write(tables['split'].to_string(header=True, name=True))
    f.write('\n\n')
    f.write('Merge Rates\n')
    f.write(tables['merge'].to_string(header=True, name=True))
    f.close()

if __name__ == '__main__':
    CONFIG = Config(unet='/Users/arianrahbar/Dropbox/Unet/OutLabels/',
                    mrcnn='/Users/arianrahbar/Dropbox/Mrcnn/OutLabels/',
                    annot='/Users/arianrahbar/Dropbox/raw_annotations',
                    file='/Users/arianrahbar/Dropbox/test.txt')
    process_all_images(unet_dir=CONFIG.unet, mrcnn_dir=CONFIG.mrcnn, annot_dir=CONFIG.annot, filenames=CONFIG.file)

