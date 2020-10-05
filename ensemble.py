import os
import argparse
import numpy as np
import pandas as pd
import skimage.io
import skimage.segmentation
import skimage.measure
import skimage.morphology
import cv2
import compute_metrics as comp
import matplotlib.pyplot as plt
import predictor as pred
import shutil
import seaborn as sns
from random_walker import random_walker
import cProfile


class Config:
    def __init__(self, unet, mrcnn, annot, file, instance=False):
        self.unet = unet
        self.mrcnn = mrcnn
        self.annot = annot
        f = open('/Users/arianrahbar/Dropbox/' + file)
        self.files = f.readlines()
        self.instance = instance
        self.filename = file.split('.')[0]
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


def process_all_images(config):
    filenames = sorted(config.files)
    jaccard_results = pd.DataFrame(columns=['unet', 'mrcnn', 'walker_label', 'walker_binary', 'union'], copy=True)
    f1_results = pd.DataFrame(columns=['unet', 'mrcnn', 'walker_label', 'walker_binary', 'union'], copy=True)
    split_results = pd.DataFrame(columns=['unet', 'mrcnn', 'walker_label', 'walker_binary', 'union'], copy=True)
    merge_results = pd.DataFrame(columns=['unet', 'mrcnn', 'walker_label', 'walker_binary', 'union'], copy=True)
    os.makedirs(config.filename, exist_ok=True)
    if config.instance:
        root_dir = os.path.join(config.filename, 'instance')
    else:
        root_dir = os.path.join(config.filename, 'images')
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    counter = 0
    for file in filenames:
        if counter % 50 == 0:
            print(counter)
        unet_path = os.path.join(config.unet, file.strip())
        mrcnn_path = os.path.join(config.mrcnn, file.strip())
        annot_path = os.path.join(config.annot, file.strip())
        unet = skimage.io.imread(unet_path, as_gray=True)
        mrcnn = skimage.io.imread(mrcnn_path, as_gray=True)
        annot = skimage.io.imread(annot_path, as_gray=True)
        ensemble_walker_label = ensemble_segmentations_label(unet, mrcnn, erosions=5)
        ensemble_walker_binary = ensemble_segmentations_binary(unet, mrcnn, erosions=5)
        ensemble_union = ensemble_segmentations_union(unet, mrcnn)
        unet_results = comp.get_per_image_metrics(annot, unet, False)
        mrcnn_results = comp.get_per_image_metrics(annot, mrcnn, False)
        ensemble_walker_label_results = comp.get_per_image_metrics(annot, ensemble_walker_label, False)
        ensemble_walker_binary_results = comp.get_per_image_metrics(annot, ensemble_walker_binary, False)
        ensemble_union_results = comp.get_per_image_metrics(annot, ensemble_union, False)

        jaccard_results = jaccard_results.append({'unet': unet_results['jac'], 'mrcnn': mrcnn_results['jac'],
                                                  'walker_label': ensemble_walker_label_results['jac'],
                                                  'walker_binary': ensemble_walker_binary_results['jac'],
                                                  'union': ensemble_union_results['jac']}, ignore_index=True)
        f1_results = f1_results.append({'unet': unet_results['af1'], 'mrcnn': mrcnn_results['af1'],
                                        'walker_label': ensemble_walker_label_results['af1'],
                                        'walker_binary': ensemble_walker_binary_results['af1'],
                                        'union': ensemble_union_results['af1']}, ignore_index=True)
        split_results = split_results.append({'unet': unet_results['split_rate'], 'mrcnn': mrcnn_results['split_rate'],
                                              'walker_label': ensemble_walker_label_results['split_rate'],
                                              'walker_binary': ensemble_walker_binary_results['split_rate'],
                                              'union': ensemble_union_results['split_rate']}, ignore_index=True)
        merge_results = merge_results.append({'unet': unet_results['merge_rate'], 'mrcnn': mrcnn_results['merge_rate'],
                                              'walker_label': ensemble_walker_label_results['merge_rate'],
                                              'walker_binary': ensemble_walker_binary_results['merge_rate'],
                                              'union': ensemble_union_results['merge_rate']}, ignore_index=True)

        unet_mask = np.where(unet > 0, 255, 0)
        mrcnn_mask = np.where(mrcnn > 0, 255, 0)
        ensemble_walker_label_mask = np.where(ensemble_walker_label > 0, 255, 0)
        ensemble_walker_binary_mask = np.where(ensemble_walker_binary > 0, 255, 0)
        ensemble_union_mask = np.where(ensemble_union > 0, 255, 0)
        annot_mask = np.where(annot > 0, 255, 0)
        file_dir = os.path.join(root_dir, file)
        os.makedirs(file_dir, exist_ok=True)
        cv2.imwrite(os.path.join(file_dir, 'unet.png'), unet_mask)
        cv2.imwrite(os.path.join(file_dir, 'mrcnn.png'), mrcnn_mask)
        cv2.imwrite(os.path.join(file_dir, 'ensemble_walker_label.png'), ensemble_walker_label_mask)
        cv2.imwrite(os.path.join(file_dir, 'ensemble_walker_binary.png'), ensemble_walker_binary_mask)
        cv2.imwrite(os.path.join(file_dir, 'ensemble_union.png'), ensemble_union_mask)
        cv2.imwrite(os.path.join(file_dir, 'annot.png'), annot_mask)
        counter += 1



    tables = {'jac': jaccard_results, 'f1': f1_results, 'split': split_results, 'merge': merge_results}
    os.makedirs(os.path.join(root_dir, 'stats'), exist_ok=True)
    output_charts_two(tables, os.path.join(root_dir, 'stats'))
    output_multiple_tables(tables, os.path.join(root_dir, 'stats'))

def process_all_images_erosions(config):
    filenames = sorted(config.files)
    erosion_results = pd.DataFrame(columns=['erosions', 'jaccard', 'f1', 'split', 'merge'])
    n = len(filenames)
    print(n)
    for erosion in range(1, 11):
        print("Erosion {}".format(erosion))
        counter = 0
        jaccard = 0
        f1 = 0
        split = 0
        merge = 0
        erosions = []
        for file in filenames:
            if counter % 50 == 0:
                print(counter)
            unet_path = os.path.join(config.unet, file.strip())
            mrcnn_path = os.path.join(config.mrcnn, file.strip())
            annot_path = os.path.join(config.annot, file.strip())
            unet = skimage.io.imread(unet_path, as_gray=True)
            mrcnn = skimage.io.imread(mrcnn_path, as_gray=True)
            annot = skimage.io.imread(annot_path, as_gray=True)
            ensemble = ensemble_segmentations_label(unet, mrcnn, erosion)
            ensemble_results = comp.get_per_image_metrics(annot, ensemble, False)
            jaccard += ensemble_results['jac']
            f1 += ensemble_results['af1']
            merge += ensemble_results['merge_rate']
            split += ensemble_results['split_rate']
            counter += 1
        erosions.append(erosion)
        erosion_results = erosion_results.append({'erosions': erosion, 'jaccard': jaccard / n, 'f1': f1 / n,
                                                  'split': split / n, 'merge': merge / n}, ignore_index=True)
    erosion_results = erosion_results.set_index('erosions')
    chart = erosion_results.plot.line()
    plt.title('Ensemble results vs erosions (training)')
    plt.xlabel('Number of erosions')
    plt.savefig('erosions_training.png')

def process_all_images_beta(config):
    filenames = sorted(config.files)
    beta_results = pd.DataFrame(columns=['beta', 'jaccard', 'f1', 'split', 'merge'])
    n = len(filenames)
    print(n)
    for beta in range(10, 110, 10):
        print("Beta {}".format(beta))
        counter = 0
        jaccard = 0
        f1 = 0
        split = 0
        merge = 0
        betas = []
        for file in filenames:
            if counter % 50 == 0:
                print(counter)
            unet_path = os.path.join(config.unet, file.strip())
            mrcnn_path = os.path.join(config.mrcnn, file.strip())
            annot_path = os.path.join(config.annot, file.strip())
            unet = skimage.io.imread(unet_path, as_gray=True)
            mrcnn = skimage.io.imread(mrcnn_path, as_gray=True)
            annot = skimage.io.imread(annot_path, as_gray=True)
            ensemble = ensemble_segmentations_label(unet, mrcnn, beta=beta)
            ensemble_results = comp.get_per_image_metrics(annot, ensemble, False)
            jaccard += ensemble_results['jac']
            f1 += ensemble_results['af1']
            merge += ensemble_results['merge_rate']
            split += ensemble_results['split_rate']
            counter += 1
        betas.append(beta)
        beta_results = beta_results.append({'beta': beta, 'jaccard': jaccard / n, 'f1': f1 / n,
                                                  'split': split / n, 'merge': merge / n}, ignore_index=True)
    beta_results = beta_results.set_index('beta')
    chart = beta_results.plot.line()
    plt.title('Ensemble results vs Beta (training)')
    plt.xlabel('Beta')
    plt.savefig('beta_training.png')


def ensemble_segmentations_label(unet, mrcnn, erosions=5, beta=30):
    unet_seed, mrcnn_seed = erode_images(unet, mrcnn, erosions)
    unet_mask = np.where(unet > 0, 1, 0)
    mrcnn_mask = np.where(mrcnn > 0, 1, 0)
    labels = np.where(unet_seed + mrcnn_seed > 0, 1, 0)
    labels = skimage.measure.label(labels)
    labels, _, _ = skimage.segmentation.relabel_sequential(labels, offset=2)
    labels[unet_mask + mrcnn_mask == 0] = 1
    data = np.dstack((unet, mrcnn))
    final = random_walker(data, labels, beta=beta, mode='bf')
    final[final == 1] = 0
    return final

def ensemble_segmentations_binary(unet, mrcnn, erosions=5, beta=30):
    unet_seed, mrcnn_seed = erode_images(unet, mrcnn, erosions)
    unet_mask = np.where(unet > 0, 1, 0)
    mrcnn_mask = np.where(mrcnn > 0, 1, 0)
    labels = np.where(unet_seed + mrcnn_seed > 0, 1, 0)
    labels[unet_mask + mrcnn_mask == 0] = 2
    data = np.dstack((unet, mrcnn))
    final = random_walker(data, labels, beta=beta, mode='bf')
    final[final == 2] = 0
    return skimage.measure.label(final)

def ensemble_segmentations_union(unet, mrcnn):
    final = unet + mrcnn
    final = np.where(final > 0, 1, 0)
    final = skimage.measure.label(final)
    return final


def erode_images(img1, img2, num):
    img1_seed = img1.copy()
    img2_seed = img2.copy()
    for i in range(num):
        img1_seed = skimage.morphology.erosion(img1_seed)
        img2_seed = skimage.morphology.erosion(img2_seed)
    return img1_seed, img2_seed

def output_charts_two(tables, root_dir):
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()

    bins = np.linspace(0, 1, 100)
    sns.distplot(tables['jac']['unet'].tolist(), hist=False, kde=True,
                     kde_kws = {'shade': True, 'linewidth': 3}, label='unet', bins=bins)
    sns.distplot(tables['jac']['mrcnn'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='mrcnn')
    sns.distplot(tables['jac']['walker_label'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='walker_label')
    sns.distplot(tables['jac']['walker_binary'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='walker_binary')
    sns.distplot(tables['jac']['union'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='union')
    plt.legend(loc='upper left')
    plt.title('Jaccard Scores Test')
    plt.xlabel('Jaccard Score')
    plt.savefig(os.path.join(root_dir, 'jaccard.png'))
    plt.clf()

    sns.distplot(tables['f1']['unet'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, label='unet', bins=bins)
    sns.distplot(tables['f1']['mrcnn'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='mrcnn')
    sns.distplot(tables['f1']['walker_label'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='walker_label')
    sns.distplot(tables['f1']['walker_binary'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='walker_binary')
    sns.distplot(tables['f1']['union'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='union')
    plt.legend(loc='upper left')
    plt.title('F1 Scores Test')
    plt.xlabel('F1 Score')
    plt.savefig(os.path.join(root_dir, 'f1.png'))
    plt.clf()
    sns.distplot(tables['merge']['unet'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, label='unet', bins=bins)
    sns.distplot(tables['merge']['mrcnn'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='mrcnn')
    sns.distplot(tables['merge']['walker_label'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='walker_label')
    sns.distplot(tables['merge']['walker_binary'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='walker_binary')
    sns.distplot(tables['merge']['union'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=100, label='union')
    plt.legend()
    plt.title('Merge Rates Test')
    plt.xlabel('Merge Rate')
    plt.savefig(os.path.join(root_dir, 'merge.png'))
    plt.clf()

    sns.distplot(tables['split']['unet'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, label='unet', bins=100)
    sns.distplot(tables['split']['mrcnn'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=100, label='mrcnn')
    sns.distplot(tables['split']['walker_label'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=100, label='walker_label')
    sns.distplot(tables['split']['walker_binary'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=100, label='walker_binary')
    sns.distplot(tables['split']['union'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=100, label='union')
    plt.title('Split Rates Test')
    plt.xlabel('Split Rate')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'split.png'))


def output_multiple_tables(tables, root_dir):
    f = open(os.path.join(root_dir, 'tables.txt'), 'w')
    f.write('Jaccard Scores\n')
    f.write(tables['jac'].to_string(index=True))
    f.write('\n\n')
    f.write('F1 Scores\n')
    f.write(tables['f1'].to_string(header=True))
    f.write('\n\n')
    f.write('Split Rates\n')
    f.write(tables['split'].to_string(header=True))
    f.write('\n\n')
    f.write('Merge Rates\n')
    f.write(tables['merge'].to_string(header=True))
    f.close()

if __name__ == '__main__':
    CONFIG = Config(unet='/Users/arianrahbar/Dropbox/Unet/ContourAdjusted/',
                    mrcnn='/Users/arianrahbar/Dropbox/Mrcnn/OutLabels/',
                    annot='/Users/arianrahbar/Dropbox/raw_annotations/',
                    file='training.txt',
                    instance=False)
    process_all_images(CONFIG)

