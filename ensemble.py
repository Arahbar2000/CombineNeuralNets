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
    jaccard_results = pd.DataFrame(columns=['unet', 'mrcnn', 'ensemble_walker', 'ensemble_union'], copy=True)
    f1_results = pd.DataFrame(columns=['unet', 'mrcnn', 'ensemble_walker', 'ensemble_union'], copy=True)
    split_results = pd.DataFrame(columns=['unet', 'mrcnn', 'ensemble_walker', 'ensemble_union'], copy=True)
    merge_results = pd.DataFrame(columns=['unet', 'mrcnn', 'ensemble_walker', 'ensemble_union'], copy=True)
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
        if counter % 10 == 0:
            print(counter)
        if counter == 50:
            break
        unet_path = os.path.join(config.unet, file.strip())
        mrcnn_path = os.path.join(config.mrcnn, file.strip())
        annot_path = os.path.join(config.annot, file.strip())
        unet = pred.predict(skimage.io.imread(unet_path))
        mrcnn = skimage.io.imread(mrcnn_path, as_gray=True)
        annot = skimage.io.imread(annot_path, cv2.IMREAD_GRAYSCALE)
        ensemble_walker = ensemble_segmentations(unet, mrcnn, 6)
        ensemble_union = ensemble_segmentations_union(unet, mrcnn)
        unet_results = comp.get_per_image_metrics(annot, unet, False)
        mrcnn_results = comp.get_per_image_metrics(annot, mrcnn, False)
        ensemble_walker_results = comp.get_per_image_metrics(annot, ensemble_walker, False)
        ensemble_union_results = comp.get_per_image_metrics(annot, ensemble_union, False)

        jaccard_results = jaccard_results.append({'unet': unet_results['jac'], 'mrcnn': mrcnn_results['jac'],
                                                  'ensemble_walker': ensemble_walker_results['jac'],
                                                  'ensemble_union': ensemble_union_results['jac']}, ignore_index=True)
        f1_results = f1_results.append({'unet': unet_results['af1'], 'mrcnn': mrcnn_results['af1'],
                                        'ensemble_walker': ensemble_walker_results['af1'],
                                        'ensemble_union': ensemble_union_results['af1']}, ignore_index=True)
        split_results = split_results.append({'unet': unet_results['split_rate'], 'mrcnn': mrcnn_results['split_rate'],
                                              'ensemble_walker': ensemble_walker_results['split_rate'],
                                              'ensemble_union': ensemble_union_results['split_rate']}, ignore_index=True)
        merge_results = merge_results.append({'unet': unet_results['merge_rate'], 'mrcnn': mrcnn_results['merge_rate'],
                                              'ensemble_walker': ensemble_walker_results['merge_rate'],
                                              'ensemble_union': ensemble_union_results['merge_rate']}, ignore_index=True)

        unet_mask = np.where(unet > 0, 255, 0)
        mrcnn_mask = np.where(mrcnn > 0, 255, 0)
        ensemble_walker_mask = np.where(ensemble_walker > 0, 255, 0)
        ensemble_union_mask = np.where(ensemble_union > 0, 255, 0)
        annot_mask = np.where(annot > 0, 255, 0)
        file_dir = os.path.join(root_dir, file)
        os.makedirs(file_dir, exist_ok=True)
        cv2.imwrite(os.path.join(file_dir, 'unet.png'), unet_mask)
        cv2.imwrite(os.path.join(file_dir, 'mrcnn.png'), mrcnn_mask)
        cv2.imwrite(os.path.join(file_dir, 'ensemble_walker.png'), ensemble_walker_mask)
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
    os.makedirs(config.filename, exist_ok=True)
    if config.instance:
        root_dir = os.path.join(config.filename, 'instance')
    else:
        root_dir = os.path.join(config.filename, 'images')
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)
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
            unet = pred.predict(skimage.io.imread(unet_path))
            mrcnn = skimage.io.imread(mrcnn_path, as_gray=True)
            annot = skimage.io.imread(annot_path, cv2.IMREAD_GRAYSCALE)
            ensemble = ensemble_segmentations(unet, mrcnn, erosion)
            unet_results = comp.get_per_image_metrics(annot, unet, False)
            mrcnn_results = comp.get_per_image_metrics(annot, mrcnn, False)
            ensemble_results = comp.get_per_image_metrics(annot, ensemble, False)
            jaccard += ensemble_results['jac']
            f1 += ensemble_results['af1']
            merge += ensemble_results['merge_rate']
            split += ensemble_results['split_rate']

            unet_mask = np.where(unet > 0, 255, 0)
            mrcnn_mask = np.where(mrcnn > 0, 255, 0)
            ensemble_mask = np.where(ensemble > 0, 255, 0)
            file_dir = os.path.join(root_dir, file)
            os.makedirs(file_dir, exist_ok=True)
            cv2.imwrite(os.path.join(file_dir, 'unet.png'), unet_mask)
            cv2.imwrite(os.path.join(file_dir, 'mrcnn.png'), mrcnn_mask)
            cv2.imwrite(os.path.join(file_dir, 'ensemble.png'), ensemble_mask)
            counter += 1
        erosions.append(erosion)
        erosion_results = erosion_results.append({'erosions': erosion, 'jaccard': jaccard / 50, 'f1': f1 / 50,
                                                  'split': split / 50, 'merge': merge / 50}, ignore_index=True)
    erosion_results = erosion_results.set_index('erosions')
    chart = erosion_results.plot.line()
    plt.title('Ensemble results vs erosions (training)')
    plt.xlabel('Number of erosions')
    plt.savefig('erosions_training.png')





    os.makedirs(os.path.join(root_dir, 'stats'), exist_ok=True)


def generate_instances(unet, mrcnn, erosions):
    unet, unet_num = skimage.measure.label(unet, return_num=True)
    mrcnn, mrcnn_num = skimage.measure.label(mrcnn, return_num=True)
    unet_label = 1
    final = np.zeros_like(unet)
    mrcnn_labels = set(range(1, mrcnn_num+1))
    for i in range(1, unet_num+1):
        max_overlap = 0
        mrcnn_label = 0
        for j in range(1, mrcnn_num + 1):
            unet_test = np.where(unet == i, 1, 0)
            mrcnn_test = np.where(mrcnn == j, 1, 0)
            overlap_img = unet_test * mrcnn_test
            overlap = np.count_nonzero(overlap_img)
            if overlap > max_overlap:
                mrcnn_label = j
                max_overlap = overlap
        if mrcnn_label in mrcnn_labels:
            mrcnn_labels.remove(mrcnn_label)
        if mrcnn_label != 0:
            unet_instance = np.where(unet == i, 1, 0)
            mrcnn_instance = np.where(mrcnn == mrcnn_label, 1, 0)
            instance = ensemble_instances(unet_instance, mrcnn_instance, erosions)
            final[instance > 0] = i

    for label, m_label in enumerate(mrcnn_labels, unet_num+1):
        final[mrcnn == m_label] = label
    return final


def ensemble_segmentations(unet, mrcnn, erosions):
    unet_seed, mrcnn_seed = erode_images(unet, mrcnn, erosions)
    unet_mask = np.where(unet > 0, 1, 0)
    mrcnn_mask = np.where(mrcnn > 0, 1, 0)
    labels = np.where(unet_seed + mrcnn_seed > 0, 1, 0)
    labels = skimage.measure.label(labels)
    labels, _, _ = skimage.segmentation.relabel_sequential(labels, offset=2)
    labels[unet_mask + mrcnn_mask == 0] = 1
    data = np.dstack((unet, mrcnn))
    final = random_walker(data, labels, beta=30, mode='bf')
    final[final == 1] = 0
    return final

def ensemble_segmentations_union(unet, mrcnn):
    final = unet + mrcnn
    final = np.where(final > 0, 1, 0)
    final = skimage.measure.label(final)
    return final

def ensemble_instances(unet, mrcnn, erosions):
    unet_seed, mrcnn_seed = erode_images(unet, mrcnn, erosions)
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

def output_charts(tables, root_dir):
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()

    bins = np.linspace(0, 1, 50)
    a11 = fig1.add_subplot(3, 1, 1)
    a11.hist(tables['jac']['unet'].tolist(),
                   bins=bins, label='unet', histtype='step')
    a11.set_title("Jaccard Unet")

    a12 = fig1.add_subplot(3, 1, 2)
    a12.hist(tables['jac']['mrcnn'].tolist(),
                   bins=bins, label='mrcnn', histtype='step')
    a12.set_title("Jaccard Mrcnn")

    a13 = fig1.add_subplot(3, 1, 3)
    a13.hist(tables['jac']['ensemble'].tolist(),
                   bins=bins, label='ensemble', histtype='step')
    a13.set_title("Jaccard Ensemble")


    a21 = fig2.add_subplot(3, 1, 1)
    a21.hist(tables['f1']['unet'].tolist(),
                   bins=bins, label='unet', histtype='step')
    a21.set_title("F1 Unet")

    a22 = fig2.add_subplot(3, 1, 2)
    a22.hist(tables['f1']['mrcnn'].tolist(),
                   bins=bins, label='mrcnn', histtype='step')
    a22.set_title("F1 Mrcnn")

    a23 = fig2.add_subplot(3, 1, 3)
    a23.hist(tables['f1']['ensemble'].tolist(),
             bins=bins, label='ensemble', histtype='step')
    a23.set_title("F1 Ensemble")


    a31 = fig3.add_subplot(3, 1, 1)
    a31.hist(tables['split']['unet'].tolist(),
             bins=bins, label='unet', histtype='step')
    a31.set_title("Split Rate Unet")

    a32 = fig3.add_subplot(3, 1, 2)
    a32.hist(tables['split']['mrcnn'].tolist(),
             bins=bins, label='mrcnn', histtype='step')
    a32.set_title("Split Rate Mrcnn")

    a33 = fig3.add_subplot(3, 1, 3)
    a33.hist(tables['split']['ensemble'].tolist(),
             bins=bins, label='ensemble', histtype='step')
    a33.set_title("Split Rate Ensemble")

    a41 = fig4.add_subplot(3, 1, 1)
    a41.hist(tables['merge']['unet'].tolist(),
             bins=bins, label='unet', histtype='step')
    a41.set_title("Merge Rate Unet")

    a42 = fig4.add_subplot(3, 1, 2)
    a42.hist(tables['merge']['mrcnn'].tolist(),
             bins=bins, label='mrcnn', histtype='step')
    a42.set_title("Merge Rate Mrcnn")

    a43 = fig4.add_subplot(3, 1, 3)
    a43.hist(tables['merge']['ensemble'].tolist(),
             bins=bins, label='ensemble', histtype='step')
    a43.set_title("Merge Rate Ensemble")

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig1.savefig(os.path.join(root_dir, 'jaccard.png'))
    fig2.savefig(os.path.join(root_dir, 'f1.png'))
    fig3.savefig(os.path.join(root_dir, 'split.png'))
    fig4.savefig(os.path.join(root_dir, 'merge.png'))

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
    sns.distplot(tables['jac']['ensemble_walker'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='ensemble_walker')
    sns.distplot(tables['jac']['ensemble_union'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='ensemble_union')
    plt.legend(loc='upper left')
    plt.title('Jaccard Scores Test')
    plt.xlabel('Jaccard Score')
    plt.savefig(os.path.join(root_dir, 'jaccard.png'))
    plt.clf()

    sns.distplot(tables['f1']['unet'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, label='unet', bins=bins)
    sns.distplot(tables['f1']['mrcnn'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='mrcnn')
    sns.distplot(tables['f1']['ensemble_walker'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='ensemble_walker')
    sns.distplot(tables['f1']['ensemble_union'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='ensemble_union')
    plt.legend(loc='upper left')
    plt.title('F1 Scores Test')
    plt.xlabel('F1 Score')
    plt.savefig(os.path.join(root_dir, 'f1.png'))
    plt.clf()
    sns.distplot(tables['merge']['unet'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, label='unet', bins=bins)
    sns.distplot(tables['merge']['mrcnn'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='mrcnn')
    sns.distplot(tables['merge']['ensemble_walker'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label='ensemble_walker')
    sns.distplot(tables['merge']['ensemble_union'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=100, label='ensemble_union')
    plt.legend()
    plt.title('Merge Rates Test')
    plt.xlabel('Merge Rate')
    plt.savefig(os.path.join(root_dir, 'merge.png'))
    plt.clf()

    sns.distplot(tables['split']['unet'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, label='unet', bins=100)
    sns.distplot(tables['split']['mrcnn'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=100, label='mrcnn')
    sns.distplot(tables['split']['ensemble_walker'].tolist(), hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, bins=100, label='ensemble_walker')
    sns.distplot(tables['split']['ensemble_union'].tolist(), hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, bins=100, label='ensemble_union')
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
    CONFIG = Config(unet='/Users/arianrahbar/Dropbox/Unet/OutProbs//',
                    mrcnn='/Users/arianrahbar/Dropbox/Mrcnn/OutLabels/',
                    annot='/Users/arianrahbar/Dropbox/raw_annotations/',
                    file='training.txt',
                    instance=False)
    process_all_images(CONFIG)

