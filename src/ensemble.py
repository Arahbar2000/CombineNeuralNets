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
    def __init__(self, unet, mrcnn, annot, file, counter=None, erosions=5, beta=30, instance=False):
        self.unet = unet
        self.mrcnn = mrcnn
        self.annot = annot
        f = open('/Users/arianrahbar/Dropbox/' + file)
        self.files = f.readlines()
        self.instance = instance
        self.filename = file.split('.')[0]
        self.erosions = erosions
        self.beta = beta
        self.counter = counter

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
        'mrcnn': args.mrcnn,
        'unet': args.unet,
        'annot': args.annot,
        'file': args.file
    }


def process_all_images(config):
    filenames = sorted(config.files)
    tables = {}
    methods = {'unet': {}, 'mrcnn': {}, 'walker_label': {}, 'walker_binary': {}, 'walker_old': {}, 'union': {}}
    for key in ['jac', 'af1', 'merge_rate', 'split_rate']:
        tables[key] = pd.DataFrame(columns=list(methods.keys()), copy=True)
    os.makedirs('../' + config.filename, exist_ok=True)
    root_dir = os.path.join('..', config.filename, 'images')
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    counter = 0
    for file in filenames:
        if counter % 50 == 0:
            print(counter)
        if counter == config.counter:
            break
        unet_path = os.path.join(config.unet, file.strip())
        mrcnn_path = os.path.join(config.mrcnn, file.strip())
        annot_path = os.path.join(config.annot, file.strip())
        methods['unet']['orig'] = skimage.io.imread(unet_path, as_gray=True)
        methods['mrcnn']['orig'] = skimage.io.imread(mrcnn_path, as_gray=True)
        annot = skimage.io.imread(annot_path, as_gray=True)
        methods['walker_label']['orig'] = ensemble_segmentations_label(methods['unet']['orig'],
                                                                       methods['mrcnn']['orig'],
                                                                       erosions=config.erosions, beta=config.beta)
        methods['walker_binary']['orig'] = ensemble_segmentations_binary(methods['unet']['orig'],
                                                                         methods['mrcnn']['orig'],
                                                                         erosions=config.erosions, beta=config.beta)
        methods['walker_old']['orig'] = ensemble_segmentations_old(methods['unet']['orig'],
                                                                   methods['mrcnn']['orig'],
                                                                   erosions=2, beta=config.beta)
        methods['union']['orig'] = ensemble_segmentations_union(methods['unet']['orig'], methods['mrcnn']['orig'])
        for method in methods.keys():
            methods[method]['results'] = comp.get_per_image_metrics(annot, methods[method]['orig'], False)
            methods[method]['mask'] = np.where(methods[method]['orig'] > 0, 255, 0)
        for key in tables.keys():
            tables[key] = tables[key].append(
                {'unet': methods['unet']['results'][key], 'mrcnn': methods['mrcnn']['results'][key],
                 'walker_label': methods['walker_label']['results'][key],
                 'walker_binary': methods['walker_binary']['results'][key],
                 'walker_old': methods['walker_old']['results'][key],
                 'union': methods['union']['results'][key]}, ignore_index=True)
        annot_mask = np.where(annot > 0, 255, 0)
        file_dir = os.path.join(root_dir, file)
        os.makedirs(file_dir, exist_ok=True)
        for method in methods:
            cv2.imwrite(os.path.join(file_dir, f'{method}.png'), methods[method]['mask'])
        cv2.imwrite(os.path.join(file_dir, 'annot.png'), annot_mask)
        counter += 1

    os.makedirs(os.path.join(config.filename, 'stats'), exist_ok=True)
    output_charts(tables, list(methods.keys()), os.path.join(config.filename, 'stats'), config)
    output_multiple_tables(tables, os.path.join(config.filename, 'stats'))


def process_all_images_erosions(config):
    filenames = sorted(config.files)
    erosion_results = {'label': pd.DataFrame(columns=['erosions', 'jaccard', 'f1', 'split', 'merge']),
                       'binary': pd.DataFrame(columns=['erosions', 'jaccard', 'f1', 'split', 'merge'])}
    n = len(filenames)
    for erosion in range(1, 16):
        print("Erosion {}".format(erosion))
        counter = 0
        results = {'binary': {'jac': 0, 'af1': 0, 'merge_rate': 0, 'split_rate': 0},
                   'label': {'jac': 0, 'af1': 0, 'merge_rate': 0, 'split_rate': 0}}
        for file in filenames:
            if counter % 50 == 0:
                print(counter)
            unet_path = os.path.join(config.unet, file.strip())
            mrcnn_path = os.path.join(config.mrcnn, file.strip())
            annot_path = os.path.join(config.annot, file.strip())
            unet = skimage.io.imread(unet_path, as_gray=True)
            mrcnn = skimage.io.imread(mrcnn_path, as_gray=True)
            annot = skimage.io.imread(annot_path, as_gray=True)
            ensemble_label = ensemble_segmentations_label(unet, mrcnn, erosions=erosion, beta=config.beta)
            ensemble_binary = ensemble_segmentations_binary(unet, mrcnn, erosions=erosion, beta=config.beta)
            ensemble_label_results = comp.get_per_image_metrics(annot, ensemble_label, False)
            ensemble_binary_results = comp.get_per_image_metrics(annot, ensemble_binary, False)
            for stat in ensemble_binary_results.keys():
                results['binary'][stat] += ensemble_binary_results[stat]
                results['label'][stat] += ensemble_label_results[stat]
            counter += 1
        erosion_results['binary'] = erosion_results['binary'].append(
            {'erosions': erosion, 'jaccard': results['binary']['jac'] / n,
             'f1': results['binary']['af1'] / n,
             'split': results['binary']['split_rate'] / n,
             'merge': results['binary']['merge_rate'] / n},
            ignore_index=True)
        erosion_results['label'] = erosion_results['label'].append(
            {'erosions': erosion, 'jaccard': results['label']['jac'] / n,
             'f1': results['label']['af1'] / n,
             'split': results['label']['split_rate'] / n,
             'merge': results['label']['merge_rate'] / n},
            ignore_index=True)
    for method in results.keys():
        erosion_results[method] = erosion_results[method].set_index('erosions')
        erosion_results[method].plot.line()
        plt.title(f'Walker_{method} Result Averages vs Erosions ({config.filename}, beta={config.beta})')
        plt.xlabel('Number of erosions')
        plt.savefig(f'erosions_{method}_{config.filename}.png')
        plt.clf()


def process_all_images_beta(config):
    filenames = sorted(config.files)
    beta_results = {'label': pd.DataFrame(columns=['beta', 'jaccard', 'f1', 'split', 'merge']),
                       'binary': pd.DataFrame(columns=['beta', 'jaccard', 'f1', 'split', 'merge'])}
    n = len(filenames)
    for beta in range(10, 110, 10):
        print("Beta {}".format(beta))
        counter = 0
        results = {'binary': {'jac': 0, 'af1': 0, 'merge_rate': 0, 'split_rate': 0},
                   'label': {'jac': 0, 'af1': 0, 'merge_rate': 0, 'split_rate': 0}}
        for file in filenames:
            if counter % 50 == 0:
                print(counter)
            unet_path = os.path.join(config.unet, file.strip())
            mrcnn_path = os.path.join(config.mrcnn, file.strip())
            annot_path = os.path.join(config.annot, file.strip())
            unet = skimage.io.imread(unet_path, as_gray=True)
            mrcnn = skimage.io.imread(mrcnn_path, as_gray=True)
            annot = skimage.io.imread(annot_path, as_gray=True)
            ensemble_label = ensemble_segmentations_label(unet, mrcnn, erosions=6, beta=beta)
            ensemble_binary = ensemble_segmentations_binary(unet, mrcnn, erosions=6, beta=beta)
            ensemble_label_results = comp.get_per_image_metrics(annot, ensemble_label, False)
            ensemble_binary_results = comp.get_per_image_metrics(annot, ensemble_binary, False)
            for stat in ensemble_binary_results.keys():
                results['binary'][stat] += ensemble_binary_results[stat]
                results['label'][stat] += ensemble_label_results[stat]
            counter += 1
        beta_results['binary'] = beta_results['binary'].append(
            {'beta': beta, 'jaccard': results['binary']['jac'] / n,
             'f1': results['binary']['af1'] / n,
             'split': results['binary']['split_rate'] / n,
             'merge': results['binary']['merge_rate'] / n},
            ignore_index=True)
        beta_results['label'] = beta_results['label'].append(
            {'beta': beta, 'jaccard': results['label']['jac'] / n,
             'f1': results['label']['af1'] / n,
             'split': results['label']['split_rate'] / n,
             'merge': results['label']['merge_rate'] / n},
            ignore_index=True)
    for method in results.keys():
        beta_results[method] = beta_results[method].set_index('beta')
        beta_results[method].plot.line()
        plt.title(f'Walker_{method} Result Averages vs Beta ({config.filename}, beta={config.beta})')
        plt.xlabel('Beta')
        plt.savefig(f'beta_{method}_{config.filename}.png')
        plt.clf()


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


def ensemble_segmentations_old(unet, mrcnn, erosions=5, beta=30):
    unet_seed, mrcnn_seed = erode_images(unet, mrcnn, erosions)
    ensemble = unet + mrcnn
    # ensemble = skimage.measure.label(ensemble)
    ensemble_seed = np.where(unet_seed + mrcnn_seed > 0, 1, 0)
    labels = np.where(ensemble > 0, 0, 2)
    labels[ensemble_seed > 0] = 1
    final = skimage.segmentation.random_walker(ensemble, labels, beta=beta)
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


def output_charts(tables, methods, root_dir, config):
    bins = np.linspace(0, 1, 100)
    reference = {
        'jac': 'Jaccard Score',
        'af1': 'F1 Score',
        'merge_rate': 'Merge Rate',
        'split_rate': 'Split Rate'
    }
    # methods.remove('walker_old')
    methods.remove('unet')
    methods.remove('mrcnn')
    for key in tables.keys():
        for method in methods:
            sns.distplot(tables[key][method].tolist(), hist=False, kde=True,
                         kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label=method)
        plt.legend(loc='upper left') if (key == 'jac' or key == 'f1') else plt.legend()
        plt.title(f"{reference[key]} {config.filename.capitalize()} (erosions={config.erosions}, beta={config.beta})")
        plt.xlabel(reference[key])
        plt.savefig(os.path.join(root_dir, f'{reference[key].split()[0]}.png'))
        plt.clf()
        avg = tables[key].mean()
        std = tables[key].std()
        avg = avg.drop(['unet', 'mrcnn'])
        std = std.drop(['unet', 'mrcnn'])
        # avg = avg.drop('walker_old')
        # std = std.drop('walker_old')
        avg.plot.bar(yerr=std.tolist())
        plt.ylim(bottom=0)
        plt.xticks(rotation='horizontal')
        plt.title(
            f'{reference[key]} Averages {config.filename.capitalize()} (erosions={config.erosions}, beta={config.beta})')
        plt.xlabel('Method')
        plt.ylabel(f'{reference[key]}')
        plt.savefig(os.path.join(root_dir, f'{reference[key].split()[0]}_averages.png'))
        plt.clf()


def output_multiple_tables(tables, root_dir):
    reference = {
        'jac': 'Jaccard Score',
        'af1': 'F1 Score',
        'merge_rate': 'Merge Rate',
        'split_rate': 'Split Rate'
    }
    f = open(os.path.join(root_dir, 'tables.txt'), 'w')
    for key in tables.keys():
        f.write(f'{reference[key]}')
        f.write(tables[key].to_string(index=True))
        f.write('\n\n')
    f.close()


if __name__ == '__main__':
    options = parse_arguments()

    CONFIG = Config(unet=options['unet'],
                    mrcnn=options['mrcnn'],
                    annot=options['annot'],
                    file=options['file'],
                    instance=False,
                    erosions=6,
                    beta=30,
                    counter=None)
    process_all_images(CONFIG)
